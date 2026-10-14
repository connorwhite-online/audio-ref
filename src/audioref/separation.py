"""Source separation: Demucs (htdemucs family) or an HPSS + crossover fallback ("pseudo-stems")."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import librosa
import numpy as np
from scipy import signal

from . import models
from .util import log

STEMS = ("drums", "bass", "other", "vocals")


def _load_demucs(name: str, device: str):
    import torch  # noqa: F401
    from demucs.pretrained import get_model

    m = get_model(name)
    m.eval()
    return m


def demucs_separate(stereo: np.ndarray, sr: int, model_name: str = "htdemucs", device: str = "auto",
                    shifts: int = 0, progress: Callable[[float], None] | None = None) -> dict[str, np.ndarray]:
    """Separate a (2, n) signal at 44.1 kHz. Returns {stem: (2, n) float32}."""
    import torch
    from demucs.apply import apply_model

    dev = models.torch_device(device)
    model = models.get(f"demucs:{model_name}", lambda: _load_demucs(model_name, dev))
    if sr != model.samplerate:
        raise ValueError(f"demucs expects {model.samplerate} Hz")
    wav = torch.from_numpy(np.ascontiguousarray(stereo, dtype=np.float32))
    ref = wav.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    x = ((wav - mean) / std)[None]

    def run(device: str):
        with torch.no_grad():
            return apply_model(model, x, device=device, shifts=shifts, split=True, overlap=0.25, progress=False,
                               num_workers=0)[0]

    try:
        out = run(dev)
    except Exception as e:  # MPS can fail on some ops/shapes; CPU always works
        if dev != "cpu":
            log.warning("demucs on %s failed (%s); retrying on cpu", dev, e)
            out = run("cpu")
        else:
            raise
    out = (out * std + mean).cpu().numpy().astype(np.float32)
    stems = {name: out[i] for i, name in enumerate(model.sources)}
    if progress:
        progress(1.0)
    return stems


def pseudo_stems(stereo: np.ndarray, sr: int) -> dict[str, np.ndarray]:
    """Cheap fallback: HPSS (percussive -> drums) + low/high crossover of the harmonic part."""
    out: dict[str, list[np.ndarray]] = {k: [] for k in STEMS}
    for ch in stereo:
        S = librosa.stft(ch, n_fft=4096, hop_length=1024)
        H, P = librosa.decompose.hpss(S, kernel_size=(17, 31), margin=1.0)
        n = ch.shape[-1]
        drums = librosa.istft(P, hop_length=1024, length=n)
        harm = librosa.istft(H, hop_length=1024, length=n)
        sos_lo = signal.butter(4, 180 / (sr / 2), "lowpass", output="sos")
        sos_hi = signal.butter(4, 180 / (sr / 2), "highpass", output="sos")
        out["drums"].append(drums)
        out["bass"].append(signal.sosfiltfilt(sos_lo, harm))
        out["other"].append(signal.sosfiltfilt(sos_hi, harm))
        out["vocals"].append(np.zeros(n))
    return {k: np.asarray(v, dtype=np.float32) for k, v in out.items()}


def separate(stereo: np.ndarray, sr: int, method: str = "htdemucs", device: str = "auto",
             progress: Callable[[float], None] | None = None) -> tuple[dict[str, np.ndarray], str]:
    """Returns (stems, method_used)."""
    if method != "none" and models.capabilities()["demucs"]:
        try:
            return demucs_separate(stereo, sr, method, device, progress=progress), method
        except Exception as e:
            log.warning("demucs failed (%s); falling back to HPSS pseudo-stems", e)
    return pseudo_stems(stereo, sr), "hpss-fallback"


def save_stems(stems: dict[str, np.ndarray], folder: Path, sr: int, fmt: str = "flac") -> dict[str, str]:
    """Write stems as 16-bit files. A shared scale factor (scale.json) avoids clipping loud stems."""
    import json

    import soundfile as sf

    folder.mkdir(parents=True, exist_ok=True)
    peak = max((float(np.max(np.abs(v))) for v in stems.values() if v.size), default=0.0)
    scale = 0.99 / peak if peak > 0.99 else 1.0
    paths = {}
    for k, v in stems.items():
        p = folder / f"{k}.{fmt}"
        sf.write(str(p), (v * scale).T, sr, subtype="PCM_16")
        paths[k] = str(p)
    (folder / "scale.json").write_text(json.dumps({"scale": scale}))
    return paths


def load_stems(folder: Path, names=None, fmt: str = "flac") -> dict[str, np.ndarray] | None:
    import json

    import soundfile as sf

    names = names or STEMS
    scale = 1.0
    if (folder / "scale.json").is_file():
        scale = float(json.loads((folder / "scale.json").read_text()).get("scale", 1.0))
    out = {}
    for k in list(names) + ["piano", "guitar"]:
        p = folder / f"{k}.{fmt}"
        if not p.is_file():
            if k in ("drums", "bass", "other"):
                return None
            continue
        x, _ = sf.read(str(p), dtype="float32", always_2d=True)
        out[k] = x.T.copy() / scale
    return out or None
