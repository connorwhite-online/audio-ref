"""Decode audio (ffmpeg), hash it, measure loudness, build a normalized working copy."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soxr

from . import loudness
from .cache import audio_hash
from .util import log

SR = 44100  # ingest / separation / onset-precision rate
SR_ANALYSIS = 22050  # librosa feature rate
TARGET_LUFS = -23.0
AUDIO_EXTS = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".aiff", ".aif", ".ogg", ".opus", ".wma", ".mp4", ".webm"}


class IngestError(RuntimeError):
    pass


@dataclass
class Audio:
    path: Path
    stereo: np.ndarray  # (2, n) float32 @ SR, original level
    sr: int
    start: float
    end: float | None
    track_id: str
    loudness: dict
    gain_db: float  # gain applied to reach TARGET_LUFS in the working copy
    _mono: np.ndarray | None = field(default=None, repr=False)

    @property
    def duration(self) -> float:
        return self.stereo.shape[-1] / self.sr

    @property
    def gain(self) -> float:
        return float(10 ** (self.gain_db / 20.0))

    @property
    def stereo_norm(self) -> np.ndarray:
        return self.stereo * self.gain

    @property
    def mono(self) -> np.ndarray:
        """Normalized mono mix at SR_ANALYSIS (librosa features)."""
        if self._mono is None:
            m = self.stereo.mean(axis=0) * self.gain
            self._mono = soxr.resample(m, self.sr, SR_ANALYSIS).astype(np.float32)
        return self._mono

    @property
    def mono_full(self) -> np.ndarray:
        """Normalized mono mix at full SR (precise onsets)."""
        return (self.stereo.mean(axis=0) * self.gain).astype(np.float32)


def have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def is_url(s: str) -> bool:
    return urllib.parse.urlparse(str(s)).scheme in ("http", "https")


def download(url: str, allow_download: bool, dest_dir: Path) -> Path:
    """Fetch a direct audio-file URL. Refused unless explicitly enabled (rights/ToS)."""
    if not allow_download:
        raise IngestError(
            "URL inputs are disabled. Supply a local file, or enable allow_download "
            "(--allow-download / allow_download=true) only for audio you own or have rights to."
        )
    name = Path(urllib.parse.urlparse(url).path).name or "download"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / name
    log.info("downloading %s", url)
    req = urllib.request.Request(url, headers={"User-Agent": "audioref/0.1"})
    with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)
    return dest


def decode(path: str | Path, sr: int = SR, start: float | None = None, end: float | None = None) -> np.ndarray:
    """Decode any ffmpeg-readable file to (2, n) float32 at `sr`."""
    path = Path(path)
    if not path.is_file():
        raise IngestError(f"file not found: {path}")
    if have_ffmpeg():
        cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-v", "error"]
        if start:
            cmd += ["-ss", f"{float(start):.6f}"]
        cmd += ["-i", str(path)]
        if end is not None:
            dur = float(end) - float(start or 0.0)
            if dur <= 0:
                raise IngestError("end must be greater than start")
            cmd += ["-t", f"{dur:.6f}"]
        cmd += ["-vn", "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "2", "-ar", str(sr), "-"]
        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode != 0:
            raise IngestError(f"ffmpeg failed to decode {path.name}: {proc.stderr.decode(errors='replace').strip()[:400]}")
        data = np.frombuffer(proc.stdout, dtype=np.float32)
        if data.size < 2:
            raise IngestError(f"no audio decoded from {path.name}")
        return data.reshape(-1, 2).T.copy()
    # fallback without ffmpeg: soundfile handles wav/flac/aiff/ogg
    import soundfile as sf

    info = sf.info(str(path))
    s = int((start or 0) * info.samplerate)
    e = int(end * info.samplerate) if end is not None else None
    x, file_sr = sf.read(str(path), start=s, stop=e, dtype="float32", always_2d=True)
    x = x.T
    if x.shape[0] == 1:
        x = np.repeat(x, 2, axis=0)
    x = x[:2]
    if file_sr != sr:
        x = soxr.resample(x.T, file_sr, sr).T
    return np.ascontiguousarray(x, dtype=np.float32)


def load(path: str | Path, start: float | None = None, end: float | None = None, track_id: str | None = None) -> Audio:
    path = Path(path).expanduser()
    stereo = decode(path, SR, start, end)
    n_min = int(SR * 2.0)
    if stereo.shape[-1] < n_min:
        raise IngestError(f"audio too short ({stereo.shape[-1] / SR:.2f}s); need at least 2 s")
    tid = track_id or audio_hash(stereo)[:16]
    loud = loudness.measure(stereo, SR)
    il = loud["integrated_lufs"]
    gain_db = float(np.clip(TARGET_LUFS - il, -40, 60)) if il is not None else 0.0
    return Audio(path=path, stereo=stereo, sr=SR, start=float(start or 0.0), end=end, track_id=tid, loudness=loud, gain_db=gain_db)


def list_audio_files(folder: str | Path) -> list[Path]:
    folder = Path(folder).expanduser()
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTS and not p.name.startswith("."))


def write_wav(path: Path, x: np.ndarray, sr: int, subtype: str = "PCM_16") -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.atleast_2d(x)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if subtype.startswith("PCM") and peak > 1.0:
        x = x / peak * 0.999
    sf.write(str(path), x.T, sr, subtype=subtype)


def temp_dir() -> Path:
    return Path(tempfile.mkdtemp(prefix="audioref_"))
