"""Pitched note transcription: Spotify Basic Pitch (CoreML/ONNX/TF) with pYIN / NNLS fallbacks.

Note events are tuples (start_s, end_s, midi_pitch, amplitude, bend_semitones) where bend_semitones is the
largest pitch-bend magnitude inside the note (glide/slide indicator; 0 when unknown).
"""

from __future__ import annotations

import contextlib
import sys
import tempfile
from pathlib import Path

import numpy as np

from . import models
from .util import hz_to_midi, log

VERSION = "2"

PARAMS = {
    "bass": dict(onset_threshold=0.45, frame_threshold=0.3, minimum_note_length=70.0, minimum_frequency=27.0,
                 maximum_frequency=420.0),
    "keys": dict(onset_threshold=0.5, frame_threshold=0.3, minimum_note_length=60.0, minimum_frequency=60.0,
                 maximum_frequency=3500.0),
    "melody": dict(onset_threshold=0.5, frame_threshold=0.3, minimum_note_length=80.0, minimum_frequency=80.0,
                   maximum_frequency=2000.0),
}


@contextlib.contextmanager
def quiet_stdout():
    """Basic Pitch's CoreML path print()s debug lines to stdout (which would corrupt an MCP stdio stream)
    and its imports log noisy warnings; send all of it to /dev/null."""
    import logging
    import os
    import warnings

    root = logging.getLogger()
    level = root.level
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull), \
            warnings.catch_warnings():
        warnings.simplefilter("ignore")
        root.setLevel(logging.ERROR)
        try:
            yield
        finally:
            root.setLevel(level)


def _bp_model():
    with quiet_stdout():
        from basic_pitch import ICASSP_2022_MODEL_PATH
        from basic_pitch.inference import Model

        return Model(ICASSP_2022_MODEL_PATH)


def basic_pitch_notes(y: np.ndarray, sr: int, kind: str) -> list[tuple]:
    import soundfile as sf

    model = models.get("basic_pitch", _bp_model)
    with quiet_stdout():
        from basic_pitch.inference import predict
    with tempfile.TemporaryDirectory(prefix="audioref_bp_") as d:
        p = Path(d) / f"{kind}.wav"
        sf.write(str(p), np.asarray(y, dtype=np.float32), sr, subtype="FLOAT")
        with quiet_stdout():
            _, _, events = predict(str(p), model, multiple_pitch_bends=False, melodia_trick=True,
                                   **PARAMS[kind])
    out = []
    for ev in events:
        t0, t1, pitch, amp = float(ev[0]), float(ev[1]), int(ev[2]), float(ev[3])
        bends = np.asarray(ev[4] if len(ev) > 4 and ev[4] is not None else [], dtype=float)  # 1/3-semitone bins
        bend = 0.0
        if len(bends) >= 6:
            head = float(np.median(bends[:3]))
            body = float(np.median(bends[len(bends) // 3:]))
            # a glide starts away from the target pitch and converges onto it
            if abs(head) >= 3 and abs(body) <= 1:
                bend = abs(head) / 3.0
        out.append((t0, t1, pitch, amp, round(bend, 2)))
    out.sort()
    return out


# ------------------------------------------------------------------ fallbacks
def pyin_notes(y: np.ndarray, sr: int, fmin: float, fmax: float, min_len: float = 0.07) -> list[tuple]:
    """Monophonic transcription: pYIN f0 track segmented at pitch changes and onsets."""
    import librosa

    target_sr = 4000 if fmax <= 500 else 11025
    if sr != target_sr:
        import soxr

        y = soxr.resample(y, sr, target_sr)
        sr = target_sr
    if np.max(np.abs(y)) < 1e-4:
        return []
    hop = 64 if target_sr == 4000 else 128
    frame = 512 if target_sr == 4000 else 1024
    f0, voiced, prob = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=sr, frame_length=frame, hop_length=hop,
                                    fill_na=np.nan)
    times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop)
    rms = librosa.feature.rms(y=y, frame_length=frame, hop_length=hop)[0]
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    onsets = set(librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=hop, units="frames").tolist())
    midi = np.where(np.isfinite(f0), 69 + 12 * np.log2(np.where(np.isfinite(f0), f0, 1) / 440.0), np.nan)
    notes = []
    cur = None
    for i in range(len(midi)):
        m = midi[i]
        if not np.isfinite(m) or not voiced[i]:
            if cur:
                notes.append(cur)
                cur = None
            continue
        if cur is None or abs(m - np.median(cur["p"])) > 0.7 or (i in onsets and i - cur["i0"] > 2):
            if cur:
                notes.append(cur)
            cur = {"i0": i, "i1": i, "p": [m], "a": [rms[min(i, len(rms) - 1)]]}
        else:
            cur["i1"] = i
            cur["p"].append(m)
            cur["a"].append(rms[min(i, len(rms) - 1)])
    if cur:
        notes.append(cur)
    out = []
    amax = max((max(n["a"]) for n in notes), default=1.0) or 1.0
    for n in notes:
        t0, t1 = times[n["i0"]], times[n["i1"]] + hop / sr
        if t1 - t0 < min_len:
            continue
        p = np.array(n["p"])
        bend = float(np.max(np.abs(p - np.median(p))))
        out.append((float(t0), float(t1), int(round(float(np.median(p)))), float(max(n["a"]) / amax), round(bend, 2)))
    return out


def salience_notes(X: np.ndarray, beats: np.ndarray, midi_lo: int = 21, rel_thr: float = 0.25,
                   register=(45, 88)) -> list[tuple]:
    """Polyphonic fallback from beat-synchronous NNLS note salience (beat resolution)."""
    if X is None or X.size == 0:
        return []
    beats = np.asarray(beats, dtype=float)
    ends = np.concatenate([beats[1:], [beats[-1] + np.median(np.diff(beats))]])
    out = []
    active: dict[int, list] = {}
    for i in range(X.shape[1]):
        col = X[:, i]
        mx = col.max()
        on = set()
        if mx > 0:
            for k in np.where(col >= rel_thr * mx)[0]:
                m = midi_lo + int(k)
                if register[0] <= m <= register[1]:
                    on.add(m)
        for m in list(active):
            if m not in on:
                a = active.pop(m)
                out.append((a[0], float(beats[i]), m, a[1], 0.0))
        for m in on:
            if m in active:
                active[m][1] = max(active[m][1], float(col[m - midi_lo] / (mx + 1e-12)))
            else:
                active[m] = [float(beats[i]), float(col[m - midi_lo] / (mx + 1e-12))]
    for m, a in active.items():
        out.append((a[0], float(ends[-1]), m, a[1], 0.0))
    out.sort()
    return out


def clean_bass(notes: list[tuple], y: np.ndarray, sr: int) -> list[tuple]:
    """Bass-specific cleanup: drop octave/2-octave doublings and merge same-pitch fragments that have no
    real re-attack (e.g. a sustained note split by a sidechain duck)."""
    import librosa

    if not notes:
        return notes
    notes = sorted(notes)
    keep = [True] * len(notes)
    for i, a in enumerate(notes):
        for j in range(i + 1, len(notes)):
            b = notes[j]
            if b[0] > a[1]:
                break
            ov = min(a[1], b[1]) - max(a[0], b[0])
            if ov <= 0.5 * min(a[1] - a[0], b[1] - b[0]):
                continue
            if abs(a[2] - b[2]) in (12, 24, 19):
                lo, hi = (i, j) if a[2] < b[2] else (j, i)
                if notes[hi][3] > 3 * notes[lo][3]:
                    keep[lo] = False
                else:
                    keep[hi] = False
    notes = [n for n, k in zip(notes, keep) if k]
    hop = 256
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    env = env / (np.percentile(env, 99) + 1e-9)
    peaks = librosa.util.peak_pick(env, pre_max=3, post_max=3, pre_avg=10, post_avg=10, delta=0.08, wait=4)
    pt = librosa.frames_to_time(peaks, sr=sr, hop_length=hop)
    merged: list[list] = []
    for n in notes:
        if merged:
            prev = merged[-1]
            reattack = len(pt) and np.min(np.abs(pt - n[0])) < 0.035
            if n[2] == prev[2] and n[0] - prev[1] < 0.06 and not reattack:
                prev[1] = max(prev[1], n[1])
                prev[3] = max(prev[3], n[3])
                continue
        merged.append(list(n))
    return [tuple(m) for m in merged]


# ------------------------------------------------------------------ entry
def transcribe(kind: str, y: np.ndarray, sr: int, backend: str = "auto", X=None, beats=None) -> dict:
    """kind: bass | keys | melody. Returns {'notes': [...], 'method': str}."""
    if y is None or np.max(np.abs(y)) < 1e-4:
        return {"notes": [], "method": "silent"}
    use_bp = backend in ("auto", "basic_pitch") and models.capabilities()["basic_pitch"]
    if use_bp:
        try:
            notes = basic_pitch_notes(y, sr, kind)
            if kind == "bass":
                notes = clean_bass(notes, y, sr)
            return {"notes": notes, "method": "basic_pitch"}
        except Exception as e:  # pragma: no cover - depends on platform runtime
            log.warning("basic-pitch failed (%s); using fallback", e)
    if kind == "bass":
        return {"notes": pyin_notes(y, sr, 30.0, 400.0), "method": "pyin"}
    if kind == "melody":
        return {"notes": pyin_notes(y, sr, 90.0, 1400.0, 0.08), "method": "pyin"}
    if X is not None and beats is not None:
        return {"notes": salience_notes(X, beats), "method": "nnls-salience"}
    return {"notes": [], "method": "none"}


def midi_range(notes: list[tuple]) -> tuple[int, int] | None:
    if not notes:
        return None
    p = np.array([n[2] for n in notes])
    return int(np.percentile(p, 3)), int(np.percentile(p, 97))


__all__ = ["transcribe", "basic_pitch_notes", "pyin_notes", "salience_notes", "quiet_stdout", "hz_to_midi"]
