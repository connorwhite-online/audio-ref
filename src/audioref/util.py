"""Small shared helpers: JSON conversion, pitch naming, robust statistics."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

log = logging.getLogger("audioref")

PC_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
PC_FLAT = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]


def pc_name(pc: int, flats: bool = False) -> str:
    return (PC_FLAT if flats else PC_SHARP)[int(pc) % 12]


def midi_name(m: int | float, flats: bool = False) -> str:
    m = int(round(m))
    return f"{pc_name(m % 12, flats)}{m // 12 - 1}"


def name_to_pc(name: str) -> int:
    name = name.strip()
    base = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[name[0].upper()]
    for ch in name[1:]:
        if ch in "#♯":
            base += 1
        elif ch in "b♭":
            base -= 1
        else:
            break
    return base % 12


def hz_to_midi(f: float) -> float:
    return 69.0 + 12.0 * math.log2(max(f, 1e-6) / 440.0)


def midi_to_hz(m: float) -> float:
    return 440.0 * 2.0 ** ((m - 69.0) / 12.0)


def db(x: float | np.ndarray, floor: float = 1e-10):
    return 20.0 * np.log10(np.maximum(np.abs(x), floor))


def power_db(x: float | np.ndarray, floor: float = 1e-20):
    return 10.0 * np.log10(np.maximum(x, floor))


def robust_norm(x: np.ndarray, lo: float = 5, hi: float = 95) -> np.ndarray:
    """Scale to roughly [0, 1] using percentiles (outlier tolerant)."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    a, b = np.percentile(x, [lo, hi])
    if b - a < 1e-12:
        return np.zeros_like(x)
    return np.clip((x - a) / (b - a), 0.0, 1.0)


def round_floats(obj: Any, ndigits: int = 4) -> Any:
    return to_jsonable(obj, ndigits)


def to_jsonable(obj: Any, ndigits: int | None = 4) -> Any:
    """Convert numpy/pathlib containers into plain JSON types, rounding floats."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v, ndigits) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist(), ndigits)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        if math.isnan(f) or math.isinf(f):
            return None
        if ndigits is None:
            return f
        r = round(f, ndigits)
        return int(r) if r == int(r) and abs(r) < 1e15 and ndigits <= 0 else r
    if isinstance(obj, Path):
        return str(obj)
    return obj


def write_json(path: Path, obj: Any, ndigits: int | None = 4, indent: int | None = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(to_jsonable(obj, ndigits), indent=indent))
    tmp.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def mode(values: Iterable[Any]) -> Any:
    counts: dict[Any, int] = {}
    order: list[Any] = []
    for v in values:
        if v not in counts:
            order.append(v)
            counts[v] = 0
        counts[v] += 1
    if not counts:
        return None
    return max(order, key=lambda v: counts[v])


def median_or(x: Iterable[float], default: float | None = None) -> float | None:
    arr = np.asarray(list(x), dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else default


def otsu_threshold(x: np.ndarray, bins: int = 64) -> float:
    """Otsu's method on a 1-D sample (used for adaptive present/absent splits). When the classes are well
    separated every threshold in the gap is optimal; return the middle of that plateau."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2 or np.ptp(x) < 1e-12:
        return float(x.mean()) if x.size else 0.0
    hist, edges = np.histogram(x, bins=bins)
    centers = 0.5 * (edges[1:] + edges[:-1])
    w = hist.astype(float)
    cum_w = np.cumsum(w)[:-1]
    cum_m = np.cumsum(w * centers)[:-1]
    total_w, total_m = w.sum(), (w * centers).sum()
    w0, w1 = cum_w, total_w - cum_w
    valid = (w0 > 0) & (w1 > 0)
    m0 = np.where(valid, cum_m / np.maximum(w0, 1e-12), 0)
    m1 = np.where(valid, (total_m - cum_m) / np.maximum(w1, 1e-12), 0)
    between = np.where(valid, w0 * w1 * (m0 - m1) ** 2, -1)
    best = between.max()
    idx = np.where(between >= best * (1 - 1e-9))[0]
    cand = edges[1:-1][idx]
    return float(0.5 * (cand.min() + cand.max()))


def frames_to_time(frames, sr: int, hop: int):
    return np.asarray(frames, dtype=float) * hop / sr


def time_to_frames(t, sr: int, hop: int):
    return np.round(np.asarray(t, dtype=float) * sr / hop).astype(int)


def moving_average(x: np.ndarray, n: int) -> np.ndarray:
    if n <= 1 or len(x) == 0:
        return np.asarray(x, dtype=float)
    k = np.ones(n) / n
    pad = n // 2
    xp = np.pad(np.asarray(x, dtype=float), (pad, n - 1 - pad), mode="edge")
    return np.convolve(xp, k, mode="valid")


def parabolic_peak(y: np.ndarray, i: int) -> float:
    """Sub-sample peak location via parabolic interpolation around index i."""
    if i <= 0 or i >= len(y) - 1:
        return float(i)
    a, b, c = y[i - 1], y[i], y[i + 1]
    den = a - 2 * b + c
    if abs(den) < 1e-12:
        return float(i)
    return i + 0.5 * (a - c) / den


def safe_float(x: Any, default: float | None = None) -> float | None:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default
