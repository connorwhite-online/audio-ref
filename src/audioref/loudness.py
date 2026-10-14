"""EBU R128 / ITU-R BS.1770 loudness: integrated, short-term curve, LRA, true peak."""

from __future__ import annotations

import math

import numpy as np
from scipy import signal

ABS_GATE = -70.0


def _biquad_high_shelf(sr: int, G: float = 4.0, Q: float = 1 / math.sqrt(2), fc: float = 1500.0):
    A = 10 ** (G / 40.0)
    w0 = 2.0 * math.pi * fc / sr
    alpha = math.sin(w0) / (2.0 * Q)
    cw = math.cos(w0)
    sa = 2.0 * math.sqrt(A) * alpha
    b = [A * ((A + 1) + (A - 1) * cw + sa), -2 * A * ((A - 1) + (A + 1) * cw), A * ((A + 1) + (A - 1) * cw - sa)]
    a = [(A + 1) - (A - 1) * cw + sa, 2 * ((A - 1) - (A + 1) * cw), (A + 1) - (A - 1) * cw - sa]
    return np.array(b) / a[0], np.array(a) / a[0]


def _biquad_high_pass(sr: int, Q: float = 0.5, fc: float = 38.0):
    w0 = 2.0 * math.pi * fc / sr
    alpha = math.sin(w0) / (2.0 * Q)
    cw = math.cos(w0)
    b = [(1 + cw) / 2, -(1 + cw), (1 + cw) / 2]
    a = [1 + alpha, -2 * cw, 1 - alpha]
    return np.array(b) / a[0], np.array(a) / a[0]


def k_weight(x: np.ndarray, sr: int) -> np.ndarray:
    """Apply the BS.1770 K-weighting pre-filter. x: (channels, n)."""
    b1, a1 = _biquad_high_shelf(sr)
    b2, a2 = _biquad_high_pass(sr)
    y = signal.lfilter(b1, a1, x, axis=-1)
    return signal.lfilter(b2, a2, y, axis=-1)


def _block_power(kw: np.ndarray, sr: int, win_s: float, hop_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Mean-square power summed over channels for sliding windows. Returns (times, power)."""
    n = kw.shape[-1]
    win = int(round(win_s * sr))
    hop = int(round(hop_s * sr))
    if n < win:
        p = float(np.sum(np.mean(kw**2, axis=-1)))
        return np.array([n / sr / 2]), np.array([p])
    # cumulative sum per channel for O(n) windowing
    csum = np.concatenate([np.zeros((kw.shape[0], 1)), np.cumsum(kw.astype(np.float64) ** 2, axis=-1)], axis=-1)
    starts = np.arange(0, n - win + 1, hop)
    ms = (csum[:, starts + win] - csum[:, starts]) / win
    power = ms.sum(axis=0)
    times = (starts + win / 2) / sr
    return times, power


def _lufs(power: np.ndarray) -> np.ndarray:
    return -0.691 + 10.0 * np.log10(np.maximum(power, 1e-20))


def integrated_loudness(x: np.ndarray, sr: int, kw: np.ndarray | None = None) -> float:
    """Gated integrated loudness (LUFS), 400 ms blocks with 75% overlap."""
    if kw is None:
        kw = k_weight(x, sr)
    _, p = _block_power(kw, sr, 0.4, 0.1)
    l = _lufs(p)
    p = p[l > ABS_GATE]
    if p.size == 0:
        return -math.inf
    rel = _lufs(np.array([p.mean()]))[0] - 10.0
    p2 = p[_lufs(p) > rel]
    if p2.size == 0:
        return -math.inf
    return float(_lufs(np.array([p2.mean()]))[0])


def short_term_curve(x: np.ndarray, sr: int, hop_s: float = 1.0, kw: np.ndarray | None = None):
    """Short-term (3 s) loudness sampled every hop_s seconds. Returns (times, lufs)."""
    if kw is None:
        kw = k_weight(x, sr)
    t, p = _block_power(kw, sr, 3.0, hop_s)
    return t, _lufs(p)


def momentary_curve(x: np.ndarray, sr: int, hop_s: float = 0.1, kw: np.ndarray | None = None):
    if kw is None:
        kw = k_weight(x, sr)
    t, p = _block_power(kw, sr, 0.4, hop_s)
    return t, _lufs(p)


def loudness_range(x: np.ndarray, sr: int, kw: np.ndarray | None = None) -> float:
    """EBU Tech 3342 LRA: spread (P95 - P10) of gated short-term loudness."""
    if kw is None:
        kw = k_weight(x, sr)
    _, p = _block_power(kw, sr, 3.0, 0.1)
    l = _lufs(p)
    p = p[l > ABS_GATE]
    if p.size < 2:
        return 0.0
    rel = _lufs(np.array([p.mean()]))[0] - 20.0
    l = _lufs(p)
    l = l[l > rel]
    if l.size < 2:
        return 0.0
    lo, hi = np.percentile(l, [10, 95])
    return float(hi - lo)


def true_peak_dbtp(x: np.ndarray, oversample: int = 4) -> float:
    """True peak via 4x polyphase oversampling (BS.1770-4 Annex 2 approximation)."""
    peak = 0.0
    for ch in np.atleast_2d(x):
        # process in chunks to bound memory on long files
        step = 1 << 20
        for i in range(0, ch.shape[-1], step):
            seg = ch[max(0, i - 64): i + step + 64]
            up = signal.resample_poly(seg, oversample, 1)
            peak = max(peak, float(np.max(np.abs(up))) if up.size else 0.0)
    return float(20 * np.log10(max(peak, 1e-10)))


def measure(x: np.ndarray, sr: int) -> dict:
    """Full loudness report for a (channels, n) signal."""
    x = np.atleast_2d(x)
    kw = k_weight(x, sr)
    integrated = integrated_loudness(x, sr, kw)
    lra = loudness_range(x, sr, kw)
    tp = true_peak_dbtp(x)
    sample_peak = float(20 * np.log10(max(float(np.max(np.abs(x))), 1e-10)))
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    crest = sample_peak - float(20 * np.log10(max(rms, 1e-10)))
    return {
        "integrated_lufs": integrated if math.isfinite(integrated) else None,
        "lra": lra,
        "true_peak_dbtp": tp,
        "sample_peak_dbfs": sample_peak,
        "crest_factor_db": crest,
    }
