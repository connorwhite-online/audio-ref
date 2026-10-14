"""Tempo, beats, downbeats, meter, tempo curve.

Beat/downbeat candidates come from Beat This! (if installed) or librosa's DP tracker; both are then
refined against a high-resolution onset envelope so the grid is accurate to a few ms (needed for
swing and microtiming).
"""

from __future__ import annotations

import math

import librosa
import numpy as np
from scipy import ndimage, signal

from . import models
from .util import log, mode, parabolic_peak

VERSION = "3"
ENV_SR = 22050
ENV_HOP = 128  # 5.8 ms


# ---------------------------------------------------------------- onset envelopes
def onset_envelopes(y: np.ndarray, sr: int = ENV_SR, hop: int = ENV_HOP) -> dict[str, np.ndarray]:
    """Log-mel spectral-flux onset envelopes: full band plus low/mid/high bands."""
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=hop, center=True)) ** 2
    mel = librosa.feature.melspectrogram(S=S, sr=sr, n_mels=80, fmin=20, fmax=sr / 2)
    logmel = librosa.power_to_db(mel, ref=np.max(mel) + 1e-12, top_db=80)
    flux = np.maximum(0.0, np.diff(logmel, axis=1, prepend=logmel[:, :1]))
    freqs = librosa.mel_frequencies(n_mels=80, fmin=20, fmax=sr / 2)
    bands = {"low": freqs < 160, "mid": (freqs >= 160) & (freqs < 3000), "high": freqs >= 5000}
    out = {"all": flux.mean(axis=0)}
    for k, m in bands.items():
        out[k] = flux[m].mean(axis=0) if m.any() else np.zeros(flux.shape[1])
    out["times"] = librosa.frames_to_time(np.arange(flux.shape[1]), sr=sr, hop_length=hop)
    return out


def _interp(env: np.ndarray, times: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.interp(t, times, env, left=0.0, right=0.0)


# ---------------------------------------------------------------- tempo candidates
def tempo_candidates(env: np.ndarray, sr: int = ENV_SR, hop: int = ENV_HOP, lo: float = 50, hi: float = 220) -> list[dict]:
    """Autocorrelation tempo salience with a mild log-normal prior around 115 BPM."""
    env = env - env.mean()
    ac = librosa.autocorrelate(env, max_size=int(60 / lo * sr / hop) + 2)
    ac = ac / (ac[0] + 1e-12)
    lags = np.arange(len(ac))
    with np.errstate(divide="ignore"):
        bpms = 60.0 * sr / hop / np.maximum(lags, 1e-9)
    valid = (bpms >= lo) & (bpms <= hi)
    peaks, _ = signal.find_peaks(np.where(valid, ac, 0), distance=3)
    cands = []
    for p in peaks:
        lag = parabolic_peak(ac, int(p))
        bpm = 60.0 * sr / hop / lag
        # also reward lags whose multiples are strong (beat-level periodicity)
        harm = np.mean([ac[min(int(round(lag * k)), len(ac) - 1)] for k in (2, 3, 4) if lag * k < len(ac)] or [0])
        prior = math.exp(-0.5 * (math.log2(bpm / 115.0) / 0.9) ** 2)
        cands.append({"bpm": bpm, "salience": float(ac[p]), "score": float((ac[p] + 0.5 * harm) * prior)})
    cands.sort(key=lambda c: -c["score"])
    return cands[:8]


# ---------------------------------------------------------------- beat backends
def _beat_this(y: np.ndarray, sr: int, device: str) -> tuple[np.ndarray, np.ndarray]:
    from beat_this.inference import Audio2Beats

    dev = "cpu" if device == "auto" else device  # beat_this is fast on CPU; MPS gives no benefit here
    model = models.get(f"beat_this:{dev}", lambda: Audio2Beats("final0", device=dev, dbn=False))
    beats, downbeats = model(y.astype(np.float32), sr)
    return np.asarray(beats, dtype=float), np.asarray(downbeats, dtype=float)


def _librosa_beats(env: np.ndarray, times: np.ndarray, sr: int, hop: int, bpm_hint: float | None) -> np.ndarray:
    _, beats = librosa.beat.beat_track(onset_envelope=env, sr=sr, hop_length=hop, bpm=bpm_hint, tightness=300,
                                       trim=False, units="frames")
    return times[np.clip(beats, 0, len(times) - 1)]


# ---------------------------------------------------------------- grid refinement
def _beat_indices(beats: np.ndarray) -> np.ndarray:
    """Integer beat index for each detected beat, tolerant of skipped beats."""
    if len(beats) < 2:
        return np.arange(len(beats))
    ibi = np.median(np.diff(beats))
    idx = [0]
    for i in range(1, len(beats)):
        local = np.median(np.diff(beats[max(0, i - 8): i + 1])) if i >= 2 else ibi
        idx.append(idx[-1] + max(1, int(round((beats[i] - beats[i - 1]) / local))))
    return np.asarray(idx)


def _fit_grid(t0: float, period: float, n: int, env_s: np.ndarray, times: np.ndarray,
              span_period: float = 0.004, span_phase: float = 0.03) -> tuple[float, float, float]:
    """Search period/phase that maximise envelope energy on the grid. Returns (t0, period, score)."""
    best = (t0, period, -1.0)
    k = np.arange(n)
    for pass_ in range(2):
        pr = np.linspace(-span_period, span_period, 41) if pass_ == 0 else np.linspace(-span_period / 10, span_period / 10, 21)
        ph = np.linspace(-span_phase, span_phase, 61) if pass_ == 0 else np.linspace(-span_phase / 15, span_phase / 15, 31)
        c_t0, c_p = best[0] if pass_ else t0, best[1] if pass_ else period
        for dp in pr:
            p = c_p * (1 + dp)
            grid = c_t0 + k[None, :] * p + ph[:, None]
            # anchor the period change at the middle so phase and period decouple
            grid = grid - (n / 2) * (p - c_p)
            vals = _interp(env_s, times, grid).sum(axis=1)
            j = int(np.argmax(vals))
            if vals[j] > best[2]:
                best = (float(c_t0 + ph[j] - (n / 2) * (p - c_p)), float(p), float(vals[j]))
    return best


def refine_grid(beats: np.ndarray, env: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, dict]:
    """Turn raw beat estimates into a smooth, onset-aligned grid. Returns (grid, info)."""
    beats = np.asarray(sorted(beats), dtype=float)
    if len(beats) < 4:
        return beats, {"steady": False, "residual_ms": None}
    idx = _beat_indices(beats)
    A = np.vstack([idx, np.ones_like(idx)]).T
    (period, t0), *_ = np.linalg.lstsq(A, beats, rcond=None)
    resid = beats - (t0 + period * idx)
    resid_ms = float(np.std(resid) * 1000)
    env_s = ndimage.gaussian_filter1d(env / (np.max(env) + 1e-12), 1.0)
    n = int(idx[-1]) + 1
    # steady (programmed) music: one global grid
    if resid_ms < 18.0:
        t0r, pr, _ = _fit_grid(float(t0), float(period), n, env_s, times)
        grid = t0r + pr * np.arange(n)
        return grid, {"steady": True, "residual_ms": resid_ms, "period": pr}
    # drifting tempo: local refinement + smoothing
    full_idx = np.arange(n)
    est = np.interp(full_idx, idx, beats)
    refined = est.copy()
    for i, t in enumerate(est):
        lo, hi = t - 0.035, t + 0.035
        m = (times >= lo) & (times <= hi)
        if m.sum() >= 3:
            seg = env_s[m]
            j = int(np.argmax(seg))
            if seg[j] > 0.05:
                refined[i] = np.interp(parabolic_peak(seg, j), np.arange(m.sum()), times[m])
    win = min(len(refined) - (1 - len(refined) % 2), 9)
    if win >= 5:
        refined = signal.savgol_filter(refined, win, 2)
    return refined, {"steady": False, "residual_ms": resid_ms}


# ---------------------------------------------------------------- meter / downbeats
def _beat_features(grid: np.ndarray, envs: dict, chroma: np.ndarray | None, chroma_times: np.ndarray | None) -> dict[str, np.ndarray]:
    times = envs["times"]
    feats = {}
    for band in ("low", "mid", "high", "all"):
        e = ndimage.maximum_filter1d(envs[band], 5)
        feats[band] = _interp(e, times, grid)
    if chroma is not None and chroma_times is not None and len(grid) > 2:
        # harmonic change at each beat: cosine distance between the half-beats before/after
        c = chroma / (np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-9)
        hc = np.zeros(len(grid))
        ibi = np.median(np.diff(grid))
        for i, t in enumerate(grid):
            a = (chroma_times >= t - ibi) & (chroma_times < t)
            b = (chroma_times >= t) & (chroma_times < t + ibi)
            if a.any() and b.any():
                va, vb = c[:, a].mean(axis=1), c[:, b].mean(axis=1)
                hc[i] = 1 - float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9))
        feats["harm"] = hc
    return feats


def _z(x: np.ndarray) -> np.ndarray:
    s = np.std(x)
    return (x - np.mean(x)) / s if s > 1e-9 else np.zeros_like(x)


def estimate_meter_and_phase(feats: dict[str, np.ndarray], meters=(4, 3)) -> tuple[int, int, float, dict]:
    """Heuristic meter + downbeat phase from beat-level cues (kick, backbeat, harmonic change)."""
    n = len(feats["low"])
    w = {"low": 0.8, "mid": -0.5, "harm": 1.4, "all": 0.3}
    cue = sum(w[k] * _z(feats[k]) for k in w if k in feats)
    scores = {}
    for m in meters:
        if n < 2 * m:
            continue
        ph_scores = []
        for p in range(m):
            sel = np.arange(p, n, m)
            others = np.setdiff1d(np.arange(n), sel)
            ph_scores.append(float(cue[sel].mean() - cue[others].mean()))
        # periodicity of the raw cues at the bar lag (and double) supports the meter itself
        per = 0.0
        for k in ("low", "harm", "mid"):
            if k in feats and n > 2 * m + 2:
                z = _z(feats[k])
                per += float(np.mean(z[:-m] * z[m:])) + 0.5 * float(np.mean(z[:-2 * m] * z[2 * m:]))
        scores[m] = (max(ph_scores), int(np.argmax(ph_scores)), ph_scores, per)
    if not scores:
        return 4, 0, 0.0, {}
    best_m = max(scores, key=lambda m: scores[m][0] + 0.6 * scores[m][3])
    s, phase, ph_scores, _ = scores[best_m]
    srt = sorted(ph_scores, reverse=True)
    conf = float(np.clip((srt[0] - srt[1]) / 1.0, 0, 1)) if len(srt) > 1 else 0.5
    return best_m, phase, conf, {m: {"phase_scores": v[2], "periodicity": v[3]} for m, v in scores.items()}


# ---------------------------------------------------------------- main entry
def analyze(mono: np.ndarray, sr: int, drums: np.ndarray | None = None, chroma: np.ndarray | None = None,
            chroma_times: np.ndarray | None = None, backend: str = "auto", device: str = "auto") -> dict:
    """Rhythm analysis on the normalized mono mix (sr = 22050). `drums` = drum stem mono (same sr) if any."""
    envs_mix = onset_envelopes(mono, sr)
    envs = onset_envelopes(drums, sr) if drums is not None and np.max(np.abs(drums)) > 1e-4 else envs_mix
    times = envs["times"]
    # envelope used for grid refinement: percussive band-sum (kick emphasised)
    ref_env = envs["all"] + 0.7 * envs["low"]
    if np.max(ref_env) < 1e-6:
        ref_env = envs_mix["all"]

    cands = tempo_candidates(envs_mix["all"], sr)
    use_bt = backend in ("auto", "beat_this") and models.capabilities()["beat_this"]
    raw_down = None
    method = "librosa"
    if use_bt:
        try:
            raw_beats, raw_down = _beat_this(mono, sr, device)
            method = "beat_this"
        except Exception as e:  # network/weights problems -> fallback
            log.warning("beat_this failed (%s); using librosa", e)
            use_bt = False
    if not use_bt:
        hint = _choose_octave(cands, envs_mix)
        raw_beats = _librosa_beats(envs_mix["all"], envs_mix["times"], sr, ENV_HOP, hint)

    if len(raw_beats) < 4:
        return {"bpm": None, "beats": raw_beats.tolist(), "downbeats": [], "meter": None, "method": method,
                "confidence": 0.0, "tempo_candidates": cands[:4]}

    grid, ginfo = refine_grid(raw_beats, ref_env, times)
    # keep only the span where music actually plays
    grid = grid[(grid >= 0) & (grid <= len(mono) / sr)]
    ibis = np.diff(grid)
    bpm = 60.0 / ginfo["period"] if ginfo.get("steady") else 60.0 / float(np.median(ibis))

    feats = _beat_features(grid, envs, chroma, chroma_times)
    h_meter, h_phase, h_conf, h_detail = estimate_meter_and_phase(feats)
    meter, phase, dconf, dsource = h_meter, h_phase, h_conf, "heuristic"
    if raw_down is not None and len(raw_down) >= 2:
        # map model downbeats onto grid indices
        di = np.array([int(np.argmin(np.abs(grid - d))) for d in raw_down
                       if np.min(np.abs(grid - d)) < 0.5 * np.median(ibis)])
        if len(di) >= 2:
            diffs = np.diff(di)
            m_model = int(mode(diffs[(diffs >= 2) & (diffs <= 7)].tolist()) or 4)
            phases = (di % m_model).tolist()
            p_model = int(mode(phases))
            agree = phases.count(p_model) / len(phases)
            meter, phase, dconf, dsource = m_model, p_model, float(agree), "beat_this"
            if agree < 0.6 and h_conf > 0.5 and h_meter == m_model:
                phase, dsource = h_phase, "heuristic(model inconsistent)"
    downbeats = grid[phase::meter]

    # tempo curve (local BPM every ~4 bars)
    curve = []
    step = max(4, meter * 4)
    for i in range(0, len(grid) - 1, step):
        seg = np.diff(grid[i: i + step + 1])
        if len(seg):
            curve.append([float(grid[i]), float(60.0 / np.mean(seg))])
    bpm_vals = np.array([c[1] for c in curve]) if curve else np.array([bpm])

    alternatives = _octave_alternatives(bpm, cands)
    conf_tempo = float(np.clip(1.0 - (ginfo.get("residual_ms") or 30) / 60.0, 0.2, 1.0))
    return {
        "bpm": float(bpm),
        "bpm_curve": curve,
        "tempo_stability": {"steady": bool(ginfo.get("steady")), "bpm_std": float(np.std(bpm_vals)),
                            "bpm_range": [float(bpm_vals.min()), float(bpm_vals.max())],
                            "beat_residual_ms": ginfo.get("residual_ms")},
        "meter": f"{meter}/4",
        "beats_per_bar": int(meter),
        "beats": grid.tolist(),
        "downbeats": downbeats.tolist(),
        "first_downbeat": float(downbeats[0]) if len(downbeats) else None,
        "method": method,
        "downbeat_source": dsource,
        "confidence": {"tempo": conf_tempo, "downbeat": float(dconf), "meter": float(max(dconf, h_conf))},
        "tempo_candidates": alternatives,
        "_meter_detail": h_detail,
    }


def _choose_octave(cands: list[dict], envs: dict) -> float | None:
    if not cands:
        return None
    best = cands[0]["bpm"]
    # prefer the octave inside 70..160 BPM when its salience is comparable
    options = [best / 2, best, best * 2]
    in_range = [b for b in options if 70 <= b < 160]
    return float(in_range[0]) if in_range else float(best)


def _octave_alternatives(bpm: float, cands: list[dict]) -> list[dict]:
    """Report the chosen tempo plus half/double-time readings with relative support."""
    out = []
    for factor, label in ((1.0, "detected"), (0.5, "half-time"), (2.0, "double-time")):
        b = bpm * factor
        support = 0.0
        for c in cands:
            if abs(c["bpm"] / b - 1) < 0.04:
                support = max(support, c["salience"])
        out.append({"bpm": round(b, 2), "reading": label, "support": round(support, 3)})
    return out
