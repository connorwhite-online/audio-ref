"""Drum transcription (semi-adaptive NMF on the drum stem) and per-voice sound descriptors.

Pipeline:
  1. log-frequency magnitude spectrogram of the drum stem (5.8 ms hop)
  2. seed kick/snare/hat templates from the track's own onsets; add free components for other sounds
  3. KL-NMF with semi-adaptive seeded templates -> activation per component
  4. label every component (kick, snare, clap, rim, hat_closed, hat_open, ride, crash, tom, perc) from its
     spectrum, attack, decay and burst structure; merge components that are parts of the same sound
  5. peak-pick activations -> hits; refine each hit time in the time domain to ~1 ms; velocity from activation
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage, signal

from .util import log, parabolic_peak

VERSION = "9"
SR = 44100
HOP = 128
N_FFT = 1024
N_BANDS = 64
FMIN, FMAX = 30.0, 16000.0
VOICES = ("kick", "snare", "clap", "rim", "hat_closed", "hat_open", "ride", "crash", "tom", "perc")


# ------------------------------------------------------------------ spectrogram
def _log_filterbank(sr: int = SR, n_fft: int = N_FFT, n_bands: int = N_BANDS) -> tuple[np.ndarray, np.ndarray]:
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    edges = np.geomspace(FMIN, FMAX, n_bands + 2)
    fb = np.zeros((n_bands, len(freqs)))
    for i in range(n_bands):
        lo, c, hi = edges[i], edges[i + 1], edges[i + 2]
        up = (freqs - lo) / (c - lo)
        down = (hi - freqs) / (hi - c)
        fb[i] = np.maximum(0, np.minimum(up, down))
        if fb[i].sum() == 0:  # very narrow low bands: nearest bin
            fb[i, int(np.argmin(np.abs(freqs - c)))] = 1.0
    fb /= fb.sum(axis=1, keepdims=True)
    return fb, edges[1:-1]


def spectrogram(x: np.ndarray, sr: int = SR) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (V [bands x frames] magnitude, band centre freqs, frame times)."""
    f, t, Z = signal.stft(x, fs=sr, nperseg=N_FFT, noverlap=N_FFT - HOP, boundary="even", padded=True)
    fb, centers = _log_filterbank(sr)
    V = fb @ np.abs(Z)
    times = t
    return V.astype(np.float64), centers, times


# ------------------------------------------------------------------ component analysis
def _template_features(w: np.ndarray, centers: np.ndarray) -> dict:
    p = w / (w.sum() + 1e-12)
    centroid = float(np.exp(np.sum(p * np.log(centers))))
    flat = float(np.exp(np.mean(np.log(w + 1e-12))) / (np.mean(w) + 1e-12))
    return {
        "centroid_hz": centroid,
        "low": float(p[centers < 150].sum()),
        "lowmid": float(p[(centers >= 150) & (centers < 500)].sum()),
        "mid": float(p[(centers >= 500) & (centers < 4000)].sum()),
        "high": float(p[centers >= 5000].sum()),
        "flatness": flat,
        "peak_hz": float(centers[int(np.argmax(w))]),
    }


def _label_component(tf: dict, shape: dict, n_events: int, duration: float) -> str:
    rate = n_events / max(duration, 1e-6)
    if tf["low"] > 0.45 or (tf["centroid_hz"] < 140 and tf["high"] < 0.2):
        return "kick"
    if tf["high"] > 0.55:
        if shape["attack_ms"] > 7:
            return "perc"
        if shape["decay_ms"] > 450 and rate < 1.0:
            return "crash"
        if shape["decay_ms"] > 260:
            return "ride"
        if shape["decay_ms"] > 85:
            return "hat_open"
        return "hat_closed"
    if tf["peak_hz"] < 420 and tf["flatness"] < 0.25 and shape["decay_ms"] > 90 and tf["high"] < 0.15:
        return "tom"
    if tf["high"] > 0.35 and tf["mid"] < 0.35:
        return "perc"
    return "snare"  # refined to clap/rim per hit below


# ------------------------------------------------------------------ hit refinement
_BANDS = {"kick": (30, 180), "snare": (150, 8000), "clap": (600, 5000), "rim": (800, 6000), "tom": (60, 1500),
          "hat_closed": (5000, 16000), "hat_open": (5000, 16000), "ride": (3000, 16000), "crash": (3000, 16000),
          "perc": (3000, 14000)}


def _hilbert_env(y: np.ndarray, chunk: int = 1 << 19, pad: int = 4096) -> np.ndarray:
    """Analytic-signal envelope computed in overlapping chunks (bounded memory/FFT size)."""
    out = np.empty_like(y)
    for i in range(0, len(y), chunk):
        a, b = max(0, i - pad), min(len(y), i + chunk + pad)
        e = np.abs(signal.hilbert(y[a:b]))
        out[i: min(i + chunk, len(y))] = e[i - a: i - a + min(chunk, len(y) - i)]
    return out


def _band_envelopes(x: np.ndarray, sr: int) -> dict[str, np.ndarray]:
    out = {}
    cache: dict[tuple, np.ndarray] = {}
    for v, (lo, hi) in _BANDS.items():
        if (lo, hi) not in cache:
            sos = signal.butter(4, [lo / (sr / 2), min(hi / (sr / 2), 0.99)], "bandpass", output="sos")
            y = signal.sosfiltfilt(sos, x)
            env = _hilbert_env(y)
            cache[(lo, hi)] = ndimage.uniform_filter1d(env, int(0.0007 * sr))
        out[v] = cache[(lo, hi)]
    return out


def _refine_time(env: np.ndarray, t: float, sr: int, pre: float = 0.015, post: float = 0.03) -> tuple[float, float]:
    """Attack time = first crossing of 30% of the local peak in [t-pre, t+post]. Returns (t, peak)."""
    i0 = max(0, int((t - pre) * sr))
    i1 = min(len(env), int((t + post) * sr))
    seg = env[i0:i1]
    if seg.size < 4:
        return t, 0.0
    pk = int(np.argmax(seg))
    level = seg[pk]
    # walk back from the peak to the 30% crossing
    j = pk
    while j > 0 and seg[j] > 0.3 * level:
        j -= 1
    # sub-sample linear interpolation
    if seg[j + 1] != seg[j]:
        frac = (0.3 * level - seg[j]) / (seg[j + 1] - seg[j])
    else:
        frac = 0.0
    return (i0 + j + float(np.clip(frac, 0, 1))) / sr, float(level)


def _is_clap(env_mid: np.ndarray, t: float, sr: int) -> bool:
    """Claps are 2-4 micro-bursts 6-18 ms apart with deep dips between them (snare noise just decays)."""
    i0, i1 = int((t - 0.003) * sr), int((t + 0.045) * sr)
    seg = env_mid[max(0, i0): i1]
    if seg.size < 10:
        return False
    seg = ndimage.uniform_filter1d(seg, int(0.0025 * sr))
    pk, _ = signal.find_peaks(seg, height=0.25 * seg.max(), distance=int(0.005 * sr))
    deep = 0
    for a, b in zip(pk[:-1], pk[1:]):
        gap_ms = (b - a) / sr * 1000
        if 6 <= gap_ms <= 18 and seg[a:b].min() < 0.35 * min(seg[a], seg[b]):
            deep += 1
    return deep >= 1


def _sustain_db(env: np.ndarray, t: float, after_s: float, sr: int) -> float:
    """Level `after_s` after a hit relative to its peak (dB), background removed."""
    i0 = int(t * sr)
    bg = float(np.min(env[max(0, i0 - int(0.012 * sr)): max(1, i0 - int(0.001 * sr))])) if i0 > 10 else 0.0
    pk = float(np.max(env[i0: i0 + int(0.01 * sr)])) if i0 < len(env) else 0.0
    j = i0 + int(after_s * sr)
    if j >= len(env) or pk - bg <= 0:
        return -99.0
    lvl = float(np.mean(env[j: j + int(0.02 * sr)])) - bg
    return float(20 * np.log10(max(lvl, 1e-9) / (pk - bg)))


def _band_energy(x: np.ndarray, t: float, sr: int, lo: float, hi: float, dur: float = 0.04) -> float:
    i0 = max(0, int((t - 0.002) * sr))
    seg = x[i0: i0 + int(dur * sr)]
    if seg.size < 64:
        return 0.0
    sp = np.abs(np.fft.rfft(seg * np.hanning(seg.size))) ** 2
    f = np.fft.rfftfreq(seg.size, 1 / sr)
    return float(sp[(f >= lo) & (f < hi)].sum())


def _mid_hf_ratio(x: np.ndarray, t: float, sr: int) -> float:
    """Attack energy ratio (0.8-5 kHz) / (5-12 kHz) in the first 20 ms after t."""
    i0 = max(0, int((t - 0.002) * sr))
    seg = x[i0: i0 + int(0.022 * sr)]
    if seg.size < 64:
        return 1.0
    sp = np.abs(np.fft.rfft(seg * np.hanning(seg.size))) ** 2
    f = np.fft.rfftfreq(seg.size, 1 / sr)
    mid = sp[(f >= 800) & (f < 5000)].sum()
    hf = sp[(f >= 5000) & (f < 12000)].sum() + 1e-12
    return float(mid / hf)


def _presence_threshold(c: np.ndarray, ref: float, lo: float = 0.12, hi: float = 0.4) -> float:
    """Split an atom's per-event coefficients into cross-talk vs real hits (Otsu in the log domain),
    bounded to [lo, hi] x the atom's typical hit level."""
    from .util import otsu_threshold

    pos = c[c > 1e-4 * ref]
    if len(pos) < 6:
        return 0.3 * ref
    thr = float(np.exp(otsu_threshold(np.log(pos))))
    return float(np.clip(thr, lo * ref, hi * ref))


def _first_onset(env: np.ndarray, t_lo: float, t_hi: float, sr: int, max_gap_s: float = 0.008) -> float:
    """Earliest 30%-of-peak crossing in [t_lo, t_hi] that stays continuous (no gap below 12% of the peak
    longer than max_gap) up to the peak. Finds a clap's first micro-burst rather than its loudest one."""
    i0, i1 = max(0, int(t_lo * sr)), min(len(env), int(t_hi * sr))
    seg = env[i0:i1]
    if seg.size < 8:
        return t_lo
    p = int(np.argmax(seg))
    on, gap_thr = 0.3 * seg[p], 0.12 * seg[p]
    above = np.where(seg[: p + 1] >= on)[0]
    if not len(above):
        return (i0 + p) / sr
    i = int(above[0])
    max_gap = int(max_gap_s * sr)
    run = 0
    for j in range(i, p + 1):
        if seg[j] < gap_thr:
            run += 1
            if run > max_gap:
                nxt = above[above > j]
                if not len(nxt):
                    break
                i, run = int(nxt[0]), 0
        else:
            run = 0
    # sub-sample interpolation of the crossing
    if i > 0 and seg[i] != seg[i - 1]:
        frac = (on - seg[i - 1]) / (seg[i] - seg[i - 1])
        return (i0 + i - 1 + float(np.clip(frac, 0, 1))) / sr
    return (i0 + i) / sr


def _snare_family(env_mid: np.ndarray, t: float, sr: int, atom_decay_ms: float, near_kick: bool) -> str:
    """clap (micro-bursts) | rim (very short: per hit when away from a kick click, else per atom) | snare."""
    if _is_clap(env_mid, t, sr):
        return "clap"
    if 0 < atom_decay_ms < 40:
        return "rim"
    if not near_kick:
        i0 = int(t * sr)
        seg = env_mid[i0: i0 + int(0.15 * sr)]
        if seg.size > 10:
            pk = int(np.argmax(seg[: int(0.02 * sr)]))
            below = np.where(seg[pk:] < 0.1 * seg[pk])[0]
            if len(below) and below[0] / sr < 0.035:
                return "rim"
    return "snare"


def _shape_from_envelope(env: np.ndarray, times: list[float], sr: int) -> dict:
    """Median attack (10->90 %) and decay (-20 dB) of a voice from its band envelope at isolated hits."""
    atts, decs = [], []
    for t in times:
        i0 = int((t - 0.01) * sr)
        i1 = int((t + 1.2) * sr)
        if i0 < 0 or i1 > len(env):
            continue
        seg = env[i0:i1]
        pk = int(np.argmax(seg[: int(0.06 * sr)]))
        lvl = seg[pk] + 1e-12
        pre = seg[: pk + 1]
        a10 = np.where(pre >= 0.1 * lvl)[0]
        a90 = np.where(pre >= 0.9 * lvl)[0]
        if len(a10) and len(a90):
            atts.append((a90[0] - a10[0]) / sr * 1000)
        post = seg[pk:]
        below = np.where(post < 0.1 * lvl)[0]
        decs.append((below[0] if len(below) else len(post)) / sr * 1000)
    if not decs:
        return {"attack_ms": 0.0, "decay_ms": 0.0}
    return {"attack_ms": float(np.median(atts)) if atts else 0.0, "decay_ms": float(np.median(decs))}


# ------------------------------------------------------------------ main
def _env_band_for(tf: dict) -> str:
    if tf["low"] > 0.45 or tf["centroid_hz"] < 140:
        return "kick"
    if tf["high"] > 0.55:
        return "hat_closed"
    if tf["peak_hz"] < 420 and tf["high"] < 0.15:
        return "tom"
    return "snare"


KICK_BIAS_S = 0.0019  # zero-phase band-pass pre-ringing makes the low-band 30% crossing ~2 ms early


def _decay_slope_db_per_ms(env: np.ndarray, t: float, gap_s: float, sr: int) -> float:
    """Initial decay rate (dB/ms) of a hit's own energy above the background that was already sounding
    (e.g. a ringing open hat under a closed hat), measured until the next hit or 45 ms."""
    i0 = int(t * sr)
    pk_win = env[i0: i0 + int(0.008 * sr)]
    if pk_win.size < 4:
        return 0.0
    bg = float(np.min(env[max(0, i0 - int(0.012 * sr)): max(1, i0 - int(0.001 * sr))])) if i0 > 10 else 0.0
    pk = i0 + int(np.argmax(pk_win))
    end = min(pk + int(min(0.045, max(gap_s - 0.004, 0.01)) * sr), len(env))
    seg = env[pk + int(0.002 * sr): end] - 0.95 * bg
    top = env[pk] - 0.95 * bg
    if seg.size < 8 or top <= 0:
        return 0.0
    d = 20 * np.log10(np.maximum(seg, top * 1e-3) / top)
    tt = np.arange(seg.size) / sr * 1000
    return float(np.polyfit(tt, d, 1)[0])


def _flux_peaks(L: np.ndarray, frame_s: float, offset: float) -> tuple[np.ndarray, np.ndarray]:
    flux = np.maximum(0, L[:, 2:] - L[:, :-2]).sum(axis=0)
    flux = np.concatenate([[0, 0], flux])
    flux = flux / (np.percentile(flux, 99.5) + 1e-12)
    thr = ndimage.median_filter(flux, int(0.4 / frame_s) | 1) + offset
    peaks, _ = signal.find_peaks(flux, height=thr, distance=max(2, int(0.03 / frame_s)))
    return peaks, flux


def _detect_events(L: np.ndarray, frame_s: float, centers: np.ndarray | None = None) -> np.ndarray:
    """Onset frames: full-band whitened log flux, plus a high-band detector so quiet hats under ringing
    cymbals are not lost (lag-2 difference gives a sharp peak at small hops)."""
    peaks, flux = _flux_peaks(L, frame_s, 0.035)
    if centers is None:
        return peaks
    hp, _ = _flux_peaks(L[centers >= 5000], frame_s, 0.05)
    tol = int(0.02 / frame_s)
    extra = [p for p in hp if len(peaks) == 0 or np.min(np.abs(peaks - p)) > tol]
    return np.sort(np.concatenate([peaks, np.asarray(extra, dtype=int)])).astype(int)


def _attack_spectra(Vw: np.ndarray, events: np.ndarray, frame_s: float) -> np.ndarray:
    """Per-event attack spectrum: max over the first 25 ms minus the level just before the onset."""
    post_n = max(2, int(0.032 / frame_s))
    pre_a, pre_b = int(0.024 / frame_s), int(0.006 / frame_s)
    A = np.zeros((Vw.shape[0], len(events)))
    for j, f in enumerate(events):
        post = Vw[:, f: f + post_n].max(axis=1)
        pre = Vw[:, max(0, f - pre_a): max(1, f - pre_b)].max(axis=1)
        A[:, j] = np.maximum(post - pre, 0)
    return A


def _cluster_events(An: np.ndarray, max_dist: float = 0.14, max_n: int = 2500) -> np.ndarray:
    """Average-linkage cosine clustering of normalised attack spectra. Returns a label per event."""
    from scipy.cluster.hierarchy import fcluster, linkage

    n = An.shape[1]
    if n < 2:
        return np.zeros(n, dtype=int)
    idx = np.arange(n)
    if n > max_n:  # cluster a subsample, assign the rest to the nearest centroid
        idx = np.random.default_rng(0).choice(n, max_n, replace=False)
    Z = linkage(An[:, idx].T, method="average", metric="cosine")
    lab_sub = fcluster(Z, t=max_dist, criterion="distance")
    if len(idx) == n:
        return lab_sub
    cents = {l: An[:, idx[lab_sub == l]].mean(axis=1) for l in np.unique(lab_sub)}
    keys = list(cents)
    C = np.stack([cents[k] / (np.linalg.norm(cents[k]) + 1e-12) for k in keys], axis=1)
    return np.array([keys[int(np.argmax(An[:, j] @ C))] for j in range(n)])


def _band_masks(centers: np.ndarray) -> dict[str, np.ndarray]:
    return {"low": centers < 120, "lowmid": (centers >= 120) & (centers < 500),
            "mid": (centers >= 900) & (centers < 5000), "high": centers >= 7000}


def _profile(X: np.ndarray, own: np.ndarray) -> np.ndarray:
    """Median in the voice's own bands, 20th percentile elsewhere (drops sometimes-layered sounds)."""
    return np.where(own, np.median(X, axis=1), np.percentile(X, 20, axis=1))


def _exemplar_atoms(A: np.ndarray, centers: np.ndarray, ev_times: np.ndarray) -> tuple[list[np.ndarray], list[str], list[np.ndarray]]:
    """Base atoms (kick, snare, hat) from each voice's most characteristic events. Using a low percentile
    across exemplars keeps only what every hit of that voice shares, so layered hats/shakers drop out."""
    from .util import otsu_threshold

    m = _band_masks(centers)
    E = {k: A[v].sum(axis=0) for k, v in m.items()}
    tot = A.sum(axis=0) + 1e-12
    atoms, names, exemplars = [], [], []

    lo = E["low"]
    # "tails": a kick's pitch sweep reaching the lowest bands ~30-45 ms after the real onset
    tail = np.zeros(len(lo), dtype=bool)
    for j in range(1, len(lo)):
        i = j - 1
        while i >= 0 and ev_times[j] - ev_times[i] < 0.05:
            if lo[i] > lo[j]:
                tail[j] = True
                break
            i -= 1
    kick_ex = np.array([], dtype=int)
    if np.any(lo > 0):
        thr = otsu_threshold(np.log(lo[lo > 0] + 1e-9))
        cand = np.where((np.log(lo + 1e-9) > thr) & ~tail)[0]
        if len(cand) >= 3:
            kick_ex = cand[lo[cand] >= np.percentile(lo[cand], 40)]
            prof = _profile(A[:, kick_ex] / lo[kick_ex], centers < 500) * np.median(lo[kick_ex])
            atoms.append(prof)
            names.append("kick")
            exemplars.append(kick_ex)
    kick_level = float(np.median(lo[kick_ex])) if len(kick_ex) else np.inf

    hi = E["high"]
    f_hi = hi / tot
    hat_ex = np.where((f_hi > 0.5) & (lo < 0.1 * kick_level))[0]
    if len(hat_ex) >= 3:
        hat_ex = hat_ex[f_hi[hat_ex] >= np.percentile(f_hi[hat_ex], 30)]
        prof = np.median(A[:, hat_ex] / hi[hat_ex], axis=1) * np.median(hi[hat_ex])
        atoms.append(prof)
        names.append("hat")
        exemplars.append(hat_ex)

    md = E["mid"] + E["lowmid"]
    f_md = md / tot
    cand = np.where((lo < 0.15 * kick_level) & (f_md > 0.35) & (f_hi < 0.6))[0]
    if len(cand) >= 3:
        thr = otsu_threshold(np.log(md[cand] + 1e-9))
        sn_ex = cand[np.log(md[cand] + 1e-9) >= thr]
        if len(sn_ex) >= 3:
            prof = _profile(A[:, sn_ex] / md[sn_ex], centers < 5000) * np.median(md[sn_ex])
            atoms.append(prof)
            names.append("snare")
            exemplars.append(sn_ex)
    return atoms, names, exemplars


def _purify_layered_hats(A, base, names, exemplars, centers, ev_times) -> None:
    """If pure hats tick on a steady grid around the kick/snare exemplars, the hat is almost certainly still
    playing under those hits: remove the hat-shaped high end from the kick/snare atoms so NNLS can report it."""
    from scipy.optimize import nnls

    if "hat" not in names:
        return
    h = names.index("hat")
    hat = base[h]
    hf = centers >= 5000
    M = np.stack(base, axis=1)
    coefs = np.array([nnls(M, A[:, j])[0] for j in range(A.shape[1])]).T
    ref = float(np.median(coefs[h, exemplars[h]])) if len(exemplars[h]) else float(coefs[h].max())
    hat_t = ev_times[coefs[h] > 0.3 * ref]
    if len(hat_t) < 8:
        return
    d = np.diff(hat_t)
    d = d[(d > 0.06) & (d < 0.8)]
    if len(d) < 4:
        return
    step = float(np.median(d))
    for k, name in enumerate(names):
        if name == "hat" or not len(exemplars[k]):
            continue
        on_grid = 0
        for j in exemplars[k]:
            t = ev_times[j]
            for s_ in (step / 2, step):  # masked hats make the detected spacing look doubled
                if np.any(np.abs(hat_t - (t - s_)) < 0.02) or np.any(np.abs(hat_t - (t + s_)) < 0.02):
                    on_grid += 1
                    break
        frac = on_grid / len(exemplars[k])
        x_hf, hat_hf = base[k][hf], hat[hf]
        cos = float(x_hf @ hat_hf / (np.linalg.norm(x_hf) * np.linalg.norm(hat_hf) + 1e-12))
        if frac > 0.6 and cos > 0.9:
            beta = float(x_hf @ hat_hf / (hat_hf @ hat_hf + 1e-12))
            base[k] = np.maximum(base[k] - beta * hat, 0)


def _extra_atoms(A: np.ndarray, base: list[np.ndarray], min_size: int) -> list[np.ndarray]:
    """Sounds the base atoms cannot explain (clap layered on kick, rim, toms, cowbell, crash ...):
    cluster the per-event NNLS residuals and keep each cluster's median residual as an extra atom."""
    from scipy.optimize import nnls

    if not base:
        return []
    M = np.stack(base, axis=1)
    R = np.zeros_like(A)
    for j in range(A.shape[1]):
        coef, _ = nnls(M, A[:, j])
        R[:, j] = np.maximum(A[:, j] - M @ coef, 0)
    frac = R.sum(axis=0) / (A.sum(axis=0) + 1e-12)
    cand = np.where(frac > 0.25)[0]
    if len(cand) < min_size:
        return []
    Rn = R[:, cand] / (np.linalg.norm(R[:, cand], axis=0, keepdims=True) + 1e-12)
    labels = _cluster_events(Rn, max_dist=0.2)
    extras = []
    for l in np.unique(labels):
        members = cand[labels == l]
        if len(members) >= min_size:
            extras.append(np.median(R[:, members], axis=1))
    return extras


def transcribe(drums_mono: np.ndarray, sr: int = SR) -> dict:
    """Transcribe a mono drum signal. Returns {'hits': {voice: [[t, vel], ...]}, 'atoms': [...]}"""
    from scipy.optimize import nnls

    x = np.asarray(drums_mono, dtype=np.float64)
    duration = len(x) / sr
    if np.max(np.abs(x)) < 1e-4:
        return {"hits": {}, "atoms": [], "method": "onset-atoms", "empty": True}
    V, centers, times = spectrogram(x, sr)
    frame_s = HOP / sr
    scale = np.percentile(V, 97, axis=1)
    scale = np.maximum(scale, 0.01 * scale.max()) + 1e-9  # never boost a near-empty band by > 40 dB
    Vw = V / scale[:, None]  # whitened magnitude
    L = np.log1p(10 * Vw)
    events = _detect_events(L, frame_s, centers)
    if len(events) < 4:
        return {"hits": {}, "atoms": [], "method": "onset-atoms", "empty": True}
    A = _attack_spectra(Vw, events, frame_s) ** 0.7
    energy = A.sum(axis=0)
    keep = energy > 0.02 * np.percentile(energy, 95)
    events, A = events[keep], A[:, keep]
    base, base_names, base_ex = _exemplar_atoms(A, centers, times[events])
    _purify_layered_hats(A, base, base_names, base_ex, centers, times[events])
    extras = _extra_atoms(A, base, min_size=max(4, int(0.01 * len(events))))
    atoms = base + extras
    if not atoms:
        return {"hits": {}, "atoms": [], "method": "onset-atoms", "empty": True}
    M = np.stack(atoms, axis=1)
    coefs = np.zeros((M.shape[1], A.shape[1]))
    for j in range(A.shape[1]):
        coefs[:, j], _ = nnls(M, A[:, j])
    envs = _band_envelopes(x, sr)
    ev_times = times[events]
    contrib = coefs * M.sum(axis=0)[:, None]  # energy explained by each atom per event
    share = contrib / (contrib.sum(axis=0, keepdims=True) + 1e-12)

    info = []
    for k in range(M.shape[1]):
        c = coefs[k]
        if k < len(base_ex) and len(base_ex[k]):
            ref = float(np.median(c[base_ex[k]]))
        else:
            strong = c > 0.2 * (c.max() + 1e-12)
            ref = float(np.percentile(c[strong], 60)) if strong.any() else float(c.max())
        w_orig = (M[:, k] ** (1 / 0.7)) * scale
        tf = _template_features(w_orig, centers)
        dom = np.where(share[k] > 0.6)[0]
        iso = [ev_times[j] for j in dom if (j + 1 >= len(ev_times) or ev_times[j + 1] - ev_times[j] > 0.06)]
        shape = _shape_from_envelope(envs[_env_band_for(tf)], iso[:20], sr)
        bounds = (0.3, 0.5) if _env_band_for(tf) == "kick" else (0.12, 0.4)
        present = c > _presence_threshold(c, ref, *bounds)
        label = _label_component(tf, shape, int(present.sum()), duration)
        info.append({"k": k, "tf": tf, "ref": ref, "present": present, "label": label, **shape})

    raw: dict[str, list[list]] = {}
    kick_events = {}
    order = sorted(info, key=lambda a: a["label"] != "kick")  # kicks first: snare/clap timing uses them
    for a in order:
        label = a["label"]
        for j in np.where(a["present"])[0]:
            t0 = float(ev_times[j])
            if label == "snare":
                t_ref = _first_onset(envs["clap"], t0 - 0.012, t0 + 0.04, sr)
                if j in kick_events:
                    kt = kick_events[j]
                    t_late = _first_onset(envs["clap"], kt + 0.004, t0 + 0.04, sr)
                    # a laid-back snare/clap: the first onset found was the kick's click
                    if abs(t_ref - kt) < 0.004 and t_late - kt > 0.006:
                        t_ref = t_late
            else:
                band = "clap" if label == "snare" else label
                t_ref, _ = _refine_time(envs[band], t0, sr, pre=0.012, post=0.04)
                if label == "kick":
                    t_ref += KICK_BIAS_S
            if abs(t_ref - t0) > 0.045:
                t_ref = t0
            if label == "kick":
                kick_events[j] = t_ref
            raw.setdefault(label, []).append([t_ref, float(coefs[a["k"], j] / (a["ref"] + 1e-12)), a])
    # validate snare-family hits: a real snare/clap/rim has at least as much 0.8-5 kHz attack energy as
    # 5-12 kHz energy (hat/shaker bleed is HF-heavy), and on a kick event it needs a clear coefficient
    if "snare" in raw:
        kts = np.array(sorted(h[0] for h in raw.get("kick", [])))
        sn_t = np.array(sorted(h[0] for h in raw["snare"]))
        # typical 0.8-5 kHz attack energy of kicks that have no snare-family hit (click + bleed baseline)
        solo = [k for k in kts if not len(sn_t) or np.min(np.abs(sn_t - k)) > 0.03]
        base = float(np.median([_band_energy(x, k, sr, 800, 5000) for k in solo[:200]])) if len(solo) >= 4 else 0.0
        kept = []
        for h in raw["snare"]:
            t = h[0]
            if _mid_hf_ratio(x, t, sr) < 0.35:
                continue  # HF-heavy: hat/shaker bleed, not a snare/clap/rim
            if base > 0 and len(kts) and np.min(np.abs(kts - t)) < 0.025 and \
                    _band_energy(x, t, sr, 800, 5000) < 2.0 * base:
                continue  # nothing beyond what a kick alone puts in the mid band
            kept.append(h)
        raw["snare"] = kept
    hits: dict[str, list[list[float]]] = {}
    all_t = np.array(sorted(h[0] for hs in raw.values() for h in hs))
    kick_t = np.array(sorted(h[0] for h in raw.get("kick", [])))
    for label, hs in raw.items():
        hs.sort(key=lambda h: h[0])
        hat_default = "hat_open" if label == "hat_open" else "hat_closed"
        for i, (t, amp, a) in enumerate(hs):
            voice = label
            if label == "snare":
                near_kick = bool(len(kick_t) and np.min(np.abs(kick_t - t)) < 0.02)
                voice = _snare_family(envs["clap"], t, sr, a["decay_ms"], near_kick)
            elif label in ("hat_closed", "hat_open"):
                layered = np.sum(np.abs(all_t - t) < 0.015) > 1
                gap = hs[i + 1][0] - t if i + 1 < len(hs) else 1.0
                if gap > 0.45 and _sustain_db(envs["crash"], t, 0.4, sr) > -10:
                    voice = "crash"  # long, loud ring-out: crash/ride wash rather than an open hat
                elif layered:
                    voice = hat_default  # a co-sounding snare/kick masks the hat's own decay
                else:
                    slope = _decay_slope_db_per_ms(envs["hat_closed"], t, gap, sr)
                    voice = "hat_open" if slope > -0.22 else "hat_closed"
            hits.setdefault(voice, []).append([t, amp])

    # a clap's/snare's tail can trigger a second event ~20-35 ms later: keep the first of the family
    fam = sorted([(t, amp, v) for v in ("snare", "clap", "rim") for t, amp in hits.get(v, [])])
    kept: list[tuple] = []
    for t, amp, v in fam:
        if kept and t - kept[-1][0] < 0.04:
            continue
        kept.append((t, amp, v))
    for v in ("snare", "clap", "rim"):
        if v in hits:
            hits[v] = [[t, amp] for t, amp, vv in kept if vv == v]
            if not hits[v]:
                del hits[v]
    out_hits: dict[str, list[list[float]]] = {}
    for voice, hs in hits.items():
        hs.sort()
        merged: list[list[float]] = []
        window = 0.045 if voice in ("kick", "tom") else 0.012
        for t, amp in hs:
            if merged and t - merged[-1][0] < window:
                if amp > merged[-1][1]:
                    merged[-1] = [min(t, merged[-1][0]), amp]
                continue
            merged.append([t, amp])
        amps = np.array([m[1] for m in merged])
        ref = np.percentile(amps, 95) if len(amps) else 1.0
        out_hits[voice] = [[round(t, 5), round(float(np.clip(amp / (ref + 1e-12), 0.05, 1.0)), 3)] for t, amp in merged]

    atom_info = [{"label": a["label"], "n_hits": int(a["present"].sum()), "centroid_hz": round(a["tf"]["centroid_hz"], 1),
                  "decay_ms": round(a["decay_ms"], 1), "attack_ms": round(a["attack_ms"], 1),
                  "low": round(a["tf"]["low"], 2), "high": round(a["tf"]["high"], 2)} for a in info]
    return {"hits": out_hits, "atoms": atom_info, "method": "onset-atoms", "n_events": int(len(events))}


# ------------------------------------------------------------------ sound descriptors
_DESC_BANDS = {"kick": (25, 2500), "snare": (120, 12000), "clap": (500, 8000), "rim": (300, 8000), "tom": (50, 3000),
               "hat_closed": (4000, 18000), "hat_open": (4000, 18000), "ride": (2500, 18000), "crash": (2500, 18000),
               "perc": (2000, 16000)}


def describe_sounds(drums_mono: np.ndarray, hits: dict[str, list], sr: int = SR, max_hits: int = 24) -> dict:
    """Per-voice descriptors: pitch centre, decay, brightness, transient sharpness, saturation.
    Each voice is measured in its own band so co-sounding voices (a kick under a clap) don't leak in."""
    x = np.asarray(drums_mono, dtype=np.float64)
    out = {}
    all_times = sorted(t for hs in hits.values() for t, _ in hs)
    all_times_arr = np.array(all_times) if all_times else np.array([0.0])
    for voice, hs in hits.items():
        if len(hs) < 2:
            continue
        lo, hi = _DESC_BANDS.get(voice, (30, 16000))
        sos = signal.butter(4, [lo / (sr / 2), min(hi / (sr / 2), 0.99)], "bandpass", output="sos")
        xb = signal.sosfiltfilt(sos, x)
        cands = []
        for t, v in hs:
            others = all_times_arr[np.abs(all_times_arr - t) > 0.004]
            gap_after = np.min(others[others > t] - t) if np.any(others > t) else 1.0
            cands.append((v * min(gap_after, 0.4), t))
        cands.sort(reverse=True)
        chosen = [t for sc, t in cands[:max_hits] if sc > 0] or [hs[0][0]]
        seg_len = int(0.5 * sr)
        segs, raw = [], []
        for t in chosen:
            i = int(t * sr)
            if i + seg_len < len(x) and i >= 0:
                segs.append(xb[i: i + seg_len])
                raw.append(x[i: i + seg_len])
        if not segs:
            continue
        d = _describe_segments(np.array(segs), sr, voice, np.array(raw))
        d["count"] = len(hs)
        out[voice] = d
    return out


def _describe_segments(segs: np.ndarray, sr: int, voice: str, raw: np.ndarray) -> dict:
    env = np.abs(signal.hilbert(segs, axis=1))
    env = ndimage.uniform_filter1d(env, int(0.002 * sr), axis=1).mean(axis=0)
    pk = int(np.argmax(env[: int(0.05 * sr)]))
    peak = env[pk] + 1e-12
    after = env[pk:]
    below = np.where(after < peak * 0.1)[0]  # -20 dB
    decay_ms = (below[0] if len(below) else len(after)) / sr * 1000
    attack_ms = pk / sr * 1000
    n_att = int(0.012 * sr)
    win = int(0.2 * sr)
    spec_full = np.abs(np.fft.rfft(segs[:, :win] * np.hanning(win), axis=1)).mean(axis=0)
    freqs = np.fft.rfftfreq(win, 1 / sr)
    p = spec_full ** 2
    centroid = float((freqs * p).sum() / (p.sum() + 1e-12))
    att = np.abs(np.fft.rfft(raw[:, :n_att] * np.hanning(n_att), axis=1)).mean(axis=0) ** 2
    att_f = np.fft.rfftfreq(n_att, 1 / sr)
    hf_attack = float(att[att_f > 2000].sum() / (att.sum() + 1e-12))
    rms = np.sqrt(np.mean(segs[:, : max(int(decay_ms / 1000 * sr), 64)] ** 2))
    crest = float(20 * np.log10(np.max(np.abs(segs)) / (rms + 1e-12)))
    d = {"decay_ms": round(float(decay_ms), 1), "attack_ms": round(float(attack_ms), 1),
         "brightness_hz": round(centroid, 1), "transient": _level(hf_attack, (0.15, 0.4), ("soft", "medium", "sharp")),
         "crest_db": round(crest, 1)}
    if voice in ("kick", "tom", "snare"):
        lo_hz, hi_hz = {"kick": (30, 200), "tom": (60, 500), "snare": (100, 400)}[voice]
        body = raw[:, int(0.02 * sr): int(0.15 * sr)]
        n = body.shape[1]
        bs = np.abs(np.fft.rfft(body * np.hanning(n), n=8 * n, axis=1)).mean(axis=0)
        bf = np.fft.rfftfreq(8 * n, 1 / sr)
        m = (bf >= lo_hz) & (bf <= hi_hz)
        if m.any():
            d["f0_hz"] = round(float(bf[m][int(np.argmax(bs[m]))]), 1)
        if voice == "kick":
            head = raw[:, : int(0.015 * sr)]
            nh = head.shape[1]
            hs = np.abs(np.fft.rfft(head * np.hanning(nh), n=16 * nh, axis=1)).mean(axis=0)
            hf = np.fft.rfftfreq(16 * nh, 1 / sr)
            mm = (hf >= 30) & (hf <= 400)
            d["start_hz"] = round(float(hf[mm][int(np.argmax(hs[mm]))]), 1)
            d["click"] = _level(hf_attack, (0.05, 0.2), ("low", "medium", "high"))
            f0 = d.get("f0_hz", 55.0)
            harm = sum(bs[np.argmin(np.abs(bf - f0 * k))] for k in (2, 3, 4, 5))
            fund = bs[np.argmin(np.abs(bf - f0))] + 1e-12
            d["saturation"] = _level(float(harm / fund), (0.25, 0.8), ("clean", "medium", "heavy"))
            d["sub_heavy"] = bool(d.get("f0_hz", 60) < 60 and decay_ms > 250)
    if voice in ("hat_closed", "hat_open", "ride", "crash", "perc"):
        d["openness"] = "open" if decay_ms > 120 else "closed"
    if voice in ("snare", "clap"):
        rp = np.abs(np.fft.rfft(raw[:, :win] * np.hanning(win), axis=1)).mean(axis=0) ** 2
        d["noise_ratio"] = round(float(rp[freqs > 1000].sum() / (rp.sum() + 1e-12)), 2)
    return d


def _level(v: float, cuts: tuple[float, float], names: tuple[str, str, str]) -> str:
    return names[0] if v < cuts[0] else names[1] if v < cuts[1] else names[2]
