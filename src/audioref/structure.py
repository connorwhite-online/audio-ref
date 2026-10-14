"""Structure: energy curve, per-bar layer activity, sections (bar-aligned), builds/drops/stops/climax."""

from __future__ import annotations

import numpy as np
from scipy import ndimage, signal

from . import loudness
from .util import robust_norm

VERSION = "7"
SR = 22050


# ------------------------------------------------------------------ helpers
def _segment_rms_db(y: np.ndarray, edges: np.ndarray, sr: int) -> np.ndarray:
    out = np.full(len(edges) - 1, -120.0)
    if y is None:
        return out
    for i in range(len(edges) - 1):
        a, b = int(edges[i] * sr), int(edges[i + 1] * sr)
        seg = y[max(0, a): max(a + 1, b)]
        if seg.size:
            out[i] = 20 * np.log10(np.sqrt(np.mean(seg.astype(np.float64) ** 2)) + 1e-9)
    return out


def _edges(times: np.ndarray, end: float) -> np.ndarray:
    times = np.asarray(times, dtype=float)
    if len(times) < 2:
        return np.array([0.0, end])
    step = float(np.median(np.diff(times)))
    return np.concatenate([times, [min(end, times[-1] + step)]])


# ------------------------------------------------------------------ energy & layers
def energy_per_beat(y: np.ndarray, beats: np.ndarray, sr: int, drum_hits: dict | None) -> np.ndarray:
    import librosa

    edges = _edges(beats, len(y) / sr)
    rms_db = _segment_rms_db(y, edges, sr)
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=512)
    cent = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    flux = np.r_[0, np.maximum(0, np.diff(np.log1p(S), axis=1)).sum(axis=0)]
    bright = np.zeros(len(edges) - 1)
    fl = np.zeros(len(edges) - 1)
    for i in range(len(edges) - 1):
        m = (times >= edges[i]) & (times < edges[i + 1])
        if m.any():
            bright[i] = np.median(cent[m])
            fl[i] = np.mean(flux[m])
    dens = np.zeros(len(edges) - 1)
    if drum_hits:
        allt = np.sort(np.concatenate([np.array([h[0] for h in hs]) for hs in drum_hits.values() if hs] or [np.array([])]))
        if allt.size:
            idx = np.searchsorted(edges, allt, side="right") - 1
            idx = idx[(idx >= 0) & (idx < len(dens))]
            np.add.at(dens, idx, 1)
    e = 0.5 * robust_norm(rms_db, 2, 98) + 0.2 * robust_norm(dens, 2, 98) + 0.15 * robust_norm(np.log(bright + 1), 2, 98) \
        + 0.15 * robust_norm(fl, 2, 98)
    return np.clip(e, 0, 1)


def layer_activity(stems: dict[str, np.ndarray], mix: np.ndarray, downbeats: np.ndarray, sr: int,
                   drum_hits: dict | None, melody_notes: list | None = None) -> dict[str, np.ndarray]:
    """Presence (bool per bar) of each layer."""
    end = len(mix) / sr
    edges = _edges(downbeats, end)
    mix_db = _segment_rms_db(mix, edges, sr)
    out: dict[str, np.ndarray] = {}
    for name, y in stems.items():
        if y is None or name == "drums":
            continue
        db = _segment_rms_db(y, edges, sr)
        ref = np.percentile(db, 95)
        if ref < mix_db.max() - 40:
            continue  # stem essentially silent everywhere
        out[name] = (db > ref - 18) & (db > mix_db - 30)
    if drum_hits:
        groups = {"kick": ("kick",), "snare/clap": ("snare", "clap", "rim"), "hats": ("hat_closed", "hat_open", "ride"),
                  "perc": ("perc", "tom"), "cymbals": ("crash",)}
        for g, voices in groups.items():
            cnt = np.zeros(len(edges) - 1)
            for v in voices:
                for t, _ in drum_hits.get(v, []):
                    i = np.searchsorted(edges, t, side="right") - 1
                    if 0 <= i < len(cnt):
                        cnt[i] += 1
            if cnt.sum() > 0:
                out[g] = cnt >= (1 if g in ("kick", "cymbals") else 2)
    elif "drums" in stems:
        db = _segment_rms_db(stems["drums"], edges, sr)
        out["drums"] = db > np.percentile(db, 95) - 18
    if melody_notes:
        cnt = np.zeros(len(edges) - 1)
        for n in melody_notes:
            i = np.searchsorted(edges, n[0], side="right") - 1
            if 0 <= i < len(cnt):
                cnt[i] += 1
        out["lead"] = cnt >= 2
    return out


# ------------------------------------------------------------------ segmentation
def _bar_features(y: np.ndarray, sr: int, downbeats: np.ndarray, chroma_beats: np.ndarray | None,
                  beats: np.ndarray, layers: dict[str, np.ndarray]) -> np.ndarray:
    import librosa

    edges = _edges(downbeats, len(y) / sr)
    mf = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=1024)
    times = librosa.frames_to_time(np.arange(mf.shape[1]), sr=sr, hop_length=1024)
    feats = []
    for i in range(len(edges) - 1):
        m = (times >= edges[i]) & (times < edges[i + 1])
        v = mf[:, m].mean(axis=1) if m.any() else np.zeros(13)
        feats.append(v)
    F = np.array(feats)
    F = (F - F.mean(axis=0)) / (F.std(axis=0) + 1e-9)
    parts = [F]
    if chroma_beats is not None and chroma_beats.size:
        C = []
        for i in range(len(edges) - 1):
            m = (beats >= edges[i] - 1e-3) & (beats < edges[i + 1] - 1e-3)
            c = chroma_beats[:, m].mean(axis=1) if m.any() else np.zeros(12)
            C.append(c / (np.linalg.norm(c) + 1e-9))
        parts.append(1.5 * np.array(C))
    if layers:
        L = np.array([layers[k].astype(float) for k in sorted(layers)]).T
        parts.append(2.0 * L)
    rms = _segment_rms_db(y, edges, sr)
    parts.append(2.0 * robust_norm(rms)[:, None])
    return np.hstack(parts)


def _novelty(F: np.ndarray, k: int) -> np.ndarray:
    n = len(F)
    Fn = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-9)
    S = Fn @ Fn.T
    nov = np.zeros(n)
    g = np.outer(signal.windows.gaussian(2 * k, k / 2), signal.windows.gaussian(2 * k, k / 2))
    sign = np.ones((2 * k, 2 * k))
    sign[:k, k:] = -1
    sign[k:, :k] = -1
    K = g * sign
    Sp = np.pad(S, k, mode="edge")
    for i in range(n):
        nov[i] = np.sum(Sp[i: i + 2 * k, i: i + 2 * k] * K)
    nov = np.maximum(nov, 0)
    return nov / (nov.max() + 1e-9)


def segment(F: np.ndarray, layers: dict[str, np.ndarray], min_len: int = 2, phrase: int = 4,
            cues: np.ndarray | None = None) -> list[int]:
    """Bar indices of section starts (always includes 0). `cues` adds boundary evidence per bar (e.g. the
    start/end of a snare roll, the bar after a fill)."""
    n = len(F)
    if n < 2 * min_len:
        return [0]
    nov = 0.5 * _novelty(F, 4) + 0.5 * _novelty(F, 2) if n >= 8 else _novelty(F, 2)
    # layer entries/exits are strong section cues
    if layers:
        L = np.array([layers[k].astype(float) for k in sorted(layers)])
        lc = np.zeros(n)
        lc[1:] = np.abs(np.diff(L, axis=1)).sum(axis=0)
        lc = lc / (lc.max() + 1e-9)
        nov = 0.6 * nov + 0.4 * lc
    bonus = np.array([0.2 if i % (2 * phrase) == 0 else 0.15 if i % phrase == 0 else 0.03 if i % 2 == 0 else 0.0
                      for i in range(n)])
    score = nov + bonus * (nov > 0.1)
    if cues is not None and len(cues) == n:
        score = score + cues
    thr = max(0.35, float(np.percentile(score, 75)))
    cands = [i for i in range(1, n) if score[i] >= thr and score[i] >= score[max(0, i - 1)] and
             score[i] >= score[min(n - 1, i + 1)]]
    cands.sort(key=lambda i: -score[i])
    chosen = [0]
    for c in cands:
        if all(abs(c - b) >= min_len for b in chosen) and n - c >= min_len:
            chosen.append(c)
    return sorted(chosen)


def _cluster_letters(sec_feats: list[np.ndarray], thr: float = 0.85, chord_sets: list[set] | None = None) -> list[str]:
    """Similarity letters (A, B, ...). With chord sets, sections must also share their harmony to group."""
    letters = []
    protos: list[tuple[str, np.ndarray, set | None]] = []
    nxt = iter("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    for i, f in enumerate(sec_feats):
        fn = f / (np.linalg.norm(f) + 1e-9)
        cs = chord_sets[i] if chord_sets else None
        best, bs = None, -1.0
        for lab, p, pcs in protos:
            s = float(fn @ p)
            if cs and pcs and len(cs | pcs) and len(cs & pcs) / len(cs | pcs) < 0.5:
                continue  # different chord vocabulary: not the same section type
            if s > bs:
                best, bs = lab, s
        if best is not None and bs >= thr:
            letters.append(best)
        else:
            lab = next(nxt)
            protos.append((lab, fn, cs))
            letters.append(lab)
    return letters


# ------------------------------------------------------------------ events
def _centroid_trend(y: np.ndarray, sr: int, t0: float, t1: float) -> float:
    import librosa

    seg = y[int(t0 * sr): int(t1 * sr)]
    if seg.size < sr:
        return 0.0
    c = librosa.feature.spectral_centroid(y=seg, sr=sr, hop_length=1024)[0]
    if len(c) < 4:
        return 0.0
    a, b = np.median(c[: len(c) // 4]), np.median(c[-len(c) // 4:])
    return float(np.log2((b + 1) / (a + 1)))


def _hf_noise_trend(y: np.ndarray, sr: int, t0: float, t1: float) -> float:
    seg = y[int(t0 * sr): int(t1 * sr)]
    if seg.size < sr:
        return 0.0
    sos = signal.butter(4, 4000 / (sr / 2), "highpass", output="sos")
    hf = signal.sosfilt(sos, seg)
    n = len(hf) // 4
    a = np.sqrt(np.mean(hf[:n] ** 2)) + 1e-9
    b = np.sqrt(np.mean(hf[-n:] ** 2)) + 1e-9
    return float(20 * np.log10(b / a))


def beat_stops(mix: np.ndarray, beats: np.ndarray, sr: int, downbeats: np.ndarray) -> list[dict]:
    """Sudden silences of >= 1 beat (everything stops), e.g. the beat before a drop."""
    edges = _edges(beats, len(mix) / sr)
    db = _segment_rms_db(mix, edges, sr)
    loud = np.percentile(db, 90)
    quiet = db < loud - 28
    stops = []
    i = 0
    while i < len(quiet):
        if quiet[i]:
            j = i
            while j + 1 < len(quiet) and quiet[j + 1]:
                j += 1
            before = db[max(0, i - 4): i]
            after = db[j + 1: j + 5]
            if len(before) >= 3 and len(after) and before.min() > loud - 20 and after.max() > loud - 12 and j - i + 1 <= 16:
                bar = int(np.searchsorted(downbeats, edges[i] + 1e-3, side="right") - 1)
                stops.append({"start": round(float(edges[i]), 3), "end": round(float(edges[j + 1]), 3),
                              "beats": int(j - i + 1), "bar": bar})
            i = j + 1
        else:
            i += 1
    return stops


# ------------------------------------------------------------------ main
def analyze(mix: np.ndarray, sr: int, beats, downbeats, meter: int, stems: dict[str, np.ndarray],
            drum_hits: dict | None = None, groove: dict | None = None, chroma_beats: np.ndarray | None = None,
            melody_notes: list | None = None, stereo_orig: np.ndarray | None = None, sr_orig: int = 44100,
            electronic: bool = True, bar_chords: list | None = None, loop_bars: int | None = None) -> dict:
    beats = np.asarray(beats, dtype=float)
    downbeats = np.asarray(downbeats, dtype=float)
    if len(downbeats) < 2:
        return {"sections": [], "events": [], "energy_per_beat": []}
    e_beat = energy_per_beat(mix, beats, sr, drum_hits)
    layers = layer_activity(stems, mix, downbeats, sr, drum_hits, melody_notes)
    nb = len(downbeats)
    F = _bar_features(mix, sr, downbeats, chroma_beats, beats, layers)
    cues = np.zeros(nb)
    pats = (groove or {}).get("patterns", {}) if groove else {}
    for r in pats.get("rolls", []):  # a build roll starts and ends sections (its last bar may hold a stop)
        for b, w in ((r["start_bar"], 0.6), (r["end_bar"], 0.4), (r["end_bar"] + 1, 0.4)):
            if 0 < b < nb:
                cues[b] += w
    for f in pats.get("fills", []):  # the bar after a fill usually starts something new
        if 0 < f["bar"] + 1 < nb:
            cues[f["bar"] + 1] += 0.2
    if bar_chords and len(bar_chords) >= 8:  # the chord loop changes -> a new section (e.g. waltz A/B)
        L = loop_bars if loop_bars and 2 <= loop_bars <= 8 else 4
        bc = [tuple(b) for b in bar_chords]
        for b in range(L, min(nb, len(bc)) - 1, L):
            prev, nxt = bc[b - L: b], bc[b: b + L]
            if len(nxt) == L and all(x and x != ("N",) for x in prev + nxt):
                same = sum(p == q for p, q in zip(prev, nxt)) / L
                if same <= 0.5:
                    cues[b] += 0.5
    starts = segment(F, layers, min_len=4 if nb >= 32 else 2, cues=cues)
    bounds = starts + [nb]
    # bar energy
    db_idx = np.array([int(np.argmin(np.abs(beats - d))) for d in downbeats])
    bar_e = np.zeros(nb)
    for b in range(nb):
        seg = e_beat[db_idx[b]: (db_idx[b + 1] if b + 1 < nb else len(e_beat))]
        bar_e[b] = float(np.mean(seg)) if len(seg) else 0.0
    edges = _edges(downbeats, len(mix) / sr)
    # loudness per section (original level)
    lufs_curve = []
    st_times, st_lufs = (np.array([]), np.array([]))
    if stereo_orig is not None:
        st_times, st_lufs = loudness.short_term_curve(stereo_orig, sr_orig, hop_s=1.0)
        lufs_curve = [[round(float(t), 1), round(float(l), 1)] for t, l in zip(st_times, st_lufs) if np.isfinite(l)]
    secs = []
    for k in range(len(starts)):
        a, b = bounds[k], bounds[k + 1]
        t0, t1 = float(edges[a]), float(edges[b])
        lay = [name for name, arr in layers.items() if arr[a:b].mean() >= 0.5]
        sec = {"start_bar": int(a), "end_bar": int(b), "bars": int(b - a), "start": round(t0, 3), "end": round(t1, 3),
               "energy": round(float(bar_e[a:b].mean()), 3), "layers": _order_layers(lay)}
        if stereo_orig is not None:
            seg = stereo_orig[:, int(t0 * sr_orig): int(t1 * sr_orig)]
            if seg.shape[-1] > sr_orig:
                pk = 20 * np.log10(np.max(np.abs(seg)) + 1e-9)
                rms = 20 * np.log10(np.sqrt(np.mean(seg.astype(np.float64) ** 2)) + 1e-9)
                sec["crest_db"] = round(float(pk - rms), 1)
                m = (st_times >= t0) & (st_times < t1) & np.isfinite(st_lufs)
                if m.any():
                    sec["lufs"] = round(float(np.median(st_lufs[m])), 1)
        secs.append(sec)
    chord_sets = None
    if bar_chords:
        chord_sets = [{c for b in bar_chords[s["start_bar"]: s["end_bar"]] for c in b if c != "N"} for s in secs]
    letters = _cluster_letters([F[s["start_bar"]: s["end_bar"]].mean(axis=0) for s in secs], chord_sets=chord_sets)
    for s, l in zip(secs, letters):
        s["group"] = l

    # events
    events: list[dict] = []
    rolls = (groove or {}).get("patterns", {}).get("rolls", []) if groove else []
    other = stems.get("other")
    nondrum = None
    for k in ("other", "bass", "vocals"):
        if stems.get(k) is not None:
            nondrum = stems[k] if nondrum is None else nondrum + stems[k]
    stops = beat_stops(mix, beats, sr, downbeats)
    energies = np.array([s["energy"] for s in secs])
    hi_thr = np.percentile(energies, 66) if len(energies) >= 3 else energies.max()
    for k, s in enumerate(secs):
        if k == 0:
            continue
        prev = secs[k - 1]
        jump = s["energy"] - prev["energy"]
        kick_in = "kick" in s["layers"] and "kick" not in prev["layers"]
        if jump > 0.12 and (kick_in or s["energy"] >= hi_thr):
            events.append({"type": "drop", "bar": s["start_bar"], "time": s["start"], "energy_jump": round(jump, 3),
                           "kick_returns": kick_in})
        # build: the section/bars leading into a big rise
        if jump > 0.08:
            b0 = max(prev["start_bar"], s["start_bar"] - 8)
            t0, t1 = float(edges[b0]), s["start"]
            tech = []
            for r in rolls:
                if r["end_bar"] >= s["start_bar"] - 1 and r["start_bar"] < s["start_bar"]:
                    tech.append("snare roll accelerating")
                    b0 = min(b0, r["start_bar"])
            if other is not None and _centroid_trend(other, sr, t0, t1) > 0.6:
                tech.append("filter opening")
            if nondrum is not None and _hf_noise_trend(nondrum, sr, t0, t1) > 6:
                tech.append("noise riser / sweep")
            e_slope = np.polyfit(np.arange(s["start_bar"] - b0), bar_e[b0: s["start_bar"]], 1)[0] if s["start_bar"] - b0 >= 2 else 0
            if e_slope > 0.02:
                tech.append("energy ramp")
            if "kick" in prev["layers"] and "kick" not in [l for l in layers if layers[l][b0: s["start_bar"]].mean() > 0.5]:
                tech.append("kick removed")
            if any(st["start"] >= t0 and st["end"] <= t1 + 0.05 for st in stops):
                tech.append("stop before the drop")
            if set(tech) - {"energy ramp"}:
                events.append({"type": "build", "start_bar": int(b0), "end_bar": int(s["start_bar"]),
                               "start": round(t0, 3), "end": round(t1, 3), "technique": tech})
    for st in stops:  # whole-mix silences: nothing continues (maybe a reverb tail)
        events.append({"type": "stop", "bar": st["bar"], "start": st["start"], "end": st["end"], "beats": st["beats"],
                       "what_continues": []})
    for ds in (groove or {}).get("patterns", {}).get("stops", []) if groove else []:
        a, b = ds["start_bar"], ds["end_bar"]
        what = [name for name, arr in layers.items() if name not in ("kick", "snare/clap", "hats", "perc", "cymbals", "drums")
                and arr[a:b].mean() >= 0.5]
        events.append({"type": "drum_stop", "start_bar": a, "end_bar": b, "bars": b - a,
                       "start": round(float(edges[a]), 3), "what_continues": _order_layers(what)})
    # climax: start of the highest-energy section (ties -> later), refined to the bar with the max smoothed energy
    if len(secs):
        smooth = ndimage.uniform_filter1d(bar_e, 4)
        top = max(range(len(secs)), key=lambda i: (round(secs[i]["energy"], 2), secs[i]["start_bar"]))
        cb = secs[top]["start_bar"]
        events.append({"type": "climax", "bar": int(cb), "time": secs[top]["start"],
                       "peak_bar": int(np.argmax(smooth)), "energy": secs[top]["energy"]})
    events.sort(key=lambda e: e.get("bar", e.get("start_bar", 0)))
    _label_functions(secs, events, electronic)
    return {
        "sections": secs,
        "events": events,
        "energy_per_beat": [round(float(x), 3) for x in e_beat],
        "energy_per_bar": [round(float(x), 3) for x in bar_e],
        "layers_per_bar": {k: v.astype(int).tolist() for k, v in layers.items()},
        "loudness_curve_lufs": lufs_curve,
        "stops": stops,
    }


_LAYER_ORDER = ["kick", "snare/clap", "hats", "perc", "cymbals", "drums", "bass", "other", "piano", "guitar", "lead", "vocals"]


def _order_layers(lay: list[str]) -> list[str]:
    return sorted(lay, key=lambda x: _LAYER_ORDER.index(x) if x in _LAYER_ORDER else 99)


def _label_functions(secs: list[dict], events: list[dict], electronic: bool) -> None:
    if not secs:
        return
    e = np.array([s["energy"] for s in secs])
    hi = np.percentile(e, 66) if len(e) >= 3 else e.max()
    lo = np.percentile(e, 33) if len(e) >= 3 else e.min()
    real_builds = {ev["end_bar"] for ev in events if ev["type"] == "build"
                   and set(ev.get("technique", [])) - {"energy ramp"}}
    n = len(secs)
    max_layers = max(len(s["layers"]) for s in secs)
    drums = ("kick", "snare/clap", "hats", "drums")
    for i, s in enumerate(secs):
        has_drums = any(l in s["layers"] for l in drums)
        nxt = secs[i + 1] if i + 1 < n else None
        prev = secs[i - 1] if i > 0 else None
        fewer = len(s["layers"]) < max_layers
        if i == 0 and n > 1 and (s["energy"] <= np.median(e) - 0.05 or not has_drums or
                                 (fewer and nxt is not None and len(nxt["layers"]) > len(s["layers"]))):
            label = "intro"
        elif i == n - 1 and n > 2 and prev is not None and (len(s["layers"]) < len(prev["layers"]) or
                                                             s["energy"] < prev["energy"] - 0.08):
            label = "outro"
        elif nxt is not None and nxt["start_bar"] in real_builds and s["bars"] <= 8 and nxt["energy"] > s["energy"]:
            label = "build"
        elif not has_drums or ("kick" not in s["layers"] and s["energy"] <= lo + 0.05):
            label = "breakdown"
        elif prev is not None and (prev.get("label") in ("build", "breakdown") or s["start_bar"] in real_builds) \
                and s["energy"] >= lo:
            label = "drop" if electronic else "chorus"
        elif s["energy"] >= hi and prev is not None and prev.get("label") not in ("intro",):
            label = "drop" if electronic else "chorus"
        else:
            label = "groove" if electronic else "verse"
        s["label"] = label
