"""Harmony: NNLS note salience, key, chords (with extensions), Roman numerals, loops, harmonic rhythm."""

from __future__ import annotations

import numpy as np

from .theory import FAMILY, QUALITIES, Chord, Key, roman, spell_chord
from .util import mode

VERSION = "3"
SR = 22050
HOP = 512
BPO = 36  # CQT bins per octave (3 per semitone)
MIDI_LO, MIDI_HI = 21, 104  # A0 .. G#7
BASE_QUALITIES = ("", "m", "dim", "sus4", "sus2", "aug")

# key profiles (major, minor)
_KK = (np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]),
       np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]))
_TEMPERLEY = (np.array([5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0]),
              np.array([5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 4.0]))
# EDM-oriented profile (diatonic, tonic/fifth emphasised, leading tone de-emphasised)
_EDMA = (np.array([1.0, 0.3, 0.65, 0.3, 0.75, 0.7, 0.3, 0.85, 0.3, 0.65, 0.4, 0.45]),
         np.array([1.0, 0.3, 0.6, 0.8, 0.3, 0.65, 0.3, 0.85, 0.6, 0.3, 0.6, 0.35]))
PROFILES = {"krumhansl": _KK, "temperley": _TEMPERLEY, "edma": _EDMA}


# ------------------------------------------------------------------ note salience
_DICT_CACHE: dict = {}


def _note_dictionary(n_bins: int) -> np.ndarray:
    if n_bins in _DICT_CACHE:
        return _DICT_CACHE[n_bins]
    notes = np.arange(MIDI_LO, MIDI_HI + 1)
    D = np.zeros((n_bins, len(notes)))
    bins = np.arange(n_bins)
    for k, n in enumerate(notes):
        for h in range(1, 9):
            centre = 3 * (n - MIDI_LO) + 1 + BPO * np.log2(h)
            if centre >= n_bins:
                break
            D[:, k] += 0.6 ** (h - 1) * np.exp(-0.5 * ((bins - centre) / 0.7) ** 2)
    D /= np.linalg.norm(D, axis=0, keepdims=True) + 1e-12
    _DICT_CACHE[n_bins] = D
    return D


def note_salience(y: np.ndarray, beats: np.ndarray, sr: int = SR) -> tuple[np.ndarray, np.ndarray, float]:
    """Beat-synchronous NNLS note activations (notes x segments). Returns (X, segment_starts, tuning)."""
    import librosa

    tuning = float(librosa.estimate_tuning(y=y, sr=sr, bins_per_octave=36))
    n_bins = BPO * 8
    # first bin a third of a semitone below A0 so each semitone's centre bin is index 3k+1
    fmin = librosa.note_to_hz("A0") * 2 ** (-1 / 36)
    C = np.abs(librosa.cqt(y, sr=sr, hop_length=HOP, fmin=fmin, n_bins=n_bins, bins_per_octave=BPO,
                           tuning=tuning))
    times = librosa.frames_to_time(np.arange(C.shape[1]), sr=sr, hop_length=HOP)
    starts = np.asarray(beats, dtype=float)
    edges = np.concatenate([starts, [starts[-1] + (starts[-1] - starts[-2] if len(starts) > 1 else 0.5)]])
    D = _note_dictionary(n_bins)
    S = np.zeros((n_bins, len(starts)))
    for i in range(len(starts)):
        m = (times >= edges[i]) & (times < edges[i + 1])
        if not m.any():
            continue
        s = np.median(C[:, m], axis=1)
        # mild spectral whitening against broadband noise: subtract a running floor
        floor = np.convolve(s, np.ones(18) / 18, mode="same")
        S[:, i] = np.maximum(s - 0.6 * floor, 0)
    return _batched_nnls(D, S), starts, tuning


def _batched_nnls(D: np.ndarray, S: np.ndarray, n_iter: int = 400) -> np.ndarray:
    """Non-negative least squares for many columns at once (multiplicative updates)."""
    DtS = D.T @ S
    DtD = D.T @ D
    X = np.maximum(DtS, 0) + 1e-9
    for _ in range(n_iter):
        X *= DtS / (DtD @ X + 1e-12)
    X[X < 1e-6 * (X.max() + 1e-12)] = 0
    return X


def chromas(X: np.ndarray, X_bass: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Treble chroma, bass chroma, and per-segment energy from note activations."""
    notes = np.arange(MIDI_LO, MIDI_HI + 1)
    pcs = notes % 12
    w_treble = 1 / (1 + np.exp(-(notes - 47) / 2.0))  # fades in from ~B2
    w_treble *= 1 / (1 + np.exp((notes - 90) / 3.0))
    Xb = X if X_bass is None else X_bass
    w_bass = np.exp(-0.5 * ((notes - 38) / 7.0) ** 2) * (notes <= 57)
    treble = np.zeros((12, X.shape[1]))
    bass = np.zeros((12, X.shape[1]))
    for pc in range(12):
        m = pcs == pc
        treble[pc] = (X[m] * w_treble[m, None]).sum(axis=0)
        bass[pc] = (Xb[m] * w_bass[m, None]).sum(axis=0)
    energy = treble.sum(axis=0) + bass.sum(axis=0)
    return treble, bass, energy


# ------------------------------------------------------------------ chord recognition
def _template(root: int, quality: str, ext_weight: float = 0.7) -> np.ndarray:
    t = np.zeros(12)
    for k, iv in enumerate(QUALITIES[quality]):
        w = 1.0 if k < 3 else ext_weight
        t[(root + iv) % 12] = max(t[(root + iv) % 12], w)
    return t / np.linalg.norm(t)


def _emissions(treble: np.ndarray, bass: np.ndarray, energy: np.ndarray, qualities=BASE_QUALITIES,
               bass_w: float = 0.6) -> tuple[np.ndarray, list[Chord | None]]:
    states: list[Chord | None] = [Chord(r, q) for q in qualities for r in range(12)] + [None]
    tn = treble / (np.linalg.norm(treble, axis=0, keepdims=True) + 1e-9)
    bn = bass / (bass.max(axis=0, keepdims=True) + 1e-9)
    T = np.stack([_template(c.root, c.quality) for c in states[:-1]])  # S x 12
    score = T @ tn  # S x n
    roots = np.array([c.root for c in states[:-1]])
    score += bass_w * bn[roots]
    # mild prior: plain major/minor triads are the most common
    prior = np.array([0.0 if c.quality in ("", "m") else -0.08 for c in states[:-1]])
    score += prior[:, None]
    e_rel = energy / (np.percentile(energy[energy > 0], 90) + 1e-9) if np.any(energy > 0) else energy
    n_score = np.where(e_rel < 0.08, 1.6, 0.2)[None, :]
    return np.vstack([score, n_score]), states


def _viterbi(E: np.ndarray, beat_in_bar: np.ndarray, meter: int, temp: float = 0.08) -> np.ndarray:
    S, n = E.shape
    logp = (E - E.max(axis=0, keepdims=True)) / temp
    change_cost = np.where(beat_in_bar == 0, 1.5, np.where(beat_in_bar == meter // 2, 4.0, 8.0))
    dp = logp[:, 0].copy()
    back = np.zeros((S, n), dtype=int)
    for i in range(1, n):
        best_prev = int(np.argmax(dp))
        stay = dp
        switch = dp[best_prev] - change_cost[i]
        take_switch = switch > stay
        back[:, i] = np.where(take_switch, best_prev, np.arange(S))
        dp = np.maximum(stay, switch) + logp[:, i]
    path = np.zeros(n, dtype=int)
    path[-1] = int(np.argmax(dp))
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[path[i], i]
    return path


_EXT_RULES = {
    # family -> list of (quality, required intervals beyond the triad)
    "maj": [("13", (10, 9)), ("maj13", (11, 9)), ("9", (10, 2)), ("maj9", (11, 2)), ("7", (10,)), ("maj7", (11,)),
            ("6/9", (9, 2)), ("6", (9,)), ("add9", (2,))],
    "min": [("m11", (10, 2, 5)), ("m9", (10, 2)), ("m7", (10,)), ("mmaj7", (11,)), ("m6", (9,)), ("madd9", (2,))],
    "dim": [("m7b5", (10,)), ("dim7", (9,))],
    "sus": [("7sus4", (10,))],
}


def _extend(chord: Chord, treble_seg: np.ndarray, note_pcs: np.ndarray | None, thr: float = 0.38) -> tuple[Chord, float]:
    """Pick the richest extension whose extra pitch classes are clearly present."""
    fam = chord.family
    if fam not in _EXT_RULES:
        return chord, 0.0
    tones = [(chord.root + iv) % 12 for iv in QUALITIES[chord.quality]]
    evidence = treble_seg / (np.mean(treble_seg[tones]) + 1e-9)
    if note_pcs is not None and note_pcs.sum() > 0:
        nev = note_pcs / (np.mean(note_pcs[tones]) + 1e-9)
        evidence = 0.5 * evidence + 0.5 * nev
    for q, ivs in _EXT_RULES[fam]:
        if chord.quality == "sus2" and q == "7sus4":
            continue
        pcs = [(chord.root + iv) % 12 for iv in ivs]
        if all(evidence[p] >= thr for p in pcs):
            # a major 7th next to a minor 7th is noise, not an extension
            if q in ("maj7", "maj9", "maj13", "mmaj7") and evidence[(chord.root + 10) % 12] > evidence[(chord.root + 11) % 12]:
                continue
            return Chord(chord.root, q, chord.bass), float(min(evidence[p] for p in pcs))
    return chord, 0.0


# ------------------------------------------------------------------ key
def estimate_key(treble: np.ndarray, bass: np.ndarray, chords: list[dict]) -> tuple[Key, float, list[dict]]:
    prof_chroma = treble.sum(axis=1) + 0.7 * bass.sum(axis=1)
    pc = prof_chroma / (prof_chroma.sum() + 1e-9)
    scores = []
    total = sum(c["end"] - c["start"] for c in chords if c.get("_chord") is not None) or 1.0
    for mode_i, mode_name in enumerate(("major", "minor")):
        for tonic in range(12):
            key = Key(tonic, mode_name)
            corr = np.mean([np.corrcoef(np.roll(PROFILES[p][mode_i], tonic), pc)[0, 1] for p in PROFILES])
            fit = 0.0
            tonic_w = 0.0
            for c in chords:
                ch = c.get("_chord")
                if ch is None:
                    continue
                dur = c["end"] - c["start"]
                diat = {(ch.root + iv) % 12 for iv in QUALITIES[ch.quality][:3]} <= key.pitch_classes
                if not diat and mode_name == "minor" and (ch.root - tonic) % 12 == 7 and ch.family == "maj":
                    diat = True  # harmonic-minor V
                fit += dur * diat
                if ch.root == tonic and ((ch.family == "maj") == (mode_name == "major")):
                    tonic_w += dur
            fit /= total
            tonic_w /= total
            scores.append({"key": key, "score": float(corr + 0.8 * fit + 0.6 * tonic_w), "corr": float(corr),
                           "chord_fit": float(fit), "tonic_weight": float(tonic_w)})
    scores.sort(key=lambda s: -s["score"])
    best, second = scores[0], scores[1]
    conf = float(np.clip((best["score"] - second["score"]) / 0.25, 0.0, 1.0) * 0.6 + 0.4 * np.clip(best["corr"], 0, 1))
    alts = [{"key": s["key"].name, "score": round(s["score"], 3)} for s in scores[:4]]
    return best["key"], conf, alts


# ------------------------------------------------------------------ loops
def find_loop(bar_chords: list[tuple], max_len: int = 8, tonic_pc: int | None = None) -> dict:
    """Smallest bar period whose repetition explains the chord sequence; returns the loop bars."""
    seq = list(bar_chords)
    n = len(seq)
    valid = [i for i, b in enumerate(seq) if b and b != ("N",)]
    if len(valid) < 2:
        return {"length_bars": len(valid), "bars": [seq[i] for i in valid], "agreement": 1.0 if valid else 0.0}
    scores = {}
    for L in (1, 2, 4, 8):
        if L > max_len or L >= len(valid):
            break
        pairs = [(i, i + L) for i in valid if i + L < n and seq[i + L] and seq[i + L] != ("N",)]
        if pairs:
            scores[L] = float(np.mean([seq[a] == seq[b] for a, b in pairs]))
    if not scores:
        return {"length_bars": 1, "bars": [seq[valid[0]]], "agreement": 0.0}
    # smallest period that repeats well; otherwise the best-agreeing one
    good = [L for L in sorted(scores) if scores[L] >= 0.75]
    L = good[0] if good else max(scores, key=lambda k: scores[k])
    best = {"length_bars": L, "agreement": scores[L]}
    # choose the window phase with the most repetitions (ties: starts on the tonic, then earliest)
    cands = []
    for ph in range(L):
        windows = []
        for st in range(ph, n - L + 1, L):
            w = tuple(seq[st: st + L])
            if all(b and b != ("N",) for b in w):
                windows.append((st, w))
        if not windows:
            continue
        loop = mode([w for _, w in windows])
        count = sum(1 for _, w in windows if w == loop)
        first = min(st for st, w in windows if w == loop)
        starts_on_tonic = tonic_pc is not None and loop[0] and _root_pc(loop[0][0]) == tonic_pc
        cands.append((count, starts_on_tonic, -first, loop))
    cands.sort(reverse=True)
    best["bars"] = list(cands[0][3]) if cands else [seq[valid[0]]]
    best["occurrences"] = cands[0][0] if cands else 1
    return best


def find_loop_chords(chords: list[dict], bar_len_s: float, tonic_pc: int | None = None) -> dict | None:
    """Chord-level cycle finder (robust to sections restarting the progression mid-way):
    the n-gram of consecutive chord symbols that covers the most time when repeated."""
    seq = [c for c in chords if c["symbol"] != "N"]
    if len(seq) < 3:
        return None
    syms = [c["symbol"] for c in seq]
    best = None
    for p_len in range(2, 9):
        counts: dict[tuple, list[int]] = {}
        for i in range(len(syms) - p_len + 1):
            g = tuple(syms[i: i + p_len])
            if len(set(g)) < 2:
                continue
            counts.setdefault(g, []).append(i)
        for g, idxs in counts.items():
            # non-overlapping occurrences
            occ, last = [], -10 ** 9
            for i in idxs:
                if i >= last + p_len:
                    occ.append(i)
                    last = i
            if len(occ) < 2:
                continue
            cover = sum(seq[i + p_len - 1]["end"] - seq[i]["start"] for i in occ) / bar_len_s
            key = (cover, -p_len)
            if best is None or key > best["key"]:
                best = {"key": key, "symbols": list(g), "occ": occ, "p": p_len}
    if best is None:
        return None
    g = best["symbols"]
    # rotate so it starts on the tonic when the tonic is in the cycle
    if tonic_pc is not None:
        for r in range(len(g)):
            if _root_pc(g[r]) == tonic_pc:
                g = g[r:] + g[:r]
                break
    durs = [seq[i + k]["end"] - seq[i + k]["start"] for i in best["occ"] for k in range(best["p"])]
    return {"symbols": g, "coverage_bars": best["key"][0], "occurrences": len(best["occ"]),
            "bars_per_chord": float(np.median(durs) / bar_len_s)}


def _root_pc(symbol: str) -> int | None:
    from .theory import parse_chord

    try:
        c = parse_chord(symbol)
    except ValueError:
        return None
    return c.root if c else None


def _flatten_loop(bars: list[tuple]) -> list[str]:
    out: list[str] = []
    for b in bars:
        for c in b:
            if not out or out[-1] != c:
                out.append(c)
    return out


# ------------------------------------------------------------------ main
def analyze(y_harm: np.ndarray, beats, downbeats, meter: int, y_bass: np.ndarray | None = None,
            note_events: list | None = None, sr: int = SR) -> dict:
    """y_harm: harmonic signal (bass+other stems or HPSS harmonic part) at 22050 Hz."""
    beats = np.asarray(beats, dtype=float)
    downbeats = np.asarray(downbeats, dtype=float)
    if len(beats) < 4 or np.max(np.abs(y_harm)) < 1e-5:
        return {"key": None, "chords": [], "loop": None, "confidence": 0.0}
    X, starts, tuning = note_salience(y_harm, beats, sr)
    Xb = note_salience(y_bass, beats, sr)[0] if y_bass is not None and np.max(np.abs(y_bass)) > 1e-5 else None
    treble, bass, energy = chromas(X, Xb)
    db_idx = np.array([int(np.argmin(np.abs(beats - d))) for d in downbeats]) if len(downbeats) else np.array([0])
    beat_in_bar = np.zeros(len(beats), dtype=int)
    for i in range(len(beats)):
        k = np.searchsorted(db_idx, i, side="right") - 1
        beat_in_bar[i] = (i - db_idx[k]) % meter if k >= 0 else (i - db_idx[0]) % meter
    E, states = _emissions(treble, bass, energy)
    path = _viterbi(E, beat_in_bar, meter)
    ibi = float(np.median(np.diff(beats)))
    ends = np.concatenate([beats[1:], [beats[-1] + ibi]])

    # note-event pitch classes per beat (from transcription), used to confirm extensions
    note_pc = None
    if note_events:
        note_pc = np.zeros((12, len(beats)))
        for (t0, t1, pitch, *_rest) in note_events:
            if pitch < 45:
                continue
            i0 = max(0, int(np.searchsorted(beats, t0, side="right") - 1))
            i1 = max(i0, int(np.searchsorted(beats, t1, side="right") - 1))
            note_pc[int(pitch) % 12, i0: i1 + 1] += 1

    # short "no chord" gaps (sparse stabs, rests) inherit the previous chord
    n_idx = len(states) - 1
    i = 0
    while i < len(path):
        if path[i] == n_idx:
            j = i
            while j + 1 < len(path) and path[j + 1] == n_idx:
                j += 1
            if j - i + 1 < 2 * meter:
                fill = path[i - 1] if i > 0 else (path[j + 1] if j + 1 < len(path) else n_idx)
                path[i: j + 1] = fill
            i = j + 1
        else:
            i += 1
    segs = []
    i = 0
    while i < len(path):
        j = i
        while j + 1 < len(path) and path[j + 1] == path[i]:
            j += 1
        segs.append((i, j, states[path[i]]))
        i = j + 1
    bass_typ = float(np.median(bass.sum(axis=0)[bass.sum(axis=0) > 0])) if np.any(bass > 0) else 0.0
    chords = []
    for i, j, ch in segs:
        seg = {"start": float(beats[i]), "end": float(ends[j]), "beat_start": int(i), "beats": int(j - i + 1)}
        if ch is None:
            seg["_chord"] = None
            chords.append(seg)
            continue
        tseg = treble[:, i: j + 1].mean(axis=1)
        bseg = bass[:, i: j + 1].mean(axis=1)
        npc = note_pc[:, i: j + 1].sum(axis=1) if note_pc is not None else None
        ext, _ = _extend(ch, tseg, npc)
        # slash chord: strong bass on another chord tone
        bpc = int(np.argmax(bseg))
        if (bseg.sum() > 0.3 * bass_typ and bpc != ext.root and bpc in ext.pitch_classes
                and bseg[bpc] > 1.6 * bseg[ext.root]):
            ext = Chord(ext.root, ext.quality, bpc)
        # confidence: emission margin of the chosen base chord over the best alternative
        Ei = E[:, i: j + 1].mean(axis=1)
        k_sel = states.index(ch)
        alt = np.max(np.delete(Ei, k_sel))
        seg["_chord"] = ext
        seg["confidence"] = float(np.clip((Ei[k_sel] - alt) / 0.25 + 0.3, 0, 1))
        chords.append(seg)

    key, key_conf, key_alts = estimate_key(treble, bass, chords)
    for c in chords:
        ch = c.pop("_chord")
        c["symbol"] = spell_chord(ch, key)
        c["roman"] = roman(ch, key) if ch is not None else "N"
        c["family"] = ch.family if ch is not None else None
        c["_obj"] = ch

    # merge identical neighbours that differ only by extension noise? keep as-is but merge exact repeats
    merged = []
    for c in chords:
        if merged and merged[-1]["symbol"] == c["symbol"]:
            merged[-1]["end"] = c["end"]
            merged[-1]["beats"] += c["beats"]
            continue
        merged.append(c)
    chords = merged

    # bar-level chord sequence (chord sounding on each beat)
    beat_sym = []
    for c in chords:
        beat_sym.extend([c["symbol"]] * c["beats"])
    bar_chords = []
    for b, d in enumerate(db_idx):
        nxt = db_idx[b + 1] if b + 1 < len(db_idx) else d + meter
        syms = beat_sym[d:nxt]
        bar = []
        for s in syms:
            if not bar or bar[-1] != s:
                bar.append(s)
        bar_chords.append(tuple(bar))
    loop = find_loop(bar_chords, tonic_pc=key.tonic)
    loop_syms = _flatten_loop(loop["bars"]) if loop.get("bars") else []
    bar_len_s = float(np.median(np.diff(downbeats))) if len(downbeats) > 1 else meter * float(np.median(np.diff(beats)))
    bpc = loop["length_bars"] / max(len(loop_syms), 1) if loop_syms else None
    bar_cover = loop.get("occurrences", 1) * loop["length_bars"]
    alt = find_loop_chords(chords, bar_len_s, key.tonic)
    if alt and alt["coverage_bars"] > 1.25 * bar_cover and len(alt["symbols"]) >= len(loop_syms):
        loop_syms = alt["symbols"]
        bpc = alt["bars_per_chord"]
        loop = {"length_bars": int(round(bpc * len(loop_syms))), "agreement": loop.get("agreement", 0),
                "occurrences": alt["occurrences"], "method": "chord-cycle"}
    sym_to_obj = {c["symbol"]: c["_obj"] for c in chords}
    loop_roman = [roman(sym_to_obj[s], key) if sym_to_obj.get(s) is not None else "N" for s in loop_syms]
    harmonic_bars = [b for b in bar_chords if b and b != ("N",)]
    changes = [max(0, len(b) - 1) + 1 for b in harmonic_bars]
    chords_per_bar = float(np.mean([len(b) for b in harmonic_bars])) if harmonic_bars else 0.0

    quality_hist: dict[str, float] = {}
    total = 0.0
    for c in chords:
        if c["_obj"] is None:
            continue
        d = c["end"] - c["start"]
        q = c["_obj"].quality or "maj"
        quality_hist[q] = quality_hist.get(q, 0.0) + d
        total += d
    quality_hist = {k: round(v / total, 3) for k, v in sorted(quality_hist.items(), key=lambda kv: -kv[1])} if total else {}
    extensions = sorted({str(iv) for c in chords if c["_obj"] is not None
                         for iv in _ext_names(c["_obj"].quality)})
    for c in chords:
        c.pop("_obj", None)
        c.pop("family", None)
    low_conf = [c for c in chords if c.get("confidence", 1) < 0.35 and c["symbol"] != "N"]
    return {
        "key": key.name,
        "key_confidence": round(key_conf, 3),
        "key_alternatives": key_alts,
        "tuning_cents": round(tuning * 100, 1),
        "chords": chords,
        "bar_chords": [list(b) for b in bar_chords],
        "loop": {"symbols": loop_syms, "roman": loop_roman, "length_bars": loop["length_bars"],
                 "bars_per_chord": round(bpc, 3) if bpc else None, "agreement": round(loop.get("agreement", 0), 3),
                 "occurrences": loop.get("occurrences")},
        "harmonic_rhythm": {"chords_per_bar": round(chords_per_bar, 2),
                            "changes_on_downbeat": round(_downbeat_change_ratio(chords, downbeats), 2)},
        "vocabulary": quality_hist,
        "extensions": extensions,
        "uncertain_spans": [{"start": round(c["start"], 2), "end": round(c["end"], 2), "symbol": c["symbol"]} for c in low_conf],
        "confidence": round(float(np.mean([c.get("confidence", 0.5) for c in chords if c["symbol"] != "N"] or [0])), 3),
        "_arrays": {"treble_chroma": treble.astype(np.float32), "bass_chroma": bass.astype(np.float32),
                    "note_salience": X.astype(np.float32), "beat_times": beats.astype(np.float32)},
    }


def _ext_names(q: str) -> list[str]:
    ivs = QUALITIES.get(q, ())
    names = {9: "6", 10: "b7", 11: "maj7", 14: "9", 2: "9", 17: "11", 5: "11", 21: "13", 13: "b9", 15: "#9"}
    return [names[iv] for iv in ivs[3:] if iv in names] if FAMILY.get(q) != "sus" else (["b7"] if "7" in q else [])


def _downbeat_change_ratio(chords: list[dict], downbeats: np.ndarray) -> float:
    if len(chords) < 2 or len(downbeats) == 0:
        return 0.0
    changes = [c["start"] for c in chords[1:]]
    on = sum(1 for t in changes if np.min(np.abs(downbeats - t)) < 0.06)
    return on / len(changes)
