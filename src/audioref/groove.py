"""Groove: swing, microtiming, groove templates, per-bar step grids, pattern mining, fills, stops, rolls."""

from __future__ import annotations

import numpy as np

from .util import mode

VERSION = "4"
CORE_WEIGHT = {"kick": 1.0, "snare": 1.0, "clap": 1.0, "rim": 0.6, "hat_closed": 0.5, "hat_open": 0.5,
               "ride": 0.4, "perc": 0.4, "tom": 0.7, "crash": 0.2}
GRID_VOICES = ("kick", "snare", "clap", "rim", "hat_closed", "hat_open", "ride", "tom", "perc", "crash")


# ------------------------------------------------------------------ beat-relative positions
def beat_positions(times: np.ndarray, beats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fractional beat index and local inter-beat interval for each time (extrapolates at the edges)."""
    times = np.asarray(times, dtype=float)
    beats = np.asarray(beats, dtype=float)
    i = np.searchsorted(beats, times, side="right") - 1
    i = np.clip(i, 0, len(beats) - 2)
    ibi = beats[i + 1] - beats[i]
    pos = i + (times - beats[i]) / ibi
    return pos, ibi


def _wmedian(x: np.ndarray, w: np.ndarray) -> float:
    o = np.argsort(x)
    cw = np.cumsum(w[o])
    return float(x[o][np.searchsorted(cw, cw[-1] / 2)])


# ------------------------------------------------------------------ swing
def _swing_from(frac: np.ndarray, w: np.ndarray) -> dict:
    """Swing from beat fractions. 16th swing: where the 2nd 16th of each 8th-pair lands (0.5 = straight)."""
    n = len(frac)
    if n < 4:
        return {"level": None, "ratio_16th": None, "ratio_8th": None, "n": int(n)}
    u = (frac % 0.5) / 0.5  # position inside an 8th pair
    off16 = (u > 0.3) & (u < 0.84)
    near_8th = (frac > 0.4) & (frac < 0.8)
    trip = (np.abs(frac - 1 / 3) < 0.04) | (np.abs(frac - 2 / 3) < 0.04)
    straight16 = (np.abs(frac - 0.25) < 0.04) | (np.abs(frac - 0.75) < 0.04)
    out: dict = {"n": int(n)}
    if off16.sum() >= max(4, 0.1 * n):
        out["level"] = "16th"
        out["ratio_16th"] = _wmedian(u[off16], w[off16])
        on8 = np.abs(frac - 0.5) < 0.12
        out["ratio_8th"] = _wmedian(frac[on8], w[on8]) if on8.sum() >= 3 else 0.5
    elif near_8th.sum() >= max(3, 0.1 * n):
        out["level"] = "8th"
        out["ratio_8th"] = _wmedian(frac[near_8th], w[near_8th])
        out["ratio_16th"] = None
    else:
        out["level"] = "quarter"
        out["ratio_16th"] = None
        out["ratio_8th"] = None
    out["triplet_grid"] = bool(trip.sum() > 0.25 * n and trip.sum() > 2 * straight16.sum()
                               and np.any(np.abs(frac - 1 / 3) < 0.04) and np.any(np.abs(frac - 2 / 3) < 0.04))
    return out


def _voice_lag_beats(frac: np.ndarray) -> float:
    """A voice's constant push/lag (in beats) measured on its straight 8th positions."""
    d = frac[:, None] - np.array([0.0, 0.5, 1.0])[None, :]
    off = d[np.arange(len(d)), np.argmin(np.abs(d), axis=1)]
    sel = np.abs(off) < 0.06
    return float(np.median(off[sel])) if sel.sum() >= 3 else 0.0


def estimate_swing(hits: dict[str, list], beats: np.ndarray) -> dict:
    """Swing per voice is measured relative to that voice's own on-grid timing, so a laid-back hat
    (constant lag) is reported as microtiming, not as extra swing."""
    voices = [v for v in hits if v not in ("crash",) and len(hits[v])]
    allf, allw = [], []
    by_voice = {}
    for v in voices:
        t = np.array([h[0] for h in hits[v]])
        w = np.array([h[1] for h in hits[v]])
        pos, _ = beat_positions(t, beats)
        frac = pos - np.floor(pos)
        frac = (frac - _voice_lag_beats(frac)) % 1.0
        allf.append(frac)
        allw.append(w)
        sv = _swing_from(frac, w)
        if sv.get("level") == "16th" and sv.get("ratio_16th") is not None:
            by_voice[v] = round(sv["ratio_16th"], 3)
        elif sv.get("level") == "8th" and sv.get("ratio_8th") is not None:
            by_voice[v] = round(sv["ratio_8th"], 3)
    if not allf:
        return {"level": None, "ratio_16th": None, "ratio_8th": None, "by_voice": {}, "confidence": 0.0}
    frac = np.concatenate(allf)
    w = np.concatenate(allw)
    sw = _swing_from(frac, w)
    r = sw.get("ratio_16th") if sw.get("level") == "16th" else sw.get("ratio_8th")
    sw["by_voice"] = by_voice
    sw["percent"] = round(100 * r, 1) if r else None
    if r is None:
        feel = "straight / sparse"
    elif sw.get("triplet_grid"):
        feel = "triplet / shuffle grid"
    elif r < 0.53:
        feel = "straight"
    elif r < 0.6:
        feel = "light swing"
    elif r < 0.65:
        feel = "medium swing"
    else:
        feel = "heavy (triplet) swing"
    sw["feel"] = feel
    # confidence: how consistent the swung onsets are
    if r is not None:
        u = (frac % 0.5) / 0.5 if sw["level"] == "16th" else frac
        sel = (u > 0.3) & (u < 0.84) if sw["level"] == "16th" else (frac > 0.4) & (frac < 0.8)
        spread = float(np.std(u[sel])) if sel.sum() > 2 else 0.1
        sw["confidence"] = round(float(np.clip(1 - spread / 0.08, 0.1, 1.0)), 2)
    else:
        sw["confidence"] = 0.3
    return sw


# ------------------------------------------------------------------ microtiming
def _slot_positions(swing: dict) -> np.ndarray:
    """Expected beat-fraction of the four 16ths in a beat given the swing estimate."""
    r16 = swing.get("ratio_16th") if swing.get("level") == "16th" else None
    if r16:
        return np.array([0.0, 0.5 * r16, 0.5, 0.5 + 0.5 * r16])
    r8 = swing.get("ratio_8th") if swing.get("level") == "8th" else None
    if r8:
        return np.array([0.0, 0.25, r8, r8 + 0.5 * (1 - r8)])
    return np.array([0.0, 0.25, 0.5, 0.75])


def microtiming(hits: dict[str, list], beats: np.ndarray, swing: dict) -> dict:
    """Mean push(-)/lag(+) in ms of each voice's on-grid hits (quarters + 8ths) vs the beat grid."""
    out = {}
    for v, hs in hits.items():
        if v == "crash" or len(hs) < 4:
            continue
        t = np.array([h[0] for h in hs])
        pos, ibi = beat_positions(t, beats)
        frac = pos - np.floor(pos)
        # nearest straight 8th position (8ths are unaffected by 16th swing)
        if swing.get("level") == "8th" and swing.get("ratio_8th"):
            targets = np.array([0.0, swing["ratio_8th"], 1.0])
        else:
            targets = np.array([0.0, 0.5, 1.0])
        d = frac[:, None] - targets[None, :]
        j = np.argmin(np.abs(d), axis=1)
        off = d[np.arange(len(d)), j]
        sel = np.abs(off) < 0.09  # only hits that are clearly meant to be on an 8th
        if sel.sum() < 3:
            continue
        ms = off[sel] * ibi[sel] * 1000
        lo, hi = np.percentile(ms, [10, 90])
        core = ms[(ms >= lo) & (ms <= hi)] if len(ms) >= 10 else ms
        out[v] = {"mean_ms": round(float(np.mean(core)), 1), "std_ms": round(float(np.std(ms)), 1), "n": int(sel.sum())}
    if "kick" in out:
        k = out["kick"]["mean_ms"]
        for v in out:
            out[v]["vs_kick_ms"] = round(out[v]["mean_ms"] - k, 1)
    return out


# ------------------------------------------------------------------ step grids
def bar_grids(hits: dict[str, list], beats: np.ndarray, downbeats: np.ndarray, meter: int, swing: dict,
              resolution: int = 16) -> list[dict]:
    """Quantize hits into per-bar step grids. resolution 16 -> 4 steps/beat (swing-aware), 32 -> 8 steps/beat."""
    beats = np.asarray(beats, dtype=float)
    downbeats = np.asarray(downbeats, dtype=float)
    if len(downbeats) == 0 or len(beats) < 2:
        return []
    db_idx = np.array([int(np.argmin(np.abs(beats - d))) for d in downbeats])
    spb = 4 if resolution == 16 else 8
    steps = meter * spb
    slot_pos = _slot_positions(swing) if resolution == 16 else np.arange(8) / 8.0
    n_bars = len(downbeats)
    bars = [{"bar": b, "start": float(downbeats[b]), "voices": {}, "extra": []} for b in range(n_bars)]
    for v, hs in hits.items():
        if not hs:
            continue
        t = np.array([h[0] for h in hs])
        vel = np.array([h[1] for h in hs])
        pos, ibi = beat_positions(t, beats)
        for ti, p, vi, ib in zip(t, pos, vel, ibi):
            b = int(np.searchsorted(db_idx, p + 0.5 / spb, side="right") - 1)  # early hits belong to the next bar
            if b < 0 or b >= n_bars:
                continue
            rel = p - db_idx[b]  # beats from this bar's downbeat
            beat = int(np.floor(rel + 0.5 / spb))
            frac = rel - beat
            cands = np.concatenate([slot_pos, [1.0]])
            j = int(np.argmin(np.abs(frac - cands)))
            dist = abs(frac - cands[j])
            if j == len(slot_pos):
                beat, j = beat + 1, 0
            bar_len = (db_idx[b + 1] - db_idx[b]) if b + 1 < n_bars else meter
            if beat >= bar_len:
                if b + 1 >= n_bars:
                    continue
                b += 1
                beat -= bar_len
                rel = p - db_idx[b]
            if beat < 0 or beat >= meter:
                continue
            slot = beat * spb + j
            if slot >= steps:
                continue
            grid = bars[b]["voices"].setdefault(v, {"vel": np.zeros(steps), "off_ms": np.full(steps, np.nan)})
            off_ms = (rel - (beat + j / spb)) * ib * 1000  # vs the straight (unswung) grid position
            if dist > 0.3 / spb and resolution == 16:
                bars[b]["extra"].append({"voice": v, "t": round(float(ti), 4), "beat": round(float(rel), 3),
                                         "vel": round(float(vi), 2)})
                continue
            if vi > grid["vel"][slot]:
                grid["vel"][slot] = vi
                grid["off_ms"][slot] = off_ms
    return bars


def grid_string(vel: np.ndarray, accents: bool = False) -> str:
    if not accents:
        return "".join("x" if v > 0 else "." for v in vel)
    return "".join("." if v <= 0 else "o" if v < 0.45 else "X" if v >= 0.85 else "x" for v in vel)


# ------------------------------------------------------------------ templates
def groove_template(bars: list[dict], steps: int) -> dict:
    """Per-voice, per-step mean offset (ms vs straight grid), mean velocity and hit-rate across the track."""
    voices = sorted({v for b in bars for v in b["voices"]})
    active_bars = [b for b in bars if b["voices"]]
    out = {}
    for v in voices:
        vel = np.zeros(steps)
        off = np.zeros(steps)
        cnt = np.zeros(steps)
        for b in active_bars:
            g = b["voices"].get(v)
            if g is None:
                continue
            m = g["vel"] > 0
            vel[m] += g["vel"][m]
            o = np.nan_to_num(g["off_ms"][m])
            off[m] += o
            cnt[m] += 1
        with np.errstate(invalid="ignore", divide="ignore"):
            mv = np.where(cnt > 0, vel / np.maximum(cnt, 1), 0.0)
            mo = np.where(cnt > 0, off / np.maximum(cnt, 1), np.nan)
        out[v] = {"offset_ms": [None if np.isnan(x) else round(float(x), 1) for x in mo],
                  "velocity": [round(float(x), 2) for x in mv],
                  "hit_rate": [round(float(c / max(len(active_bars), 1)), 2) for c in cnt]}
    return out


# ------------------------------------------------------------------ patterns
def _bar_matrix(bar: dict, voices: list[str], steps: int) -> np.ndarray:
    m = np.zeros((len(voices), steps))
    for i, v in enumerate(voices):
        g = bar["voices"].get(v)
        if g is not None:
            m[i] = g["vel"] > 0
    return m


def _dist(a: np.ndarray, b: np.ndarray, w: np.ndarray, cols: slice | None = None) -> float:
    if cols is not None:
        a, b = a[:, cols], b[:, cols]
    diff = (np.abs(a - b) * w[:, None]).sum()
    union = (np.maximum(a, b) * w[:, None]).sum()
    return float(diff / union) if union > 0 else 0.0


def mine_patterns(bars: list[dict], meter: int, steps: int) -> dict:
    voices = [v for v in GRID_VOICES if any(v in b["voices"] for b in bars)]
    if not voices:
        return {"voices": [], "patterns": [], "bar_patterns": [], "fills": [], "stops": [], "rolls": []}
    w = np.array([CORE_WEIGHT.get(v, 0.5) for v in voices])
    mats = [_bar_matrix(b, voices, steps) for b in bars]
    counts = np.array([m.sum() for m in mats])
    med = float(np.median(counts[counts > 0])) if np.any(counts > 0) else 0.0
    empty = counts <= max(1, 0.15 * med)

    # 1) exact-ish clusters
    clusters: list[dict] = []
    assign = [None] * len(bars)
    for i, m in enumerate(mats):
        if empty[i]:
            continue
        best, bd = None, 1e9
        for c in clusters:
            d = _dist(m, c["proto"], w)
            if d < bd:
                best, bd = c, d
        if best is not None and bd <= 0.12:
            best["bars"].append(i)
            best["proto"] = (np.mean([mats[j] for j in best["bars"]], axis=0) >= 0.5).astype(float)
            assign[i] = best
        else:
            c = {"bars": [i], "proto": m.copy()}
            clusters.append(c)
            assign[i] = c
    # 2) label: frequent clusters get letters, close relatives become variations (A', A'')
    clusters.sort(key=lambda c: c["bars"][0])
    by_size = sorted(clusters, key=lambda c: -len(c["bars"]))
    letters = iter("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    for c in by_size:
        c["label"] = None
    mains: list[dict] = []
    for c in sorted(clusters, key=lambda c: (-len(c["bars"]), c["bars"][0])):
        parent = None
        for mcl in mains:
            if _dist(c["proto"], mcl["proto"], w) <= 0.25:
                parent = mcl
                break
        if parent is None and (len(c["bars"]) >= 2 or not mains):
            mains.append(c)
            c["variation_of"] = None
        else:
            c["variation_of"] = parent
    # letters by first appearance among mains
    for c in sorted(mains, key=lambda c: c["bars"][0]):
        c["label"] = next(letters)
    for c in clusters:
        if c.get("variation_of") is not None:
            p = c["variation_of"]
            p.setdefault("_nvar", 0)
            p["_nvar"] += 1
            c["label"] = p["label"] + "'" * min(p["_nvar"], 3) + (str(p["_nvar"]) if p["_nvar"] > 3 else "")
        elif c["label"] is None:
            c["label"] = "X" + str(c["bars"][0])  # one-off bar unlike anything else
    bar_patterns = ["-" if a is None else a["label"] for a in assign]

    # 3) fills: a bar that follows its pattern in the first half but deviates near the end
    fills = []
    half = slice(steps // 2, steps)
    first = slice(0, steps // 2)
    last_beat = slice(steps - steps // meter, steps)
    for i, m in enumerate(mats):
        if empty[i] or assign[i] is None:
            continue
        ctx = _context_proto(i, assign, mats)
        if ctx is None:
            continue
        d1 = _dist(m, ctx, w, first)
        d2 = _dist(m, ctx, w, half)
        dl = _dist(m, ctx, w, last_beat)
        extra = bars[i]["extra"]
        # a bar whose pattern continues in the next bar starts a new groove; it is not a fill
        if i + 1 < len(assign) and assign[i + 1] is assign[i] and len(assign[i]["bars"]) > 1:
            continue
        if (d1 <= 0.25 and (d2 >= 0.3 or dl >= 0.4)) or len(extra) >= 3:
            f = _describe_fill(i, bars[i], m, ctx, voices, meter, steps)
            if f["kind"] != "variation" or d2 >= 0.5:
                fills.append(f)
    # 4) stops / drop-outs (whole bars without drums)
    stops = []
    i = 0
    while i < len(bars):
        if empty[i] and _has_drums_near(i, empty):
            j = i
            while j + 1 < len(bars) and empty[j + 1]:
                j += 1
            stops.append({"start_bar": i, "end_bar": j + 1, "bars": j + 1 - i, "start": bars[i]["start"]})
            i = j + 1
        else:
            i += 1
    rolls = _snare_rolls(bars, meter, steps)
    in_roll = {b for r in rolls for b in range(r["start_bar"], r["end_bar"])}
    fills = [f for f in fills if f["bar"] not in in_roll]
    patterns = []
    for c in sorted(clusters, key=lambda c: c["bars"][0]):
        proto_bars = c["bars"]
        grid, accent, velo = {}, {}, {}
        for vi, v in enumerate(voices):
            vel = np.zeros(steps)
            n = np.zeros(steps)
            for j in proto_bars:
                g = bars[j]["voices"].get(v)
                if g is not None:
                    vel += g["vel"]
                    n += g["vel"] > 0
            present = n >= max(1, 0.5 * len(proto_bars))
            if not present.any():
                continue
            mv = np.where(present, vel / np.maximum(n, 1), 0.0)
            grid[v] = grid_string(mv)
            accent[v] = grid_string(mv, accents=True)
            velo[v] = [round(float(x), 2) for x in mv]
        patterns.append({"id": c["label"], "bars": proto_bars, "count": len(proto_bars),
                         "variation_of": c["variation_of"]["label"] if c.get("variation_of") else None,
                         "grid_16": grid, "accents": accent, "velocity": velo})
    return {"voices": voices, "patterns": patterns, "bar_patterns": bar_patterns, "fills": fills, "stops": stops,
            "rolls": rolls}


def _context_proto(i: int, assign: list, mats: list) -> np.ndarray | None:
    """Prototype of the groove around bar i (previous non-empty bar's cluster, else next)."""
    for j in list(range(i - 1, max(-1, i - 4), -1)) + list(range(i + 1, min(len(assign), i + 3))):
        if assign[j] is not None and assign[j] is not assign[i] or (assign[j] is not None and len(assign[j]["bars"]) > 2):
            return assign[j]["proto"]
    return None


def _has_drums_near(i: int, empty: np.ndarray) -> bool:
    return bool((~empty[:i]).any() and (~empty[i + 1:]).any())


def _describe_fill(i: int, bar: dict, m: np.ndarray, ctx: np.ndarray, voices: list[str], meter: int, steps: int) -> dict:
    spb = steps // meter
    diff = np.abs(m - ctx)
    cols = np.where(diff.any(axis=0))[0]
    start_step = int(cols.min()) if len(cols) else steps - spb
    beat = start_step // spb + 1
    involved = [voices[k] for k in range(len(voices)) if diff[k].any()]
    added = {voices[k]: int(((m[k] > 0) & (ctx[k] == 0)).sum()) for k in range(len(voices))}
    removed = [voices[k] for k in range(len(voices)) if ((m[k] == 0) & (ctx[k] > 0)).sum() >= 2]
    extra = bar["extra"]
    kinds, notes = [], []
    hat_32 = [e for e in extra if e["voice"].startswith("hat")]
    if len(hat_32) >= 2:
        kinds.append("hat_ratchet_32nd")
    sn = added.get("snare", 0) + sum(1 for e in extra if e["voice"] == "snare")
    if sn >= 3:
        kinds.append("snare_roll")
    if added.get("tom", 0) >= 2 or any(e["voice"] == "tom" for e in extra):
        kinds.append("tom_fill")
    claps = [e for e in extra if e["voice"] == "clap"]
    if claps:
        kinds.append("clap_flam")
    if added.get("kick", 0) >= 2:
        kinds.append("kick_pickup")
    if removed and not kinds:
        kinds.append("dropout")
    if not kinds:
        kinds.append("variation")
    # crescendo?
    vels = []
    for v in involved:
        g = bar["voices"].get(v)
        if g is not None:
            vels.extend([x for x in g["vel"][start_step:] if x > 0])
    vels.extend(e["vel"] for e in extra)
    if len(vels) >= 3 and np.corrcoef(np.arange(len(vels)), vels)[0, 1] > 0.5:
        notes.append("crescendo into the next downbeat")
    if removed:
        notes.append("drops " + ", ".join(removed))
    return {"bar": i, "beat": int(beat), "kind": kinds[0], "kinds": kinds, "voices": involved, "notes": "; ".join(notes)}


def _snare_rolls(bars: list[dict], meter: int, steps: int) -> list[dict]:
    """Runs of >= 2 bars whose snare/clap density (or, with no kick, total non-kick density - separated stems
    often scatter a roll's 32nds into hats) keeps increasing: the classic build roll."""
    def count(b, voices, extra=True):
        n = 0
        for v in voices:
            g = b["voices"].get(v)
            if g is not None:
                n += int((g["vel"] > 0).sum())
        if extra:
            n += sum(1 for e in b["extra"] if e["voice"] in voices)
        return n

    sn = [count(b, ("snare", "clap")) for b in bars]
    allv = [count(b, ("snare", "clap", "rim", "hat_closed", "hat_open", "perc", "tom")) for b in bars]
    kick = [count(b, ("kick",), False) for b in bars]
    rolls = []
    i = 0
    while i < len(bars) - 1:
        found = None
        for dens, need_no_kick in ((sn, False), (allv, True)):
            if dens[i] < meter or (need_no_kick and kick[i] > 0):
                continue
            j = i
            while j + 1 < len(dens) and dens[j + 1] > dens[j] and (not need_no_kick or kick[j + 1] == 0):
                j += 1
            if j - i >= 1 and dens[j] >= 2 * max(dens[i], 1) and dens[j] >= steps // 2:
                # keep going through dense, kick-less bars (the roll's last bar often holds a stop)
                while j + 1 < len(bars) and kick[j + 1] == 0 and allv[j + 1] >= 0.5 * allv[j] and allv[j + 1] >= meter:
                    j += 1
                found = (i, j, dens)
                break
        if found:
            a, b, dens = found
            rolls.append({"start_bar": a, "end_bar": b + 1, "density_per_bar": dens[a: b + 1],
                          "technique": "snare roll accelerating"})
            i = b + 1
        else:
            i += 1
    return rolls


# ------------------------------------------------------------------ main
def analyze(hits: dict[str, list], beats, downbeats, meter: int) -> dict:
    beats = np.asarray(beats, dtype=float)
    downbeats = np.asarray(downbeats, dtype=float)
    if len(beats) < 4 or not hits:
        return {"swing": {"level": None, "ratio_16th": None, "by_voice": {}}, "microtiming": {}, "template": {},
                "patterns": {"voices": [], "patterns": [], "bar_patterns": [], "fills": [], "stops": [], "rolls": []},
                "bars": []}
    swing = estimate_swing(hits, beats)
    micro = microtiming(hits, beats, swing)
    steps = meter * 4
    bars = bar_grids(hits, beats, downbeats, meter, swing, 16)
    template = groove_template(bars, steps)
    patterns = mine_patterns(bars, meter, steps)
    compact_bars = [{"bar": b["bar"], "start": round(b["start"], 4), "pattern": patterns["bar_patterns"][b["bar"]],
                     "grid": {v: grid_string(g["vel"]) for v, g in b["voices"].items()},
                     "n_extra": len(b["extra"])} for b in bars]
    return {"swing": swing, "microtiming": micro, "template": template, "patterns": patterns, "bars": compact_bars}


def grid_for_bars(hits, beats, downbeats, meter, swing, resolution=16, bars=None, voices=None) -> list[dict]:
    """On-demand step grids (used by get_drum_grid), optionally at 32nd resolution."""
    out = []
    for b in bar_grids(hits, beats, downbeats, meter, swing, resolution):
        if bars is not None and b["bar"] not in bars:
            continue
        row = {"bar": b["bar"], "start": round(b["start"], 4), "grid": {}, "velocity": {}, "offset_ms": {}}
        for v, g in b["voices"].items():
            if voices and v not in voices:
                continue
            row["grid"][v] = grid_string(g["vel"])
            row["velocity"][v] = [round(float(x), 2) for x in g["vel"]]
            row["offset_ms"][v] = [None if np.isnan(x) else round(float(x), 1) for x in g["off_ms"]]
        if b["extra"]:
            row["off_grid"] = b["extra"]
        out.append(row)
    return out


def most_common_pattern(patterns: dict) -> dict | None:
    pats = [p for p in patterns.get("patterns", []) if p.get("variation_of") is None]
    if not pats:
        return None
    return max(pats, key=lambda p: p["count"])


__all__ = ["analyze", "estimate_swing", "microtiming", "bar_grids", "grid_for_bars", "mine_patterns", "mode"]
