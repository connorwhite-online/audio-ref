"""Part analysis from note events: bass line, melody/hook, keys voicing & comping."""

from __future__ import annotations

import numpy as np

from .groove import _slot_positions, beat_positions, grid_string
from .theory import parse_chord
from .util import midi_name, mode

VERSION = "2"


# ------------------------------------------------------------------ helpers
def _monophonic(notes: list[tuple], lowest: bool = True) -> list[tuple]:
    """Collapse simultaneous notes (octave doublings etc.) to a single line."""
    if not notes:
        return []
    notes = sorted(notes, key=lambda n: (n[0], n[2] if lowest else -n[2]))
    out = [notes[0]]
    for n in notes[1:]:
        if n[0] - out[-1][0] < 0.03:
            # same onset: keep the lower (bass) / higher (melody) unless much weaker
            prev = out[-1]
            better = (n[2] < prev[2]) if lowest else (n[2] > prev[2])
            if better and n[3] > 0.5 * prev[3]:
                out[-1] = n
            continue
        out.append(n)
    return out


def _onset_slots(times: np.ndarray, beats: np.ndarray, downbeats: np.ndarray, meter: int, swing: dict) -> list[tuple[int, int]]:
    """(bar, slot) for each onset on the swing-aware 16th grid."""
    if len(times) == 0 or len(beats) < 2 or len(downbeats) == 0:
        return []
    db_idx = np.array([int(np.argmin(np.abs(beats - d))) for d in downbeats])
    slots = np.concatenate([_slot_positions(swing), [1.0]])
    pos, _ = beat_positions(np.asarray(times), beats)
    out = []
    for p in pos:
        b = int(np.searchsorted(db_idx, p + 0.125, side="right") - 1)
        if b < 0:
            continue
        rel = p - db_idx[b]
        beat = int(np.floor(rel + 0.125))
        j = int(np.argmin(np.abs((rel - beat) - slots)))
        if j == 4:
            beat, j = beat + 1, 0
        if beat >= meter:
            b, beat = b + 1, beat - meter
        if 0 <= beat < meter and b < len(downbeats):
            out.append((b, beat * 4 + j))
    return out


def _rhythm_grid(slots: list[tuple[int, int]], meter: int, n_bars_active: int) -> tuple[str, list[float], str | None]:
    steps = meter * 4
    if not slots or n_bars_active == 0:
        return "." * steps, [0.0] * steps, None
    hist = np.zeros(steps)
    per_bar: dict[int, list[int]] = {}
    for b, s in slots:
        per_bar.setdefault(b, []).append(s)
    for b, ss in per_bar.items():
        for s in set(ss):
            hist[s] += 1
    rate = hist / max(len(per_bar), 1)
    grid = "".join("x" if r >= 0.4 else "." for r in rate)
    common = mode(["".join("x" if s in set(ss) else "." for s in range(steps)) for ss in per_bar.values()])
    return grid, [round(float(r), 2) for r in rate], common


def _chord_at(chords: list[dict], t: float):
    for c in chords:
        if c["start"] <= t < c["end"]:
            return c
    return None


# ------------------------------------------------------------------ bass
def analyze_bass(notes: list[tuple], beats, downbeats, meter: int, swing: dict, chords: list[dict],
                 kick_times: list[float] | None = None) -> dict:
    beats = np.asarray(beats, dtype=float)
    downbeats = np.asarray(downbeats, dtype=float)
    notes = [n for n in notes if 20 <= n[2] <= 67]
    line = _monophonic(notes, lowest=True)
    if len(line) < 3:
        return {"present": False, "n_notes": len(line)}
    pitches = np.array([n[2] for n in line])
    on = np.array([n[0] for n in line])
    dur = np.array([n[1] - n[0] for n in line])
    ibi = float(np.median(np.diff(beats))) if len(beats) > 1 else 0.5
    slots = _onset_slots(on, beats, downbeats, meter, swing)
    active_bars = len({b for b, _ in slots})
    grid, rate, common = _rhythm_grid(slots, meter, active_bars)
    iv = np.diff(pitches)
    gaps = np.diff(on)
    octave = float(np.mean(np.abs(iv) == 12)) if len(iv) else 0.0
    glides = float(np.mean(np.array([n[4] for n in line]) >= 0.8))  # notes that slide into their pitch
    # root following
    root_hits = third = fifth = other = 0.0
    for n in line:
        c = _chord_at(chords, (n[0] + n[1]) / 2)
        if not c or c["symbol"] == "N":
            continue
        ch = parse_chord(c["symbol"])
        d = (n[2] - ch.root) % 12
        w = n[1] - n[0]
        if d == 0:
            root_hits += w
        elif d in (3, 4):
            third += w
        elif d == 7:
            fifth += w
        else:
            other += w
    tot = root_hits + third + fifth + other
    relation, locked, kick_cov = None, None, None
    if kick_times:
        k = np.asarray(sorted(kick_times))
        near = [np.min(np.abs(k - t)) < 0.035 for t in on]
        locked = float(np.mean(near))
        kick_cov = float(np.mean([np.min(np.abs(on - t)) < 0.035 for t in k]))
        offbeat = np.mean([s % 4 != 0 for _, s in slots]) if slots else 0.0
        if locked >= 0.6:
            relation = "locked to the kick"
        elif locked <= 0.3 and offbeat >= 0.5:
            relation = "syncopated around the kick (plays between kicks)"
        elif locked <= 0.3:
            relation = "independent of the kick"
        else:
            relation = "partly locked to the kick, with syncopated pickups"
    legato = float(np.median(dur[:-1] / np.maximum(gaps, 1e-3))) if len(gaps) else 1.0
    lo, hi = int(np.percentile(pitches, 3)), int(np.percentile(pitches, 97))
    return {
        "present": True,
        "n_notes": len(line),
        "range_midi": [lo, hi],
        "range_names": [midi_name(lo), midi_name(hi)],
        "register": "sub" if hi < 40 else "low" if hi < 52 else "mid-low",
        "rhythm_grid_16": grid,
        "rhythm_hit_rate": rate,
        "most_common_bar": common,
        "notes_per_bar": round(len(line) / max(active_bars, 1), 2),
        "median_length_16ths": round(float(np.median(dur) / (ibi / 4)), 2),
        "articulation": "legato" if legato > 0.85 else "staccato" if legato < 0.5 else "detached",
        "glides": round(glides, 2),
        "octave_jumps": round(octave, 2),
        "chord_tone_use": {"root": round(root_hits / tot, 2), "third": round(third / tot, 2),
                           "fifth": round(fifth / tot, 2), "other": round(other / tot, 2)} if tot else None,
        "kick_lock": None if locked is None else round(locked, 2),
        "kick_coverage": None if kick_cov is None else round(kick_cov, 2),
        "relation_to_kick": relation,
    }


# ------------------------------------------------------------------ melody
def analyze_melody(notes: list[tuple], beats, downbeats, meter: int, source: str) -> dict:
    beats = np.asarray(beats, dtype=float)
    downbeats = np.asarray(downbeats, dtype=float)
    line = _monophonic([n for n in notes if n[2] >= 52], lowest=False)
    if len(line) < 4:
        return {"present": False, "source": source}
    p = np.array([n[2] for n in line])
    on = np.array([n[0] for n in line])
    iv = np.diff(p)
    steps = np.mean(np.abs(iv) <= 2) if len(iv) else 0.0
    leaps = np.mean(np.abs(iv) >= 5) if len(iv) else 0.0
    # motif repetition: share of 4-note interval patterns that occur more than once
    grams = [tuple(iv[i: i + 3]) for i in range(len(iv) - 2)]
    rep = 0.0
    top = None
    if grams:
        counts: dict[tuple, int] = {}
        for g in grams:
            counts[g] = counts.get(g, 0) + 1
        rep = float(np.mean([counts[g] > 1 for g in grams]))
        top = max(counts, key=lambda g: counts[g])
    bars = len(downbeats) or 1
    bar_idx = np.searchsorted(downbeats, on, side="right") - 1
    active = len(set(bar_idx[bar_idx >= 0].tolist()))
    contour = "rising" if np.polyfit(np.arange(len(p)), p, 1)[0] > 0.05 else "falling" if np.polyfit(np.arange(len(p)), p, 1)[0] < -0.05 else "arch/level"
    return {
        "present": True,
        "source": source,
        "n_notes": len(line),
        "range_midi": [int(np.percentile(p, 3)), int(np.percentile(p, 97))],
        "range_names": [midi_name(np.percentile(p, 3)), midi_name(np.percentile(p, 97))],
        "notes_per_bar": round(len(line) / max(active, 1), 2),
        "bars_with_melody": round(active / bars, 2),
        "stepwise": round(float(steps), 2),
        "leaps": round(float(leaps), 2),
        "contour": contour,
        "motif_repetition": round(rep, 2),
        "top_motif_intervals": list(map(int, top)) if top else None,
        "pitch_classes": _pc_hist(p),
    }


def _pc_hist(p: np.ndarray) -> dict[str, float]:
    from .util import pc_name

    h = np.bincount(p % 12, minlength=12) / len(p)
    order = np.argsort(-h)[:5]
    return {pc_name(int(i)): round(float(h[i]), 2) for i in order if h[i] > 0}


# ------------------------------------------------------------------ keys / voicing
def analyze_voicing(notes: list[tuple], chords: list[dict], beats, downbeats, meter: int, swing: dict) -> dict:
    beats = np.asarray(beats, dtype=float)
    downbeats = np.asarray(downbeats, dtype=float)
    notes = [n for n in notes if 40 <= n[2] <= 96]
    if len(notes) < 4:
        return {"present": False}
    # group simultaneous onsets into stabs/voicings
    groups: list[list[tuple]] = []
    for n in sorted(notes):
        if groups and n[0] - groups[-1][0][0] < 0.035:
            groups[-1].append(n)
        else:
            groups.append([n])
    sizes = np.array([len(g) for g in groups])
    ibi = float(np.median(np.diff(beats))) if len(beats) > 1 else 0.5
    durs = np.array([np.median([n[1] - n[0] for n in g]) for g in groups])
    poly = float(np.median(sizes))
    ext_counts: dict[str, int] = {}
    inv_counts: dict[str, int] = {}
    spreads = []
    lows, highs = [], []
    for g in groups:
        if len(g) < 2:
            continue
        c = _chord_at(chords, g[0][0] + 0.02)
        ps = sorted(n[2] for n in g)
        spreads.append(ps[-1] - ps[0])
        lows.append(ps[0])
        highs.append(ps[-1])
        if not c or c["symbol"] == "N":
            continue
        ch = parse_chord(c["symbol"])
        for pch in ps:
            d = (pch - ch.root) % 12
            name = {2: "9", 5: "11", 9: "13", 10: "b7", 11: "maj7", 1: "b9", 3: "b3/#9", 6: "#11/b5", 8: "b13/#5"}.get(d)
            if name in ("9", "11", "13", "b7", "maj7", "b9", "#11/b5", "b13/#5"):
                ext_counts[name] = ext_counts.get(name, 0) + 1
        lowd = (ps[0] - ch.root) % 12
        inv = {0: "root", 3: "3rd", 4: "3rd", 7: "5th", 10: "7th", 11: "7th"}.get(lowd, "other")
        inv_counts[inv] = inv_counts.get(inv, 0) + 1
    n_g = max(sum(inv_counts.values()), 1)
    slots = _onset_slots(np.array([g[0][0] for g in groups]), beats, downbeats, meter, swing)
    active = len({b for b, _ in slots})
    grid, rate, common = _rhythm_grid(slots, meter, active)
    med_dur_beats = float(np.median(durs) / ibi)
    sequential = poly < 1.5
    style = ("arpeggiated / single-note line" if sequential else
             "short stabs" if med_dur_beats < 0.6 else "sustained pads/chords" if med_dur_beats > 1.5 else "comping chords")
    offbeat = float(np.mean([s % 4 != 0 for _, s in slots])) if slots else 0.0
    total_ext = sum(ext_counts.values()) or 1
    return {
        "present": True,
        "polyphony": round(poly, 1),
        "register_midi": [int(np.percentile(lows, 10)) if lows else None, int(np.percentile(highs, 90)) if highs else None],
        "spread_semitones": round(float(np.median(spreads)), 1) if spreads else None,
        "extensions": [k for k, v in sorted(ext_counts.items(), key=lambda kv: -kv[1]) if v / total_ext >= 0.08],
        "inversions": {k: round(v / n_g, 2) for k, v in sorted(inv_counts.items(), key=lambda kv: -kv[1])},
        "comping_grid_16": grid,
        "comping_hit_rate": rate,
        "most_common_bar": common,
        "stab_length_beats": round(med_dur_beats, 2),
        "style": style,
        "syncopation": round(offbeat, 2),
        "stabs_per_bar": round(len(groups) / max(active, 1), 2),
    }
