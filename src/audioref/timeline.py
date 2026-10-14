"""Map a target spec onto a video timeline: solve BPM + bar offset so cues land on the grid, then plan
sections, builds, stops, fills and the climax around the cues.

Cue sheet: {"duration": 42.0, "cues": [[t, kind, opts], ...]}  (opts optional: {"label", "weight",
"must_hit", "level": "bar"|"beat"|"half"|"8th", "duration"}). Dict cues {"t", "kind", ...} also work.
"""

from __future__ import annotations

import math

import numpy as np

BAR_KINDS = {"drop", "climax", "scene", "section", "start", "intro", "downbeat", "chorus", "reveal", "logo", "end",
             "outro", "breakdown", "button", "impact"}
BEAT_KINDS = {"hit", "accent", "whoosh", "beat", "cut", "transition", "stop", "hold", "build"}
STRUCTURAL = {"scene", "section", "drop", "climax", "build", "hold", "stop", "breakdown", "outro", "end", "intro",
              "chorus", "start", "reveal", "whoosh", "impact", "logo", "button"}
DEFAULT_ENERGY = {"intro": 0.25, "groove": 0.55, "build": 0.65, "drop": 0.85, "climax": 1.0, "breakdown": 0.3,
                  "stop": 0.0, "outro": 0.35}
DEFAULT_LAYERS = {
    "intro": ["pad", "keys"], "groove": ["kick", "snare/clap", "hats", "bass", "keys"],
    "build": ["snare roll", "bass", "keys", "riser"], "drop": ["kick", "snare/clap", "hats", "perc", "bass", "keys"],
    "climax": ["kick", "snare/clap", "hats", "perc", "bass", "keys", "lead", "pad"], "breakdown": ["pad", "keys"],
    "stop": [], "outro": ["pad", "keys"],
}


# ------------------------------------------------------------------ parsing
def parse_timeline(tl: dict) -> tuple[float, list[dict]]:
    if not isinstance(tl, dict) or "cues" not in tl:
        raise ValueError("timeline must be {duration, cues: [[t, kind, opts], ...]}")
    cues = []
    for c in tl["cues"]:
        if isinstance(c, dict):
            t, kind, opts = c.get("t", c.get("time")), c.get("kind", "hit"), {k: v for k, v in c.items() if k not in ("t", "time", "kind")}
        else:
            t, kind = c[0], c[1] if len(c) > 1 else "hit"
            opts = c[2] if len(c) > 2 and isinstance(c[2], dict) else ({"label": c[2]} if len(c) > 2 else {})
        if t is None:
            raise ValueError(f"cue without time: {c}")
        kind = str(kind).lower()
        level = opts.get("level") or ("bar" if kind in BAR_KINDS else "beat")
        cues.append({"t": float(t), "kind": kind, "label": opts.get("label") or kind, "weight": float(opts.get("weight", 1.0)),
                     "must_hit": bool(opts.get("must_hit", kind in ("drop", "climax"))), "level": level,
                     "duration": opts.get("duration"), "opts": opts})
    cues.sort(key=lambda c: c["t"])
    dur = float(tl.get("duration") or (cues[-1]["t"] + 4.0 if cues else 30.0))
    return dur, cues


# ------------------------------------------------------------------ grid solving
def _level_beats(level: str, meter: int) -> float:
    return {"bar": float(meter), "half": meter / 2.0, "beat": 1.0, "8th": 0.5, "16th": 0.25}.get(level, 1.0)


def solve(cues: list[dict], target_bpm: float, meter: int, constraints: dict | None = None) -> list[dict]:
    """Return candidate (bpm, offset) solutions, best first."""
    c = constraints or {}
    lo, hi = c.get("bpm_range") or (target_bpm * 0.92, target_bpm * 1.08)
    lo, hi = float(min(lo, hi)), float(max(lo, hi))
    integer = bool(c.get("integer_bpm", False))
    tol = float(c.get("tolerance_ms", 25.0))
    must = set(c.get("must_hit") or [])
    step = 1.0 if integer else 0.05
    bpms = np.arange(math.ceil(lo / step) * step, hi + 1e-9, step)
    if not len(bpms):
        bpms = np.array([round(target_bpm, 2)])
    if not cues:
        b = float(np.clip(target_bpm, lo, hi))
        return [{"bpm": round(b, 2), "offset_s": 0.0, "rms_error_ms": 0.0, "max_error_ms": 0.0, "must_hits_ok": True, "cost": 0.0}]
    T = np.array([q["t"] for q in cues])
    Lb = np.array([_level_beats(q["level"], meter) for q in cues])
    W = np.array([q["weight"] for q in cues])
    M = np.array([q["must_hit"] or q["label"] in must or str(i) in must for i, q in enumerate(cues)])
    W = W * np.where(M, 4.0, 1.0)
    cap = 150.0
    results = []
    for bpm in bpms:
        beat = 60.0 / bpm
        bar = beat * meter
        offs = np.arange(0.0, bar, 0.002)
        P = Lb * beat  # grid period per cue (s)
        r = np.mod(T[None, :] - offs[:, None], P[None, :])
        d = np.minimum(r, P[None, :] - r) * 1000.0  # ms
        dc = np.minimum(d, cap)
        cost = (W[None, :] * dc ** 2).sum(axis=1) / W.sum()
        miss = ((d > tol) & M[None, :]).sum(axis=1)
        cost = cost + miss * 1e5
        prior = (20.0 ** 2) * (math.log2(bpm / target_bpm) / 0.05) ** 2
        if not integer:
            prior += (3.0 ** 2) * min(abs(bpm - round(bpm)), 0.5)
        cost = cost + prior
        j = int(np.argmin(cost))
        results.append((float(cost[j]), float(bpm), float(offs[j]), d[j]))
    results.sort(key=lambda x: x[0])
    out = []
    seen = []
    for cost, bpm, off, d in results:
        if any(abs(bpm - s) < 0.5 for s in seen):
            continue
        seen.append(bpm)
        out.append({"bpm": round(bpm, 2), "offset_s": round(off, 4), "rms_error_ms": round(float(np.sqrt(np.mean(d ** 2))), 1),
                    "max_error_ms": round(float(d.max()), 1), "must_hits_ok": bool(np.all(d[M] <= tol)) if M.any() else True,
                    "cost": round(cost, 1)})
        if len(out) >= 3:
            break
    return out


# ------------------------------------------------------------------ planning
def _quality(err: float) -> str:
    e = abs(err)
    return "tight" if e < 10 else "good" if e < 25 else "loose" if e < 60 else "off-grid"


def plan(spec: dict, timeline: dict, constraints: dict | None = None) -> dict:
    duration, cues = parse_timeline(timeline)
    tempo = spec.get("tempo") or {}
    meter = int(tempo.get("beats_per_bar") or 4)
    target = float((constraints or {}).get("target_bpm") or tempo.get("bpm") or 120.0)
    cands = solve(cues, target, meter, constraints)
    best = cands[0]
    bpm, off = best["bpm"], best["offset_s"]
    beat = 60.0 / bpm
    bar = beat * meter
    tol = float((constraints or {}).get("tolerance_ms", 25.0))
    # bar lines covering [0, duration]
    k0 = -1 if off > 1e-6 else 0
    bar_starts = [off + k * bar for k in range(k0, int(math.ceil((duration - off) / bar)) + 1)]
    bar_starts = [b for b in bar_starts if b < duration + 1e-6 and b + bar > 0]

    def bar_of(t: float) -> tuple[int, float]:
        """1-based bar number (bar 1 = first full bar at `off`; pickup bar = 0) and 1-based beat position."""
        k = math.floor((t - off) / bar + 1e-9)
        beat_pos = (t - (off + k * bar)) / beat + 1
        return k + 1, beat_pos

    sync = []
    for q in cues:
        p = _level_beats(q["level"], meter) * beat
        k = round((q["t"] - off) / p)
        g = off + k * p
        err = (q["t"] - g) * 1000
        bnum, bpos = bar_of(g)
        sync.append({"t": q["t"], "kind": q["kind"], "label": q["label"], "level": q["level"], "grid_time": round(g, 4),
                     "bar": bnum, "beat": round(bpos, 2), "error_ms": round(err, 1), "quality": _quality(err),
                     "ok": abs(err) <= tol, "must_hit": q["must_hit"]})

    # structural boundaries snapped to the grid
    st = spec.get("structure") or {}
    e_by = {**DEFAULT_ENERGY, **{k: v for k, v in (st.get("energy_by_label") or {}).items() if k}}
    l_by = {**DEFAULT_LAYERS, **{k: v for k, v in (st.get("layers_by_label") or {}).items() if k and v}}
    if "climax" not in (st.get("energy_by_label") or {}):
        e_by["climax"] = min(1.0, e_by.get("drop", 0.85) + 0.12)
        l_by["climax"] = sorted(set(l_by.get("drop", DEFAULT_LAYERS["drop"])) | {"lead", "pad"})
    bounds: list[tuple[float, dict]] = []
    for q, s in zip(cues, sync):
        if q["kind"] in STRUCTURAL:
            bounds.append((s["grid_time"], q))
    segs = []
    times = [0.0] + [b[0] for b in bounds] + [duration]
    labels_q = [None] + [b[1] for b in bounds]
    for i in range(len(times) - 1):
        t0, t1 = times[i], times[i + 1]
        if t1 - t0 < beat * 0.5:
            continue
        q = labels_q[i]
        segs.append({"start_s": t0, "end_s": t1, "cue": q})
    seen_drop = False
    for i, sgm in enumerate(segs):
        q = sgm["cue"]
        kind = q["kind"] if q else ("intro" if i == 0 else "groove")
        nxt = segs[i + 1]["cue"]["kind"] if i + 1 < len(segs) and segs[i + 1]["cue"] else None
        if kind in ("hold", "stop"):
            label = "stop"
            d = q.get("duration")
            if d:
                sgm["end_s"] = min(sgm["end_s"], sgm["start_s"] + float(d))
        elif kind == "build":
            label = "build"
        elif kind in ("drop", "reveal", "impact", "whoosh", "chorus"):
            label = "drop"
            seen_drop = True
        elif kind == "climax":
            label = "climax"
            seen_drop = True
        elif kind == "breakdown":
            label = "breakdown"
        elif kind in ("outro", "end", "button", "logo"):
            label = "outro"
        elif kind in ("intro", "start"):
            label = "intro"
        else:
            label = "groove" if (seen_drop or i > 0) else "intro"
            if nxt in ("drop", "climax", "reveal", "impact") and not seen_drop and i > 0:
                label = "groove"
        sgm["label"] = label
    # fill gaps left by shortened stops with the previous mood
    filled = []
    for sgm in segs:
        if filled and sgm["start_s"] > filled[-1]["end_s"] + 1e-6:
            filled.append({"start_s": filled[-1]["end_s"], "end_s": sgm["start_s"], "cue": None, "label": "breakdown"})
        filled.append(sgm)
    segs = filled
    # automatic builds into drops / climax (spec build length), and a stop before the drop if the reference does it
    builds = st.get("builds") or []
    build_bars = int(np.median([b["bars"] for b in builds])) if builds else 4
    techniques = sorted({t for b in builds for t in (b.get("technique") or [])}) or ["snare roll accelerating", "filter opening", "noise riser / sweep"]
    stop_before = bool(st.get("stops_before_drops"))
    events = []
    new_segs = []
    for i, sgm in enumerate(segs):
        nxt = segs[i + 1] if i + 1 < len(segs) else None
        if nxt and nxt["label"] in ("drop", "climax") and sgm["label"] not in ("build", "stop"):
            span_bars = (sgm["end_s"] - sgm["start_s"]) / bar
            nb = min(build_bars, max(1, int(span_bars // 2)))
            if span_bars >= 2 and sgm["label"] != "build":
                cut = sgm["end_s"] - nb * bar
                head = {**sgm, "end_s": cut}
                tail = {"start_s": cut, "end_s": sgm["end_s"], "cue": None, "label": "build", "auto": True}
                if cut - sgm["start_s"] > beat * 0.5:
                    new_segs.append(head)
                new_segs.append(tail)
                continue
        new_segs.append(sgm)
    segs = new_segs
    for i, sgm in enumerate(segs):
        lab = sgm["label"]
        sgm["energy"] = round(float(e_by.get(lab, 0.5)), 2)
        sgm["layers"] = list(l_by.get(lab, []))
        b0, _ = bar_of(sgm["start_s"] + 1e-6)
        b1, _ = bar_of(sgm["end_s"] - 1e-6)
        sgm["start_bar"], sgm["end_bar"] = b0, b1
        sgm["bars"] = round((sgm["end_s"] - sgm["start_s"]) / bar, 2)
        if lab == "build":
            sgm["technique"] = techniques
            sgm["energy_ramp"] = [round(e_by.get("groove", 0.55), 2), round(e_by.get("drop", 0.85), 2)]
            events.append({"type": "build", "start_s": round(sgm["start_s"], 3), "end_s": round(sgm["end_s"], 3),
                           "start_bar": b0, "end_bar": b1, "technique": techniques})
        if lab in ("drop", "climax"):
            events.append({"type": lab if lab == "climax" else "drop", "time": round(sgm["start_s"], 3), "bar": b0,
                           "beat": round(bar_of(sgm["start_s"] + 1e-6)[1], 2)})
            prev = segs[i - 1] if i > 0 else None
            if stop_before and prev is not None and prev["label"] != "stop":
                events.append({"type": "stop", "time": round(sgm["start_s"] - beat, 3), "end": round(sgm["start_s"], 3),
                               "bar": bar_of(sgm["start_s"] - beat + 1e-6)[0], "beats": 1, "notes": "silence the last beat before the drop"})
        if lab == "stop":
            events.append({"type": "stop", "time": round(sgm["start_s"], 3), "end": round(sgm["end_s"], 3),
                           "bar": b0, "beats": round((sgm["end_s"] - sgm["start_s"]) / beat, 2),
                           "notes": "everything out (let reverb tails ring)"})
    # fills: last beat before each new section (not into a stop), plus phrase fills in long sections
    fills_rate = float((spec.get("drums") or {}).get("fills_per_8_bars") or 0.5)
    fill_kind = next((f.get("kind") for f in (spec.get("drums") or {}).get("fills") or []), "snare_roll")
    for i, sgm in enumerate(segs[1:], 1):
        if sgm["label"] == "stop" or segs[i - 1]["label"] in ("stop", "intro", "build"):
            continue
        t = sgm["start_s"] - beat
        if t > 0:
            bn, bp = bar_of(t + 1e-6)
            events.append({"type": "fill", "time": round(t, 3), "bar": bn, "beat": round(bp, 2), "kind": fill_kind,
                           "notes": f"into {sgm['label']}"})
    if fills_rate >= 0.4:
        for sgm in segs:
            if sgm["label"] in ("groove", "drop", "climax") and sgm["bars"] >= 8:
                k = 8
                while k < sgm["bars"] - 0.5:
                    t = sgm["start_s"] + k * bar - beat
                    bn, bp = bar_of(t + 1e-6)
                    events.append({"type": "fill", "time": round(t, 3), "bar": bn, "beat": round(bp, 2), "kind": fill_kind,
                                   "notes": "phrase end"})
                    k += 8
    if not any(e["type"] == "climax" for e in events):
        hot = [s for s in segs if s["label"] in ("drop", "climax")]
        target = max(hot, key=lambda s: (s["start_s"] >= 0.5 * duration, s["energy"], s["start_s"])) if hot else \
            max(segs, key=lambda s: (s["energy"], s["start_s"]))
        target["label"] = "climax" if target["label"] == "drop" else target["label"]
        target["energy"] = round(float(e_by.get("climax", 1.0)), 2)
        target["layers"] = list(l_by.get("climax", target["layers"]))
        events.append({"type": "climax", "time": round(target["start_s"], 3), "bar": target["start_bar"], "auto": True})
    events.sort(key=lambda e: e.get("time", e.get("start_s", 0)))
    # bar map
    bar_map = []
    for bs in bar_starts:
        seg = next((s for s in segs if s["start_s"] - 1e-6 <= max(bs, 0) < s["end_s"] - 1e-6), segs[-1] if segs else None)
        bn = bar_of(bs + 1e-6)[0]
        evs = [e["type"] + (f"@beat{e['beat']}" if e.get("beat") and e["type"] == "fill" else "") for e in events
               if bs - 1e-6 <= e.get("time", e.get("start_s", -1)) < bs + bar - 1e-6]
        evs += [f"cue:{s['label']}" for s in sync if bs - 1e-6 <= s["grid_time"] < bs + bar - 1e-6]
        bar_map.append({"bar": bn, "start_s": round(max(bs, 0.0), 3), "section": seg["label"] if seg else None,
                        "energy": seg["energy"] if seg else None, "events": evs})
    sections = [{"label": s["label"], "start_s": round(s["start_s"], 3), "end_s": round(s["end_s"], 3), "start_bar": s["start_bar"],
                 "end_bar": s["end_bar"], "bars": s["bars"], "energy": s["energy"], "layers": s["layers"],
                 **({"technique": s["technique"]} if s.get("technique") else {}),
                 **({"cue": s["cue"]["label"]} if s.get("cue") else {}), **({"auto": True} if s.get("auto") else {})}
                for s in segs]
    notes = _notes(sections, sync, events, bpm, off, meter)
    out = {"kind": "audioref.score_plan", "bpm": bpm, "meter": f"{meter}/4", "beats_per_bar": meter, "offset_s": off,
           "first_downbeat_s": off, "beat_s": round(beat, 5), "bar_s": round(bar, 5), "duration_s": duration,
           "candidates": cands, "cue_sync": sync, "sections": sections, "events": events, "bar_map": bar_map,
           "notes": notes, "spec_sources": spec.get("sources")}
    out["brief_md"] = plan_brief(out)
    return out


def _notes(sections, sync, events, bpm, off, meter) -> list[str]:
    n = [f"Use {bpm} bpm in {meter}/4 with bar 1 starting at {off:.3f} s" + (" (pickup before it)" if off > 0.05 else "") + "."]
    bad = [s for s in sync if not s["ok"]]
    if bad:
        n.append("Off-grid cues (consider nudging the edit or a tempo change): " +
                 ", ".join(f"{s['label']} @ {s['t']:.2f}s ({s['error_ms']:+.0f} ms)" for s in bad))
    else:
        n.append(f"All {len(sync)} cues land within tolerance of the grid.")
    for e in events:
        if e["type"] == "build":
            n.append(f"Build bars {e['start_bar']}–{e['end_bar']} ({', '.join(e['technique'])}).")
        elif e["type"] == "stop":
            n.append(f"Stop at {e['time']:.2f}s (bar {e['bar']}) for {e['beats']} beat(s): {e.get('notes', '')}.")
        elif e["type"] in ("drop", "climax"):
            n.append(f"{e['type'].capitalize()} at bar {e['bar']} ({e['time']:.2f}s).")
    return n


def plan_brief(p: dict) -> str:
    lines = ["# Score plan", "", f"**{p['bpm']} bpm, {p['meter']}, bar 1 at {p['offset_s']:.3f} s** "
             f"(beat {p['beat_s'] * 1000:.0f} ms, bar {p['bar_s']:.3f} s; {p['duration_s']:.1f} s total)", "",
             "| cue | t (s) | lands on | error | |", "|---|---|---|---|---|"]
    for s in p["cue_sync"]:
        lines.append(f"| {s['label']} ({s['kind']}) | {s['t']:.2f} | bar {s['bar']} beat {s['beat']:.2f} | {s['error_ms']:+.0f} ms | {s['quality']} |")
    lines += ["", "| bars | section | energy | layers |", "|---|---|---|---|"]
    for s in p["sections"]:
        lines.append(f"| {s['start_bar']}–{s['end_bar']} | {s['label']}{' (auto)' if s.get('auto') else ''} | {s['energy']} | {', '.join(s['layers'])} |")
    lines += ["", "Notes:"] + [f"- {x}" for x in p["notes"]]
    if len(p["candidates"]) > 1:
        lines.append("- Alternatives: " + ", ".join(f"{c['bpm']} bpm (rms {c['rms_error_ms']} ms)" for c in p["candidates"][1:]))
    return "\n".join(lines) + "\n"
