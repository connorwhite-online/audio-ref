"""Multi-reference comparison (per-dimension similarity) and blending into a target spec."""

from __future__ import annotations

import math

import numpy as np

from .theory import parse_chord, parse_key

DIMENSIONS = ("tempo", "groove", "drums", "harmony", "bass", "sound", "structure")
DIM_WEIGHTS = {"tempo": 1.0, "groove": 1.0, "drums": 1.2, "harmony": 1.2, "bass": 0.8, "sound": 1.0, "structure": 0.8}
ROLES = {"kick": ("kick",), "backbeat": ("snare", "clap", "rim"), "hats": ("hat_closed", "hat_open", "ride", "perc")}


# ------------------------------------------------------------------ features
def _main_pattern(a: dict) -> dict | None:
    pats = [p for p in a.get("drums", {}).get("patterns", []) if p.get("variation_of") is None]
    return max(pats, key=lambda p: p["count"]) if pats else None


def _role_vectors(p: dict | None, steps: int = 16) -> dict[str, np.ndarray]:
    out = {}
    for role, voices in ROLES.items():
        v = np.zeros(steps)
        if p:
            for voice in voices:
                g = p.get("grid_16", {}).get(voice)
                if g and len(g) == steps:
                    v = np.maximum(v, np.array([c == "x" for c in g], dtype=float))
        out[role] = v
    return out


def _swing(a: dict) -> tuple[float, str | None]:
    sw = a.get("rhythm", {}).get("swing") or {}
    lvl = sw.get("level")
    r = sw.get("ratio_16th") if lvl == "16th" else sw.get("ratio_8th") if lvl == "8th" else None
    return (float(r) if r else 0.5), lvl


def features(a: dict) -> dict:
    r, h, s, st = a.get("rhythm", {}), a.get("harmony", {}), a.get("sound", {}), a.get("structure", {})
    main = _main_pattern(a)
    meter = r.get("beats_per_bar") or 4
    swing, lvl = _swing(a)
    md = r.get("microtiming_detail") or {}
    backlag = np.mean([d.get("vs_kick_ms", 0) for v, d in md.items() if v in ("snare", "clap", "rim")] or [0.0])
    voc = h.get("vocabulary") or {}
    b = a.get("bass") or {}
    e = np.asarray(st.get("energy_per_bar") or [0.0], dtype=float)
    e64 = np.interp(np.linspace(0, len(e) - 1, 64), np.arange(len(e)), e) if len(e) > 1 else np.zeros(64)
    stems = (s.get("stems") or {})
    mix = s.get("mix") or {}
    bands = mix.get("bands") or {}
    timbre = list(np.asarray(s.get("_timbre_vector") or [0.0] * 13)[1:] / 40.0)
    timbre += [math.log((mix.get("centroid_hz") or 1500) / 1500.0)]
    timbre += [bands.get(k, 0.0) * 3 for k in ("sub", "low", "low_mid", "high_mid", "high")]
    return {
        "bpm": r.get("bpm") or 0.0, "meter": meter, "swing": swing, "swing_level": lvl, "backbeat_lag_ms": float(backlag),
        "roles": _role_vectors(main, meter * 4), "density": float(sum((a.get("drums", {}).get("hit_counts") or {}).values()))
        / max(a.get("track", {}).get("duration_s") or 1, 1),
        "key": h.get("key"), "mode": (h.get("key") or " ").split(" ")[-1], "vocab": voc,
        "loop_roman": (h.get("loop") or {}).get("roman") or [], "extensions": set(h.get("extensions") or []),
        "chords_per_bar": (h.get("harmonic_rhythm") or {}).get("chords_per_bar") or 0.0,
        "bass_rate": np.asarray(b.get("rhythm_hit_rate") or [0.0] * (meter * 4), dtype=float),
        "bass_glides": b.get("glides") or 0.0, "bass_oct": b.get("octave_jumps") or 0.0,
        "bass_lock": b.get("kick_lock") if b.get("kick_lock") is not None else 0.5,
        "timbre": np.asarray(timbre), "clap": np.asarray(s.get("_clap_embedding")) if s.get("_clap_embedding") else None,
        "energy": e64, "labels": [x.get("label") for x in st.get("sections", [])],
        "brightness": mix.get("brightness"), "centroid": mix.get("centroid_hz"),
        "sidechain": (s.get("sidechain") or {}).get("depth_db") if (s.get("sidechain") or {}).get("detected") else 0.0,
        "lufs": (a.get("track", {}).get("loudness") or {}).get("integrated_lufs"),
    }


# ------------------------------------------------------------------ similarities
def _cos(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or not a.size:
        return 0.0
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 and nb == 0:
        return 1.0
    if na == 0 or nb == 0:
        return 0.0
    return float(a @ b / (na * nb))


def _jac(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return 0.0
    u = np.maximum(a, b).sum()
    return float(np.minimum(a, b).sum() / u) if u else 1.0


def _edit_sim(a: list, b: list) -> float:
    if not a and not b:
        return 1.0
    n, m = len(a), len(b)
    d = np.zeros((n + 1, m + 1))
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (a[i - 1] != b[j - 1]))
    return float(1 - d[n, m] / max(n, m))


def tempo_ratio(a: float, b: float) -> tuple[float, float]:
    """Octave-aware log2 tempo distance and the octave factor applied to b."""
    best = (9.0, 1.0)
    for f in (1.0, 2.0, 0.5):
        d = abs(math.log2(max(a, 1e-6) / max(b * f, 1e-6)))
        if d < best[0] - 1e-9:
            best = (d, f)
    return best


def similarity(fa: dict, fb: dict) -> dict[str, float]:
    d, _ = tempo_ratio(fa["bpm"], fb["bpm"])
    tempo = math.exp(-((d / 0.06) ** 2)) * (1.0 if fa["meter"] == fb["meter"] else 0.5)
    groove = 0.6 * math.exp(-(((fa["swing"] - fb["swing"]) / 0.04) ** 2)) + \
        0.4 * math.exp(-(((fa["backbeat_lag_ms"] - fb["backbeat_lag_ms"]) / 10.0) ** 2))
    drums = sum(w * _jac(fa["roles"][r], fb["roles"][r]) for r, w in (("kick", 0.4), ("backbeat", 0.35), ("hats", 0.25)))
    dens = min(fa["density"], fb["density"]) / max(fa["density"], fb["density"], 1e-6)
    drums = 0.85 * drums + 0.15 * dens
    keys = sorted(set(fa["vocab"]) | set(fb["vocab"]))
    voc = _cos([fa["vocab"].get(k, 0) for k in keys], [fb["vocab"].get(k, 0) for k in keys]) if keys else 0.0
    ext_u = fa["extensions"] | fb["extensions"]
    ext = len(fa["extensions"] & fb["extensions"]) / len(ext_u) if ext_u else 1.0
    harmony = 0.2 * (fa["mode"] == fb["mode"]) + 0.35 * voc + 0.25 * _edit_sim(fa["loop_roman"], fb["loop_roman"]) + 0.2 * ext
    bass = 0.5 * max(0.0, _cos(fa["bass_rate"], fb["bass_rate"])) + 0.5 * (
        1 - (abs(fa["bass_glides"] - fb["bass_glides"]) + abs(fa["bass_oct"] - fb["bass_oct"]) + abs(fa["bass_lock"] - fb["bass_lock"])) / 3)
    if fa["clap"] is not None and fb["clap"] is not None:
        sound = (_cos(fa["clap"], fb["clap"]) + 1) / 2
        sound = float(np.clip((sound - 0.6) / 0.4, 0, 1))  # CLAP cosines cluster high; stretch
    else:
        sound = max(0.0, _cos(fa["timbre"], fb["timbre"]))
    corr = np.corrcoef(fa["energy"], fb["energy"])[0, 1] if np.std(fa["energy"]) > 0 and np.std(fb["energy"]) > 0 else 0.0
    structure = 0.7 * (corr + 1) / 2 + 0.3 * _edit_sim(fa["labels"], fb["labels"])
    out = {"tempo": tempo, "groove": groove, "drums": drums, "harmony": harmony, "bass": bass, "sound": sound,
           "structure": structure}
    out = {k: round(float(np.clip(v, 0, 1)), 3) for k, v in out.items()}
    out["overall"] = round(sum(out[k] * DIM_WEIGHTS[k] for k in DIMENSIONS) / sum(DIM_WEIGHTS.values()), 3)
    return out


def differences(na: str, fa: dict, nb: str, fb: dict) -> list[str]:
    out = []
    d, f = tempo_ratio(fa["bpm"], fb["bpm"])
    if f != 1.0:
        out.append(f"{nb} ({fb['bpm']:.0f} bpm) is a {'half' if f == 2.0 else 'double'}-time reading relative to {na} ({fa['bpm']:.0f} bpm)")
    elif abs(fa["bpm"] - fb["bpm"]) / max(fa["bpm"], 1) > 0.03:
        faster, slower = (na, nb) if fa["bpm"] > fb["bpm"] else (nb, na)
        out.append(f"{faster} is {abs(fa['bpm'] - fb['bpm']):.0f} bpm faster than {slower}")
    if abs(fa["swing"] - fb["swing"]) >= 0.03:
        more = na if fa["swing"] > fb["swing"] else nb
        out.append(f"{more} swings harder ({max(fa['swing'], fb['swing']) * 100:.0f}% vs {min(fa['swing'], fb['swing']) * 100:.0f}%)")
    if fa["meter"] != fb["meter"]:
        out.append(f"different meters: {na} {fa['meter']}/4 vs {nb} {fb['meter']}/4")
    for role in ("kick", "backbeat", "hats"):
        if _jac(fa["roles"][role], fb["roles"][role]) < 0.5 and (fa["roles"][role].any() or fb["roles"][role].any()):
            ga = "".join("x" if v else "." for v in fa["roles"][role])
            gb = "".join("x" if v else "." for v in fb["roles"][role])
            out.append(f"{role} patterns differ: {na} `{ga}` vs {nb} `{gb}`")
    if fa["key"] and fb["key"] and fa["key"] != fb["key"]:
        out.append(f"keys differ: {na} {fa['key']} vs {nb} {fb['key']}")
    rich = lambda v: sum(w for q, w in v.items() if q not in ("maj", "m", "", "5"))  # noqa: E731
    ra, rb = rich(fa["vocab"]), rich(fb["vocab"])
    if abs(ra - rb) > 0.3:
        more = na if ra > rb else nb
        out.append(f"{more} uses far more extended chords (7ths/9ths/13ths: {max(ra, rb) * 100:.0f}% vs {min(ra, rb) * 100:.0f}%)")
    if fa["centroid"] and fb["centroid"] and abs(math.log2(fa["centroid"] / fb["centroid"])) > 0.4:
        brighter = na if fa["centroid"] > fb["centroid"] else nb
        out.append(f"{brighter} is noticeably brighter ({max(fa['centroid'], fb['centroid']):.0f} vs {min(fa['centroid'], fb['centroid']):.0f} Hz centroid)")
    if abs((fa["sidechain"] or 0) - (fb["sidechain"] or 0)) > 2:
        more = na if (fa["sidechain"] or 0) > (fb["sidechain"] or 0) else nb
        out.append(f"{more} has a much stronger sidechain pump")
    if abs(fa["bass_glides"] - fb["bass_glides"]) > 0.25:
        out.append(f"{na if fa['bass_glides'] > fb['bass_glides'] else nb} has a gliding bass")
    return out


def compare(analyses: list[dict]) -> dict:
    feats = [features(a) for a in analyses]
    names = [_short(a) for a in analyses]
    pairs = []
    n = len(analyses)
    mats = {k: np.eye(n).tolist() for k in list(DIMENSIONS) + ["overall"]}
    diffs = []
    for i in range(n):
        for j in range(i + 1, n):
            sim = similarity(feats[i], feats[j])
            pairs.append({"a": analyses[i]["track"]["id"], "b": analyses[j]["track"]["id"], "names": [names[i], names[j]],
                          **sim})
            for k, v in sim.items():
                mats[k][i][j] = mats[k][j][i] = v
            diffs.extend(f"[{names[i]} vs {names[j]}] {d}" for d in differences(names[i], feats[i], names[j], feats[j]))
    tracks = [{"id": a["track"]["id"], "name": names[k], "bpm": a["rhythm"].get("bpm"), "meter": a["rhythm"].get("meter"),
               "key": a["harmony"].get("key"), "swing": feats[k]["swing"], "loop": (a["harmony"].get("loop") or {}).get("symbols")}
              for k, a in enumerate(analyses)]
    return {"tracks": tracks, "pairwise": pairs, "matrix": mats, "differences": diffs,
            "dimensions": list(DIMENSIONS), "note": "similarities are 0..1 per dimension; overall is a weighted mean"}


def _short(a: dict) -> str:
    return a["track"].get("name") or a["track"]["id"][:8]


# ------------------------------------------------------------------ blend
def blend(analyses: list[dict], weights: list[float] | None = None, overrides: dict[str, str] | None = None) -> dict:
    """Target spec mixing references: weighted averages for numbers, exemplars for categorical parts."""
    n = len(analyses)
    if n == 0:
        raise ValueError("need at least one analysis")
    w = np.asarray(weights if weights else [1.0] * n, dtype=float)
    if len(w) != n or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("weights must be non-negative, one per track, not all zero")
    w = w / w.sum()
    ids = [a["track"]["id"] for a in analyses]
    names = [_short(a) for a in analyses]
    overrides = dict(overrides or {})
    for k in overrides:
        if k not in DIMENSIONS:
            raise ValueError(f"unknown dimension '{k}' (use {', '.join(DIMENSIONS)})")

    def resolve(ref: str) -> int:
        ref = str(ref)
        for i, (tid, nm) in enumerate(zip(ids, names)):
            if ref == tid or tid.startswith(ref) or ref == nm or ref == str(i):
                return i
        raise ValueError(f"override refers to unknown track '{ref}'")

    src = {k: resolve(v) for k, v in overrides.items()}
    heaviest = int(np.argmax(w))
    g = src.get("groove", heaviest)
    for dim, default in (("tempo", g), ("drums", g), ("bass", g), ("harmony", heaviest), ("sound", heaviest),
                         ("structure", heaviest)):
        src.setdefault(dim, default)
    src.setdefault("groove", g)
    A = analyses
    rationale = []

    # tempo: weighted geometric mean in the tempo source's octave, unless tempo is overridden
    base_bpm = A[src["tempo"]]["rhythm"]["bpm"] or 120.0
    if "tempo" in overrides:
        bpm = base_bpm
        rationale.append(f"tempo {bpm:.1f} bpm taken from {names[src['tempo']]}")
    else:
        logs = []
        for a in A:
            b = a["rhythm"].get("bpm") or base_bpm
            _, f = tempo_ratio(base_bpm, b)
            logs.append(math.log(b * f))
        bpm = float(math.exp(np.dot(w, logs)))
        parts = ", ".join("%s %.1f" % (nm, a["rhythm"]["bpm"] or 0) for nm, a in zip(names, A))
        rationale.append(f"tempo {bpm:.1f} bpm = weighted (octave-aligned) mean of {parts}")
    meter = A[src["groove"]]["rhythm"].get("beats_per_bar") or 4

    gs = A[src["groove"]]["rhythm"]
    sw_vals = [_swing(a)[0] for a in A]
    swing_ratio = float(np.dot(w, sw_vals)) if "groove" not in overrides else _swing(A[src["groove"]])[0]
    level = _swing(A[src["groove"]])[1] or "16th"
    rationale.append(f"swing {swing_ratio * 100:.0f}% on {level}s" + (f" from {names[src['groove']]}" if "groove" in overrides else " (weighted)"))

    da = A[src["drums"]]["drums"]
    main = _main_pattern(A[src["drums"]])
    pats = [p for p in da.get("patterns", []) if p.get("count", 0) >= 2][:4]
    rationale.append(f"drum patterns from {names[src['drums']]} (pattern {main['id'] if main else '?'})")

    ha = A[src["harmony"]]["harmony"]
    rationale.append(f"harmony from {names[src['harmony']]}: {ha.get('key')}, loop {' – '.join((ha.get('loop') or {}).get('symbols') or [])}")
    ba = A[src["bass"]].get("bass") or {}
    so = A[src["sound"]]["sound"]
    side = [((a["sound"].get("sidechain") or {}).get("depth_db") or 0.0) if (a["sound"].get("sidechain") or {}).get("detected") else 0.0 for a in A]
    rel = [((a["sound"].get("sidechain") or {}).get("release_ms") or 0.0) for a in A]
    side_depth = float(np.dot(w, side)) if "sound" not in overrides else side[src["sound"]]
    rationale.append(f"sound palette from {names[src['sound']]}" + (f"; sidechain {side_depth:.1f} dB" if side_depth > 0.5 else ""))
    st = A[src["structure"]]["structure"]
    rationale.append(f"arrangement/energy template from {names[src['structure']]}")

    # energy template by section label (mean over all references, weighted)
    template: dict[str, list] = {}
    for wi, a in zip(w, A):
        for s in a["structure"].get("sections", []):
            template.setdefault(s.get("label"), []).append((wi, s["energy"], s.get("layers") or [], s.get("bars")))
    energy_by_label = {k: round(sum(x[0] * x[1] for x in v) / max(sum(x[0] for x in v), 1e-9), 3) for k, v in template.items()}
    layers_by_label = {s.get("label"): s.get("layers") for s in st.get("sections", [])}
    builds = [e for e in st.get("events", []) if e["type"] == "build"]
    fills = A[src["drums"]]["drums"].get("fills") or []
    n_bars = len(A[src["drums"]]["drums"].get("bar_patterns") or []) or 1
    spec = {
        "kind": "audioref.target_spec",
        "sources": {k: {"track_id": ids[v], "name": names[v]} for k, v in src.items()},
        "weights": {names[i]: round(float(w[i]), 3) for i in range(n)},
        "tempo": {"bpm": round(bpm, 2), "meter": f"{meter}/4", "beats_per_bar": meter},
        "groove": {"swing_ratio": round(swing_ratio, 3), "swing_level": level, "swing_percent": round(swing_ratio * 100, 1),
                   "microtiming_ms": gs.get("microtiming_ms"), "groove_template_16": gs.get("groove_template_16")},
        "drums": {"main_pattern": main, "patterns": pats, "fills": [{k: f.get(k) for k in ("kind", "beat", "voices", "notes")} for f in fills[:6]],
                  "fills_per_8_bars": round(8 * len(fills) / n_bars, 2), "sound": da.get("sound")},
        "harmony": {"key": ha.get("key"), "loop": ha.get("loop"), "vocabulary": ha.get("vocabulary"),
                    "extensions": ha.get("extensions"), "voicing": ha.get("voicing"), "harmonic_rhythm": ha.get("harmonic_rhythm")},
        "bass": {k: ba.get(k) for k in ("rhythm_grid_16", "rhythm_hit_rate", "range_midi", "range_names", "glides", "octave_jumps",
                                        "relation_to_kick", "articulation", "chord_tone_use")} if ba.get("present") else None,
        "sound": {"synthesis_hints": so.get("synthesis_hints"), "tags": so.get("tags"), "mix": so.get("mix"),
                  "sidechain": {"depth_db": round(side_depth, 1),
                                "release_ms": round(float(np.dot(w, rel)), 0) if side_depth > 0.5 else None}},
        "structure": {"sections": [{k: s.get(k) for k in ("label", "bars", "energy", "layers")} for s in st.get("sections", [])],
                      "energy_by_label": energy_by_label, "layers_by_label": layers_by_label,
                      "builds": [{"bars": e["end_bar"] - e["start_bar"], "technique": e.get("technique")} for e in builds],
                      "stops_before_drops": any("stop before the drop" in (e.get("technique") or []) for e in builds),
                      "energy_curve": _resample(st.get("energy_per_bar") or [], 32)},
        "rationale": rationale,
    }
    spec["brief_md"] = blend_brief(spec)
    return spec


def _resample(x: list, n: int) -> list[float]:
    if not x:
        return []
    a = np.asarray(x, dtype=float)
    return [round(float(v), 3) for v in np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a)]


def spec_from_analysis(a: dict) -> dict:
    """A single reference as a target spec (used by map_to_timeline with a track id)."""
    return blend([a])


def blend_brief(spec: dict) -> str:
    t, g, d, h, s, st = spec["tempo"], spec["groove"], spec["drums"], spec["harmony"], spec["sound"], spec["structure"]
    lines = ["# Blend target", "", "Sources: " + ", ".join(f"{k} ← {v['name']}" for k, v in spec["sources"].items()), "",
             f"- **Tempo** {t['bpm']} bpm, {t['meter']}; swing {g['swing_percent']}% on {g['swing_level']}s."]
    mp = d.get("main_pattern")
    if mp:
        lines.append(f"- **Drums** pattern {mp['id']} (`X` accent, `x` hit, `o` ghost):")
        lines.append("```\n" + "\n".join(f"{v.ljust(11)} {p}" for v, p in mp.get("accents", {}).items()) + "\n```")
    loop = h.get("loop") or {}
    if loop.get("symbols"):
        lines.append(f"- **Harmony** {h.get('key')}: `{' – '.join(loop['symbols'])}` = `{' – '.join(loop.get('roman') or [])}`"
                     + (f", {(h.get('voicing') or {}).get('style')}" if (h.get("voicing") or {}).get("present") else ""))
    b = spec.get("bass")
    if b:
        lines.append(f"- **Bass** `{b.get('rhythm_grid_16')}`, {b.get('relation_to_kick')}, glides {int(100 * (b.get('glides') or 0))}%")
    hints = s.get("synthesis_hints") or {}
    if hints:
        lines.append("- **Sound** " + " | ".join(v for k, v in hints.items() if k != "mix"))
    if (s.get("sidechain") or {}).get("depth_db", 0) > 0.5:
        lines.append(f"- **Mix** sidechain ~{s['sidechain']['depth_db']} dB, ~{s['sidechain'].get('release_ms')} ms release")
    if st.get("sections"):
        lines.append("- **Arrangement template** " + " → ".join(f"{x['label']} {x['bars']}" for x in st["sections"]))
    lines += ["", "Rationale:"] + [f"- {r}" for r in spec.get("rationale", [])]
    return "\n".join(lines) + "\n"


__all__ = ["compare", "blend", "features", "similarity", "spec_from_analysis", "parse_chord", "parse_key"]
