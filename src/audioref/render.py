"""Quick preview renders (sanity checks, and for closing the loop by re-analysing your own output).

- render_midi: any MIDI file (GM drums on channel 10, other programs mapped to simple synth voices)
- render_plan: a score plan (from map_to_timeline) + target spec -> a rough arrangement
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pretty_midi

from . import synth
from .ingest import write_wav
from .theory import parse_chord

SR = synth.SR


def _voice_for_program(program: int, name: str) -> str:
    n = (name or "").lower()
    if "bass" in n or 32 <= program <= 39:
        return "bass"
    if "melody" in n or "lead" in n or 80 <= program <= 87:
        return "lead"
    if "pad" in n or 88 <= program <= 95:
        return "pad"
    return "epiano"


def render_midi(path: str | Path, out: str | Path, tail_s: float = 1.5) -> dict:
    pm = pretty_midi.PrettyMIDI(str(path))
    dur = pm.get_end_time() + tail_s
    mx = synth.Mixer(dur)
    counts: dict[str, int] = {}
    for inst in pm.instruments:
        if inst.is_drum:
            for n in inst.notes:
                v = synth.GM_DRUMS.get(n.pitch, "perc")
                mx.add("drums", synth.DRUMS[v](vel=n.velocity / 127.0), n.start,
                       {"kick": 1.0, "snare": 0.7, "clap": 0.6, "hat_closed": 0.3, "hat_open": 0.28}.get(v, 0.4))
                counts["drums"] = counts.get("drums", 0) + 1
            continue
        voice = _voice_for_program(inst.program, inst.name)
        bus = "bass" if voice == "bass" else "other"
        for n in inst.notes:
            d = max(0.03, n.end - n.start)
            v = n.velocity / 127.0
            if voice == "bass":
                sig = synth.bass(n.pitch, d, v)
            elif voice == "lead":
                sig = synth.lead(n.pitch, d, v * 0.7)
            elif voice == "pad":
                sig = synth.pad(n.pitch, d, v * 0.5)
            else:
                sig = synth.epiano(n.pitch, d, v * 0.7)
            mx.add(bus, sig, n.start, 0.5)
            counts[voice] = counts.get(voice, 0) + 1
    mix, _ = mx.mix()
    write_wav(Path(out), mix, SR)
    return {"path": str(out), "duration_s": round(dur, 2), "notes_rendered": counts}


VEL = {".": 0.0, "o": 0.38, "x": 0.75, "X": 1.0}
DRUM_GAIN = {"kick": 1.0, "snare": 0.7, "clap": 0.6, "rim": 0.4, "hat_closed": 0.3, "hat_open": 0.28, "perc": 0.22,
             "ride": 0.25, "tom": 0.6, "crash": 0.35}


def render_plan(plan: dict, spec: dict, out: str | Path) -> dict:
    """Very rough arrangement of a score plan using the spec's groove, chords and bass grid."""
    bpm = plan["bpm"]
    meter = plan["beats_per_bar"]
    beat = 60.0 / bpm
    bar = beat * meter
    off = plan["offset_s"]
    dur = plan["duration_s"] + 1.5
    mx = synth.Mixer(dur)
    swing = (spec.get("groove") or {}).get("swing_ratio") or 0.5
    swung = set(((spec.get("groove") or {}).get("groove_template_16") or {}).keys()) or {"hat_closed", "perc"}
    main = (spec.get("drums") or {}).get("main_pattern") or {}
    grids = main.get("accents") or main.get("grid_16") or {"kick": "X...X...X...X...", "clap": "....x.......x...",
                                                               "hat_closed": "..x...x...x...x."}
    loop = ((spec.get("harmony") or {}).get("loop") or {}).get("symbols") or ["C", "Am", "F", "G"]
    bpc = ((spec.get("harmony") or {}).get("loop") or {}).get("bars_per_chord") or 1.0
    bass_grid = ((spec.get("bass") or {}) or {}).get("rhythm_grid_16") or "x.......x......."
    steps = meter * 4

    def step_t(bar_start: float, s: int, sw: float) -> float:
        b, sub = divmod(s, 4)
        frac = (sub // 2) * 0.5 + (sw * 0.5 if sub % 2 else 0.0)
        return bar_start + (b + frac) * beat

    stop_spans = [(e["time"], e.get("end", e["time"] + beat)) for e in plan["events"] if e["type"] == "stop"]
    fills = {(e["bar"]) for e in plan["events"] if e["type"] == "fill"}
    bar_idx = 0
    for b in plan["bar_map"]:
        bs = off + (b["bar"] - 1) * bar
        if bs + bar <= 0:
            continue
        sec = next((s for s in plan["sections"] if s["start_s"] - 1e-6 <= max(bs, 0) < s["end_s"] - 1e-6), None)
        if sec is None:
            continue
        lab = sec["label"]
        layers = set(sec.get("layers") or [])
        drums_on = lab in ("groove", "drop", "climax") or bool(layers & {"kick", "snare/clap", "hats"})
        if lab in ("intro", "breakdown", "stop", "outro") and not (layers & {"kick"}):
            drums_on = False
        if drums_on:
            for v, g in grids.items():
                if v == "crash":
                    continue
                for s_, ch in enumerate(g[:steps]):
                    if VEL.get(ch, 0) <= 0:
                        continue
                    if b["bar"] in fills and s_ >= steps - 4 and v in ("snare", "clap", "hat_closed", "hat_open"):
                        continue
                    t = step_t(bs, s_, swing if v in swung else 0.5)
                    mx.add("drums", synth.DRUMS.get(v, synth.DRUMS["perc"])(vel=VEL[ch]), t, DRUM_GAIN.get(v, 0.4))
        if b["bar"] in fills and lab not in ("stop",):
            for k in range(4):
                mx.add("drums", synth.snare(0.5 + 0.12 * k), bs + (meter - 1 + k / 4) * beat, 0.6)
        if lab == "build":
            pos = (bs - sec["start_s"]) / max(sec["end_s"] - sec["start_s"], 1e-6)
            per_beat = [1, 2, 4, 8][min(3, int(4 * pos))]
            for bt in range(meter):
                for k in range(per_beat):
                    mx.add("drums", synth.snare(0.35 + 0.6 * pos), bs + (bt + k / per_beat) * beat, 0.6)
        if any(e["type"] in ("drop", "climax") and e.get("bar") == b["bar"] for e in plan["events"]):
            mx.add("drums", synth.crash(0.9), bs, 0.35)
        # harmony
        if lab != "stop":
            ci = int(bar_idx / max(bpc, 1e-6)) % len(loop)
            ch = parse_chord(loop[ci])
            if ch is not None:
                for m in ch.voicing(center=62):
                    if lab in ("intro", "breakdown", "build", "outro"):
                        mx.add("other", synth.pad(m, bar, 0.3), bs, 0.45)
                    else:
                        for s_ in (3, 6, 11, 14):
                            mx.add("other", synth.epiano(m, beat * 0.4, 0.5), step_t(bs, s_, swing), 0.4)
                if lab in ("groove", "drop", "climax", "build"):
                    root = 36 + ch.root - (12 if ch.root >= 8 else 0)
                    hits = [s_ for s_, c in enumerate(bass_grid[:steps]) if c != "."]
                    for i, s_ in enumerate(hits):
                        nxt = hits[i + 1] if i + 1 < len(hits) else steps
                        mx.add("bass", synth.bass(root, max(0.08, (nxt - s_) * beat / 4 * 0.85), 0.9), step_t(bs, s_, 0.5), 0.55)
        if lab == "build":
            pass
        bar_idx += 1
    for sec in plan["sections"]:
        if sec["label"] == "build":
            mx.add("other", synth.riser(sec["end_s"] - sec["start_s"]), sec["start_s"], 0.3)
            mx.filter_sweep("other", sec["start_s"], sec["end_s"], 400.0, 12000.0)
    for a, b_ in stop_spans:
        mx.gate(a, b_)
    mix, _ = mx.mix()
    write_wav(Path(out), mix, SR)
    return {"path": str(out), "duration_s": round(dur, 2), "bpm": bpm, "sections": [s["label"] for s in plan["sections"]]}
