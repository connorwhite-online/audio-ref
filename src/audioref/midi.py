"""MIDI export: drums, bass, chords (from symbols), keys (transcribed), melody, full.

Notes keep their absolute audio times (so the MIDI lines up with the reference), and the file tempo is set to
the detected BPM; `analysis.rhythm.first_downbeat` tells where bar 1 starts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pretty_midi

from .synth import VOICE_TO_GM
from .theory import parse_chord

PROGRAMS = {"bass": 38, "chords": 4, "keys": 4, "melody": 80}


def _inst(name: str, program: int = 0, is_drum: bool = False) -> pretty_midi.Instrument:
    return pretty_midi.Instrument(program=program, is_drum=is_drum, name=name)


def drums_instrument(hits: dict[str, list]) -> pretty_midi.Instrument:
    inst = _inst("drums", 0, True)
    for voice, hs in hits.items():
        note = VOICE_TO_GM.get(voice, 39)
        for t, vel in hs:
            inst.notes.append(pretty_midi.Note(velocity=int(np.clip(round(vel * 127), 1, 127)), pitch=note,
                                               start=float(t), end=float(t) + 0.06))
    inst.notes.sort(key=lambda n: n.start)
    return inst


def notes_instrument(name: str, notes: list, program: int) -> pretty_midi.Instrument:
    inst = _inst(name, program)
    amps = np.array([n[3] for n in notes]) if notes else np.array([1.0])
    ref = float(np.percentile(amps, 95)) or 1.0
    for n in notes:
        t0, t1, pitch, amp = float(n[0]), float(n[1]), int(n[2]), float(n[3])
        if t1 <= t0:
            t1 = t0 + 0.05
        vel = int(np.clip(round(40 + 87 * min(amp / ref, 1.0)), 1, 127))
        inst.notes.append(pretty_midi.Note(velocity=vel, pitch=int(np.clip(pitch, 0, 127)), start=t0, end=t1))
    return inst


def chords_instrument(chords: list[dict]) -> pretty_midi.Instrument:
    inst = _inst("chords", PROGRAMS["chords"])
    for c in chords:
        if c.get("symbol") in (None, "N"):
            continue
        ch = parse_chord(c["symbol"])
        voicing = ch.voicing(center=62)
        if ch.bass is not None:
            voicing = [48 + ((ch.bass - 48) % 12)] + voicing
        for p in voicing:
            inst.notes.append(pretty_midi.Note(velocity=80, pitch=int(p), start=float(c["start"]), end=float(c["end"]) - 0.01))
    return inst


def _pm(bpm: float | None, meter: int) -> pretty_midi.PrettyMIDI:
    pm = pretty_midi.PrettyMIDI(initial_tempo=float(bpm) if bpm else 120.0)
    pm.time_signature_changes.append(pretty_midi.TimeSignature(meter, 4, 0.0))
    return pm


def export_all(folder: Path, analysis: dict, hits: dict, notes: dict) -> dict[str, str]:
    folder.mkdir(parents=True, exist_ok=True)
    bpm = analysis["rhythm"].get("bpm")
    meter = analysis["rhythm"].get("beats_per_bar") or 4
    parts = {}
    if hits:
        parts["drums"] = drums_instrument(hits)
    if notes.get("bass"):
        parts["bass"] = notes_instrument("bass", notes["bass"], PROGRAMS["bass"])
    if analysis["harmony"].get("chords"):
        parts["chords"] = chords_instrument(analysis["harmony"]["chords"])
    if notes.get("keys"):
        parts["keys"] = notes_instrument("keys", notes["keys"], PROGRAMS["keys"])
    if notes.get("melody"):
        parts["melody"] = notes_instrument("melody", notes["melody"], PROGRAMS["melody"])
    out = {}
    for name, inst in parts.items():
        pm = _pm(bpm, meter)
        pm.instruments.append(inst)
        p = folder / f"{name}.mid"
        pm.write(str(p))
        out[name] = str(p)
    if parts:
        pm = _pm(bpm, meter)
        for name, inst in parts.items():
            if name == "keys" and "chords" in parts:
                continue  # full.mid keeps the clean chord track; keys.mid has the raw transcription
            pm.instruments.append(inst)
        p = folder / "full.mid"
        pm.write(str(p))
        out["full"] = str(p)
    return out


def summarize(path: str | Path) -> dict:
    """Note summary of a MIDI file: counts, ranges, density."""
    pm = pretty_midi.PrettyMIDI(str(path))
    out = {"path": str(path), "duration_s": round(pm.get_end_time(), 2), "tempo_bpm": None, "instruments": []}
    _, tempi = pm.get_tempo_changes()
    if len(tempi):
        out["tempo_bpm"] = round(float(tempi[0]), 2)
    for inst in pm.instruments:
        ns = inst.notes
        if not ns:
            continue
        pitches = [n.pitch for n in ns]
        dur = max(n.end for n in ns) - min(n.start for n in ns)
        d = {"name": inst.name, "is_drum": inst.is_drum, "program": inst.program, "notes": len(ns),
             "notes_per_second": round(len(ns) / max(dur, 1e-3), 2),
             "velocity_mean": round(float(np.mean([n.velocity for n in ns])), 1)}
        if inst.is_drum:
            counts: dict[int, int] = {}
            for p in pitches:
                counts[p] = counts.get(p, 0) + 1
            d["drum_notes"] = {pretty_midi.note_number_to_drum_name(k) or str(k): v for k, v in sorted(counts.items())}
        else:
            d["range"] = [pretty_midi.note_number_to_name(min(pitches)), pretty_midi.note_number_to_name(max(pitches))]
            d["mean_duration_s"] = round(float(np.mean([n.end - n.start for n in ns])), 3)
        out["instruments"].append(d)
    return out
