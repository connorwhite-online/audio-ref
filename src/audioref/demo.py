"""Ground-truth demo songs rendered with `audioref.synth`.

Each song comes with an exact truth record (beats, downbeats, swing, drum hits, chords, bass notes,
sections, events), which the test-suite and `scripts/validate.py` use to measure accuracy.
`audioref demo <name> out.wav` writes one to disk so humans can try the tool without a reference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import synth
from .theory import Key, parse_chord, parse_key

VEL = {".": 0.0, "o": 0.38, "x": 0.75, "X": 1.0}
HARMONIC_LAYERS = {"keys", "pad", "bass", "lead"}
STOP_EDGE = 0.008  # hits jittered this close to the end of a stop belong to the next downbeat


@dataclass
class Section:
    name: str
    bars: int
    layers: tuple[str, ...]
    pattern: str = "main"
    fill: str | None = None  # snare_roll | hat_ratchet | tom_fill
    build: bool = False  # accelerating snare roll + filter sweep + riser, no kick
    stop_beats: int = 0  # silence everything for the last N beats
    chords: list[str] | None = None  # overrides the song loop
    crash: bool = False  # crash on first downbeat
    climax: bool = False


@dataclass
class SongSpec:
    name: str
    bpm: float
    key: str
    chords: list[str]
    drum_patterns: dict[str, dict[str, str]]
    sections: list[Section]
    meter: int = 4
    bars_per_chord: float = 1.0
    swing: float = 0.5
    swing_voices: tuple[str, ...] = ("hat_closed", "hat_open", "perc", "snare", "clap", "rim", "bass", "keys")
    microtiming_ms: dict[str, float] = field(default_factory=dict)
    jitter_ms: float = 1.5
    bass_pattern: str = "x..............."
    bass_octave: int = 2
    bass_glide: bool = False
    bass_cutoff: float = 420.0
    keys_pattern: str | None = None  # comping grid; None = sustained chords
    keys_voice: str = "epiano"
    sidechain_db: float = 0.0
    sidechain_release_ms: float = 100.0
    lead_motif: list[tuple[int, int, int]] | None = None  # (step, semitones above tonic, length in steps)
    start_offset: float = 0.3
    tempo_drift_bpm: float = 0.0  # linear tempo change across the song
    tail_s: float = 1.5
    seed: int = 7

    @property
    def steps_per_bar(self) -> int:
        return self.meter * 4

    @property
    def total_bars(self) -> int:
        return sum(s.bars for s in self.sections)


# ----------------------------------------------------------------------------- songs
SONGS: dict[str, SongSpec] = {}

SONGS["house"] = SongSpec(
    name="house", bpm=112.0, key="F major", chords=["Bbmaj9", "Am9", "Gm9", "C13"],
    swing=0.58, swing_voices=("hat_closed", "perc", "keys"),
    microtiming_ms={"kick": 0.0, "clap": 10.0, "hat_closed": 5.0, "perc": 5.0},
    drum_patterns={
        "main": {"kick": "X...X...X...X...", "clap": "....x.......x...", "hat_closed": ".oXo.oXo.oXo.oXo",
                 "perc": "x.o.x.o.x.o.x.o."},
        "drop": {"kick": "X...X...X...X...", "clap": "....X.......X...", "hat_closed": ".o.o.o.o.o.o.o.o",
                 "hat_open": "..x...x...x...x.", "perc": "x.o.x.o.x.o.x.o."},
        "outro": {"kick": "X...X...X...X...", "hat_closed": ".oXo.oXo.oXo.oXo"},
    },
    bass_pattern="x..o..x...x...o.", bass_glide=True, keys_pattern="...x..x....x..x.",
    sidechain_db=4.5, sidechain_release_ms=110.0,
    lead_motif=[(0, 12, 3), (3, 14, 3), (6, 16, 2), (10, 19, 4)],
    sections=[
        Section("intro", 4, ("keys", "pad")),
        Section("groove", 8, ("kick", "clap", "hat_closed", "perc", "bass", "keys"), fill="hat_ratchet"),
        Section("build", 4, ("keys", "pad", "bass"), build=True, stop_beats=1),
        Section("drop", 8, ("kick", "clap", "hat_closed", "hat_open", "perc", "bass", "keys"), pattern="drop",
                crash=True, fill="snare_roll"),
        Section("breakdown", 4, ("pad", "keys")),
        Section("drop", 8, ("kick", "clap", "hat_closed", "hat_open", "perc", "bass", "keys", "lead", "pad"),
                pattern="drop", crash=True, climax=True),
        Section("outro", 4, ("kick", "hat_closed", "pad"), pattern="outro"),
    ],
)

SONGS["hiphop"] = SongSpec(
    name="hiphop", bpm=88.0, key="A minor", chords=["Am9", "Dm9", "Fmaj7", "E7"],
    swing=0.62, swing_voices=("hat_closed", "snare", "kick", "bass", "keys", "rim"),
    microtiming_ms={"kick": -2.0, "snare": 15.0, "hat_closed": 6.0}, jitter_ms=3.0,
    drum_patterns={
        "main": {"kick": "X.....x...X..x..", "snare": "....X.......X...", "hat_closed": "x.xox.x.x.xox.x.",
                 "rim": "...........o...."},
    },
    bass_pattern="x.....x...o..x..", keys_pattern="x.......x..x....", keys_voice="epiano",
    lead_motif=[(0, 7, 2), (2, 10, 2), (4, 12, 4), (10, 10, 2), (12, 7, 4)],
    sections=[
        Section("intro", 2, ("keys",)),
        Section("verse", 8, ("kick", "snare", "hat_closed", "rim", "bass", "keys"), fill="tom_fill"),
        Section("hook", 8, ("kick", "snare", "hat_closed", "rim", "bass", "keys", "lead"), climax=True),
        Section("verse", 4, ("kick", "snare", "hat_closed", "rim", "bass", "keys")),
        Section("outro", 2, ("keys",)),
    ],
)

SONGS["waltz"] = SongSpec(
    name="waltz", bpm=150.0, key="C major", chords=["C", "F", "G7", "C"], meter=3,
    drum_patterns={"main": {"kick": "X...........", "rim": "....x...x...", "hat_closed": "x.x.x.x.x.x."}},
    bass_pattern="x...........", keys_pattern="....x...x...", keys_voice="epiano",
    sections=[
        Section("A", 8, ("kick", "rim", "hat_closed", "bass", "keys", "pad")),
        Section("B", 8, ("kick", "rim", "hat_closed", "bass", "keys", "pad"), chords=["Am", "Dm", "G7", "C"]),
        Section("A", 8, ("kick", "rim", "hat_closed", "bass", "keys", "pad")),
    ],
)

SONGS["techno"] = SongSpec(
    name="techno", bpm=128.0, key="A minor", chords=["Am", "F", "C", "G"], bars_per_chord=2,
    drum_patterns={
        "main": {"kick": "X...X...X...X...", "hat_open": "..x...x...x...x.", "clap": "....x.......x..."},
        "intro": {"kick": "X...X...X...X...", "hat_open": "..x...x...x...x."},
    },
    bass_pattern=".xxx.xxx.xxx.xxx", bass_cutoff=600.0, keys_voice="pad", sidechain_db=6.0, sidechain_release_ms=150,
    sections=[
        Section("intro", 4, ("kick", "hat_open"), pattern="intro"),
        Section("groove", 8, ("kick", "hat_open", "clap", "bass", "pad"), fill="snare_roll"),
        Section("break", 4, ("pad",)),
        Section("build", 4, ("pad", "bass"), build=True, stop_beats=2),
        Section("drop", 8, ("kick", "hat_open", "clap", "bass", "pad"), crash=True, climax=True),
        Section("outro", 4, ("kick", "hat_open"), pattern="intro"),
    ],
)

SONGS["live"] = SongSpec(
    name="live", bpm=96.0, key="G major", chords=["G", "D", "Em", "C"], tempo_drift_bpm=4.0, jitter_ms=7.0,
    drum_patterns={"main": {"kick": "X.....x.X.......", "snare": "....X.......X...", "hat_closed": "x.x.x.x.x.x.x.x."}},
    bass_pattern="x.......x.x.....", keys_pattern="x...x...x...x...", keys_voice="epiano",
    sections=[Section("verse", 8, ("kick", "snare", "hat_closed", "bass", "keys")),
              Section("chorus", 8, ("kick", "snare", "hat_closed", "bass", "keys", "pad"), climax=True)],
)


# --------------------------------------------------------------------------- render
class _Clock:
    def __init__(self, spec: SongSpec):
        self.spec = spec
        n_beats = spec.total_bars * spec.meter + 1
        bpms = spec.bpm + spec.tempo_drift_bpm * np.arange(n_beats) / max(n_beats - 1, 1)
        ibis = 60.0 / bpms
        self.beats = spec.start_offset + np.concatenate([[0.0], np.cumsum(ibis[:-1])])
        self.ibis = ibis

    def beat(self, b: float) -> float:
        i = int(math.floor(b))
        i = min(max(i, 0), len(self.beats) - 2)
        return float(self.beats[i] + (b - i) * self.ibis[i])

    def step_time(self, bar: int, step: int, swing: float) -> float:
        """Time of 16th-step `step` in `bar`, with 16th swing ratio `swing`."""
        beat = bar * self.spec.meter + step // 4
        sub = step % 4
        frac = (sub // 2) * 0.5 + (swing * 0.5 if sub % 2 else 0.0)
        return self.beat(beat + frac)


def render_song(spec: SongSpec | str) -> tuple[np.ndarray, dict[str, np.ndarray], dict]:
    """Render a song. Returns (stereo mix (2,n) float32, stems dict, truth dict)."""
    if isinstance(spec, str):
        spec = SONGS[spec]
    rng = np.random.default_rng(spec.seed)
    clock = _Clock(spec)
    key = parse_key(spec.key)
    spb = spec.steps_per_bar
    duration = clock.beat(spec.total_bars * spec.meter) + spec.tail_s
    mx = synth.Mixer(duration)

    truth: dict = {
        "name": spec.name, "bpm": spec.bpm, "meter": f"{spec.meter}/4", "key": key.name, "swing": spec.swing,
        "swing_voices": list(spec.swing_voices), "microtiming_ms": dict(spec.microtiming_ms),
        "beats": [], "downbeats": [], "drums": {}, "chords": [], "bass": [], "lead": [], "sections": [],
        "events": [], "fills": [], "stops": [], "sidechain": {"depth_db": spec.sidechain_db,
                                                             "release_ms": spec.sidechain_release_ms},
        "duration": duration, "tempo_drift_bpm": spec.tempo_drift_bpm,
    }
    n_beats_total = spec.total_bars * spec.meter
    truth["beats"] = [clock.beat(b) for b in range(n_beats_total)]
    truth["downbeats"] = [clock.beat(b * spec.meter) for b in range(spec.total_bars)]

    def jitter(voice: str) -> float:
        return (spec.microtiming_ms.get(voice, 0.0) + rng.normal(0, spec.jitter_ms)) / 1000.0

    def hit(voice: str, t: float, vel: float, pan: float = 0.0, gain: float = 1.0) -> None:
        if vel <= 0:
            return
        vel = float(np.clip(vel * (1 + rng.normal(0, 0.04)), 0.05, 1.0))
        t = t + jitter(voice)
        fn = synth.DRUMS[voice]
        g = {"kick": 1.0, "snare": 0.7, "clap": 0.6, "hat_closed": 0.32, "hat_open": 0.28, "rim": 0.4,
             "perc": 0.22, "tom": 0.7, "crash": 0.35, "ride": 0.25}[voice] * gain
        p = {"hat_closed": 0.25, "hat_open": -0.2, "perc": -0.35, "rim": 0.15, "tom": 0.2}.get(voice, pan)
        mx.add("drums", fn(vel=vel), t, g, p)
        truth["drums"].setdefault(voice, []).append((round(t, 5), round(vel, 3)))

    kick_times: list[float] = []
    bar0 = 0
    stop_spans: list[tuple[float, float]] = []
    for si, sec in enumerate(spec.sections):
        sec_start_t = clock.beat(bar0 * spec.meter)
        truth["sections"].append({"label": sec.name, "start_bar": bar0, "end_bar": bar0 + sec.bars,
                                  "start": sec_start_t, "end": clock.beat((bar0 + sec.bars) * spec.meter),
                                  "layers": list(sec.layers)})
        pattern = spec.drum_patterns.get(sec.pattern, spec.drum_patterns["main"])
        loop = sec.chords or spec.chords
        harmonic = bool(HARMONIC_LAYERS & set(sec.layers))
        for b in range(sec.bars):
            bar = bar0 + b
            last_bar = b == sec.bars - 1
            fill_region = (spec.meter - 1) * 4 if (last_bar and sec.fill) else spb
            # ---- drums
            for voice, steps in pattern.items():
                if voice not in sec.layers or sec.build:
                    continue
                for s, ch in enumerate(steps[:spb]):
                    if VEL[ch] == 0:
                        continue
                    if s >= fill_region and voice in ("snare", "clap", "hat_closed", "hat_open", "rim"):
                        continue
                    sw = spec.swing if voice in spec.swing_voices else 0.5
                    t = clock.step_time(bar, s, sw)
                    hit(voice, t, VEL[ch])
                    if voice == "kick":
                        kick_times.append(t)
            if sec.crash and b == 0:
                hit("crash", clock.step_time(bar, 0, 0.5), 0.9)
            if sec.build:
                # accelerating snare roll: quarters -> 8ths -> 16ths -> 32nds
                level = min(3, int(4 * b / sec.bars))
                per_beat = [1, 2, 4, 8][level]
                for beat in range(spec.meter):
                    for k in range(per_beat):
                        frac = (b * spec.meter + beat + k / per_beat) / (sec.bars * spec.meter)
                        hit("snare", clock.beat(bar * spec.meter + beat + k / per_beat), 0.35 + 0.6 * frac)
            if last_bar and sec.fill:
                fb = bar * spec.meter + spec.meter - 1  # last beat
                if sec.fill == "snare_roll":
                    for k in range(4):
                        hit("snare", clock.beat(fb + k / 4), 0.55 + 0.15 * k)
                elif sec.fill == "hat_ratchet":
                    for k in range(8):
                        hit("hat_closed", clock.beat(fb + k / 8), 0.4 + 0.08 * k)
                    hit("clap", clock.beat(fb), 0.8)
                    hit("clap", clock.beat(fb) + 0.018, 0.6)
                elif sec.fill == "tom_fill":
                    for k, f in enumerate((180, 150, 120, 95)):
                        t = clock.beat(fb + k / 4)
                        t += jitter("tom")
                        mx.add("drums", synth.tom(0.85, f), t, 0.6)
                        truth["drums"].setdefault("tom", []).append((round(t, 5), 0.85))
                truth["fills"].append({"bar": bar, "beat": spec.meter, "kind": sec.fill})
            # ---- harmony
            chords_per_bar = max(1, int(round(1 / spec.bars_per_chord))) if spec.bars_per_chord < 1 else 1
            for c in range(chords_per_bar):
                idx = int((bar - bar0) / spec.bars_per_chord) if spec.bars_per_chord >= 1 else (bar - bar0) * chords_per_bar + c
                sym = loop[idx % len(loop)]
                chord = parse_chord(sym)
                c_start_beat = bar * spec.meter + c * spec.meter / chords_per_bar
                c_end_beat = c_start_beat + spec.meter / chords_per_bar
                t0, t1 = clock.beat(c_start_beat), clock.beat(c_end_beat)
                truth["chords"].append((t0, t1, sym if harmonic else "N"))
                if "pad" in sec.layers or ("keys" in sec.layers and spec.keys_pattern is None):
                    voice_fn = synth.pad
                    for m in chord.voicing(center=60):
                        mx.add("other", voice_fn(m, t1 - t0, 0.35, cutoff=1500.0), t0, 0.5)
                if "keys" in sec.layers and spec.keys_pattern is not None:
                    grid = spec.keys_pattern
                    steps_in_chord = range(int(c * spb / chords_per_bar), int((c + 1) * spb / chords_per_bar))
                    for s in steps_in_chord:
                        ch = grid[s % len(grid)]
                        if VEL[ch] == 0:
                            continue
                        sw = spec.swing if "keys" in spec.swing_voices else 0.5
                        t = clock.step_time(bar, s, sw) + jitter("keys")
                        dur = 1.5 * clock.ibis[0] / 4
                        for m in chord.voicing(center=62):
                            mx.add("other", synth.epiano(m, dur, 0.55, bright=0.5), t, 0.45)
                # bass
                if "bass" in sec.layers:
                    _render_bass(spec, clock, mx, truth, bar, c, chords_per_bar, chord, jitter)
            # ---- lead
            if "lead" in sec.layers and spec.lead_motif:
                for step, semis, length in spec.lead_motif:
                    t = clock.step_time(bar, step, 0.5) + jitter("lead")
                    m = 60 + (key.tonic % 12) + semis
                    if m > 84:
                        m -= 12
                    dur = length * clock.ibis[0] / 4
                    mx.add("other", synth.lead(m, dur, 0.5), t, 0.35, 0.1)
                    truth["lead"].append((round(t, 5), round(dur, 4), m))
        if sec.build:
            t0, t1 = sec_start_t, clock.beat((bar0 + sec.bars) * spec.meter)
            mx.add("other", synth.riser(t1 - t0), t0, 0.35)
            truth["events"].append({"type": "build", "start_bar": bar0, "end_bar": bar0 + sec.bars,
                                    "technique": ["snare roll accelerating", "filter opening", "noise riser"]})
        if sec.stop_beats:
            end_beat = (bar0 + sec.bars) * spec.meter
            stop_spans.append((clock.beat(end_beat - sec.stop_beats), clock.beat(end_beat)))
            truth["stops"].append({"start_bar": bar0 + sec.bars - 1, "beats": sec.stop_beats,
                                   "start": clock.beat(end_beat - sec.stop_beats), "end": clock.beat(end_beat)})
        if si > 0 and "kick" in sec.layers and "kick" not in spec.sections[si - 1].layers:
            truth["events"].append({"type": "drop", "bar": bar0})
        if sec.climax:
            truth["events"].append({"type": "climax", "bar": bar0})
        bar0 += sec.bars

    # filter sweeps for builds (applied before sidechain so the duck stays visible)
    bar0 = 0
    for sec in spec.sections:
        if sec.build:
            t0, t1 = clock.beat(bar0 * spec.meter), clock.beat((bar0 + sec.bars) * spec.meter)
            mx.filter_sweep("other", t0, t1, 300.0, 14000.0)
        bar0 += sec.bars
    if spec.sidechain_db > 0:
        for bus in ("bass", "other"):
            mx.sidechain(bus, kick_times, spec.sidechain_db, spec.sidechain_release_ms)
    for t0, t1 in stop_spans:
        mx.gate(t0, t1 - STOP_EDGE)
    # drop truth hits that fall inside stops
    for voice, hits in truth["drums"].items():
        truth["drums"][voice] = sorted(h for h in hits if not any(a - 0.005 <= h[0] < b - STOP_EDGE for a, b in stop_spans))
    mix, stems = mx.mix()
    return mix, stems, truth


def _render_bass(spec, clock, mx, truth, bar, c, chords_per_bar, chord, jitter):
    spb = spec.steps_per_bar
    grid = spec.bass_pattern
    steps = [s for s in range(int(c * spb / chords_per_bar), int((c + 1) * spb / chords_per_bar))
             if grid[s % len(grid)] != "."]
    base = 12 * (spec.bass_octave + 1) + chord.root
    if chord.root >= 8:
        base -= 12
    prev = truth["bass"][-1][2] if truth["bass"] else None
    for i, s in enumerate(steps):
        ch = grid[s % len(grid)]
        m = base + (12 if ch == "o" else 7 if ch == "5" else 0)
        sw = spec.swing if "bass" in spec.swing_voices else 0.5
        t = clock.step_time(bar, s, sw) + jitter("bass")
        nxt = steps[i + 1] if i + 1 < len(steps) else int((c + 1) * spb / chords_per_bar)
        gap = (nxt - s) * clock.ibis[0] / 4
        dur = max(0.06, min(gap * 0.85, clock.ibis[0] * 1.5))
        glide = prev if (spec.bass_glide and prev is not None and abs(prev - m) >= 5 and gap < 0.5) else None
        mx.add("bass", synth.bass(m, dur, 0.9, cutoff=spec.bass_cutoff, glide_from=glide), t, 0.55)
        truth["bass"].append((round(t, 5), round(dur, 4), m, glide is not None))
        prev = m


def write_song(name: str, out: str | Path, stems_dir: str | Path | None = None) -> dict:
    import json

    from .ingest import write_wav

    mix, stems, truth = render_song(name)
    out = Path(out)
    write_wav(out, mix, synth.SR)
    if stems_dir:
        for k, v in stems.items():
            write_wav(Path(stems_dir) / f"{k}.wav", v, synth.SR)
    out.with_suffix(".truth.json").write_text(json.dumps(truth, indent=1, default=float))
    return truth


def key_of(spec: SongSpec) -> Key:
    return parse_key(spec.key)
