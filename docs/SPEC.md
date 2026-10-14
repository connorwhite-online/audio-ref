# Reference Audio Analyzer — Spec

A local tool that lets a coding agent (Claude) "listen" to reference tracks by turning them into structured, machine-readable musical descriptions, so it can compose or arrange new music (e.g. synthesized scores for product launch videos) that matches the references' groove, harmony, sound and energy.

The agent cannot hear audio. Today it guesses from genre names. This tool closes that gap: drop in 1–3 reference tracks, get back a precise brief (tempo, swing, drum pattern, chords, bass line, timbre, arrangement and energy over time) plus MIDI and stems the agent can read and imitate.

---

## 1. Goals and non-goals

**Goals**
- Extract everything a producer would note when studying a reference: tempo, meter, swing/groove, drum patterns per section, key, chord progression and voicing tendencies, bass line, melodic motifs, timbre/sound palette, mix balance, loudness and energy curve, arrangement sections, build/drop/fill moments.
- Output in forms an LLM uses well: a concise Markdown brief, a detailed JSON document, MIDI files per part, and optional stems.
- Compare multiple references and produce a "blend" target ("70% of A's groove, B's chords, C's sound").
- Map analysis onto a target timeline (e.g. a video cue sheet): suggest where builds, drops, stops and the climax should land.
- Run locally, offline, on a Mac (Apple Silicon) without a GPU requirement; GPU/MPS optional for speed.
- Be callable from Claude Code as an MCP server, and also usable as a plain CLI.

**Non-goals**
- Not a DAW, not a sample library, not a generative music model.
- No redistribution of the reference audio. Analysis output is descriptive (numbers, MIDI, text); stems stay in a local cache and are never committed or uploaded.
- Not trying to perfectly transcribe dense mixes; "good enough to imitate the feel" is the bar.

---

## 2. How the agent accesses it

### Recommended: local MCP server (primary) + CLI (same core)

- **MCP server** `audio-ref` (stdio transport), registered in the repo's `.mcp.json` or user-level Claude Code config. The agent calls tools directly; long jobs return a job id and stream progress.
- **CLI** `audioref` (same Python package) for humans and for scripts/CI: `audioref analyze track.mp3 --out analysis/`.
- **Python package** `audioref` with the analysis pipeline; MCP server and CLI are thin wrappers.

Why MCP: the agent can discover tools, pass file paths, get structured JSON back without parsing shell output, and the server can hold models warm in memory between calls (big speed win for stem separation and transcription).

### Inputs the agent can pass
- Local file path (mp3, wav, flac, m4a, aiff, ogg).
- A folder of references.
- Optional: a time range (`start`, `end`) to analyze only a section.
- Optional: a target timeline (JSON cue sheet: `{ duration, cues: [[t, kind, opts], …] }`) for mapping.
- URLs: **not** fetched by default (copyright and ToS). If enabled, only for audio the user owns/has rights to, behind an explicit `--allow-download` flag. Recommended default: user supplies local files.

### Outputs
Written to `./.audioref/<track-hash>/` (gitignored) and returned inline (trimmed):
- `brief.md` — the LLM-facing summary (target: < 2 pages).
- `analysis.json` — full structured result (schema in §5).
- `midi/drums.mid`, `midi/bass.mid`, `midi/chords.mid`, `midi/melody.mid`, `midi/full.mid`.
- `stems/{drums,bass,vocals,other}.wav` (optional, local only).
- `plots/` — PNG spectrogram, chromagram, energy curve, beat grid (for humans; the agent can view images too).

---

## 3. Analysis pipeline

Each stage is independently cacheable (keyed by audio hash + stage version).

### 3.1 Ingest
- Decode with ffmpeg → 44.1 kHz stereo float32; mono mix for most analyses.
- Loudness normalize a working copy (EBU R128 to -23 LUFS) so features are comparable across references; record original integrated loudness, LRA, true peak.

### 3.2 Source separation (optional but high value)
- **Demucs (htdemucs / htdemucs_6s)** → drums, bass, vocals, other (+ piano, guitar with 6s).
- Runs on CPU or MPS; cache stems. Allows per-instrument analysis (drum pattern from the drum stem, bass line from the bass stem, chords from other/piano).

### 3.3 Rhythm: tempo, meter, beats, swing, groove
- Beat and downbeat tracking: **madmom** (RNN + DBN) or **Beat This!** / **BeatNet** for robust downbeats. Fallback: librosa beat_track.
- Output: BPM (global + tempo curve), time signature, beat times, downbeat times, bar grid.
- **Swing/microtiming**: onset detection (librosa/madmom) on the drum stem; quantize to the 16th grid; measure mean offset of off-16ths vs on-16ths → swing ratio (e.g. 0.58). Per-instrument microtiming (kick/snare/hat push or lag in ms). This is the "Dilla/Kaytranada feel" number.
- **Groove template**: per-16th average onset offset and velocity across the track (a 16-slot vector per instrument) — directly reusable by a sequencer.

### 3.4 Drums: transcription and patterns
- Drum transcription on the drum stem: **ADTOF** / **Omnizart drum** / a lightweight CNN classifier on onsets (kick, snare/clap, closed hat, open hat, ride, crash, toms, rim, percussion). Output drum MIDI with velocities.
- Pattern mining: for each bar, a 16-step (or 32-step) grid per drum voice; cluster bars into patterns (A, A', B, fill). Report:
  - the core loop(s) as step grids (`x...x..x...x...` style) with velocities,
  - how often variations occur and where,
  - **fills**: bars whose pattern deviates strongly, with timing (phrase ends, pre-drop) and content (e.g. "32nd hat ratchet + clap flam on beat 4"),
  - drop-outs/stops (bars with no drums) and what plays during them.
- Drum sound descriptors per voice: pitch center, decay time, brightness, transient sharpness, saturation (so "kick: short, punchy, ~55 Hz fundamental, 160 ms decay, clicky attack").

### 3.5 Harmony: key, chords, voicings
- Key/mode: **Essentia** KeyExtractor (edma/temperley profiles), confidence.
- Chords: **Chordino (NNLS-chroma via Vamp)** or **madmom CRF chord recognition** or **BTC (Bi-directional Transformer for Chord recognition)** for extended chords (7ths, 9ths, sus, 13). Output chord symbols with start/end times aligned to the bar grid.
- Progression summary: the loop in Roman numerals and symbols (e.g. `IVmaj9 – iiim9 – iim9 – V13` in F), loop length, harmonic rhythm (chords per bar).
- Voicing tendencies (from piano/other stem via polyphonic transcription, see 3.6): extensions used, inversions, register, spread, rhythm of comping (stabs vs pads, syncopation positions).

### 3.6 Pitched transcription: bass, keys, melody
- **Basic Pitch (Spotify)** polyphonic note transcription per stem → MIDI. Optionally **MT3** for multi-instrument if available.
- Bass: pitch contour, note lengths, slides/glides, octave jumps, rhythmic positions relative to kick (locked vs syncopated), register range.
- Keys/chords: comping rhythm grid (which 16ths the stabs land on), sustain lengths.
- Melody/vocal hook: contour, range, motif repetition (useful even if we won't copy it — to understand density/space).

### 3.7 Timbre and sound palette
- Per stem and full mix: spectral centroid, rolloff, flatness, bandwidth, MFCC summaries, attack/decay envelopes, stereo width, reverb/space estimate (decay of onsets, DRR proxy), saturation/distortion estimate.
- Instrument tagging: **CLAP / PANNs / Essentia models** (e.g. `discogs-effnet`, `mtg-jamendo-instrument`, `genre_discogs400`, `mood/theme`) → labels like "electric piano (Rhodes-like)", "sub bass", "808", "shaker", "warm", "lo-fi", "jazzy", genre probabilities.
- Translate to synthesis hints the agent can use: e.g. "keys: FM electric piano, soft tines, tremolo ~5 Hz, low-passed ~2.5 kHz; bass: saw through a 300–500 Hz low-pass with fast envelope; mix: sidechain pump ~40% at 100 ms release".
- **Sidechain/pump detection**: amplitude modulation of non-drum stems correlated with kick onsets → depth (dB) and release (ms).

### 3.8 Structure, energy and arrangement
- Section segmentation: **MSAF** / self-similarity + novelty (librosa) / all-in-one music structure models → intro, verse/groove, build, drop, breakdown, outro, with bar-aligned boundaries.
- Energy curve: RMS + onset density + spectral flux per beat; per-stem activity (which layers are present in each section) as a layer chart.
- Builds and releases: detect risers (rising spectral centroid / filter sweeps / increasing onset density), snare rolls (accelerating onsets), drop points (energy jump after a dip), stops (sudden silence), and the climax (global energy peak). Report each with time, bars before/after, technique used.
- Loudness/dynamics: LUFS short-term curve, crest factor per section.

### 3.9 Similarity and blend (multi-reference)
- Normalize each reference's features; compute per-dimension distances (tempo, swing, drum grid similarity, harmonic vocabulary, timbre embedding cosine, energy-curve shape).
- `blend` produces a target spec: weighted averages for numeric features, chosen exemplars for categorical (e.g. drum loop from A, chord vocabulary from B), with a short rationale.

### 3.10 Mapping to a target timeline (the launch-video use)
- Input: target duration + cue sheet (scene changes, hits, holds, climax).
- Output: a **score plan**: tempo suggestion that makes key cues land on beats (solve for BPM and offset minimizing cue-to-beat error), section plan per scene (which layers, energy level), where to put fills, drop-outs and the climax, and per-cue sync quality (ms off-beat). This directly answers "build into the doorstep, stop on the hold, come back on the whoosh".

---

## 4. MCP tool surface

All tools take/return JSON. Paths are local. Long-running tools return `{ job_id }` and support `get_job`.

| Tool | Input | Output |
|---|---|---|
| `analyze_track` | `path`, `sections?` (`start`,`end`), `depth`: `quick` \| `standard` \| `deep`, `stems?`: bool | `{ track_id, brief_md, analysis (trimmed), files: { json, midi[], stems[], plots[] } }` |
| `get_analysis` | `track_id`, `fields?` (JSONPath list) | the requested slice of `analysis.json` |
| `get_brief` | `track_id`, `focus?`: `groove` \| `harmony` \| `sound` \| `structure` \| `all` | Markdown brief |
| `get_drum_grid` | `track_id`, `bars?`, `voices?`, `resolution`: 16 \| 32 | step grids + velocities + microtiming per voice; labelled patterns and fills |
| `get_chords` | `track_id`, `format`: `symbols` \| `roman` \| `midi` | chord timeline, loop summary, voicing notes |
| `get_midi` | `track_id`, `part`: `drums` \| `bass` \| `keys` \| `melody` \| `full` | file path + note summary (counts, range, density) |
| `compare_tracks` | `track_ids[]` | per-dimension similarity table + notable differences |
| `blend_references` | `track_ids[]`, `weights?`, `dimension_overrides?` (e.g. `{ groove: A, harmony: B, sound: C }`) | target spec JSON + brief |
| `map_to_timeline` | `target_spec` or `track_id`, `timeline` (duration + cues), `constraints?` (BPM range, must-hit cues) | score plan: BPM, offset, per-section layer plan, fill/drop/climax placements, cue sync report |
| `render_preview` (optional) | `midi path` or `score plan` | quick GM/sine-synth render WAV for sanity checks |
| `list_tracks` / `delete_track` | — | cache management |
| `get_job` | `job_id` | status, progress, partial results |

Resources (MCP resources): `audioref://track/<id>/brief`, `.../analysis`, `.../plots/<name>` so the agent can pull large artifacts on demand instead of in tool results.

Tool-result size discipline: default responses ≤ ~4k tokens; everything else via `get_analysis` slices or files.

---

## 5. `analysis.json` schema (abridged)

```json
{
  "track": { "id": "sha1…", "path": "…", "duration_s": 213.4, "sample_rate": 44100,
             "loudness": { "integrated_lufs": -9.1, "lra": 5.2, "true_peak_dbtp": -0.8 } },
  "rhythm": {
    "bpm": 112.0, "bpm_curve": [[0, 111.8], [30, 112.1]], "meter": "4/4",
    "beats": [0.42, 0.955, …], "downbeats": [0.42, 2.56, …],
    "swing": { "ratio_16th": 0.58, "by_voice": { "hat": 0.6, "snare": 0.52 } },
    "microtiming_ms": { "kick": -3, "snare": +12, "hat": +6 },
    "groove_template_16": { "hat": { "offset_ms": [0, 18, 0, 21, …], "velocity": [0.9, 0.5, …] } }
  },
  "drums": {
    "voices": ["kick", "snare", "clap", "hat_closed", "hat_open", "rim", "perc"],
    "patterns": [
      { "id": "A", "bars": [5, 6, 7, 9], "grid_16": { "kick": "x......xx.......", "clap": "....x.......x...", "hat_closed": ".x.x.x.x.x.x.x.x" },
        "velocity": { "hat_closed": [0, 0.6, 0, 1.0, …] } }
    ],
    "fills": [ { "bar": 8, "beat": 4, "kind": "hat_ratchet_32nd", "voices": ["hat_closed", "clap"], "notes": "crescendo into downbeat" } ],
    "stops": [ { "start_bar": 16, "end_bar": 17, "what_continues": ["keys"] } ],
    "sound": { "kick": { "f0_hz": 54, "decay_ms": 160, "click": "high", "saturation": "medium" } }
  },
  "harmony": {
    "key": "F major", "key_confidence": 0.81,
    "chords": [ { "start": 0.42, "end": 2.56, "symbol": "Bbmaj9", "roman": "IVmaj9" } ],
    "loop": { "symbols": ["Bbmaj9", "Am9", "Gm9", "C13"], "roman": ["IVmaj9", "iiim9", "iim9", "V13"], "bars_per_chord": 1 },
    "voicing": { "extensions": ["9", "11", "13"], "register_midi": [55, 74], "comping_grid_16": "...x..x....x..x.", "style": "short stabs" }
  },
  "bass": { "midi": "midi/bass.mid", "range_midi": [33, 48], "rhythm_grid_16": "x..x..x...x...x.", "glides": 0.35, "octave_jumps": 0.2,
            "relation_to_kick": "syncopated around kick", "sound": "saw + sub, LP ~400 Hz" },
  "sound": {
    "tags": { "genre": { "house": 0.41, "hip hop": 0.33, "neo soul": 0.21 }, "instruments": ["electric piano", "sub bass", "shaker"], "mood": ["warm", "groovy"] },
    "mix": { "sidechain": { "depth_db": 4.5, "release_ms": 110 }, "stereo_width": 0.62, "reverb": "short room", "brightness": "mid" },
    "synthesis_hints": { "keys": "FM e-piano, index decays ~120 ms, tremolo 5 Hz", "pad": "detuned saws LP 1.1 kHz" }
  },
  "structure": {
    "sections": [ { "label": "intro", "start_bar": 0, "end_bar": 4, "layers": ["keys", "pad"], "energy": 0.2 } ],
    "events": [ { "type": "build", "start_bar": 14, "end_bar": 16, "technique": ["snare roll accelerating", "filter opening"] },
                { "type": "drop", "bar": 16 }, { "type": "climax", "bar": 48 } ],
    "energy_per_beat": [0.21, 0.22, …]
  }
}
```

---

## 6. The brief (`brief.md`) format

Short, opinionated, in producer language, ordered by what matters for imitation:

1. **One-line vibe** ("Swung 112 bpm neo-soul house: punchy kick, layered clap, Rhodes stabs over a gliding saw bass, heavy sidechain").
2. **Groove**: BPM, swing %, kick/clap/hat grids (16-step strings), microtiming notes, sidechain.
3. **Harmony**: key, loop in symbols + Roman numerals, voicing/comping style.
4. **Bass**: rhythm grid, register, glides, relation to kick.
5. **Sound palette**: per instrument with synthesis hints.
6. **Arrangement & energy**: sections with layers, where builds/fills/stops/climax happen and how.
7. **"To imitate this, do:"** — 5–10 concrete, actionable instructions.

---

## 7. Implementation

- **Language**: Python 3.11. Package `audioref`, `pyproject.toml`, `uv` for env management.
- **Core deps**: `librosa`, `soundfile`, `numpy`, `scipy`, `madmom` (or `beat_this`), `essentia` (+ `essentia-tensorflow` models), `basic-pitch`, `demucs`, `pretty_midi`/`mido`, `pyloudnorm`, `msaf` (optional), `torch` (CPU/MPS), `matplotlib` (plots).
- **Chord model**: Chordino via `vamp` host (`vamp` Python bindings + NNLS-Chroma plugin) or BTC weights; pick the one that best handles extended chords on test set.
- **MCP**: the official Python MCP SDK (`mcp`), stdio server `audioref-mcp`; job queue in-process (asyncio + thread pool), models loaded lazily and kept warm.
- **Caching**: content hash of decoded PCM; each stage writes `stage_name@version.json`; re-runs skip completed stages.
- **Performance targets** (M-series Mac, 3-minute track): `quick` (no stems: tempo, key, chords, energy) < 20 s; `standard` (+ stems, drums, bass, transcription) < 2 min; `deep` (+ structure models, tagging, plots) < 4 min.
- **Config**: `audioref.toml` (cache dir, default depth, model choices, allow_download=false).
- **Install**: `uv tool install audioref` → provides `audioref` CLI and `audioref-mcp`. Claude Code config:

```json
{ "mcpServers": { "audioref": { "command": "audioref-mcp", "args": ["--cache", "~/.audioref"] } } }
```

---

## 8. Accuracy, validation, failure modes

- **Test set**: ~20 tracks with known metadata (stems/MIDI from open datasets: MUSDB18, Groove MIDI Dataset for drums, GuitarSet/MAESTRO-style for pitch, Isophonics/Billboard chord annotations).
- **Metrics**: beat F-measure, tempo accuracy (±2%), downbeat accuracy, chord WCSR (majmin + sevenths), drum transcription F1 per voice, swing estimate error vs annotated.
- **Confidence everywhere**: every field carries a confidence; the brief flags low-confidence items ("chords uncertain in bars 20–24: dense mix").
- **Failure modes**: dense/distorted mixes (chords), ambiguous downbeats (half/double tempo — report both candidates), swung vs triplet confusion, live tempo drift (use tempo curve), stems bleeding (cross-check with full mix).

---

## 9. Legal and privacy

- Works on audio the user provides locally; no uploading anywhere by default.
- Output is analysis and MIDI for reference/imitation of style, not copies; stems are cached locally only and excluded from git (`.audioref/` in `.gitignore`).
- No URL downloading unless explicitly enabled by the user for audio they have rights to.

---

## 10. Phased build plan

1. **MVP (CLI)**: ingest, tempo/beats/downbeats, swing, key, chords, energy curve, `brief.md` + `analysis.json`. Validate on 5 tracks.
2. **Stems + drums + bass**: Demucs, drum transcription → grids/patterns/fills, Basic Pitch on bass/keys, MIDI export.
3. **MCP server**: tool surface in §4, jobs, caching, resources.
4. **Sound palette**: tagging models, timbre descriptors, sidechain detection, synthesis hints.
5. **Multi-reference**: `compare_tracks`, `blend_references`.
6. **Timeline mapping**: `map_to_timeline` with BPM/offset solving against a cue sheet; integrates with a launch-video cue sheet format (`{ duration, cues: [[t, kind, opts]] }`).
7. **Preview render** (optional) for quick sanity checks.

---

## 11. Example agent workflow

1. User: "Score this video like these three tracks" + `refs/a.mp3 refs/b.mp3 refs/c.mp3`.
2. Agent: `analyze_track` ×3 (`standard`), reads briefs.
3. Agent: `blend_references` with `{ groove: a, harmony: b, sound: c }`.
4. Agent: `map_to_timeline(target_spec, timeline=cue_sheet.json)` → BPM 112, offset so the drop lands on "parts fly to screen", stops on holds, build into the climax cue.
5. Agent writes/updates its synth score (e.g. a Node/Python synth like this repo's `sfx.mjs`) from the plan + MIDI grids, renders, then optionally re-runs `analyze_track` on its own output and `compare_tracks` against the references to check it's close (closing the loop without ears).
