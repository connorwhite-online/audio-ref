# audio-ref — reference audio analyzer

Lets a coding agent (Claude) "listen" to reference tracks. Drop in 1–3 references and get back what a producer would
note when studying them — tempo, swing, drum patterns, chords, bass line, sound palette, arrangement and energy — as a
short Markdown **brief**, a detailed **`analysis.json`**, per-part **MIDI**, optional **stems** and **plots**. Then
**compare** references, **blend** them into a target spec ("A's groove, B's chords, C's sound") and **map** that onto a
video cue sheet so builds, drops, stops and the climax land on the cuts.

Runs locally and offline on a Mac (Apple Silicon; MPS used when available). Usable as an **MCP server** (`audioref-mcp`)
or a **CLI** (`audioref`) over the same Python package. See [docs/SPEC.md](docs/SPEC.md) for the original spec and
[docs/VALIDATION.md](docs/VALIDATION.md) for measured accuracy.

```text
**Vibe:** Swung 112 bpm house in F major: punchy short kick, clap on 2 & 4, e-piano stabs, gliding bass, sidechain pump.
- Swing 58% swing on 16ths (light swing) — swung voices: hat_closed 58%.
- Microtiming clap +10 ms vs kick (laid back); hat_closed +5 ms vs kick (laid back).
- Loop (4 bars, 1 bar(s)/chord): `Bbmaj9 – Am9 – Gm9 – C13`  =  `IVmaj9 – iiim9 – iim9 – V13`
kick        X...X...X...X...
clap        ....x.......x...
hat_closed  .oXo.oXo.oXo.oXo
```

## Install

Requires Python 3.11, [uv](https://docs.astral.sh/uv/) and ffmpeg (`brew install ffmpeg uv`).

```bash
uv sync --all-extras          # in this repo: core + Demucs, Basic Pitch, Beat This!, CLAP
```

```bash
uv tool install ".[all]"      # or install the `audioref` and `audioref-mcp` commands globally
```

The extras are optional; without them the analyzer still works with built-in fallbacks
(HPSS pseudo-stems, librosa beat tracking, pYIN/NNLS transcription, heuristic tags):

| extra | adds | first-use download |
|---|---|---|
| `stems` | Demucs `htdemucs` source separation (drums/bass/vocals/other) | ~80 MB |
| `beats` | Beat This! beat + downbeat tracker | ~77 MB |
| `transcribe` | Spotify Basic Pitch note transcription (CoreML on macOS, bundled) | – |
| `tagging` | LAION CLAP zero-shot genre/instrument/mood tags (`deep` only) | ~780 MB |

Model weights download once on first use and are then loaded from the local cache with **no network access**
(the loaders force Hugging Face offline mode first).

## Use from Claude Code (MCP)

This repo ships a `.mcp.json` that runs the server from the checkout:

```json
{ "mcpServers": { "audioref": { "command": "uv", "args": ["run", "--directory", ".", "audioref-mcp", "--cache", ".audioref"] } } }
```

With a global install (`uv tool install ".[all]"`) use the spec's form instead:

```json
{ "mcpServers": { "audioref": { "command": "audioref-mcp", "args": ["--cache", "~/.audioref"] } } }
```

| tool | what it does |
|---|---|
| `analyze_track(path, depth?, start?, end?, stems?, plots?, force?, wait_s?)` | analyze a file or folder → `brief_md`, trimmed analysis, file paths. Long runs return a `job_id` and stream progress. |
| `get_job(job_id, wait_s?)` | status / result of a background analysis |
| `get_analysis(track_id, fields?)` | JSONPath slices of `analysis.json` (`rhythm.swing`, `harmony.chords[0:8]`, `structure.sections[*].label`, `drums.patterns[?(@.id=='A')]`) |
| `get_brief(track_id, focus?)` | Markdown brief; focus `groove` \| `harmony` \| `sound` \| `structure` \| `all` |
| `get_drum_grid(track_id, bars?, voices?, resolution?)` | per-bar step grids + velocities + microtiming, labelled patterns, fills, stops, rolls (16 swing-aware, or 32) |
| `get_chords(track_id, format?)` | chord timeline as `symbols` \| `roman` \| `midi`, loop summary, voicing notes |
| `get_midi(track_id, part?)` | path + note summary for `drums` \| `bass` \| `keys` \| `chords` \| `melody` \| `full` |
| `compare_tracks(track_ids)` | per-dimension similarity (tempo, groove, drums, harmony, bass, sound, structure) + notable differences |
| `blend_references(track_ids, weights?, dimension_overrides?)` | target spec + brief, e.g. `{"groove": "a", "harmony": "b", "sound": "c"}` |
| `map_to_timeline(target, timeline, constraints?)` | BPM + bar offset solving cue-to-beat error, section plan, builds/stops/fills/climax, bar map, per-cue sync |
| `render_preview(source, out?, target?)` | quick WAV render of a MIDI file or score plan (re-analyze it to close the loop) |
| `list_tracks()` / `delete_track(track_id)` / `status()` | cache management, backends and warm models |

Resources: `audioref://tracks`, `audioref://track/<id>/brief`, `audioref://track/<id>/analysis`,
`audioref://track/<id>/plots/<name>` (PNG). Tool results stay ≤ ~4k tokens; everything else is one slice/resource away.

Typical agent flow: `analyze_track` ×3 → read the briefs → `blend_references(…, {groove: a, harmony: b, sound: c})` →
`map_to_timeline(spec_id, cue_sheet)` → write the synth score → `render_preview` / `analyze_track` your render →
`compare_tracks` against the references.

## Use from the shell (CLI)

```bash
audioref analyze refs/a.mp3 refs/b.mp3 --depth standard --out analysis/   # brief to stdout, files to analysis/
audioref analyze refs/ --depth quick                                      # a whole folder
audioref analyze long.mp3 --start 60 --end 120 --plots                    # just a section, with PNG plots
audioref brief a --focus groove                                           # ids, id prefixes or names work
audioref get a rhythm.swing 'harmony.chords[0:8]'
audioref grid a --bars 0-7 --voices kick,clap,hat_closed --resolution 32
audioref chords a --format roman
audioref midi a --part bass
audioref compare a b c
audioref blend a b c --groove a --harmony b --sound c --out target.json
audioref map target.json --timeline cues.json --bpm-range 100-120 --render plan.wav
audioref render analysis/midi/full.mid --out check.wav
audioref list | audioref delete <id> | audioref status
audioref demo house house.wav --stems house_stems/                        # a ground-truth test song
```

Add `--json` to any command for machine-readable output (`audioref analyze x.mp3 --json`).

### Cue sheets

```json
{ "duration": 42.0,
  "cues": [[0.0, "start"], [6.9, "scene"], [13.7, "build"], [17.1, "drop", {"label": "parts fly to screen", "must_hit": true}],
           [27.4, "hold", {"duration": 1.1}], [28.6, "whoosh"], [34.3, "climax"], [41.5, "end"]] }
```

Kinds: `start`/`intro`, `scene`/`section`, `build`, `drop`/`reveal`/`impact`, `hit`/`accent`, `whoosh`, `hold`/`stop`,
`breakdown`, `climax`, `end`/`outro`/`button`. Options: `label`, `weight`, `must_hit`, `level` (`bar`|`half`|`beat`|`8th`),
`duration`. Bar-level kinds (drop, climax, scene, end…) are solved onto downbeats; the rest onto beats. Constraints:
`bpm_range`, `integer_bpm`, `must_hit` (labels), `tolerance_ms` (default 25), `target_bpm`.

## Depths and speed

| depth | runs | measured (M1, 16 GB) |
|---|---|---|
| `quick` | HPSS pseudo-stems, beats/downbeats, drums, swing, key, chords, energy/sections | ~6–12 s per minute of audio |
| `standard` (default) | + Demucs stems, Basic Pitch bass/keys/vocal transcription, parts, sound palette, sidechain, MIDI | ~20–30 s per minute (first run of a session adds ~10 s model load) |
| `deep` | + CLAP tags and PNG plots | + ~5–15 s |

Every stage is cached under `<cache>/<track-id>/stages/` (keyed by audio hash + stage version), so re-runs, deeper
re-runs and code upgrades only recompute what changed. The track id is a hash of the decoded PCM, so renamed or
moved copies of the same file reuse the cache.

## Outputs

`<cache>/<track_id>/` (default cache `./.audioref`, gitignored):

- `brief.md` — the LLM-facing summary: one-line vibe, groove (tempo, swing, microtiming, step grids, fills, sidechain),
  harmony (key, loop in symbols + Roman numerals, vocabulary, voicing/comping), bass, sound palette with synthesis
  hints, arrangement & energy table with events, **"To imitate this, do:"** steps, and flagged uncertainties.
- `analysis.json` — everything (schema below), with confidences and warnings.
- `midi/{drums,bass,chords,keys,melody,full}.mid` — tempo set to the detected BPM, notes at absolute audio times
  (bar 1 = `rhythm.first_downbeat`). `chords.mid` is block chords from the recognised symbols; `keys.mid` is the raw
  transcription of the other/keys stem.
- `stems/{drums,bass,other,vocals}.wav` (with `--stems`), `plots/*.png` (deep or `--plots`): spectrogram, chromagram,
  energy, beat grid, drum pattern, layer chart.

`analysis.json` top level: `track` (id, path, duration, loudness: integrated LUFS, LRA, true peak) · `rhythm` (bpm,
bpm_curve, tempo_candidates incl. half/double readings, meter, beats, downbeats, swing {ratio_16th, ratio_8th, level,
by_voice}, microtiming_ms, groove_template_16 per voice) · `drums` (voices, patterns with 16-step grids/accents/
velocities, bar_patterns, fills, stops, rolls, per-voice sound) · `harmony` (key, chords with roman numerals and
confidence, loop, harmonic_rhythm, vocabulary, extensions, voicing) · `bass` · `melody` · `sound` (tags, mix,
sidechain, per-stem descriptors, synthesis_hints) · `structure` (sections with layers/energy/LUFS, events: build /
drop / stop / drum_stop / climax, energy_per_beat, layers_per_bar, loudness curve) · `confidence` · `warnings`.

## How it works

| stage | method |
|---|---|
| ingest | ffmpeg → 44.1 kHz stereo float; BS.1770 loudness, LRA, 4× true peak; working copy normalised to −23 LUFS |
| separation | Demucs `htdemucs` on MPS/CPU (stems cached as FLAC); fallback HPSS + crossover |
| beats | Beat This! (or librosa DP) → robust line fit / local refinement on an onset envelope, extended over the audible span, then **anchored to the kick** so other voices' microtiming is measured against it; meter + downbeat phase from the model with a cue-based heuristic fallback; tempo curve, half/double candidates |
| drums | onset events on the drum stem → per-event attack spectra → exemplar atoms (kick/snare/hat) + residual-cluster atoms → NNLS per event with adaptive (Otsu) thresholds → per-hit refinement to ~1 ms and clap/rim/open-hat/crash classification; hats masked by snares are recovered by hat-grid reasoning |
| groove | per-voice swing relative to the voice's own lag, microtiming vs kick, swing-aware 16/32-step grids, pattern clustering (A, A′, B…), fills (kind + notes), drop-outs, accelerating rolls, groove template |
| harmony | constant-Q → beat-synchronous NNLS note salience (Chordino-style harmonic dictionary) → bass/treble chroma → bar-aware Viterbi over triad families → per-segment extensions (7/9/11/13/sus/6, slash bass) from chroma + transcribed notes → key (Krumhansl/Temperley/EDM profiles + chord fit + tonic weight) → Roman numerals, loop (bar-period or chord-cycle), harmonic rhythm |
| notes | Basic Pitch per stem (bass cleanup: octave de-dup, fragment merge, glide detection) with pYIN / NNLS fallbacks |
| sound | spectral/envelope descriptors per stem, stereo width, space (HF floor between hits), tremolo, low-pass estimate, **sidechain** (kick-locked V-shaped dips above the kick band, vs kick-free controls, consistency across kicks), CLAP tags, synthesis hints |
| structure | bar features (MFCC, chroma, layers, loudness) → Foote novelty + layer changes + phrase prior → sections, similarity groups, functional labels (club vs song vocabulary); builds (roll / filter / riser / kick removed / stop), drops, stops, climax |
| compare / blend | per-dimension features and similarities; weighted averages for numbers, exemplar sources per dimension |
| timeline | vectorised BPM × offset search with tempo prior and must-hit constraints; section planning with reference energies/layers, automatic builds, stop-before-drop, fills, climax |

Spec deviations (all for macOS/Python 3.11 installability): madmom → Beat This!; Essentia key models → own profile
ensemble; Chordino/BTC → own NNLS-chroma + Viterbi chord recogniser; ADTOF/Omnizart → own onset-atom drum transcriber;
MSAF → own novelty segmentation; Essentia TF taggers → CLAP via transformers.

## Accuracy

`uv run python scripts/validate.py --real` renders the five ground-truth songs (`audioref demo …`: swung house with
sidechain/build/stop, laid-back hip hop, 3/4 waltz, techno with 2-bar harmonic rhythm, drifting "live" take), runs the
full pipeline and each stage on true stems, and writes [docs/VALIDATION.md](docs/VALIDATION.md). Headlines: tempo within
0.1 bpm, beat grids within ~2 ms, swing within ±0.01, microtiming within ~1 ms, keys and chord loops exact on all five
songs, drum F1 ≥ 0.95 for core voices on clean stems. Real CC-licensed recordings (librosa examples) are analysed in the
same report.

Known limits: dense real mixes reduce chord-extension and drum-voice accuracy (always check `confidence` / `warnings`);
percussion that never sounds alone (a shaker always layered on hats) may not get its own voice; crash detection is
coarse; release times measured on pads read long (note envelopes); half/double-time is ambiguous by nature — both
readings are reported.

## Configuration

`audioref.toml` (searched in `./` then `~/.config/audioref/`, or `--config`); every key can also be set with
`AUDIOREF_<KEY>` env vars. See [audioref.toml.example](audioref.toml.example).

## Legal and privacy

Works only on local audio you provide; nothing is uploaded. URL inputs are refused unless explicitly enabled
(`--allow-download` / `allow_download = true`, direct file URLs only — for audio you own or have rights to). Outputs
are descriptive (numbers, MIDI, text); stems stay in the local cache, which is gitignored.

## Development

```bash
uv run pytest -m "not slow"   # fast unit tests (~2 s)
uv run pytest                 # + accuracy and end-to-end tests: pipeline, CLI, MCP in-process and over stdio (~5–10 min)
uv run python scripts/validate.py --real
```
