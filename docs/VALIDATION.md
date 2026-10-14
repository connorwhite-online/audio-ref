# Validation report

_Generated 2026-10-14 14:37 by `scripts/validate.py` · backends: demucs, beat_this, basic_pitch, clap_

Ground truth comes from `audioref.demo` (rendered with exact beats, drum hits, chords, sections, swing and microtiming). **Pipeline** = full standard-depth analysis of the mixed WAV (Demucs separation, beat tracking, transcription). **Oracle** = each stage fed the true stems/grid, isolating that stage.

## Pipeline (mixed audio → analysis)

| song | bpm (true→est) | tempo ±2% | beat F | beat err | downbeat F | meter | swing (true→est) | key | loop | chords majmin / 7ths | sections ±1 bar | time |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| house | 112.0→112.0 | ✅ | 1.0 | 0.3 ms | 1.0 | ✅ | 0.58→0.583 | ✅ | ✅ | 0.899 / 0.899 | 7/7 | 28.8 s for 87.5 s |
| hiphop | 88.0→88.0 | ✅ | 1.0 | 2.0 ms | 1.0 | ✅ | 0.62→0.622 | ✅ | ✅ | 0.979 / 0.979 | 3/5 | 17.4 s for 67.3 s |
| waltz | 150.0→150.0 | ✅ | 1.0 | 0.1 ms | 1.0 | ✅ | 0.5→0.5 | ✅ | ✅ | 1.0 / 1.0 | 3/3 | 8.5 s for 30.6 s |
| techno | 128.0→128.0 | ✅ | 1.0 | 0.2 ms | 1.0 | ✅ | 0.5→0.5 | ✅ | ✅ | 0.99 / 0.99 | 6/6 | 15.2 s for 61.8 s |
| live | 98.0→98.0 | ✅ | 1.0 | 2.3 ms | 1.0 | ✅ | 0.5→0.501 | ✅ | ✅ | 1.0 / 1.0 | 2/2 | 11.4 s for 41.0 s |

Drum F1 per voice (pipeline, ±30 ms):

| song | kick | snare | clap | rim | hat_closed | hat_open |
|---|---|---|---|---|---|---|
| house | 1.0 | 0.697 | 0.968 | – | 0.929 | 0.913 |
| hiphop | 0.994 | 0.886 | – | 0.0 | 0.961 | – |
| waltz | 1.0 | – | – | 0.0 | 0.909 | – |
| techno | 1.0 | 0.847 | 0.984 | – | – | 0.964 |
| live | 1.0 | 0.821 | – | – | 0.775 | – |

Section labels (pipeline): 

- **house** est `intro groove build drop breakdown drop outro` · truth `intro groove build drop breakdown drop outro`
- **hiphop** est `intro verse chorus outro` · truth `intro verse hook verse outro`
- **waltz** est `intro verse verse outro` · truth `A B A`
- **techno** est `intro groove breakdown build drop outro` · truth `intro groove break build drop outro`
- **live** est `verse verse` · truth `verse chorus`

## Oracle stems (stage accuracy)

| song | drum F1 | swing (true→est) | microtiming vs kick est (true) ms | key | majmin / 7ths | loop |
|---|---|---|---|---|---|---|
| house | kick 1.0, snare 0.982, clap 0.968, hat_closed 0.987, hat_open 0.992 | 0.58→0.579 | clap 8.3 (10.0), hat_closed 5.2 (5.0), perc None (5.0) | ✅ | 0.805 / 0.78 | Bbmaj9 – Am9 – Gm9 – C13 |
| hiphop | kick 0.988, snare 0.975, rim 1.0, hat_closed 0.997 | 0.62→0.62 | snare 17.6 (17.0), hat_closed 7.2 (8.0) | ✅ | 0.937 / 0.937 | Am9 – Dm9 – Fmaj7 – E7 |
| waltz | kick 1.0, rim 1.0, hat_closed 0.909 | 0.5→0.5 | – | ✅ | 1.0 / 1.0 | C – F – G7 – C |
| techno | kick 1.0, snare 1.0, clap 0.984, hat_open 0.984 | 0.5→0.5 | – | ✅ | 1.0 / 1.0 | Am – F – C – G |
| live | kick 1.0, snare 1.0, hat_closed 0.775 | 0.5→0.501 | – | ✅ | 1.0 / 1.0 | G – D – Em – C |

## Real recordings (CC-licensed librosa examples; no ground truth)

| track | note | bpm | meter | key | loop | sections | time |
|---|---|---|---|---|---|---|---|
| choice | drum+bass loop (no chords) | 68.0 | 4/4 | G major | G7sus4 – Gsus2 – Asus2 – Csus2 – F#sus2 | verse | 7.0 s for 25.026 s |
| vibeace | Kevin MacLeod, electronic | 130.05 | 4/4 | E major | E – G#aug – A | intro groove drop drop groove outro | 15.9 s for 61.459 s |
| fishin | folk song w/ vocals | 88.5 | 4/4 | Bb major | Bb – Eb | intro verse build chorus verse build chorus outro | 35.7 s for 132.989 s |
| sweetwaltz | 3/4 waltz | 149.99 | 3/4 | Bb major | F – Cm – Eb – Cm – F7sus4 | intro verse chorus verse verse verse | 12.5 s for 49.2 s |
| pistachio | ragtime piano | 144.34 | 4/4 | F major | F – C6 – Cdim7 – Gb – Dsus2 – Fmaj13 – Dm7 – Bbsus2 – C7sus4 | intro verse verse verse chorus chorus verse chorus outro | 18.1 s for 70.766 s |
| brahms | Hungarian Dance 5 (orchestral; G minor arr.) | 150.56 | 2/4 | G minor | Gm | verse chorus verse chorus verse verse chorus | 12.2 s for 45.845 s |
