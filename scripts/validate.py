"""Accuracy report (spec §8): runs the analyzer on the ground-truth demo songs and writes docs/VALIDATION.md.

Two modes per song:
  * pipeline : the real end-to-end path (Demucs stems, beat tracking, transcription) on the mixed WAV
  * oracle   : each stage fed the true stems, isolating that stage's accuracy

Metrics: tempo accuracy (±2 %), beat F-measure (±70 ms), median beat error, downbeat accuracy, meter,
swing error, chord accuracy (root / majmin / sevenths, beat-sampled WCSR), key, drum F1 per voice,
section boundaries (±1 bar) and labels.

Usage: uv run python scripts/validate.py [--songs house,hiphop] [--out docs/VALIDATION.md] [--real]
       --real also runs the CC-licensed librosa example tracks (no ground truth beyond meter/key notes).
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from chord_eval import compare as chord_compare  # noqa: E402
from eval_utils import match_onsets  # noqa: E402

from audioref import demo, models  # noqa: E402
from audioref.config import load_config  # noqa: E402
from audioref.ingest import write_wav  # noqa: E402
from audioref.service import Service  # noqa: E402

VOICES = ("kick", "snare", "clap", "rim", "hat_closed", "hat_open")


def beat_f(est, ref, tol=0.07):
    p, r, f, errs = match_onsets(list(est), list(ref), tol)
    return f, (float(np.median(np.abs(errs))) if errs else None)


def eval_song(svc: Service, name: str, wav: Path, truth: dict) -> dict:
    t0 = time.time()
    r = svc.analyze(str(wav), depth="standard", trim=False, force=False)
    elapsed = time.time() - t0
    a = r["analysis"]
    rh, h = a["rhythm"], a["harmony"]
    true_bpm = truth["bpm"] + truth["tempo_drift_bpm"] / 2
    span = (truth["beats"][0] - 0.2, truth["beats"][-1] + 0.2)
    est_beats = [b for b in rh["beats"] if span[0] <= b <= span[1]]
    bf, berr = beat_f(est_beats, truth["beats"])
    est_db = [d for d in rh["downbeats"] if span[0] <= d <= span[1]]
    dbf, _ = beat_f(est_db, truth["downbeats"])
    sw = rh["swing"]
    est_swing = sw.get("ratio_16th") if sw.get("level") == "16th" else sw.get("ratio_8th") or 0.5
    chords = chord_compare(truth["chords"], h["chords"], truth["beats"])
    hits = json.loads((Path(r["files"]["json"]).parent / "hits.json").read_text())
    drum_f = {}
    for v in VOICES:
        if v in truth["drums"]:
            _, _, f, _ = match_onsets([t for t, _ in hits.get(v, [])], [t for t, _ in truth["drums"][v]], 0.03)
            drum_f[v] = f
    secs_true = [s["start_bar"] for s in truth["sections"]]
    secs_est = [s["start_bar"] for s in a["structure"]["sections"]]
    hit_bounds = sum(any(abs(e - t) <= 1 for e in secs_est) for t in secs_true)
    return {
        "song": name, "elapsed_s": round(elapsed, 1), "duration_s": round(truth["duration"], 1),
        "bpm_true": round(true_bpm, 2), "bpm_est": rh["bpm"], "tempo_ok": abs(rh["bpm"] - true_bpm) / true_bpm <= 0.02,
        "beat_f": round(bf, 3), "beat_err_ms": None if berr is None else round(berr, 1), "downbeat_f": round(dbf, 3),
        "meter_ok": rh["meter"] == truth["meter"], "swing_true": truth["swing"], "swing_est": round(est_swing, 3),
        "key_true": truth["key"], "key_est": h["key"], "key_ok": h["key"] == truth["key"],
        "loop_ok": (h.get("loop") or {}).get("symbols") == demo.SONGS[name].chords,
        "chord_root": round(chords["root"], 3), "chord_majmin": round(chords["majmin"], 3),
        "chord_sevenths": round(chords["sevenths"], 3), "drum_f1": {k: round(v, 3) for k, v in drum_f.items()},
        "section_boundaries": f"{hit_bounds}/{len(secs_true)}",
        "labels_est": [s["label"] for s in a["structure"]["sections"]], "labels_true": [s["label"] for s in truth["sections"]],
    }


def eval_oracle(name: str) -> dict:
    """Stage accuracy with true stems + true grid (isolates transcription / harmony from separation)."""
    import soxr

    from audioref import drums, groove, harmony

    mix, stems, truth = demo.render_song(name)
    m22 = lambda x: soxr.resample(x.mean(0), 44100, 22050)  # noqa: E731
    res = drums.transcribe(stems["drums"].mean(0))
    drum_f = {}
    for v in VOICES:
        if v in truth["drums"]:
            _, _, f, _ = match_onsets([t for t, _ in res["hits"].get(v, [])], [t for t, _ in truth["drums"][v]], 0.03)
            drum_f[v] = round(f, 3)
    meter = int(truth["meter"][0])
    g = groove.analyze(res["hits"], truth["beats"], truth["downbeats"], meter)
    sw = g["swing"]
    est_swing = sw.get("ratio_16th") if sw.get("level") == "16th" else sw.get("ratio_8th") or 0.5
    hh = harmony.analyze(m22(stems["bass"] + stems["other"]), truth["beats"], truth["downbeats"], meter, y_bass=m22(stems["bass"]))
    ch = chord_compare(truth["chords"], hh["chords"], truth["beats"])
    micro = {v: d.get("vs_kick_ms") for v, d in g["microtiming"].items()}
    true_micro = {v: round(ms - truth["microtiming_ms"].get("kick", 0.0), 1) for v, ms in truth["microtiming_ms"].items()}
    return {"song": name, "drum_f1": drum_f, "swing_true": truth["swing"], "swing_est": round(est_swing, 3),
            "microtiming_vs_kick_est": micro, "microtiming_vs_kick_true": true_micro, "key_ok": hh["key"] == truth["key"],
            "chord_majmin": round(ch["majmin"], 3), "chord_sevenths": round(ch["sevenths"], 3),
            "loop": hh["loop"]["symbols"]}


def real_tracks(svc: Service) -> list[dict]:
    import librosa

    notes = {"choice": "drum+bass loop (no chords)", "vibeace": "Kevin MacLeod, electronic", "fishin": "folk song w/ vocals",
             "sweetwaltz": "3/4 waltz", "pistachio": "ragtime piano", "brahms": "Hungarian Dance 5 (orchestral; G minor arr.)"}
    out = []
    for n, note in notes.items():
        p = librosa.ex(n)
        t0 = time.time()
        r = svc.analyze(p, depth="standard", trim=False)
        a = r["analysis"]
        out.append({"track": n, "note": note, "elapsed_s": round(time.time() - t0, 1), "duration_s": a["track"]["duration_s"],
                    "bpm": a["rhythm"]["bpm"], "meter": a["rhythm"]["meter"], "key": a["harmony"]["key"],
                    "loop": (a["harmony"].get("loop") or {}).get("symbols"),
                    "sections": " ".join(s["label"] for s in a["structure"]["sections"]),
                    "warnings": len(a.get("warnings") or [])})
    return out


def to_markdown(pipe: list[dict], oracle: list[dict], real: list[dict] | None, caps: dict) -> str:
    L = ["# Validation report", "", f"_Generated {time.strftime('%Y-%m-%d %H:%M')} by `scripts/validate.py` · backends: "
         + ", ".join(k for k, v in caps.items() if v) + "_", "",
         "Ground truth comes from `audioref.demo` (rendered with exact beats, drum hits, chords, sections, swing and "
         "microtiming). **Pipeline** = full standard-depth analysis of the mixed WAV (Demucs separation, beat tracking, "
         "transcription). **Oracle** = each stage fed the true stems/grid, isolating that stage.", "",
         "## Pipeline (mixed audio → analysis)", "",
         "| song | bpm (true→est) | tempo ±2% | beat F | beat err | downbeat F | meter | swing (true→est) | key | loop | chords majmin / 7ths | sections ±1 bar | time |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in pipe:
        L.append(f"| {r['song']} | {r['bpm_true']}→{r['bpm_est']} | {'✅' if r['tempo_ok'] else '❌'} | {r['beat_f']} | "
                 f"{r['beat_err_ms']} ms | {r['downbeat_f']} | {'✅' if r['meter_ok'] else '❌'} | {r['swing_true']}→{r['swing_est']} | "
                 f"{'✅' if r['key_ok'] else '❌ ' + r['key_est']} | {'✅' if r['loop_ok'] else '❌'} | {r['chord_majmin']} / {r['chord_sevenths']} | "
                 f"{r['section_boundaries']} | {r['elapsed_s']} s for {r['duration_s']} s |")
    L += ["", "Drum F1 per voice (pipeline, ±30 ms):", "", "| song | " + " | ".join(VOICES) + " |", "|---" * (len(VOICES) + 1) + "|"]
    for r in pipe:
        L.append(f"| {r['song']} | " + " | ".join(str(r['drum_f1'].get(v, '–')) for v in VOICES) + " |")
    L += ["", "Section labels (pipeline): ", ""]
    for r in pipe:
        L.append(f"- **{r['song']}** est `{' '.join(r['labels_est'])}` · truth `{' '.join(r['labels_true'])}`")
    L += ["", "## Oracle stems (stage accuracy)", "", "| song | drum F1 | swing (true→est) | microtiming vs kick est (true) ms | key | majmin / 7ths | loop |",
          "|---|---|---|---|---|---|---|"]
    for r in oracle:
        df = ", ".join(f"{k} {v}" for k, v in r["drum_f1"].items())
        mt = ", ".join(f"{k} {r['microtiming_vs_kick_est'].get(k)} ({v})" for k, v in r["microtiming_vs_kick_true"].items() if k != "kick")
        L.append(f"| {r['song']} | {df} | {r['swing_true']}→{r['swing_est']} | {mt or '–'} | {'✅' if r['key_ok'] else '❌'} | "
                 f"{r['chord_majmin']} / {r['chord_sevenths']} | {' – '.join(r['loop'])} |")
    if real:
        L += ["", "## Real recordings (CC-licensed librosa examples; no ground truth)", "",
              "| track | note | bpm | meter | key | loop | sections | time |", "|---|---|---|---|---|---|---|---|"]
        for r in real:
            L.append(f"| {r['track']} | {r['note']} | {r['bpm']} | {r['meter']} | {r['key']} | {' – '.join(r['loop'] or [])} | "
                     f"{r['sections']} | {r['elapsed_s']} s for {r['duration_s']} s |")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--songs", default=",".join(demo.SONGS))
    ap.add_argument("--out", default=str(ROOT / "docs" / "VALIDATION.md"))
    ap.add_argument("--cache", default=None)
    ap.add_argument("--real", action="store_true")
    args = ap.parse_args()
    cache = args.cache or tempfile.mkdtemp(prefix="audioref_validate_")
    svc = Service(load_config(None, cache_dir=cache))
    wav_dir = Path(tempfile.mkdtemp(prefix="audioref_songs_"))
    pipe, oracle = [], []
    for name in args.songs.split(","):
        mix, _, truth = demo.render_song(name)
        wav = wav_dir / f"{name}.wav"
        write_wav(wav, mix, 44100)
        print(f"[pipeline] {name} ...", flush=True)
        pipe.append(eval_song(svc, name, wav, truth))
        print(f"[oracle]   {name} ...", flush=True)
        oracle.append(eval_oracle(name))
    real = real_tracks(svc) if args.real else None
    md = to_markdown(pipe, oracle, real, models.capabilities())
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(md)
    (Path(args.out).with_suffix(".json")).write_text(json.dumps({"pipeline": pipe, "oracle": oracle, "real": real}, indent=1, default=str))
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
