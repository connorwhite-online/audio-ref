"""Drum transcription F1 on the Demucs drum stem cached by a pipeline run (vs demo truth)."""
import sys
sys.path.insert(0, "scripts")
import collections
import numpy as np
from audioref import demo, drums, separation
from eval_utils import match_onsets

name, stem_dir = sys.argv[1], sys.argv[2]
_, _, truth = demo.render_song(name)
st = separation.load_stems(__import__("pathlib").Path(stem_dir))
res = drums.transcribe(st["drums"].mean(0))
print("atoms:", [(a["label"], a["n_hits"], a["centroid_hz"], a["decay_ms"]) for a in res["atoms"]])
allhits = sorted((t, v) for v, hs in truth["drums"].items() for t, _ in hs)
for v in sorted(set(truth["drums"]) | set(res["hits"])):
    ref = [t for t, _ in truth["drums"].get(v, [])]
    est = [t for t, _ in res["hits"].get(v, [])]
    p, r, f, errs = match_onsets(est, ref, 0.03)
    fp = collections.Counter()
    for t in est:
        if not any(abs(t - x) < 0.03 for x in ref):
            fp[tuple(sorted(set(vv for th, vv in allhits if abs(th - t) < 0.03)))] += 1
    print(f"  {v:11s} ref {len(ref):4d} est {len(est):4d} P {p:.2f} R {r:.2f} F {f:.2f}  FP ctx {dict(fp.most_common(3))}")
