import sys, time
sys.path.insert(0, "scripts")
import numpy as np
from audioref import demo, drums
from eval_utils import match_onsets
name = sys.argv[1] if len(sys.argv) > 1 else "house"
mix, stems, truth = demo.render_song(name)
t = time.time()
res = drums.transcribe(stems["drums"].mean(0))
print(name, "transcribe", round(time.time() - t, 2), "s")
for c in res.get("atoms", res.get("components", [])):
    print("   comp", c)
groups = {"kick": ["kick"], "snare": ["snare"], "clap": ["clap"], "hat_closed": ["hat_closed"], "hat_open": ["hat_open"], "perc": ["perc"], "rim": ["rim"], "tom": ["tom"], "crash": ["crash"]}
for v, ref in truth["drums"].items():
    est = [h[0] for h in res["hits"].get(v, [])]
    p, r, f, errs = match_onsets(est, [h[0] for h in ref], 0.03)
    print(f"  {v:11s} ref {len(ref):4d} est {len(est):4d}  P {p:.2f} R {r:.2f} F {f:.2f}  err med {np.median(errs) if errs else float('nan'):+.1f}ms sd {np.std(errs) if errs else float('nan'):.1f}")
extra = set(res["hits"]) - set(truth["drums"])
for v in extra:
    print("  EXTRA voice", v, len(res["hits"][v]))
