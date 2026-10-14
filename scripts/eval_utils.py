"""Evaluation helpers shared by tests and scripts/validate.py."""
import numpy as np


def match_onsets(est, ref, tol=0.03):
    """Greedy one-to-one onset matching. Returns (precision, recall, f1, matched_errors_ms)."""
    est = sorted(est)
    ref = sorted(ref)
    used = set()
    errs = []
    tp = 0
    for r in ref:
        best, bj = None, None
        for j, e in enumerate(est):
            if j in used:
                continue
            d = e - r
            if abs(d) <= tol and (best is None or abs(d) < abs(best)):
                best, bj = d, j
        if bj is not None:
            used.add(bj)
            tp += 1
            errs.append(best * 1000)
    p = tp / len(est) if est else (1.0 if not ref else 0.0)
    r = tp / len(ref) if ref else (1.0 if not est else 0.0)
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f, errs
