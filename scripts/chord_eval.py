"""Chord evaluation helpers (beat-sampled root / majmin / sevenths / exact accuracy)."""
from audioref.theory import parse_chord, QUALITIES

SEVENTH_CLASS = {"": "maj", "6": "maj", "add9": "maj", "6/9": "maj", "5": "maj",
                 "7": "7", "9": "7", "11": "7", "13": "7", "7#9": "7", "7b9": "7",
                 "maj7": "maj7", "maj9": "maj7", "maj13": "maj7",
                 "m": "min", "m6": "min", "madd9": "min", "m7": "min7", "m9": "min7", "m11": "min7", "m13": "min7",
                 "mmaj7": "minmaj7", "dim": "dim", "m7b5": "hdim7", "dim7": "dim7", "aug": "aug",
                 "sus2": "sus2", "sus4": "sus4", "7sus4": "7sus4"}


def chord_at(chords, t):
    for c in chords:
        if c["start"] <= t < c["end"]:
            return c["symbol"]
    return "N"


def compare(truth_chords, est_chords, beats):
    """truth_chords: [(start, end, symbol)], est: [{'start','end','symbol'}]. Sampled at beat midpoints."""
    n = root = majmin = sev = exact = 0
    for i in range(len(beats) - 1):
        t = (beats[i] + beats[i + 1]) / 2
        ts = next((s for a, b, s in truth_chords if a <= t < b), "N")
        if ts == "N":
            continue
        es = chord_at(est_chords, t)
        n += 1
        tc, ec = parse_chord(ts), parse_chord(es) if es != "N" else None
        if ec is None:
            continue
        if tc.root == ec.root:
            root += 1
            if tc.family == ec.family:
                majmin += 1
            if SEVENTH_CLASS.get(tc.quality) == SEVENTH_CLASS.get(ec.quality):
                sev += 1
            if set((tc.root + iv) % 12 for iv in QUALITIES[tc.quality]) == set((ec.root + iv) % 12 for iv in QUALITIES[ec.quality]):
                exact += 1
    n = max(n, 1)
    return {"root": root / n, "majmin": majmin / n, "sevenths": sev / n, "exact": exact / n, "n": n}
