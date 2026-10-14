"""Accuracy against the ground-truth demo songs (oracle stems isolate each analysis stage)."""
import numpy as np
import pytest

from audioref import drums, groove, harmony, models, rhythm, sound, structure
from chord_eval import compare as chord_compare
from conftest import mono22
from eval_utils import match_onsets

pytestmark = pytest.mark.slow

DRUM_TARGETS = {  # minimum F1 per voice on oracle drum stems
    "house": {"kick": 0.95, "clap": 0.9, "snare": 0.9, "hat_closed": 0.9, "hat_open": 0.9},
    "hiphop": {"kick": 0.95, "snare": 0.95, "hat_closed": 0.85, "rim": 0.85},
    "techno": {"kick": 0.95, "clap": 0.9, "snare": 0.9, "hat_open": 0.9},
    "waltz": {"kick": 0.95, "rim": 0.95, "hat_closed": 0.8},
    "live": {"kick": 0.95, "snare": 0.9},
}


@pytest.mark.parametrize("song", list(DRUM_TARGETS))
def test_drum_transcription_f1(demo_songs, song):
    _, truth, stems = demo_songs[song]
    res = drums.transcribe(stems["drums"].mean(0))
    for voice, target in DRUM_TARGETS[song].items():
        ref = [t for t, _ in truth["drums"][voice]]
        est = [t for t, _ in res["hits"].get(voice, [])]
        p, r, f, errs = match_onsets(est, ref, 0.03)
        assert f >= target, f"{song}/{voice}: F1 {f:.2f} < {target} (P {p:.2f} R {r:.2f})"
        assert abs(float(np.median(errs))) < 3.0, f"{song}/{voice}: median timing error {np.median(errs):.1f} ms"


@pytest.mark.skipif(not models.capabilities()["beat_this"], reason="beat_this not installed")
@pytest.mark.parametrize("song", ["house", "hiphop", "techno", "waltz", "live"])
def test_tempo_grid_and_downbeats(demo_songs, song):
    _, truth, stems = demo_songs[song]
    mix = demo_songs[song][2]
    from audioref import demo

    m, _, _ = demo.render_song(song)
    rh = rhythm.analyze(mono22(m), 22050, drums=mono22(stems["drums"]))
    hits = drums.transcribe(stems["drums"].mean(0))["hits"]
    rh = rhythm.anchor_to_hits(rh, hits)
    true_bpm = truth["bpm"] + truth["tempo_drift_bpm"] / 2
    assert abs(rh["bpm"] - true_bpm) / true_bpm < 0.01
    assert rh["meter"] == truth["meter"]
    tb = np.array(truth["beats"])
    err = np.array([np.min(np.abs(tb - b)) for b in rh["beats"] if tb[0] - 0.2 < b < tb[-1] + 0.2]) * 1000
    assert np.median(err) < (6 if song == "live" else 3), f"median beat error {np.median(err):.1f} ms"
    td = np.array(truth["downbeats"])
    hit = np.mean([np.min(np.abs(td - d)) < 0.05 for d in rh["downbeats"] if td[0] - 0.2 < d < td[-1] + 0.2])
    assert hit >= 0.9
    if song == "live":
        assert not rh["tempo_stability"]["steady"]
    else:
        assert rh["tempo_stability"]["steady"]


@pytest.mark.parametrize("song,ratio", [("house", 0.58), ("hiphop", 0.62)])
def test_swing_and_microtiming(demo_songs, song, ratio):
    _, truth, stems = demo_songs[song]
    hits = drums.transcribe(stems["drums"].mean(0))["hits"]
    g = groove.analyze(hits, truth["beats"], truth["downbeats"], 4)
    assert g["swing"]["level"] == "16th"
    assert abs(g["swing"]["ratio_16th"] - ratio) < 0.015
    micro = g["microtiming"]
    for voice, ms in truth["microtiming_ms"].items():
        if voice in micro and voice != "kick" and voice in ("clap", "snare", "hat_closed"):
            rel = ms - truth["microtiming_ms"].get("kick", 0.0)
            assert abs(micro[voice]["vs_kick_ms"] - rel) < 3.0, f"{voice}: {micro[voice]['vs_kick_ms']} vs {rel}"


def test_patterns_and_fills_house(demo_songs):
    _, truth, _ = demo_songs["house"]
    hits = {v: [[t, vel] for t, vel in hs] for v, hs in truth["drums"].items()}
    g = groove.analyze(hits, truth["beats"], truth["downbeats"], 4)
    grids = [p["grid_16"] for p in g["patterns"]["patterns"]]
    assert any(p.get("kick") == "x...x...x...x..." and p.get("clap") == "....x.......x..." for p in grids)
    fill_bars = {f["bar"] for f in g["patterns"]["fills"]}
    assert {f["bar"] for f in truth["fills"]} <= fill_bars
    kinds = {f["bar"]: f["kind"] for f in g["patterns"]["fills"]}
    assert kinds[11] == "hat_ratchet_32nd" and kinds[23] == "snare_roll"
    assert any(s["start_bar"] == 24 for s in g["patterns"]["stops"])
    assert any(r["start_bar"] == 12 for r in g["patterns"]["rolls"])


@pytest.mark.parametrize("song,majmin", [("house", 0.75), ("hiphop", 0.9), ("waltz", 0.95), ("techno", 0.95), ("live", 0.95)])
def test_key_chords_loop(demo_songs, song, majmin):
    from audioref import demo

    _, truth, stems = demo_songs[song]
    meter = int(truth["meter"][0])
    h = harmony.analyze(mono22(stems["bass"] + stems["other"]), truth["beats"], truth["downbeats"], meter,
                        y_bass=mono22(stems["bass"]))
    assert h["key"] == truth["key"]
    acc = chord_compare(truth["chords"], h["chords"], truth["beats"])
    assert acc["majmin"] >= majmin, acc
    assert h["loop"]["symbols"] == demo.SONGS[song].chords


@pytest.mark.parametrize("song,expect", [
    ("house", [("intro", 0), ("groove", 4), ("build", 12), ("drop", 16), ("breakdown", 24), ("drop", 28), ("outro", 36)]),
    ("techno", [("intro", 0), ("groove", 4), ("breakdown", 12), ("build", 16), ("drop", 20), ("outro", 28)]),
])
def test_structure_sections_and_events(demo_songs, song, expect):
    from audioref import demo

    m, stems, truth = demo.render_song(song)
    st = {k: mono22(v) for k, v in stems.items() if np.abs(v).max() > 1e-4}
    hits = {v: [[a, b] for a, b in hs] for v, hs in truth["drums"].items()}
    g = groove.analyze(hits, truth["beats"], truth["downbeats"], 4)
    res = structure.analyze(mono22(m), 22050, truth["beats"], truth["downbeats"], 4, st, hits, g, stereo_orig=m)
    got = [(s["label"], s["start_bar"]) for s in res["sections"]]
    assert got == expect
    types = {e["type"] for e in res["events"]}
    assert {"build", "drop", "climax"} <= types
    build = next(e for e in res["events"] if e["type"] == "build")
    assert "snare roll accelerating" in build["technique"]
    climax = next(e for e in res["events"] if e["type"] == "climax")
    assert climax["bar"] == next(e["bar"] for e in truth["events"] if e["type"] == "climax")


def test_sidechain_detection(demo_songs):
    _, truth, stems = demo_songs["house"]
    kicks = [t for t, _ in truth["drums"]["kick"]]
    sc = sound.sidechain(mono22(stems["bass"]), kicks)
    assert sc["detected"] and abs(sc["depth_db"] - 4.5) < 1.5 and 60 <= sc["release_ms"] <= 220
    _, truth_t, stems_t = demo_songs["techno"]
    sc_t = sound.sidechain(mono22(stems_t["bass"]), [t for t, _ in truth_t["drums"]["kick"]])
    assert sc_t["detected"] and abs(sc_t["depth_db"] - 6.0) < 1.5 and 100 <= sc_t["release_ms"] <= 220
    _, truth_l, stems_l = demo_songs["live"]
    sc_l = sound.sidechain(mono22(stems_l["bass"]), [t for t, _ in truth_l["drums"]["kick"]])
    assert not sc_l["detected"]  # musical gaps after some kicks are not a pump
    _, truth_h, stems_h = demo_songs["hiphop"]
    sc_h = sound.sidechain(mono22(stems_h["bass"]), [t for t, _ in truth_h["drums"]["kick"]])
    assert not (sc_h and sc_h["detected"])


def test_drum_sound_descriptors(demo_songs):
    _, truth, stems = demo_songs["house"]
    x = stems["drums"].mean(0)
    hits = {v: [[t, vel] for t, vel in hs] for v, hs in truth["drums"].items()}
    d = drums.describe_sounds(x, hits)
    assert 45 <= d["kick"]["f0_hz"] <= 75 and d["kick"]["start_hz"] > d["kick"]["f0_hz"]
    assert 100 <= d["kick"]["decay_ms"] <= 260  # synth: tau 70 ms -> ~160 ms to -20 dB
    assert d["hat_closed"]["decay_ms"] < 80 and d["hat_closed"]["brightness_hz"] > 6000
    assert 1000 < d["clap"]["brightness_hz"] < 4000
