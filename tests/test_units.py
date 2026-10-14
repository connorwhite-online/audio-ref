"""Fast unit tests for the building blocks."""
import json
import math
import time

import numpy as np
import pytest

from audioref import cache, config, jsonpath, loudness, theory, util


def test_theory_parse_and_spell():
    c = theory.parse_chord("Bbmaj9")
    assert c.root == 10 and c.quality == "maj9" and c.family == "maj"
    assert theory.parse_chord("F#m7b5").quality == "m7b5"
    assert theory.parse_chord("C/E").bass == 4
    assert theory.parse_chord("N") is None
    with pytest.raises(ValueError):
        theory.parse_chord("H7")
    key = theory.parse_key("F major")
    assert key.flats and key.name == "F major"
    assert theory.parse_key("A minor").mode == "minor"
    assert theory.spell_chord(theory.Chord(10, "maj9"), key) == "Bbmaj9"


@pytest.mark.parametrize("sym,expect", [("Bbmaj9", "IVmaj9"), ("Am9", "iiim9"), ("Gm9", "iim9"), ("C13", "V13"),
                                        ("Eb", "bVII"), ("Edim", "vii°"), ("Db", "bVI")])
def test_roman_numerals_in_f(sym, expect):
    assert theory.roman(theory.parse_chord(sym), theory.parse_key("F major")) == expect


def test_roman_minor_and_roundtrip():
    k = theory.parse_key("A minor")
    assert theory.roman(theory.parse_chord("E7"), k) == "V7"
    assert theory.roman(theory.parse_chord("F"), k) == "VI"
    for num in ("IVmaj9", "iim9", "V13", "bVII"):
        ch = theory.chord_from_roman(num, theory.parse_key("F major"))
        assert theory.roman(ch, theory.parse_key("F major")) == num


def test_voicing_contains_chord_tones():
    ch = theory.parse_chord("C13")
    v = ch.voicing(center=62)
    assert {n % 12 for n in v} == ch.pitch_classes


def test_otsu_and_jsonable():
    x = np.concatenate([np.random.default_rng(0).normal(0, 0.1, 200), np.random.default_rng(1).normal(5, 0.1, 200)])
    assert 0.5 < util.otsu_threshold(x) < 4.5
    j = util.to_jsonable({"a": np.float32(1.23456), "b": np.arange(3), "c": float("nan"), "d": (1, 2)}, 2)
    assert j == {"a": 1.23, "b": [0, 1, 2], "c": None, "d": [1, 2]}


def test_loudness_matches_pyloudnorm():
    import pyloudnorm as pyln

    sr = 44100
    t = np.arange(sr * 5) / sr
    x = 0.1 * np.sin(2 * np.pi * 997 * t) * (1 + 0.5 * np.sin(2 * np.pi * 0.5 * t))
    st = np.vstack([x, 0.7 * x])
    ours = loudness.integrated_loudness(st, sr)
    ref = pyln.Meter(sr).integrated_loudness(st.T)
    assert abs(ours - ref) < 0.1
    m = loudness.measure(st, sr)
    assert m["true_peak_dbtp"] >= m["sample_peak_dbfs"] - 0.01
    assert 0 <= m["lra"] < 20


def test_jsonpath():
    d = {"a": {"b": [{"id": "A", "n": 1}, {"id": "B", "n": 3}], "c": 5}, "x": [1, 2, 3, 4]}
    assert jsonpath.query(d, "a.c") == 5
    assert jsonpath.query(d, "$.x[1:3]") == [2, 3]
    assert jsonpath.query(d, "x[-1]") == 4
    assert jsonpath.query(d, "a.b[*].id") == ["A", "B"]
    assert jsonpath.query(d, "a.b[?(@.id=='B')]") == [{"id": "B", "n": 3}]
    assert jsonpath.query(d, "a.b[?(@.n>1)].id") == ["B"]
    assert jsonpath.query(d, "['a']['c']") == 5
    sel = jsonpath.select(d, ["a.c", "nope.z", "a..c"])
    assert sel["a.c"] == 5 and sel["nope.z"] is None and "error" in sel["a..c"]


def test_stage_cache_roundtrip(tmp_path):
    tc = cache.TrackCache(tmp_path, "abc")
    calls = []

    def compute():
        calls.append(1)
        return {"x": 1, "_arrays": {"arr": np.arange(5, dtype=np.float32)}}

    a = tc.cached("stage", "1", compute)
    b = tc.cached("stage", "1", compute)
    assert len(calls) == 1 and b["x"] == 1 and np.allclose(b["_arrays"]["arr"], np.arange(5))
    tc.cached("stage", "2", compute)
    assert len(calls) == 2
    assert a["_elapsed_s"] >= 0


def test_index_resolve(tmp_path):
    for tid, name in (("aaaa1111", "Song One"), ("bbbb2222", "Other")):
        tc = cache.TrackCache(tmp_path, tid)
        tc.write_meta({"name": name})
    idx = cache.CacheIndex(tmp_path)
    assert idx.resolve("aaaa").track_id == "aaaa1111"
    assert idx.resolve("Other").track_id == "bbbb2222"
    assert idx.resolve("song").track_id == "aaaa1111"
    with pytest.raises(KeyError):
        idx.resolve("zzz")


def test_config_toml_env_overrides(tmp_path, monkeypatch):
    p = tmp_path / "audioref.toml"
    p.write_text('cache_dir = "/tmp/x"\ndefault_depth = "quick"\nallow_download = true\n')
    cfg = config.load_config(p)
    assert str(cfg.cache_dir) == "/tmp/x" and cfg.default_depth == "quick" and cfg.allow_download
    monkeypatch.setenv("AUDIOREF_DEFAULT_DEPTH", "deep")
    assert config.load_config(p).default_depth == "deep"
    assert config.load_config(p, default_depth="standard").default_depth == "standard"
    with pytest.raises(ValueError):
        config.load_config(p, default_depth="bogus")


def test_ingest_decode_trim_and_errors(tmp_path):
    from audioref import ingest

    sr = 44100
    t = np.arange(sr * 4) / sr
    x = np.vstack([0.2 * np.sin(2 * np.pi * 220 * t)] * 2).astype(np.float32)
    p = tmp_path / "tone.wav"
    ingest.write_wav(p, x, sr)
    a = ingest.load(p)
    assert abs(a.duration - 4.0) < 0.01 and a.stereo.shape[0] == 2
    b = ingest.load(p, start=1.0, end=3.0)
    assert abs(b.duration - 2.0) < 0.01 and b.track_id != a.track_id
    assert abs((a.loudness["integrated_lufs"] + a.gain_db) - ingest.TARGET_LUFS) < 0.01
    with pytest.raises(ingest.IngestError):
        ingest.load(tmp_path / "missing.wav")
    with pytest.raises(ingest.IngestError):
        ingest.load(p, start=0, end=1.0)  # too short
    with pytest.raises(ingest.IngestError):
        ingest.download("https://example.com/a.mp3", allow_download=False, dest_dir=tmp_path)
    # the id is a deterministic content hash of the decoded PCM
    assert ingest.load(p).track_id == a.track_id
    ingest.write_wav(tmp_path / "copy.wav", x, sr)
    assert ingest.load(tmp_path / "copy.wav").track_id == a.track_id


def test_jobs_success_and_error():
    from audioref.jobs import JobManager

    jm = JobManager(1)

    def ok(progress, n):
        progress(0.5, "half")
        return n * 2

    def bad(progress):
        raise ValueError("boom")

    j1 = jm.submit("ok", ok, n=21)
    j2 = jm.submit("bad", bad)
    assert jm.wait(j1, 5).result == 42
    assert jm.wait(j2, 5).status == "error" and "boom" in jm.get(j2).error
    assert {j["job_id"] for j in jm.list()} >= {j1, j2}
    with pytest.raises(KeyError):
        jm.get("nope")


def test_swing_and_microtiming_from_onsets():
    from audioref import groove

    bpm = 100.0
    beat = 60 / bpm
    beats = np.arange(0, 64) * beat
    r = 0.6
    hats, kicks, snares = [], [], []
    for b in range(60):
        t0 = beats[b]
        hats += [[t0 + 0.005, 0.8], [t0 + 0.5 * r * beat + 0.005, 0.5], [t0 + 0.5 * beat + 0.005, 0.8],
                 [t0 + (0.5 + 0.5 * r) * beat + 0.005, 0.5]]
        kicks.append([t0, 1.0]) if b % 2 == 0 else snares.append([t0 + 0.012, 0.9])
    hits = {"hat_closed": hats, "kick": kicks, "snare": snares}
    sw = groove.estimate_swing(hits, beats)
    assert sw["level"] == "16th" and abs(sw["ratio_16th"] - r) < 0.01
    mt = groove.microtiming(hits, beats, sw)
    assert abs(mt["snare"]["vs_kick_ms"] - 12) < 1.5 and abs(mt["hat_closed"]["vs_kick_ms"] - 5) < 1.5
    res = groove.analyze(hits, beats, beats[::4], 4)
    main = groove.most_common_pattern(res["patterns"])
    assert main["grid_16"]["kick"] == "x.......x......."
    assert main["grid_16"]["snare"] == "....x.......x..."
    assert main["grid_16"]["hat_closed"] == "xxxxxxxxxxxxxxxx"


def test_find_loop_and_chord_cycle():
    from audioref import harmony

    bars = [("Bbmaj9",), ("Am9",), ("Gm9",), ("C13",)] * 4
    lp = harmony.find_loop(bars)
    assert lp["length_bars"] == 4 and lp["bars"] == bars[:4]
    chords = []
    t = 0.0
    for s in ["Am", "F", "C", "G"] * 2 + ["Am", "F"] * 2:
        chords.append({"start": t, "end": t + 4.0, "symbol": s})
        t += 4.0
    cyc = harmony.find_loop_chords(chords, bar_len_s=2.0, tonic_pc=9)
    assert cyc["symbols"] == ["Am", "F", "C", "G"] and abs(cyc["bars_per_chord"] - 2.0) < 1e-6


def test_timeline_solver_recovers_tempo_and_offset():
    from audioref import timeline

    bpm, off = 117.0, 0.37
    bar = 4 * 60 / bpm
    cues = [[off + 2 * bar, "scene"], [off + 6 * bar, "build"], [off + 8 * bar, "drop", {"label": "parts fly"}],
            [off + 12 * bar + 60 / bpm, "hit"], [off + 14 * bar, "hold", {"duration": 60 / bpm * 2}],
            [off + 15 * bar, "whoosh"], [off + 20 * bar, "climax"], [off + 24 * bar, "end"]]
    spec = {"tempo": {"bpm": 115.0, "beats_per_bar": 4},
            "structure": {"energy_by_label": {"drop": 0.9}, "builds": [{"bars": 2, "technique": ["snare roll accelerating"]}],
                          "stops_before_drops": True}}
    plan = timeline.plan(spec, {"duration": off + 26 * bar, "cues": cues}, {"bpm_range": [105, 125]})
    assert abs(plan["bpm"] - bpm) < 0.06 and abs(plan["offset_s"] - off) < 0.004
    assert all(c["ok"] for c in plan["cue_sync"])
    labels = [s["label"] for s in plan["sections"]]
    assert "drop" in labels and "climax" in labels and "stop" in labels and "build" in labels
    drop_sync = next(c for c in plan["cue_sync"] if c["label"] == "parts fly")
    assert drop_sync["beat"] == 1.0 and drop_sync["bar"] == 9
    assert any(e["type"] == "fill" for e in plan["events"])
    assert plan["bar_map"][0]["bar"] in (0, 1)
    with pytest.raises(ValueError):
        timeline.parse_timeline({"duration": 3})


def test_timeline_integer_bpm_and_must_hit():
    from audioref import timeline

    cues = [[1.0, "hit"], [3.05, "drop", {"must_hit": True}], [5.0, "hit"]]
    plan = timeline.plan({"tempo": {"bpm": 120, "beats_per_bar": 4}}, {"duration": 8, "cues": cues},
                         {"integer_bpm": True, "bpm_range": [110, 130]})
    assert float(plan["bpm"]).is_integer()
    assert next(c for c in plan["cue_sync"] if c["kind"] == "drop")["ok"]


def test_midi_export_and_summary(tmp_path):
    from audioref import midi

    analysis = {"rhythm": {"bpm": 120.0, "beats_per_bar": 4},
                "harmony": {"chords": [{"start": 0.0, "end": 2.0, "symbol": "Am7"}, {"start": 2.0, "end": 4.0, "symbol": "D9"}]}}
    hits = {"kick": [[0.0, 1.0], [0.5, 0.9]], "hat_closed": [[0.25, 0.5]]}
    notes = {"bass": [(0.0, 0.4, 33, 0.9, 0.0), (0.5, 0.9, 38, 0.7, 0.0)], "keys": [], "melody": []}
    files = midi.export_all(tmp_path, analysis, hits, notes)
    assert set(files) == {"drums", "bass", "chords", "full"}
    s = midi.summarize(files["full"])
    names = {i["name"] for i in s["instruments"]}
    assert {"drums", "bass", "chords"} <= names and abs(s["tempo_bpm"] - 120) < 0.01
    drums = next(i for i in s["instruments"] if i["is_drum"])
    assert drums["notes"] == 3


def test_render_midi(tmp_path):
    import pretty_midi
    import soundfile as sf

    from audioref import render

    pm = pretty_midi.PrettyMIDI(initial_tempo=100)
    d = pretty_midi.Instrument(0, is_drum=True, name="drums")
    b = pretty_midi.Instrument(38, name="bass")
    for i in range(8):
        d.notes.append(pretty_midi.Note(100, 36 if i % 2 == 0 else 38, i * 0.3, i * 0.3 + 0.05))
        b.notes.append(pretty_midi.Note(90, 40, i * 0.3, i * 0.3 + 0.25))
    pm.instruments += [d, b]
    pm.write(str(tmp_path / "x.mid"))
    r = render.render_midi(tmp_path / "x.mid", tmp_path / "x.wav")
    y, sr = sf.read(r["path"])
    assert sr == 44100 and y.shape[0] > sr and np.abs(y).max() > 0.1
    assert r["notes_rendered"]["drums"] == 8 and r["notes_rendered"]["bass"] == 8
