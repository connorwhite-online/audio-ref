"""End-to-end: pipeline depths, caching, CLI, MCP (in-process and stdio), compare/blend/map/render."""
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from audioref import models
from audioref.cli import main as cli_main

pytestmark = pytest.mark.slow
ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ pipeline
def test_standard_analysis_schema_and_accuracy(house_standard):
    res, truth = house_standard
    a = res["analysis"]
    for key in ("track", "rhythm", "drums", "harmony", "bass", "melody", "sound", "structure", "confidence", "warnings",
                "files", "generator", "schema_version"):
        assert key in a
    assert abs(a["rhythm"]["bpm"] - 112.0) < 0.2 and a["rhythm"]["meter"] == "4/4"
    assert abs(a["rhythm"]["first_downbeat"] - truth["downbeats"][0]) < 0.02
    assert abs(a["rhythm"]["swing"]["ratio_16th"] - 0.58) < 0.03
    assert a["harmony"]["key"] == "F major"
    assert a["harmony"]["loop"]["symbols"] == ["Bbmaj9", "Am9", "Gm9", "C13"]
    assert a["harmony"]["loop"]["roman"] == ["IVmaj9", "iiim9", "iim9", "V13"]
    main = max((p for p in a["drums"]["patterns"] if p["variation_of"] is None), key=lambda p: p["count"])
    assert main["grid_16"]["kick"] == "x...x...x...x..."
    assert a["track"]["loudness"]["integrated_lufs"] is not None
    if models.capabilities()["demucs"]:
        assert a["generator"]["methods"]["separation"].startswith("htdemucs")
        assert a["bass"]["present"] and a["sound"]["sidechain"]["detected"]
    files = res["files"]
    assert Path(files["json"]).is_file() and Path(files["brief"]).is_file()
    mids = {Path(p).stem for p in files["midi"]}
    assert {"drums", "bass", "chords", "full"} <= mids
    labels = [s["label"] for s in a["structure"]["sections"]]
    assert labels[0] == "intro" and labels[-1] == "outro" and "drop" in labels


def test_brief_format(house_standard, service):
    res, _ = house_standard
    brief = Path(res["files"]["brief"]).read_text()
    for heading in ("**Vibe:**", "## Groove", "## Harmony", "## Bass", "## Sound palette", "## Arrangement & energy",
                    "## To imitate this, do:"):
        assert heading in brief, heading
    assert len(brief.split()) < 1300, "brief should stay under ~2 pages"
    assert "IVmaj9" in brief and "112" in brief
    for focus in ("groove", "harmony", "sound", "structure"):
        b = service.get_brief(res["track_id"], focus)
        assert b.startswith("# Reference brief")
    with pytest.raises(ValueError):
        service.get_brief(res["track_id"], "nope")


def test_cache_hit_is_fast(service, house_standard, demo_songs):
    res, _ = house_standard
    t = time.time()
    again = service.analyze(str(demo_songs["house"][0]), depth="quick")
    assert again["track_id"] == res["track_id"] and time.time() - t < 3.0  # deeper result reused


def test_quick_depth_and_section_range(service, demo_songs):
    path = demo_songs["techno"][0]
    t = time.time()
    r = service.analyze(str(path), depth="quick", trim=False)
    assert time.time() - t < 30
    a = r["analysis"]
    assert a["generator"]["depth"] == "quick" and a["generator"]["methods"]["separation"] == "hpss-fallback"
    assert abs(a["rhythm"]["bpm"] - 128) < 0.5 and a["harmony"]["key"] == "A minor"
    part = service.analyze(str(path), depth="quick", start=15.0, end=45.0, trim=False)
    assert part["track_id"] != r["track_id"] and abs(part["analysis"]["track"]["duration_s"] - 30.0) < 0.05
    assert part["analysis"]["track"]["start"] == 15.0


def test_folder_analysis(service, demo_songs, tmp_path):
    import shutil

    for n in ("waltz", "live"):
        shutil.copy(demo_songs[n][0], tmp_path / f"{n}.wav")
    (tmp_path / "notes.txt").write_text("ignore me")
    out = service.analyze(str(tmp_path), depth="quick")
    assert isinstance(out, list) and len(out) == 2
    meters = sorted(o["analysis"]["rhythm"]["meter"] for o in out)
    assert meters == ["3/4", "4/4"]


def test_getters(service, house_standard):
    res, _ = house_standard
    tid = res["track_id"]
    sl = service.get_analysis(tid[:8], ["rhythm.bpm", "harmony.loop.roman", "structure.sections[*].label"])
    assert sl["rhythm.bpm"] > 100 and sl["harmony.loop.roman"][0] == "IVmaj9"
    g = service.get_drum_grid(tid, bars=[4, 5], voices=["kick", "clap"])
    assert [b["bar"] for b in g["bars"]] == [4, 5] and set(g["bars"][0]["grid"]) <= {"kick", "clap"}
    g32 = service.get_drum_grid(tid, bars=[11], resolution=32)
    assert g32["steps_per_bar"] == 32 and len(g32["bars"][0]["grid"]["kick"]) == 32
    ch = service.get_chords(tid, "roman")
    assert ch["chords"] and all("roman" in c for c in ch["chords"])
    assert service.get_chords(tid, "midi")["midi"]["instruments"]
    m = service.get_midi(tid, "drums")
    assert m["summary"]["instruments"][0]["is_drum"] and m["timing"]["bpm"] > 100
    with pytest.raises(ValueError):
        service.get_midi(tid, "kazoo")


# ------------------------------------------------------------------ multi-reference + timeline + render
def test_compare_blend_map_render(service, house_standard, demo_songs, tmp_path):
    res, _ = house_standard
    hh = service.analyze(str(demo_songs["hiphop"][0]), depth="standard")
    cmp = service.compare_tracks([res["track_id"], hh["track_id"]])
    pair = cmp["pairwise"][0]
    assert 0 <= pair["overall"] <= 1 and set(cmp["dimensions"]) <= set(pair)
    assert any("bpm faster" in d for d in cmp["differences"])
    same = service.compare_tracks([res["track_id"], res["track_id"]])["pairwise"][0]
    assert same["overall"] > 0.95

    b = service.blend_references([res["track_id"], hh["track_id"]], weights=[0.7, 0.3],
                                 dimension_overrides={"groove": hh["track_id"], "harmony": res["track_id"]})
    spec = b["target_spec"]
    assert spec["sources"]["groove"]["track_id"] == hh["track_id"]
    assert spec["harmony"]["key"] == "F major"
    assert abs(spec["groove"]["swing_ratio"] - 0.62) < 0.04
    assert "Blend target" in b["brief_md"]

    tl = {"duration": 40.0, "cues": [[0.0, "start"], [6.9, "scene"], [13.7, "build"], [17.1, "drop", {"label": "parts fly"}],
                                     [27.4, "hold", {"duration": 1.1}], [28.6, "whoosh"], [34.3, "climax"], [39.0, "end"]]}
    plan = service.map_to_timeline(b["spec_id"], tl, {"bpm_range": [80, 100]})
    assert 80 <= plan["bpm"] <= 100 and plan["cue_sync"] and plan["bar_map"]
    assert next(c for c in plan["cue_sync"] if c["label"] == "parts fly")["ok"]
    assert {"drop", "stop", "climax"} <= {e["type"] for e in plan["events"]}
    plan2 = service.map_to_timeline(res["track_id"], tl)  # straight from a track id
    assert abs(plan2["bpm"] - 112) < 112 * 0.09

    r = service.render_preview(plan["plan_id"], str(tmp_path / "plan.wav"))
    import soundfile as sf

    y, sr = sf.read(r["path"])
    assert abs(len(y) / sr - 41.5) < 1.0 and np.abs(y).max() > 0.1
    rm = service.render_preview(service.get_midi(res["track_id"], "full")["path"], str(tmp_path / "full.wav"))
    assert Path(rm["path"]).is_file()

    # close the loop: re-analyze the rendered plan and check the tempo survived
    back = service.analyze(r["path"], depth="quick")
    assert abs(back["analysis"]["rhythm"]["bpm"] - plan["bpm"]) / plan["bpm"] < 0.02


def test_list_and_delete(service, demo_songs):
    r = service.analyze(str(demo_songs["waltz"][0]), depth="quick", start=0.0, end=10.0)
    ids = {t["track_id"] for t in service.list_tracks()}
    assert r["track_id"] in ids
    d = service.delete_track(r["track_id"])
    assert d["deleted"] == r["track_id"]
    assert r["track_id"] not in {t["track_id"] for t in service.list_tracks()}
    with pytest.raises(KeyError):
        service.get_brief(r["track_id"])


# ------------------------------------------------------------------ CLI
def test_cli_end_to_end(cache_dir, demo_songs, house_standard, tmp_path, capsys):
    res, _ = house_standard
    c = ["--cache", str(cache_dir)]
    assert cli_main(c + ["analyze", str(demo_songs["house"][0]), "-q", "--out", str(tmp_path / "out")]) == 0
    out = capsys.readouterr().out
    assert "# Reference brief" in out and (tmp_path / "out" / "analysis.json").is_file()
    assert (tmp_path / "out" / "midi" / "drums.mid").is_file()
    assert cli_main(c + ["get", res["track_id"], "rhythm.bpm"]) == 0
    assert json.loads(capsys.readouterr().out)["rhythm.bpm"] > 100
    for args in (["brief", res["track_id"], "--focus", "groove"], ["grid", res["track_id"], "--bars", "4-5"],
                 ["chords", res["track_id"], "--format", "roman"], ["midi", res["track_id"], "--part", "bass"],
                 ["list"], ["status"], ["compare", res["track_id"], res["track_id"]]):
        assert cli_main(c + args) == 0, args
    capsys.readouterr()
    assert cli_main(c + ["--json", "list"]) == 0
    assert any(t["track_id"] == res["track_id"] for t in json.loads(capsys.readouterr().out))
    tl = tmp_path / "cues.json"
    tl.write_text(json.dumps({"duration": 20, "cues": [[4.3, "drop"], [12.9, "climax"]]}))
    assert cli_main(c + ["map", res["track_id"], "--timeline", str(tl), "--render", str(tmp_path / "p.wav")]) == 0
    assert (tmp_path / "p.wav").is_file()
    assert cli_main(c + ["blend", res["track_id"], res["track_id"], "--out", str(tmp_path / "spec.json")]) == 0
    assert json.loads((tmp_path / "spec.json").read_text())["kind"] == "audioref.target_spec"
    assert cli_main(c + ["demo", "waltz", str(tmp_path / "w.wav")]) == 0 and (tmp_path / "w.truth.json").is_file()
    assert cli_main(c + ["brief", "does-not-exist"]) == 1
    assert "unknown track" in capsys.readouterr().err


# ------------------------------------------------------------------ MCP
def test_mcp_in_process(cache_dir, house_standard):
    from mcp import Client

    from audioref import mcp_server

    res, _ = house_standard
    srv = mcp_server.build(cache=str(cache_dir))

    async def run():
        async with Client(srv) as c:
            names = {t.name for t in (await c.list_tools()).tools}
            assert {"analyze_track", "get_analysis", "get_brief", "get_drum_grid", "get_chords", "get_midi",
                    "compare_tracks", "blend_references", "map_to_timeline", "render_preview", "list_tracks",
                    "delete_track", "get_job", "status"} <= names
            r = await c.call_tool("analyze_track", {"path": res["analysis"]["track"]["path"], "depth": "standard"})
            d = json.loads(r.content[0].text)
            assert d["track_id"] == res["track_id"] and len(r.content[0].text) < 20000
            r = await c.call_tool("analyze_track", {"path": res["analysis"]["track"]["path"], "depth": "deep", "wait_s": 0})
            job = json.loads(r.content[0].text)
            assert job["status"] in ("queued", "running") and job["job_id"]
            r = await c.call_tool("get_job", {"job_id": job["job_id"], "wait_s": 600})
            done = json.loads(r.content[0].text)
            assert done["status"] == "done" or done.get("track_id")
            r = await c.call_tool("get_brief", {"track_id": res["track_id"], "focus": "harmony"})
            assert "## Harmony" in r.content[0].text
            r = await c.call_tool("get_brief", {"track_id": "zzzz"})
            assert r.is_error and "unknown track" in r.content[0].text
            r = await c.call_tool("analyze_track", {"path": "/no/such/file.mp3"})
            assert r.is_error and "file not found" in r.content[0].text
            rr = await c.read_resource(f"audioref://track/{res['track_id']}/brief")
            assert rr.contents[0].text.startswith("# Reference brief")
            rr = await c.read_resource(f"audioref://track/{res['track_id']}/analysis")
            assert json.loads(rr.contents[0].text)["track"]["id"] == res["track_id"]

    asyncio.run(run())


def test_mcp_stdio_subprocess(cache_dir):
    from mcp import Client
    from mcp.client.stdio import StdioServerParameters

    exe = Path(sys.executable).with_name("audioref-mcp")
    params = StdioServerParameters(command=str(exe), args=["--cache", str(cache_dir)], cwd=str(ROOT))

    async def run():
        async with Client(params) as c:
            tools = await c.list_tools()
            assert len(tools.tools) >= 14
            r = await c.call_tool("status", {})
            st = json.loads(r.content[0].text)
            assert st["cache_dir"] == str(cache_dir)

    asyncio.run(run())


@pytest.mark.skipif(not models.capabilities()["clap"], reason="transformers not installed")
def test_deep_depth_tags_and_plots(service, house_standard):
    res, _ = house_standard
    r = service.analyze(res["analysis"]["track"]["path"], depth="deep", trim=False)
    a = r["analysis"]
    assert a["generator"]["depth"] == "deep"
    assert a["sound"]["tags"]["method"] == "clap", a["warnings"]
    genre = a["sound"]["tags"]["genre"]
    assert genre and abs(sum(genre.values())) <= 1.0 and max(genre.values()) > 0.1  # a real, non-flat distribution
    assert a["sound"]["tags"]["instruments"]
    plots = {Path(p).stem for p in r["files"].get("plots", [])}
    assert {"spectrogram", "energy", "beat_grid", "drum_grid", "layers"} <= plots
