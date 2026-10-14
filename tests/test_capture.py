"""System-audio capture (macOS): gating/trimming via the helper's simulate mode, the menu bar app's control socket,
CLI/service/MCP wiring. Real system-audio capture needs the user's permission and is not run here."""
import asyncio
import json
import sys

import numpy as np
import pytest
import soundfile as sf

from audioref import capture as cap

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="macOS only")
needs_helper = pytest.mark.skipif(cap.find_helper() is None, reason="capture helper not built (audioref app build)")
needs_app = pytest.mark.skipif(cap.app_status() is None, reason="AudioRef menu bar app not running")


@pytest.fixture(scope="module")
def padded_clip(tmp_path_factory, demo_songs):
    """1.5 s silence + 8 s of the house demo (the source adds trailing silence like a track ending)."""
    y, sr = sf.read(str(demo_songs["house"][0]), dtype="float32")
    x = np.concatenate([np.zeros((int(1.5 * sr), 2), np.float32), y[int(4 * sr): int(12 * sr)]])
    p = tmp_path_factory.mktemp("cap") / "clip.wav"
    sf.write(p, x, sr, subtype="FLOAT")
    return p, x, sr


def _check_trim(path, x, sr):
    y, sr2 = sf.read(str(path), dtype="float32")
    assert sr2 == sr
    k = int(np.argmax(np.abs(y[:, 0]) > 1e-4))
    assert 0.25 <= k / sr <= 0.35  # ~0.3 s pre-roll before the first sound
    assert abs(len(y) / sr - (0.3 + 8.0 + 0.5)) < 0.1  # pre-roll + music + 0.5 s tail
    kin = int(np.argmax(np.abs(x[:, 0]) > 1e-4))
    off = kin - k
    n = min(len(y), len(x) - off)
    assert np.abs(y[:n] - x[off: off + n]).max() < 1e-6  # bit-exact


@needs_helper
def test_helper_simulate_gating(padded_clip, tmp_path):
    clip, x, sr = padded_clip
    events = []
    res = cap.capture(tmp_path / "out.wav", simulate=str(clip), on_event=lambda e: events.append(e["event"]), via="helper")
    assert res["via"] == "helper" and res["reason"] == "silence"
    assert events[0] == "waiting" and "started" in events and events[-1] == "finished"
    _check_trim(res["path"], x, sr)


@needs_helper
def test_helper_no_audio_and_permission_check(tmp_path):
    sf.write(tmp_path / "silent.wav", np.zeros((44100 * 3, 2), np.float32), 44100)
    with pytest.raises(cap.CaptureError, match="No audio detected"):
        cap.capture(tmp_path / "x.wav", simulate=str(tmp_path / "silent.wav"), wait_for_audio=1, via="helper")
    assert cap.helper_permission() in (True, False)


@needs_app
def test_app_socket_capture(padded_clip, tmp_path):
    clip, x, sr = padded_clip
    st = cap.app_status()
    assert st["ok"] and st["app"] == "AudioRef" and "permission" in st
    res = cap.capture(tmp_path / "app.wav", simulate=str(clip), via="app")
    assert res["via"] == "app"
    _check_trim(res["path"], x, sr)
    assert cap.app_status()["phase"] == "done"


@needs_helper
def test_service_capture_and_analyze(service, padded_clip):
    clip, _, _ = padded_clip
    r = service.capture(name="Captured clip", depth="quick", via="helper", simulate=str(clip))
    assert r["name"] == "Captured clip" and r["capture"]["reason"] == "silence"
    assert abs(r["summary"]["bpm"] - 112) < 1 and r["summary"]["key"]  # 8 s / 4 chords: key itself is ambiguous
    assert any(t["name"] == "Captured clip" for t in service.list_tracks())


def test_mcp_lists_capture_tools(cache_dir):
    from mcp import Client

    from audioref import mcp_server

    srv = mcp_server.build(cache=str(cache_dir))

    async def run():
        async with Client(srv) as c:
            tools = {t.name: t for t in (await c.list_tools()).tools}
            assert {"capture_audio", "stop_capture"} <= set(tools)
            assert "press play" in tools["capture_audio"].description
            r = await c.call_tool("stop_capture", {})
            assert json.loads(r.content[0].text)["stopped"] in (True, False)

    asyncio.run(run())
