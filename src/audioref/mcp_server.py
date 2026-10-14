"""`audioref-mcp`: MCP server (stdio) exposing the analyzer to coding agents.

Long-running work runs in a background job queue; `analyze_track` waits up to `wait_s` seconds (streaming
progress notifications) and otherwise returns a job id to poll with `get_job`. Models stay loaded between calls.
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import json
import logging
import sys
import time
from typing import Any

from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ResourceError, ToolError
from mcp.types import ToolAnnotations

from . import __version__
from .config import load_config
from .capture import CaptureError
from .ingest import IngestError

INSTRUCTIONS = """\
audio-ref lets you "listen" to reference tracks. Typical flow:
0. No file? capture_audio(name) records what the user plays on this Mac (ask them to press play), then analyzes it.
1. analyze_track(path) for each reference (depth: quick ~tempo/key/chords/energy, standard = + stems, drums, bass,
   transcription, MIDI [default], deep = + CLAP tags + plots). Read the returned brief_md first.
2. Drill down with get_drum_grid / get_chords / get_midi / get_analysis(fields=[JSONPath...]).
3. compare_tracks / blend_references(track_ids, dimension_overrides={"groove": A, "harmony": B, "sound": C}).
4. map_to_timeline(target, timeline={"duration": s, "cues": [[t, kind, opts], ...]}) to get BPM, bar offset,
   section plan, builds/stops/fills/climax and per-cue sync for a video. Cue kinds: scene, hit, whoosh, build,
   drop, hold/stop, climax, end.
5. Optionally render_preview(plan_id or MIDI path) and re-analyze your own render to compare against the refs.
Paths are local files only (no URLs unless the server was started with allow_download). Bar numbers in JSON are
0-based; briefs use 1-based bars."""

log = logging.getLogger("audioref.mcp")
server = MCPServer(name="audio-ref", instructions=INSTRUCTIONS, version=__version__)
_svc = None
RO = ToolAnnotations(read_only_hint=True, idempotent_hint=True, open_world_hint=False)


def svc():
    global _svc
    if _svc is None:
        from .service import Service

        _svc = Service(load_config())
    return _svc


def _tool_errors(fn):
    """Turn expected failures into ToolError so the agent sees an actionable message."""

    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        try:
            return await fn(*args, **kwargs)
        except ToolError:
            raise
        except KeyError as e:
            raise ToolError(e.args[0] if e.args else str(e)) from e
        except (ValueError, FileNotFoundError, IngestError, CaptureError, RuntimeError) as e:
            raise ToolError(str(e)) from e

    return wrapper


async def _run(fn, *args, **kwargs):
    return await asyncio.to_thread(fn, *args, **kwargs)


async def _wait_job(job_id: str, wait_s: float, ctx: Context | None) -> dict:
    s = svc()
    t0 = time.time()
    last = -1.0
    while True:
        job = s.jobs.get(job_id)
        if job.status in ("done", "error"):
            break
        if ctx is not None and job.progress != last:
            last = job.progress
            try:
                await ctx.report_progress(round(job.progress * 100, 1), 100, job.message or job.status)
            except Exception:  # client may not support progress
                pass
        if time.time() - t0 >= wait_s:
            return {"job_id": job_id, "status": job.status, "progress": round(job.progress, 3), "message": job.message,
                    "hint": "still running: call get_job(job_id, wait_s=...) to wait for the result"}
        await asyncio.sleep(0.4)
    if job.status == "error":
        raise ToolError(f"{job.kind} failed: {job.error}")
    return job.result


# ---------------------------------------------------------------------------- tools
@server.tool(description="Analyze a local audio file (mp3/wav/flac/m4a/aiff/ogg) or a folder of them. Returns the Markdown "
             "brief, a trimmed analysis and file paths (JSON, MIDI, stems, plots). depth: quick | standard | deep. "
             "start/end (seconds) analyze only a section. stems=true exports stem WAVs (local only). Waits up to wait_s "
             "seconds, then returns a job_id to poll with get_job.")
@_tool_errors
async def analyze_track(path: str, depth: str = "standard", start: float | None = None, end: float | None = None,
                        stems: bool = False, plots: bool | None = None, force: bool = False, wait_s: float = 50.0,
                        ctx: Context | None = None) -> dict:
    s = svc()
    job_id = s.submit_analyze(path=path, depth=depth, start=start, end=end, stems=stems, plots=plots, force=force)
    return await _wait_job(job_id, max(0.0, float(wait_s)), ctx)


@server.tool(description="Record whatever is playing on this Mac (system audio) and analyze it: no downloading needed. "
             "Right after calling, ask the user to press play on their reference (Spotify, YouTube, Music, ...). "
             "Recording starts when sound starts and stops after stop_on_silence_s seconds of silence or max_seconds; "
             "the WAV is then analyzed like analyze_track (name labels it). Uses the AudioRef menu bar app if installed "
             "(it needs macOS Screen & System Audio Recording permission). Waits up to wait_s, then returns a job_id "
             "for get_job; stop_capture ends the recording early.")
@_tool_errors
async def capture_audio(name: str | None = None, max_seconds: float = 600, stop_on_silence_s: float = 3.0,
                        wait_for_audio_s: float = 90, depth: str = "standard", analyze: bool = True,
                        wait_s: float = 50.0, ctx: Context | None = None) -> dict:
    s = svc()
    job_id = s.submit_capture(name=name, max_seconds=max_seconds, stop_on_silence=stop_on_silence_s,
                              wait_for_audio=wait_for_audio_s, depth=depth, analyze=analyze)
    res = await _wait_job(job_id, max(0.0, float(wait_s)), ctx)
    if "job_id" in res:
        res["hint"] = ("capturing: make sure the user pressed play; call get_job(job_id, wait_s=...) for the result "
                       "or stop_capture() to end the recording now")
    return res


@server.tool(description="Stop the system-audio capture in progress (what was recorded so far is kept and analyzed).")
@_tool_errors
async def stop_capture() -> dict:
    from . import capture as cap

    return {"stopped": await _run(cap.stop)}


@server.tool(description="Status/result of a background job (from analyze_track). wait_s > 0 waits for completion.",
             annotations=RO)
@_tool_errors
async def get_job(job_id: str, wait_s: float = 0.0, ctx: Context | None = None) -> dict:
    s = svc()
    job = s.jobs.get(job_id)
    if job.status in ("done", "error") or wait_s <= 0:
        d = job.to_dict(include_result=True)
        if job.status == "error":
            d["hint"] = "fix the input and call analyze_track again"
        return d
    res = await _wait_job(job_id, float(wait_s), ctx)
    return res if "job_id" in res else {"job_id": job_id, "status": "done", "result": res}


@server.tool(description="Slices of analysis.json by JSONPath, e.g. ['rhythm.swing', 'harmony.chords[0:8]', "
             "'structure.sections[*].label', \"drums.patterns[?(@.id=='A')]\"]. No fields = trimmed overview.",
             annotations=RO)
@_tool_errors
async def get_analysis(track_id: str, fields: list[str] | None = None) -> dict:
    return await _run(svc().get_analysis, track_id, fields)


@server.tool(description="The Markdown brief for a track; focus: groove | harmony | sound | structure | all.",
             annotations=RO)
@_tool_errors
async def get_brief(track_id: str, focus: str = "all") -> str:
    return await _run(svc().get_brief, track_id, focus)


@server.tool(description="Drum step grids: per-bar grids + velocities + microtiming offsets per voice, labelled patterns "
             "(A, A', B...), fills, stops, rolls and the groove template. bars = 0-based bar indices; voices e.g. "
             "['kick','clap','hat_closed']; resolution 16 (swing-aware) or 32.", annotations=RO)
@_tool_errors
async def get_drum_grid(track_id: str, bars: list[int] | None = None, voices: list[str] | None = None,
                        resolution: int = 16) -> dict:
    return await _run(svc().get_drum_grid, track_id, bars, voices, resolution)


@server.tool(description="Chord timeline (format: symbols | roman | midi), loop summary, harmonic rhythm and voicing notes.",
             annotations=RO)
@_tool_errors
async def get_chords(track_id: str, format: str = "symbols") -> dict:
    return await _run(svc().get_chords, track_id, format)


@server.tool(description="MIDI file path + note summary (counts, range, density). part: drums | bass | keys | chords | "
             "melody | full. Notes are at absolute audio times (bar 1 = rhythm.first_downbeat).", annotations=RO)
@_tool_errors
async def get_midi(track_id: str, part: str = "full") -> dict:
    return await _run(svc().get_midi, track_id, part)


@server.tool(description="Per-dimension similarity (tempo, groove, drums, harmony, bass, sound, structure) between analysed "
             "tracks plus notable differences.", annotations=RO)
@_tool_errors
async def compare_tracks(track_ids: list[str]) -> dict:
    return await _run(svc().compare_tracks, track_ids)


@server.tool(description="Blend references into a target spec: weighted averages for numbers, exemplars for categorical "
             "parts. dimension_overrides maps tempo|groove|drums|harmony|bass|sound|structure -> track id/name/index, e.g. "
             "{'groove': 'a', 'harmony': 'b', 'sound': 'c'}. Returns spec_id (reusable in map_to_timeline) + brief.")
@_tool_errors
async def blend_references(track_ids: list[str], weights: list[float] | None = None,
                           dimension_overrides: dict[str, str] | None = None) -> dict:
    return await _run(svc().blend_references, track_ids, weights, dimension_overrides)


@server.tool(description="Map a target (spec_id, target_spec object, spec JSON path, or a track id) onto a video timeline "
             "{duration, cues: [[t, kind, opts], ...]} (kinds: scene, hit, whoosh, build, drop, hold/stop, climax, end; "
             "opts: label, weight, must_hit, level bar|beat|half|8th, duration). constraints: bpm_range [lo, hi], "
             "integer_bpm, must_hit [labels], tolerance_ms. Returns BPM + offset solving cue-to-beat error, section plan "
             "with layers/energy, fills/stops/builds/climax, a bar map and a per-cue sync report.")
@_tool_errors
async def map_to_timeline(target: Any, timeline: dict, constraints: dict | None = None) -> dict:
    return await _run(svc().map_to_timeline, target, timeline, constraints)


@server.tool(description="Render a quick preview WAV from a MIDI path or a score plan (plan_id / plan path) for sanity "
             "checks; re-analyze the WAV with analyze_track and compare_tracks to close the loop.")
@_tool_errors
async def render_preview(source: str, out: str | None = None, target: str | None = None) -> dict:
    return await _run(svc().render_preview, source, out, target)


@server.tool(description="List cached tracks (id, name, depth, summary).", annotations=RO)
@_tool_errors
async def list_tracks() -> list[dict]:
    return await _run(svc().list_tracks)


@server.tool(description="Delete a cached track (analysis, MIDI, stems, plots).",
             annotations=ToolAnnotations(destructive_hint=True, idempotent_hint=True))
@_tool_errors
async def delete_track(track_id: str) -> dict:
    return await _run(svc().delete_track, track_id)


@server.tool(description="Server status: cache dir, config, installed model backends, models loaded in memory.",
             annotations=RO)
@_tool_errors
async def status() -> dict:
    return await _run(svc().status)


# ---------------------------------------------------------------------------- resources
@server.resource("audioref://tracks", mime_type="application/json", description="Cached tracks")
def res_tracks() -> str:
    return json.dumps(svc().list_tracks(), indent=1)


@server.resource("audioref://track/{track_id}/brief", mime_type="text/markdown", description="Markdown brief")
def res_brief(track_id: str) -> str:
    try:
        return svc().get_brief(track_id)
    except KeyError as e:
        raise ResourceError(str(e)) from e


@server.resource("audioref://track/{track_id}/analysis", mime_type="application/json", description="Full analysis.json")
def res_analysis(track_id: str) -> str:
    try:
        tc = svc()._tc(track_id)
    except KeyError as e:
        raise ResourceError(str(e)) from e
    return tc.analysis_path.read_text()


@server.resource("audioref://track/{track_id}/plots/{name}", mime_type="image/png", description="Plot PNG (spectrogram, "
                 "chromagram, energy, beat_grid, drum_grid, layers); analyze with depth=deep or plots=true first")
def res_plot(track_id: str, name: str) -> bytes:
    try:
        tc = svc()._tc(track_id)
    except KeyError as e:
        raise ResourceError(str(e)) from e
    p = tc.plots_dir / (name if name.endswith(".png") else f"{name}.png")
    if not p.is_file():
        raise ResourceError(f"no plot '{name}' for {track_id} (available: {[f.stem for f in tc.plots_dir.glob('*.png')]})")
    return p.read_bytes()


# ---------------------------------------------------------------------------- entry
def build(cache: str | None = None, config: str | None = None, **overrides: Any):
    """(Re)configure the shared service; returns the MCPServer (used by tests for in-process clients)."""
    global _svc
    from .service import Service

    _svc = Service(load_config(config, cache_dir=cache, **{k: v for k, v in overrides.items() if v is not None}))
    return server


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="audioref-mcp", description="audio-ref MCP server (stdio)")
    ap.add_argument("--cache", help="cache directory (default ./.audioref)")
    ap.add_argument("--config", help="audioref.toml path")
    ap.add_argument("--allow-download", action="store_true", help="allow http(s) audio URLs (audio you own/have rights to)")
    ap.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"])
    ap.add_argument("--preload", action="store_true", help="load models at startup instead of on first use")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, stream=sys.stderr,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    build(args.cache, args.config, allow_download=True if args.allow_download else None, device=args.device)
    if args.preload:
        _preload()
    server.run("stdio")
    return 0


def _preload() -> None:  # pragma: no cover - warms model caches
    from . import models, rhythm, separation

    caps = models.capabilities()
    if caps["demucs"]:
        models.get("demucs:htdemucs", lambda: separation._load_demucs("htdemucs", "auto"))
    if caps["beat_this"]:
        import numpy as np

        rhythm._beat_this(np.zeros(22050 * 3, dtype=np.float32), 22050, "auto")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
