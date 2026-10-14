"""`audioref` command-line interface (same core as the MCP server)."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path

from . import __version__
from .config import DEPTHS, load_config


def _svc(args):
    from .service import Service

    overrides = {"cache_dir": args.cache, "device": args.device}
    if getattr(args, "allow_download", False):
        overrides["allow_download"] = True
    if getattr(args, "separation", None):
        overrides["separation_model"] = args.separation
    cfg = load_config(args.config, **overrides)
    return Service(cfg)


def _print(obj, as_json: bool = True) -> None:
    if isinstance(obj, str) and not as_json:
        sys.stdout.write(obj if obj.endswith("\n") else obj + "\n")
    else:
        sys.stdout.write(json.dumps(obj, indent=2, default=str) + "\n")


def _progress_printer(quiet: bool):
    t0 = time.time()

    def cb(frac: float, msg: str) -> None:
        if not quiet:
            sys.stderr.write(f"\r  [{frac * 100:5.1f}%] {msg:<48} {time.time() - t0:6.1f}s")
            sys.stderr.flush()
            if frac >= 1.0:
                sys.stderr.write("\n")

    return cb


def _parse_bars(s: str | None) -> list[int] | None:
    if not s:
        return None
    out: list[int] = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part.strip():
            out.append(int(part))
    return out


# ------------------------------------------------------------------ commands
def cmd_analyze(args) -> int:
    svc = _svc(args)
    results = []
    for path in args.paths:
        sys.stderr.write(f"analyzing {path} ({args.depth or svc.cfg.default_depth})\n")
        r = svc.analyze(path, args.depth, args.start, args.end, args.stems, args.plots, args.force,
                        _progress_printer(args.quiet), trim=not args.full_json)
        results.extend(r if isinstance(r, list) else [r])
    if args.out:
        out_root = Path(args.out).expanduser()
        for r in results:
            dest = out_root / (r["name"] if len(results) > 1 or args.out_per_track else "")
            _export(svc, r["track_id"], dest)
            sys.stderr.write(f"wrote {dest}\n")
    if args.json:
        _print(results if len(results) > 1 else results[0])
    else:
        for r in results:
            brief = r.get("brief_md") or svc.get_brief(r["track_id"])
            _print(brief, as_json=False)
            sys.stdout.write(f"\n[track_id {r['track_id']}] files: {r['files'].get('json')}\n\n")
    return 0


def _export(svc, track_id: str, dest: Path) -> None:
    tc = svc.index.resolve(track_id)
    dest.mkdir(parents=True, exist_ok=True)
    for f in ("analysis.json", "brief.md"):
        if (tc.dir / f).is_file():
            shutil.copy2(tc.dir / f, dest / f)
    for d in ("midi", "stems", "plots"):
        if (tc.dir / d).is_dir():
            shutil.copytree(tc.dir / d, dest / d, dirs_exist_ok=True)


def cmd_brief(args) -> int:
    _print(_svc(args).get_brief(args.track, args.focus), as_json=False)
    return 0


def cmd_get(args) -> int:
    _print(_svc(args).get_analysis(args.track, args.fields or None))
    return 0


def cmd_grid(args) -> int:
    svc = _svc(args)
    g = svc.get_drum_grid(args.track, _parse_bars(args.bars), args.voices.split(",") if args.voices else None, args.resolution)
    if args.json:
        _print(g)
        return 0
    from .brief import LEGEND

    sys.stdout.write(f"swing: {g['swing']}\nmicrotiming (ms): {g['microtiming_ms']}\npatterns ({LEGEND}):\n")
    for p in g["patterns"]:
        if p.get("count", 0) < 2 and p.get("variation_of") is not None:
            continue
        sys.stdout.write(f"  {p['id']} ({p['count']} bars{', variation of ' + p['variation_of'] if p.get('variation_of') else ''})\n")
        for v, s in p.get("accents", {}).items():
            sys.stdout.write(f"    {v:<11} {s}\n")
    sys.stdout.write(f"bar patterns: {' '.join(g.get('bar_patterns') or [])}\n")
    for row in g["bars"]:
        sys.stdout.write(f"bar {row['bar']:3d} @ {row['start']:7.3f}s\n")
        for v, s in row["grid"].items():
            sys.stdout.write(f"    {v:<11} {s}\n")
    if g.get("note"):
        sys.stdout.write(g["note"] + "\n")
    return 0


def cmd_chords(args) -> int:
    c = _svc(args).get_chords(args.track, args.format)
    if args.json or args.format == "midi":
        _print(c)
        return 0
    loop = c.get("loop") or {}
    sys.stdout.write(f"key: {c['key']} ({c['key_confidence']})\nloop: {' - '.join(loop.get('symbols') or [])}  "
                     f"[{' - '.join(loop.get('roman') or [])}] ({loop.get('length_bars')} bars)\n")
    for ch in c["chords"]:
        sys.stdout.write(f"  {ch['start']:8.2f}-{ch['end']:8.2f}  bar {ch['bar']!s:>3}  {ch.get('roman') or ch['symbol']}\n")
    return 0


def cmd_midi(args) -> int:
    _print(_svc(args).get_midi(args.track, args.part))
    return 0


def cmd_compare(args) -> int:
    c = _svc(args).compare_tracks(args.tracks)
    if args.json:
        _print(c)
        return 0
    for p in c["pairwise"]:
        sys.stdout.write(f"{p['names'][0]} vs {p['names'][1]}: overall {p['overall']:.2f}\n")
        for k in c["dimensions"]:
            sys.stdout.write(f"  {k:<10} {p[k]:.2f} {'#' * int(round(p[k] * 20))}\n")
    sys.stdout.write("differences:\n" + "".join(f"  - {d}\n" for d in c["differences"]))
    return 0


def cmd_blend(args) -> int:
    ov = {k: getattr(args, k) for k in ("tempo", "groove", "drums", "harmony", "bass", "sound", "structure") if getattr(args, k)}
    w = [float(x) for x in args.weights.split(",")] if args.weights else None
    r = _svc(args).blend_references(args.tracks, w, ov)
    if args.out:
        Path(args.out).expanduser().write_text(json.dumps(r["target_spec"], indent=1))
        sys.stderr.write(f"wrote {args.out}\n")
    if args.json:
        _print(r)
    else:
        _print(r["brief_md"], as_json=False)
        sys.stdout.write(f"spec_id {r['spec_id']} ({r['path']})\n")
    return 0


def cmd_map(args) -> int:
    svc = _svc(args)
    cons = {}
    if args.bpm_range:
        lo, hi = args.bpm_range.split("-")
        cons["bpm_range"] = [float(lo), float(hi)]
    if args.integer_bpm:
        cons["integer_bpm"] = True
    if args.must_hit:
        cons["must_hit"] = args.must_hit.split(",")
    if args.tolerance_ms:
        cons["tolerance_ms"] = args.tolerance_ms
    tl = json.loads(Path(args.timeline).expanduser().read_text())
    plan = svc.map_to_timeline(args.target, tl, cons or None)
    if args.out:
        Path(args.out).expanduser().write_text(json.dumps(plan, indent=1))
        sys.stderr.write(f"wrote {args.out}\n")
    if args.render:
        r = svc.render_preview(plan["path"], args.render)
        sys.stderr.write(f"rendered {r['path']}\n")
    if args.json:
        _print(plan)
    else:
        _print(plan["brief_md"], as_json=False)
        sys.stdout.write(f"plan_id {plan['plan_id']} ({plan['path']})\n")
    return 0


def cmd_render(args) -> int:
    _print(_svc(args).render_preview(args.source, args.out, args.target))
    return 0


def cmd_list(args) -> int:
    rows = _svc(args).list_tracks()
    if args.json:
        _print(rows)
        return 0
    if not rows:
        sys.stdout.write("(no cached tracks)\n")
    for r in rows:
        sys.stdout.write(f"{r['track_id']}  {r['depth'] or '-':<8} {r['status'] or '-':<7} {r['summary'] or '':<26} {r['name']}\n")
    return 0


def cmd_delete(args) -> int:
    _print(_svc(args).delete_track(args.track))
    return 0


def cmd_status(args) -> int:
    _print(_svc(args).status())
    return 0


def cmd_demo(args) -> int:
    from .demo import SONGS, write_song

    if args.name not in SONGS:
        sys.stderr.write(f"unknown demo '{args.name}'; choose from {', '.join(SONGS)}\n")
        return 2
    truth = write_song(args.name, args.out, args.stems)
    sys.stderr.write(f"wrote {args.out} ({truth['duration']:.1f}s, {truth['bpm']} bpm, {truth['key']}); truth in "
                     f"{Path(args.out).with_suffix('.truth.json')}\n")
    return 0


def cmd_mcp(args) -> int:
    from .mcp_server import main as mcp_main

    argv = []
    if args.cache:
        argv += ["--cache", args.cache]
    if args.config:
        argv += ["--config", args.config]
    return mcp_main(argv)


# ------------------------------------------------------------------ parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="audioref", description="Reference audio analyzer: briefs, JSON, MIDI and stems from reference tracks.")
    p.add_argument("--version", action="version", version=f"audioref {__version__}")
    p.add_argument("--cache", help="cache directory (default ./.audioref or audioref.toml)")
    p.add_argument("--config", help="path to audioref.toml")
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], help="torch device for Demucs/CLAP")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--json", action="store_true", help="machine-readable JSON output")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true", default=argparse.SUPPRESS, help="machine-readable JSON output")
    sub = p.add_subparsers(dest="cmd", required=True)
    _add = sub.add_parser

    def add_parser(name, **kw):
        return _add(name, parents=[common], **kw)

    sub.add_parser = add_parser  # every subcommand also accepts --json after its name

    a = sub.add_parser("analyze", help="analyze one or more files/folders")
    a.add_argument("paths", nargs="+")
    a.add_argument("--depth", choices=DEPTHS)
    a.add_argument("--start", type=float, help="analyze from this time (s)")
    a.add_argument("--end", type=float, help="analyze until this time (s)")
    a.add_argument("--stems", action="store_true", help="export stems as WAV (local only)")
    a.add_argument("--plots", dest="plots", action="store_true", default=None, help="render PNG plots")
    a.add_argument("--no-plots", dest="plots", action="store_false")
    a.add_argument("--force", action="store_true", help="ignore cached results")
    a.add_argument("--out", help="also copy brief.md, analysis.json, midi/, stems/, plots/ here")
    a.add_argument("--out-per-track", action="store_true", help="always create a sub-folder per track under --out")
    a.add_argument("--full-json", action="store_true", help="with --json: untrimmed analysis")
    a.add_argument("--separation", choices=["htdemucs", "htdemucs_6s", "htdemucs_ft", "none"])
    a.add_argument("--allow-download", action="store_true", help="allow http(s) URLs (only audio you own/have rights to)")
    a.add_argument("-q", "--quiet", action="store_true")
    a.set_defaults(func=cmd_analyze)

    b = sub.add_parser("brief", help="print the Markdown brief")
    b.add_argument("track")
    b.add_argument("--focus", default="all", choices=["groove", "harmony", "sound", "structure", "all"])
    b.set_defaults(func=cmd_brief)

    g = sub.add_parser("get", help="slice analysis.json with JSONPath (e.g. rhythm.swing 'harmony.chords[0:8]')")
    g.add_argument("track")
    g.add_argument("fields", nargs="*")
    g.set_defaults(func=cmd_get)

    gr = sub.add_parser("grid", help="drum step grids")
    gr.add_argument("track")
    gr.add_argument("--bars", help="e.g. 0-7 or 4,5,12")
    gr.add_argument("--voices", help="comma list, e.g. kick,clap,hat_closed")
    gr.add_argument("--resolution", type=int, default=16, choices=[16, 32])
    gr.set_defaults(func=cmd_grid)

    c = sub.add_parser("chords", help="chord timeline")
    c.add_argument("track")
    c.add_argument("--format", default="symbols", choices=["symbols", "roman", "midi"])
    c.set_defaults(func=cmd_chords)

    m = sub.add_parser("midi", help="MIDI file path + note summary")
    m.add_argument("track")
    m.add_argument("--part", default="full", choices=["drums", "bass", "keys", "chords", "melody", "full"])
    m.set_defaults(func=cmd_midi)

    cp = sub.add_parser("compare", help="per-dimension similarity between analysed tracks")
    cp.add_argument("tracks", nargs="+")
    cp.set_defaults(func=cmd_compare)

    bl = sub.add_parser("blend", help="blend references into a target spec")
    bl.add_argument("tracks", nargs="+")
    bl.add_argument("--weights", help="comma list, one per track")
    for dim in ("tempo", "groove", "drums", "harmony", "bass", "sound", "structure"):
        bl.add_argument(f"--{dim}", help=f"take {dim} from this track (id/prefix/name/index)")
    bl.add_argument("--out", help="write the target spec JSON here")
    bl.set_defaults(func=cmd_blend)

    mp = sub.add_parser("map", help="map a target spec / track onto a video cue sheet")
    mp.add_argument("target", help="track id, spec_id, or spec JSON path")
    mp.add_argument("--timeline", required=True, help='cue sheet JSON: {"duration": s, "cues": [[t, kind, opts], ...]}')
    mp.add_argument("--bpm-range", help="e.g. 100-120")
    mp.add_argument("--integer-bpm", action="store_true")
    mp.add_argument("--must-hit", help="comma list of cue labels that must land on the grid")
    mp.add_argument("--tolerance-ms", type=float)
    mp.add_argument("--out", help="write the plan JSON here")
    mp.add_argument("--render", help="also render a rough preview WAV here")
    mp.set_defaults(func=cmd_map)

    r = sub.add_parser("render", help="render a MIDI file or score plan to WAV")
    r.add_argument("source", help="MIDI path, plan_id or plan JSON path")
    r.add_argument("--out")
    r.add_argument("--target", help="spec for a plan (track id, spec_id or path)")
    r.set_defaults(func=cmd_render)

    sub.add_parser("list", help="list cached tracks").set_defaults(func=cmd_list)
    d = sub.add_parser("delete", help="delete a cached track")
    d.add_argument("track")
    d.set_defaults(func=cmd_delete)
    sub.add_parser("status", help="config, capabilities, cache info").set_defaults(func=cmd_status)

    de = sub.add_parser("demo", help="write a ground-truth demo song (house, hiphop, waltz, techno, live)")
    de.add_argument("name")
    de.add_argument("out")
    de.add_argument("--stems", help="also write the true stems to this folder")
    de.set_defaults(func=cmd_demo)

    sub.add_parser("mcp", help="run the MCP server on stdio").set_defaults(func=cmd_mcp)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, stream=sys.stderr,
                        format="%(levelname)s %(name)s: %(message)s")
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        return 130
    except (KeyError, ValueError, FileNotFoundError, RuntimeError) as e:
        msg = e.args[0] if isinstance(e, KeyError) and e.args else str(e)
        sys.stderr.write(f"error: {msg}\n")
        if args.verbose:
            raise
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
