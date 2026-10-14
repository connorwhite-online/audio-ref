"""High-level API shared by the CLI and the MCP server."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable

from . import compare as compare_mod
from . import groove as groove_mod
from . import ingest, models
from . import midi as midi_mod
from . import render as render_mod
from . import timeline as timeline_mod
from .brief import FOCI, make_brief
from .cache import CacheIndex, TrackCache
from .config import DEPTHS, Config, load_config
from .jobs import JobManager
from .jsonpath import select
from .pipeline import Pipeline, list_files
from .util import read_json, to_jsonable, write_json

MAX_CHARS = 16000  # ~4k tokens for default tool results
MIDI_PARTS = ("drums", "bass", "keys", "chords", "melody", "full")


class Service:
    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or load_config()
        self.root = Path(self.cfg.cache_dir).expanduser()
        self.pipeline = Pipeline(self.cfg)
        self.index = CacheIndex(self.root)
        self.jobs = JobManager(self.cfg.max_workers)

    # ------------------------------------------------------------------ helpers
    def _tc(self, track: str) -> TrackCache:
        tc = self.index.resolve(track)
        if not tc.analysis_path.is_file():
            raise KeyError(f"track '{track}' has no completed analysis yet (status: {tc.read_meta().get('status')})")
        return tc

    def analysis(self, track: str) -> dict:
        return read_json(self._tc(track).analysis_path)

    # ------------------------------------------------------------------ analyze
    def analyze(self, path: str, depth: str | None = None, start: float | None = None, end: float | None = None,
                stems: bool = False, plots: bool | None = None, force: bool = False,
                progress: Callable[[float, str], None] | None = None, trim: bool = True) -> dict | list[dict]:
        """Analyze a file (or every audio file in a folder)."""
        p = Path(str(path)).expanduser()
        if not ingest.is_url(str(path)) and p.is_dir():
            files = ingest.list_audio_files(p)
            if not files:
                raise ingest.IngestError(f"no audio files in {p}")
            out = []
            for i, f in enumerate(files):
                sub = (lambda fr, msg, i=i: progress((i + fr) / len(files), f"[{i + 1}/{len(files)}] {f.name}: {msg}")) if progress else None
                r = self.pipeline.analyze(f, depth, start, end, stems, plots, force, sub)
                out.append(self._result(r, trim=True, brief=False))
            return out
        r = self.pipeline.analyze(str(path), depth, start, end, stems, plots, force, progress)
        return self._result(r, trim=trim)

    def _result(self, r: dict, trim: bool = True, brief: bool = True) -> dict:
        tc: TrackCache = r["cache"]
        out = {"track_id": r["track_id"], "name": r["analysis"]["track"]["name"], "files": r["files"]}
        if brief:
            out["brief_md"] = tc.brief_path.read_text() if tc.brief_path.is_file() else ""
        if trim:
            # the brief already carries the detail; keep the tool result inside the ~4k-token budget
            out["summary"] = summarize_analysis(r["analysis"])
            out["more"] = ("get_analysis(track_id) for the trimmed analysis, get_analysis(track_id, fields=[JSONPath]) for "
                           "slices, get_drum_grid / get_chords / get_midi for parts, or resource audioref://track/<id>/analysis")
        else:
            out["analysis"] = r["analysis"]
        if not brief:
            out["summary_line"] = tc.read_meta().get("summary")
        return out

    def submit_analyze(self, **kw: Any) -> str:
        params = {k: v for k, v in kw.items() if k != "progress"}
        return self.jobs.submit("analyze_track", lambda progress, **k: self.analyze(progress=progress, **k), params, **kw)

    # ------------------------------------------------------------------ getters
    def get_analysis(self, track: str, fields: list[str] | None = None) -> dict:
        a = self.analysis(track)
        if not fields:
            return trim_analysis(a)
        return select(a, fields)

    def get_brief(self, track: str, focus: str = "all") -> str:
        if focus not in FOCI:
            raise ValueError(f"focus must be one of {FOCI}")
        a = self.analysis(track)
        return make_brief(a, focus)

    def get_drum_grid(self, track: str, bars: list[int] | None = None, voices: list[str] | None = None,
                      resolution: int = 16) -> dict:
        if resolution not in (16, 32):
            raise ValueError("resolution must be 16 or 32")
        tc = self._tc(track)
        a = read_json(tc.analysis_path)
        hits = read_json(tc.dir / "hits.json") if (tc.dir / "hits.json").is_file() else {}
        r = a["rhythm"]
        swing = {k: (r.get("swing") or {}).get(k) for k in ("level", "ratio_16th", "ratio_8th")}
        meter = r.get("beats_per_bar") or 4
        bar_set = set(bars) if bars else None
        rows = groove_mod.grid_for_bars(hits, r.get("beats") or [], r.get("downbeats") or [], meter, swing, resolution,
                                        bar_set, voices)
        pats = a["drums"].get("patterns", [])
        if voices:
            pats = [{**p, "grid_16": {v: g for v, g in p.get("grid_16", {}).items() if v in voices},
                     "accents": {v: g for v, g in p.get("accents", {}).items() if v in voices},
                     "velocity": {v: g for v, g in p.get("velocity", {}).items() if v in voices}} for p in pats]
        out = {"track_id": tc.track_id, "resolution": resolution, "steps_per_bar": meter * (4 if resolution == 16 else 8),
               "meter": r.get("meter"), "swing": r.get("swing"), "microtiming_ms": r.get("microtiming_ms"),
               "legend": "grid: x = hit, . = rest (bar numbers are 0-based); accents: X accent, x hit, o ghost; "
                         "offset_ms is vs the straight grid (swing shows as positive offsets on off-16ths)",
               "patterns": pats, "bar_patterns": a["drums"].get("bar_patterns"), "fills": a["drums"].get("fills"),
               "stops": a["drums"].get("stops"), "rolls": a["drums"].get("rolls"),
               "groove_template_16": r.get("groove_template_16") if resolution == 16 else None}
        if bars is None and len(rows) > 16:
            out["bars"] = rows[:16]
            out["note"] = f"showing bars 0-15 of {len(rows)}; pass bars=[...] for others"
        else:
            out["bars"] = rows
        return out

    def get_chords(self, track: str, fmt: str = "symbols") -> dict:
        if fmt not in ("symbols", "roman", "midi"):
            raise ValueError("format must be symbols | roman | midi")
        tc = self._tc(track)
        a = read_json(tc.analysis_path)
        h = a["harmony"]
        downs = a["rhythm"].get("downbeats") or []
        import numpy as np

        d = np.asarray(downs)
        timeline = []
        for c in h.get("chords", []):
            bar = int(np.searchsorted(d, c["start"] + 1e-3, side="right") - 1) if len(d) else None
            item = {"start": c["start"], "end": c["end"], "bar": bar, "beats": c.get("beats")}
            if fmt == "roman":
                item["roman"] = c.get("roman")
                item["symbol"] = c.get("symbol")
            else:
                item["symbol"] = c.get("symbol")
            if c.get("confidence") is not None:
                item["confidence"] = c["confidence"]
            timeline.append(item)
        out = {"track_id": tc.track_id, "key": h.get("key"), "key_confidence": h.get("key_confidence"),
               "loop": h.get("loop"), "harmonic_rhythm": h.get("harmonic_rhythm"), "vocabulary": h.get("vocabulary"),
               "voicing": h.get("voicing"), "bar_chords": h.get("bar_chords"), "uncertain_spans": h.get("uncertain_spans")}
        if fmt == "midi":
            p = tc.midi_dir / "chords.mid"
            out["midi"] = midi_mod.summarize(p) if p.is_file() else None
            out["path"] = str(p) if p.is_file() else None
        else:
            out["chords"] = timeline
        return out

    def get_midi(self, track: str, part: str = "full") -> dict:
        if part not in MIDI_PARTS:
            raise ValueError(f"part must be one of {MIDI_PARTS}")
        tc = self._tc(track)
        p = tc.midi_dir / f"{part}.mid"
        if not p.is_file() and part == "keys":
            p = tc.midi_dir / "chords.mid"
        if not p.is_file():
            raise FileNotFoundError(f"no {part}.mid for this track (part not detected or depth too low)")
        a = read_json(tc.analysis_path)
        return {"track_id": tc.track_id, "part": part, "path": str(p), "summary": midi_mod.summarize(p),
                "timing": {"bpm": a["rhythm"].get("bpm"), "first_downbeat_s": a["rhythm"].get("first_downbeat"),
                           "note": "notes are at absolute audio times; bar 1 starts at first_downbeat_s"}}

    # ------------------------------------------------------------------ multi-reference
    def compare_tracks(self, track_ids: list[str]) -> dict:
        if len(track_ids) < 2:
            raise ValueError("compare_tracks needs at least two track ids")
        return to_jsonable(compare_mod.compare([self.analysis(t) for t in track_ids]), 3)

    def blend_references(self, track_ids: list[str], weights: list[float] | None = None,
                         dimension_overrides: dict[str, str] | None = None) -> dict:
        analyses = [self.analysis(t) for t in track_ids]
        ids = [a["track"]["id"] for a in analyses]
        # allow overrides by the same refs the caller used for track_ids
        ov = {}
        for k, v in (dimension_overrides or {}).items():
            ref = str(v)
            if ref in track_ids:
                ref = ids[track_ids.index(ref)]
            ov[k] = ref
        spec = to_jsonable(compare_mod.blend(analyses, weights, ov), 4)
        sid = _hash({"ids": ids, "w": weights, "ov": ov})
        path = self.root / "blends" / f"{sid}.json"
        write_json(path, spec)
        return {"spec_id": sid, "path": str(path), "brief_md": spec["brief_md"], "target_spec": spec}

    def _load_spec(self, target: Any) -> dict:
        if isinstance(target, dict):
            if target.get("kind") == "audioref.target_spec" or "tempo" in target:
                return target
            if "target_spec" in target:
                return target["target_spec"]
        if isinstance(target, str):
            p = Path(target).expanduser()
            if p.is_file():
                return read_json(p)
            bp = self.root / "blends" / f"{target}.json"
            if bp.is_file():
                return read_json(bp)
            return compare_mod.spec_from_analysis(self.analysis(target))
        raise ValueError("target must be a target_spec object, a spec_id/path, or a track id")

    def map_to_timeline(self, target: Any, timeline: dict | str, constraints: dict | None = None) -> dict:
        spec = self._load_spec(target)
        if isinstance(timeline, str):
            timeline = read_json(Path(timeline).expanduser())
        plan = to_jsonable(timeline_mod.plan(spec, timeline, constraints), 4)
        pid = _hash({"spec": spec.get("sources"), "tempo": spec.get("tempo"), "tl": timeline, "c": constraints})
        path = self.root / "plans" / f"{pid}.json"
        write_json(path, {"plan": plan, "spec": spec})
        plan["plan_id"] = pid
        plan["path"] = str(path)
        return plan

    def render_preview(self, source: Any, out: str | None = None, target: Any = None) -> dict:
        """source: a MIDI path, a score plan (dict / plan_id / plan path)."""
        prev = self.root / "previews"
        prev.mkdir(parents=True, exist_ok=True)
        if isinstance(source, str) and source.lower().endswith((".mid", ".midi")):
            p = Path(source).expanduser()
            if not p.is_file():
                raise FileNotFoundError(str(p))
            o = Path(out).expanduser() if out else prev / f"{p.stem}-{_hash(str(p.resolve()))[:6]}.wav"
            return render_mod.render_midi(p, o)
        plan, spec = None, None
        if isinstance(source, dict):
            plan = source.get("plan", source)
            spec = source.get("spec")
        elif isinstance(source, str):
            p = Path(source).expanduser()
            if not p.is_file():
                p = self.root / "plans" / f"{source}.json"
            if not p.is_file():
                raise FileNotFoundError(f"no MIDI file, plan file or plan_id '{source}'")
            d = read_json(p)
            plan, spec = d.get("plan", d), d.get("spec")
        if target is not None:
            spec = self._load_spec(target)
        if not plan or plan.get("kind") != "audioref.score_plan":
            raise ValueError("source is not a score plan")
        if spec is None:
            raise ValueError("rendering a plan needs its target spec (pass target=...)")
        o = Path(out).expanduser() if out else prev / f"plan-{plan.get('plan_id') or _hash(plan.get('cue_sync'))}.wav"
        return render_mod.render_plan(plan, spec, o)

    # ------------------------------------------------------------------ cache management
    def list_tracks(self) -> list[dict]:
        out = []
        for tc in self.index.tracks():
            m = tc.read_meta()
            out.append({"track_id": tc.track_id, "name": m.get("name"), "path": m.get("path"), "duration_s": m.get("duration_s"),
                        "depth": m.get("depth"), "status": m.get("status"), "summary": m.get("summary"),
                        "start": m.get("start"), "end": m.get("end"),
                        "updated": time.strftime("%Y-%m-%d %H:%M", time.localtime(m.get("updated", m.get("created", 0))))})
        return out

    def delete_track(self, track: str) -> dict:
        tc = self.index.resolve(track)
        size = sum(f.stat().st_size for f in tc.dir.rglob("*") if f.is_file())
        tc.delete()
        self.index.forget(tc.track_id)
        return {"deleted": tc.track_id, "freed_mb": round(size / 1e6, 1)}

    def status(self) -> dict:
        return {"cache_dir": str(self.root), "config": self.cfg.to_dict(), "capabilities": models.capabilities(),
                "loaded_models": models.loaded(), "depths": DEPTHS, "tracks": len(self.index.tracks()),
                "ffmpeg": ingest.have_ffmpeg()}

    def files(self, track: str) -> dict:
        return list_files(self._tc(track))


def summarize_analysis(a: dict) -> dict:
    """The handful of fields an agent needs first (everything else is in the brief or get_analysis)."""
    r, h, st = a["rhythm"], a["harmony"], a["structure"]
    sw = r.get("swing") or {}
    return {
        "duration_s": a["track"].get("duration_s"), "loudness_lufs": (a["track"].get("loudness") or {}).get("integrated_lufs"),
        "bpm": r.get("bpm"), "meter": r.get("meter"), "first_downbeat_s": r.get("first_downbeat"),
        "swing": {k: sw.get(k) for k in ("level", "percent", "feel")}, "microtiming_ms": r.get("microtiming_ms"),
        "key": h.get("key"), "key_confidence": h.get("key_confidence"),
        "loop": {k: (h.get("loop") or {}).get(k) for k in ("symbols", "roman", "length_bars")},
        "sections": [f"{s['label']} {s['start_bar'] + 1}-{s['end_bar']}" for s in st.get("sections", [])],
        "events": [e["type"] + "@bar" + str((e.get("bar") if e.get("bar") is not None else e.get("start_bar", -1)) + 1)
                   for e in st.get("events", [])][:12],
        "tags": {k: (a["sound"].get("tags") or {}).get(k) for k in ("genre", "instruments", "mood", "method")},
        "methods": a["generator"].get("methods"), "depth": a["generator"].get("depth"),
        "confidence": {k: v.get("level") for k, v in (a.get("confidence") or {}).items()},
        "warnings": (a.get("warnings") or [])[:6],
    }


def _hash(obj: Any) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]


# ---------------------------------------------------------------------- trimming
def _summ(arr: list, n: int = 4) -> dict:
    return {"count": len(arr), "first": arr[:n], "last": arr[-1] if arr else None}


def trim_analysis(a: dict, max_chars: int = MAX_CHARS) -> dict:
    """Compact view for tool results; big arrays become summaries (use get_analysis with JSONPath for the rest)."""
    r, d, h, s, st = a["rhythm"], a["drums"], a["harmony"], a["sound"], a["structure"]
    pats = [p for p in d.get("patterns", []) if p.get("count", 0) >= 2 or p.get("variation_of") is None]
    main_first = sorted(pats, key=lambda p: -p["count"])[:3]
    t = {
        "track": a["track"],
        "rhythm": {**{k: r.get(k) for k in ("bpm", "meter", "beats_per_bar", "first_downbeat", "swing", "microtiming_ms",
                                              "tempo_stability", "tempo_candidates", "confidence", "grid_anchor")},
                   "beats": _summ(r.get("beats") or []), "downbeats": _summ(r.get("downbeats") or [])},
        "drums": {"voices": d.get("voices"), "hit_counts": d.get("hit_counts"),
                  "patterns": [{k: p.get(k) for k in ("id", "count", "variation_of", "accents")} for p in main_first],
                  "bar_patterns": " ".join(d.get("bar_patterns") or []), "fills": (d.get("fills") or [])[:6],
                  "stops": d.get("stops"), "rolls": d.get("rolls"),
                  "sound": {v: {k: x for k, x in dd.items() if k in ("decay_ms", "f0_hz", "start_hz", "brightness_hz", "click",
                                                                     "saturation", "openness")} for v, dd in (d.get("sound") or {}).items()}},
        "harmony": {**{k: h.get(k) for k in ("key", "key_confidence", "loop", "harmonic_rhythm", "vocabulary", "extensions")},
                    "voicing": {k: v for k, v in (h.get("voicing") or {}).items() if k not in ("comping_hit_rate",)},
                    "chords": _summ([f"{c['symbol']}@{c['start']:.2f}" for c in h.get("chords", [])], 8)},
        "bass": {k: v for k, v in (a.get("bass") or {}).items() if k not in ("rhythm_hit_rate",)},
        "melody": a.get("melody"),
        "sound": {"tags": s.get("tags"), "mix": {k: v for k, v in (s.get("mix") or {}).items() if k != "bands"},
                  "sidechain": s.get("sidechain"), "synthesis_hints": s.get("synthesis_hints")},
        "structure": {"sections": [{k: x.get(k) for k in ("label", "group", "start_bar", "end_bar", "start", "end", "energy", "layers")}
                                   for x in st.get("sections", [])], "events": st.get("events")},
        "confidence": a.get("confidence"), "warnings": a.get("warnings"), "generator": a.get("generator"),
        "more": "full arrays (beats, chords, energy curves, groove templates, layer chart) via get_analysis(fields=[...]) "
                "or the resource audioref://track/<id>/analysis",
    }
    if len(json.dumps(t, default=str)) > max_chars:
        t["drums"]["patterns"] = t["drums"]["patterns"][:1]
        t["structure"]["events"] = (t["structure"]["events"] or [])[:10]
        t["rhythm"].pop("tempo_candidates", None)
    if len(json.dumps(t, default=str)) > max_chars:
        t["structure"]["sections"] = t["structure"]["sections"][:12]
        t["drums"]["fills"] = t["drums"]["fills"][:3]
        t["sound"]["synthesis_hints"] = {k: v[:160] for k, v in (t["sound"]["synthesis_hints"] or {}).items()}
    return t
