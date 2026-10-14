"""Analysis pipeline: runs the stages for a depth, caches each one, assembles analysis.json + artifacts."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import numpy as np
import soxr

from . import __version__, drums, groove, harmony, ingest, models, parts, rhythm, separation, sound, structure, transcription
from .cache import CacheIndex, TrackCache
from .config import DEPTHS, Config
from .util import log, to_jsonable, write_json

SCHEMA_VERSION = "1.0"
ASSEMBLY_VERSION = "2"


def pipeline_version() -> str:
    """Changes whenever any stage version changes, invalidating the assembled analysis (stage caches with
    unchanged versions are still reused)."""
    from . import brief, midi, plots  # noqa: F401

    versions = [SCHEMA_VERSION, ASSEMBLY_VERSION, rhythm.VERSION, drums.VERSION, groove.VERSION, harmony.VERSION,
                transcription.VERSION, parts.VERSION, sound.VERSION, structure.VERSION]
    return ".".join(versions)
SR22 = 22050
DEPTH_ORDER = {d: i for i, d in enumerate(DEPTHS)}

Progress = Callable[[float, str], None]


def _noop(p: float, msg: str) -> None:  # pragma: no cover - trivial
    pass


def _r22(x: np.ndarray) -> np.ndarray:
    """(2, n) or (n,) @ 44.1k -> mono float32 @ 22.05k."""
    m = x.mean(axis=0) if x.ndim == 2 else x
    return soxr.resample(np.ascontiguousarray(m, dtype=np.float32), ingest.SR, SR22).astype(np.float32)


class Pipeline:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.root = Path(cfg.cache_dir).expanduser()
        self.index = CacheIndex(self.root)

    # ------------------------------------------------------------------ public
    def analyze(self, path: str | Path, depth: str | None = None, start: float | None = None, end: float | None = None,
                stems: bool = False, plots: bool | None = None, force: bool = False,
                progress: Progress | None = None) -> dict:
        """Analyze a file. Returns {'track_id', 'cache': TrackCache, 'analysis': dict, 'files': dict}."""
        depth = depth or self.cfg.default_depth
        if depth not in DEPTHS:
            raise ValueError(f"depth must be one of {DEPTHS}")
        progress = progress or _noop
        src = str(path)
        if ingest.is_url(src):
            src = str(ingest.download(src, self.cfg.allow_download, self.root / "_downloads"))
        path = Path(src).expanduser()
        if not path.is_file():
            raise ingest.IngestError(f"file not found: {path}")
        t_start = time.time()
        progress(0.01, "decoding")
        audio = ingest.load(path, start, end)
        tc = TrackCache(self.root, audio.track_id)
        fp = CacheIndex.fingerprint(path, start, end)
        meta = tc.read_meta()
        want_plots = (depth == "deep") if plots is None else plots
        have = meta.get("depth")
        if (not force and have and DEPTH_ORDER.get(have, -1) >= DEPTH_ORDER[depth] and tc.analysis_path.is_file()
                and meta.get("pipeline_version") == pipeline_version()
                and (not stems or meta.get("stems_exported")) and (not want_plots or meta.get("plots"))):
            progress(1.0, "cached")
            self.index.record(fp, audio.track_id)
            return self._result(tc)
        tc.dir.mkdir(parents=True, exist_ok=True)
        tc.write_meta({**meta, "track_id": audio.track_id, "name": path.stem, "path": str(path.resolve()),
                       "start": audio.start, "end": audio.end, "duration_s": round(audio.duration, 3),
                       "created": meta.get("created", time.time()), "status": "running", "depth_requested": depth})
        self.index.record(fp, audio.track_id)
        try:
            analysis = self._run(audio, tc, depth, stems, want_plots, force, progress)
        except Exception:
            tc.update_meta(status="error")
            raise
        elapsed = round(time.time() - t_start, 2)
        analysis["generator"]["elapsed_s"] = elapsed
        write_json(tc.analysis_path, analysis, ndigits=4, indent=1)
        from .brief import make_brief

        tc.brief_path.write_text(make_brief(analysis))
        tc.update_meta(status="done", pipeline_version=pipeline_version(), depth=depth if DEPTH_ORDER[depth] >= DEPTH_ORDER.get(have or "quick", 0) or force else have,
                       elapsed_s=elapsed, stems_exported=bool(stems) or meta.get("stems_exported", False),
                       plots=bool(want_plots) or meta.get("plots", False),
                       summary=_summary_line(analysis))
        progress(1.0, "done")
        return self._result(tc)

    def _result(self, tc: TrackCache) -> dict:
        from .util import read_json

        analysis = read_json(tc.analysis_path)
        return {"track_id": tc.track_id, "cache": tc, "analysis": analysis, "files": list_files(tc)}

    # ------------------------------------------------------------------ stages
    def _run(self, audio: ingest.Audio, tc: TrackCache, depth: str, export_stems: bool, want_plots: bool, force: bool,
             progress: Progress) -> dict:
        cfg = self.cfg
        caps = models.capabilities()
        deep = depth == "deep"
        quick = depth == "quick"
        warnings: list[str] = []
        mono = audio.mono  # 22.05k normalized

        # ---- separation
        progress(0.05, "separating stems")
        sep_method = "none" if quick or cfg.separation_model == "none" else cfg.separation_model
        stems44, sep_used = self._stems(audio, tc, sep_method, force)
        if sep_used == "hpss-fallback" and not quick:
            warnings.append("Demucs unavailable: stems are HPSS pseudo-stems (drums/bass/other only); per-instrument analysis is coarser.")
        tag = "demucs" if sep_used.startswith("htdemucs") else "hpss"
        st22 = {k: _r22(v) for k, v in stems44.items()}
        drums44 = stems44["drums"].mean(axis=0).astype(np.float32)

        # ---- rhythm
        progress(0.35, "tracking beats")
        rh = tc.cached(f"rhythm-{tag}", rhythm.VERSION, lambda: rhythm.analyze(
            mono, SR22, drums=st22.get("drums"), backend=cfg.beat_backend, device=cfg.device), force)
        beats = np.asarray(rh.get("beats") or [], dtype=float)
        downbeats = np.asarray(rh.get("downbeats") or [], dtype=float)
        meter = int(rh.get("beats_per_bar") or 4)
        if len(beats) < 8:
            warnings.append("Very few beats detected: rhythm-dependent fields are unreliable (free-time or very sparse audio).")

        # ---- drums
        progress(0.45, "transcribing drums")
        dr = tc.cached(f"drums-{tag}", drums.VERSION, lambda: drums.transcribe(drums44, ingest.SR), force)
        hits = dr.get("hits", {})
        rh = rhythm.anchor_to_hits(rh, hits)  # kick-anchored grid (snare/hat microtiming measured against it)
        beats = np.asarray(rh.get("beats") or [], dtype=float)
        downbeats = np.asarray(rh.get("downbeats") or [], dtype=float)
        gr = tc.cached(f"groove-{tag}", groove.VERSION + "." + drums.VERSION + "." + rhythm.VERSION,
                       lambda: groove.analyze(hits, beats, downbeats, meter), force)
        dsound = tc.cached(f"drumsound-{tag}", drums.VERSION, lambda: {"sound": drums.describe_sounds(drums44, hits)}, force)["sound"]

        # ---- transcription (standard+)
        notes = {"bass": [], "keys": [], "melody": []}
        methods = {}
        if not quick:
            progress(0.55, "transcribing notes")
            vocals_on = "vocals" in st22 and _active(st22["vocals"], mono)

            def _tx():
                out = {}
                for kind, key in (("bass", "bass"), ("keys", "other"), ("melody", "vocals")):
                    if kind == "melody" and not vocals_on:
                        out[kind] = {"notes": [], "method": "no-vocals"}
                        continue
                    if key in st22:
                        out[kind] = transcription.transcribe(kind, st22[key], SR22, cfg.transcription_backend)
                return out

            tx = tc.cached(f"transcription-{tag}", transcription.VERSION, _tx, force)
            for k, v in tx.items():
                if isinstance(v, dict) and "notes" in v:
                    notes[k] = [tuple(n) for n in v["notes"]]
                    methods[k] = v.get("method")

        # ---- harmony
        progress(0.7, "analysing harmony")
        harm_sig = sum(st22[k] for k in ("bass", "other", "piano", "guitar") if k in st22)
        ha = tc.cached(f"harmony-{tag}", harmony.VERSION + "." + rhythm.VERSION, lambda: harmony.analyze(
            harm_sig, beats, downbeats, meter, y_bass=st22.get("bass"), note_events=notes["keys"] or None), force)
        chords = ha.get("chords", [])
        if not quick and "other" in st22 and not _active(st22["other"], mono, -24) and \
                not any(k in st22 and _active(st22[k], mono, -24) for k in ("piano", "guitar")):
            warnings.append("No chordal instruments detected (bass/drums only): chord qualities are implied by the bass "
                            "line and are low-confidence; treat the loop as root movement.")
            ha = {**ha, "confidence": min(ha.get("confidence") or 0.3, 0.3), "implied_by_bass": True}
        if quick or not notes["keys"]:
            X = (ha.get("_arrays") or {}).get("note_salience")
            if X is not None and not notes["keys"]:
                notes["keys"] = transcription.salience_notes(np.asarray(X), beats)
                methods.setdefault("keys", "nnls-salience")
        if quick or not notes["bass"]:
            pass

        # ---- parts
        progress(0.78, "analysing parts")
        kick_t = [h[0] for h in hits.get("kick", [])]
        swing = gr.get("swing", {})
        bass_info = parts.analyze_bass(notes["bass"], beats, downbeats, meter, swing, chords, kick_t) if notes["bass"] else {"present": False}
        voicing = parts.analyze_voicing(notes["keys"], chords, beats, downbeats, meter, swing) if notes["keys"] else {"present": False}
        mel_src = "vocals" if notes["melody"] else "keys top line"
        mel_notes = notes["melody"] or [n for n in notes["keys"] if n[2] >= 64]
        melody_info = parts.analyze_melody(mel_notes, beats, downbeats, meter, mel_src) if mel_notes and not quick else {"present": False}

        # ---- sound
        progress(0.84, "describing sound")
        bpm = rh.get("bpm")

        def _sound():
            desc = {}
            for k in (() if quick else ("bass", "other", "vocals", "piano", "guitar")):
                if k in st22 and _active(st22[k], mono):
                    desc[k] = sound.describe(st22[k], SR22, stems44[k], bpm=bpm)
            full = sound.describe(mono, SR22, audio.stereo_norm, bpm=bpm)
            space = sound.space_estimate(st22["drums"]) if "drums" in st22 else {"label": "unknown"}
            side = {}
            for k in ("bass", "other"):
                if k in st22:
                    r = sound.sidechain(st22[k], kick_t)
                    if r:
                        side[k] = r
            return {"stems": desc, "full": full, "space": space, "sidechain": side}

        snd = tc.cached(f"sound-{tag}{'-q' if quick else ''}", sound.VERSION + "." + drums.VERSION, _sound, force)
        tags = None
        if deep and cfg.tagging_backend != "none" and caps["clap"]:
            progress(0.88, "tagging (CLAP)")
            try:
                tg = tc.cached("tagging", "1", lambda: _clap(mono, cfg), force)
                tags = tg
            except Exception as e:  # pragma: no cover - network/model issues
                warnings.append(f"CLAP tagging failed ({type(e).__name__}); using heuristic tags.")
                log.warning("CLAP failed: %s", e)

        # ---- structure
        progress(0.9, "segmenting structure")
        tc_ = (ha.get("_arrays") or {}).get("treble_chroma")
        electronic = _is_electronic(gr, rh, bool(notes["melody"]))
        stru = tc.cached(f"structure-{tag}-{'edm' if electronic else 'song'}", structure.VERSION + "." + rhythm.VERSION, lambda: structure.analyze(
            mono, SR22, beats, downbeats, meter, {k: v for k, v in st22.items() if _active(v, mono)}, hits, gr,
            np.asarray(tc_) if tc_ is not None else None, notes["melody"] or None, audio.stereo, ingest.SR,
            electronic), force)

        analysis = assemble(audio, tc, depth, rh, dr, gr, dsound, ha, bass_info, voicing, melody_info, snd, tags, stru,
                            {"separation": sep_used, "beats": rh.get("method"), "drums": dr.get("method"),
                             "transcription": methods, "tagging": "clap" if tags else "heuristic"}, warnings, notes)

        # ---- artifacts
        progress(0.94, "writing MIDI")
        from . import midi as midi_mod

        midi_files = midi_mod.export_all(tc.midi_dir, analysis, hits, notes)
        analysis["files"]["midi"] = midi_files
        # raw events for on-demand tools (32nd-note grids, MIDI summaries) without re-analysis
        write_json(tc.dir / "hits.json", hits, ndigits=5, indent=None)
        write_json(tc.dir / "notes.json", {k: [list(n) for n in v] for k, v in notes.items()}, ndigits=4, indent=None)
        if export_stems:
            exported = {}
            for k, v in stems44.items():
                p = tc.stems_dir / f"{k}.wav"
                ingest.write_wav(p, v, ingest.SR)
                exported[k] = str(p)
            analysis["files"]["stems"] = exported
        if want_plots:
            progress(0.97, "rendering plots")
            from . import plots as plots_mod

            try:
                analysis["files"]["plots"] = plots_mod.render_all(tc.plots_dir, analysis, mono, SR22, ha)
            except Exception as e:  # pragma: no cover
                warnings.append(f"plot rendering failed: {e}")
        return analysis

    def _stems(self, audio: ingest.Audio, tc: TrackCache, method: str, force: bool) -> tuple[dict, str]:
        folder = tc.stem_cache_dir / method
        if not force and method != "none" and self.cfg.keep_stem_cache:
            got = separation.load_stems(folder)
            if got:
                return got, method
        if method == "none":
            hp = tc.stem_cache_dir / "hpss"
            got = separation.load_stems(hp) if not force else None
            if got:
                return got, "hpss-fallback"
            st = separation.pseudo_stems(audio.stereo_norm, ingest.SR)
            separation.save_stems(st, hp, ingest.SR)
            return st, "hpss-fallback"
        st, used = separation.separate(audio.stereo_norm, ingest.SR, method, self.cfg.device)
        if self.cfg.keep_stem_cache:
            separation.save_stems(st, tc.stem_cache_dir / (used if used != "hpss-fallback" else "hpss"), ingest.SR)
        return st, used


# ---------------------------------------------------------------------- helpers
def _active(y: np.ndarray, ref: np.ndarray, thr_db: float = -30.0) -> bool:
    if y is None or y.size == 0:
        return False
    a = float(np.sqrt(np.mean(y.astype(np.float64) ** 2)))
    b = float(np.sqrt(np.mean(ref.astype(np.float64) ** 2))) + 1e-12
    return 20 * np.log10(a / b + 1e-12) > thr_db


def _clap(mono: np.ndarray, cfg: Config) -> dict:
    t = sound.clap_tags(mono, SR22, cfg.clap_model, cfg.device)
    emb = t.pop("embedding")
    t = {k: v for k, v in t.items() if not k.startswith("_")}
    t["_arrays"] = {"clap_embedding": emb}
    return t


def _is_electronic(gr: dict, rh: dict, vocals: bool = False) -> bool:
    """Four-on-the-floor kick on a rock-steady grid -> use club vocabulary (groove/build/drop) for sections;
    otherwise verse/chorus."""
    main = groove.most_common_pattern(gr.get("patterns", {}))
    steady = (rh.get("tempo_stability") or {}).get("steady", False)
    kick = (main or {}).get("grid_16", {}).get("kick", "")
    four = len(kick) == 16 and all(kick[i] == "x" for i in (0, 4, 8, 12))
    return bool(four and steady and not vocals)


def _summary_line(a: dict) -> str:
    r, h = a.get("rhythm", {}), a.get("harmony", {})
    bpm = r.get("bpm")
    return f"{bpm:.1f} bpm {r.get('meter', '')} {h.get('key') or ''}".strip() if bpm else (h.get("key") or "")


def list_files(tc: TrackCache) -> dict:
    files = {"json": str(tc.analysis_path), "brief": str(tc.brief_path)}
    for k, d, pat in (("midi", tc.midi_dir, "*.mid"), ("stems", tc.stems_dir, "*.wav"), ("plots", tc.plots_dir, "*.png")):
        if d.is_dir():
            files[k] = sorted(str(p) for p in d.glob(pat))
    prev = tc.dir / "preview"
    if prev.is_dir():
        files["preview"] = sorted(str(p) for p in prev.glob("*.wav"))
    return files


# ---------------------------------------------------------------------- assembly
def _confidence_level(x: float | None) -> str:
    if x is None:
        return "unknown"
    return "high" if x >= 0.7 else "medium" if x >= 0.4 else "low"


def assemble(audio, tc, depth, rh, dr, gr, dsound, ha, bass_info, voicing, melody_info, snd, tags, stru, methods,
             warnings, notes) -> dict:
    swing = gr.get("swing", {})
    micro = gr.get("microtiming", {})
    pats = gr.get("patterns", {})
    stems_desc = snd.get("stems", {})
    side_all = snd.get("sidechain", {})
    side = None
    if side_all:
        det = [v for v in side_all.values() if v.get("detected")]
        if det:
            side = max(det, key=lambda v: v["depth_db"] - v.get("control_depth_db", 0))
            side = {**side, "on": [k for k, v in side_all.items() if v.get("detected")]}
        else:
            side = {"detected": False}
    mix = sound.mix_summary(snd.get("full", {}), audio.stereo_norm, snd.get("space", {}), audio.loudness)
    hints = sound.synthesis_hints(stems_desc, dsound, bass_info, voicing, mix, side)

    a: dict = {
        "schema_version": SCHEMA_VERSION,
        "generator": {"name": "audioref", "version": __version__, "depth": depth, "methods": methods,
                      "created": time.strftime("%Y-%m-%dT%H:%M:%S")},
        "track": {"id": audio.track_id, "name": audio.path.stem, "path": str(audio.path.resolve()),
                  "start": audio.start, "end": audio.end, "duration_s": round(audio.duration, 3),
                  "sample_rate": audio.sr, "loudness": {k: v for k, v in audio.loudness.items()},
                  "analysis_gain_db": round(audio.gain_db, 2)},
        "rhythm": {
            "bpm": round(rh["bpm"], 2) if rh.get("bpm") else None,
            "bpm_curve": rh.get("bpm_curve", []),
            "tempo_candidates": rh.get("tempo_candidates", []),
            "tempo_stability": rh.get("tempo_stability"),
            "meter": rh.get("meter"),
            "beats_per_bar": rh.get("beats_per_bar"),
            "first_downbeat": rh.get("first_downbeat"),
            "beats": rh.get("beats", []),
            "downbeats": rh.get("downbeats", []),
            "swing": {k: swing.get(k) for k in ("ratio_16th", "ratio_8th", "level", "percent", "feel", "triplet_grid",
                                                "by_voice", "confidence")},
            "microtiming_ms": {v: d["mean_ms"] for v, d in micro.items()},
            "microtiming_detail": micro,
            "groove_template_16": gr.get("template", {}),
            "confidence": rh.get("confidence"),
            "method": rh.get("method"),
            "grid_anchor": rh.get("grid_anchor"),
        },
        "drums": {
            "method": dr.get("method"),
            "voices": pats.get("voices", []),
            "hit_counts": {v: len(h) for v, h in dr.get("hits", {}).items()},
            "patterns": pats.get("patterns", []),
            "bar_patterns": pats.get("bar_patterns", []),
            "fills": pats.get("fills", []),
            "stops": pats.get("stops", []),
            "rolls": pats.get("rolls", []),
            "sound": dsound,
        },
        "harmony": {
            "key": ha.get("key"),
            "key_confidence": ha.get("key_confidence"),
            "key_alternatives": ha.get("key_alternatives"),
            "tuning_cents": ha.get("tuning_cents"),
            "chords": ha.get("chords", []),
            "bar_chords": ha.get("bar_chords", []),
            "loop": ha.get("loop"),
            "harmonic_rhythm": ha.get("harmonic_rhythm"),
            "vocabulary": ha.get("vocabulary"),
            "extensions": ha.get("extensions"),
            "voicing": voicing,
            "uncertain_spans": ha.get("uncertain_spans", []),
            "confidence": ha.get("confidence"),
            "implied_by_bass": bool(ha.get("implied_by_bass")),
        },
        "bass": {**bass_info, "midi": "midi/bass.mid" if notes.get("bass") else None, "sound": hints.get("bass")},
        "melody": {**melody_info, "midi": "midi/melody.mid" if melody_info.get("present") else None},
        "sound": {
            "tags": None,
            "mix": mix,
            "stems": {k: {kk: vv for kk, vv in v.items() if kk != "mfcc_mean"} for k, v in stems_desc.items()},
            "sidechain": side,
            "sidechain_by_stem": side_all,
            "synthesis_hints": hints,
            "_timbre_vector": snd.get("full", {}).get("mfcc_mean"),
        },
        "structure": {
            "sections": stru.get("sections", []),
            "events": stru.get("events", []),
            "energy_per_beat": stru.get("energy_per_beat", []),
            "energy_per_bar": stru.get("energy_per_bar", []),
            "layers_per_bar": stru.get("layers_per_bar", {}),
            "loudness_curve_lufs": stru.get("loudness_curve_lufs", []),
        },
        "files": {"json": str(tc.analysis_path), "brief": str(tc.brief_path)},
        "warnings": warnings,
    }
    if tags:
        a["sound"]["tags"] = {"genre": tags.get("genre"), "instruments": list((tags.get("instruments") or {}).keys())[:6],
                              "instrument_scores": tags.get("instruments"), "mood": list((tags.get("mood") or {}).keys())[:4],
                              "method": "clap"}
        emb = (tags.get("_arrays") or {}).get("clap_embedding")
        if emb is not None:
            a["sound"]["_clap_embedding"] = np.asarray(emb).round(5).tolist()
    else:
        a["sound"]["tags"] = sound.heuristic_tags(a)
        inst = []
        if a["drums"]["hit_counts"]:
            inst.append("drum machine" if (rh.get("tempo_stability") or {}).get("steady") else "drums")
        if bass_info.get("present"):
            inst.append("bass")
        if voicing.get("present"):
            inst.append("keys/pads")
        if melody_info.get("present") and melody_info.get("source") == "vocals":
            inst.append("vocals")
        a["sound"]["tags"]["instruments"] = inst

    # confidence roll-up + uncertainty warnings
    conf = {
        "tempo": (rh.get("confidence") or {}).get("tempo"),
        "downbeats": (rh.get("confidence") or {}).get("downbeat"),
        "swing": swing.get("confidence"),
        "key": ha.get("key_confidence"),
        "chords": ha.get("confidence"),
    }
    a["confidence"] = {k: {"value": v, "level": _confidence_level(v)} for k, v in conf.items()}
    for k, v in conf.items():
        if v is not None and v < 0.4:
            warnings.append(f"low confidence: {k} ({v:.2f})")
    cands = rh.get("tempo_candidates") or []
    alt = [c for c in cands if c["reading"] != "detected" and c["support"] > 0.5 * max((c2["support"] for c2 in cands), default=1)]
    if alt:
        warnings.append("tempo octave ambiguity: also plausible at " + ", ".join(f"{c['bpm']} bpm ({c['reading']})" for c in alt))
    for span in (ha.get("uncertain_spans") or [])[:3]:
        warnings.append(f"chords uncertain {span['start']:.1f}-{span['end']:.1f}s (read as {span['symbol']})")
    return to_jsonable(a, ndigits=4)
