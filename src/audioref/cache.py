"""On-disk cache layout.

<cache_dir>/
  index.json                      source-path fingerprint -> track_id
  <track_id>/
    meta.json                     track info, depth completed, stage list
    stages/<name>@<version>.json  per-stage results (+ .npz arrays)
    stages/stems/<name>.flac      cached separated stems (internal)
    analysis.json  brief.md
    midi/*.mid  stems/*.wav  plots/*.png  preview/*.wav
"""

from __future__ import annotations

import hashlib
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .util import log, read_json, write_json

_index_lock = threading.Lock()


def audio_hash(stereo: np.ndarray) -> str:
    """Content hash of decoded PCM (float32), so re-encodes of the same audio map together."""
    h = hashlib.sha1()
    arr = np.ascontiguousarray(stereo, dtype=np.float32)
    # quantize to 16-bit first so tiny decoder float differences don't change the id
    q = np.round(arr * 32767.0).astype(np.int16)
    h.update(q.tobytes())
    h.update(str(arr.shape).encode())
    return h.hexdigest()


class TrackCache:
    def __init__(self, root: Path, track_id: str):
        self.root = Path(root)
        self.track_id = track_id
        self.dir = self.root / track_id

    # --- paths -----------------------------------------------------------
    @property
    def stages_dir(self) -> Path:
        return self.dir / "stages"

    @property
    def meta_path(self) -> Path:
        return self.dir / "meta.json"

    @property
    def analysis_path(self) -> Path:
        return self.dir / "analysis.json"

    @property
    def brief_path(self) -> Path:
        return self.dir / "brief.md"

    @property
    def midi_dir(self) -> Path:
        return self.dir / "midi"

    @property
    def stems_dir(self) -> Path:
        return self.dir / "stems"

    @property
    def plots_dir(self) -> Path:
        return self.dir / "plots"

    @property
    def stem_cache_dir(self) -> Path:
        return self.stages_dir / "stems"

    def exists(self) -> bool:
        return self.meta_path.is_file()

    # --- meta ------------------------------------------------------------
    def read_meta(self) -> dict[str, Any]:
        return read_json(self.meta_path) if self.meta_path.is_file() else {}

    def write_meta(self, meta: dict[str, Any]) -> None:
        write_json(self.meta_path, meta, ndigits=None, indent=2)

    def update_meta(self, **kw: Any) -> dict[str, Any]:
        meta = self.read_meta()
        meta.update(kw)
        meta["updated"] = time.time()
        self.write_meta(meta)
        return meta

    # --- stages ----------------------------------------------------------
    def stage_path(self, name: str, version: str) -> Path:
        return self.stages_dir / f"{name}@{version}.json"

    def has_stage(self, name: str, version: str) -> bool:
        return self.stage_path(name, version).is_file()

    def load_stage(self, name: str, version: str) -> dict[str, Any] | None:
        p = self.stage_path(name, version)
        if not p.is_file():
            return None
        data = read_json(p)
        npz = p.with_suffix(".npz")
        if npz.is_file():
            with np.load(npz, allow_pickle=False) as z:
                data["_arrays"] = {k: z[k] for k in z.files}
        return data

    def save_stage(self, name: str, version: str, data: dict[str, Any]) -> None:
        p = self.stage_path(name, version)
        p.parent.mkdir(parents=True, exist_ok=True)
        arrays = data.get("_arrays")
        body = {k: v for k, v in data.items() if k != "_arrays"}
        if arrays:
            np.savez_compressed(p.with_suffix(".npz"), **{k: np.asarray(v) for k, v in arrays.items()})
        write_json(p, body, ndigits=6, indent=None)

    def cached(self, name: str, version: str, fn: Callable[[], dict[str, Any]], force: bool = False) -> dict[str, Any]:
        if not force:
            got = self.load_stage(name, version)
            if got is not None:
                log.debug("stage %s@%s: cache hit", name, version)
                return got
        t0 = time.time()
        data = fn()
        data.setdefault("_elapsed_s", round(time.time() - t0, 3))
        self.save_stage(name, version, data)
        return data

    def delete(self) -> None:
        if self.dir.is_dir():
            shutil.rmtree(self.dir)


class CacheIndex:
    """Maps a cheap source fingerprint (path, size, mtime, range) to a track id."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.path = self.root / "index.json"

    def _load(self) -> dict[str, Any]:
        if self.path.is_file():
            try:
                return read_json(self.path)
            except Exception:  # corrupt index is just a cache miss
                return {}
        return {}

    @staticmethod
    def fingerprint(path: Path, start: float | None, end: float | None) -> str:
        st = path.stat()
        return f"{path.resolve()}|{st.st_size}|{int(st.st_mtime)}|{start}|{end}"

    def lookup(self, fp: str) -> str | None:
        tid = self._load().get(fp)
        if tid and (self.root / tid / "meta.json").is_file():
            return tid
        return None

    def record(self, fp: str, track_id: str) -> None:
        with _index_lock:
            data = self._load()
            data[fp] = track_id
            write_json(self.path, data, ndigits=None, indent=1)

    def forget(self, track_id: str) -> None:
        with _index_lock:
            data = {k: v for k, v in self._load().items() if v != track_id}
            write_json(self.path, data, ndigits=None, indent=1)

    def tracks(self) -> list[TrackCache]:
        if not self.root.is_dir():
            return []
        out = []
        for d in sorted(self.root.iterdir()):
            if d.is_dir() and (d / "meta.json").is_file():
                out.append(TrackCache(self.root, d.name))
        return out

    def resolve(self, ref: str) -> TrackCache:
        """Resolve a full id, unique id prefix, or track name to a TrackCache."""
        ref = ref.strip()
        direct = TrackCache(self.root, ref)
        if direct.exists():
            return direct
        tracks = self.tracks()
        matches = [t for t in tracks if t.track_id.startswith(ref)]
        if len(matches) == 1:
            return matches[0]
        by_name = [t for t in tracks if t.read_meta().get("name") in (ref, Path(ref).name, Path(ref).stem)]
        if len(by_name) == 1:
            return by_name[0]
        if len(matches) + len(by_name) > 1:
            raise KeyError(f"ambiguous track reference '{ref}'")
        raise KeyError(f"unknown track '{ref}' (run analyze first; list_tracks shows cached ids)")
