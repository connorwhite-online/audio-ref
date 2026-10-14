"""Configuration: audioref.toml + environment + explicit overrides."""

from __future__ import annotations

import os
import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

DEPTHS = ("quick", "standard", "deep")

CONFIG_SEARCH = (
    Path("audioref.toml"),
    Path("~/.config/audioref/audioref.toml").expanduser(),
)


@dataclass
class Config:
    cache_dir: Path = field(default_factory=lambda: Path(".audioref"))
    default_depth: str = "standard"
    # separation: "htdemucs" | "htdemucs_6s" | "htdemucs_ft" | "none" (HPSS pseudo-stems)
    separation_model: str = "htdemucs"
    # beats: "auto" (beat_this if installed) | "beat_this" | "librosa"
    beat_backend: str = "auto"
    # transcription: "auto" (basic_pitch if installed) | "basic_pitch" | "fallback"
    transcription_backend: str = "auto"
    # tagging: "auto" (CLAP if installed, deep only) | "clap" | "none"
    tagging_backend: str = "auto"
    clap_model: str = "laion/larger_clap_music"
    # torch device: "auto" | "cpu" | "mps"
    device: str = "auto"
    # URL inputs are refused unless this is true (only for audio you own/have rights to)
    allow_download: bool = False
    # keep a FLAC copy of separated stems in the cache so re-analysis skips Demucs
    keep_stem_cache: bool = True
    # worker threads for the MCP job queue
    max_workers: int = 1

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["cache_dir"] = str(self.cache_dir)
        return d


def _coerce(name: str, value: Any) -> Any:
    if name == "cache_dir":
        return Path(str(value)).expanduser()
    if name in ("allow_download", "keep_stem_cache"):
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)
    if name == "max_workers":
        return int(value)
    return value


def load_config(path: str | Path | None = None, **overrides: Any) -> Config:
    """Load config: defaults < toml file < AUDIOREF_* env vars < explicit overrides."""
    cfg = Config()
    valid = {f.name for f in fields(Config)}

    candidates = [Path(path).expanduser()] if path else list(CONFIG_SEARCH)
    env_path = os.environ.get("AUDIOREF_CONFIG")
    if env_path and not path:
        candidates.insert(0, Path(env_path).expanduser())
    for cand in candidates:
        if cand.is_file():
            data = tomllib.loads(cand.read_text())
            data = data.get("audioref", data)
            for k, v in data.items():
                if k in valid:
                    setattr(cfg, k, _coerce(k, v))
            break
        if path:
            raise FileNotFoundError(f"config file not found: {cand}")

    for k in valid:
        env = os.environ.get(f"AUDIOREF_{k.upper()}")
        if env is not None:
            setattr(cfg, k, _coerce(k, env))
    # AUDIOREF_CACHE is the short alias used in docs
    if os.environ.get("AUDIOREF_CACHE"):
        cfg.cache_dir = _coerce("cache_dir", os.environ["AUDIOREF_CACHE"])

    for k, v in overrides.items():
        if v is None:
            continue
        if k not in valid:
            raise ValueError(f"unknown config key: {k}")
        setattr(cfg, k, _coerce(k, v))

    if cfg.default_depth not in DEPTHS:
        raise ValueError(f"default_depth must be one of {DEPTHS}")
    return cfg
