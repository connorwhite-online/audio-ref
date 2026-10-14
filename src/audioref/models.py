"""Lazy, process-wide registry of ML models so the MCP server keeps them warm between calls."""

from __future__ import annotations

import contextlib
import importlib.util
import os
import threading
import time
from typing import Any, Callable

from .util import log

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
# transformers otherwise spawns a background thread that queries the Hub to convert .bin checkpoints
os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")

_lock = threading.RLock()
_models: dict[str, Any] = {}
_load_times: dict[str, float] = {}


def available(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def capabilities() -> dict[str, bool]:
    return {
        "demucs": available("demucs") and available("torch"),
        "beat_this": available("beat_this") and available("torch"),
        "basic_pitch": available("basic_pitch"),
        "clap": available("transformers") and available("torch"),
    }


def torch_device(pref: str = "auto") -> str:
    if pref in ("cpu", "mps", "cuda"):
        return pref
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


@contextlib.contextmanager
def hf_offline():
    """Temporarily force huggingface_hub into offline mode (cache only, no network)."""
    try:
        import huggingface_hub.constants as hfc
    except ImportError:  # pragma: no cover
        yield
        return
    old = hfc.HF_HUB_OFFLINE
    hfc.HF_HUB_OFFLINE = True
    try:
        yield
    finally:
        hfc.HF_HUB_OFFLINE = old


def local_first(loader: Callable[[], Any]) -> Any:
    """Load from local caches without touching the network; only go online if that fails (first use)."""
    try:
        with hf_offline():
            return loader()
    except Exception as e:  # not cached yet
        log.info("model not in local cache (%s); downloading", type(e).__name__)
        return loader()


def get(name: str, loader: Callable[[], Any]) -> Any:
    with _lock:
        if name not in _models:
            t0 = time.time()
            log.info("loading model %s", name)
            _models[name] = loader()
            _load_times[name] = round(time.time() - t0, 2)
        return _models[name]


def loaded() -> dict[str, float]:
    return dict(_load_times)


def unload(name: str | None = None) -> None:
    with _lock:
        if name is None:
            _models.clear()
            _load_times.clear()
        else:
            _models.pop(name, None)
            _load_times.pop(name, None)
