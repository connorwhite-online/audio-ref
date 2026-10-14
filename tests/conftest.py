import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

FIXTURES = ROOT / "tests" / ".fixtures"


@pytest.fixture(scope="session")
def demo_songs():
    """Render the ground-truth demo songs once per session -> {name: (wav_path, truth, stems)}."""
    from audioref import demo
    from audioref.ingest import write_wav

    FIXTURES.mkdir(parents=True, exist_ok=True)
    out = {}
    for name in demo.SONGS:
        mix, stems, truth = demo.render_song(name)
        p = FIXTURES / f"{name}.wav"
        if not p.is_file():
            write_wav(p, mix, 44100)
            (FIXTURES / f"{name}.truth.json").write_text(json.dumps(truth, default=float))
        out[name] = (p, truth, stems)
    return out


@pytest.fixture(scope="session")
def cache_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("audioref_cache")


@pytest.fixture(scope="session")
def service(cache_dir):
    from audioref.config import load_config
    from audioref.service import Service

    return Service(load_config(None, cache_dir=str(cache_dir)))


@pytest.fixture(scope="session")
def house_standard(service, demo_songs):
    """Full standard-depth analysis of the house demo (Demucs if installed)."""
    path, truth, _ = demo_songs["house"]
    res = service.analyze(str(path), depth="standard", trim=False)
    return res, truth


def mono22(x):
    import soxr

    return soxr.resample(x.mean(0), 44100, 22050).astype(np.float32)
