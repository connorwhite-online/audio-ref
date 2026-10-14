"""Reference audio analyzer: turns reference tracks into machine-readable musical briefs, MIDI and stems."""

import warnings as _warnings

__version__ = "0.1.0"

# third-party deprecation noise (resampy -> pkg_resources, coremltools/sklearn version checks)
_warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
_warnings.filterwarnings("ignore", category=UserWarning, module="resampy")
_warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")
