"""Build / install the AudioRef menu bar app (macOS) from the Swift package in macos/AudioRef."""

from __future__ import annotations

import json
import plistlib
import shutil
import subprocess
import sys
from pathlib import Path

from . import __version__
from .capture import APP_LOCATIONS, REPO_PKG, SUPPORT_DIR, app_status

BUNDLE_ID = "dev.audioref.AudioRef"


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw)


def signing_identity() -> str:
    """A stable code-signing identity if one exists (keeps the macOS permission across rebuilds), else ad-hoc."""
    try:
        out = _run(["security", "find-identity", "-v", "-p", "codesigning"]).stdout
    except (OSError, subprocess.CalledProcessError):
        return "-"
    for line in out.splitlines():
        if '"Apple Development' in line or '"Developer ID Application' in line:
            return line.split('"')[1]
    return "-"


def _icon(resources: Path) -> str | None:
    """Render a simple app icon (waveform on a gradient squircle) to AppIcon.icns."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.patches import FancyBboxPatch
    except ImportError:
        return None
    work = resources / "AppIcon.iconset"
    work.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(10.24, 10.24), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    grad = np.linspace(0, 1, 256)[:, None]
    ax.imshow(np.dstack([0.16 + 0.25 * grad, 0.12 + 0.05 * grad, 0.42 + 0.3 * grad]), extent=(0.08, 0.92, 0.08, 0.92),
              origin="lower", clip_path=None, zorder=0)
    box = FancyBboxPatch((0.08, 0.08), 0.84, 0.84, boxstyle="round,pad=0,rounding_size=0.19", transform=ax.transData,
                         facecolor="none", edgecolor="none")
    ax.add_patch(box)
    ax.images[0].set_clip_path(box)
    xs = np.linspace(0.22, 0.78, 9)
    hs = [0.12, 0.26, 0.42, 0.3, 0.52, 0.34, 0.44, 0.24, 0.12]
    for x, h in zip(xs, hs):
        ax.plot([x, x], [0.5 - h / 2, 0.5 + h / 2], color="white", linewidth=34, solid_capstyle="round", zorder=2)
    master = work / "icon_512x512@2x.png"
    fig.savefig(master, transparent=True)
    plt.close(fig)
    for size in (16, 32, 128, 256, 512):
        for scale in (1, 2):
            px = size * scale
            name = f"icon_{size}x{size}{'@2x' if scale == 2 else ''}.png"
            if name != master.name:
                _run(["sips", "-z", str(px), str(px), str(master), "--out", str(work / name)])
    icns = resources / "AppIcon.icns"
    _run(["iconutil", "-c", "icns", str(work), "-o", str(icns)])
    shutil.rmtree(work, ignore_errors=True)
    return "AppIcon"


def build(verbose: bool = False) -> Path:
    """swift build the package and assemble build/AudioRef.app (signed)."""
    if sys.platform != "darwin":
        raise RuntimeError("the AudioRef app is macOS-only")
    if not (REPO_PKG / "Package.swift").is_file():
        raise RuntimeError(f"Swift sources not found at {REPO_PKG} (build from an audio-ref checkout)")
    if shutil.which("swift") is None:
        raise RuntimeError("Swift toolchain not found: install Xcode or the Command Line Tools (xcode-select --install)")
    r = subprocess.run(["swift", "build", "-c", "release", "--package-path", str(REPO_PKG)], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("swift build failed:\n" + (r.stdout + r.stderr)[-3000:])
    if verbose:
        print(r.stdout[-1500:])
    bins = REPO_PKG / ".build" / "release"
    app = REPO_PKG / "build" / "AudioRef.app"
    if app.exists():
        shutil.rmtree(app)
    macos = app / "Contents" / "MacOS"
    resources = app / "Contents" / "Resources"
    macos.mkdir(parents=True)
    resources.mkdir(parents=True)
    for exe in ("AudioRef", "audioref-capture"):
        shutil.copy2(bins / exe, macos / exe)
    icon = _icon(resources)
    info = {
        "CFBundleIdentifier": BUNDLE_ID, "CFBundleName": "AudioRef", "CFBundleDisplayName": "AudioRef",
        "CFBundleExecutable": "AudioRef", "CFBundlePackageType": "APPL", "CFBundleShortVersionString": __version__,
        "CFBundleVersion": __version__, "LSMinimumSystemVersion": "13.0", "LSUIElement": True,
        "NSHighResolutionCapable": True, "NSSupportsAutomaticTermination": False,
        "NSHumanReadableCopyright": "audio-ref: reference audio analyzer",
    }
    if icon:
        info["CFBundleIconFile"] = icon
    with open(app / "Contents" / "Info.plist", "wb") as f:
        plistlib.dump(info, f)
    subprocess.run(["xattr", "-cr", str(app)], capture_output=True)  # codesign rejects Finder/provenance metadata
    sign(app)
    return app


def sign(app: Path) -> str:
    """Sign with a stable identity when possible (macOS then keeps the capture permission across rebuilds);
    fall back to ad-hoc signing if that identity can't be used non-interactively."""
    ident = signing_identity()
    for candidate in ([ident, "-"] if ident != "-" else ["-"]):
        r = subprocess.run(["codesign", "--force", "--deep", "--sign", candidate, "--identifier", BUNDLE_ID, str(app)],
                           capture_output=True, text=True)
        if r.returncode == 0:
            return candidate
    raise RuntimeError("codesign failed: " + r.stderr.strip()[-500:])


def cli_command() -> list[str]:
    """How the app should invoke the analyzer: this checkout via uv, else an installed `audioref`."""
    repo = REPO_PKG.parents[1]
    uv = shutil.which("uv")
    if uv and (repo / "pyproject.toml").is_file():
        return [uv, "run", "--directory", str(repo), "audioref"]
    exe = shutil.which("audioref")
    if exe:
        return [exe]
    raise RuntimeError("could not locate the audioref CLI for the app to use")


def write_config(cache_dir: str = "~/.audioref") -> Path:
    SUPPORT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = {"cli": cli_command(), "cache_dir": str(Path(cache_dir).expanduser())}
    p = SUPPORT_DIR / "config.json"
    p.write_text(json.dumps(cfg, indent=2))
    return p


def install(cache_dir: str = "~/.audioref", dest: Path | None = None, launch: bool = True) -> dict:
    """Build, copy to ~/Applications, write the app config, (re)launch."""
    app = build()
    dest_dir = dest or APP_LOCATIONS[0].parent
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / "AudioRef.app"
    was_running = app_status() is not None
    if was_running:
        subprocess.run(["osascript", "-e", 'quit app "AudioRef"'], capture_output=True)
        subprocess.run(["pkill", "-x", "AudioRef"], capture_output=True)
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(app, target, symlinks=True)
    cfg = write_config(cache_dir)
    if launch:
        subprocess.run(["open", str(target)], check=False)
    signed = subprocess.run(["codesign", "-dvv", str(target)], capture_output=True, text=True).stderr
    authority = next((l.split("=", 1)[1] for l in signed.splitlines() if l.startswith("Authority=")), "ad-hoc")
    return {"app": str(target), "config": str(cfg), "signed_with": authority, "launched": launch}


def uninstall() -> dict:
    removed = []
    subprocess.run(["pkill", "-x", "AudioRef"], capture_output=True)
    for p in APP_LOCATIONS:
        if p.exists() and p.name == "AudioRef.app":
            shutil.rmtree(p)
            removed.append(str(p))
    return {"removed": removed, "kept": str(SUPPORT_DIR)}
