"""Record what is playing on this Mac (system audio) for analysis.

Two transports, same JSON event protocol:
  * the AudioRef menu bar app (preferred): a Unix socket in ~/Library/Application Support/AudioRef/; macOS
    "Screen & System Audio Recording" permission belongs to AudioRef.app
  * the headless `audioref-capture` helper (fallback): permission belongs to whatever launched it (Terminal, ...)

Events: {"event": "waiting" | "started" | "level" (db, elapsed) | "finished" (path, duration, reason) | "error" (message)}
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable

SUPPORT_DIR = Path("~/Library/Application Support/AudioRef").expanduser()
SOCKET_PATH = SUPPORT_DIR / "control.sock"
APP_LOCATIONS = (Path("~/Applications/AudioRef.app").expanduser(), Path("/Applications/AudioRef.app"))
REPO_PKG = Path(__file__).resolve().parents[2] / "macos" / "AudioRef"

Event = dict
OnEvent = Callable[[Event], None]

_lock = threading.Lock()
_current: dict = {}


class CaptureError(RuntimeError):
    pass


# ------------------------------------------------------------------ discovery
def installed_app() -> Path | None:
    for p in APP_LOCATIONS:
        if (p / "Contents" / "MacOS" / "AudioRef").is_file():
            return p
    return None


def find_helper() -> Path | None:
    cands = [p / "Contents" / "MacOS" / "audioref-capture" for p in APP_LOCATIONS]
    cands += [REPO_PKG / "build" / "AudioRef.app" / "Contents" / "MacOS" / "audioref-capture",
              REPO_PKG / ".build" / "release" / "audioref-capture"]
    return next((c for c in cands if c.is_file() and os.access(c, os.X_OK)), None)


def _connect(timeout: float = 2.0) -> socket.socket:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    s.connect(str(SOCKET_PATH))
    return s


def app_status(timeout: float = 1.0) -> dict | None:
    """Status of the running menu bar app, or None if it isn't running."""
    if not SOCKET_PATH.exists():
        return None
    try:
        with _connect(timeout) as s:
            s.sendall(b'{"cmd": "status"}\n')
            line = s.makefile("r").readline()
            return json.loads(line) if line else None
    except (OSError, ValueError):
        return None


def launch_app(wait: float = 8.0) -> bool:
    """Open the installed menu bar app (in the background) and wait for its control socket."""
    app = installed_app()
    if app is None:
        return False
    subprocess.run(["open", "-g", str(app)], check=False, capture_output=True)
    t0 = time.time()
    while time.time() - t0 < wait:
        if app_status(0.5):
            return True
        time.sleep(0.25)
    return False


# ------------------------------------------------------------------ transports
def _params(out: Path, max_seconds: float, stop_on_silence: float, wait_for_audio: float, threshold_db: float,
            simulate: str | None) -> dict:
    p = {"out": str(out), "max_seconds": float(max_seconds), "stop_on_silence": float(stop_on_silence),
         "wait_for_audio": float(wait_for_audio), "threshold_db": float(threshold_db)}
    if simulate:
        p["simulate"] = str(Path(simulate).expanduser().resolve())
    return p


def _run_app(params: dict, on_event: OnEvent) -> dict:
    s = _connect(timeout=10.0)
    with _lock:
        _current.clear()
        _current.update({"via": "app"})
    try:
        s.sendall((json.dumps({"cmd": "capture", **params}) + "\n").encode())
        s.settimeout(params["max_seconds"] + params["wait_for_audio"] + 30)
        f = s.makefile("r")
        for line in f:
            ev = json.loads(line)
            if "event" not in ev:
                raise CaptureError(ev.get("error") or f"unexpected reply: {line.strip()}")
            on_event(ev)
            if ev["event"] == "finished":
                return {**ev, "via": "app"}
            if ev["event"] == "error":
                raise CaptureError(ev.get("message", "capture failed"))
        raise CaptureError("AudioRef closed the connection before the capture finished")
    finally:
        s.close()
        with _lock:
            _current.clear()


def _run_helper(params: dict, on_event: OnEvent) -> dict:
    helper = find_helper()
    if helper is None:
        raise CaptureError("No capture backend: install the menu bar app (`audioref app install`) or build the helper "
                           "(`audioref app build`).")
    cmd = [str(helper), "--out", params["out"], "--max-seconds", str(params["max_seconds"]),
           "--stop-on-silence", str(params["stop_on_silence"]), "--wait-for-audio", str(params["wait_for_audio"]),
           "--threshold-db", str(params["threshold_db"])]
    if params.get("simulate"):
        cmd += ["--simulate", params["simulate"]]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    with _lock:
        _current.clear()
        _current.update({"via": "helper", "proc": proc})
    try:
        result = None
        assert proc.stdout is not None
        for line in proc.stdout:
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            on_event(ev)
            if ev.get("event") == "finished":
                result = {**ev, "via": "helper"}
            elif ev.get("event") == "error":
                raise CaptureError(ev.get("message", "capture failed"))
        proc.wait(timeout=10)
        if result is None:
            err = proc.stderr.read() if proc.stderr else ""
            raise CaptureError(f"capture helper exited ({proc.returncode}) without a recording. {err.strip()[:300]}")
        return result
    finally:
        if proc.poll() is None:
            proc.terminate()
        with _lock:
            _current.clear()


# ------------------------------------------------------------------ public API
def capture(out: str | Path, max_seconds: float = 600, stop_on_silence: float = 3.0, wait_for_audio: float = 60,
            threshold_db: float = -55, simulate: str | None = None, on_event: OnEvent | None = None,
            via: str = "auto") -> dict:
    """Record system audio (or `simulate` a file) into `out` (WAV). Blocks until the capture ends.
    via: "app" | "helper" | "auto" (app if running/installed, else helper). Returns the "finished" event."""
    if sys.platform != "darwin":
        raise CaptureError("System audio capture is only available on macOS.")
    out = Path(out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    params = _params(out, max_seconds, stop_on_silence, wait_for_audio, threshold_db, simulate)
    on_event = on_event or (lambda e: None)
    if via not in ("auto", "app", "helper"):
        raise ValueError("via must be auto | app | helper")
    use_app = via == "app" or (via == "auto" and (app_status() is not None or launch_app()))
    if use_app:
        if app_status() is None and not launch_app():
            raise CaptureError("The AudioRef menu bar app is not running (install it with `audioref app install`).")
        return _run_app(params, on_event)
    return _run_helper(params, on_event)


def stop() -> bool:
    """Stop the capture in progress (keeps what was recorded). Returns True if one was running."""
    with _lock:
        cur = dict(_current)
    if cur.get("via") == "helper" and cur.get("proc") is not None:
        cur["proc"].send_signal(signal.SIGINT)
        return True
    if cur.get("via") == "app" or (app_status() or {}).get("phase") in ("waiting", "recording"):
        try:
            with _connect(2.0) as s:
                s.sendall(b'{"cmd": "stop"}\n')
                s.makefile("r").readline()
            return True
        except OSError:
            return False
    return False


def helper_permission() -> bool | None:
    """Whether the headless helper (launched from this process) may capture; None if no helper."""
    helper = find_helper()
    if helper is None:
        return None
    r = subprocess.run([str(helper), "--check"], capture_output=True, text=True, timeout=10)
    try:
        return bool(json.loads(r.stdout.strip().splitlines()[-1])["permission"])
    except (ValueError, IndexError, KeyError):
        return None
