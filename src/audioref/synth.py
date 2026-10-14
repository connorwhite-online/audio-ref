"""A small numpy synthesizer: drum voices, tonal voices, a bus mixer with sidechain + filter sweeps.

Used by `render_preview` and by `audioref.demo` (ground-truth test songs).
"""

from __future__ import annotations

import functools
import math

import numpy as np
from scipy import signal

SR = 44100
_rng = np.random.default_rng(1234)


def _noise(n: int, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed) if seed is not None else _rng
    return rng.standard_normal(n).astype(np.float64)


def _sos(kind: str, freq, sr: int = SR, order: int = 2):
    nyq = sr / 2
    if isinstance(freq, (tuple, list)):
        wn = [min(max(f / nyq, 1e-4), 0.999) for f in freq]
    else:
        wn = min(max(freq / nyq, 1e-4), 0.999)
    return signal.butter(order, wn, btype=kind, output="sos")


def lowpass(x: np.ndarray, f: float, sr: int = SR, order: int = 2) -> np.ndarray:
    return signal.sosfilt(_sos("lowpass", f, sr, order), x, axis=-1)


def highpass(x: np.ndarray, f: float, sr: int = SR, order: int = 2) -> np.ndarray:
    return signal.sosfilt(_sos("highpass", f, sr, order), x, axis=-1)


def bandpass(x: np.ndarray, lo: float, hi: float, sr: int = SR, order: int = 2) -> np.ndarray:
    return signal.sosfilt(_sos("bandpass", (lo, hi), sr, order), x, axis=-1)


def exp_env(n: int, tau_s: float, sr: int = SR, attack_s: float = 0.0) -> np.ndarray:
    t = np.arange(n) / sr
    env = np.exp(-t / max(tau_s, 1e-4))
    if attack_s > 0:
        na = max(1, int(attack_s * sr))
        env[:na] *= np.linspace(0, 1, na)
    return env


def adsr(n: int, sr: int, a: float, d: float, s: float, r: float, gate_s: float) -> np.ndarray:
    t = np.arange(n) / sr
    env = np.zeros(n)
    a = max(a, 1e-4)
    d = max(d, 1e-4)
    att = t < a
    env[att] = t[att] / a
    dec = (t >= a) & (t < gate_s)
    env[dec] = s + (1 - s) * np.exp(-(t[dec] - a) / d)
    gate_level = s + (1 - s) * math.exp(-max(gate_s - a, 0) / d) if gate_s > a else min(gate_s / a, 1.0)
    rel = t >= gate_s
    env[rel] = gate_level * np.exp(-(t[rel] - gate_s) / max(r, 1e-4))
    return env


# ----------------------------------------------------------------- oscillators
def _polyblep(t: np.ndarray, dt: np.ndarray) -> np.ndarray:
    out = np.zeros_like(t)
    m1 = t < dt
    x = t[m1] / dt[m1]
    out[m1] = x + x - x * x - 1.0
    m2 = t > 1.0 - dt
    x = (t[m2] - 1.0) / dt[m2]
    out[m2] = x * x + x + x + 1.0
    return out


def osc(wave: str, freq: np.ndarray | float, n: int, sr: int = SR, phase0: float = 0.0) -> np.ndarray:
    f = np.broadcast_to(np.asarray(freq, dtype=float), (n,)).copy()
    dt = f / sr
    ph = (phase0 + np.cumsum(dt) - dt[0]) % 1.0
    if wave == "sine":
        return np.sin(2 * np.pi * ph)
    if wave == "saw":
        return 2.0 * ph - 1.0 - _polyblep(ph, dt)
    if wave == "square":
        sq = np.where(ph < 0.5, 1.0, -1.0)
        return sq + _polyblep(ph, dt) - _polyblep((ph + 0.5) % 1.0, dt)
    if wave == "tri":
        return 2.0 * np.abs(2.0 * ph - 1.0) - 1.0
    raise ValueError(wave)


def mtof(m: float) -> float:
    return 440.0 * 2 ** ((m - 69) / 12)


# ------------------------------------------------------------------ drum voices
@functools.lru_cache(maxsize=256)
def _kick(f0: float, f_start: float, sweep_s: float, decay_s: float, click: float, drive: float, sr: int) -> np.ndarray:
    n = int(sr * (decay_s * 6 + 0.03))
    t = np.arange(n) / sr
    f = f0 + (f_start - f0) * np.exp(-t / sweep_s)
    body = np.sin(2 * np.pi * np.cumsum(f) / sr) * np.exp(-t / decay_s)
    clk = bandpass(_noise(n, 7), 1500, 7000, sr) * np.exp(-t / 0.0025) * click * 3
    y = body + clk
    if drive > 0:
        g = 1 + 6 * drive
        y = np.tanh(g * y) / math.tanh(g)
    return y / max(np.max(np.abs(y)), 1e-9)


def kick(vel=1.0, f0=52.0, f_start=170.0, sweep_s=0.03, decay_s=0.07, click=0.35, drive=0.15, sr=SR):
    return vel * _kick(f0, f_start, sweep_s, decay_s, click, drive, sr)


@functools.lru_cache(maxsize=64)
def _snare(body_hz: float, decay_s: float, tone: float, sr: int) -> np.ndarray:
    n = int(sr * (decay_s * 6 + 0.02))
    t = np.arange(n) / sr
    tonal = (np.sin(2 * np.pi * body_hz * t) + 0.5 * np.sin(2 * np.pi * body_hz * 1.78 * t)) * np.exp(-t / 0.045)
    nz = highpass(_noise(n, 11), 1200, sr) * np.exp(-t / decay_s)
    y = tone * tonal + 0.9 * nz
    return y / np.max(np.abs(y))


def snare(vel=1.0, body_hz=190.0, decay_s=0.06, tone=0.6, sr=SR):
    return vel * _snare(body_hz, decay_s, tone, sr)


@functools.lru_cache(maxsize=32)
def _clap(decay_s: float, spacing_s: float, bursts: int, sr: int) -> np.ndarray:
    n = int(sr * (decay_s * 6 + spacing_s * bursts + 0.02))
    t = np.arange(n) / sr
    env = np.zeros(n)
    for b in range(bursts):
        t0 = b * spacing_s
        m = t >= t0
        env[m] += np.exp(-(t[m] - t0) / 0.004) * (0.8 if b < bursts - 1 else 1.0)
    tail_start = (bursts - 1) * spacing_s
    m = t >= tail_start
    env[m] += 0.7 * np.exp(-(t[m] - tail_start) / decay_s)
    y = bandpass(_noise(n, 13), 800, 3200, sr, 2) * env
    return y / np.max(np.abs(y))


def clap(vel=1.0, decay_s=0.05, spacing_s=0.011, bursts=3, sr=SR):
    return vel * _clap(decay_s, spacing_s, bursts, sr)


@functools.lru_cache(maxsize=32)
def _hat(decay_s: float, cutoff: float, sr: int) -> np.ndarray:
    n = int(sr * (decay_s * 6 + 0.01))
    t = np.arange(n) / sr
    # 808-style metallic cluster of square waves + noise
    ratios = [2.0, 3.0, 4.16, 5.43, 6.79, 8.21]
    metal = sum(osc("square", 40 * r * 4.1, n, sr) for r in ratios)
    y = highpass(0.6 * metal + _noise(n, 17), cutoff, sr, 4) * exp_env(n, decay_s, sr, 0.0005)
    return y / np.max(np.abs(y))


def hat(vel=1.0, open_=False, decay_s=None, cutoff=7000.0, sr=SR):
    d = decay_s if decay_s is not None else (0.12 if open_ else 0.018)
    return vel * _hat(d, cutoff, sr)


@functools.lru_cache(maxsize=8)
def _rim(sr: int) -> np.ndarray:
    n = int(sr * 0.08)
    t = np.arange(n) / sr
    y = (np.sin(2 * np.pi * 1700 * t) * 0.6 + np.sin(2 * np.pi * 480 * t)) * np.exp(-t / 0.008)
    y += bandpass(_noise(n, 19), 1500, 5000, sr) * np.exp(-t / 0.004)
    return y / np.max(np.abs(y))


def rim(vel=1.0, sr=SR):
    return vel * _rim(sr)


@functools.lru_cache(maxsize=16)
def _tom(f: float, sr: int) -> np.ndarray:
    n = int(sr * 0.6)
    t = np.arange(n) / sr
    fr = f * (1 + 0.4 * np.exp(-t / 0.05))
    y = np.sin(2 * np.pi * np.cumsum(fr) / sr) * np.exp(-t / 0.12)
    y += 0.2 * bandpass(_noise(n, 23), 200, 3000, sr) * np.exp(-t / 0.01)
    return y / np.max(np.abs(y))


def tom(vel=1.0, f=110.0, sr=SR):
    return vel * _tom(f, sr)


@functools.lru_cache(maxsize=4)
def _crash(sr: int) -> np.ndarray:
    n = int(sr * 2.5)
    y = highpass(_noise(n, 29), 3500, sr, 2) * exp_env(n, 0.55, sr, 0.001)
    return y / np.max(np.abs(y))


def crash(vel=1.0, sr=SR):
    return vel * _crash(sr)


@functools.lru_cache(maxsize=4)
def _shaker(sr: int) -> np.ndarray:
    n = int(sr * 0.12)
    t = np.arange(n) / sr
    env = np.minimum(t / 0.012, 1.0) * np.exp(-np.maximum(t - 0.012, 0) / 0.025)
    y = bandpass(_noise(n, 31), 4500, 11000, sr) * env
    return y / np.max(np.abs(y))


def shaker(vel=1.0, sr=SR):
    return vel * _shaker(sr)


DRUMS = {
    "kick": kick, "snare": snare, "clap": clap, "hat_closed": lambda vel=1.0, sr=SR: hat(vel, False, sr=sr),
    "hat_open": lambda vel=1.0, sr=SR: hat(vel, True, sr=sr), "rim": rim, "tom": tom, "crash": crash,
    "perc": shaker, "ride": lambda vel=1.0, sr=SR: hat(vel, True, decay_s=0.35, cutoff=5000, sr=sr),
}

# General MIDI drum map (note -> voice) used when rendering drum MIDI
GM_DRUMS = {
    35: "kick", 36: "kick", 37: "rim", 38: "snare", 40: "snare", 39: "clap", 42: "hat_closed", 44: "hat_closed",
    46: "hat_open", 49: "crash", 57: "crash", 51: "ride", 59: "ride", 41: "tom", 43: "tom", 45: "tom", 47: "tom",
    48: "tom", 50: "tom", 54: "perc", 69: "perc", 70: "perc", 82: "perc",
}
VOICE_TO_GM = {"kick": 36, "snare": 38, "clap": 39, "hat_closed": 42, "hat_open": 46, "ride": 51, "crash": 49,
               "tom": 45, "rim": 37, "perc": 70}


# ----------------------------------------------------------------- tonal voices
@functools.lru_cache(maxsize=2048)
def _epiano(midi: int, dur: float, sr: int, bright: float) -> np.ndarray:
    n = int(sr * (dur + 0.6))
    t = np.arange(n) / sr
    f = mtof(midi)
    index = (0.4 + 2.2 * bright) * np.exp(-t / 0.12)
    mod = np.sin(2 * np.pi * f * t)
    car = np.sin(2 * np.pi * f * t + index * mod)
    tine = 0.15 * np.sin(2 * np.pi * f * 14 * t) * np.exp(-t / 0.02)
    env = adsr(n, sr, 0.002, 0.9, 0.35, 0.15, dur)
    return (car + tine) * env


def epiano(midi: int, dur: float, vel: float = 0.8, bright: float = 0.5, sr: int = SR) -> np.ndarray:
    return vel * _epiano(int(midi), round(dur, 3), sr, round(bright, 2))


@functools.lru_cache(maxsize=1024)
def _pad(midi: int, dur: float, sr: int, cutoff: float) -> np.ndarray:
    n = int(sr * (dur + 0.8))
    f = mtof(midi)
    y = sum(osc("saw", f * 2 ** (c / 1200), n, sr, phase0=k * 0.31) for k, c in enumerate((-9, 0, 8)))
    y = lowpass(y / 3, cutoff, sr, 2)
    return y * adsr(n, sr, 0.25, 0.5, 0.8, 0.5, dur)


def pad(midi: int, dur: float, vel: float = 0.6, cutoff: float = 1400.0, sr: int = SR) -> np.ndarray:
    return vel * _pad(int(midi), round(dur, 3), sr, float(cutoff))


@functools.lru_cache(maxsize=1024)
def _bass(midi: int, dur: float, sr: int, cutoff: float, glide_from: int | None, sub: float) -> np.ndarray:
    n = int(sr * (dur + 0.08))
    t = np.arange(n) / sr
    f_target = mtof(midi)
    if glide_from is not None:
        f0 = mtof(glide_from)
        f = f_target + (f0 - f_target) * np.exp(-t / 0.04)
    else:
        f = np.full(n, f_target)
    y = 0.7 * osc("saw", f, n, sr) + sub * osc("sine", f, n, sr)
    # filter envelope: open then close towards cutoff
    y = 0.6 * lowpass(y, cutoff * 2.5, sr, 2) * np.exp(-t / 0.05) + lowpass(y, cutoff, sr, 2) * (1 - np.exp(-t / 0.05))
    return y * adsr(n, sr, 0.003, 0.25, 0.7, 0.03, dur)


def bass(midi: int, dur: float, vel: float = 0.9, cutoff: float = 420.0, glide_from: int | None = None,
         sub: float = 0.6, sr: int = SR) -> np.ndarray:
    return vel * _bass(int(midi), round(dur, 3), sr, float(cutoff), glide_from, sub)


@functools.lru_cache(maxsize=1024)
def _lead(midi: int, dur: float, sr: int) -> np.ndarray:
    n = int(sr * (dur + 0.15))
    t = np.arange(n) / sr
    f = mtof(midi) * (1 + 0.004 * np.sin(2 * np.pi * 5.5 * t) * np.minimum(t / 0.3, 1))
    y = lowpass(osc("square", f, n, sr) * 0.5 + osc("saw", f * 1.003, n, sr) * 0.5, 3000, sr)
    return y * adsr(n, sr, 0.01, 0.2, 0.7, 0.1, dur)


def lead(midi: int, dur: float, vel: float = 0.7, sr: int = SR) -> np.ndarray:
    return vel * _lead(int(midi), round(dur, 3), sr)


def sine_tone(midi: int, dur: float, vel: float = 0.7, sr: int = SR) -> np.ndarray:
    n = int(sr * (dur + 0.05))
    return vel * osc("sine", mtof(midi), n, sr) * adsr(n, sr, 0.005, 0.3, 0.6, 0.04, dur)


def riser(dur: float, sr: int = SR) -> np.ndarray:
    n = int(sr * dur)
    t = np.arange(n) / sr
    x = _noise(n, 37)
    out = np.zeros(n)
    block = 1024
    zi = None
    for i in range(0, n, block):
        frac = i / n
        fc = 400 * (12000 / 400) ** frac
        sos = _sos("bandpass", (fc * 0.7, min(fc * 1.4, 20000)), sr, 1)
        if zi is None:
            zi = signal.sosfilt_zi(sos) * 0
        seg, zi = signal.sosfilt(sos, x[i:i + block], zi=zi)
        out[i:i + block] = seg
    return out * (t / dur) ** 2 * 0.8


TONAL = {"epiano": epiano, "pad": pad, "bass": bass, "lead": lead, "sine": sine_tone}


# ----------------------------------------------------------------------- mixer
class Mixer:
    """Accumulates sounds on named stereo buses; applies sidechain and filter sweeps per bus."""

    def __init__(self, duration_s: float, sr: int = SR, buses=("drums", "bass", "other", "vocals")):
        self.sr = sr
        self.n = int(math.ceil(duration_s * sr))
        self.buses = {b: np.zeros((2, self.n)) for b in buses}

    def add(self, bus: str, sig: np.ndarray, t: float, gain: float = 1.0, pan: float = 0.0) -> None:
        i = int(round(t * self.sr))
        if i >= self.n or sig.size == 0:
            return
        if i < 0:
            sig = sig[-i:]
            i = 0
        seg = sig[: self.n - i] * gain
        if seg.ndim == 1:
            lg = math.cos((pan + 1) * math.pi / 4)
            rg = math.sin((pan + 1) * math.pi / 4)
            self.buses[bus][0, i:i + seg.size] += seg * lg * math.sqrt(2)
            self.buses[bus][1, i:i + seg.size] += seg * rg * math.sqrt(2)
        else:
            self.buses[bus][:, i:i + seg.shape[-1]] += seg

    def sidechain(self, bus: str, times: list[float], depth_db: float, release_ms: float, attack_ms: float = 4.0) -> None:
        if depth_db <= 0 or not times:
            return
        floor = 10 ** (-depth_db / 20)
        deficit = np.zeros(self.n)
        tau = release_ms / 1000.0
        L = int(self.sr * tau * 6)
        ramp_n = max(1, int(attack_ms / 1000 * self.sr))
        t = np.arange(L) / self.sr
        shape = np.exp(-t / tau)
        shape[:ramp_n] = np.linspace(0, 1, ramp_n)  # fast duck
        for tk in times:
            i = int(round(tk * self.sr))
            if i >= self.n:
                continue
            seg = shape[: self.n - i] * (1 - floor)
            deficit[i:i + seg.size] = np.maximum(deficit[i:i + seg.size], seg)
        self.buses[bus] *= (1 - deficit)[None, :]

    def filter_sweep(self, bus: str, t0: float, t1: float, f0: float, f1: float, hold_before: bool = True) -> None:
        """Time-varying low-pass between t0 and t1 (exponential cutoff ramp)."""
        x = self.buses[bus]
        i0, i1 = int(t0 * self.sr), min(int(t1 * self.sr), self.n)
        block = 512
        zi = None
        for i in range(i0, i1, block):
            frac = (i - i0) / max(i1 - i0, 1)
            fc = f0 * (f1 / f0) ** frac
            sos = _sos("lowpass", fc, self.sr, 2)
            if zi is None:
                zi = np.zeros((sos.shape[0], 2, 2))
            for ch in range(2):
                x[ch, i:i + block], zi[:, ch, :] = signal.sosfilt(sos, x[ch, i:i + block], zi=zi[:, ch, :])

    def gate(self, t0: float, t1: float, buses=None, fade_s: float = 0.01) -> None:
        """Silence buses between t0 and t1 (stops/drop-outs)."""
        i0, i1 = int(t0 * self.sr), min(int(t1 * self.sr), self.n)
        f = int(fade_s * self.sr)
        for b in buses or self.buses:
            self.buses[b][:, i0:i1] = 0.0
            if i0 - f > 0:
                self.buses[b][:, i0 - f:i0] *= np.linspace(1, 0, f)[None, :]

    def mix(self, headroom_db: float = -1.0) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        total = sum(self.buses.values())
        peak = float(np.max(np.abs(total))) or 1.0
        g = 10 ** (headroom_db / 20) / peak
        stems = {k: (v * g).astype(np.float32) for k, v in self.buses.items()}
        return (total * g).astype(np.float32), stems
