"""PNG plots for humans (and agents that can read images): spectrogram, chroma, energy, beat grid, drums, layers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PC = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
SEC_COLORS = {"intro": "#9ecae1", "groove": "#a1d99b", "verse": "#a1d99b", "build": "#fdae6b", "drop": "#fb6a4a",
              "chorus": "#fb6a4a", "breakdown": "#bcbddc", "outro": "#d9d9d9"}


def _sections(ax, analysis, ymax=None):
    for s in analysis["structure"].get("sections", []):
        ax.axvspan(s["start"], s["end"], color=SEC_COLORS.get(s.get("label"), "#eeeeee"), alpha=0.25, lw=0)
        ax.text(s["start"], (ymax if ymax is not None else ax.get_ylim()[1]), f" {s.get('label')}", fontsize=7,
                va="top", ha="left")


def render_all(folder: Path, analysis: dict, mono: np.ndarray, sr: int, harmony_stage: dict) -> dict[str, str]:
    import librosa
    import librosa.display  # noqa: F401

    folder.mkdir(parents=True, exist_ok=True)
    out = {}
    dur = len(mono) / sr
    downs = np.asarray(analysis["rhythm"].get("downbeats") or [])

    # spectrogram
    fig, ax = plt.subplots(figsize=(12, 3.6), dpi=100)
    M = librosa.power_to_db(librosa.feature.melspectrogram(y=mono, sr=sr, n_mels=96, hop_length=1024), ref=np.max)
    ax.imshow(M, origin="lower", aspect="auto", extent=[0, dur, 0, 96], cmap="magma")
    for d in downs[::4]:
        ax.axvline(d, color="w", lw=0.3, alpha=0.5)
    _sections(ax, analysis, 96)
    ax.set_title("Mel spectrogram (lines every 4 bars)")
    ax.set_xlabel("s")
    ax.set_yticks([])
    out["spectrogram"] = _save(fig, folder / "spectrogram.png")

    # chromagram with chord labels
    arr = (harmony_stage.get("_arrays") or {})
    ch = arr.get("treble_chroma")
    bt = arr.get("beat_times")
    if ch is not None and bt is not None and len(bt) > 1:
        ch = np.asarray(ch)
        bt = np.asarray(bt)
        fig, ax = plt.subplots(figsize=(12, 3.2), dpi=100)
        chn = ch / (ch.max(axis=0, keepdims=True) + 1e-9)
        edges = np.concatenate([bt, [bt[-1] + np.median(np.diff(bt))]])
        ax.pcolormesh(edges, np.arange(13), chn, cmap="Greys", shading="flat")
        ax.set_yticks(np.arange(12) + 0.5)
        ax.set_yticklabels(PC, fontsize=7)
        last = None
        for c in analysis["harmony"].get("chords", []):
            if c["symbol"] != last:
                ax.text(c["start"], 12.2, c["symbol"], fontsize=6, rotation=60, va="bottom")
                last = c["symbol"]
        ax.set_ylim(0, 14.5)
        ax.set_title(f"Chroma (beat-synchronous) - key {analysis['harmony'].get('key')}", fontsize=9)
        out["chromagram"] = _save(fig, folder / "chromagram.png")

    # energy curve + events + loudness
    stru = analysis["structure"]
    eb = np.asarray(stru.get("energy_per_beat") or [])
    beats = np.asarray(analysis["rhythm"].get("beats") or [])
    if len(eb) and len(beats) >= len(eb):
        fig, ax = plt.subplots(figsize=(12, 3.2), dpi=100)
        ax.plot(beats[: len(eb)], eb, lw=1, color="k", label="energy")
        lc = stru.get("loudness_curve_lufs") or []
        if lc:
            t, l = np.array(lc).T
            ax2 = ax.twinx()
            ax2.plot(t, l, color="tab:blue", lw=0.8, alpha=0.7)
            ax2.set_ylabel("LUFS (short-term)", fontsize=7, color="tab:blue")
        ax.set_ylim(0, 1.15)
        _sections(ax, analysis, 1.15)
        for e in stru.get("events", []):
            t = e.get("time", e.get("start"))
            if t is None:
                continue
            ax.axvline(t, color={"drop": "r", "build": "orange", "stop": "purple", "climax": "magenta",
                                 "drum_stop": "grey"}.get(e["type"], "k"), lw=1, ls="--")
            ax.text(t, 1.02, e["type"], fontsize=6, rotation=90, va="bottom")
        ax.set_title("Energy per beat, sections and events")
        out["energy"] = _save(fig, folder / "energy.png")

    # beat grid (first 8 bars) with drum hits
    if len(downs) >= 2:
        t1 = downs[min(8, len(downs) - 1)]
        t0 = downs[0]
        fig, ax = plt.subplots(figsize=(12, 3), dpi=100)
        env = librosa.onset.onset_strength(y=mono[int(t0 * sr): int(t1 * sr)], sr=sr, hop_length=256)
        tt = t0 + np.arange(len(env)) * 256 / sr
        ax.plot(tt, env / (env.max() + 1e-9), color="0.6", lw=0.8)
        for b in beats[(beats >= t0) & (beats < t1)]:
            ax.axvline(b, color="tab:blue", lw=0.6, alpha=0.6)
        for d in downs[(downs >= t0) & (downs < t1)]:
            ax.axvline(d, color="tab:red", lw=1.2)
        ax.set_xlim(t0, t1)
        ax.set_title("Beat grid (first 8 bars): blue = beats, red = downbeats")
        out["beat_grid"] = _save(fig, folder / "beat_grid.png")

    # main drum pattern heatmap
    pats = [p for p in analysis["drums"].get("patterns", []) if p.get("variation_of") is None]
    if pats:
        p = max(pats, key=lambda p: p["count"])
        voices = list(p["velocity"].keys())
        if voices:
            V = np.array([p["velocity"][v] for v in voices])
            fig, ax = plt.subplots(figsize=(8, 0.4 * len(voices) + 1.2), dpi=100)
            ax.imshow(V, cmap="Greens", vmin=0, vmax=1, aspect="auto")
            ax.set_yticks(range(len(voices)))
            ax.set_yticklabels(voices, fontsize=8)
            ax.set_xticks(range(V.shape[1]))
            ax.set_xticklabels([str(i % 4 + 1) if i % 4 == 0 else "" for i in range(V.shape[1])], fontsize=7)
            for x in range(0, V.shape[1], 4):
                ax.axvline(x - 0.5, color="k", lw=0.5)
            ax.set_title(f"Drum pattern {p['id']} ({p['count']} bars)", fontsize=9)
            out["drum_grid"] = _save(fig, folder / "drum_grid.png")

    # layer chart
    lay = stru.get("layers_per_bar") or {}
    if lay:
        names = list(lay.keys())
        L = np.array([lay[n] for n in names])
        fig, ax = plt.subplots(figsize=(12, 0.35 * len(names) + 1.2), dpi=100)
        ax.imshow(L, cmap="Blues", aspect="auto", vmin=0, vmax=1, interpolation="nearest")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("bar")
        for s in stru.get("sections", []):
            ax.axvline(s["start_bar"] - 0.5, color="r", lw=0.8)
        ax.set_title("Layers per bar (red = section boundaries)", fontsize=9)
        out["layers"] = _save(fig, folder / "layers.png")
    return out


def _save(fig, path: Path) -> str:
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return str(path)
