"""Timbre & mix: descriptors per stem, stereo width, space, sidechain pump, tagging, synthesis hints."""

from __future__ import annotations

import numpy as np
from scipy import ndimage, signal

from . import models
from .util import log

VERSION = "4"
SR = 22050

GENRES = ["house", "deep house", "tech house", "techno", "trance", "drum and bass", "dubstep", "trap", "hip hop",
          "boom bap", "lo-fi hip hop", "r&b", "neo soul", "funk", "disco", "nu disco", "pop", "synthpop", "electropop",
          "indie pop", "rock", "indie rock", "jazz", "ambient", "downtempo", "chillout", "cinematic orchestral",
          "classical", "afrobeats", "reggaeton", "uk garage", "future bass", "synthwave", "edm", "folk", "country",
          "latin", "soul", "gospel", "breakbeat"]
INSTRUMENTS = ["electric piano", "acoustic piano", "synth pad", "synth lead", "synth bass", "sub bass", "808 bass",
               "electric bass", "acoustic guitar", "electric guitar", "strings", "brass", "saxophone", "flute",
               "organ", "vocals", "vocal chops", "choir", "drum machine", "acoustic drums", "hand claps", "shaker",
               "congas", "bells", "marimba", "plucked synth", "arpeggiator", "orchestral percussion", "risers and sweeps"]
MOODS = ["warm", "groovy", "uplifting", "euphoric", "dark", "melancholic", "chill", "energetic", "aggressive",
         "dreamy", "nostalgic", "playful", "tense", "epic", "romantic", "minimal", "lo-fi", "polished"]


# ------------------------------------------------------------------ descriptors
def describe(y: np.ndarray, sr: int = SR, stereo: np.ndarray | None = None, stereo_sr: int = 44100,
             bpm: float | None = None) -> dict:
    import librosa

    if y is None or np.max(np.abs(y)) < 1e-4:
        return {"present": False}
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    rms = librosa.feature.rms(S=S)[0]
    active = rms > 0.1 * np.percentile(rms, 95)
    if active.sum() < 4:
        active = rms > 0
    Sa = S[:, active]
    cent = librosa.feature.spectral_centroid(S=Sa, sr=sr)[0]
    roll = librosa.feature.spectral_rolloff(S=Sa, sr=sr, roll_percent=0.85)[0]
    flat = librosa.feature.spectral_flatness(S=Sa)[0]
    bw = librosa.feature.spectral_bandwidth(S=Sa, sr=sr)[0]
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(librosa.feature.melspectrogram(S=Sa ** 2, sr=sr)), n_mfcc=13)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    spec = (Sa ** 2).mean(axis=1)
    d = {
        "present": True,
        "active_fraction": round(float(active.mean()), 2),
        "centroid_hz": round(float(np.median(cent)), 1),
        "rolloff_hz": round(float(np.median(roll)), 1),
        "flatness": round(float(np.median(flat)), 4),
        "bandwidth_hz": round(float(np.median(bw)), 1),
        "lowpass_hz": _lowpass_estimate(spec, freqs),
        "mfcc_mean": [round(float(x), 2) for x in mfcc.mean(axis=1)],
        "bands": _band_balance(spec, freqs),
    }
    d.update(_envelope_shape(y, sr))
    trem = _tremolo(rms, sr / 512, bpm)
    if trem:
        d["tremolo_hz"] = trem
    if stereo is not None and stereo.shape[0] == 2:
        d["stereo_width"] = stereo_width(stereo)
    return d


def _band_balance(spec: np.ndarray, freqs: np.ndarray) -> dict:
    edges = [("sub", 0, 60), ("low", 60, 250), ("low_mid", 250, 2000), ("high_mid", 2000, 6000), ("high", 6000, 1e9)]
    tot = spec.sum() + 1e-12
    return {name: round(float(spec[(freqs >= lo) & (freqs < hi)].sum() / tot), 3) for name, lo, hi in edges}


def _lowpass_estimate(spec: np.ndarray, freqs: np.ndarray, drop_db: float = 30.0) -> float | None:
    """Frequency above which the (smoothed) spectrum stays > drop_db below its peak."""
    p = 10 * np.log10(spec + 1e-12)
    p = ndimage.uniform_filter1d(p, 9)
    pk = p.max()
    above = np.where(p > pk - drop_db)[0]
    if not len(above):
        return None
    return round(float(freqs[above[-1]]), 0)


def _envelope_shape(y: np.ndarray, sr: int) -> dict:
    """Median attack (10->90 %) and decay (peak -> -20 dB) around onsets."""
    import librosa

    hop = 128
    env = np.abs(signal.hilbert(y)) if len(y) < 2 ** 22 else np.abs(y)
    env = ndimage.uniform_filter1d(env, int(0.003 * sr))
    onsets = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop, units="samples", backtrack=True)
    atts, decs = [], []
    for i, o in enumerate(onsets[:400]):
        nxt = onsets[i + 1] if i + 1 < len(onsets) else len(y)
        seg = env[o: min(nxt, o + int(2.0 * sr))]
        if seg.size < int(0.02 * sr):
            continue
        pk = int(np.argmax(seg[: int(0.25 * sr)]))
        lvl = seg[pk] + 1e-12
        pre = seg[: pk + 1]
        a10 = np.argmax(pre >= 0.1 * lvl)
        a90 = np.argmax(pre >= 0.9 * lvl)
        atts.append((a90 - a10) / sr * 1000)
        post = seg[pk:]
        below = np.where(post < 0.1 * lvl)[0]
        if len(below):
            decs.append(below[0] / sr * 1000)
        else:
            decs.append(len(post) / sr * 1000)  # lower bound: sustained until the next onset
    if not atts:
        return {}
    return {"attack_ms": round(float(np.median(atts)), 1), "decay_ms": round(float(np.median(decs)), 1)}


def _tremolo(rms: np.ndarray, fps: float, bpm: float | None = None) -> float | None:
    """Amplitude modulation 4-9 Hz that is not just the rhythm (tempo multiples are excluded)."""
    if len(rms) < 64:
        return None
    x = np.log(rms + 1e-6)
    x = x - ndimage.uniform_filter1d(x, int(fps))  # remove slow level changes
    f, p = signal.welch(x, fs=fps, nperseg=min(256, len(x)))
    m = (f >= 4) & (f <= 9)
    if bpm:
        for k in (0.5, 1, 1.5, 2, 3, 4, 6):
            m &= np.abs(f - k * bpm / 60.0) > 0.08 * k * bpm / 60.0 + 0.15
    if not m.any():
        return None
    k = int(np.argmax(p[m]))
    if p[m][k] > 12 * np.median(p[(f > 1) & (f < 15)]):
        return round(float(f[m][k]), 1)
    return None


def stereo_width(x: np.ndarray) -> float:
    """0 = mono, ~0.5 = decorrelated stereo, -> 1 = out of phase."""
    m = (x[0] + x[1]) / 2
    s = (x[0] - x[1]) / 2
    em, es = float(np.sqrt(np.mean(m ** 2))), float(np.sqrt(np.mean(s ** 2)))
    return round(es / (em + es + 1e-12), 3)


def space_estimate(y: np.ndarray, sr: int = SR) -> dict:
    """Reverb/space proxy from a percussive signal: how far the >2 kHz envelope falls between hits.
    Dry mixes drop 35+ dB between transients; reverb tails keep the floor up."""
    import librosa

    if y is None or np.max(np.abs(y)) < 1e-4:
        return {"label": "unknown", "tail_db": None}
    sos = signal.butter(4, 2000 / (sr / 2), "highpass", output="sos")
    hf = signal.sosfilt(sos, y)
    hop = 128
    rms = librosa.feature.rms(y=hf, frame_length=512, hop_length=hop)[0]
    db = 20 * np.log10(rms + 1e-9)
    on = librosa.onset.onset_detect(y=hf, sr=sr, hop_length=hop, units="frames")
    fps = sr / hop
    drops, slopes = [], []
    for i, o in enumerate(on):
        nxt = on[i + 1] if i + 1 < len(on) else len(db)
        if nxt - o < int(0.1 * fps):
            continue
        seg = db[o: nxt]
        pk = int(np.argmax(seg[: int(0.03 * fps) + 1]))
        if seg[pk] < db.max() - 40:
            continue
        drops.append(float(np.min(seg[pk:]) - seg[pk]))
        tail = seg[pk + int(0.02 * fps): pk + int(0.25 * fps)]
        if len(tail) > 8:
            slopes.append(np.polyfit(np.arange(len(tail)) / fps, tail, 1)[0])
    if not drops:
        return {"label": "unknown", "tail_db": None}
    tail_db = float(np.median(drops))
    label = "dry" if tail_db < -35 else "short room" if tail_db < -25 else "medium plate/hall" if tail_db < -15 else "long, washed-out"
    rt60 = float(np.clip(-60.0 / np.median(slopes), 0.05, 8.0)) if slopes and np.median(slopes) < -1 else None
    return {"label": label, "tail_db": round(tail_db, 1), "rt60_s": round(rt60, 2) if rt60 else None}


# ------------------------------------------------------------------ sidechain
def sidechain(y: np.ndarray, kick_times: list[float], sr: int = SR) -> dict | None:
    """Average level of a non-drum signal around kick hits: a dip right after the kick that recovers
    over ~50-300 ms is a sidechain pump. Compared against random control times."""
    if y is None or not kick_times or len(kick_times) < 8 or np.max(np.abs(y)) < 1e-4:
        return None
    hop = int(0.004 * sr)
    frame = int(0.012 * sr)
    env = np.sqrt(ndimage.uniform_filter1d(y.astype(np.float64) ** 2, frame)[::hop] + 1e-12)
    db = 20 * np.log10(env + 1e-9)
    fps = sr / hop
    k = np.array(sorted(kick_times))
    gaps = np.diff(k)
    win = int(min(0.45, float(np.median(gaps)) * 0.95) * fps) if len(gaps) else int(0.4 * fps)
    pre = int(0.03 * fps)

    def curve(times):
        segs = []
        for t in times:
            i = int(t * fps)
            if i - pre < 0 or i + win >= len(db):
                continue
            seg = db[i - pre: i + win]
            ref = np.median(db[i - pre: i - int(0.004 * fps)])
            if ref < db.max() - 45:  # silence: no information
                continue
            segs.append(seg - ref)
        return np.median(np.array(segs), axis=0) if len(segs) >= 6 else None

    c = curve(k)
    if c is None:
        return None
    rng = np.random.default_rng(0)
    span = (k.min(), k.max())
    ctrl_t = rng.uniform(span[0], span[1], size=len(k))
    ctrl_t = ctrl_t[np.min(np.abs(ctrl_t[:, None] - k[None, :]), axis=1) > 0.08]
    cc = curve(ctrl_t)
    post = c[pre:]
    j = int(np.argmin(post[: int(0.12 * fps)]))
    depth = float(-post[j])
    ctrl_depth = float(-np.min(cc[pre: pre + int(0.12 * fps)])) if cc is not None else 0.0
    # release: time from the dip until 63 % of the depth has been recovered
    target = post[j] + 0.63 * depth
    rec = np.where(post[j:] >= target)[0]
    release = float(rec[0] / fps * 1000) if len(rec) else float(len(post[j:]) / fps * 1000)
    detected = (depth > 1.5 and depth - ctrl_depth > 1.0 and 0.004 <= j / fps < 0.08 and release >= 25)
    return {"detected": bool(detected), "depth_db": round(depth, 1) if detected else round(max(depth, 0), 1),
            "release_ms": round(release, 0) if detected else None, "dip_at_ms": round(j / fps * 1000, 0),
            "control_depth_db": round(ctrl_depth, 1)}


# ------------------------------------------------------------------ tagging
def _load_clap(name: str, device: str):
    from transformers import ClapModel, ClapProcessor

    proc = ClapProcessor.from_pretrained(name)
    model = ClapModel.from_pretrained(name).eval()
    return model, proc


def clap_tags(y: np.ndarray, sr: int, model_name: str, device: str = "auto", n_windows: int = 6) -> dict:
    import soxr
    import torch

    model, proc = models.get(f"clap:{model_name}", lambda: _load_clap(model_name, device))
    y48 = soxr.resample(y.astype(np.float32), sr, 48000)
    win = 10 * 48000
    if len(y48) <= win:
        chunks = [y48]
    else:
        starts = np.linspace(0, len(y48) - win, n_windows).astype(int)
        chunks = [y48[s: s + win] for s in starts]
    with torch.no_grad():
        a_in = proc(audio=chunks, sampling_rate=48000, return_tensors="pt")
        a_emb = model.get_audio_features(**a_in)
        a_emb = getattr(a_emb, "pooler_output", a_emb)
        a_emb = torch.nn.functional.normalize(a_emb, dim=-1).mean(0, keepdim=True)
        a_emb = torch.nn.functional.normalize(a_emb, dim=-1)
        out = {}
        for cat, labels, tmpl in (("genre", GENRES, "{} music"), ("instruments", INSTRUMENTS, "the sound of {}"),
                                  ("mood", MOODS, "{} music")):
            t_in = proc(text=[tmpl.format(l) for l in labels], return_tensors="pt", padding=True)
            t_emb = model.get_text_features(**t_in)
            t_emb = getattr(t_emb, "pooler_output", t_emb)
            t_emb = torch.nn.functional.normalize(t_emb, dim=-1)
            sims = (a_emb @ t_emb.T)[0]
            probs = torch.softmax(sims * 100.0 / 1.0, dim=0).numpy()
            order = np.argsort(-probs)
            k = 5 if cat != "instruments" else 8
            out[cat] = {labels[i]: round(float(probs[i]), 3) for i in order[:k]}
            out[f"_{cat}_sims"] = {labels[i]: round(float(sims[i]), 4) for i in order[:k]}
    out["embedding"] = a_emb[0].numpy().astype(np.float32)
    return out


def heuristic_tags(analysis: dict) -> dict:
    """Coarse genre/mood/instrument guesses from the analysis itself (used when CLAP is unavailable)."""
    r = analysis.get("rhythm", {})
    bpm = r.get("bpm") or 0
    swing = (r.get("swing") or {}).get("ratio_16th") or (r.get("swing") or {}).get("ratio_8th") or 0.5
    meter = r.get("meter")
    drums = analysis.get("drums", {})
    main = next((p for p in drums.get("patterns", []) if p.get("variation_of") is None), None)
    kick = (main or {}).get("grid_16", {}).get("kick", "")
    four_floor = len(kick) == 16 and all(kick[i] == "x" for i in (0, 4, 8, 12))
    backbeat = any(s and len(s) == 16 and s[4] == "x" and s[12] == "x" for v, s in (main or {}).get("grid_16", {}).items()
                   if v in ("snare", "clap", "rim"))
    mode = (analysis.get("harmony", {}).get("key") or "").split(" ")[-1]
    genre: dict[str, float] = {}
    if meter == "3/4":
        genre["waltz / 3-4 ballad"] = 0.6
    elif four_floor and 118 <= bpm <= 135:
        genre["techno" if swing < 0.53 and mode == "minor" else "house"] = 0.5
        genre["tech house" if swing < 0.53 else "deep house"] = 0.25
    elif four_floor and 105 <= bpm < 118:
        genre["house" if swing >= 0.53 else "nu disco"] = 0.45
        genre["deep house"] = 0.25
    elif backbeat and 70 <= bpm <= 105:
        genre["hip hop" if swing >= 0.55 else "pop / r&b"] = 0.45
        genre["neo soul" if swing >= 0.58 else "pop"] = 0.25
    elif 160 <= bpm <= 180:
        genre["drum and bass"] = 0.4
    elif bpm and bpm < 90 and not kick:
        genre["ambient / downtempo"] = 0.4
    else:
        genre["pop / electronic"] = 0.3
    mood = []
    bright = analysis.get("sound", {}).get("mix", {}).get("brightness")
    mood.append("warm" if bright in ("dark", "mid") else "bright")
    if swing >= 0.55:
        mood.append("groovy")
    mood.append("melancholic" if mode == "minor" and bpm < 100 else "uplifting" if mode == "major" and bpm >= 110 else "laid-back")
    return {"genre": genre, "mood": mood, "method": "heuristic"}


# ------------------------------------------------------------------ synthesis hints
def synthesis_hints(stems_desc: dict, drum_sounds: dict, bass: dict, voicing: dict, mix: dict, side: dict | None) -> dict:
    hints: dict[str, str] = {}
    o = stems_desc.get("other") or {}
    if o.get("present"):
        att, dec, cent = o.get("attack_ms", 10), o.get("decay_ms", 300), o.get("centroid_hz", 1500)
        lp = o.get("lowpass_hz")
        width = o.get("stereo_width")
        style = (voicing or {}).get("style", "")
        if att is not None and att > 60:
            h = f"pad: detuned saws (±8 cents), slow attack ~{att:.0f} ms, low-pass ~{_k(lp)}"
        elif "stab" in style or (dec is not None and dec < 350):
            h = (f"keys/stabs: {'FM electric piano (soft tines, mod index decaying ~120 ms)' if cent < 1800 else 'bright pluck (saw/square, fast filter env)'}"
                 f", attack ~{att:.0f} ms, decay ~{dec:.0f} ms, low-pass ~{_k(lp)}")
        else:
            h = f"chords: sustained, attack ~{att:.0f} ms, centroid {cent:.0f} Hz, low-pass ~{_k(lp)}"
        if o.get("tremolo_hz"):
            h += f", tremolo ~{o['tremolo_hz']} Hz"
        if width is not None:
            h += ", wide stereo (chorus/unison)" if width > 0.25 else ", fairly mono/centred"
        hints["keys"] = h
    b = stems_desc.get("bass") or {}
    if b.get("present"):
        lp = b.get("lowpass_hz")
        flat = b.get("flatness", 0)
        cent = b.get("centroid_hz", 100)
        if cent < 90 and flat < 0.01:
            src = "sine/sub (almost no harmonics)"
        elif cent < 160:
            src = "saw + sine sub, heavily low-passed"
        else:
            src = "saw/square with audible harmonics"
        h = f"bass: {src}, low-pass ~{_k(lp)}"
        if bass and bass.get("glides", 0) >= 0.15:
            h += f", portamento/glide on ~{int(100 * bass['glides'])}% of notes (~40 ms)"
        if bass and bass.get("articulation"):
            h += f", {bass['articulation']} notes"
        if b.get("decay_ms") and b["decay_ms"] > 600 and drum_sounds.get("kick"):
            h += " (long decaying tail: 808-style)"
        hints["bass"] = h
    k = drum_sounds.get("kick")
    if k:
        hints["kick"] = (f"kick: sine body ~{k.get('f0_hz', '?')} Hz swept down from ~{k.get('start_hz', '?')} Hz, "
                         f"{k.get('decay_ms', '?')} ms decay, {k.get('click', 'medium')} click, {k.get('saturation', 'clean')} saturation")
    for v in ("clap", "snare"):
        d = drum_sounds.get(v)
        if d:
            hints[v] = (f"{v}: {'band-passed noise bursts (3 x ~10 ms apart) + short tail' if v == 'clap' else 'tonal body ~' + str(d.get('f0_hz', 190)) + ' Hz + high-passed noise'}"
                        f", decay ~{d.get('decay_ms')} ms, brightness ~{d.get('brightness_hz')} Hz")
    hh = drum_sounds.get("hat_closed") or drum_sounds.get("hat_open")
    if hh:
        hints["hats"] = f"hats: high-passed metallic noise (>6-7 kHz), decay ~{hh.get('decay_ms')} ms, {hh.get('transient')} transient"
    m = []
    if side and side.get("detected"):
        m.append(f"sidechain pump ~{side['depth_db']} dB with ~{side['release_ms']:.0f} ms release on bass/pads")
    if mix.get("space") and mix["space"].get("label") not in (None, "unknown"):
        m.append(f"space: {mix['space'].get('label')}")
    if mix.get("stereo_width") is not None:
        m.append(f"stereo width {mix['stereo_width']}")
    if m:
        hints["mix"] = "; ".join(m)
    return hints


def _k(hz) -> str:
    if hz is None:
        return "?"
    return f"{hz / 1000:.1f} kHz" if hz >= 1000 else f"{hz:.0f} Hz"


def mix_summary(full: dict, stereo: np.ndarray, space: dict, loud: dict) -> dict:
    cent = full.get("centroid_hz", 2000)
    brightness = "dark" if cent < 1200 else "mid" if cent < 2600 else "bright"
    return {"brightness": brightness, "centroid_hz": cent, "stereo_width": stereo_width(stereo) if stereo is not None else None,
            "bands": full.get("bands"), "space": space, "reverb": space.get("label"),
            "crest_factor_db": loud.get("crest_factor_db")}


__all__ = ["describe", "sidechain", "clap_tags", "heuristic_tags", "synthesis_hints", "mix_summary", "space_estimate",
           "stereo_width", "log"]
