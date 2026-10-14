"""The LLM-facing Markdown brief: short, opinionated, in producer language, ordered by what matters to imitate."""

from __future__ import annotations

FOCI = ("groove", "harmony", "sound", "structure", "all")
BAR_NOTE = "Bar numbers in this brief are 1-based (analysis.json uses 0-based indices)."
LEGEND = "`X` accent · `x` hit · `o` ghost · `.` rest"


def _f(x, nd=1, unit=""):
    if x is None:
        return "?"
    return f"{x:.{nd}f}{unit}"


def _ms(x):
    if x is None:
        return "?"
    return f"{'+' if x >= 0 else ''}{x:.0f} ms"


def _main_pattern(a):
    pats = [p for p in a["drums"].get("patterns", []) if p.get("variation_of") is None]
    return max(pats, key=lambda p: p["count"]) if pats else None


def _swing_text(sw):
    if not sw or not sw.get("level") or sw.get("percent") is None:
        return "straight / not measurable"
    lvl = sw["level"]
    pct = sw["percent"]
    if pct < 52.5:
        return f"straight {lvl}s ({pct:.0f}%)"
    return f"{pct:.0f}% swing on {lvl}s ({sw.get('feel')})"


def vibe_line(a: dict) -> str:
    r, h, s = a["rhythm"], a["harmony"], a["sound"]
    bpm = r.get("bpm")
    tags = s.get("tags") or {}
    genre = next(iter((tags.get("genre") or {}).keys()), None)
    sw = r.get("swing") or {}
    feel = ""
    if sw.get("percent") and sw["percent"] >= 53:
        feel = "Swung " if sw["percent"] >= 57 else "Lightly swung "
    else:
        feel = "Straight "
    parts = []
    ds = a["drums"].get("sound", {})
    k = ds.get("kick")
    if k:
        parts.append(f"{'punchy short' if (k.get('decay_ms') or 0) < 250 else 'boomy long'} kick")
    main = _main_pattern(a)
    if main:
        g = main.get("grid_16", {})
        bb = next((v for v in ("clap", "snare", "rim") if v in g and len(g[v]) == 16 and g[v][4] == "x" and g[v][12] == "x"), None)
        if bb:
            parts.append(f"{bb} on 2 & 4")
    voicing = h.get("voicing") or {}
    keys_hint = (s.get("synthesis_hints") or {}).get("keys", "")
    if voicing.get("present"):
        kind = "e-piano" if "electric piano" in keys_hint else "pad" if keys_hint.startswith("pad") else "keys"
        style = voicing.get("style", "chords")
        parts.append(f"{kind} {'stabs' if 'stab' in style else 'chords'}")
    b = a.get("bass") or {}
    if b.get("present"):
        desc = "gliding" if (b.get("glides") or 0) >= 0.15 else "syncopated" if "syncopated" in (b.get("relation_to_kick") or "") else "steady"
        parts.append(f"{desc} bass")
    side = s.get("sidechain") or {}
    tail = ", sidechain pump" if side.get("detected") else ""
    key = h.get("key") or "unknown key"
    head = f"{feel}{_f(bpm, 0)} bpm {genre + ' ' if genre else ''}in {key}"
    return head + (": " + ", ".join(parts) if parts else "") + tail + "."


def _grid_block(p: dict, accents: bool = True) -> str:
    grids = p.get("accents" if accents else "grid_16", {})
    if not grids:
        return ""
    w = max(len(v) for v in grids)
    lines = [f"{v.ljust(w)}  {g}" for v, g in grids.items()]
    return "```\n" + "\n".join(lines) + "\n```"


def section_groove(a: dict) -> list[str]:
    r = a["rhythm"]
    out = ["## Groove"]
    st = r.get("tempo_stability") or {}
    alts = [c for c in (r.get("tempo_candidates") or []) if c["reading"] != "detected" and c["support"] > 0.3]
    alt_txt = f"; also reads as {', '.join(str(c['bpm']) for c in alts)} bpm" if alts else ""
    curve = r.get("bpm_curve") or []
    drift = ""
    if not st.get("steady") and curve:
        drift = f", drifts {_f(st.get('bpm_range', [None])[0], 1)}–{_f(st.get('bpm_range', [None, None])[1], 1)} bpm (live feel)"
    out.append(f"- **Tempo** {_f(r.get('bpm'), 1)} bpm{' (rock steady)' if st.get('steady') else drift}, {r.get('meter')}, "
               f"bar 1 starts at {_f(r.get('first_downbeat'), 2)} s{alt_txt}.")
    sw = r.get("swing") or {}
    byv = sw.get("by_voice") or {}
    swung = [f"{v} {round(x * 100)}%" for v, x in byv.items() if x >= 0.53]
    out.append(f"- **Swing** {_swing_text(sw)}" + (f" — swung voices: {', '.join(swung)}" if swung else "") + ".")
    md = r.get("microtiming_detail") or {}
    if md:
        bits = []
        for v, d in md.items():
            rel = d.get("vs_kick_ms")
            if v == "kick" or rel is None or abs(rel) < 3:
                continue
            bits.append(f"{v} {_ms(rel)} vs kick ({'laid back' if rel > 0 else 'pushed'})")
        if bits:
            out.append("- **Microtiming** " + "; ".join(bits) + ".")
    main = _main_pattern(a)
    if main:
        out.append(f"- **Core loop** pattern {main['id']} ({main['count']} bars) — {LEGEND}:")
        out.append(_grid_block(main))
        others = [p for p in a["drums"].get("patterns", []) if p is not main and p["count"] >= 2]
        for p in others[:3]:
            rel = f"variation of {p['variation_of']}" if p.get("variation_of") else "different groove"
            bars = ", ".join(str(b + 1) for b in p["bars"][:6]) + ("…" if len(p["bars"]) > 6 else "")
            out.append(f"- Pattern {p['id']} ({rel}, {p['count']} bars: {bars}):")
            out.append(_grid_block(p))
    fills = a["drums"].get("fills") or []
    if fills:
        fl = [f"bar {f['bar'] + 1} beat {f['beat']}: {f['kind'].replace('_', ' ')}" + (f" ({f['notes']})" if f.get("notes") else "")
              for f in fills[:6]]
        out.append("- **Fills** " + "; ".join(fl) + ".")
    rolls = a["drums"].get("rolls") or []
    for rl in rolls[:2]:
        out.append(f"- **Roll** bars {rl['start_bar'] + 1}–{rl['end_bar']}: snare density {rl['density_per_bar']} hits/bar (accelerating).")
    side = a["sound"].get("sidechain") or {}
    if side.get("detected"):
        out.append(f"- **Sidechain** ~{side['depth_db']} dB duck, ~{_f(side.get('release_ms'), 0)} ms release on {', '.join(side.get('on', []))}.")
    return out


def section_harmony(a: dict) -> list[str]:
    h = a["harmony"]
    out = ["## Harmony"]
    alts = [k["key"] for k in (h.get("key_alternatives") or [])[1:2]]
    out.append(f"- **Key** {h.get('key')} (confidence {_f(h.get('key_confidence'), 2)}"
               + (f"; runner-up {alts[0]}" if alts else "") + ").")
    loop = h.get("loop") or {}
    if loop.get("symbols"):
        bpc = loop.get("bars_per_chord")
        rate = f"{_f(bpc, 2).rstrip('0').rstrip('.')} bar(s)/chord" if bpc else ""
        out.append(f"- **Loop** ({loop.get('length_bars')} bars, {rate}): `{' – '.join(loop['symbols'])}`  =  "
                   f"`{' – '.join(loop.get('roman') or [])}`")
    hr = h.get("harmonic_rhythm") or {}
    if hr:
        out.append(f"- **Harmonic rhythm** {hr.get('chords_per_bar')} chords/bar; "
                   f"{int(100 * (hr.get('changes_on_downbeat') or 0))}% of changes on the downbeat.")
    voc = h.get("vocabulary") or {}
    if voc:
        out.append("- **Vocabulary** " + ", ".join(f"{('maj' if q == 'maj' else q)} {int(round(100 * w))}%" for q, w in list(voc.items())[:6])
                   + (f"; extensions used: {', '.join(h.get('extensions') or [])}" if h.get("extensions") else "") + ".")
    v = h.get("voicing") or {}
    if v.get("present"):
        from .util import midi_name

        reg = v.get("register_midi") or [None, None]
        inv = v.get("inversions") or {}
        top_inv = next(iter(inv), None)
        out.append(f"- **Voicing/comping** {v.get('style')}, ~{v.get('polyphony')} notes, register "
                   f"{midi_name(reg[0]) if reg[0] else '?'}–{midi_name(reg[1]) if reg[1] else '?'}, spread ~{v.get('spread_semitones')} st"
                   + (f", mostly {top_inv}-position" if top_inv else "")
                   + (f"; colour tones {', '.join(v.get('extensions') or [])}" if v.get("extensions") else "")
                   + f"; comping grid `{v.get('comping_grid_16')}` (syncopation {int(100 * (v.get('syncopation') or 0))}%).")
    return out


def section_bass(a: dict) -> list[str]:
    b = a.get("bass") or {}
    if not b.get("present"):
        return []
    ctu = b.get("chord_tone_use") or {}
    return ["## Bass",
            f"- Grid `{b.get('rhythm_grid_16')}` (most common bar `{b.get('most_common_bar')}`), "
            f"{b.get('notes_per_bar')} notes/bar, register {'–'.join(b.get('range_names') or [])} ({b.get('register')}), "
            f"{b.get('articulation')}.",
            f"- {b.get('relation_to_kick') or 'relation to kick unknown'}; glides {int(100 * (b.get('glides') or 0))}% of notes, "
            f"octave jumps {int(100 * (b.get('octave_jumps') or 0))}%; plays roots {int(100 * (ctu.get('root') or 0))}% of the time."]


def section_sound(a: dict) -> list[str]:
    s = a["sound"]
    out = ["## Sound palette"]
    tags = s.get("tags") or {}
    g = tags.get("genre") or {}
    gtxt = ", ".join(f"{k} {int(round(100 * v))}%" for k, v in list(g.items())[:3])
    inst = ", ".join(tags.get("instruments") or [])
    mood = ", ".join(tags.get("mood") or [])
    out.append(f"- **Tags** ({tags.get('method', '?')}): {gtxt}" + (f" · instruments: {inst}" if inst else "") + (f" · mood: {mood}" if mood else "") + ".")
    for k, v in (s.get("synthesis_hints") or {}).items():
        if k == "mix":
            continue
        out.append(f"- {v}")
    mix = s.get("mix") or {}
    loud = a["track"].get("loudness") or {}
    bands = mix.get("bands") or {}
    btxt = ", ".join(f"{k} {int(round(100 * v))}%" for k, v in bands.items()) if bands else ""
    out.append(f"- **Mix** {mix.get('brightness')} ({_f(mix.get('centroid_hz'), 0)} Hz centroid), width {mix.get('stereo_width')}, "
               f"space: {(mix.get('space') or {}).get('label')}; {_f(loud.get('integrated_lufs'), 1)} LUFS integrated, "
               f"LRA {_f(loud.get('lra'), 1)} LU, true peak {_f(loud.get('true_peak_dbtp'), 1)} dBTP"
               + (f"; band energy {btxt}" if btxt else "") + ".")
    if (s.get("synthesis_hints") or {}).get("mix"):
        out.append(f"- {s['synthesis_hints']['mix']}")
    return out


def _bar(e: float) -> str:
    blocks = "▁▂▃▄▅▆▇█"
    return blocks[min(7, max(0, int(e * 8)))]


def section_structure(a: dict) -> list[str]:
    st = a["structure"]
    secs = st.get("sections") or []
    out = ["## Arrangement & energy"]
    if not secs:
        return out + ["- (no sections detected)"]
    out.append("| bars | section | energy | layers |")
    out.append("|---|---|---|---|")
    for s in secs[:14]:
        out.append(f"| {s['start_bar'] + 1}–{s['end_bar']} | {s.get('label')} ({s.get('group')}) | {_bar(s['energy'])} {s['energy']:.2f} | "
                   f"{', '.join(s.get('layers') or []) or '—'} |")
    if len(secs) > 14:
        out.append(f"| … | {len(secs) - 14} more sections | | |")
    ev = []
    for e in st.get("events") or []:
        t = e["type"]
        if t == "build":
            ev.append(f"build bars {e['start_bar'] + 1}–{e['end_bar']} ({', '.join(e.get('technique', []))})")
        elif t == "drop":
            ev.append(f"drop at bar {e['bar'] + 1}" + (" (kick returns)" if e.get("kick_returns") else ""))
        elif t == "stop":
            ev.append(f"full stop {e['beats']} beat(s) in bar {e['bar'] + 1}")
        elif t == "drum_stop":
            ev.append(f"drums out bars {e['start_bar'] + 1}–{e['end_bar']} (" + (", ".join(e.get("what_continues") or []) or "nothing") + " continues)")
        elif t == "climax":
            ev.append(f"climax at bar {e['bar'] + 1}")
    if ev:
        out.append("- **Events** " + "; ".join(ev) + ".")
    return out


def imitation_steps(a: dict) -> list[str]:
    r, h, s = a["rhythm"], a["harmony"], a["sound"]
    steps = []
    sw = r.get("swing") or {}
    swing_txt = f", {sw['level']} swing {sw['percent']:.0f}%" if sw.get("percent") and sw["percent"] >= 52.5 else ", straight grid"
    steps.append(f"Set {_f(r.get('bpm'), 1)} bpm in {r.get('meter')}{swing_txt}"
                 + (f" (apply to {', '.join(v for v, x in (sw.get('by_voice') or {}).items() if x >= 0.53)})" if sw.get("by_voice") else "") + ".")
    main = _main_pattern(a)
    if main:
        g = main.get("accents", {})
        core = "; ".join(f"{v} `{p}`" for v, p in list(g.items())[:4])
        steps.append(f"Program the core drum loop: {core}.")
    md = r.get("microtiming_detail") or {}
    lag = [(v, d["vs_kick_ms"]) for v, d in md.items() if d.get("vs_kick_ms") is not None and abs(d["vs_kick_ms"]) >= 3 and v != "kick"]
    if lag:
        steps.append("Nudge timing: " + ", ".join(f"{v} {_ms(x)}" for v, x in lag) + " relative to the kick.")
    fills = a["drums"].get("fills") or []
    if fills:
        kinds = sorted({f["kind"].replace("_", " ") for f in fills})
        steps.append(f"Add fills on phrase ends ({', '.join(kinds)}) on the last beat before new sections.")
    loop = h.get("loop") or {}
    v = h.get("voicing") or {}
    if loop.get("symbols"):
        steps.append(f"Loop `{' – '.join(loop['symbols'])}` ({' – '.join(loop.get('roman') or [])} in {h.get('key')})"
                     + (f" as {v.get('style')} on `{v.get('comping_grid_16')}`" if v.get("present") else "") + ".")
    b = a.get("bass") or {}
    if b.get("present"):
        steps.append(f"Bass on `{b.get('rhythm_grid_16')}`, roots in {'–'.join(b.get('range_names') or [])}, "
                     f"{b.get('relation_to_kick') or ''}" + (f", glide into ~{int(100 * b['glides'])}% of notes" if (b.get("glides") or 0) >= 0.15 else "") + ".")
    hints = s.get("synthesis_hints") or {}
    sd = [hints[k] for k in ("kick", "clap", "snare", "keys", "bass") if k in hints][:3]
    if sd:
        steps.append("Sound design — " + " | ".join(sd) + ".")
    side = s.get("sidechain") or {}
    loud = a["track"].get("loudness") or {}
    mixbits = []
    if side.get("detected"):
        mixbits.append(f"sidechain bass/pads to the kick (~{side['depth_db']} dB, ~{_f(side.get('release_ms'), 0)} ms release)")
    space = ((s.get("mix") or {}).get("space") or {}).get("label")
    if space and space != "unknown":
        mixbits.append(f"keep the space {space}")
    if loud.get("integrated_lufs") is not None:
        mixbits.append(f"master around {loud['integrated_lufs']:.0f} LUFS")
    if mixbits:
        steps.append("Mix: " + ", ".join(mixbits) + ".")
    secs = a["structure"].get("sections") or []
    if secs:
        plan = " → ".join(f"{x.get('label')} {x['bars']}" for x in secs[:10])
        ev = a["structure"].get("events") or []
        builds = [e for e in ev if e["type"] == "build"]
        tech = sorted({t for e in builds for t in e.get("technique", [])})
        steps.append(f"Arrange (bars): {plan}" + (f"; builds use {', '.join(tech)}" if tech else "") + ".")
    return steps[:10]


def make_brief(a: dict, focus: str = "all") -> str:
    if focus not in FOCI:
        raise ValueError(f"focus must be one of {FOCI}")
    t = a["track"]
    r = a["rhythm"]
    head = [f"# Reference brief — {t.get('name')}",
            f"_{_f(t.get('duration_s'), 1)} s · {_f(r.get('bpm'), 1)} bpm · {r.get('meter')} · {a['harmony'].get('key')} · "
            f"depth {a['generator'].get('depth')} · id `{t.get('id')}`_", "",
            f"**Vibe:** {vibe_line(a)}", ""]
    body: list[str] = []
    if focus in ("groove", "all"):
        body += section_groove(a) + [""]
    if focus in ("harmony", "all"):
        body += section_harmony(a) + [""] + section_bass(a) + ([""] if section_bass(a) else [])
    if focus in ("sound", "all"):
        body += section_sound(a) + [""]
    if focus in ("structure", "all"):
        body += section_structure(a) + [""]
    if focus == "all":
        body.append("## To imitate this, do:")
        body += [f"{i}. {s}" for i, s in enumerate(imitation_steps(a), 1)]
        body.append("")
    warn = a.get("warnings") or []
    if warn:
        body.append("## Uncertain")
        body += [f"- {w}" for w in warn[:8]]
        body.append("")
    body.append(f"_{BAR_NOTE} Full data: `analysis.json`; MIDI in `midi/`._")
    return "\n".join(head + body) + "\n"
