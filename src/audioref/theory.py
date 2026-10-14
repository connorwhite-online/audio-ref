"""Music theory helpers: chord qualities, symbol parsing/spelling, keys, Roman numerals."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .util import PC_FLAT, PC_SHARP, name_to_pc

# quality -> intervals in semitones (extensions above the octave kept as >12)
QUALITIES: dict[str, tuple[int, ...]] = {
    "": (0, 4, 7),
    "m": (0, 3, 7),
    "dim": (0, 3, 6),
    "aug": (0, 4, 8),
    "sus2": (0, 2, 7),
    "sus4": (0, 5, 7),
    "6": (0, 4, 7, 9),
    "m6": (0, 3, 7, 9),
    "7": (0, 4, 7, 10),
    "maj7": (0, 4, 7, 11),
    "m7": (0, 3, 7, 10),
    "mmaj7": (0, 3, 7, 11),
    "m7b5": (0, 3, 6, 10),
    "dim7": (0, 3, 6, 9),
    "7sus4": (0, 5, 7, 10),
    "add9": (0, 4, 7, 14),
    "madd9": (0, 3, 7, 14),
    "9": (0, 4, 7, 10, 14),
    "maj9": (0, 4, 7, 11, 14),
    "m9": (0, 3, 7, 10, 14),
    "11": (0, 7, 10, 14, 17),
    "m11": (0, 3, 7, 10, 14, 17),
    "13": (0, 4, 7, 10, 14, 21),
    "maj13": (0, 4, 7, 11, 14, 21),
    "m13": (0, 3, 7, 10, 14, 21),
    "7#9": (0, 4, 7, 10, 15),
    "7b9": (0, 4, 7, 10, 13),
    "6/9": (0, 4, 7, 9, 14),
    "5": (0, 7),
}

# triad family used for Roman numeral case and for coarse (majmin) evaluation
FAMILY = {
    "": "maj", "6": "maj", "7": "maj", "maj7": "maj", "add9": "maj", "9": "maj", "maj9": "maj",
    "11": "maj", "13": "maj", "maj13": "maj", "7#9": "maj", "7b9": "maj", "6/9": "maj", "5": "maj",
    "m": "min", "m6": "min", "m7": "min", "mmaj7": "min", "madd9": "min", "m9": "min", "m11": "min", "m13": "min",
    "dim": "dim", "m7b5": "dim", "dim7": "dim",
    "aug": "aug",
    "sus2": "sus", "sus4": "sus", "7sus4": "sus",
}

ALIASES = {
    "maj": "", "M": "", "major": "", "min": "m", "-": "m", "minor": "m", "mi": "m",
    "M7": "maj7", "Δ": "maj7", "Δ7": "maj7", "ma7": "maj7", "maj7": "maj7", "min7": "m7", "-7": "m7", "mi7": "m7",
    "M9": "maj9", "ma9": "maj9", "Δ9": "maj9", "min9": "m9", "-9": "m9", "min11": "m11", "-11": "m11",
    "ø": "m7b5", "ø7": "m7b5", "m7-5": "m7b5", "°": "dim", "o": "dim", "°7": "dim7", "o7": "dim7",
    "sus": "sus4", "7sus": "7sus4", "+": "aug", "dom7": "7", "min6": "m6", "-6": "m6", "M13": "maj13",
    "69": "6/9", "add2": "add9", "2": "sus2", "4": "sus4",
}

MAJOR_SCALE = (0, 2, 4, 5, 7, 9, 11)
MINOR_SCALE = (0, 2, 3, 5, 7, 8, 10)

# which tonics prefer flat spelling
FLAT_MAJOR = {5, 10, 3, 8, 1}  # F Bb Eb Ab Db
FLAT_MINOR = {2, 7, 0, 5, 10, 3}  # D G C F Bb Eb


@dataclass(frozen=True)
class Chord:
    root: int  # pitch class 0-11
    quality: str  # key of QUALITIES
    bass: int | None = None  # slash bass pitch class (None = root)

    @property
    def intervals(self) -> tuple[int, ...]:
        return QUALITIES[self.quality]

    @property
    def pitch_classes(self) -> set[int]:
        return {(self.root + i) % 12 for i in self.intervals}

    @property
    def family(self) -> str:
        return FAMILY.get(self.quality, "maj")

    def symbol(self, flats: bool = False) -> str:
        s = (PC_FLAT if flats else PC_SHARP)[self.root] + self.quality
        if self.bass is not None and self.bass != self.root:
            s += "/" + (PC_FLAT if flats else PC_SHARP)[self.bass]
        return s

    def voicing(self, center: int = 60, spread: bool = False) -> list[int]:
        """MIDI notes for a playable voicing around `center` (root position)."""
        root_midi = center - 12 + ((self.root - (center - 12)) % 12)
        if root_midi > center - 5:
            root_midi -= 12
        notes = [root_midi + i for i in self.intervals]
        if spread and len(notes) >= 4:
            notes = [notes[0]] + [n + 12 if n - notes[0] < 7 and k > 1 else n for k, n in enumerate(notes[1:], 1)]
        return sorted(set(notes))


_SYMBOL_RE = re.compile(r"^\s*([A-Ga-g])([#b♯♭]?)(.*?)(?:/([A-Ga-g][#b♯♭]?))?\s*$")


def parse_chord(symbol: str) -> Chord | None:
    """Parse 'Bbmaj9', 'F#m7b5', 'C/E', 'N' (no chord -> None)."""
    if not symbol or symbol.strip().upper() in ("N", "NC", "N.C.", "X"):
        return None
    m = _SYMBOL_RE.match(symbol)
    if not m:
        raise ValueError(f"unparseable chord symbol: {symbol!r}")
    root = name_to_pc(m.group(1).upper() + (m.group(2) or ""))
    q = (m.group(3) or "").strip()
    q = ALIASES.get(q, q)
    if q not in QUALITIES:
        # tolerate e.g. 'min7' 'maj' variants with parentheses
        q2 = q.replace("(", "").replace(")", "")
        q = ALIASES.get(q2, q2)
        if q not in QUALITIES:
            raise ValueError(f"unknown chord quality in {symbol!r}")
    bass = name_to_pc(m.group(4)) if m.group(4) else None
    return Chord(root, q, bass)


@dataclass(frozen=True)
class Key:
    tonic: int
    mode: str  # "major" | "minor"

    @property
    def flats(self) -> bool:
        return self.tonic in (FLAT_MAJOR if self.mode == "major" else FLAT_MINOR)

    @property
    def scale(self) -> tuple[int, ...]:
        return MAJOR_SCALE if self.mode == "major" else MINOR_SCALE

    @property
    def pitch_classes(self) -> set[int]:
        return {(self.tonic + i) % 12 for i in self.scale}

    @property
    def name(self) -> str:
        return f"{(PC_FLAT if self.flats else PC_SHARP)[self.tonic]} {self.mode}"

    def relative(self) -> "Key":
        if self.mode == "major":
            return Key((self.tonic + 9) % 12, "minor")
        return Key((self.tonic + 3) % 12, "major")

    def spell(self, pc: int) -> str:
        return (PC_FLAT if self.flats else PC_SHARP)[pc % 12]


def parse_key(name: str) -> Key:
    parts = name.replace(":", " ").split()
    tonic = name_to_pc(parts[0])
    mode = "minor" if len(parts) > 1 and parts[1].lower().startswith("min") or parts[0].endswith("m") and len(parts) == 1 else "major"
    return Key(tonic, mode)


_NUMERALS = ["I", "II", "III", "IV", "V", "VI", "VII"]


def degree_label(key: Key, pc: int) -> str:
    """Scale-degree label with accidentals relative to the key's own scale (e.g. 'b7', '4', '#4')."""
    d = (pc - key.tonic) % 12
    scale = key.scale
    if d in scale:
        return str(scale.index(d) + 1)
    # non-diatonic: prefer flat of the degree above in major, sharp of the degree below in minor for 4/6/7
    if key.mode == "major":
        above = {1: 2, 3: 3, 6: 5, 8: 6, 10: 7}
        if d == 6:
            return "#4"
        return "b" + str(above[d])
    sharp_of = {4: 3, 6: 4, 9: 6, 11: 7}
    if d in sharp_of:
        return "#" + str(sharp_of[d])
    return "b2"


def roman(chord: Chord, key: Key) -> str:
    deg = degree_label(key, chord.root)
    acc = "".join(c for c in deg if c in "#b")
    num = _NUMERALS[int(deg.strip("#b")) - 1]
    fam = chord.family
    if fam in ("min", "dim"):
        num = num.lower()
    q = chord.quality
    if q in ("", "m"):
        suffix = ""
    elif q == "dim":
        suffix = "°"
    elif q == "dim7":
        suffix = "°7"
    elif q == "m7b5":
        suffix = "ø7"
    elif q == "aug":
        suffix = "+"
    else:
        suffix = q
    out = f"{acc}{num}{suffix}"
    if chord.bass is not None and chord.bass != chord.root:
        out += "/" + degree_label(key, chord.bass)
    return out


def spell_chord(chord: Chord | None, key: Key | None) -> str:
    if chord is None:
        return "N"
    return chord.symbol(flats=key.flats if key else False)


def chord_from_roman(numeral: str, key: Key) -> Chord:
    """Inverse of roman() for plain numerals like 'IVmaj9', 'iim7', 'bVII', 'V13' (used by tests/render)."""
    m = re.match(r"^([#b]?)([ivIV]+)(.*)$", numeral)
    if not m:
        raise ValueError(numeral)
    acc, num, suf = m.groups()
    idx = [n.lower() for n in _NUMERALS].index(num.lower())
    pc = (key.tonic + key.scale[idx] + (1 if acc == "#" else -1 if acc == "b" else 0)) % 12
    minor = num.islower()
    if suf in ("", None):
        q = "m" if minor else ""
    elif suf == "°":
        q = "dim"
    elif suf == "ø7":
        q = "m7b5"
    elif suf == "+":
        q = "aug"
    else:
        q = suf if suf in QUALITIES else ("m" + suf if minor and "m" + suf in QUALITIES else suf)
    return Chord(pc, q)
