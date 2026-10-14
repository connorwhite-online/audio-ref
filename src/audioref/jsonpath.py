"""A small JSONPath subset for slicing analysis.json.

Supported: `$` (optional), `.key`, `['key']`, `[n]` (negative ok), `[a:b]`, `[*]`, `.*`,
`[?(@.field == value)]` / `!=` / `<` / `>` / `<=` / `>=` (numbers, quoted strings, true/false/null).
Examples: `rhythm.bpm`, `$.harmony.chords[0:8]`, `structure.sections[*].label`, `drums.patterns[?(@.id=='A')]`.
"""

from __future__ import annotations

import json
import re
from typing import Any

_TOKEN = re.compile(
    r"""\.\*|\.\.|\.([A-Za-z_][\w-]*)|\[\s*'([^']*)'\s*\]|\[\s*"([^"]*)"\s*\]|\[\s*(-?\d*)\s*:\s*(-?\d*)\s*\]"""
    r"""|\[\s*(-?\d+)\s*\]|\[\s*\*\s*\]|\[\s*\?\(\s*@\.([\w.]+)\s*(==|!=|<=|>=|<|>)\s*(.+?)\s*\)\s*\]""")

_MISSING = object()


class JSONPathError(ValueError):
    pass


def _tokens(path: str) -> list[tuple]:
    p = path.strip()
    if p.startswith("$"):
        p = p[1:]
    if p and not p.startswith((".", "[")):
        p = "." + p
    out = []
    pos = 0
    while pos < len(p):
        m = _TOKEN.match(p, pos)
        if not m:
            raise JSONPathError(f"cannot parse JSONPath near '{p[pos:]}' in '{path}'")
        g = m.groups()
        tok = m.group(0)
        if tok == "..":
            raise JSONPathError("recursive descent '..' is not supported")
        if tok == ".*" or re.fullmatch(r"\[\s*\*\s*\]", tok):
            out.append(("wild",))
        elif g[0] or g[1] is not None or g[2] is not None:
            out.append(("key", g[0] or (g[1] if g[1] is not None else g[2])))
        elif g[5] is not None:
            out.append(("index", int(g[5])))
        elif g[3] is not None or g[4] is not None:
            out.append(("slice", int(g[3]) if g[3] else None, int(g[4]) if g[4] else None))
        elif g[6]:
            out.append(("filter", g[6], g[7], _literal(g[8])))
        pos = m.end()
    return out


def _literal(s: str) -> Any:
    s = s.strip()
    if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')):
        return s[1:-1]
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return s


def _get_field(obj: Any, dotted: str) -> Any:
    for part in dotted.split("."):
        if isinstance(obj, dict) and part in obj:
            obj = obj[part]
        else:
            return _MISSING
    return obj


def _cmp(a: Any, op: str, b: Any) -> bool:
    try:
        return {"==": a == b, "!=": a != b, "<": a < b, ">": a > b, "<=": a <= b, ">=": a >= b}[op]
    except TypeError:
        return False


def query(data: Any, path: str) -> Any:
    """Evaluate a path. Returns a single value, or a list when the path contains a wildcard/slice/filter."""
    toks = _tokens(path)
    multi = any(t[0] in ("wild", "slice", "filter") for t in toks)
    nodes = [data]
    for t in toks:
        nxt = []
        for n in nodes:
            if t[0] == "key":
                if isinstance(n, dict) and t[1] in n:
                    nxt.append(n[t[1]])
            elif t[0] == "index":
                if isinstance(n, list) and -len(n) <= t[1] < len(n):
                    nxt.append(n[t[1]])
            elif t[0] == "slice":
                if isinstance(n, list):
                    nxt.extend(n[t[1]:t[2]])
            elif t[0] == "wild":
                if isinstance(n, list):
                    nxt.extend(n)
                elif isinstance(n, dict):
                    nxt.extend(n.values())
            elif t[0] == "filter":
                if isinstance(n, list):
                    for item in n:
                        v = _get_field(item, t[1])
                        if v is not _MISSING and _cmp(v, t[2], t[3]):
                            nxt.append(item)
        nodes = nxt
    if multi:
        return nodes
    if not nodes:
        raise KeyError(path)
    return nodes[0]


def select(data: Any, paths: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for p in paths:
        try:
            out[p] = query(data, p)
        except KeyError:
            out[p] = None
        except JSONPathError as e:
            out[p] = {"error": str(e)}
    return out
