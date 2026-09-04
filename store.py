"""The store's path, and the text layout of the reports printed beside it.

Not a script. `DB_DEFAULT` is where the database sits, `sentence_bounds` cuts
the sentence around a span out of a document, and `md_table` and `wrap_text`
lay out what a script prints.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parent
DB_DEFAULT = REPO / "db" / "ffw_span-ablation_database.sqlite"

_STOPS = ".!?"
_TRAIL = " \"'”’)"


def sentence_bounds(text, a, b):
    """Character bounds of the sentence containing [a, b).

    A full stop only closes a sentence when what follows it, after any closing
    quote, is a capital or a line break, so an abbreviation inside the sentence
    does not split it.
    """
    start = 0
    for i in range(a - 1, -1, -1):
        ch = text[i]
        if ch == "\n":
            start = i + 1
            break
        if ch in _STOPS:
            j = i + 1
            while j < len(text) and text[j] in _TRAIL:
                j += 1
            if j >= len(text) or text[j].isupper() or text[j] == "\n":
                start = i + 1
                break
    end = len(text)
    for i in range(b, len(text)):
        ch = text[i]
        if ch == "\n":
            end = i
            break
        if ch in _STOPS:
            j = i + 1
            while j < len(text) and text[j] in _TRAIL:
                j += 1
            if j >= len(text) or text[j].isupper() or text[j] == "\n":
                end = j
                break
    while start < a and text[start] in " \n":
        start += 1
    while end > b and text[end - 1] in " \n":
        end -= 1
    return start, end


def md_table(header, rows, aligns=None):
    """A markdown table as a string, columns padded to their widths."""
    cols = len(header)
    aligns = aligns or ["l"] * cols
    w = [len(h) for h in header]
    for r in rows:
        for i, c in enumerate(r):
            w[i] = max(w[i], len(str(c)))

    def line(cells):
        out = []
        for i, c in enumerate(cells):
            c = str(c)
            out.append(c.ljust(w[i]) if aligns[i] == "l" else c.rjust(w[i]))
        return "| " + " | ".join(out) + " |"

    sep = "|" + "|".join(("-" * (w[i] + 2)) if aligns[i] == "l"
                         else ("-" * (w[i] + 1)) + ":" for i in range(cols)) + "|"
    return "\n".join([line(header), sep] + [line(r) for r in rows])


def wrap_text(text, width=96, indent=""):
    """Collapse whitespace and wrap to `width`, every line given `indent`."""
    return "\n".join(textwrap.wrap(" ".join(text.split()), width,
                                   initial_indent=indent,
                                   subsequent_indent=indent) or [""])
