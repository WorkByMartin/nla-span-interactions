"""Shared helpers for the worked examples: the store, text layout and a tee.

Not a script. The three example scripts import it, and every number they print
comes from the same reader functions the analysis uses.
"""
from __future__ import annotations

import sqlite3
import sys
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
DB_DEFAULT = HERE.parent.parent / "db" / "ffw_span-ablation_database.sqlite"

SWAP = "corpus-swap/pos+len"
DELETION = "deletion"
SHUFFLE = "shuffle"
MLM = "modernbert-large_filler_model/textsub"

FLOOR = 0.044          # harness noise floor in FVE points

sys.path.insert(0, str(HERE.parent.parent))
import figstyle  # noqa: E402

C_SWAP = figstyle.SWAP
C_MLM = figstyle.MLM
C_DEL = figstyle.DELETION
C_GREY = figstyle.GREY
RESULTS = HERE.parent / "results"

# advance width of DejaVu Sans Mono, in units of the font size
MONO_ADVANCE = 0.6021


def connect(path=DB_DEFAULT):
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def doc_text(conn, doc_id):
    return conn.execute("SELECT text FROM docs WHERE doc_id = ?",
                        (int(doc_id),)).fetchone()["text"]


def baseline(conn, run_id, doc_id):
    r = conn.execute("SELECT base_fve FROM v_baseline WHERE run_id = ? "
                     "AND doc_id = ?", (int(run_id), int(doc_id))).fetchone()
    return float(r["base_fve"])


def singles(conn, run_id, source, doc_id=None, span_id=None):
    """Draw-level rows of a one-substitution arm, as FVE points lost."""
    q = ("SELECT doc_id, span_id, span_text, substitute, depth, draw_idx, "
         "       fve, base_fve, -100.0 * dfve AS effect "
         "FROM v_single WHERE run_id = ? AND source = ?")
    args = [int(run_id), source]
    if doc_id is not None:
        q += " AND doc_id = ?"
        args.append(int(doc_id))
    if span_id is not None:
        q += " AND span_id = ?"
        args.append(int(span_id))
    return [dict(r) for r in conn.execute(q, args)]


def span_pos(conn, doc_id):
    """{span_id: (pos, text, char_start, char_end)} for one document."""
    q = ("SELECT span_id, pos, text, char_start, char_end FROM v_pos "
         "WHERE doc_id = ?")
    return {int(r["span_id"]): (r["pos"], r["text"], int(r["char_start"]),
                                int(r["char_end"]))
            for r in conn.execute(q, (int(doc_id),))}


# ------------------------------------------------------------------ sentences

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


# --------------------------------------------------------------- text drawing

def wrap_with_offsets(text, width):
    """Greedy wrap into [(line, offset of the line's first character)]."""
    lines, pos = [], 0
    words = []
    i = 0
    while i < len(text):
        if text[i].isspace():
            i += 1
            continue
        j = i
        while j < len(text) and not text[j].isspace():
            j += 1
        words.append((text[i:j], i))
        i = j
    cur, cur_off = "", None
    for w, off in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append((cur, cur_off))
            cur, cur_off = w, off
        elif not cur:
            cur, cur_off = w, off
        else:
            cur += " " + w
    if cur:
        lines.append((cur, cur_off))
    return lines


def mono_char_width(ax, fontsize):
    """Width of one monospace character, in axes fractions, measured not assumed."""
    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    probe = ax.text(0, 0, "M" * 50, transform=ax.transAxes,
                    family="monospace", fontsize=fontsize)
    w = probe.get_window_extent(r).width / 50.0
    probe.remove()
    return w / ax.get_window_extent(r).width


def draw_mono_block(ax, text, width, fontsize, highlight=None,
                    colour="#222222", hl_colour="#000000",
                    hl_face="#ffe08a", top=1.0, line_gap=1.45):
    """Draw wrapped monospace text in axes coordinates, one span highlighted.

    `highlight` is a (start, end) character range into `text`. Every character
    of a monospace face has the same advance, so a line is drawn in three
    pieces around the highlight rather than overdrawn, and nothing is measured
    twice.
    """
    import matplotlib.patches as mpatches

    fig = ax.figure
    ax_h_in = ax.get_position().height * fig.get_size_inches()[1]
    cw = mono_char_width(ax, fontsize)
    lh = fontsize * line_gap / 72.0 / ax_h_in
    lines = wrap_with_offsets(text, width)
    for k, (line, off) in enumerate(lines):
        y = top - (k + 0.8) * lh
        a = b = None
        if highlight is not None:
            hs, he = highlight
            a = max(hs - off, 0)
            b = min(he - off, len(line))
            if not (hs < off + len(line) and he > off and b > a):
                a = b = None

        def put(x_char, s, bold=False, col=colour):
            if not s:
                return
            ax.text(x_char * cw, y, s, transform=ax.transAxes,
                    family="monospace", fontsize=fontsize, color=col,
                    va="baseline", ha="left",
                    fontweight="bold" if bold else "normal")

        if a is None:
            put(0, line)
        else:
            put(0, line[:a])
            ax.add_patch(mpatches.Rectangle(
                (a * cw, y - 0.28 * lh), (b - a) * cw, 1.05 * lh,
                transform=ax.transAxes, facecolor=hl_face, edgecolor="none",
                zorder=0))
            put(a, line[a:b], bold=True, col=hl_colour)
            put(b, line[b:])
    return len(lines)


def head_words(text, n):
    w = text.split()
    if len(w) <= n:
        return " ".join(w), False
    return " ".join(w[:n]) + " ...", True


class Tee:
    """Write everything printed to stdout and to a file at once."""

    def __init__(self, path):
        self.f = open(path, "w")

    def write(self, s):
        sys.__stdout__.write(s)
        self.f.write(s)

    def flush(self):
        sys.__stdout__.flush()
        self.f.flush()

    def close(self):
        self.f.close()


def tee_to(path):
    sys.stdout = Tee(path)
    return sys.stdout


def untee():
    sys.stdout.flush()
    sys.stdout.close()
    sys.stdout = sys.__stdout__


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


def wrap_md(text, width=96, indent=""):
    return "\n".join(textwrap.wrap(" ".join(text.split()), width,
                                   initial_indent=indent,
                                   subsequent_indent=indent) or [""])
