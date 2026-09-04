"""One figure style for every experiment. Import and call apply() before drawing.

Not a script. Colours, font sizes, the harness floor shading and the save
routine live here so that every figure in the repo is drawn the same way.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# the three ablation arms, and neutral greys
SWAP = "#0072B2"
MLM = "#E69F00"
DELETION = "#009E73"
GREY = "#8c8c8c"
INK = "#222222"
FLOOR_FACE = "#e8e8e8"
HIGHLIGHT = "#ffe08a"

FLOOR = 0.044          # harness reproduction floor, FVE points
DPI = 150

_RC = {
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "x",
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": DPI,
}


def apply():
    matplotlib.rcParams.update(_RC)


# Universal Dependencies names for the spaCy coarse tags, lower case.
POS_NAME = {
    "ADJ": "adjective",
    "ADP": "adposition",
    "ADV": "adverb",
    "AUX": "auxiliary",
    "CCONJ": "coordinating conjunction",
    "DET": "determiner",
    "INTJ": "interjection",
    "NOUN": "noun",
    "NUM": "numeral",
    "PART": "particle",
    "PRON": "pronoun",
    "PROPN": "proper noun",
    "SCONJ": "subordinating conjunction",
    "VERB": "verb",
}


def pos_name(label):
    """A class label with its coarse tags replaced by their UD names.

    Handles both a bare tag and the pooled `other (TAG, TAG)` label that
    swap_analysis.class_groups builds for the small classes.
    """
    label = str(label)
    if label in POS_NAME:
        return POS_NAME[label]
    if label.startswith("other (") and label.endswith(")"):
        tags = [t.strip() for t in label[len("other ("):-1].split(",")]
        return "other (" + ", ".join(POS_NAME.get(t, t) for t in tags) + ")"
    return label


# zorder: row shading and other backdrops at 0, the floor band above them at
# FLOOR_Z so it is never hidden, the zero line just above that, data at 3+.
FLOOR_Z = 1.5


def floor(ax, axis="x"):
    """Shade the harness floor either side of zero and draw the zero line."""
    span = ax.axvspan if axis == "x" else ax.axhspan
    line = ax.axvline if axis == "x" else ax.axhline
    span(-FLOOR, FLOOR, color=FLOOR_FACE, zorder=FLOOR_Z, alpha=0.9, lw=0)
    line(0.0, color="#666666", lw=0.9, zorder=FLOOR_Z + 0.1)


def floor_handle():
    return plt.Line2D([], [], color=FLOOR_FACE, lw=8,
                      label=f"harness floor, +-{FLOOR} points")


def save(fig, path, copies=()):
    """Write the figure, then copy the file to each of `copies` (dirs or files)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    for c in copies:
        c = Path(c)
        if c.is_dir() or not c.suffix:
            c.mkdir(parents=True, exist_ok=True)
            c = c / path.name
        shutil.copyfile(path, c)
    return path


# spaCy dependency labels used in 05 and 06, glossed for a reader
DEP_NAME = {
    "dobj": "object noun to verb",
    "nsubj": "subject noun to verb",
    "conj": "coordinated words",
    "appos": "noun to its apposition",
    "nmod": "noun modifier",
    "advcl": "adverbial clause to main verb",
    "ccomp": "clausal complement to main verb",
    "attr": "predicate noun to copula",
    "poss": "possessor to noun",
    "amod": "adjective to noun",
    "compound": "compound noun parts",
    "prep": "preposition to its head",
    "pobj": "object of a preposition",
    "det": "determiner to noun",
    "advmod": "adverb to its head",
    "cc": "conjunction to its head",
    "acl": "clause modifying a noun",
}


def dep_name(label):
    return DEP_NAME.get(label, label)
