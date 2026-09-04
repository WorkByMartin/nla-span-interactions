#!/usr/bin/env python3
"""One word, all draws: what one word's ablation looks like, draw by draw.

    python one_word_all_draws.py

One word of document 1934 with its sixteen corpus-swap draws, its masked-LM
draws and its single deletion. Prints the report, writes the same text to
one_word_all_draws.md, the figure to ../results/one_word_all_draws.png and a
copy beside the text. No GPU.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as K  # noqa: E402
import figstyle  # noqa: E402

DOC = 1934
WANT = ("NOUN", "PROPN")
SWAP_RUN = 5
MLM_RUN = 3


def pick(conn, want):
    """The span of one of `want` classes with the largest mean swap effect."""
    pos = K.span_pos(conn, DOC)
    rows = K.singles(conn, SWAP_RUN, K.SWAP, doc_id=DOC)
    per = defaultdict(list)
    for r in rows:
        per[int(r["span_id"])].append(float(r["effect"]))
    cand = [(s, float(np.mean(v))) for s, v in per.items()
            if s in pos and pos[s][0] in want]
    span_id, mean = max(cand, key=lambda t: t[1])
    return span_id, mean, pos[span_id], cand


def mlm_unique(rows, original):
    """Deduplicated masked-LM draws: (substitute, n, effect, identical)."""
    by = defaultdict(list)
    for r in rows:
        by[r["substitute"]].append(r)
    out = []
    for w, rs in by.items():
        e = float(np.mean([r["effect"] for r in rs]))
        out.append((w, len(rs), e, w.strip().lower() == original.strip().lower()))
    return sorted(out, key=lambda t: -t[2])


def bars(ax, labels, values, colours, deletion, xlim):
    y = np.arange(len(values))
    ax.barh(y, values, color=colours, height=0.72, edgecolor="none")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.axvline(0.0, color="#444444", lw=0.8)
    ax.axvline(deletion, color=K.C_DEL, lw=1.6, ls="--")
    ax.set_xlim(*xlim)
    ax.set_xlabel("FVE points lost")
    ax.spines["left"].set_visible(False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=str(K.DB_DEFAULT))
    args = ap.parse_args()
    figstyle.apply()

    conn = K.connect(args.db)
    text = K.doc_text(conn, DOC)
    base = K.baseline(conn, SWAP_RUN, DOC)
    span_id, mean, (p, word, a, b), cand = pick(conn, WANT)
    s0, s1 = K.sentence_bounds(text, a, b)
    sent, hl = text[s0:s1], (a - s0, b - s0)
    swap = sorted(K.singles(conn, SWAP_RUN, K.SWAP, span_id=span_id),
                  key=lambda r: -r["effect"])
    mlm = K.singles(conn, MLM_RUN, K.MLM, span_id=span_id)
    mlmu = mlm_unique(mlm, word)
    deletion = float(K.singles(conn, SWAP_RUN, K.DELETION, span_id=span_id)[0]["effect"])
    ident = sum(n for _, n, _, i in mlmu if i)

    # ------------------------------------------------------------------ text
    K.tee_to(K.HERE / "one_word_all_draws.md")
    print("# One word, all draws")
    print()
    print(f"One ablated word, every draw of it. Document {DOC}, baseline FVE "
          f"{base:.4f}. Effect is FVE points")
    print("lost, -100 x (fve - base_fve), so a positive number means the "
          "reconstruction got worse.")
    print(f"The harness floor is {K.FLOOR} points.")
    print()
    print(f"The word is {word!r} ({p}), the {' or '.join(WANT)} span of the "
          f"document with the largest mean")
    print(f"swap effect out of {len(cand)}. Its sentence, the word in capitals "
          f"between double angle brackets:")
    print()
    i, j = hl
    print(K.wrap_md(sent[:i] + "<<" + sent[i:j].upper() + ">>" + sent[j:], 92,
                    indent="    "))
    print()
    print(f"Mean swap effect {mean:+.3f}, deletion {deletion:+.3f}, masked-LM "
          f"mean over all {len(mlm)} draws")
    print(f"{np.mean([r['effect'] for r in mlm]):+.3f} FVE points.")
    print()
    print(f"Corpus swap, {len(swap)} draws from other documents matched on "
          f"part of speech, token count")
    print("and space parity:")
    print()
    print(K.md_table(["substitute", "effect"],
                     [[r["substitute"], f"{r['effect']:+.3f}"] for r in swap],
                     ["l", "r"]))
    print()
    print(f"Masked-LM marginalisation, {len(mlm)} draws over six candidate "
          f"depths, deduplicated by")
    print(f"substitute. {ident} of the {len(mlm)} draws put the original word "
          f"back.")
    print()
    print(K.md_table(["substitute", "draws", "effect", "note"],
                     [[w, n, f"{e:+.3f}", "original word" if idn else ""]
                      for w, n, e, idn in mlmu], ["l", "r", "r", "l"]))
    print()
    print(f"Deletion, one variant: {deletion:+.3f} FVE points.")
    print()

    # ---------------------------------------------------------------- figure
    import matplotlib.pyplot as plt

    vals = [0.0, deletion] + [r["effect"] for r in swap] + [e for _, _, e, _ in mlmu]
    lo, hi = min(vals), max(vals)
    pad = 0.10 * (hi - lo)
    xlim = (lo - pad, hi + pad)
    n_bar = max(len(swap), len(mlmu))

    fig = plt.figure(figsize=(8.0, 5.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.7, 3.2], hspace=0.30,
                          wspace=0.55, left=0.13, right=0.97, top=0.96,
                          bottom=0.09)
    axt = fig.add_subplot(gs[0, :])
    axt.axis("off")
    axt.text(0.0, 1.02, f"the word \"{word}\" in document {DOC}, baseline "
                        f"FVE {base:.4f}",
             transform=axt.transAxes, fontsize=10.5, fontweight="bold",
             va="baseline")
    K.draw_mono_block(axt, sent, 104, 7.6, highlight=hl, top=0.72)

    ax1 = fig.add_subplot(gs[1, 0])
    bars(ax1, [r["substitute"] for r in swap], [r["effect"] for r in swap],
         [K.C_SWAP] * len(swap), deletion, xlim)
    ax1.set_title(f"corpus swap, {len(swap)} draws", color=K.C_SWAP, pad=6)
    ax1.set_ylim(n_bar - 0.4, -0.6)

    ax2 = fig.add_subplot(gs[1, 1])
    bars(ax2, [f"{w} ({n})" + (", original" if idn else "")
               for w, n, _, idn in mlmu],
         [e for _, _, e, _ in mlmu],
         [K.C_GREY if idn else K.C_MLM for _, _, _, idn in mlmu],
         deletion, xlim)
    ax2.set_title(f"masked-LM, {len(mlm)} draws", color=K.C_MLM, pad=6)
    ax2.set_ylim(n_bar - 0.4, -0.6)
    ax2.text(0.99, 0.02, f"dashed line: deletion {deletion:+.3f}",
             transform=ax2.transAxes, ha="right", va="bottom", fontsize=7.5,
             color=K.C_DEL)

    figstyle.save(fig, K.RESULTS / "one_word_all_draws.png", copies=[K.HERE])
    print("Figure: ../results/one_word_all_draws.png, copied here. Left the corpus-swap "
          "draws, right the masked-LM")
    print("draws deduplicated by substitute with the draw count in the label, "
          "grey where the draw put the")
    print("original word back. The dashed line on both panels is the word's "
          "single deletion.")
    K.untee()
    conn.close()


if __name__ == "__main__":
    main()
