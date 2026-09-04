#!/usr/bin/env python3
"""Three word pairs from one sentence of document 1664, with their interactions.

    python three_pairs_one_sentence.py

An adjacent pair, a pair on a dependency arc three or more tokens apart, and a
pair with no arc at the same token distance, each the pair of its kind nearest
that kind's mean interaction in the document. Prints the sentence with the
three pairs marked and the eight-draw interaction of each. Writes the same
text to three_pairs_one_sentence.md. No GPU.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent / "05_dependent-pair"))
sys.path.insert(0, str(HERE.parent.parent / "04_ablation-strategy" / "examples"))
import tree_vs_linear_analysis as T  # noqa: E402
import _common as K                  # noqa: E402
import figstyle                      # noqa: E402

RUN = 8
DOC = 1664


def sentences(text):
    out, start = [], 0
    while start < len(text):
        s0, s1 = K.sentence_bounds(text, start, start + 1)
        if s1 <= start:
            s1 = start + 1
        out.append((s0, s1))
        start = s1
        while start < len(text) and text[start] in " \n":
            start += 1
    return out


def main():
    conn = T.connect(str(K.DB_DEFAULT))
    base, sing, joint, _ = T.load_run(conn, RUN)
    joint = [j for j in joint if j["doc_id"] == DOC]
    ids = {x for j in joint for x in (j["lo"], j["hi"])}
    meta = T.span_meta(conn, ids)
    head = T.head_map(conn)
    cats = T.recorded_categories(conn)
    paths = T.tree_paths(conn, head, {DOC}, {(j["lo"], j["hi"]) for j in joint})
    rows, _ = T.cells(sing, joint, meta, head, cats, paths)
    text = K.doc_text(conn, DOC)
    pos = {s: (r["char_start"], r["char_end"]) for s, r in
           ((int(r["span_id"]), r) for r in conn.execute(
               "SELECT span_id, char_start, char_end FROM v_span_text WHERE doc_id = ?", (DOC,)))}
    sents = sentences(text)

    def sent_of(span):
        a = pos[span][0]
        return next(i for i, (s0, s1) in enumerate(sents) if s0 <= a < s1)

    for r in rows:
        r["sent"] = sent_of(r["lo"]) if sent_of(r["lo"]) == sent_of(r["hi"]) else None
    kinds = {
        "adjacent": [r for r in rows if r["distance"] == 1],
        "arc": [r for r in rows if r["arc"] and r["distance"] >= 3],
        "unrelated": [r for r in rows if not r["arc"] and r["distance"] >= 3],
    }
    means = {k: float(np.mean([r["inter"] for r in v])) for k, v in kinds.items()}
    # only pairs whose own draw noise is at or under the harness floor are
    # candidates, so the printed numbers mean something
    kinds = {k: [r for r in v if r["se"] <= K.FLOOR] for k, v in kinds.items()}

    # the sentence that can host all three, choosing per sentence the pair of
    # each kind nearest that kind's document mean, then the sentence whose three
    # picks are jointly nearest
    best = None
    for si in range(len(sents)):
        pick = {}
        for k in ("adjacent", "arc"):
            c = [r for r in kinds[k] if r["sent"] == si]
            if not c:
                break
            pick[k] = min(c, key=lambda r: abs(r["inter"] - means[k]))
        else:
            d = pick["arc"]["distance"]
            c = [r for r in kinds["unrelated"] if r["sent"] == si and r["distance"] == d]
            if not c:
                continue
            pick["unrelated"] = min(c, key=lambda r: abs(r["inter"] - means["unrelated"]))
            score = sum(abs(pick[k]["inter"] - means[k]) for k in pick)
            if best is None or score < best[0]:
                best = (score, si, pick)
    _, si, pick = best
    s0, s1 = sents[si]

    tags = {"adjacent": "1", "arc": "2", "unrelated": "3"}
    edits = []
    for k, r in pick.items():
        for span, side in ((r["lo"], "A"), (r["hi"], "B")):
            edits.append((pos[span][0], pos[span][1], f"{side}{tags[k]}"))
    out = text[s0:s1]
    for x0, x1, tag in sorted(edits, reverse=True):
        x0, x1 = x0 - s0, x1 - s0
        out = out[:x0] + f"[{tag}:{out[x0:x1]}]" + out[x1:]

    K.tee_to(HERE / "three_pairs_one_sentence.md")
    print("# Three pairs, one sentence")
    print()
    print(f"Document {DOC}, baseline FVE {base[DOC]:.3f}, the full pairwise run at 8 draws per pair.")
    print("Interaction = e(a) + e(b) - e(both) in FVE points lost: positive means the two words carry")
    print("overlapping information, negative means they are worth more together than apart.")
    print(f"Each pair is the one of its kind, in this sentence, nearest that kind's mean over the")
    print(f"whole document (adjacent {means['adjacent']:+.3f}, arc at distance 3 or more "
          f"{means['arc']:+.3f}, no arc at distance")
    print(f"3 or more {means['unrelated']:+.3f}), among pairs whose standard error over the draws is")
    print(f"at or under the harness floor of {K.FLOOR} points.")
    print()
    print(K.wrap_md(" ".join(out.split()), 92, indent="    "))
    print()
    for k, r in pick.items():
        se = r["se"]
        print(f"Pair {tags[k]}, {k}: \"{r['text']}\", {r['pos']}, token distance {r['distance']}, "
              f"tree path {r['path'] if r['path'] != T.DISCONNECTED else 'none'}"
              + (f", dep {r['dep']}" if r["dep"] else ""))
        print(f"    interaction {r['inter']:+.3f} +- {se:.3f} points over {r['n_draws']} draws")
        print()

    # ---------------------------------------------------------------- figure
    figstyle.apply()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Arc

    colours = {"adjacent": figstyle.DELETION, "arc": figstyle.SWAP,
               "unrelated": figstyle.GREY}
    labels = {"adjacent": "adjacent words",
              "arc": f"dependency arc, {figstyle.dep_name(pick['arc']['dep'])},\n"
                     f"{pick['arc']['distance']} tokens apart",
              "unrelated": f"no dependency, {pick['unrelated']['distance']} tokens apart"}

    # words of the sentence with their character offsets
    words = []
    i = s0
    while i < s1:
        while i < s1 and text[i].isspace():
            i += 1
        j = i
        while j < s1 and not text[j].isspace():
            j += 1
        if j > i:
            words.append((i, j, text[i:j]))
        i = j
    word_of = {}
    for k, r in pick.items():
        for span in (r["lo"], r["hi"]):
            a = pos[span][0]
            word_of[next(w for w, (x0, x1, _) in enumerate(words) if x0 <= a < x1)] = k

    # lay the words out with measured widths, wrapping at the right edge and
    # never splitting a pair across lines
    fig = plt.figure(figsize=(8.0, 4.6))
    ax = fig.add_axes([0.04, 0.60, 0.94, 0.38])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    fs = 9.5
    gap = 0.012
    widths = []
    for wi, (_, _, w) in enumerate(words):
        t = ax.text(0, 0, w, fontsize=fs, transform=ax.transAxes,
                    fontweight="bold" if wi in word_of else "normal")
        widths.append(t.get_window_extent(rend).transformed(
            ax.transAxes.inverted()).width)
        t.remove()
    partner = {}
    for k in pick:
        a_, b_ = sorted(w for w, kk in word_of.items() if kk == k)
        partner[a_] = b_
    lines, cur, x = [], [], 0.0
    w = 0
    while w < len(words):
        run = list(range(w, partner[w] + 1)) if w in partner else [w]
        need = sum(widths[r] + gap for r in run)
        if cur and x + need > 1.0:
            lines.append(cur)
            cur, x = [], 0.0
        cur += run
        x += need
        w = run[-1] + 1
    if cur:
        lines.append(cur)
    line_h = 1.0 / max(len(lines), 1)
    centres = {}
    for li, line in enumerate(lines):
        y = 1 - (li + 0.7) * line_h
        x = 0.0
        for w in line:
            k = word_of.get(w)
            ax.text(x, y, words[w][2], fontsize=fs, va="center", ha="left",
                    color=colours[k] if k else figstyle.INK,
                    fontweight="bold" if k else "normal",
                    transform=ax.transAxes)
            centres[w] = (x + widths[w] / 2, y)
            x += widths[w] + gap
    for k, r in pick.items():
        a_, b_ = sorted(w for w, kk in word_of.items() if kk == k)
        (xa, ya), (xb, _) = centres[a_], centres[b_]
        ax.add_patch(Arc(((xa + xb) / 2, ya + 0.25 * line_h), xb - xa,
                         0.5 * line_h, theta1=0, theta2=180, color=colours[k],
                         lw=1.6, transform=ax.transAxes, clip_on=False))

    ax2 = fig.add_axes([0.40, 0.16, 0.57, 0.36])
    figstyle.floor(ax2)
    order = ["adjacent", "arc", "unrelated"]
    for i, k in enumerate(order):
        r = pick[k]
        ax2.errorbar([r["inter"]], [i], xerr=[[r["se"]], [r["se"]]], fmt="o",
                     ms=6, color=colours[k], ecolor=colours[k], elinewidth=1.5,
                     capsize=3, zorder=3)
    ax2.set_yticks(range(len(order)))
    ax2.set_yticklabels(['"' + pick[k]["text"].replace(" .. ", '" and "') + '"\n'
                         + labels[k] for k in order])
    ax2.set_ylim(len(order) - 0.5, -0.5)
    ax2.spines["left"].set_visible(False)
    ax2.set_xlabel("interaction, e(a) + e(b) - e(both), FVE points,\n"
                   "mean and standard error over 8 draws; grey band is the harness floor")
    figstyle.save(fig, HERE.parent / "results" / "three_pairs_one_sentence.png",
                  copies=[HERE])
    print("Figure: ../results/three_pairs_one_sentence.png, copied here.")
    K.untee()


if __name__ == "__main__":
    main()
