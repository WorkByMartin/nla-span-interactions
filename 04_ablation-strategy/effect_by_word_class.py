#!/usr/bin/env python3
"""Effect by word class: the signed mean effect of each word class.

    python effect_by_word_class.py

Recomputes the per-class signed mean effect table from the store, checks it
against the same table in results/statistics.md digit for digit, and draws it as
a dot and interval chart. Prints the report and writes the figure to
results/effect_by_word_class.png. No GPU.
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import swap_analysis as SA     # noqa: E402
import figstyle                # noqa: E402
import store                   # noqa: E402

SWAP_RUN = 5
MLM_RUN = 3
RESULTS = HERE / "results"
STATS = RESULTS / "statistics.md"
SECTION = "PER-CLASS SIGNED MEAN EFFECT"
ARMS = ["swap", "MLM all", "MLM non-id", "deletion"]
CHARTED = ["swap", "deletion"]
COLOUR = {"swap": figstyle.SWAP, "MLM non-id": figstyle.MLM,
          "deletion": figstyle.DELETION}
NICE = {"swap": "corpus swap", "MLM non-id": "masked-LM, non-identical draws",
        "deletion": "deletion"}
CELL = re.compile(r"([+-]\d+\.\d{3}) \+-\s+(\d+\.\d{3})")


def recompute(db):
    """The table as swap_analysis computes it: {class: {arm: (mean, se)}}."""
    conn = SA.connect(db)
    swap_rows = SA.singles(conn, SWAP_RUN, SA.SWAP)
    docs = sorted({int(r["doc_id"]) for r in swap_rows})
    spans = sorted({int(r["span_id"]) for r in swap_rows})
    span_set = set(spans)
    del_rows = [r for r in SA.singles(conn, SWAP_RUN, SA.DELETION, docs)
                if int(r["span_id"]) in span_set]
    mlm_rows = [r for r in SA.singles(conn, MLM_RUN, SA.MLM, docs)
                if int(r["span_id"]) in span_set]
    mlm_clean = [r for r in mlm_rows
                 if r["substitute"].lower()
                 != (r["span_text"] or "").strip().lower()]
    pos = SA.pos_of(conn, spans)
    conn.close()

    cls = SA.class_groups(spans, pos)
    cls_of = {sp: lab for lab, v in cls.items() for sp in v}

    def by_class(rows):
        out = defaultdict(list)
        for r in rows:
            lab = cls_of.get(int(r["span_id"]))
            if lab is not None:
                out[lab].append(r)
        return out

    grouped = dict(zip(ARMS, [by_class(swap_rows), by_class(mlm_rows),
                              by_class(mlm_clean), by_class(del_rows)]))
    table = {}
    for label, ss in cls.items():
        table[label] = dict(n_spans=len(ss))
        for arm in ARMS:
            r = grouped[arm].get(label, [])
            v = np.array([x["effect"] for x in r])
            table[label][arm] = (float(v.mean()),
                                 float(SA.cluster_se(v, [x["doc_id"]
                                                         for x in r])))
    return table, cls, len(docs), len(spans)


def parse_statistics(path):
    """The same table as printed in statistics.md: {class: {arm: (str, str)}}."""
    lines = path.read_text().splitlines()
    try:
        i = next(k for k, ln in enumerate(lines) if SECTION in ln)
    except StopIteration:
        raise SystemExit(f"{path} has no {SECTION!r} section")
    j = next(k for k in range(i, len(lines))
             if lines[k].lstrip().startswith("class "))
    out = {}
    for ln in lines[j + 1:]:
        cells = CELL.findall(ln)
        if len(cells) != len(ARMS):
            if out:
                break
            continue
        prefix = ln[:CELL.search(ln).start()]
        head = re.match(r"\s*(.+?)\s+(\d+)\s*$", prefix)
        if not head:
            raise SystemExit(f"cannot read the class and span count from "
                             f"{ln!r}")
        out[head.group(1)] = dict(
            n_spans=int(head.group(2)),
            **{arm: cells[k] for k, arm in enumerate(ARMS)})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=str(store.DB_DEFAULT))
    args = ap.parse_args()
    figstyle.apply()

    table, cls, n_docs, n_spans = recompute(args.db)
    printed = parse_statistics(STATS)

    # every cell compared as the string statistics.md printed it
    diffs = []
    for label in table:
        if label not in printed:
            diffs.append(f"{label}: absent from {STATS.name}")
            continue
        if printed[label]["n_spans"] != table[label]["n_spans"]:
            diffs.append(f"{label} spans: recomputed "
                         f"{table[label]['n_spans']}, printed "
                         f"{printed[label]['n_spans']}")
        for arm in ARMS:
            m, se = table[label][arm]
            mine = (f"{m:+.3f}", f"{se:.3f}")
            theirs = printed[label][arm]
            if mine != theirs:
                diffs.append(f"{label} / {arm}: recomputed "
                             f"{mine[0]} +- {mine[1]}, printed "
                             f"{theirs[0]} +- {theirs[1]}")
    for label in printed:
        if label not in table:
            diffs.append(f"{label}: in {STATS.name} but not recomputed")

    order = sorted(table, key=lambda l: -table[l]["swap"][0])

    # ------------------------------------------------------------------ text
    print("# Effect by word class")
    print()
    print(f"The signed mean effect of an ablation, by the coarse part of "
          f"speech of the word ablated,")
    print(f"recomputed from the store over {n_spans} spans in {n_docs} "
          f"documents. Effect is FVE points lost,")
    print(f"-100 x (fve - base_fve), so a positive number means the edit made "
          f"the reconstruction worse and")
    print(f"a negative number means it improved it. Classes holding fewer than "
          f"{SA.MIN_CLASS} spans are pooled")
    print(f"into one row. Every interval is a standard error clustered on the "
          f"document, over {n_docs} clusters,")
    print(f"which is enough for the asymptotics to be worth something and not "
          f"enough to lean on hard.")
    print()
    print(f"`MLM all` is every masked-LM draw. `MLM non-id` drops the draws "
          f"that put the original word")
    print(f"back, which is 45 per cent of them, and is the column the chart "
          f"uses.")
    print()
    print(store.md_table(
        ["class", "spans"] + [f"{a} (FVE points)" for a in ARMS],
        [[label, table[label]["n_spans"]]
         + [f"{table[label][a][0]:+.3f} +- {table[label][a][1]:.3f}"
            for a in ARMS]
         for label in order],
        ["l", "r", "r", "r", "r", "r"]))
    print()
    print(f"Rows are sorted by the swap column, largest first.")
    print()
    if diffs:
        print(f"## Check against {STATS.name}: FAILED")
        print()
        print(f"{len(diffs)} cell(s) disagree with the table printed in "
              f"results/{STATS.name}:")
        print()
        for d in diffs:
            print(f"  - {d}")
        print()
        print("No figure was written. The disagreement is the result.")
        raise SystemExit(1)
    print(f"## Check against {STATS.name}: matched")
    print()
    print(f"Every one of the {len(table) * len(ARMS)} cells recomputed here "
          f"({len(table)} classes x {len(ARMS)} arms), and every")
    print(f"span count, is identical to the table printed in "
          f"results/{STATS.name} at the three decimal")
    print(f"places that file prints. The chart is drawn from the recomputed "
          f"values.")
    print()

    # ---------------------------------------------------------------- figure
    import matplotlib.pyplot as plt

    vals = [(label, arm) + table[label][arm]
            for label in order for arm in CHARTED]
    inside = [(m, se) for _, _, m, se in vals if abs(m) <= 1.0]
    lo = min(m - se for m, se in inside)
    hi = max(m + se for m, se in inside)
    pad = 0.08 * (hi - lo)
    xlim = (lo - pad, hi + pad)
    offs = {"swap": -0.16, "deletion": 0.16}
    marks = {"swap": "o", "deletion": "D"}

    fig, ax = plt.subplots(figsize=(8.0, 6.6))
    fig.subplots_adjust(left=0.28, right=0.965, top=0.90, bottom=0.10)
    ax.axvline(0, color="#999999", lw=0.8, zorder=1.5)
    clipped = []
    for i, label in enumerate(order):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color="#fafafa", zorder=0)
        for arm in CHARTED:
            m, se = table[label][arm]
            y = i + offs[arm]
            if m > xlim[1]:
                ax.plot([xlim[1]], [y], marker=">", color=COLOUR[arm], ms=7,
                        clip_on=False, zorder=4)
                clipped.append((label, arm, m, se, y))
                continue
            ax.errorbar([m], [y], xerr=[[se], [se]], fmt=marks[arm], ms=5.5,
                        color=COLOUR[arm], ecolor=COLOUR[arm], elinewidth=1.4,
                        capsize=2.5, zorder=3)
    for label, arm, m, se, y in clipped:
        ax.annotate(f"{m:+.3f} +- {se:.3f}", (xlim[1], y),
                    textcoords="offset points", xytext=(-6, 8), ha="right",
                    fontsize=7.5, color=COLOUR[arm])
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{figstyle.pos_name(l)}  ({table[l]['n_spans']})"
                        for l in order], fontsize=9)
    ax.invert_yaxis()
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.set_xlim(*xlim)
    ax.set_xlabel("FVE points lost, signed. Left of zero the edit improved the "
                  "reconstruction", fontsize=9.5)
    ax.tick_params(axis="x", labelsize=8.5)
    ax.spines["left"].set_visible(False)
    handles = [plt.Line2D([], [], color=COLOUR[a], marker=marks[a], ls="none",
                          ms=5.5, label=NICE[a]) for a in CHARTED]
    ax.legend(handles=handles, fontsize=8.5, frameon=False, ncol=2,
              loc="lower center", bbox_to_anchor=(0.5, 1.005))
    ax.set_ylabel("word class, and the number of spans in it", fontsize=9.5)
    figstyle.save(fig, RESULTS / "effect_by_word_class.png")

    print("Figure: results/effect_by_word_class.png. One row per class, "
          "sorted by the swap effect, a swap and a deletion marker per row")
    print("with a document-clustered standard error either side and a line at "
          "zero.")
    if clipped:
        print()
        for label, arm, m, se, _ in clipped:
            print(f"  {label} / {arm} is off the right of the chart at "
                  f"{m:+.3f} +- {se:.3f} FVE points and is drawn as an arrow "
                  f"at the edge.")
        print("  The axis is set from every other point so the rest of the "
              "classes stay readable.")


if __name__ == "__main__":
    main()
