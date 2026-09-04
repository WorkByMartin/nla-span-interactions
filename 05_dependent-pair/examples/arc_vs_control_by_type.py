#!/usr/bin/env python3
"""Interaction on a dependency arc against its matched control, per dependency type.

    python arc_vs_control_by_type.py

Reads the PER DEP TYPE table from ../results/statistics.md, which pair_analysis.py
wrote from the store, and draws it with each dependency label glossed. Writes
../results/arc_vs_control_by_type.png and a copy here. No GPU.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
import figstyle  # noqa: E402

STATS = HERE.parent / "results" / "statistics.md"
ROW = re.compile(r"^\s*(\w+)\s+(\d+)\s+([+-]\d+\.\d+)\s+(\d+\.\d+)\s+(\d+)\s+"
                 r"([+-]\d+\.\d+)\s+(\d+\.\d+)\s+([+-]\d+\.\d+)\s+(\d+\.\d+)\s*$")


def read_table():
    lines = STATS.read_text().splitlines()
    i = next(k for k, ln in enumerate(lines) if ln.startswith("PER DEP TYPE"))
    rows = []
    for ln in lines[i + 2:]:
        m = ROW.match(ln)
        if not m:
            if rows:
                break
            continue
        dep, n_arc, am, ase, n_ctrl, cm, cse, dm, dse = m.groups()
        rows.append(dict(dep=dep, n_arc=int(n_arc), arc=(float(am), float(ase)),
                         n_ctrl=int(n_ctrl), ctrl=(float(cm), float(cse)),
                         diff=(float(dm), float(dse))))
    return rows


def main():
    figstyle.apply()
    import matplotlib.pyplot as plt

    rows = read_table()
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    fig.subplots_adjust(left=0.36, right=0.97, top=0.88, bottom=0.12)
    figstyle.floor(ax)
    for i, r in enumerate(rows):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color="#fafafa", zorder=0)
        for key, off, col, mk in (("arc", -0.18, figstyle.SWAP, "o"),
                                  ("ctrl", 0.18, figstyle.GREY, "s")):
            m, se = r[key]
            ax.errorbar([m], [i + off], xerr=[[se], [se]], fmt=mk, ms=5.5,
                        color=col, ecolor=col, elinewidth=1.4, capsize=2.5,
                        zorder=3)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{figstyle.dep_name(r['dep'])}  ({r['n_arc']})"
                        for r in rows])
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("interaction, e(a) + e(b) - e(both), FVE points. "
                  "Right of zero the two words overlap")
    handles = [plt.Line2D([], [], color=figstyle.SWAP, marker="o", ls="none",
                          ms=5.5, label="pair on a dependency arc"),
               plt.Line2D([], [], color=figstyle.GREY, marker="s", ls="none",
                          ms=5.5, label="matched control pair, same distance, no arc"),
               figstyle.floor_handle()]
    ax.legend(handles=handles, ncol=2, loc="lower center",
              bbox_to_anchor=(0.5, 1.0))
    ax.set_ylabel("dependency type, and the number of arc pairs")
    figstyle.save(fig, HERE.parent / "results" / "arc_vs_control_by_type.png",
                  copies=[HERE])
    print(f"{len(rows)} dependency types read from {STATS.name}; standard errors "
          f"are clustered on document over 100 documents")


if __name__ == "__main__":
    main()
