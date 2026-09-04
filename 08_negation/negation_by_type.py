#!/usr/bin/env python3
"""Every condition broken out by negator type, with the words that were actually edited.

    python negation_by_type.py

Reads the negation run out of the project database and writes
results/negation_by_type.png, the mean FVE points lost per negator type and
condition, with what the flip did to that negator written into the row label.
Prints, for each negator type, the instance nearest the type's mean flip
effect: the negator, its flipped form, the governed word and the four corpus
draws that replaced it, with the points lost under each. No GPU.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import figstyle  # noqa: E402
import store  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

DB = store.DB_DEFAULT
RESULTS = HERE / "results"
RUN = 10
NTYPE = {0: "not", 1: "n't", 2: "no", 3: "without", 4: "never"}
COND = {0: "flip", 1: "del_neg", 2: "del_gov", 3: "swap_gov"}
PLAIN = {"flip": "negation flipped", "del_neg": "negator deleted",
         "del_gov": "governed word deleted", "swap_gov": "governed word swapped"}
COLOUR = {"flip": figstyle.INK, "del_neg": figstyle.GREY, "del_gov": figstyle.DELETION, "swap_gov": figstyle.SWAP}
FLIP_FORM = {"not": "not deleted", "n't": "n't deleted, the verb restored (can't to can)",
             "no": "no becomes a, or is deleted", "without": "without becomes with", "never": "never becomes always"}
rng = np.random.default_rng(0)


def boot(x, n=10000):
    x = np.asarray(x, float)
    if len(x) < 2:
        return np.nan, np.nan
    m = rng.choice(x, (n, len(x)), replace=True).mean(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def load(con):
    m = pd.read_sql("SELECT v.variant_id, v.doc_id, m.metric, m.value FROM variants v JOIN measurements m "
                    "ON m.variant_id = v.variant_id WHERE v.created_run_id = ?", con, params=(RUN,))
    w = m.pivot(index=["variant_id", "doc_id"], columns="metric", values="value").reset_index()
    base = w[w["condition"].isna()].set_index("doc_id")["fve"]
    d = w[w["condition"].notna() & (w["instance"] > 0)].copy()
    d["base"] = d["doc_id"].map(base)
    d["lost"] = -100.0 * (d["fve"] - d["base"])
    for c in ("condition", "instance", "ntype"):
        d[c] = d[c].astype(int)
    d = d[d["condition"].isin(COND)]
    d["cond"] = d["condition"].map(COND)
    d["type"] = d["ntype"].map(NTYPE)
    s = pd.read_sql("SELECT s.variant_id, s.substitute, s.source, s.draw_idx, t.text AS span, t.char_start "
                    "FROM substitutions s JOIN v_span_text t ON t.span_id = s.span_id "
                    "JOIN variants v ON v.variant_id = s.variant_id WHERE v.created_run_id = ?", con, params=(RUN,))
    return d, s


def main():
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    d, s = load(con)
    per = d.groupby(["instance", "type", "cond"])["lost"].mean().unstack("cond").reset_index()
    types = [t for t in NTYPE.values() if t in set(per["type"])]
    conds = list(COND.values())

    # figure: one row per negator type, one dot per condition
    figstyle.apply()
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    fig.subplots_adjust(left=0.40, right=0.97, top=0.86, bottom=0.14)
    figstyle.floor(ax)
    off = np.linspace(-0.3, 0.3, len(conds))
    for i, t in enumerate(types):
        g = per[per["type"] == t]
        for j, c in enumerate(conds):
            v = g[c].dropna() if c in g else pd.Series(dtype=float)
            if len(v) < 2:
                continue
            lo, hi = boot(v)
            ax.errorbar([v.mean()], [i + off[j]], xerr=[[v.mean() - lo], [hi - v.mean()]], fmt="o", ms=5,
                        color=COLOUR[c], ecolor=COLOUR[c], elinewidth=1.3, capsize=2, zorder=3)
    ax.set_yticks(range(len(types)))
    ax.set_yticklabels([f"{t}  ({(per['type'] == t).sum()})\n{FLIP_FORM[t]}" for t in types], fontsize=8.5)
    ax.set_ylim(len(types) - 0.5, -0.5)
    ax.spines["left"].set_visible(False)
    for i in range(1, len(types)):
        ax.axhline(i - 0.5, color="#dddddd", lw=0.6, zorder=1)
    ax.set_xlabel("FVE points lost, mean with 95% bootstrap interval; grey band is the harness floor")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ms=5, color=COLOUR[c], lw=1.3, label=PLAIN[c]) for c in conds]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.55, 0.9), ncol=4, frameon=False, bbox_transform=fig.transFigure,
              fontsize=8.5, handletextpad=0.4, columnspacing=1.2)
    figstyle.save(fig, RESULTS / "negation_by_type.png")

    # text: for each type, the instance nearest the type's mean flip effect
    docs = {}
    def text_of(doc_id):
        if doc_id not in docs:
            docs[doc_id] = con.execute("SELECT text FROM docs WHERE doc_id = ?", (int(doc_id),)).fetchone()[0]
        return docs[doc_id]

    L = ["# Negation by type, the words that were edited", "",
         "Run 10. Effects are FVE points lost, so positive means the reconstruction got worse. For each",
         "negator type, the instance whose flip effect is nearest the type's mean, with the exact edit",
         "each condition made and the four corpus draws that replaced the governed word.", ""]
    for t in types:
        g = per[per["type"] == t].dropna(subset=["flip"])
        inst = int(g.iloc[(g["flip"] - g["flip"].mean()).abs().argsort().iloc[0]]["instance"])
        rows = d[d["instance"] == inst]
        doc_id = int(rows["doc_id"].iloc[0])
        text = text_of(doc_id)
        fl = rows[rows["cond"] == "flip"]
        subs = s[s["variant_id"] == int(fl["variant_id"].iloc[0])].sort_values("char_start")
        neg_span = subs.iloc[-1] if t != "n't" else subs[subs["span"].str.strip().str.lower().isin(["n't", "n’t", "nt"])].iloc[0]
        lo = int(subs["char_start"].min()); hi = int(neg_span["char_start"]) + len(neg_span["span"])
        gov_rows = rows[rows["cond"] == "del_gov"]
        gov = s[s["variant_id"] == int(gov_rows["variant_id"].iloc[0])].iloc[0]
        a, b = min(lo, int(gov["char_start"])), max(hi, int(gov["char_start"]) + len(gov["span"]))
        sa, sb = store.sentence_bounds(text, a, b)
        flipped = " ".join(f"{r['span'].strip()!r} to {r['substitute'].strip()!r}" if r["substitute"].strip()
                           else f"{r['span'].strip()!r} deleted" for _, r in subs.iterrows())
        L += [f"## {t}: {FLIP_FORM[t]}  ({len(g)} instances, mean flip {g['flip'].mean():+.2f} points)", "",
              f"Document {doc_id}, instance {inst}, baseline FVE {rows['base'].iloc[0]:.3f}.", ""]
        L.append(store.wrap_text(text[sa:sb], indent="    "))
        L += ["", f"    negator {neg_span['span'].strip()!r}, governed word {gov['span'].strip()!r}", "",
              f"    flip: {flipped}    {fl['lost'].iloc[0]:+.2f} points"]
        dn = rows[rows["cond"] == "del_neg"]
        if len(dn):
            L.append(f"    negator deleted    {dn['lost'].iloc[0]:+.2f} points")
        L.append(f"    governed word deleted    {gov_rows['lost'].iloc[0]:+.2f} points")
        sw = rows[rows["cond"] == "swap_gov"].merge(s[["variant_id", "substitute"]], on="variant_id")
        for _, r in sw.sort_values("lost").iterrows():
            L.append(f"    governed word swapped for {r['substitute']!r}    {r['lost']:+.2f} points")
        if len(sw):
            L.append(f"    swap mean over {len(sw)} draws    {sw['lost'].mean():+.2f} points")
        L.append("")
    print("\n".join(L))


if __name__ == "__main__":
    main()
