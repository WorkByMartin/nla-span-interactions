#!/usr/bin/env python3
"""How does reconstruction quality fall as a document's eligible words are removed one at a time?

    python removal_analysis.py --run 9

Reads the removal-curve run out of the project database, along with the
single-span run whose step-one variants it repeats, prints the endpoint,
mean-curve, concavity, truncation and whole-against-parts tables, writes the
same report to results/statistics.md from the same rendering, and writes one
figure beside it. Nothing but the database is read. No GPU.
"""
import argparse, sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import figstyle
import matplotlib.pyplot as plt

CURVES = {0: "random deletion", 1: "random swap", 4: "random filler", 2: "front deletion", 3: "back deletion",
          5: "front filler", 6: "back filler"}


def load(db, run):
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    m = pd.read_sql("SELECT v.variant_id, v.doc_id, m.metric, m.value FROM variants v JOIN measurements m "
                    "ON m.variant_id = v.variant_id WHERE v.created_run_id = ?", con, params=(run,))
    w = m.pivot(index=["variant_id", "doc_id"], columns="metric", values="value").reset_index()
    base = w[w["curve"].isna()][["doc_id", "fve"]].rename(columns={"fve": "base_fve"})
    d = w[w["curve"].notna()].merge(base, on="doc_id")
    d["curve"] = d["curve"].astype(int); d["step"] = d["step"].astype(int); d["perm"] = d["perm"].astype(int)
    d["frac"] = d["step"] / d["n_words"]
    d["dfve"] = d["fve"] - d["base_fve"]
    return d, base, con


def chord_area(g):
    """Signed area between the curve and the straight line from (0, base) to (1, end), in FVE x fraction.
    Positive: the curve sits above the chord, so early removals cost less than proportional."""
    g = g.sort_values("frac")
    x = np.concatenate([[0.0], g["frac"].values]); y = np.concatenate([[0.0], g["dfve"].values])
    end = y[-1]
    chord = end * x
    return float(np.trapz(y - chord, x)), float(end)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="../db/ffw_span-ablation_database.sqlite")
    ap.add_argument("--run", type=int, default=9)
    ap.add_argument("--singles-run", type=int, default=8)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "results"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(exist_ok=True)
    d, base, con = load(a.db, a.run)
    docs = sorted(d["doc_id"].unique())
    L = [f"# Removal curves, run {a.run}", "", f"documents {len(docs)}: {docs}", "",
         f"baseline FVE mean {base['base_fve'].mean():.3f} (min {base['base_fve'].min():.3f}, max {base['base_fve'].max():.3f})", ""]

    # endpoints
    ends = d[(d["step"] == d["n_words"]) & (d["curve"] == 0)].groupby("doc_id")["fve"].first()
    ends_swap = d[(d["step"] == d["n_words"]) & (d["curve"] == 1)].groupby("doc_id")["fve"].mean()
    ends_fill = d[(d["step"] == d["n_words"]) & (d["curve"] == 4)].groupby("doc_id")["fve"].first()
    L += ["## Endpoints", "", "| | mean FVE | sd |", "|---|---|---|",
          f"| intact | {base['base_fve'].mean():.3f} | {base['base_fve'].std():.3f} |",
          f"| every eligible word deleted | {ends.mean():.3f} | {ends.std():.3f} |",
          f"| every eligible word swapped (mean over perms) | {ends_swap.mean():.3f} | {ends_swap.std():.3f} |",
          f"| every eligible word replaced by filler | {ends_fill.mean():.3f} | {ends_fill.std():.3f} |", ""]

    # mean curves on a common grid
    grid = np.linspace(0, 1, 21)
    def on_grid(g):
        g = g.sort_values("frac")
        return np.interp(grid, np.concatenate([[0], g["frac"].values]), np.concatenate([[0], g["dfve"].values]))
    mean_curves = {}
    L += ["## Mean dFVE by fraction of eligible words removed", "",
          "| fraction | " + " | ".join(CURVES.values()) + " |", "|---|" + "---|" * len(CURVES)]
    for c in CURVES:
        per = [on_grid(g) for _, g in d[d["curve"] == c].groupby(["doc_id", "perm"])]
        mean_curves[c] = (np.mean(per, axis=0), np.std(per, axis=0) / np.sqrt(len(per)), len(per))
    for i, f in enumerate(grid):
        if i % 2: continue
        L.append(f"| {f:.2f} | " + " | ".join(f"{mean_curves[c][0][i]:+.3f} ± {mean_curves[c][1][i]:.3f}" for c in CURVES) + " |")
    L.append("")

    # concavity
    L += ["## Concavity: signed area between curve and chord (positive = early removals cheaper than proportional)", "",
          "| curve | mean area | se | n curves | docs with area > 0 | mean endpoint dFVE |", "|---|---|---|---|---|---|"]
    areas = {}
    for c in CURVES:
        rows = [chord_area(g) for _, g in d[d["curve"] == c].groupby(["doc_id", "perm"])]
        ar = np.array([r[0] for r in rows]); en = np.array([r[1] for r in rows])
        areas[c] = ar
        L.append(f"| {CURVES[c]} | {ar.mean():+.4f} | {ar.std(ddof=1) / np.sqrt(len(ar)):.4f} | {len(ar)} | "
                 f"{(ar > 0).mean():.2f} | {en.mean():+.3f} |")
    L.append("")

    # front vs back, deletion and filler
    L += ["## Front vs back truncation, dFVE at fixed fractions removed", "",
          "| primitive | fraction | front | back | front − back | se | docs with front < back |", "|---|---|---|---|---|---|---|"]
    for name, (cf, cb) in (("deletion", (2, 3)), ("filler", (5, 6))):
        fb = {c: np.array([on_grid(g) for _, g in d[d["curve"] == c].groupby("doc_id")]) for c in (cf, cb)}
        for f in (0.25, 0.5, 0.75):
            i = int(round(f * 20)); diff = fb[cf][:, i] - fb[cb][:, i]
            L.append(f"| {name} | {f:.2f} | {fb[cf][:, i].mean():+.3f} | {fb[cb][:, i].mean():+.3f} | {diff.mean():+.3f} | "
                     f"{diff.std(ddof=1) / np.sqrt(len(diff)):.3f} | {(diff < 0).mean():.2f} |")
    L.append("")

    # primitives against each other, random order
    L += ["## Random order, primitive differences in mean dFVE", "",
          "| fraction | deletion − swap | deletion − filler | swap − filler |", "|---|---|---|---|"]
    for i in range(0, 21, 4):
        a0, a1, a4 = (mean_curves[c][0][i] for c in (0, 1, 4))
        L.append(f"| {grid[i]:.2f} | {a0 - a1:+.3f} | {a0 - a4:+.3f} | {a1 - a4:+.3f} |")
    L.append("")
    # early rise under filler: is dFVE positive over the first steps?
    e = d[(d["curve"] == 4) & (d["step"] <= 5)].groupby("step")["dfve"].agg(["mean", "sem", "count"])
    L += ["## Random filler, first five steps (does FVE rise at first?)", "", "| step | mean dFVE | se | n |", "|---|---|---|---|"]
    L += [f"| {i} | {r['mean']:+.4f} | {r['sem']:.4f} | {int(r['count'])} |" for i, r in e.iterrows()] + [""]

    # step-1 consistency against the singles run
    s1 = d[(d["step"] == 1)]
    if a.singles_run:
        sub = pd.read_sql("SELECT s.variant_id, s.span_id, s.substitute, s.source FROM substitutions s JOIN variants v "
                          "ON v.variant_id = s.variant_id WHERE v.created_run_id = ?", con, params=(a.run,))
        s1 = s1.merge(sub, on="variant_id")
        prev = pd.read_sql("SELECT span_id, substitute, dfve FROM v_single WHERE run_id = ?", con, params=(a.singles_run,))
        j = s1[s1["curve"] == 1].merge(prev, on=["span_id", "substitute"], suffixes=("", "_prev"))
        if len(j):
            r = np.corrcoef(j["dfve"], j["dfve_prev"])[0, 1]
            L += [f"## Step-1 swap variants that repeat a run {a.singles_run} single (same span, same substitute)", "",
                  f"n {len(j)}, correlation {r:.3f}, mean |difference| {np.abs(j['dfve'] - j['dfve_prev']).mean():.5f}, "
                  f"max |difference| {np.abs(j['dfve'] - j['dfve_prev']).max():.5f}", ""]
        L += [f"## First removal (one word), by primitive", ""]
        for c in (0, 1, 4):
            g = s1[s1["curve"] == c]
            L.append(f"{CURVES[c]}: mean dFVE {g['dfve'].mean():+.4f}, mean |dFVE| {g['dfve'].abs().mean():.4f}, n {len(g)}")
        L.append("")

    # sum of singles vs whole
    L += ["## Whole vs sum of parts", "",
          "| doc | base FVE | end FVE (deleted) | drop | sum of step-1 deletion dFVE over the doc's random curves ÷ perms x n |", "|---|---|---|---|---|"]
    for doc in docs:
        g = d[(d["doc_id"] == doc)]
        n = int(g["n_words"].iloc[0]); nperm = g[g["curve"] == 0]["perm"].nunique()
        first = g[(g["curve"] == 0) & (g["step"] == 1)]["dfve"]
        L.append(f"| {doc} | {base.set_index('doc_id').loc[doc, 'base_fve']:.3f} | {ends[doc]:.3f} | {ends[doc] - base.set_index('doc_id').loc[doc, 'base_fve']:+.3f} | {first.mean() * n:+.3f} |")
    L.append("")

    # figure: the three random-order curves on one axis, with a 95% confidence band
    # for the mean over documents (each document's permutations averaged first)
    figstyle.apply()
    fine = np.linspace(0, 1, 101)
    def on_fine(g):
        g = g.sort_values("frac")
        return np.interp(fine, np.concatenate([[0], g["frac"].values]), np.concatenate([[0], g["dfve"].values]))
    b0 = base["base_fve"].mean()
    colour = {0: figstyle.SWAP, 2: figstyle.SWAP, 3: figstyle.SWAP,
              4: figstyle.MLM, 5: figstyle.MLM, 6: figstyle.MLM, 1: figstyle.DELETION}
    style = {0: "-", 1: "-", 4: "-", 2: "--", 5: "--", 3: ":", 6: ":"}
    label = {0: "random order, deletion", 1: "random order, swap", 4: "random order, filler",
             2: "front truncation, deletion", 3: "back truncation, deletion",
             5: "front truncation, filler", 6: "back truncation, filler"}
    floor = float(ends.mean())
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    fig.subplots_adjust(left=0.10, right=0.97, top=0.95, bottom=0.12)
    for c in (0, 4, 1):                      # legend order follows the curves at 60% removed
        per_doc = np.array([np.mean([on_fine(g) for _, g in gd.groupby("perm")], axis=0)
                            for _, gd in d[d["curve"] == c].groupby("doc_id")])
        m = per_doc.mean(axis=0) + b0
        ci = 1.96 * per_doc.std(axis=0, ddof=1) / np.sqrt(len(per_doc))
        ax.fill_between(fine, m - ci, m + ci, color=colour[c], alpha=0.12, lw=0)
        ax.plot(fine, m, color=colour[c], ls=style[c], lw=2.0, label=label[c])
    ax.axhline(0, color="#999999", lw=0.8)
    ax.axhline(floor, color="#555555", lw=0.8, ls=(0, (1, 2)))
    ax.text(0.01, floor, f"every word removed, FVE {floor:.2f}", va="bottom", ha="left", fontsize=8, color="#555555")
    ax.set_xlim(0, 1); ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.set_xlabel("fraction of eligible words removed")
    ax.set_ylabel("FVE, mean over documents, 95% confidence band")
    ax.legend(loc="lower left")
    figstyle.save(fig, out / "removal_curves.png")
    # third figure: front against back truncation under deletion, one faint line per
    # document and the mean over documents in bold
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    fig.subplots_adjust(left=0.10, right=0.98, top=0.95, bottom=0.13)
    tcol = {2: figstyle.SWAP, 3: "#D55E00"}
    tname = {2: "deleting from the front", 3: "deleting from the back"}
    for c in (2, 3):
        ys = []
        for _, g in d[d["curve"] == c].groupby("doc_id"):
            g = g.sort_values("frac")
            x = np.concatenate([[0], g["frac"].values]); y = np.concatenate([[g["base_fve"].iloc[0]], g["fve"].values])
            ax.plot(x, y, color=tcol[c], lw=0.7, alpha=0.28)
            ys.append(np.interp(fine, x, y))
        ax.plot(fine, np.mean(ys, axis=0), color=tcol[c], lw=2.4, label=tname[c])
    ax.axhline(0, color="#999999", lw=0.8)
    ax.set_xlim(0, 1); ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.set_xlabel("fraction of eligible words removed"); ax.set_ylabel("FVE")
    ax.legend(loc="lower left", title=f"mean over {len(docs)} documents; faint lines are the documents")
    figstyle.save(fig, out / "truncation_front_vs_back.png")
    # second figure: the first ten removals step by step, in FVE points lost
    steps = np.arange(0, 11)
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    fig.subplots_adjust(left=0.11, right=0.97, top=0.95, bottom=0.13)
    for c in (2, 5, 0, 4, 3, 6, 1):
        per_doc = []
        for _, gd in d[(d["curve"] == c) & (d["step"] <= 10)].groupby("doc_id"):
            rows = [np.r_[0.0, gp.sort_values("step")["dfve"].values[:10]] for _, gp in gd.groupby("perm")]
            per_doc.append(np.mean(rows, axis=0))
        per_doc = -100.0 * np.array(per_doc)
        m = per_doc.mean(axis=0)
        ax.plot(steps, m, color=colour[c], ls=style[c], lw=2.0, label=label[c])
    ax.axhline(0, color="#999999", lw=0.8)
    ax.set_xticks(steps); ax.set_xlim(0, 10); ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.set_xlabel("words removed")
    ax.set_ylabel("FVE points lost, mean over documents")
    ax.legend(loc="upper left", handlelength=4.0)
    figstyle.save(fig, out / "removal_first_steps.png")
    L += ["## Figures", "", f"removal_curves.png: FVE against fraction removed under random order, one mean curve per "
          f"primitive with a 95% confidence band for the mean over the {len(docs)} documents (each document's "
          f"permutations averaged first); the dotted horizontal line is the mean FVE with every eligible word deleted",
          f"truncation_front_vs_back.png: FVE against fraction removed under deletion, truncating from the front and "
          f"from the back, one faint line per document and the mean over the {len(docs)} documents in bold",
          f"removal_first_steps.png: the same seven curve types over the first ten words removed, in FVE points "
          f"lost (100 x the drop in FVE), mean over the {len(docs)} documents", ""]
    (out / "statistics.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
