#!/usr/bin/env python3
"""What does flipping a negation cost the reconstruction, against controls that touch the same words?

    python negation_analysis.py --db ../db/ffw_span-ablation_database.sqlite --run 10

Reads the negation run out of the project database, prints the condition means,
the paired contrasts, the per-negator-type and in-quote breakdowns, the
insertion pair and the largest flip minus swap_gov instances, writes the same
report to results/statistics.md from the same rendering, and writes one figure
beside it, two figures. Nothing but the database is read. No GPU.
"""
import argparse, sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import figstyle
import matplotlib.pyplot as plt

COND = {0: "flip", 1: "del_neg", 2: "del_gov", 3: "swap_gov", 4: "ins_not", 5: "ins_ctrl"}
PLAIN = {"flip": "negation flipped", "del_neg": "negator deleted", "del_gov": "governed word deleted",
         "swap_gov": "governed word corpus-swapped", "ins_not": "\"not\" inserted after an auxiliary",
         "ins_ctrl": "\"just\" inserted, the control"}
NTYPE = {0: "not", 1: "n't", 2: "no", 3: "without", 4: "never", 5: "insertion"}
# the surface forms negation.py treats as each negator type, used to pick the
# negator's own span out of the flip variant's substitution rows
SURFACE = {0: ("not",), 1: ("n't", "n\u2019t", "nt"), 2: ("no",), 3: ("without",), 4: ("never",)}
CONTEXT = 40                            # characters either side of the negator
rng = np.random.default_rng(0)


def boot(x, n=10000):
    x = np.asarray(x, float)
    if len(x) < 2:
        return (np.nan, np.nan)
    m = rng.choice(x, (n, len(x)), replace=True).mean(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def fmt(x):
    lo, hi = boot(x)
    return f"{np.mean(x):+.4f} [{lo:+.4f}, {hi:+.4f}] n {len(x)}"


def negator_contexts(con, w):
    """One row per negator instance: its document, its type and the verbalisation
    around it, read back out of the store.

    The flip variant of an instance carries a substitution row per span it
    edited, and the negator's own span is the one holding that instance's
    negator surface form. A span folds in the space before it where there is
    one, so the negator starts one character in from the span when the span
    opens on a space, and `docs.text` is indexed from there.
    """
    rows, texts = [], {}
    for r in w[w["condition"] == 0].itertuples():
        doc_id, ntype = int(r.doc_id), int(r.ntype)
        if doc_id not in texts:
            texts[doc_id] = con.execute("SELECT text FROM docs WHERE doc_id = ?", (doc_id,)).fetchone()[0]
        text = texts[doc_id]
        spans = con.execute("SELECT t.char_start, t.text FROM substitutions s "
                            "JOIN v_span_text t ON t.span_id = s.span_id "
                            "WHERE s.variant_id = ? ORDER BY t.char_start",
                            (int(r.variant_id),)).fetchall()
        hit = [c for c, t in spans if t.strip().lower() in SURFACE[ntype]]
        start = hit[-1] if hit else spans[-1][0]
        i = start + (1 if text[start] == " " else 0)
        rows.append({"instance": int(r.instance), "doc_id": doc_id, "type": NTYPE[ntype],
                     "context": text[max(0, i - CONTEXT): i + CONTEXT].replace("\n", " ")})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True); ap.add_argument("--run", type=int, required=True)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "results"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(exist_ok=True)
    con = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
    m = pd.read_sql("SELECT v.variant_id, v.doc_id, m.metric, m.value FROM variants v JOIN measurements m "
                    "ON m.variant_id = v.variant_id WHERE v.created_run_id = ?", con, params=(a.run,))
    w = m.pivot(index=["variant_id", "doc_id"], columns="metric", values="value").reset_index()
    base = w[w["condition"].isna()][["doc_id", "fve"]].rename(columns={"fve": "base_fve"})
    d = w[w["condition"].notna()].merge(base, on="doc_id")
    d["dfve"] = d["fve"] - d["base_fve"]
    for c in ("condition", "instance", "ntype", "in_quote"):
        d[c] = d[c].astype(int)
    # one value per (instance, condition): swap draws averaged
    per = d.groupby(["instance", "ntype", "in_quote", "condition"])["dfve"].mean().unstack("condition").reset_index()
    per.columns = [COND.get(c, c) if isinstance(c, (int, np.integer)) else c for c in per.columns]
    inst = negator_contexts(con, w)

    L = [f"# Negation, run {a.run}", "", f"documents {d['doc_id'].nunique()}, baseline FVE mean {base['base_fve'].mean():.3f}",
         f"instances {per['instance'].nunique()} ({(per['instance'] > 0).sum()} negators, {(per['instance'] < 0).sum()} insertions)", ""]

    L += ["## Mean dFVE by condition (one value per instance, swap draws averaged)", "", "| condition | mean dFVE [95% bootstrap] | mean abs dFVE |", "|---|---|---|"]
    for c in COND.values():
        if c in per:
            x = per[c].dropna()
            L.append(f"| {c} | {fmt(x)} | {x.abs().mean():.4f} |")
    L.append("")

    neg = per[per["instance"] > 0]
    L += ["## Paired contrasts over negator instances", "", "| contrast | signed dFVE difference [95% bootstrap] | share > 0 | abs dFVE difference |", "|---|---|---|---|"]
    for a_, b_ in (("flip", "swap_gov"), ("flip", "del_gov"), ("flip", "del_neg"), ("del_gov", "swap_gov")):
        if a_ in neg and b_ in neg:
            p = neg[[a_, b_]].dropna()
            diff = p[a_] - p[b_]; adiff = p[a_].abs() - p[b_].abs()
            L.append(f"| {a_} − {b_} | {fmt(diff)} | {(diff > 0).mean():.2f} | {fmt(adiff)} |")
    L.append("")

    pfs = neg[["flip", "swap_gov"]].dropna()
    r_fs = float(np.corrcoef(pfs["flip"], pfs["swap_gov"])[0, 1])
    rho_fs = float(pfs["flip"].rank().corr(pfs["swap_gov"].rank()))
    L += ["## Correlation of flip with swap_gov across negator instances", "",
          f"Pearson r {r_fs:+.3f}, R squared {r_fs * r_fs:.3f}, Spearman rho {rho_fs:+.3f}, n {len(pfs)}. The paired contrast above "
          f"compares the two conditions in level; this compares them instance by instance.", ""]

    L += ["## By negator type: mean dFVE per condition, and flip − swap_gov paired", "",
          "| type | n | flip | del_neg | del_gov | swap_gov | flip − swap_gov |", "|---|---|---|---|---|---|---|"]
    for t, g in neg.groupby("ntype"):
        cells = []
        for c in ("flip", "del_neg", "del_gov", "swap_gov"):
            x = g[c].dropna() if c in g else pd.Series(dtype=float)
            cells.append(f"{x.mean():+.4f} ± {x.sem():.4f}" if len(x) > 1 else "")
        p = g[["flip", "swap_gov"]].dropna()
        L.append(f"| {NTYPE[t]} | {len(g)} | " + " | ".join(cells) + f" | {fmt(p['flip'] - p['swap_gov']) if len(p) > 1 else ''} |")
    L.append("")

    L += ["## Inside vs outside a quoted stretch, flip − swap_gov", "", "| in_quote | n | flip | swap_gov | flip − swap_gov |", "|---|---|---|---|---|"]
    for q, g in neg.groupby("in_quote"):
        p = g[["flip", "swap_gov"]].dropna()
        L.append(f"| {q} | {len(p)} | {p['flip'].mean():+.4f} | {p['swap_gov'].mean():+.4f} | {fmt(p['flip'] - p['swap_gov'])} |")
    L.append("")

    ins = per[per["instance"] < 0]
    if len(ins) and "ins_not" in ins:
        p = ins[["ins_not", "ins_ctrl"]].dropna()
        L += ["## Insertion: ' not' vs ' just' after the same auxiliary", "",
              f"ins_not {fmt(p['ins_not'])}", f"ins_ctrl {fmt(p['ins_ctrl'])}",
              f"ins_not − ins_ctrl {fmt(p['ins_not'] - p['ins_ctrl'])}, share > 0 {((p['ins_not'] - p['ins_ctrl']) > 0).mean():.2f}",
              f"abs: |ins_not| − |ins_ctrl| {fmt(p['ins_not'].abs() - p['ins_ctrl'].abs())}", ""]

    L += ["## Scale reference", "", f"harness floor from earlier runs is not recomputed here; per-instance flip |dFVE| median {neg['flip'].abs().median():.4f}, "
          f"90th percentile {neg['flip'].abs().quantile(0.9):.4f}", ""]

    j = neg.merge(inst, on="instance")
    j["excess"] = j["flip"] - j["swap_gov"]
    top = j.reindex(j["excess"].abs().sort_values(ascending=False).index).head(12)
    L += ["## Largest flip − swap_gov instances", "", "| instance | doc | type | flip | swap_gov | context |", "|---|---|---|---|---|---|"]
    for _, r in top.iterrows():
        L.append(f"| {r['instance']} | {r['doc_id']} | {r['type']} | {r['flip']:+.3f} | {r['swap_gov']:+.3f} | {r['context'].replace('|', '/')} |")
    L.append("")

    # figure, in FVE points lost so it reads like the other experiments' figures
    figstyle.apply()
    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    fig.subplots_adjust(left=0.36, right=0.97, top=0.95, bottom=0.16)
    cols = [c for c in COND.values() if c in per]
    vals = [per[c].dropna() * -100.0 for c in cols]
    figstyle.floor(ax)
    for i, v in enumerate(vals):
        lo, hi = boot(v)
        ax.errorbar([v.mean()], [i], xerr=[[v.mean() - lo], [hi - v.mean()]], fmt="o", ms=5.5,
                    color=figstyle.SWAP, ecolor=figstyle.SWAP, elinewidth=1.4, capsize=2.5, zorder=3)
    ax.set_yticks(range(len(cols))); ax.set_yticklabels([f"{PLAIN[c]}  ({len(v)})" for c, v in zip(cols, vals)])
    ax.set_ylim(len(cols) - 0.5, -0.5); ax.spines["left"].set_visible(False)
    ax.set_xlabel("FVE points lost, mean with 95% bootstrap interval; grey band is the harness floor")
    figstyle.save(fig, out / "negation_by_condition.png")

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    fig.subplots_adjust(left=0.16, right=0.97, top=0.95, bottom=0.12)
    pp = pfs * -100.0
    lim = max(pp.abs().max().max(), 1.0) * 1.05
    ax.scatter(pp["swap_gov"], pp["flip"], s=10, alpha=.55, color=figstyle.SWAP, lw=0)
    ax.plot([-lim, lim], [-lim, lim], color="#999999", lw=0.8, ls="--")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.set_xlabel("governed word corpus-swapped, FVE points lost")
    ax.set_ylabel("negation flipped, FVE points lost")
    ax.text(0.02, 0.97, f"one point per negator instance, n {len(pp)}\nPearson r {r_fs:+.2f}, Spearman rho {rho_fs:+.2f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=8, color="#444444")
    figstyle.save(fig, out / "negation_flip_vs_swap.png")
    L += ["## Figures", "",
          "negation_by_condition.png: mean FVE points lost per condition with a 95% bootstrap interval and the harness "
          "floor shaded",
          "negation_flip_vs_swap.png: flip against corpus swap of the governed word, one point per negator instance",
          "Both figures are in FVE points lost (100 x the drop in FVE), while the tables above are in raw dFVE", ""]
    (out / "statistics.md").write_text("\n".join(L)); print("\n".join(L))


if __name__ == "__main__":
    main()
