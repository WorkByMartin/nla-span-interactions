#!/usr/bin/env python3
"""Does a corpus swap measure more of a word than a masked-LM marginalisation does?

    python swap_analysis.py --run 5

Reads the corpus-swap run and the masked-LM run it repeats out of the project
database, prints the comparison, writes the same report to results/statistics.md
from the same rendering, and writes four figures beside it. No GPU.
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SWAP = "corpus-swap/pos+len"
DELETION = "deletion"
SHUFFLE = "shuffle"
MLM = "modernbert-large_filler_model/textsub"

FLOOR = 0.044      # harness noise floor in FVE points, from 2e-4 MSE
OPEN = {"NOUN", "VERB", "ADJ", "ADV", "PROPN"}

# the draw counts the per-class spread table is computed at, and the
# smallest class that gets a row of its own. Anything smaller is pooled
# into "other".
DIST_COUNTS = [1, 4, 8, 16]
MIN_CLASS = 10

# FVE points. The scatter's axes are linear inside this and logarithmic
# outside it, which is about twice the harness floor.
LINTHRESH = 0.1


# ------------------------------------------------------------------- loading

def connect(path):
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def singles(conn, run_id, source, docs=None):
    """Draw-level rows of a one-substitution arm, as FVE points lost."""
    q = ("SELECT doc_id, span_id, span_text, substitute, depth, draw_idx, "
         "       fve, base_fve, -100.0 * dfve AS effect "
         "FROM v_single WHERE run_id = ? AND source = ?")
    args = [int(run_id), source]
    if docs:
        q += " AND doc_id IN (%s)" % ",".join("?" * len(docs))
        args += [int(d) for d in docs]
    return [dict(r) for r in conn.execute(q, args)]


def pos_of(conn, span_ids):
    out = {}
    ids = [int(s) for s in span_ids]
    for i in range(0, len(ids), 400):
        chunk = ids[i:i + 400]
        q = ("SELECT span_id, pos, text FROM v_pos WHERE span_id IN (%s)"
             % ",".join("?" * len(chunk)))
        for r in conn.execute(q, chunk):
            out[int(r["span_id"])] = (r["pos"], r["text"])
    return out


def baselines(conn, run_id):
    return {int(r["doc_id"]): float(r["base_fve"]) for r in conn.execute(
        "SELECT doc_id, base_fve FROM v_baseline WHERE run_id = ?",
        (int(run_id),))}


def shuffles(conn, run_id):
    """(doc_id, fve) for every shuffle variant of this run."""
    q = ("SELECT vm.doc_id, vm.variant_id, vm.fve, COUNT(*) AS n_sub "
         "FROM v_variant_metrics vm "
         "JOIN substitutions s ON s.variant_id = vm.variant_id "
         "WHERE vm.run_id = ? AND s.source = ? "
         "GROUP BY vm.doc_id, vm.variant_id, vm.fve")
    return [(int(r["doc_id"]), float(r["fve"]), int(r["n_sub"]))
            for r in conn.execute(q, (int(run_id), SHUFFLE))]


# ------------------------------------------------------------------ measures

def cluster_se(values, clusters):
    """Standard error of a mean with errors correlated inside a cluster.

    The asymptotics this rests on want many clusters. There are forty documents
    here, and the report says so beside every interval it prints.
    """
    v = np.asarray(values, dtype=float)
    n = len(v)
    if n == 0:
        return float("nan")
    m = v.mean()
    g = defaultdict(float)
    for x, c in zip(v, clusters):
        g[c] += x - m
    G = len(g)
    if G < 2:
        return float("nan")
    s2 = sum(t * t for t in g.values()) * G / (G - 1)
    return math.sqrt(s2) / n


def spearman(a, b):
    from scipy.stats import spearmanr
    if len(a) < 3:
        return float("nan"), float("nan")
    r = spearmanr(a, b)
    return float(r.statistic), float(r.pvalue)


def by_span(rows, key="effect"):
    """{span_id: [values in draw order]}."""
    out = defaultdict(list)
    for r in sorted(rows, key=lambda r: (r["span_id"], r["depth"] or 0,
                                         r["draw_idx"] or 0)):
        out[int(r["span_id"])].append(float(r[key]))
    return dict(out)


def variance_components(groups):
    """One-way random-effects split of a per-span, per-draw sample.

    groups: {span_id: [draw values]}, balanced or not. Returns between-span
    variance, within-span variance, the intraclass correlation, and the counts.
    The between component is the method-of-moments estimator, which subtracts
    the within noise that a naive variance of the per-span means would carry.
    """
    gs = [np.asarray(v, dtype=float) for v in groups.values() if len(v) > 1]
    k = len(gs)
    if k < 2:
        return dict(between=float("nan"), within=float("nan"),
                    icc=float("nan"), k=k, n=0)
    ns = np.array([len(g) for g in gs], dtype=float)
    means = np.array([g.mean() for g in gs])
    N = ns.sum()
    grand = float((means * ns).sum() / N)
    ssb = float((ns * (means - grand) ** 2).sum())
    ssw = float(sum(((g - g.mean()) ** 2).sum() for g in gs))
    msb = ssb / (k - 1)
    msw = ssw / (N - k)
    # n0 is the harmonic-style effective group size; equals n when balanced
    n0 = (N - (ns ** 2).sum() / N) / (k - 1)
    between = max((msb - msw) / n0, 0.0)
    within = msw
    total = between + within
    return dict(between=between, within=within,
                icc=(between / total if total > 0 else float("nan")),
                k=k, n=int(N), n0=n0, grand=grand)


def se_curve(groups, counts):
    """Mean standard error of a per-span mean built from the first m draws."""
    out = {}
    for m in counts:
        ses = []
        for v in groups.values():
            v = np.asarray(v[:m], dtype=float)
            if len(v) < 2:
                continue
            ses.append(v.std(ddof=1) / math.sqrt(len(v)))
        out[m] = (float(np.mean(ses)), float(np.median(ses)), len(ses))
    return out


def budget_curve(comp, draws):
    """Variance of a class-level mean at a fixed budget of reconstructor passes.

    With N passes split as n_spans spans by d draws, N = n_spans * d, so

        SE^2 of the class mean = between / n_spans + within / (n_spans * d)
                               = (between * d + within) / N

    which is monotonically increasing in d whenever the between-span variance is
    positive. The ratio against d = 1 is what the table reports.
    """
    b, w = comp["between"], comp["within"]
    base = b + w
    return {d: ((b * d + w) / base if base > 0 else float("nan")) for d in draws}


# ------------------------------------------------------------------- figures

def fig_se_vs_draws(curve, comp, path, n_spans):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ms = sorted(curve)
    mean = [curve[m][0] for m in ms]
    med = [curve[m][1] for m in ms]
    ref = [mean[0] * math.sqrt(ms[0] / m) for m in ms]

    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=200)
    ax.plot(ms, mean, "o-", color="#1f4e79",
            label=f"mean over the {n_spans} spans")
    ax.plot(ms, med, "s--", color="#7a9cc6",
            label=f"median over the {n_spans} spans")
    ax.plot(ms, ref, ":", color="#999999",
            label=r"$1/\sqrt{m}$ from the 4-draw point")
    ax.set_xticks(ms)
    ax.set_xlabel("draws per span, m")
    ax.set_ylabel("standard error of the per-span swap mean (FVE points)")
    ax.axhline(FLOOR, color="#c00000", lw=0.8)
    ax.text(ms[-1], FLOOR, f" harness floor {FLOOR}", color="#c00000",
            va="bottom", ha="right", fontsize=8)
    ax.set_title("Precision of a per-span swap mean against draw count\n"
                 f"between-span sd {math.sqrt(comp['between']):.3f}, "
                 f"within-span sd {math.sqrt(comp['within']):.3f} points",
                 fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def fig_scatter(spans, swap_mean, mlm_mean, pos, path, n_docs):
    """Per-span swap mean against masked-LM mean, on symmetric log axes.

    A handful of spans reach several points in either direction while the bulk
    sits inside a tenth of a point, so both axes are symlog with a linear region
    of +-LINTHRESH. The linear region and the harness floor are drawn as faint
    lines, and y = x is a straight diagonal because both axes carry the same
    transform.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.array([mlm_mean[s] for s in spans])
    y = np.array([swap_mean[s] for s in spans])
    isopen = np.array([pos[s][0] in OPEN for s in spans])

    fig, ax = plt.subplots(figsize=(5.8, 5.4), dpi=200)
    lim = 1.2 * float(max(np.abs(x).max(), np.abs(y).max()))
    ax.set_xscale("symlog", linthresh=LINTHRESH, linscale=0.6)
    ax.set_yscale("symlog", linthresh=LINTHRESH, linscale=0.6)
    ax.plot([-lim, lim], [-lim, lim], "-", color="#bbbbbb", lw=0.8, zorder=0)
    for v in (-LINTHRESH, LINTHRESH):
        ax.axvline(v, color="#c8c8c8", lw=0.6, zorder=0)
        ax.axhline(v, color="#c8c8c8", lw=0.6, zorder=0)
    for v in (-FLOOR, FLOOR):
        ax.axvline(v, color="#dedede", lw=0.5, ls=":", zorder=0)
        ax.axhline(v, color="#dedede", lw=0.5, ls=":", zorder=0)
    ax.scatter(x[isopen], y[isopen], s=16, alpha=0.75, color="#1f4e79",
               label=f"open class ({int(isopen.sum())})")
    ax.scatter(x[~isopen], y[~isopen], s=16, alpha=0.75, color="#c07000",
               marker="^", label=f"closed class ({int((~isopen).sum())})")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ticks = [t for t in (-10, -1, -LINTHRESH, 0, LINTHRESH, 1, 10)
             if abs(t) < lim]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.set_yticklabels([f"{t:g}" for t in ticks])
    ax.set_xlabel("masked-LM marginalisation, mean FVE points lost\n"
                  f"symlog, linear within +-{LINTHRESH:g}; floor {FLOOR:g} "
                  f"dotted")
    ax.set_ylabel("corpus swap, mean FVE points lost\n"
                  f"symlog, linear within +-{LINTHRESH:g}")
    ax.set_title(f"Per-span effect, corpus swap against masked-LM\n"
                 f"{len(spans)} spans over {n_docs} documents, y = x drawn "
                 f"for reference", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.grid(alpha=0.2, lw=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def fig_budget(curves, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.2, 4.2), dpi=200)
    colours = {"swap, all spans": "#1f4e79", "swap, open class": "#4a7fb5",
               "swap, closed class": "#c07000",
               "masked-LM, all spans": "#7a7a7a"}
    for name, c in curves.items():
        ds = sorted(c)
        ax.plot(ds, [c[d] for d in ds], "o-", ms=3.5,
                color=colours.get(name, "#555555"), label=name)
    ax.axhline(1.0, color="#bbbbbb", lw=0.8)
    ax.set_xlabel("draws per span, at a fixed total number of reconstructor passes")
    ax.set_ylabel("variance of the class mean, relative to one draw per span")
    ax.set_title("Spending a fixed pass budget on draws rather than on spans\n"
                 "class-level means only; per-span ranking is a different question",
                 fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def class_groups(spans, pos, min_spans=MIN_CLASS):
    """{class label: [span_id]}, with the small classes pooled into `other`.

    The label is the spaCy coarse tag, except that every class holding fewer
    than min_spans spans is replaced by one `other (TAG, TAG)` bucket, so a
    panel is never drawn over a handful of spans.
    """
    by = defaultdict(list)
    for sp in spans:
        by[pos[sp][0]].append(sp)
    big = {p: v for p, v in by.items() if len(v) >= min_spans}
    small = {p: v for p, v in by.items() if len(v) < min_spans}
    if small:
        label = "other (" + ", ".join(sorted(small)) + ")"
        big[label] = [sp for v in small.values() for sp in v]
    return dict(sorted(big.items(), key=lambda kv: -len(kv[1])))


def first_m_means(groups, spans, m):
    """The per-span mean of the first m draws, for the spans given."""
    return np.array([float(np.mean(groups[sp][:m])) for sp in spans
                     if sp in groups and len(groups[sp]) >= 1])


# ---------------------------------------------------------------------- main

def describe(name, vals, clusters=None):
    v = np.asarray(vals, dtype=float)
    if not len(v):
        return f"  {name:34s} no rows"
    a = np.abs(v)
    se = cluster_se(v, clusters) if clusters is not None else float("nan")
    return (f"  {name:34s} n {len(v):6d}  mean|e| {a.mean():7.3f}  "
            f"median|e| {np.median(a):7.3f}  signed mean {v.mean():+7.3f}"
            + (f" +- {se:.3f}" if se == se else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default="../db/ffw_span-ablation_database.sqlite")
    ap.add_argument("--out", default="results", type=Path)
    ap.add_argument("--swap-run", "--run", type=int, default=5, dest="swap_run",
                    help="the corpus-swap run to report")
    ap.add_argument("--mlm-run", type=int, default=3)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    conn = connect(args.db)
    swap_run = args.swap_run

    swap_rows = singles(conn, swap_run, SWAP)
    if not swap_rows:
        raise SystemExit(f"run {swap_run} has no {SWAP} draws in this database")
    docs = sorted({int(r["doc_id"]) for r in swap_rows})
    del_rows = singles(conn, swap_run, DELETION, docs)
    mlm_rows = singles(conn, args.mlm_run, MLM, docs)
    spans = sorted({int(r["span_id"]) for r in swap_rows})
    span_set = set(spans)
    mlm_rows = [r for r in mlm_rows if int(r["span_id"]) in span_set]
    # every arm is reported over the swap's own spans. The deletion arm covers
    # every target span, but a span whose swap pool came out empty has no swap
    # draws, so it is dropped here rather than reported in one arm and not the
    # other
    n_del_spans = len({int(r["span_id"]) for r in del_rows})
    del_rows = [r for r in del_rows if int(r["span_id"]) in span_set]
    pos = pos_of(conn, spans)
    base_swap = baselines(conn, swap_run)
    base_mlm = baselines(conn, args.mlm_run)
    shuf = shuffles(conn, swap_run)
    cfg = conn.execute("SELECT config, started_at, notes FROM runs WHERE run_id = ?",
                       (swap_run,)).fetchone()
    config = json.loads(cfg["config"]) if cfg["config"] else {}

    doc_of = {int(r["span_id"]): int(r["doc_id"]) for r in swap_rows}
    swap_g = by_span(swap_rows)
    mlm_g = by_span(mlm_rows)
    del_e = {int(r["span_id"]): float(r["effect"]) for r in del_rows}
    swap_mean = {s: float(np.mean(v)) for s, v in swap_g.items()}
    mlm_mean = {s: float(np.mean(v)) for s, v in mlm_g.items()}

    class Tee:
        """Write the report to stdout and to results/statistics.md at once."""
        def __init__(self, path):
            self.f = open(path, "w")
        def write(self, s):
            sys.__stdout__.write(s)
            self.f.write(s)
        def flush(self):
            sys.__stdout__.flush()
            self.f.flush()

    sys.stdout = Tee(args.out / "statistics.md")
    sys.stdout.f.write("```\n")

    # ------------------------------------------------------------------ setup
    print("CORPUS SWAP AGAINST MASKED-LM MARGINALISATION")
    print(f"  swap run {swap_run} ({cfg['started_at']}), "
          f"masked-LM run {args.mlm_run}")
    print(f"  documents {len(docs)}: " + ", ".join(str(d) for d in docs))
    print(f"  spans {len(spans)}, swap draws {len(swap_rows)}, "
          f"deletion variants {len(del_rows)}, masked-LM draws {len(mlm_rows)}")
    if n_del_spans > len(spans):
        print(f"  {n_del_spans - len(spans)} of the {n_del_spans} target spans "
              f"had an empty swap pool and so carry no swap draws. Every arm "
              f"below,\n  deletion included, is reported over the "
              f"{len(spans)} spans the swap covers, so the comparison is like "
              f"for like")
    extra = ""
    if config.get("pool_word_types"):
        extra += f", {config['pool_word_types']} word types"
    if config.get("pool_documents"):
        extra += f", over {config['pool_documents']} documents"
    print(f"  pool {config.get('pool_occurrences', 'unknown')} lexical "
          f"occurrences{extra or ' over the other documents'}")
    if config.get("pool_source"):
        print(f"  pool construction: {config['pool_source']}")
    print(f"  effect = FVE points lost = -100 x (fve - base_fve); "
          f"harness floor {FLOOR} points")
    print(f"  every confidence interval below is clustered on the document, "
          f"and there are {len(docs)} clusters.")
    if len(docs) < 15:
        print("  that is far too few for the asymptotics to be trusted. Read "
              "the intervals as indicative.")
    else:
        print("  that is enough for the asymptotics to be worth something, "
              "though it is still forty and not four hundred.")
    print("  baseline FVE per document, this run against the masked-LM run:")
    for d in docs:
        print(f"    {d:6d}  {base_swap.get(d, float('nan')):.4f}  "
              f"{base_mlm.get(d, float('nan')):.4f}  "
              f"delta {base_swap.get(d, float('nan')) - base_mlm.get(d, float('nan')):+.2e}")

    # --------------------------------------------------------------- leakage
    exact = sum(1 for r in swap_rows
                if r["substitute"] == (r["span_text"] or "").strip())
    ci = sum(1 for r in swap_rows
             if r["substitute"].lower() == (r["span_text"] or "").strip().lower())
    mlm_exact = sum(1 for r in mlm_rows
                    if r["substitute"] == (r["span_text"] or "").strip())
    mlm_ci = sum(1 for r in mlm_rows
                 if r["substitute"].lower() == (r["span_text"] or "").strip().lower())
    print("\nLEAKAGE: draws that put the original word back")
    print(f"  swap       exact {exact}/{len(swap_rows)} "
          f"({100 * exact / len(swap_rows):.2f}%), "
          f"case-blind {ci}/{len(swap_rows)} "
          f"({100 * ci / len(swap_rows):.2f}%)")
    print(f"  masked-LM  exact {mlm_exact}/{len(mlm_rows)} "
          f"({100 * mlm_exact / len(mlm_rows):.2f}%), "
          f"case-blind {mlm_ci}/{len(mlm_rows)} "
          f"({100 * mlm_ci / len(mlm_rows):.2f}%)")
    print("  the swap excludes the original by construction, so a non-zero "
          "count here is a bug")

    # ------------------------------------------------------- above the floor
    def frac_over(rows):
        v = np.abs([r["effect"] for r in rows])
        return (v > FLOOR).mean() if len(v) else float("nan"), len(v)

    mlm_clean = [r for r in mlm_rows
                 if r["substitute"].lower() != (r["span_text"] or "").strip().lower()]
    print(f"\nABOVE THE HARNESS FLOOR: draws with |effect| > {FLOOR} points")
    for name, rows in [("swap, all draws", swap_rows),
                       ("masked-LM, all draws", mlm_rows),
                       ("masked-LM, non-identical draws", mlm_clean),
                       ("deletion, one per span", del_rows)]:
        f, n = frac_over(rows)
        print(f"  {name:34s} {f:6.3f}   n {n}")
    print("  by masked-LM candidate depth:")
    for d in sorted({int(r["depth"]) for r in mlm_rows if r["depth"]}):
        sub = [r for r in mlm_rows if int(r["depth"] or 0) == d]
        subc = [r for r in sub
                if r["substitute"].lower() != (r["span_text"] or "").strip().lower()]
        f, n = frac_over(sub)
        fc, nc = frac_over(subc)
        print(f"    depth {d:2d}   all {f:6.3f} (n {n})   "
              f"non-identical {fc:6.3f} (n {nc})")

    # ------------------------------------------------------------- magnitudes
    print(f"\nEFFECT SIZE, POOLED (FVE points; the interval is a "
          f"document-clustered standard error over {len(docs)} clusters)")
    for name, rows in [("swap", swap_rows), ("masked-LM, all", mlm_rows),
                       ("masked-LM, non-identical", mlm_clean),
                       ("deletion", del_rows)]:
        print(describe(name, [r["effect"] for r in rows],
                       [r["doc_id"] for r in rows]))
    print("  masked-LM by depth:")
    for d in sorted({int(r["depth"]) for r in mlm_rows if r["depth"]}):
        sub = [r for r in mlm_rows if int(r["depth"] or 0) == d]
        print(describe(f"depth {d}", [r["effect"] for r in sub],
                       [r["doc_id"] for r in sub]))

    print("\nEFFECT SIZE, PER DOCUMENT")
    print(f"  {'doc':>6s} {'base FVE':>9s} | {'swap n':>7s} {'mean|e|':>8s} "
          f"{'signed':>8s} | {'MLM mean|e|':>11s} {'signed':>8s} | "
          f"{'del mean|e|':>11s} {'signed':>8s}")
    for d in docs:
        sw = np.array([r["effect"] for r in swap_rows if int(r["doc_id"]) == d])
        ml = np.array([r["effect"] for r in mlm_rows if int(r["doc_id"]) == d])
        de = np.array([r["effect"] for r in del_rows if int(r["doc_id"]) == d])
        print(f"  {d:6d} {base_swap.get(d, float('nan')):9.4f} | "
              f"{len(sw):7d} {np.abs(sw).mean():8.3f} {sw.mean():+8.3f} | "
              f"{np.abs(ml).mean():11.3f} {ml.mean():+8.3f} | "
              f"{np.abs(de).mean():11.3f} {de.mean():+8.3f}")

    # -------------------------------------------------- variance decomposition
    def comp_for(groups, label):
        c = variance_components(groups)
        print(f"  {label:34s} spans {c['k']:4d}  draws/span {c['n0']:5.1f}  "
              f"between {c['between']:8.4f}  within {c['within']:8.4f}  "
              f"ICC {c['icc']:6.3f}  "
              f"sd_between {math.sqrt(c['between']):6.3f}  "
              f"sd_within {math.sqrt(c['within']):6.3f}")
        return c

    open_spans = {s for s in spans if pos[s][0] in OPEN}
    swap_open = {s: v for s, v in swap_g.items() if s in open_spans}
    swap_closed = {s: v for s, v in swap_g.items() if s not in open_spans}
    mlm_open = {s: v for s, v in mlm_g.items() if s in open_spans}
    mlm_closed = {s: v for s, v in mlm_g.items() if s not in open_spans}

    print("\nVARIANCE DECOMPOSITION of the per-draw effect (FVE points squared)")
    print("  one-way random effects: a draw's effect is a span mean plus draw "
          "noise. `between` is the")
    print("  variance of the true span means, `within` the variance of draws "
          "around their own span mean,")
    print("  and ICC = between / (between + within), the share of the spread "
          "that is real span-to-span")
    print("  difference rather than draw noise.")
    c_swap = comp_for(swap_g, "swap, all spans")
    c_swap_o = comp_for(swap_open, "swap, open class")
    c_swap_c = comp_for(swap_closed, "swap, closed class")
    c_mlm = comp_for(mlm_g, "masked-LM, all spans (all depths)")
    c_mlm_o = comp_for(mlm_open, "masked-LM, open class")
    c_mlm_c = comp_for(mlm_closed, "masked-LM, closed class")
    print("  the masked-LM rows pool six candidate depths into the within "
          "term, so their `within`")
    print("  carries a real depth effect as well as draw noise. Per depth:")
    for d in sorted({int(r["depth"]) for r in mlm_rows if r["depth"]}):
        g = by_span([r for r in mlm_rows if int(r["depth"] or 0) == d])
        comp_for(g, f"masked-LM, depth {d}")

    # ------------------------------------------------------- budget arithmetic
    DRAWS = [1, 2, 4, 8, 12, 16]
    print("\nWHERE TO SPEND A FIXED PASS BUDGET, for a CLASS-LEVEL mean")
    print("  N passes split as n_spans spans by d draws, so n_spans = N / d and")
    print("      SE^2 of the class mean = between / n_spans + within / "
          "(n_spans x d) = (between x d + within) / N")
    print("  which increases with d whenever between > 0. The table is that "
          "variance relative to d = 1,")
    print("  so a value of 3 means the same budget spent at that draw count "
          "gives three times the variance,")
    print("  and one draw on many spans is the better buy. This is about "
          "class-level means only: per-span")
    print("  ranking and pair interactions need the per-span mean itself to be "
          "precise, and there the")
    print("  draws are what buys the precision.")
    curves = {}
    for label, c in [("swap, all spans", c_swap),
                     ("swap, open class", c_swap_o),
                     ("swap, closed class", c_swap_c),
                     ("masked-LM, all spans", c_mlm)]:
        curves[label] = budget_curve(c, DRAWS)
    print(f"  {'':34s}" + "".join(f"{('d=' + str(d)):>8s}" for d in DRAWS))
    for label, c in curves.items():
        print(f"  {label:34s}" + "".join(f"{c[d]:8.2f}" for d in DRAWS))
    pos_between = [k for k, c in [("swap, all spans", c_swap),
                                  ("swap, open class", c_swap_o),
                                  ("swap, closed class", c_swap_c),
                                  ("masked-LM, all spans", c_mlm)]
                   if c["between"] > 0]
    print(f"  optimal draws per span for a class-level mean: 1, for "
          + ("every row above" if len(pos_between) == len(curves)
             else "these rows: " + "; ".join(pos_between)))
    d_used = int(round(c_swap["n0"]))
    factor = curves["swap, all spans"].get(d_used) or budget_curve(
        c_swap, [d_used])[d_used]
    n_pass = len(swap_g) * d_used
    print(f"  the {d_used} draws this run took cost {factor:.1f}x the "
          f"variance of the same budget spent one draw per span")
    print(f"  equivalently, the {len(swap_g)} spans x {d_used} draws here "
          f"({n_pass} passes) are worth about {n_pass / factor:.0f} spans at "
          f"one draw each for a class mean")

    # ---------------------------------------------------- precision vs draws
    COUNTS = [4, 8, 12, 16]
    curve = se_curve(swap_g, COUNTS)
    print("\nPRECISION OF A PER-SPAN SWAP MEAN against draw count")
    print(f"  {'draws':>6s} {'mean SE':>9s} {'median SE':>10s} {'spans':>7s} "
          f"{'x floor':>8s}")
    for m in COUNTS:
        mean_se, med_se, n = curve[m]
        print(f"  {m:6d} {mean_se:9.3f} {med_se:10.3f} {n:7d} "
              f"{mean_se / FLOOR:8.1f}")
    print("  the standard error is of one span's own mean, computed from that "
          "span's first m draws")

    # ------------------------------------------------------------- agreement
    print("\nRANK AGREEMENT over spans (Spearman rho on per-span mean effect)")
    print(f"  {'doc':>7s} {'n':>4s} {'swap~MLM':>10s} {'swap~del':>10s} "
          f"{'MLM~del':>10s} {'swap split-half':>16s}")
    for d in list(docs) + ["pooled"]:
        ss = spans if d == "pooled" else [s for s in spans if doc_of[s] == d]
        if len(ss) < 3:
            continue
        sw = [swap_mean[s] for s in ss]
        ml = [mlm_mean.get(s, float("nan")) for s in ss]
        de = [del_e.get(s, float("nan")) for s in ss]
        h1 = [float(np.mean(swap_g[s][:8])) for s in ss]
        h2 = [float(np.mean(swap_g[s][8:16])) for s in ss]
        r1, _ = spearman(sw, ml)
        r2, _ = spearman(sw, de)
        r3, _ = spearman(ml, de)
        r4, _ = spearman(h1, h2)
        print(f"  {str(d):>7s} {len(ss):4d} {r1:10.3f} {r2:10.3f} {r3:10.3f} "
              f"{r4:16.3f}")
    print("  the split half is draws 0 to 7 against draws 8 to 15 of the same "
          "spans, so it is the")
    print("  ceiling any other correlation with the swap could reach at this "
          "draw count")

    # --------------------------------------------------------- word classes
    print("\nOPEN AGAINST CLOSED CLASS (open: " + " ".join(sorted(OPEN)) + ")")
    print(f"  {'method':24s} {'class':>7s} {'spans':>6s} {'draws':>7s} "
          f"{'mean|e|':>9s} {'signed':>9s} {'over floor':>11s}")
    for name, rows in [("swap", swap_rows), ("masked-LM", mlm_rows),
                       ("deletion", del_rows)]:
        for cls, want in [("open", True), ("closed", False)]:
            sub = [r for r in rows
                   if (pos[int(r["span_id"])][0] in OPEN) == want]
            v = np.array([r["effect"] for r in sub])
            ns = len({int(r["span_id"]) for r in sub})
            se = cluster_se(v, [r["doc_id"] for r in sub])
            print(f"  {name:24s} {cls:>7s} {ns:6d} {len(v):7d} "
                  f"{np.abs(v).mean():9.3f} {v.mean():+9.3f} "
                  f"{(np.abs(v) > FLOOR).mean():11.3f}"
                  f"   (signed +- {se:.3f})")
    print("  per part of speech, swap mean |effect| and masked-LM mean |effect|:")
    tally = defaultdict(lambda: [[], [], []])
    for r in swap_rows:
        tally[pos[int(r["span_id"])][0]][0].append(abs(r["effect"]))
    for r in mlm_rows:
        tally[pos[int(r["span_id"])][0]][1].append(abs(r["effect"]))
    for r in del_rows:
        tally[pos[int(r["span_id"])][0]][2].append(abs(r["effect"]))
    print(f"    {'pos':8s} {'spans':>6s} {'swap':>8s} {'MLM':>8s} "
          f"{'deletion':>9s} {'swap/MLM':>9s}")
    for p in sorted(tally, key=lambda p: -np.mean(tally[p][0])):
        a, b, c = tally[p]
        ns = len({int(r["span_id"]) for r in del_rows
                  if pos[int(r["span_id"])][0] == p})
        ratio = np.mean(a) / np.mean(b) if b and np.mean(b) > 0 else float("nan")
        print(f"    {p:8s} {ns:6d} {np.mean(a):8.3f} "
              f"{(np.mean(b) if b else float('nan')):8.3f} "
              f"{(np.mean(c) if c else float('nan')):9.3f} {ratio:9.2f}")

    # ------------------------------------ per-class spread against draw count
    cls = class_groups(spans, pos)
    cls_of = {sp: lab for lab, v in cls.items() for sp in v}
    print("\nPER-CLASS SPREAD OF THE PER-SPAN SWAP MEAN, against the number of "
          "draws it was built from")
    print(f"  a span's value at m is the mean of its first m draws. The mean "
          f"column is over spans and barely\n  moves with m; what moves is the "
          f"spread, which is draw noise leaking into the per-span estimate.\n"
          f"  classes with fewer than {MIN_CLASS} spans are pooled into one "
          f"`other` row. sd is over spans, in FVE points.")
    head = "".join(f"{('m=' + str(m)):>16s}" for m in DIST_COUNTS)
    print(f"  {'class':22s} {'spans':>6s}" + head)
    for label, ss in cls.items():
        cells = []
        for m in DIST_COUNTS:
            v = first_m_means(swap_g, ss, m)
            cells.append(f"{v.mean():+7.3f}/{v.std(ddof=1):6.3f}")
        print(f"  {label:22s} {len(ss):6d}" + "".join(f"{c:>16s}"
                                                      for c in cells))
    print("  each cell is mean/sd over the class's spans")

    # ------------------------------------------- per-class signed class means
    def rows_by_class(rows):
        out = defaultdict(list)
        for r in rows:
            lab = cls_of.get(int(r["span_id"]))
            if lab is not None:
                out[lab].append(r)
        return out

    arms = [("swap", rows_by_class(swap_rows)),
            ("MLM all", rows_by_class(mlm_rows)),
            ("MLM non-id", rows_by_class(mlm_clean)),
            ("deletion", rows_by_class(del_rows))]
    print(f"\nPER-CLASS SIGNED MEAN EFFECT, swap against masked-LM against "
          f"deletion")
    print(f"  FVE points lost, signed, so a negative number means the edit "
          f"IMPROVED the reconstruction.\n  Every interval is a standard error "
          f"clustered on the document, over {len(docs)} clusters.")
    print(f"  {'class':22s} {'spans':>6s}"
          + "".join(f"{name:>22s}" for name, _ in arms))
    for label, ss in cls.items():
        cells = []
        for _, grouped in arms:
            r = grouped.get(label, [])
            if not r:
                cells.append("no rows")
                continue
            v = np.array([x["effect"] for x in r])
            se = cluster_se(v, [x["doc_id"] for x in r])
            cells.append(f"{v.mean():+8.3f} +- {se:6.3f}")
        print(f"  {label:22s} {len(ss):6d}"
              + "".join(f"{c:>22s}" for c in cells))
    print("  MLM non-id drops the masked-LM draws that put the original word "
          "back, which is 45% of them")

    # ------------------------------------------------ deletion against swap
    print("\nDELETION AGAINST SWAP, per class")
    print("  the swap column is the per-span mean over its draws, the deletion "
          "column that span's single\n  deletion variant, and the difference "
          "is paired within the span before it is averaged, so it\n  is not the "
          "difference of the two columns' clustered intervals.")
    print(f"  {'class':22s} {'spans':>6s} {'swap':>9s} {'deletion':>9s} "
          f"{'del - swap':>18s} | {'|swap|':>8s} {'|deletion|':>10s} "
          f"{'|del| - |swap|':>20s} {'ratio':>7s}")
    for label, ss in cls.items():
        pair = [(sp, swap_mean[sp], del_e[sp]) for sp in ss if sp in del_e]
        if not pair:
            continue
        sw = np.array([a for _, a, _ in pair])
        de = np.array([b for _, _, b in pair])
        cl = [doc_of[sp] for sp, _, _ in pair]
        d_signed = de - sw
        d_abs = np.abs(de) - np.abs(sw)
        ratio = (np.abs(de).mean() / np.abs(sw).mean()
                 if np.abs(sw).mean() > 0 else float("nan"))
        print(f"  {label:22s} {len(pair):6d} {sw.mean():+9.3f} "
              f"{de.mean():+9.3f} "
              f"{d_signed.mean():+10.3f} +- {cluster_se(d_signed, cl):5.3f} | "
              f"{np.abs(sw).mean():8.3f} {np.abs(de).mean():10.3f} "
              f"{d_abs.mean():+12.3f} +- {cluster_se(d_abs, cl):5.3f} "
              f"{ratio:7.2f}")
    allpair = [(sp, swap_mean[sp], del_e[sp]) for sp in spans if sp in del_e]
    sw = np.array([a for _, a, _ in allpair])
    de = np.array([b for _, _, b in allpair])
    cl = [doc_of[sp] for sp, _, _ in allpair]
    print(f"  {'ALL':22s} {len(allpair):6d} {sw.mean():+9.3f} {de.mean():+9.3f} "
          f"{(de - sw).mean():+10.3f} +- {cluster_se(de - sw, cl):5.3f} | "
          f"{np.abs(sw).mean():8.3f} {np.abs(de).mean():10.3f} "
          f"{(np.abs(de) - np.abs(sw)).mean():+12.3f} +- "
          f"{cluster_se(np.abs(de) - np.abs(sw), cl):5.3f} "
          f"{np.abs(de).mean() / np.abs(sw).mean():7.2f}")
    r_pool, _ = spearman(sw, de)
    print(f"  Spearman rho between the per-span swap mean and the per-span "
          f"deletion effect, pooled: {r_pool:.3f}")

    # ----------------------------------------- per-class variance and budget
    print("\nPER-CLASS VARIANCE DECOMPOSITION AND PASS BUDGET")
    print("  the same one-way random-effects split as above, and the same "
          "budget arithmetic, computed")
    print("  inside each word class. `between` is the variance of the true "
          "per-span means in that class,")
    print("  `within` the variance of draws around their own span mean, ICC "
          "the share of the spread that")
    print("  is real span-to-span difference. The d= columns are the variance "
          "of that class's mean at a")
    print("  fixed budget of reconstructor passes, relative to spending the "
          "same budget one draw per span,")
    print("  so a value of 3 means three times the variance and one draw on "
          "three times as many spans is")
    print("  the better buy. best d is the draw count that minimises it, which "
          "is 1 whenever between > 0.")
    for arm, groups in [("swap", swap_g), ("masked-LM, all depths", mlm_g)]:
        print(f"  {arm}:")
        print(f"    {'class':22s} {'spans':>6s} {'draws':>6s} {'between':>9s} "
              f"{'within':>8s} {'ICC':>6s}"
              + "".join(f"{('d=' + str(d)):>7s}" for d in DRAWS)
              + f"{'best d':>8s}")
        for label, ss in list(cls.items()) + [("ALL", spans)]:
            g = {s: groups[s] for s in ss if s in groups}
            if len(g) < 2:
                print(f"    {label:22s} {len(g):6d}   too few spans")
                continue
            c = variance_components(g)
            bc = budget_curve(c, DRAWS)
            best = min(DRAWS, key=lambda d: bc[d])
            print(f"    {label:22s} {c['k']:6d} {c['n0']:6.1f} "
                  f"{c['between']:9.4f} {c['within']:8.4f} {c['icc']:6.3f}"
                  + "".join(f"{bc[d]:7.2f}" for d in DRAWS)
                  + f"{best:8d}")

    # -------------------------------------------------------------- shuffle
    shuf_docs = sorted({d for d, _, _ in shuf})
    print(f"\nWORD-ORDER SHUFFLE, the document's own words permuted among its "
          f"own slots\n  run on {len(shuf_docs)} of the {len(docs)} documents")
    print(f"  {'doc':>7s} {'base FVE':>9s} {'shuffles':>9s} "
          f"{'mean FVE':>9s} {'sd':>7s} {'points lost':>12s} "
          f"{'vs deletion sum':>16s}")
    for d in docs:
        rows = [f for dd, f, _ in shuf if dd == d]
        if not rows:
            continue
        b = base_swap[d]
        v = np.array(rows)
        del_sum = sum(r["effect"] for r in del_rows if int(r["doc_id"]) == d)
        print(f"  {d:7d} {b:9.4f} {len(v):9d} {v.mean():9.4f} {v.std(ddof=1):7.4f} "
              f"{-100 * (v.mean() - b):12.3f} {del_sum:16.3f}")
    print("  the last column is the sum of the 40 single-word deletion effects "
          "in that document, which")
    print("  is not the same quantity: it is 40 spans, not all of them, and it "
          "ignores interaction")

    # -------------------------------------------------------------- figures
    fig_se_vs_draws(curve, c_swap, args.out / "swap_se_vs_draws.png",
                    len(swap_g))
    fig_scatter([s for s in spans if s in mlm_mean], swap_mean, mlm_mean, pos,
                args.out / "swap_vs_mlm_scatter.png", len(docs))
    fig_budget(curves, args.out / "budget_draws_vs_spans.png")
    print("\nFIGURES")
    print(f"  swap_se_vs_draws.png: the standard error of one span's "
          f"swap mean against how many draws\n    it was built from, mean and "
          f"median over the {len(swap_g)} spans, with the 1/sqrt(m) reference")
    print(f"  swap_vs_mlm_scatter.png: per-span mean effect, corpus "
          f"swap against masked-LM\n    marginalisation on symmetric "
          f"log axes, linear within +-{LINTHRESH:g}. Open class and "
          f"closed class marked,\n    y = x drawn")
    print(f"  budget_draws_vs_spans.png: variance of a class-level "
          f"mean at a fixed pass budget as the\n    budget is moved from spans "
          f"onto draws")

    sys.stdout.f.write("```\n")
    sys.stdout.flush()
    sys.stdout = sys.__stdout__
    conn.close()


if __name__ == "__main__":
    main()
