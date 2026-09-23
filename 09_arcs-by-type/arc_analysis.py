#!/usr/bin/env python3
"""Arc against matched control, per Universal Dependencies relation, for run 11.

    python arc_analysis.py --db ../db/ffw_span-ablation_database.sqlite --run 11

Read-only on the store. Rebuilds every pair's interaction

    interaction = e(a) + e(b) - e(both)

from the run's baselines, singles and joints, where e is FVE lost against the
same document's unedited baseline, times 100. Then, per relation type:

  raw        mean over arcs of (arc interaction - its matched control's), with a
             document-clustered standard error
  adjusted   the type's arc coefficient in one pooled least-squares fit over all
             pairs: type fixed effects, one arc indicator per type, and the
             covariates the harness recorded (token distance as bins, Qwen token
             count, in-quote, position, copies elsewhere capped at 3, final-token
             copy, UPOS of each word), with document-clustered errors

UPOS enters additively, one set of dummies per word, not as a pair: within a type
the arc's UPOS pair is close to constant, so a pair effect would be collinear
with the arc indicator.

Prints the report and writes the same text to results/statistics.md. No GPU.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "04_ablation-strategy"))
sys.path.insert(0, str(REPO / "05_dependent-pair"))
from pair_analysis import ols_cluster  # noqa: E402
from swap_analysis import cluster_se  # noqa: E402

RESULTS = HERE / "results"
DEFAULT_DB = REPO / "db" / "ffw_span-ablation_database.sqlite"
FLOOR = 0.044          # harness reproduction floor, FVE points
UNDERPOWERED = {"parataxis", "iobj"}
DIST_BINS = [(1, 1, "1"), (2, 2, "2"), (3, 3, "3"), (4, 6, "4-6"), (7, 10**6, "7+")]
Z95 = 1.959964
WINSOR = (1, 99)   # percentiles of the pooled interaction, for the robustness columns

def dist_bin(d):
    return next(lab for lo, hi, lab in DIST_BINS if lo <= d <= hi)


def load(db_path, run_id):
    c = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cfg = json.loads(c.execute("SELECT config FROM runs WHERE run_id = ?",
                               (run_id,)).fetchone()[0])
    pairs = pd.DataFrame(cfg["pairs"], columns=cfg["pair_columns"])
    pairs = pairs[pairs["planned"].astype(bool)].copy()

    meas = defaultdict(dict)
    for vid, metric, value in c.execute(
            "SELECT variant_id, metric, value FROM measurements "
            "WHERE run_id = ? AND metric IN ('fve', 'pair_id')", (run_id,)):
        meas[vid][metric] = value
    doc_of = dict(c.execute(
        "SELECT variant_id, doc_id FROM variants WHERE created_run_id = ?",
        (run_id,)))
    subs = defaultdict(list)
    for vid, sid in c.execute(
            "SELECT s.variant_id, s.span_id FROM substitutions s "
            "JOIN variants v ON v.variant_id = s.variant_id "
            "WHERE v.created_run_id = ?", (run_id,)):
        subs[vid].append(sid)
    c.close()

    base, single, joint = {}, {}, {}
    for vid, doc in doc_of.items():
        f = meas[vid]["fve"]
        sp = subs.get(vid, [])
        if not sp:
            assert doc not in base, f"two baselines for document {doc}"
            base[doc] = f
        elif len(sp) == 1:
            single[(doc, sp[0])] = f
        else:
            joint[int(meas[vid]["pair_id"])] = f

    def e(doc, f):
        return 100.0 * (base[doc] - f)

    rows = []
    for p in pairs.itertuples(index=False):
        d = int(p.doc_id)
        ea = e(d, single[(d, int(p.span_first))])
        eb = e(d, single[(d, int(p.span_second))])
        eab = e(d, joint[int(p.pair_id)])
        rows.append((int(p.pair_id), ea, eb, eab, ea + eb - eab))
    inter = pd.DataFrame(rows, columns=["pair_id", "e_a", "e_b", "e_both",
                                        "interaction"])
    df = pairs.merge(inter, on="pair_id")
    df["is_arc"] = (df["kind"] == "arc").astype(int)
    df["dist_bin"] = df["distance"].astype(int).map(dist_bin)
    return df, cfg, len(base)


def raw_by_type(df, lo, hi):
    arcs = df[df.kind == "arc"].set_index("pair_id")
    ctrl = df[df.kind == "control"].copy()
    ctrl["match_of"] = ctrl["match_of"].astype(int)
    ctrl = ctrl.set_index("match_of")
    out = {}
    for dep, a in arcs.groupby("dep"):
        a = a[a.index.isin(ctrl.index)]
        cm = ctrl.loc[a.index]
        diff = a["interaction"].values - cm["interaction"].values
        exact = (cm["match_quality"].values == "exact")
        wdiff = (np.clip(a["interaction"].values, lo, hi)
                 - np.clip(cm["interaction"].values, lo, hi))
        out[dep] = dict(
            n=len(a), n_docs=a["doc_id"].nunique(),
            arc=a["interaction"].mean(),
            arc_se=cluster_se(a["interaction"].values, a["doc_id"].values),
            ctrl=cm["interaction"].mean(),
            ctrl_se=cluster_se(cm["interaction"].values, cm["doc_id"].values),
            raw=diff.mean(), raw_se=cluster_se(diff, a["doc_id"].values),
            raw_median=float(np.median(diff)),
            raw_w=wdiff.mean(), raw_w_se=cluster_se(wdiff, a["doc_id"].values),
            n_exact=int(exact.sum()),
            raw_exact=diff[exact].mean() if exact.any() else np.nan,
            raw_exact_se=(cluster_se(diff[exact], a["doc_id"].values[exact])
                          if exact.sum() > 1 else np.nan),
            adjacent_arc=float((a["distance"] == 1).mean()),
            adjacent_ctrl=float((cm["distance"] == 1).mean()))
    return out


def design(df, deps):
    cols, names = [np.ones(len(df))], ["intercept"]
    for dep in deps[1:]:
        cols.append((df.dep == dep).values.astype(float)); names.append(f"type:{dep}")
    for dep in deps:
        cols.append(((df.dep == dep) & (df.is_arc == 1)).values.astype(float))
        names.append(f"arc:{dep}")
    for lab in [b[2] for b in DIST_BINS][1:]:
        cols.append((df.dist_bin == lab).values.astype(float)); names.append(f"dist:{lab}")
    for k in ("nqwen_first", "nqwen_second", "in_quote_first", "in_quote_second",
              "position", "final_copy_first", "final_copy_second"):
        cols.append(df[k].astype(float).values); names.append(k)
    for k in ("copies_first", "copies_second"):
        cols.append(np.minimum(df[k].astype(float).values, 3)); names.append(f"{k}(cap3)")
    for k in ("upos_first", "upos_second"):
        levels = sorted(df[k].unique())
        ref = df[k].value_counts().idxmax()
        for lv in levels:
            if lv == ref:
                continue
            cols.append((df[k] == lv).values.astype(float)); names.append(f"{k}:{lv}")
    return np.column_stack(cols), names


def adjusted_by_type(df, y):
    deps = sorted(df.dep.unique())
    X, names = design(df, deps)
    beta, se = ols_cluster(y, X, df["doc_id"].values)
    coef = dict(zip(names, zip(beta, se)))
    return {dep: coef[f"arc:{dep}"] for dep in deps}, coef, X.shape


def distance_table(df):
    out = []
    for lab in [b[2] for b in DIST_BINS]:
        for kind in ("arc", "control"):
            s = df[(df.dist_bin == lab) & (df.kind == kind)]
            out.append(dict(bin=lab, kind=kind, n=len(s),
                            mean=s["interaction"].mean(),
                            se=cluster_se(s["interaction"].values, s["doc_id"].values)))
    return pd.DataFrame(out)


# ------------------------------------------------------------------ report

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--run", type=int, default=11)
    a = ap.parse_args()

    df, cfg, n_base = load(a.db, a.run)
    w_lo, w_hi = np.percentile(df["interaction"], WINSOR)
    raw = raw_by_type(df, w_lo, w_hi)
    adj, coef, shape = adjusted_by_type(df, df["interaction"].values)
    adj_w, _, _ = adjusted_by_type(df, np.clip(df["interaction"].values, w_lo, w_hi))
    dt = distance_table(df)
    RESULTS.mkdir(parents=True, exist_ok=True)

    n_types = len(raw)
    z_bonf = 3.0233  # two-sided 0.05 / 25
    L = []
    P = L.append
    P(f"# run {a.run}: arc minus matched control, per relation type\n")
    P(f"{len(df)} pairs ({int(df.is_arc.sum())} arcs, {int((1 - df.is_arc).sum())} "
      f"controls) over {df.doc_id.nunique()} documents; {n_base} baselines. "
      f"{cfg['sign']}. Standard errors clustered on document. Pooled fit: "
      f"{shape[0]} rows, {shape[1]} columns. Floor {FLOOR} FVE points. "
      f"95% intervals are per type, unadjusted for {n_types} comparisons; the "
      f"`bonf` column says whether the Bonferroni interval (z {z_bonf}) excludes zero.\n")
    arcs_ = df[df.kind == "arc"]; ctrl_ = df[df.kind == "control"]
    q = ctrl_["match_quality"].value_counts()
    P("Control match quality: " + ", ".join(f"{k} {q.get(k, 0)}" for k in cfg["qualities"])
      + f"; exact {100 * q.get('exact', 0) / len(ctrl_):.1f}% of {len(ctrl_)} controls. "
      f"Adjacent pairs (distance 1): {100 * (arcs_.distance == 1).mean():.1f}% of arcs, "
      f"{100 * (ctrl_.distance == 1).mean():.1f}% of controls.\n")
    P("```")
    P(f"{'type':12s} {'n':>4s} {'docs':>4s} {'arc':>7s} {'ctrl':>7s} "
      f"{'raw':>7s} {'se':>6s} {'adj':>7s} {'se':>6s} {'adj 95% CI':>17s} "
      f"{'bonf':>4s} {'exact n':>7s} {'raw exact':>9s} {'adjacent% arc/ctrl':>18s}")
    for d in sorted(raw, key=lambda k: adj[k][0]):
        r = raw[d]; m, se = adj[d]
        lo, hi = m - Z95 * se, m + Z95 * se
        bonf = "yes" if (m - z_bonf * se > 0 or m + z_bonf * se < 0) else "no"
        P(f"{d:12s} {r['n']:4d} {r['n_docs']:4d} {r['arc']:+7.3f} {r['ctrl']:+7.3f} "
          f"{r['raw']:+7.3f} {r['raw_se']:6.3f} {m:+7.3f} {se:6.3f} "
          f"[{lo:+7.3f},{hi:+7.3f}] {bonf:>4s} {r['n_exact']:7d} "
          f"{r['raw_exact']:+9.3f} {100 * r['adjacent_arc']:5.0f}/{100 * r['adjacent_ctrl']:<5.0f}"
          + ("  underpowered" if d in UNDERPOWERED else ""))
    P("```\n")

    def tally(est, label):
        excl = [d for d in raw if abs(est[d][0]) - Z95 * est[d][1] > 0]
        clear = [d for d in excl if min(abs(est[d][0] - Z95 * est[d][1]),
                                         abs(est[d][0] + Z95 * est[d][1])) > FLOOR]
        bonf = [d for d in raw if abs(est[d][0]) - z_bonf * est[d][1] > 0]
        P(f"{label}: 95% interval excludes zero for {len(excl)} of {n_types} "
          f"({', '.join(sorted(excl)) or 'none'}); of those, interval entirely beyond "
          f"the floor for {len(clear)} ({', '.join(sorted(clear)) or 'none'}); "
          f"Bonferroni interval excludes zero for {len(bonf)} "
          f"({', '.join(sorted(bonf)) or 'none'}).\n")

    tally(adj, "Adjusted")
    tally(adj_w, f"Adjusted, winsorised")

    P(f"## robustness to the tails\n")
    P(f"Pooled interaction {WINSOR[0]}st and {WINSOR[1]}th percentiles: "
      f"{w_lo:+.3f} and {w_hi:+.3f} FVE points, range {df.interaction.min():+.3f} to "
      f"{df.interaction.max():+.3f}. Winsorised columns clip every pair's "
      f"interaction to that band before differencing or fitting.\n")
    P("```")
    P(f"{'type':12s} {'raw mean':>8s} {'median':>7s} {'wins':>7s} {'se':>6s} "
      f"{'adj':>7s} {'adj wins':>8s} {'se':>6s} {'adj wins 95% CI':>17s}")
    for d in sorted(raw, key=lambda k: adj[k][0]):
        r = raw[d]; m, se = adj_w[d]
        P(f"{d:12s} {r['raw']:+8.3f} {r['raw_median']:+7.3f} {r['raw_w']:+7.3f} "
          f"{r['raw_w_se']:6.3f} {adj[d][0]:+7.3f} {m:+8.3f} {se:6.3f} "
          f"[{m - Z95 * se:+7.3f},{m + Z95 * se:+7.3f}]"
          + ("  underpowered" if d in UNDERPOWERED else ""))
    P("```\n")

    P("## interaction by token distance, all types pooled\n")
    P("```")
    P(dt.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
    P("```\n")

    P("## pooled fit, covariate coefficients\n")
    P("```")
    for k, (b, s) in coef.items():
        if k.startswith(("type:", "arc:")):
            continue
        P(f"{k:28s} {b:+8.3f} {s:7.3f}")
    P("```")
    (RESULTS / "statistics.md").write_text("\n".join(L) + "\n")

    print((RESULTS / "statistics.md").read_text())


if __name__ == "__main__":
    main()
