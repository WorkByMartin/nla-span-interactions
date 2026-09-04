#!/usr/bin/env python3
"""Do two words on a dependency arc interact more than two matched words that are not?

    python pair_analysis.py --run 7

Reads the dependent-pair run and the pair table the run recorded in the store,
prints the comparison, writes the same report to results/statistics.md from the
same rendering, and writes two figures beside it. No GPU and no model files.

The quantity is the interaction

    interaction = e(a) + e(b) - e(both)

in FVE points, where e is the drop in fraction of variance explained against the
same document's unedited baseline, times 100. Positive means the pair costs LESS
than the sum of its two singles, so what the two words contribute to the
reconstruction overlaps. It is computed per draw against the single that spliced
the same string at the same draw, which is what the run's common random numbers
are for, and then averaged within the pair.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "04_ablation-strategy"))
sys.path.insert(0, str(HERE))

from swap_analysis import FLOOR, cluster_se, connect  # noqa: E402

SWAP = "corpus-swap/pos+len"
PAIR_SCHEME = "pair-ablation/arc+control"
ARC_DEPS = ["dobj", "nsubj", "conj", "appos", "nmod", "advcl", "ccomp",
            "attr", "poss"]
PERMUTATIONS = 2000
MIN_CELL = 12          # smallest POS combination that gets its own regressor


# ------------------------------------------------------------------- loading

def singles(conn, run_id):
    """{(span_id, draw_idx): (doc_id, effect)} for the one-span arm.

    Effect is FVE points LOST against the same document's baseline, so a larger
    number means the edit hurt the reconstruction more.
    """
    q = ("SELECT doc_id, span_id, draw_idx, -100.0 * dfve AS effect "
         "FROM v_single WHERE run_id = ? AND source = ? AND depth = 1")
    return {(int(r["span_id"]), int(r["draw_idx"])):
            (int(r["doc_id"]), float(r["effect"]))
            for r in conn.execute(q, (int(run_id), SWAP))
            if r["effect"] is not None}


def joints(conn, run_id):
    """Draw-level rows of the two-span arm, keyed by the span pair.

    The two substitutions of a joint variant carry the same draw_idx, so one
    grouped query recovers (span pair, draw) without a self join.
    """
    q = ("SELECT vm.variant_id, vm.doc_id, vm.fve, b.base_fve, "
         "       MIN(s.span_id) AS lo, MAX(s.span_id) AS hi, "
         "       MIN(s.draw_idx) AS k, COUNT(*) AS n "
         "FROM v_variant_metrics vm "
         "JOIN v_nsub n2 ON n2.variant_id = vm.variant_id AND n2.n_sub = 2 "
         "JOIN substitutions s ON s.variant_id = vm.variant_id "
         "JOIN v_baseline b ON b.doc_id = vm.doc_id AND b.run_id = vm.run_id "
         "WHERE vm.run_id = ? AND s.source = ? "
         "GROUP BY vm.variant_id, vm.doc_id, vm.fve, b.base_fve")
    out = []
    for r in conn.execute(q, (int(run_id), SWAP)):
        if r["fve"] is None or r["base_fve"] is None:
            continue
        out.append({"doc_id": int(r["doc_id"]), "lo": int(r["lo"]),
                    "hi": int(r["hi"]), "draw": int(r["k"]),
                    "effect": -100.0 * (float(r["fve"])
                                        - float(r["base_fve"]))})
    return out


def check_relations(conn, scheme, table):
    """The pair table against the edges the run wrote, or stop.

    The run recorded each planned pair twice, once as a row of its `runs.config`
    and once as an edge in `relations` under the pair scheme. They are compared
    here on (span_a, span_b, kind) before any number is computed.
    """
    want = {(int(p["span_a"]), int(p["span_b"]),
             f"arc:{p['dep']}" if p["kind"] == "arc" else "control")
            for p in table if p["planned"]}
    got = {(int(r["span_a"]), int(r["span_b"]), r["kind"]) for r in
           conn.execute("SELECT span_a, span_b, kind FROM relations "
                        "WHERE scheme = ?", (scheme,))}
    if want == got:
        return
    only_cfg, only_rel = sorted(want - got), sorted(got - want)
    raise SystemExit(
        f"the run's pair table and the {scheme} edges disagree: "
        f"{len(want)} pairs in the config, {len(got)} edges, "
        f"{len(only_cfg)} only in the config, {len(only_rel)} only in "
        f"relations. First few config-only {only_cfg[:5]}, "
        f"first few relations-only {only_rel[:5]}")


def load_pairs(conn, run_id):
    """The run's pair table, read back from the store, keyed by its two spans.

    The run wrote every column of the table into its `runs.config` row and every
    planned pair into `relations`, so nothing outside the database is needed.
    The one column the config does not carry is the part-of-speech combination
    in document order: `pos_a` and `pos_b` are the dependent and the head, and
    the two are read in position order here off the spans' character offsets.
    """
    row = conn.execute("SELECT config FROM runs WHERE run_id = ?",
                       (int(run_id),)).fetchone()
    if row is None or not row["config"]:
        raise SystemExit(f"run {run_id} has no config row to read the pair "
                         f"table from")
    cfg = json.loads(row["config"])
    cols, packed = cfg.get("pair_columns"), cfg.get("pairs")
    if not cols or packed is None:
        raise SystemExit(f"run {run_id} recorded no pair table in its config")
    table = [dict(zip(cols, p)) for p in packed]
    start = {int(r["span_id"]): int(r["char_start"])
             for r in conn.execute("SELECT span_id, char_start FROM spans")}
    for p in table:
        first = start[int(p["span_a"])] < start[int(p["span_b"])]
        p["pos_first"], p["pos_second"] = ((p["pos_a"], p["pos_b"]) if first
                                           else (p["pos_b"], p["pos_a"]))
    check_relations(conn, cfg.get("pair_scheme", PAIR_SCHEME), table)
    meta = {"draws": cfg["args"]["draws"],
            "min_distance": cfg["min_distance"],
            "unmatched_arcs": cfg.get("unmatched_arcs", []),
            "pair_scheme": cfg.get("pair_scheme", PAIR_SCHEME)}
    return meta, {(min(p["span_a"], p["span_b"]),
                   max(p["span_a"], p["span_b"])): p for p in table}


# ------------------------------------------------------------------ measures

def assemble(pairs_by_span, sing, joint):
    """Per-draw interaction rows, and the reasons rows were dropped.

    A row survives when the joint edit's two spans name a pair in the table and
    both of that draw's singles are present. The three effects then come from
    the same document's baseline, measured in the same batched call.
    """
    rows, dropped = [], Counter()
    for j in joint:
        p = pairs_by_span.get((j["lo"], j["hi"]))
        if p is None:
            dropped["joint variant is not in the pair table"] += 1
            continue
        a = sing.get((p["span_a"], j["draw"]))
        b = sing.get((p["span_b"], j["draw"]))
        if a is None or b is None:
            dropped["a single for this draw is missing"] += 1
            continue
        rows.append({"pair_id": p["pair_id"], "kind": p["kind"],
                     "dep": p["dep"], "doc_id": j["doc_id"],
                     "distance": p["distance"],
                     "combo": f"{p['pos_first']}-{p['pos_second']}",
                     "match_of": p["match_of"],
                     "quality": p["match_quality"], "draw": j["draw"],
                     "e_a": a[1], "e_b": b[1], "e_both": j["effect"],
                     "inter": a[1] + b[1] - j["effect"]})
    return rows, dropped


def by_pair(rows):
    """One record per pair: the mean of its draws, and what it is."""
    g = defaultdict(list)
    for r in rows:
        g[r["pair_id"]].append(r)
    out = {}
    for pid, rs in g.items():
        r0 = rs[0]
        out[pid] = {"pair_id": pid, "kind": r0["kind"], "dep": r0["dep"],
                    "doc_id": r0["doc_id"], "distance": r0["distance"],
                    "combo": r0["combo"], "match_of": r0["match_of"],
                    "quality": r0["quality"], "n_draws": len(rs),
                    "inter": float(np.mean([x["inter"] for x in rs])),
                    "inter_sd": float(np.std([x["inter"] for x in rs], ddof=1))
                    if len(rs) > 1 else float("nan"),
                    "e_a": float(np.mean([x["e_a"] for x in rs])),
                    "e_b": float(np.mean([x["e_b"] for x in rs])),
                    "e_both": float(np.mean([x["e_both"] for x in rs]))}
    return out


def describe(vals, clusters):
    """Mean, document-clustered standard error and count."""
    v = np.asarray(vals, dtype=float)
    if v.size == 0:
        return float("nan"), float("nan"), 0
    return float(v.mean()), cluster_se(v, clusters), int(v.size)


def ols_cluster(y, X, clusters):
    """Least squares with a cluster-robust covariance. Returns (beta, se).

    The sandwich is the usual one for errors correlated inside a cluster and
    independent across them, with the small-sample correction G / (G - 1). There
    are as many clusters as documents, and the report says how many beside every
    interval it prints.
    """
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    groups = defaultdict(list)
    for i, c in enumerate(clusters):
        groups[c].append(i)
    G = len(groups)
    meat = np.zeros((X.shape[1], X.shape[1]))
    for idx in groups.values():
        u = X[idx].T @ resid[idx]
        meat += np.outer(u, u)
    scale = G / max(1, G - 1)
    cov = XtX_inv @ (scale * meat) @ XtX_inv
    return beta, np.sqrt(np.clip(np.diag(cov), 0, None))


def permutation_null(pairs, stat, n=PERMUTATIONS, seed=0):
    """Shuffle the arc label among pairs at the same distance in a document.

    The label is what the experiment assigns, so permuting it inside a stratum
    that holds document and token distance fixed gives the distribution of the
    arc-minus-control gap under the hypothesis that the label carries nothing.
    A stratum with only one label present contributes no information and is left
    alone, which is what makes this a conditional test rather than a global one.
    """
    strata = defaultdict(list)
    for p in pairs:
        strata[(p["doc_id"], p["distance"])].append(p)
    live = [v for v in strata.values()
            if len({x["kind"] for x in v}) == 2]
    obs = stat(pairs)
    rng = random.Random(seed)
    null = []
    labels0 = [p["kind"] for p in pairs]
    for _ in range(n):
        for group in live:
            ks = [p["kind"] for p in group]
            rng.shuffle(ks)
            for p, k in zip(group, ks):
                p["kind"] = k
        null.append(stat(pairs))
    for p, k in zip(pairs, labels0):
        p["kind"] = k
    null = np.asarray(null, float)
    p_two = float((np.abs(null - null.mean()) >= abs(obs - null.mean())).mean())
    return {"observed": obs, "null_mean": float(null.mean()),
            "null_sd": float(null.std(ddof=1)), "p": p_two,
            "strata": len(strata), "informative_strata": len(live),
            "pairs_in_informative_strata": sum(len(v) for v in live)}


def gap(pairs):
    """Mean interaction of the arcs minus mean interaction of the controls."""
    a = [p["inter"] for p in pairs if p["kind"] == "arc"]
    c = [p["inter"] for p in pairs if p["kind"] == "control"]
    if not a or not c:
        return float("nan")
    return float(np.mean(a) - np.mean(c))


# ------------------------------------------------------------------- figures

def fig_per_type(stats, path, n_docs):
    """Arc against control, one row per dep type, with clustered intervals."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    deps = [d for d in ARC_DEPS if d in stats]
    y = np.arange(len(deps))
    fig, ax = plt.subplots(figsize=(6.6, 4.6), dpi=200)
    for off, kind, colour, marker in ((0.16, "arc", "#1f4e79", "o"),
                                      (-0.16, "control", "#c07000", "s")):
        m = [stats[d][kind][0] for d in deps]
        e = [stats[d][kind][1] for d in deps]
        ax.errorbar(m, y + off, xerr=e, fmt=marker, color=colour, ms=4,
                    lw=0, elinewidth=1.1, capsize=2.5,
                    label=f"{kind} pairs")
    ax.axvline(0, color="#444444", lw=0.8)
    ax.axvline(FLOOR, color="#c00000", lw=0.7, ls=":")
    ax.axvline(-FLOOR, color="#c00000", lw=0.7, ls=":")
    ax.set_yticks(y)
    ax.set_yticklabels(deps)
    ax.invert_yaxis()
    ax.set_xlabel("interaction, e(a) + e(b) - e(both), FVE points")
    ax.set_title("Interaction on a dependency arc against a matched control\n"
                 f"pair means, standard errors clustered on {n_docs} documents;"
                 " dotted lines are the harness floor", fontsize=9)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.grid(alpha=0.25, lw=0.5, axis="x")
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def fig_distance(pairs, bins, path, n_docs):
    """Interaction against token distance, arcs and controls side by side."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = max(p["distance"] for p in pairs)
    fig, ax = plt.subplots(figsize=(6.6, 4.4), dpi=200)
    for kind, colour, marker, off in (("arc", "#1f4e79", "o", -0.06),
                                      ("control", "#c07000", "s", 0.06)):
        xs, ms, es = [], [], []
        for lo, hi in bins:
            v = [p for p in pairs if p["kind"] == kind
                 and lo <= p["distance"] <= hi]
            if len(v) < 8:
                continue
            m, e, _ = describe([p["inter"] for p in v],
                               [p["doc_id"] for p in v])
            xs.append(math.sqrt(lo * min(hi, top)) * (1 + off))
            ms.append(m)
            es.append(e)
        ax.errorbar(xs, ms, yerr=es, fmt=marker + "-", color=colour, ms=4,
                    lw=1.0, elinewidth=1.0, capsize=2.5,
                    label=f"{kind} pairs")
    scatter = [(p["distance"], p["inter"]) for p in pairs]
    ax.scatter([x for x, _ in scatter], [y for _, y in scatter], s=2,
               color="#bbbbbb", alpha=0.35, zorder=0, label="single pairs")
    ax.axhline(0, color="#444444", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("token distance between the two words")
    ax.set_ylabel("interaction (FVE points)")
    ax.set_title("Interaction against token distance\n"
                 f"binned means with standard errors clustered on {n_docs} "
                 "documents", fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=str(REPO / "db"
                                        / "ffw_span-ablation_database.sqlite"))
    ap.add_argument("--run", type=int, required=True)
    ap.add_argument("--out", default=str(HERE / "results"), type=Path)
    ap.add_argument("--permutations", type=int, default=PERMUTATIONS)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    conn = connect(args.db)
    meta, pairs_by_span = load_pairs(conn, args.run)
    sing = singles(conn, args.run)
    joint = joints(conn, args.run)
    if not joint:
        raise SystemExit(f"run {args.run} has no two-span {SWAP} variants")
    rows, dropped = assemble(pairs_by_span, sing, joint)
    if not rows:
        raise SystemExit("no draw survived the join against the pair table")
    P = by_pair(rows)
    pairs = sorted(P.values(), key=lambda p: p["pair_id"])
    arcs = [p for p in pairs if p["kind"] == "arc"]
    ctrls = [p for p in pairs if p["kind"] == "control"]
    docs = sorted({p["doc_id"] for p in pairs})
    cfg = conn.execute("SELECT config, started_at, notes, script FROM runs "
                       "WHERE run_id = ?", (args.run,)).fetchone()
    config = json.loads(cfg["config"]) if cfg and cfg["config"] else {}

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

    # ------------------------------------------------------------ the setup
    print("SETUP")
    print(f"  database          {args.db}")
    print(f"  run               {args.run}, {cfg['script'] if cfg else '?'}, "
          f"started {cfg['started_at'] if cfg else '?'}")
    print(f"  pair table        the run's config row, {len(pairs_by_span)} "
          f"pairs, checked against {meta['pair_scheme']} in relations")
    print(f"  documents         {len(docs)}")
    print(f"  pairs measured    {len(pairs)}  ({len(arcs)} arc, "
          f"{len(ctrls)} control)")
    print(f"  draws per pair    {meta['draws']}, "
          f"{Counter(p['n_draws'] for p in pairs).most_common(3)}")
    print(f"  draw-level rows   {len(rows)}")
    print(f"  singles read      {len(sing)}")
    print(f"  joint variants    {len(joint)}")
    for k, v in dropped.most_common():
        print(f"  DROPPED {v} joint variants: {k}")
    print(f"  arc dep types     {', '.join(ARC_DEPS)}")
    print(f"  minimum distance  {meta['min_distance']} tokens")
    print(f"  edit              {SWAP}, the corpus swap of 04, matched on "
          f"spaCy coarse POS,")
    print(f"                    Qwen token count and leading-space parity")
    print(f"  harness floor     {FLOOR} FVE points")
    print()
    print("  SIGN CONVENTION. interaction = e(a) + e(b) - e(both) in FVE "
          "points, where e is")
    print("  the drop in fraction of variance explained against the same "
          "document's unedited")
    print("  baseline, times 100. A POSITIVE interaction means the pair costs "
          "LESS than the")
    print("  sum of its two singles, so the two words carry overlapping "
          "information. A")
    print("  NEGATIVE interaction means the pair costs MORE than the sum, so "
          "the two words")
    print("  are worth more together than apart.")
    print()
    print("  Every interaction is formed per draw against the singles that "
          "spliced the same")
    print("  strings at the same draw, then averaged inside the pair. Standard "
          "errors are")
    print(f"  clustered on document over {len(docs)} clusters unless the line "
          f"says otherwise.")

    # -------------------------------------------------------- overall
    print("\n\nOVERALL, ARC AGAINST CONTROL")
    print(f"  {'set':22s} {'pairs':>7s} {'mean':>9s} {'se':>7s} "
          f"{'median':>8s} {'mean abs':>9s} {'over floor':>11s}")
    for name, group in (("arc pairs", arcs), ("control pairs", ctrls),
                        ("all pairs", pairs)):
        m, se, n = describe([p["inter"] for p in group],
                            [p["doc_id"] for p in group])
        v = np.array([p["inter"] for p in group])
        print(f"  {name:22s} {n:7d} {m:+9.4f} {se:7.4f} "
              f"{np.median(v):+8.4f} {np.abs(v).mean():9.4f} "
              f"{(np.abs(v) > FLOOR).mean():11.3f}")
    matched = [(P[c["match_of"]], c) for c in ctrls if c["match_of"] in P]
    d = [a["inter"] - c["inter"] for a, c in matched]
    dc = [c["doc_id"] for _, c in matched]
    m, se, n = describe(d, dc)
    print(f"\n  paired within the match, arc minus its own control, over "
          f"{n} matched pairs")
    print(f"    {m:+.4f} +- {se:.4f} points")
    print(f"    the pair is more sub-additive on an arc in "
          f"{np.mean(np.array(d) > 0):.3f} of matches")

    # unmatched arcs, said plainly rather than left out
    n_unmatched = len(meta.get("unmatched_arcs", []))
    print(f"\n  {n_unmatched} sampled arcs never found a control and are in "
          f"the arc rows but not in")
    print(f"  the paired comparison.")

    # ------------------------------------------------------------ per type
    print("\n\nPER DEP TYPE")
    print(f"  {'dep':8s} {'n arc':>6s} {'arc mean':>9s} {'se':>7s} "
          f"{'n ctrl':>6s} {'ctrl mean':>10s} {'se':>7s} "
          f"{'arc - ctrl':>11s} {'se':>7s}")
    stats = {}
    for dep in ARC_DEPS:
        A = [p for p in arcs if p["dep"] == dep]
        C = [p for p in ctrls if p["dep"] == dep]
        if not A:
            continue
        am, ase, an = describe([p["inter"] for p in A],
                               [p["doc_id"] for p in A])
        cm, cse, cn = describe([p["inter"] for p in C],
                               [p["doc_id"] for p in C]) if C else \
            (float("nan"), float("nan"), 0)
        mt = [(P[c["match_of"]], c) for c in C if c["match_of"] in P]
        dm, dse, _ = describe([a["inter"] - c["inter"] for a, c in mt],
                              [c["doc_id"] for _, c in mt])
        stats[dep] = {"arc": (am, ase, an), "control": (cm, cse, cn),
                      "diff": (dm, dse)}
        print(f"  {dep:8s} {an:6d} {am:+9.4f} {ase:7.4f} "
              f"{cn:6d} {cm:+10.4f} {cse:7.4f} {dm:+11.4f} {dse:7.4f}")
    print("\n  arc - ctrl is paired inside the match, so it uses only arcs "
          "that found a control.")

    # ----------------------------------------------------------- regression
    print("\n\nREGRESSION, interaction ~ arc + distance + POS pair")
    combos = Counter(p["combo"] for p in pairs)
    keep = sorted(c for c, n in combos.items() if n >= MIN_CELL)
    base_combo = max(keep, key=lambda c: combos[c]) if keep else None
    names = ["intercept", "arc", "log2 distance"]
    cols = []
    for p in pairs:
        cols.append([1.0, 1.0 if p["kind"] == "arc" else 0.0,
                     math.log2(p["distance"])])
    dummies = [c for c in keep if c != base_combo]
    for j, c in enumerate(dummies):
        names.append(f"pos {c}")
        for i, p in enumerate(pairs):
            cols[i].append(1.0 if p["combo"] == c else 0.0)
    names.append("pos other")
    for i, p in enumerate(pairs):
        cols[i].append(0.0 if p["combo"] in keep else 1.0)
    y = [p["inter"] for p in pairs]
    beta, se = ols_cluster(y, cols, [p["doc_id"] for p in pairs])
    print(f"  {len(pairs)} pairs, {len(names)} regressors, "
          f"{len(docs)} clusters")
    print(f"  POS-pair reference cell {base_combo} "
          f"({combos[base_combo] if base_combo else 0} pairs); combinations "
          f"under {MIN_CELL} pairs are pooled into 'pos other'")
    print(f"\n  {'term':28s} {'beta':>9s} {'se':>8s} {'t':>7s}")
    for nm, b, s in zip(names, beta, se):
        if nm.startswith("pos ") and nm != "pos other" and abs(b) < 1e-12:
            continue
        t = b / s if s > 0 else float("nan")
        print(f"  {nm:28s} {b:+9.4f} {s:8.4f} {t:+7.2f}")
    print("\n  'arc' is the coefficient the experiment is about: the extra "
          "sub-additivity of a")
    print("  pair on a dependency arc, holding token distance and the POS "
          "combination fixed.")

    # a smaller model, so the arc term can be read without the POS block
    b2, s2 = ols_cluster(y, [c[:3] for c in cols],
                         [p["doc_id"] for p in pairs])
    print(f"\n  without the POS block: arc {b2[1]:+.4f} +- {s2[1]:.4f}, "
          f"log2 distance {b2[2]:+.4f} +- {s2[2]:.4f}")

    # ------------------------------------------------------ permutation null
    print("\n\nWITHIN-DOCUMENT PERMUTATION NULL")
    perm = permutation_null(pairs, gap, n=args.permutations, seed=0)
    print(f"  the arc label is shuffled among the pairs that share a document "
          f"and a token")
    print(f"  distance, {args.permutations} times. Strata {perm['strata']}, of "
          f"which {perm['informative_strata']} hold both labels and so carry "
          f"information")
    print(f"  ({perm['pairs_in_informative_strata']} pairs).")
    print(f"\n  observed arc minus control   {perm['observed']:+.4f} points")
    print(f"  null mean                    {perm['null_mean']:+.4f}")
    print(f"  null sd                      {perm['null_sd']:.4f}")
    print(f"  two-sided p                  {perm['p']:.4f}")

    # -------------------------------------------------- size against singles
    print("\n\nINTERACTION AGAINST THE SIZE OF THE SINGLES")
    print("  A large interaction on two words that barely matter is not the "
          "same finding as")
    print("  the same number on two words that matter a lot, so the ratio is "
          "reported beside")
    print("  the difference. sum is e(a) + e(b), both in FVE points lost.")
    print(f"\n  {'set':16s} {'mean e(a)':>10s} {'mean e(b)':>10s} "
          f"{'mean sum':>9s} {'mean e(both)':>13s} {'mean inter':>11s} "
          f"{'inter/sum':>10s} {'median |i|/|sum|':>17s}")
    for name, group in (("arc", arcs), ("control", ctrls)):
        ea = np.array([p["e_a"] for p in group])
        eb = np.array([p["e_b"] for p in group])
        eboth = np.array([p["e_both"] for p in group])
        it = np.array([p["inter"] for p in group])
        s = ea + eb
        ok = np.abs(s) > FLOOR
        print(f"  {name:16s} {ea.mean():10.4f} {eb.mean():10.4f} "
              f"{s.mean():9.4f} {eboth.mean():13.4f} {it.mean():+11.4f} "
              f"{it.mean() / s.mean():10.3f} "
              f"{np.median(np.abs(it[ok]) / np.abs(s[ok])):17.3f}")
    print(f"\n  {'set':16s} {'inter > 0':>10s} {'|inter| over floor':>19s} "
          f"{'|inter| > smaller single':>25s}")
    for name, group in (("arc", arcs), ("control", ctrls)):
        it = np.array([p["inter"] for p in group])
        sm = np.array([min(abs(p["e_a"]), abs(p["e_b"])) for p in group])
        print(f"  {name:16s} {(it > 0).mean():10.3f} "
              f"{(np.abs(it) > FLOOR).mean():19.3f} "
              f"{(np.abs(it) > sm).mean():25.3f}")
    print("\n  A positive fraction near 0.5 with a mean near zero would mean "
          "the pair is simply")
    print("  additive. The direction of the mean is the claim; the ratio says "
          "how much of the")
    print("  singles' cost the overlap accounts for.")

    # ------------------------------------------------------------ distance
    print("\n\nINTERACTION AGAINST TOKEN DISTANCE")
    bins = [(2, 2), (3, 3), (4, 5), (6, 9), (10, 17), (18, 1000)]
    print(f"  {'distance':12s} {'n arc':>6s} {'arc mean':>9s} {'se':>7s} "
          f"{'n ctrl':>6s} {'ctrl mean':>10s} {'se':>7s}")
    for lo, hi in bins:
        A = [p for p in arcs if lo <= p["distance"] <= hi]
        C = [p for p in ctrls if lo <= p["distance"] <= hi]
        if not A and not C:
            continue
        am, ase, an = describe([p["inter"] for p in A],
                               [p["doc_id"] for p in A])
        cm, cse, cn = describe([p["inter"] for p in C],
                               [p["doc_id"] for p in C])
        label = f"{lo}" if lo == hi else (f"{lo}+" if hi > 100
                                          else f"{lo} to {hi}")
        print(f"  {label:12s} {an:6d} {am:+9.4f} {ase:7.4f} "
              f"{cn:6d} {cm:+10.4f} {cse:7.4f}")

    # ------------------------------------------------------- match quality
    print("\n\nCONTROL MATCH QUALITY")
    print(f"  {'quality':16s} {'n':>6s} {'ctrl mean':>10s} {'se':>7s} "
          f"{'arc - ctrl':>11s} {'se':>7s}")
    for q in ["exact", "dist+-1", "other-doc", "other-doc+-1"]:
        C = [p for p in ctrls if p["quality"] == q]
        if not C:
            continue
        cm, cse, cn = describe([p["inter"] for p in C],
                               [p["doc_id"] for p in C])
        mt = [(P[c["match_of"]], c) for c in C if c["match_of"] in P]
        dm, dse, _ = describe([a["inter"] - c["inter"] for a, c in mt],
                              [c["doc_id"] for _, c in mt])
        print(f"  {q:16s} {cn:6d} {cm:+10.4f} {cse:7.4f} "
              f"{dm:+11.4f} {dse:7.4f}")
    print("\n  Exact means the same document, the same ordered POS "
          "combination and the same")
    print("  token distance. The other rows relaxed the distance by one token, "
          "or moved to a")
    print("  different document of the same domain.")

    # ------------------------------------------------------ draw-level noise
    print("\n\nDRAW-LEVEL SPREAD")
    sd = np.array([p["inter_sd"] for p in pairs if p["inter_sd"] == p["inter_sd"]])
    nd = np.array([p["n_draws"] for p in pairs])
    print(f"  within-pair sd of the interaction over draws: mean "
          f"{sd.mean():.4f}, median {np.median(sd):.4f} points")
    print(f"  so the standard error of one pair's mean at {int(np.median(nd))} "
          f"draws is about {sd.mean() / math.sqrt(np.median(nd)):.4f} points, "
          f"against a harness floor of {FLOOR}")
    print(f"  spread of the pair means: sd {np.std([p['inter'] for p in pairs], ddof=1):.4f} "
          f"points over {len(pairs)} pairs")

    # ----------------------------------------------------------- per document
    print("\n\nPER DOCUMENT")
    print(f"  {'doc':>6s} {'pairs':>6s} {'arc mean':>9s} {'ctrl mean':>10s} "
          f"{'arc - ctrl':>11s}")
    for d in docs:
        A = [p["inter"] for p in arcs if p["doc_id"] == d]
        C = [p["inter"] for p in ctrls if p["doc_id"] == d]
        if not A and not C:
            continue
        am = np.mean(A) if A else float("nan")
        cm = np.mean(C) if C else float("nan")
        print(f"  {d:6d} {len(A) + len(C):6d} {am:+9.4f} {cm:+10.4f} "
              f"{am - cm:+11.4f}")

    # -------------------------------------------------------------- figures
    fig_per_type(stats, args.out / "pair_arc_vs_control.png", len(docs))
    fig_distance(pairs, bins, args.out / "pair_interaction_vs_distance.png",
                 len(docs))
    print("\n\nFIGURES")
    print("  pair_arc_vs_control.png: interaction on an arc against its "
          "matched control, one\n    row per dep type, standard errors "
          "clustered on document, harness floor marked")
    print("  pair_interaction_vs_distance.png: interaction against token "
          "distance on a log\n    axis, arcs and controls binned separately, "
          "every pair drawn behind")

    sys.stdout.f.write("```\n")
    sys.stdout.flush()
    sys.stdout = sys.__stdout__
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
