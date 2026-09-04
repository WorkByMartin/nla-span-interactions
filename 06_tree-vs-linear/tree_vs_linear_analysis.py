#!/usr/bin/env python3
"""Does the dependency tree explain interaction that linear token distance does not?

    python tree_vs_linear_analysis.py --run 8

Reads the full pairwise run out of the store, forms the n by n interaction matrix
and its standard error matrix for each document, and asks what the syntax adds
once linear distance is already accounted for. Everything the report needs is in
the database: the spans, their token indices, the spaCy parse, the pair
categories the run wrote and the run's own variants. No JSON or CSV intermediate
is read or written.

The quantity is the interaction

    interaction = e(a) + e(b) - e(both)

in FVE points, where e is the drop in fraction of variance explained against the
same document's unedited baseline, times 100. Positive means the pair costs LESS
than the sum of its two singles. It is formed per draw against the singles that
spliced the same strings at the same draw, which is what the run's common random
numbers are for, and then averaged within the pair.

The draws also fix the ceiling. Splitting them in half gives the variance of the
pair means that is reproducible rather than draw noise, and every R squared is
reported both raw and as a fraction of that reliable variance, because no model
can explain the noise.
"""
from __future__ import annotations

import argparse
import math
import random
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for _c in (HERE, REPO / "05_dependent-pair"):
    if (_c / "pair_analysis.py").is_file():
        sys.path.insert(0, str(_c))
        break
else:
    raise SystemExit("cannot find pair_analysis.py beside this file or in "
                     "05_dependent-pair")

from pair_analysis import FLOOR, connect, describe  # noqa: E402

SWAP = "corpus-swap/pos+len"
SPACY = "spacy-en_core_web_sm-3.8.0"
PAIR_SCHEME = "tree-vs-linear/all-pairs"
PERMUTATIONS = 2000
R2_PERMUTATIONS = 500
# 05's nine dep types, so the consistency table can be read against its numbers
DEPS_05 = ["dobj", "nsubj", "conj", "appos", "nmod", "advcl", "ccomp", "attr",
           "poss"]
MIN_CELL = 12                 # smallest level that gets a column of its own
NULL_MIN_DISTANCE = 3         # the nulls run at this token distance and beyond
DIST_CAP = 30                 # exact distances above this share one level
PATH_CAP = 6                  # tree path lengths above this share one level
CATEGORIES = ["arc", "adjacent", "other"]
DIST_BINS = [(1, 1), (2, 2), (3, 3), (4, 5), (6, 9), (10, 17), (18, 33),
             (34, 10000)]
DISCONNECTED = 99


# ------------------------------------------------------------------ the store

def span_meta(conn, span_ids, scheme=SPACY):
    """{span_id: {tok_i, pos, dep, text, doc_id}} for the spans measured."""
    ids = sorted(span_ids)
    out = {s: {"span_id": s} for s in ids}
    marks = ",".join("?" * len(ids))
    for r in conn.execute(
            f"SELECT span_id, doc_id, text FROM v_span_text "
            f"WHERE span_id IN ({marks})", ids):
        out[int(r["span_id"])].update(doc_id=int(r["doc_id"]), text=r["text"])
    for r in conn.execute(
            f"SELECT span_id, key, value FROM labels WHERE scheme = ? "
            f"AND key IN ('tok_i', 'pos', 'dep') AND span_id IN ({marks})",
            [scheme] + ids):
        v = out[int(r["span_id"])]
        v[r["key"]] = int(r["value"]) if r["key"] == "tok_i" else r["value"]
    return out


def head_map(conn, scheme=SPACY):
    """{dependent span: head span} for the parse the store holds."""
    return {int(r["span_a"]): int(r["span_b"]) for r in conn.execute(
        "SELECT span_a, span_b FROM relations WHERE scheme = ? "
        "AND kind = 'head'", (scheme,))}


def recorded_categories(conn, scheme=PAIR_SCHEME):
    """{(lo, hi): kind} the run wrote, so the analysis reads its categories."""
    out = {}
    for r in conn.execute("SELECT span_a, span_b, kind FROM relations "
                          "WHERE scheme = ?", (scheme,)):
        a, b = int(r["span_a"]), int(r["span_b"])
        out[(min(a, b), max(a, b))] = r["kind"]
    return out


def load_run(conn, run_id, source=SWAP):
    """The run's baselines, singles and joint edits, from the base tables.

    Not from v_single and v_pair. Those views resolve the baseline through a
    correlated subquery that re-runs two whole-store aggregates for every row,
    which is affordable for a sample of pairs and is not for a complete matrix
    of a third of a million variants. Three index-covered queries and the join
    in Python give the same numbers.

    Returns (baselines, singles, joints) with the effects already in FVE points
    LOST against the document's own baseline, so a larger number means the edit
    hurt the reconstruction more.
    """
    run_id = int(run_id)
    doc_of = {int(v): int(d) for v, d in conn.execute(
        "SELECT variant_id, doc_id FROM variants WHERE created_run_id = ?",
        (run_id,))}
    fve, dtok = {}, {}
    for vid, metric, val in conn.execute(
            "SELECT variant_id, metric, value FROM measurements "
            "WHERE run_id = ? AND metric IN ('fve', 'dtok')", (run_id,)):
        if int(vid) in doc_of:
            (fve if metric == "fve" else dtok)[int(vid)] = float(val)
    subs = defaultdict(list)
    for vid, span, draw, src in conn.execute(
            "SELECT s.variant_id, s.span_id, s.draw_idx, s.source "
            "FROM substitutions s JOIN variants v "
            "  ON v.variant_id = s.variant_id "
            "WHERE v.created_run_id = ?", (run_id,)):
        subs[int(vid)].append((int(span), draw, src))

    base = {}
    for vid, doc in sorted(doc_of.items()):
        if vid not in subs and vid in fve and doc not in base:
            base[doc] = fve[vid]

    sing, joint, dropped = {}, [], Counter()
    for vid, doc in doc_of.items():
        s = subs.get(vid)
        if not s or vid not in fve:
            continue
        if doc not in base:
            dropped["the document has no baseline under this run"] += 1
            continue
        e = -100.0 * (fve[vid] - base[doc])
        if len(s) == 1:
            span, draw, src = s[0]
            if src == source and draw is not None:
                sing[(span, int(draw))] = (doc, e)
        elif len(s) == 2:
            (a, ka, sa), (b, kb, sb) = s
            if sa != source or sb != source or ka is None or ka != kb:
                dropped["a joint edit is not one draw of the swap"] += 1
                continue
            joint.append({"doc_id": doc, "lo": min(a, b), "hi": max(a, b),
                          "draw": int(ka), "effect": e,
                          "dtok": dtok.get(vid, 0.0)})
        else:
            dropped[f"a variant substitutes {len(s)} spans"] += 1
    return base, sing, joint, dropped


def tree_paths(conn, head, docs, wanted):
    """{(lo, hi): edges between them in the dependency tree}, undirected.

    The path may run through words this experiment could not edit, so the graph
    is every span of the document and not only the measured ones. A pair in
    different components, which happens when the parse is a forest, is
    DISCONNECTED.
    """
    doc_of = {}
    marks = ",".join("?" * len(docs))
    for r in conn.execute(f"SELECT span_id, doc_id FROM spans "
                          f"WHERE doc_id IN ({marks})", sorted(docs)):
        doc_of[int(r["span_id"])] = int(r["doc_id"])
    adj = defaultdict(list)
    for d, h in head.items():
        if d in doc_of and h in doc_of:
            adj[d].append(h)
            adj[h].append(d)
    by_doc = defaultdict(set)
    for lo, hi in wanted:
        by_doc[doc_of.get(lo)].update((lo, hi))
    out = {}
    for doc_id, nodes in by_doc.items():
        for src in sorted(nodes):
            seen = {src: 0}
            q = deque([src])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if v not in seen:
                        seen[v] = seen[u] + 1
                        q.append(v)
            for dst in nodes:
                if dst == src:
                    continue
                k = (min(src, dst), max(src, dst))
                if k in out:
                    continue
                if k[0] in seen or k[1] in seen:
                    out[k] = seen.get(dst, DISCONNECTED)
    for k in wanted:
        out.setdefault(k, DISCONNECTED)
    return out


# ------------------------------------------------------------ the interaction

def cells(sing, joint, meta, head, cats, paths=None):
    """One record per measured pair: the interaction, its SE, its halves, its type.

    The SE is over the run's draws, which is what the masking in the figure is
    against: a cell whose mean is inside two of its own standard errors is not
    separable from zero at this number of draws. The two half means, over the
    even and the odd draws, are what the split-half reliability is computed from.
    """
    per_pair = defaultdict(dict)
    shift = Counter()
    dropped = Counter()
    for j in joint:
        a = sing.get((j["lo"], j["draw"]))
        b = sing.get((j["hi"], j["draw"]))
        if a is None or b is None:
            dropped["a single for this draw is missing"] += 1
            continue
        per_pair[(j["doc_id"], j["lo"], j["hi"])][j["draw"]] = (
            a[1] + b[1] - j["effect"])
        if j.get("dtok"):
            shift[(j["doc_id"], j["lo"], j["hi"])] += 1
    out = []
    for (doc_id, lo, hi), draws in per_pair.items():
        if lo not in meta or hi not in meta:
            dropped["a span of the pair is not in the store"] += 1
            continue
        ti, tj = meta[lo].get("tok_i"), meta[hi].get("tok_i")
        if ti is None or tj is None:
            dropped["a span of the pair has no token index"] += 1
            continue
        kind = cats.get((lo, hi))
        if kind is None:
            arc = head.get(lo) == hi or head.get(hi) == lo
            cat = "arc" if arc else ("adjacent" if abs(ti - tj) == 1
                                     else "other")
            dep = (meta[lo].get("dep") if head.get(lo) == hi
                   else meta[hi].get("dep")) if arc else None
        elif kind.startswith("arc"):
            cat, dep = "arc", (kind.split(":", 1)[1] if ":" in kind else None)
        else:
            cat, dep = kind, None
        v = np.asarray([draws[k] for k in sorted(draws)], float)
        ev = [draws[k] for k in sorted(draws) if k % 2 == 0]
        od = [draws[k] for k in sorted(draws) if k % 2 == 1]
        first, second = (meta[lo], meta[hi]) if ti <= tj else (meta[hi],
                                                              meta[lo])
        out.append({"doc_id": doc_id, "lo": lo, "hi": hi,
                    "distance": abs(ti - tj), "category": cat,
                    "arc": cat == "arc", "dep": dep,
                    "adjacent": abs(ti - tj) == 1,
                    "path": (paths or {}).get((lo, hi), DISCONNECTED),
                    "pos": f"{first.get('pos')}-{second.get('pos')}",
                    "text": f"{first.get('text')} .. {second.get('text')}",
                    "shifted": shift.get((doc_id, lo, hi), 0),
                    "n_draws": len(v),
                    "inter": float(v.mean()),
                    "half_a": float(np.mean(ev)) if ev else float("nan"),
                    "half_b": float(np.mean(od)) if od else float("nan"),
                    "se": (float(v.std(ddof=1) / math.sqrt(len(v)))
                           if len(v) > 1 else float("nan"))})
    return out, dropped


def masses(rows):
    """Absolute interaction mass by category, and the count shares.

    The three categories partition the matrix, so the shares sum to one.
    """
    tot = sum(abs(r["inter"]) for r in rows)
    if tot == 0:
        return None
    out = {"total": tot, "pairs": len(rows)}
    for c in CATEGORIES:
        g = [r for r in rows if r["category"] == c]
        out[f"mass_{c}"] = sum(abs(r["inter"]) for r in g) / tot
        out[f"n_{c}"] = len(g)
        out[f"share_{c}"] = len(g) / len(rows)
        out[f"cell_{c}"] = (float(np.mean([abs(r["inter"]) for r in g]))
                            if g else float("nan"))
    return out


# -------------------------------------------------------------- reliability

def reliability(rows):
    """How much of the spread of the pair means is signal rather than draw noise.

    The two halves are the even and the odd draws, so they are independent given
    the pair. Their covariance estimates the variance of the true pair means,
    which is the ceiling any model of those means can reach.
    """
    a = np.array([r["half_a"] for r in rows], float)
    b = np.array([r["half_b"] for r in rows], float)
    m = np.array([r["inter"] for r in rows], float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b, m = a[ok], b[ok], m[ok]
    if a.size < 3:
        return None
    v_obs = float(m.var(ddof=1))
    cov = float(np.cov(a, b, ddof=1)[0, 1])
    r = float(np.corrcoef(a, b)[0, 1])
    return {"n": int(a.size), "v_obs": v_obs, "v_reliable": cov,
            "v_noise": v_obs - cov,
            "reliability": cov / v_obs if v_obs > 0 else float("nan"),
            "r_half": r,
            "r_spearman_brown": 2 * r / (1 + r) if r > -1 else float("nan"),
            "sd_obs": math.sqrt(max(v_obs, 0.0)),
            "sd_reliable": math.sqrt(max(cov, 0.0))}


# --------------------------------------------------------------- regression

def block(labels, min_cell=MIN_CELL, reference=None):
    """Dummy columns for a categorical variable, rare levels pooled into `other`.

    The most common level is the reference and gets no column, so the block is
    full rank beside an intercept.
    """
    counts = Counter(labels)
    keep = {k for k, n in counts.items() if n >= min_cell}
    lab = [x if x in keep else "other" for x in labels]
    levels = sorted(set(lab), key=lambda k: (-Counter(lab)[k], str(k)))
    ref = reference if reference in levels else (levels[0] if levels else None)
    cols = [k for k in levels if k != ref]
    X = np.zeros((len(lab), len(cols)))
    at = {k: j for j, k in enumerate(cols)}
    for i, k in enumerate(lab):
        if k in at:
            X[i, at[k]] = 1.0
    return X, [str(c) for c in cols]


def dist_level(r):
    return f"d{min(r['distance'], DIST_CAP)}"


def path_level(r):
    p = r["path"]
    if p >= DISCONNECTED:
        return "path-none"
    return f"path{min(p, PATH_CAP)}" + ("+" if p > PATH_CAP else "")


def r2_of(y, X):
    """Ordinary least squares R squared, with a pseudo-inverse for safety."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def base_design(rows, with_doc=True):
    """Intercept, document dummies and the saturated token distance block."""
    n = len(rows)
    parts = [np.ones((n, 1))]
    if with_doc:
        D, _ = block([r["doc_id"] for r in rows], min_cell=1)
        parts.append(D)
    doc_only = np.hstack(parts)
    Dist, dn = block([dist_level(r) for r in rows], min_cell=1)
    return doc_only, np.hstack([doc_only, Dist]), len(dn)


def var_contest(rows, with_doc=True):
    """Tree path length against token distance, each on its own, signed interaction.

    The registered comparison. Both are saturated categorical blocks fitted
    separately over the same rows, so neither is given the other's columns to
    lean on, and both are read against the split-half ceiling because neither
    can explain the draw noise.
    """
    if len(rows) < 8:
        return None
    y = np.array([r["inter"] for r in rows], float)
    n = len(rows)
    parts = [np.ones((n, 1))]
    if with_doc:
        D, _ = block([r["doc_id"] for r in rows], min_cell=1)
        parts.append(D)
    doc_only = np.hstack(parts)
    r2_doc = r2_of(y, doc_only)
    Dist, dn = block([dist_level(r) for r in rows], min_cell=1)
    Tree, tn = block([path_level(r) for r in rows], min_cell=1)
    r2_d = r2_of(y, np.hstack([doc_only, Dist])) - r2_doc
    r2_t = r2_of(y, np.hstack([doc_only, Tree])) - r2_doc
    rel = reliability(rows)
    ceiling = (rel["reliability"] if rel and rel["reliability"] > 0
               else float("nan"))
    return {"n": n, "ceiling": ceiling, "rel": rel,
            "distance": r2_d, "tree": r2_t, "diff": r2_t - r2_d,
            "distance_of_reliable": r2_d / ceiling,
            "tree_of_reliable": r2_t / ceiling,
            "diff_of_reliable": (r2_t - r2_d) / ceiling,
            "cols": {"distance": len(dn), "tree": len(tn)}}


def contest_diff(rows, select, with_doc=True):
    """The contest statistic on the pairs `select` picks out, as a fraction.

    `select` reads the CURRENT path length, so under a permutation the stratum
    it defines moves with the labels, which is what makes a stratified null on a
    path-defined subset coherent.
    """
    sub = [r for r in rows if select(r)]
    v = var_contest(sub, with_doc)
    if v is None or not np.isfinite(v["ceiling"]) or v["ceiling"] <= 0:
        return float("nan")
    return v["diff_of_reliable"]


def permute_contest(rows, select, n, seed=0, with_doc=True):
    """The within-distance shuffle, with the contest difference as the statistic."""
    strata = defaultdict(list)
    for i, r in enumerate(rows):
        strata[(r["doc_id"], r["distance"])].append(i)
    live = [v for v in strata.values()
            if len({rows[i]["path"] for i in v}) > 1]
    obs = contest_diff(rows, select, with_doc)
    rng = random.Random(seed)
    keep = [(r["path"], r["dep"], r["arc"], r["category"]) for r in rows]
    null = []
    for _ in range(n):
        for idx in live:
            vals = [keep[i] for i in idx]
            rng.shuffle(vals)
            for i, v in zip(idx, vals):
                (rows[i]["path"], rows[i]["dep"], rows[i]["arc"],
                 rows[i]["category"]) = v
        null.append(contest_diff(rows, select, with_doc))
    for r, v in zip(rows, keep):
        r["path"], r["dep"], r["arc"], r["category"] = v
    null = np.asarray([x for x in null if np.isfinite(x)], float)
    if null.size < 2 or not np.isfinite(obs):
        return {"observed": obs, "null_mean": float("nan"),
                "null_sd": float("nan"), "excess": float("nan"),
                "z": float("nan"), "p_upper": float("nan"), "n": int(null.size),
                "strata": len(strata), "informative_strata": len(live)}
    sd = float(null.std(ddof=1))
    return {"observed": obs, "null_mean": float(null.mean()), "null_sd": sd,
            "excess": obs - float(null.mean()),
            "z": (obs - float(null.mean())) / sd if sd > 0 else float("nan"),
            "p_upper": float(((null >= obs).sum() + 1) / (null.size + 1)),
            "n": int(null.size), "strata": len(strata),
            "informative_strata": len(live)}


def models(rows, with_doc=True):
    """The nested designs: distance, then the tree, then the tree with labels.

    Distance enters saturated, one column per exact token distance up to the
    cap, so linear distance is given every chance before the tree is asked to
    add anything.
    """
    doc_only, base, n_dist = base_design(rows, with_doc)
    Tree, tn = block([path_level(r) for r in rows])
    Dep, pn = block([r["dep"] or "no-arc" for r in rows])
    Pos, qn = block([r["pos"] for r in rows])
    return {"doc": doc_only, "base": base,
            "tree": np.hstack([base, Tree]),
            "tree+labels": np.hstack([base, Tree, Dep, Pos]),
            "cols": {"distance": n_dist, "tree": len(tn), "dep": len(pn),
                     "pos": len(qn)}}


def variance_accounted(rows, with_doc=True):
    """R squared of each nested model and the increments between them."""
    y = np.array([r["inter"] for r in rows], float)
    M = models(rows, with_doc)
    r2 = {k: r2_of(y, M[k]) for k in ("doc", "base", "tree", "tree+labels")}
    rel = reliability(rows)
    out = {"r2": r2, "cols": M["cols"], "n": len(rows), "rel": rel,
           "r2_distance": r2["base"] - (r2["doc"] if with_doc else 0.0),
           "d_tree": r2["tree"] - r2["base"],
           "d_tree_labels": r2["tree+labels"] - r2["base"],
           "d_labels": r2["tree+labels"] - r2["tree"]}
    ceiling = (rel["reliability"] if rel and rel["reliability"] > 0
               else float("nan"))
    for k in ("r2_distance", "d_tree", "d_tree_labels", "d_labels"):
        out[k + "_of_reliable"] = out[k] / ceiling
    out["ceiling"] = ceiling
    return out


def permute_tree(rows, n, seed=0, with_doc=True):
    """Shuffle the tree variables among the pairs sharing a document and distance.

    Path length, arc status and the dep label move together, because they are
    one description of the same edge. Distance and document stay where they are,
    so anything the increment finds is structure the linear model could not have
    had. The base design and its R squared do not move under the shuffle, so
    they are built once.
    """
    y = np.array([r["inter"] for r in rows], float)
    _, B, _ = base_design(rows, with_doc)
    r2_base = r2_of(y, B)

    def increment():
        Tree, _ = block([path_level(r) for r in rows])
        return r2_of(y, np.hstack([B, Tree])) - r2_base

    strata = defaultdict(list)
    for i, r in enumerate(rows):
        strata[(r["doc_id"], r["distance"])].append(i)
    live = [v for v in strata.values()
            if len({(rows[i]["path"], rows[i]["arc"]) for i in v}) > 1]
    obs = increment()
    rng = random.Random(seed)
    keep = [(r["path"], r["dep"], r["arc"], r["category"]) for r in rows]
    null = []
    for _ in range(n):
        for idx in live:
            vals = [keep[i] for i in idx]
            rng.shuffle(vals)
            for i, v in zip(idx, vals):
                (rows[i]["path"], rows[i]["dep"], rows[i]["arc"],
                 rows[i]["category"]) = v
        null.append(increment())
    for r, v in zip(rows, keep):
        r["path"], r["dep"], r["arc"], r["category"] = v
    null = np.asarray(null, float)
    sd = float(null.std(ddof=1))
    return {"observed": obs, "null_mean": float(null.mean()), "null_sd": sd,
            "excess": obs - float(null.mean()),
            "z": (obs - float(null.mean())) / sd if sd > 0 else float("nan"),
            "p_upper": float(((null >= obs).sum() + 1) / (n + 1)), "n": n,
            "strata": len(strata), "informative_strata": len(live),
            "pairs_in_informative_strata": sum(len(v) for v in live)}


# ------------------------------------------------------- the arc-cell null

def arc_cell_mean(rows):
    """Mean absolute interaction per arc cell."""
    g = [abs(r["inter"]) for r in rows if r["arc"]]
    return float(np.mean(g)) if g else float("nan")


def permutation_null(rows, stat, n=PERMUTATIONS, seed=0):
    """Reassign arc status among the pairs sharing a document and a distance.

    Interaction decays with token distance and arcs are short, so a raw
    comparison of arcs against everything else would recover distance and call
    it syntax. Permuting the arc label inside a (document, distance) stratum
    holds both fixed, and a stratum with only one label present carries no
    information and is left alone.
    """
    strata = defaultdict(list)
    for r in rows:
        strata[(r["doc_id"], r["distance"])].append(r)
    live = [v for v in strata.values() if len({x["arc"] for x in v}) == 2]
    obs = stat(rows)
    rng = random.Random(seed)
    keep = [r["arc"] for r in rows]
    null = []
    for _ in range(n):
        for group in live:
            flags = [r["arc"] for r in group]
            rng.shuffle(flags)
            for r, a in zip(group, flags):
                r["arc"] = a
        null.append(stat(rows))
    for r, a in zip(rows, keep):
        r["arc"] = a
    null = np.asarray(null, float)
    sd = float(null.std(ddof=1))
    return {"observed": obs, "null_mean": float(null.mean()), "null_sd": sd,
            "excess": obs - float(null.mean()),
            "z": (obs - float(null.mean())) / sd if sd > 0 else float("nan"),
            "n": n,
            "p_upper": float(((null >= obs).sum() + 1) / (n + 1)),
            "p_two": float((np.abs(null - null.mean())
                            >= abs(obs - null.mean())).mean()),
            "strata": len(strata), "informative_strata": len(live),
            "pairs_in_informative_strata": sum(len(v) for v in live),
            "arcs_in_informative_strata": sum(
                1 for v in live for x in v if x["arc"])}


# ---------------------------------------------------------------------- figure

def fig_matrix(doc_id, rows, meta, path):
    """One document's interaction matrix, with its head arcs drawn above it.

    Cells whose mean is inside two of their own standard errors are masked, so
    what is left is what the draws separate from zero. The arc strip above the
    matrix is the spaCy parse over the same words in the same order, so a band
    of colour can be read off against the syntax directly.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Arc

    spans = sorted({s for r in rows for s in (r["lo"], r["hi"])},
                   key=lambda s: meta[s]["tok_i"])
    at = {s: i for i, s in enumerate(spans)}
    n = len(spans)
    M = np.full((n, n), np.nan)
    SE = np.full((n, n), np.nan)
    for r in rows:
        i, j = at[r["lo"]], at[r["hi"]]
        M[i, j] = M[j, i] = r["inter"]
        SE[i, j] = SE[j, i] = r["se"]
    keep = np.abs(M) >= 2 * SE
    shown = np.ma.masked_where(~keep, M)

    side = max(7.0, 0.085 * n)
    fig = plt.figure(figsize=(side, side * 1.1), dpi=200)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 9], width_ratios=[40, 1],
                          hspace=0.02, wspace=0.02)
    axa = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[1, 0], sharex=axa)
    cax = fig.add_subplot(gs[1, 1])

    v = np.abs(M[np.isfinite(M)])
    vmax = max(float(np.percentile(v, 99)) if v.size else 1.0, 1e-6)
    cmap = matplotlib.colormaps["RdBu_r"].copy()
    cmap.set_bad("#f2f2f2")
    im = ax.imshow(shown, cmap=cmap, vmin=-vmax, vmax=vmax,
                   interpolation="nearest")
    fig.colorbar(im, cax=cax).set_label("interaction (FVE points)", fontsize=7)
    cax.tick_params(labelsize=6)

    fs = 6.0 if n <= 60 else (4.0 if n <= 100 else 3.0)
    labels = [meta[s]["text"] for s in spans]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=fs)
    ax.set_yticklabels(labels, fontsize=fs)
    ax.tick_params(length=1.5, pad=1)
    ax.set_xlim(-0.5, n - 0.5)

    drawn = 0
    for r in rows:
        if r["category"] != "arc":
            continue
        i, j = sorted((at[r["lo"]], at[r["hi"]]))
        axa.add_patch(Arc(((i + j) / 2.0, 0.0), width=max(j - i, 0.4),
                          height=2.0 * math.sqrt(max(j - i, 1)),
                          theta1=0, theta2=180, lw=0.6, color="#1f4e79",
                          alpha=0.85))
        drawn += 1
    axa.set_ylim(0, math.sqrt(max(n - 1, 1)) + 0.5)
    axa.set_yticks([])
    axa.tick_params(labelbottom=False, length=0)
    for s in axa.spines.values():
        s.set_visible(False)
    axa.set_title(
        f"doc {doc_id}: pairwise interaction over {n} eligible words, "
        f"{drawn} spaCy head arcs above\n"
        f"cells inside two standard errors of zero are left grey; "
        f"positive means the pair costs less than the sum of its singles",
        fontsize=8)

    fig.savefig(path, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return {"n": n, "cells": int(np.isfinite(M).sum() // 2),
            "kept": int(keep.sum() // 2), "arcs": drawn, "vmax": vmax}


# ------------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=str(REPO / "db"
                                        / "ffw_span-ablation_database.sqlite"))
    ap.add_argument("--run", type=int, required=True,
                    help="the run id, which is handed out at load time and has "
                         "no default")
    ap.add_argument("--out", default=str(HERE / "results"), type=Path)
    ap.add_argument("--permutations", type=int, default=PERMUTATIONS)
    ap.add_argument("--r2-permutations", type=int, default=R2_PERMUTATIONS,
                    help="permutations for the null on the R squared "
                         "increment, which refits a regression each time")
    ap.add_argument("--scheme", default=SPACY)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    conn = connect(args.db)
    base, sing, joint, load_drop = load_run(conn, args.run)
    if not joint:
        raise SystemExit(f"run {args.run} has no two-span variants")
    ids = {s for s, _ in sing} | {x for j in joint for x in (j["lo"], j["hi"])}
    meta = span_meta(conn, ids, args.scheme)
    head = head_map(conn, args.scheme)
    cats = recorded_categories(conn)
    docs_in = {v.get("doc_id") for v in meta.values() if v.get("doc_id")}
    wanted = {(j["lo"], j["hi"]) for j in joint}
    paths = tree_paths(conn, head, docs_in, wanted)
    rows, dropped = cells(sing, joint, meta, head, cats, paths)
    dropped.update(load_drop)
    if not rows:
        raise SystemExit("no pair survived the join against the store")
    cfg = conn.execute("SELECT script, notes, started_at, config FROM runs "
                       "WHERE run_id = ?", (args.run,)).fetchone()
    by_doc = defaultdict(list)
    for r in rows:
        by_doc[r["doc_id"]].append(r)
    docs = sorted(by_doc)

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

    print("SETUP")
    print(f"  database          {args.db}")
    print(f"  run               {args.run}, {cfg['script'] if cfg else '?'}, "
          f"started {cfg['started_at'] if cfg else '?'}")
    print(f"  notes             {(cfg['notes'] or '')[:300] if cfg else ''}")
    print(f"  documents         {len(docs)}: "
          f"{', '.join(str(d) for d in docs)}")
    print(f"  measured pairs    {len(rows)}")
    print(f"  draws per pair    "
          f"{Counter(r['n_draws'] for r in rows).most_common(3)}")
    print(f"  baselines         {len(base)}")
    print(f"  singles read      {len(sing)}")
    print(f"  joint variants    {len(joint)}")
    for k, v in dropped.most_common():
        print(f"  DROPPED {v} joint variants: {k}")
    print(f"  parse             {args.scheme}, {len(head)} head relations")
    print(f"  categories        {len(cats)} pairs carry a category recorded by "
          f"the run under {PAIR_SCHEME}")
    print(f"  prompt shifts     {sum(1 for r in rows if r['shifted'])} pairs "
          f"had at least one draw whose splice moved the prompt length")
    print(f"  tree path lengths "
          f"{dict(sorted(Counter(min(r['path'], DISCONNECTED) for r in rows).items())[:8])}"
          f" ... disconnected "
          f"{sum(1 for r in rows if r['path'] >= DISCONNECTED)}")
    print(f"  harness floor     {FLOOR} FVE points")
    print()
    print("  SIGN CONVENTION. interaction = e(a) + e(b) - e(both) in FVE "
          "points, where e is")
    print("  the drop in fraction of variance explained against the same "
          "document's unedited")
    print("  baseline, times 100. A POSITIVE interaction means the pair costs "
          "LESS than the")
    print("  sum of its two singles, so the two words carry overlapping "
          "information.")
    print()
    print("  CATEGORIES. A pair is on an ARC when the store holds a spaCy head "
          "relation between")
    print("  its two words in either direction, ADJACENT when their token "
          "indices differ by one")
    print("  and there is no arc, and OTHER otherwise. The three partition the "
          "matrix.")
    print()
    print("  TOKEN DISTANCE is the difference of the two token indices. TREE "
          "PATH LENGTH is the")
    print("  number of dependency edges between the two words, over every "
          "token of the document")
    print("  and not only the editable ones, so an arc is path length one.")

    # ------------------------------------------------------------ reliability
    print("\n\nSPLIT-HALF RELIABILITY OF THE INTERACTION")
    print("  The even draws and the odd draws give two independent estimates "
          "of each pair's")
    print("  interaction. Their covariance is the variance of the true pair "
          "means, which is the")
    print("  ceiling on any model of them; the rest of the observed spread is "
          "draw noise.")
    print(f"\n  {'doc':>7s} {'pairs':>7s} {'sd observed':>12s} "
          f"{'var observed':>13s} {'var reliable':>13s} {'var noise':>11s} "
          f"{'reliability':>12s} {'half r':>8s} {'S-B':>7s}")
    rel_doc = {}
    for d in docs:
        rl = reliability(by_doc[d])
        rel_doc[d] = rl
        if rl is None:
            continue
        print(f"  {d:7d} {rl['n']:7d} {rl['sd_obs']:12.4f} "
              f"{rl['v_obs']:13.5f} {rl['v_reliable']:13.5f} "
              f"{rl['v_noise']:11.5f} {rl['reliability']:12.4f} "
              f"{rl['r_half']:8.4f} {rl['r_spearman_brown']:7.4f}")
    rel = reliability(rows)
    print(f"  {'pooled':>7s} {rel['n']:7d} {rel['sd_obs']:12.4f} "
          f"{rel['v_obs']:13.5f} {rel['v_reliable']:13.5f} "
          f"{rel['v_noise']:11.5f} {rel['reliability']:12.4f} "
          f"{rel['r_half']:8.4f} {rel['r_spearman_brown']:7.4f}")
    print("\n  reliability is the reliable variance as a fraction of the "
          "observed variance, and")
    print("  it is the largest R squared any model of these pair means could "
          "reach.")

    # --------------------------------------------------- the primary comparison
    STRATA = [("tree path length at most 2", lambda r: r["path"] <= 2),
              ("tree path length over 2", lambda r: r["path"] > 2),
              ("all pairs", lambda r: True)]
    print("\n\nPRIMARY COMPARISON: TREE PATH LENGTH AGAINST TOKEN DISTANCE")
    print("  The registered question. Variance in the SIGNED interaction "
          "explained by the")
    print("  dependency path length between the two words, against the "
          "variance explained by")
    print("  their token distance. Each is a saturated categorical block "
          "fitted on its own, so")
    print("  neither is given the other's columns, and each is divided by the "
          "split-half")
    print("  ceiling because neither can explain the draw noise. A pair the "
          "parse leaves")
    print("  disconnected counts as a long tree path, not as a missing one.")
    for label, sel in STRATA:
        print(f"\n  {label}")
        print(f"  {'set':>9s} {'pairs':>7s} {'ceiling':>8s} "
              f"{'R2 distance':>12s} {'of reliable':>12s} {'R2 tree':>9s} "
              f"{'of reliable':>12s} {'tree - distance':>16s} "
              f"{'of reliable':>12s}")
        for d in docs:
            sub = [r for r in by_doc[d] if sel(r)]
            v = var_contest(sub, with_doc=False)
            if v is None:
                print(f"  {d:9d} {len(sub):7d}   too few pairs")
                continue
            print(f"  {d:9d} {v['n']:7d} {v['ceiling']:8.4f} "
                  f"{v['distance']:12.4f} {v['distance_of_reliable']:12.4f} "
                  f"{v['tree']:9.4f} {v['tree_of_reliable']:12.4f} "
                  f"{v['diff']:16.4f} {v['diff_of_reliable']:12.4f}")
        v = var_contest([r for r in rows if sel(r)], with_doc=True)
        if v is not None:
            print(f"  {'pooled':>9s} {v['n']:7d} {v['ceiling']:8.4f} "
                  f"{v['distance']:12.4f} {v['distance_of_reliable']:12.4f} "
                  f"{v['tree']:9.4f} {v['tree_of_reliable']:12.4f} "
                  f"{v['diff']:16.4f} {v['diff_of_reliable']:12.4f}")
            print(f"  {'':9s} {v['cols']['distance']} distance columns, "
                  f"{v['cols']['tree']} tree columns, document dummies in the "
                  f"pooled row only")

    print("\n  PERMUTATION NULL ON THE DIFFERENCE, AS A FRACTION OF RELIABLE "
          "VARIANCE")
    print(f"  Path length, arc status and dep label are shuffled together "
          f"among the pairs that")
    print(f"  share a document and a token distance, at token distance "
          f"{NULL_MIN_DISTANCE} and beyond. The stratum")
    print(f"  is re-selected from the shuffled path each time, so a subset "
          f"defined by path length")
    print(f"  is tested against the same procedure rather than against a "
          f"fixed set of pairs.")
    live = [r for r in rows if r["distance"] >= NULL_MIN_DISTANCE]
    print(f"  {len(live)} of {len(rows)} pairs qualify.")
    print(f"\n  {'stratum':30s} {'observed':>9s} {'null mean':>10s} "
          f"{'null sd':>8s} {'excess':>8s} {'z':>7s} {'one-sided p':>12s}")
    for label, sel in STRATA:
        q = permute_contest(live, sel, args.r2_permutations)
        print(f"  {label:30s} {q['observed']:9.4f} {q['null_mean']:10.4f} "
              f"{q['null_sd']:8.4f} {q['excess']:+8.4f} {q['z']:+7.2f} "
              f"{q['p_upper']:12.4f}")
    print("\n  A positive excess means the tree beats token distance by more "
          "than a tree drawn at")
    print("  random from the pairs at the same distance would.")

    # ------------------------------------------------- variance accounted for
    print("\n\nWHAT EXPLAINS THE INTERACTION: LINEAR DISTANCE, THEN THE TREE")
    print("  Nested least squares on the pair means. Token distance enters "
          "SATURATED, one column")
    print(f"  per exact distance up to {DIST_CAP} and one for everything "
          f"beyond, so the linear model is")
    print("  given every chance before the tree is asked to add anything. The "
          "tree block is the")
    print(f"  dependency path length, one column per length up to {PATH_CAP}, "
          f"one beyond and one for")
    print("  a pair the parse leaves disconnected. The second variant adds the "
          "dep type of the")
    print("  arc and the ordered POS pair.")
    print(f"\n  {'doc':>7s} {'pairs':>7s} {'ceiling':>8s} "
          f"{'R2 distance':>12s} {'of reliable':>12s} "
          f"{'+tree':>8s} {'of reliable':>12s} "
          f"{'+tree+labels':>13s} {'of reliable':>12s}")
    va_doc = {}
    for d in docs:
        va = variance_accounted(by_doc[d], with_doc=False)
        va_doc[d] = va
        print(f"  {d:7d} {va['n']:7d} {va['ceiling']:8.4f} "
              f"{va['r2_distance']:12.4f} "
              f"{va['r2_distance_of_reliable']:12.4f} "
              f"{va['d_tree']:8.4f} {va['d_tree_of_reliable']:12.4f} "
              f"{va['d_tree_labels']:13.4f} "
              f"{va['d_tree_labels_of_reliable']:12.4f}")
    va = variance_accounted(rows, with_doc=True)
    print(f"  {'pooled':>7s} {va['n']:7d} {va['ceiling']:8.4f} "
          f"{va['r2_distance']:12.4f} {va['r2_distance_of_reliable']:12.4f} "
          f"{va['d_tree']:8.4f} {va['d_tree_of_reliable']:12.4f} "
          f"{va['d_tree_labels']:13.4f} "
          f"{va['d_tree_labels_of_reliable']:12.4f}")
    print(f"\n  pooled model: document dummies, then {va['cols']['distance']} "
          f"distance columns, then {va['cols']['tree']} tree columns,")
    print(f"  then {va['cols']['dep']} dep columns and {va['cols']['pos']} POS "
          f"pair columns.")
    print(f"  R squared of the document dummies alone {va['r2']['doc']:.4f}; "
          f"with distance {va['r2']['base']:.4f};")
    print(f"  with the tree {va['r2']['tree']:.4f}; with the labels too "
          f"{va['r2']['tree+labels']:.4f}.")
    print(f"  The labels add {va['d_labels']:.4f} over the tree alone, "
          f"{va['d_labels_of_reliable']:.4f} of the reliable variance.")
    print("\n  'of reliable' divides the R squared by the ceiling above, so it "
          "is the share of the")
    print("  variance that could be explained at all, rather than of the "
          "variance including noise.")

    # ------------------------------ permutation null on the R squared increment
    print("\n\nPERMUTATION NULL ON THE TREE INCREMENT")
    live = [r for r in rows if r["distance"] >= NULL_MIN_DISTANCE]
    print(f"  The statistic is the increase in R squared from adding the tree "
          f"block to the")
    print(f"  saturated distance model. Path length, arc status and dep label "
          f"are shuffled")
    print(f"  together among the pairs that share a document and a token "
          f"distance, so the")
    print(f"  permuted tree is one the linear model could not tell from the "
          f"real one.")
    print(f"  Restricted to token distance {NULL_MIN_DISTANCE} and beyond: "
          f"{len(live)} of {len(rows)} pairs.")
    tp = permute_tree(live, args.r2_permutations, with_doc=True)
    print(f"\n  strata {tp['strata']}, informative {tp['informative_strata']} "
          f"({tp['pairs_in_informative_strata']} pairs), "
          f"{tp['n']} permutations")
    print(f"\n  observed increment           {tp['observed']:.5f}")
    print(f"  null mean                    {tp['null_mean']:.5f}")
    print(f"  null sd                      {tp['null_sd']:.5f}")
    print(f"  EXCESS OVER THE NULL         {tp['excess']:+.5f}")
    rl_live = reliability(live)
    if rl_live and rl_live["reliability"] > 0:
        print(f"  excess as a share of the reliable variance   "
              f"{tp['excess'] / rl_live['reliability']:+.5f}")
    print(f"  z                            {tp['z']:+.2f}")
    print(f"  one-sided p                  {tp['p_upper']:.4f}")
    print("\n  A permutation always explains a little by chance, which is why "
          "the null mean is")
    print("  above zero. The excess is the part of the tree's contribution "
          "that survives holding")
    print("  document and token distance fixed.")

    # ------------------------------------------------- per cell, per category
    print("\n\nMEAN ABSOLUTE INTERACTION PER CELL, BY CATEGORY")
    print("  The per-cell mean is the quantity the categories can be compared "
          "on: a mass share")
    print("  mostly reports how many cells a category has.")
    print(f"\n  {'set':10s} {'cells':>7s} {'share':>7s} "
          f"{'mean abs':>9s} {'se':>7s} {'mean':>9s} {'se':>7s} "
          f"{'median abs':>11s} {'over floor':>11s}")
    for c in CATEGORIES + ["all"]:
        g = rows if c == "all" else [r for r in rows if r["category"] == c]
        if not g:
            continue
        v = np.array([r["inter"] for r in g])
        am, ase, n = describe(np.abs(v), [r["doc_id"] for r in g])
        m, se, _ = describe(v, [r["doc_id"] for r in g])
        print(f"  {c:10s} {n:7d} {n / len(rows):7.4f} {am:9.4f} {ase:7.4f} "
              f"{m:+9.4f} {se:7.4f} {np.median(np.abs(v)):11.4f} "
              f"{(np.abs(v) > FLOOR).mean():11.3f}")
    print(f"\n  Standard errors are clustered on document over {len(docs)} "
          f"clusters, which is few.")

    print("\n  per document, mean absolute interaction per cell")
    print(f"  {'doc':>7s} {'words':>6s} {'cells':>7s} "
          + " ".join(f"{c:>10s}" for c in CATEGORIES)
          + f" {'arc / other':>12s}")
    per_doc = {}
    for d in docs:
        m = masses(by_doc[d])
        per_doc[d] = m
        n_words = len({s for r in by_doc[d] for s in (r["lo"], r["hi"])})
        ratio = (m["cell_arc"] / m["cell_other"]
                 if m["cell_other"] else float("nan"))
        print(f"  {d:7d} {n_words:6d} {m['pairs']:7d} "
              + " ".join(f"{m[f'cell_{c}']:10.4f}" for c in CATEGORIES)
              + f" {ratio:12.3f}")
    pooled = masses(rows)
    print(f"  {'pooled':>7s} {'':6s} {pooled['pairs']:7d} "
          + " ".join(f"{pooled[f'cell_{c}']:10.4f}" for c in CATEGORIES)
          + f" {pooled['cell_arc'] / pooled['cell_other']:12.3f}")

    # ---------------------------------------------------------- mass shares
    print("\n\nSHARE OF TOTAL ABSOLUTE INTERACTION MASS")
    print("  Fractions of the total absolute interaction over all C(n, 2) "
          "pairs of the document,")
    print("  beside the share of the cells each category holds. The two are "
          "equal when a")
    print("  category carries nothing the rest of the matrix does not.")
    print(f"\n  {'doc':>7s} {'total |i|':>10s} "
          + " ".join(f"{'mass ' + c:>13s} {'cells ' + c:>13s}"
                     for c in CATEGORIES))
    for d in docs:
        m = per_doc[d]
        print(f"  {d:7d} {m['total']:10.2f} "
              + " ".join(f"{m['mass_' + c]:13.4f} {m['share_' + c]:13.4f}"
                         for c in CATEGORIES))
    print(f"  {'pooled':>7s} {pooled['total']:10.2f} "
          + " ".join(f"{pooled['mass_' + c]:13.4f} "
                     f"{pooled['share_' + c]:13.4f}" for c in CATEGORIES))

    # ------------------------------------------------- the arc-cell null
    print("\n\nPERMUTATION NULL ON THE PER-CELL ARC INTERACTION")
    print("  The same shuffle, with a simpler statistic: the mean absolute "
          "interaction of an arc")
    print(f"  cell. Restricted to token distance {NULL_MIN_DISTANCE} and "
          f"beyond, {len(live)} of {len(rows)} pairs.")
    perm = permutation_null(live, arc_cell_mean, n=args.permutations)
    print(f"\n  strata {perm['strata']}, informative "
          f"{perm['informative_strata']} "
          f"({perm['pairs_in_informative_strata']} pairs, "
          f"{perm['arcs_in_informative_strata']} of them arcs)")
    print(f"\n  observed arc cell mean       {perm['observed']:.4f} FVE points")
    print(f"  null mean                    {perm['null_mean']:.4f}")
    print(f"  null sd                      {perm['null_sd']:.4f}")
    print(f"  EXCESS OVER THE NULL         {perm['excess']:+.4f} points"
          f"  ({100 * perm['excess'] / max(perm['null_mean'], 1e-12):+.1f}%)")
    print(f"  z                            {perm['z']:+.2f}")
    print(f"  one-sided p (arc higher)     {perm['p_upper']:.4f}")
    print(f"  two-sided p                  {perm['p_two']:.4f}")

    print(f"\n  {'doc':>7s} {'pairs':>7s} {'arcs':>5s} {'observed':>9s} "
          f"{'null mean':>10s} {'null sd':>8s} {'excess':>8s} {'z':>7s} "
          f"{'one-sided p':>12s}")
    for d in docs:
        g = [r for r in by_doc[d] if r["distance"] >= NULL_MIN_DISTANCE]
        na = sum(1 for r in g if r["arc"])
        if na < 2:
            print(f"  {d:7d} {len(g):7d} {na:5d}  too few arcs at this "
                  f"distance for a null")
            continue
        q = permutation_null(g, arc_cell_mean, n=args.permutations)
        print(f"  {d:7d} {len(g):7d} {na:5d} {q['observed']:9.4f} "
              f"{q['null_mean']:10.4f} {q['null_sd']:8.4f} "
              f"{q['excess']:+8.4f} {q['z']:+7.2f} {q['p_upper']:12.4f}")

    # ---------------------------------------------------------- distance
    print("\n\nMEAN ABSOLUTE INTERACTION AGAINST TOKEN DISTANCE")
    print(f"  {'distance':12s} {'pairs':>7s} {'mean abs':>9s} {'se':>7s} "
          f"{'mean':>9s} {'arcs':>6s} {'arc mean abs':>13s} "
          f"{'rest mean abs':>14s} {'excess':>8s}")
    for lo, hi in DIST_BINS:
        g = [r for r in rows if lo <= r["distance"] <= hi]
        if not g:
            continue
        a = [abs(r["inter"]) for r in g if r["arc"]]
        b = [abs(r["inter"]) for r in g if not r["arc"]]
        m, se, n = describe([abs(r["inter"]) for r in g],
                            [r["doc_id"] for r in g])
        label = f"{lo}" if lo == hi else (f"{lo}+" if hi > 1000
                                          else f"{lo} to {hi}")
        am = np.mean(a) if a else float("nan")
        bm = np.mean(b) if b else float("nan")
        print(f"  {label:12s} {n:7d} {m:9.4f} {se:7.4f} "
              f"{np.mean([r['inter'] for r in g]):+9.4f} {len(a):6d} "
              f"{am:13.4f} {bm:14.4f} {am - bm:+8.4f}")

    print("\n\nMEAN ABSOLUTE INTERACTION AGAINST TREE PATH LENGTH")
    print(f"  {'path':12s} {'pairs':>7s} {'mean abs':>9s} {'se':>7s} "
          f"{'mean':>9s} {'median token distance':>22s}")
    seen_paths = sorted({min(r["path"], DISCONNECTED) for r in rows})
    for p in seen_paths:
        g = [r for r in rows if min(r["path"], DISCONNECTED) == p]
        if len(g) < 4:
            continue
        m, se, n = describe([abs(r["inter"]) for r in g],
                            [r["doc_id"] for r in g])
        label = "disconnected" if p >= DISCONNECTED else str(p)
        print(f"  {label:12s} {n:7d} {m:9.4f} {se:7.4f} "
              f"{np.mean([r['inter'] for r in g]):+9.4f} "
              f"{np.median([r['distance'] for r in g]):22.1f}")

    # ------------------------------------------------------- consistency, 05
    print("\n\nCONSISTENCY CHECK: MEAN INTERACTION BY ARC DEP TYPE")
    print("  05 measured a sample of arcs of nine dep types with the two words "
          "at least two")
    print("  tokens apart, against matched controls. These are the same "
          "quantity on the same")
    print("  scale over every arc of these documents, so the two are "
          "comparable type by type.")
    print(f"\n  {'dep':10s} {'in 05':>6s} {'arcs':>6s} {'mean':>9s} "
          f"{'se':>7s} {'mean abs':>9s} {'median dist':>12s}")
    arcs = [r for r in rows if r["category"] == "arc"]
    by_dep = defaultdict(list)
    for r in arcs:
        by_dep[r["dep"] or "?"].append(r)
    for dep in sorted(by_dep, key=lambda d: -len(by_dep[d])):
        g = by_dep[dep]
        if len(g) < MIN_CELL and dep not in DEPS_05:
            continue
        m, se, n = describe([r["inter"] for r in g],
                            [r["doc_id"] for r in g])
        print(f"  {dep:10s} {'yes' if dep in DEPS_05 else 'no':>6s} {n:6d} "
              f"{m:+9.4f} {se:7.4f} "
              f"{np.mean([abs(r['inter']) for r in g]):9.4f} "
              f"{np.median([r['distance'] for r in g]):12.1f}")
    far = [r for r in arcs if r["distance"] >= 2 and r["dep"] in DEPS_05]
    if far:
        m, se, n = describe([r["inter"] for r in far],
                            [r["doc_id"] for r in far])
        print(f"\n  restricted to 05's nine types at distance two or more, "
              f"which is 05's own arc set:")
        print(f"    {n} arcs, mean {m:+.4f} +- {se:.4f} points")
    print(f"  {sum(1 for r in arcs if r['dep'] not in DEPS_05)} of "
          f"{len(arcs)} arcs carry a dep type 05 did not sample.")

    # ----------------------------------------------------------- separability
    print("\n\nWHAT THE DRAWS SEPARATE FROM ZERO")
    print(f"  {'doc':>7s} {'pairs':>7s} {'|mean| >= 2 se':>15s} "
          f"{'fraction':>9s} {'median se':>10s}")
    for d in docs:
        g = by_doc[d]
        k = [r for r in g if abs(r["inter"]) >= 2 * r["se"]]
        print(f"  {d:7d} {len(g):7d} {len(k):15d} {len(k) / len(g):9.4f} "
              f"{np.median([r['se'] for r in g]):10.4f}")
    keep = [r for r in rows if abs(r["inter"]) >= 2 * r["se"]]
    print(f"  {'pooled':>7s} {len(rows):7d} {len(keep):15d} "
          f"{len(keep) / len(rows):9.4f} "
          f"{np.median([r['se'] for r in rows]):10.4f}")
    if keep:
        mk = masses(keep)
        print("\n  over the cells that survive the two standard error test, "
              "the per-cell means are")
        print("    " + ", ".join(f"{c} {mk[f'cell_{c}']:.4f}"
                                 for c in CATEGORIES)
              + f", and arcs are {mk['share_arc']:.4f} of them.")

    # ---------------------------------------------------------------- figures
    print("\n\nFIGURES")
    for d in docs:
        path = args.out / f"interaction_matrix_doc{d}.png"
        info = fig_matrix(d, by_doc[d], meta, path)
        print(f"  {path.name}: {info['n']} words, {info['cells']} cells, "
              f"{info['kept']} outside two standard errors, {info['arcs']} "
              f"head arcs drawn above the axis, colour limit "
              f"+-{info['vmax']:.3f} FVE points")
    print("  Each figure is one document's interaction matrix in reading "
          "order, symmetric by")
    print("  construction, with the spaCy head arcs over the same words drawn "
          "above it.")

    sys.stdout.f.write("```\n")
    sys.stdout.flush()
    sys.stdout = sys.__stdout__
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
