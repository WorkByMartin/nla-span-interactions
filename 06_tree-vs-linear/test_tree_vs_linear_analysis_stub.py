#!/usr/bin/env python3
"""Drive tree_vs_linear_analysis.py over a synthetic store with a known answer.

    python test_tree_vs_linear_analysis_stub.py

Builds a throwaway database shaped like the real one: documents, spans with
token indices, a spaCy parse, the pair categories the run records, a corpus-swap
run of singles at eight draws and joint edits over EVERY pair, with the
interaction planted. The plant carries a decay with token distance on every
pair and, in the positive case, an extra amount on the pairs that sit on a
dependency arc.

Two scenarios are run. In the first the arc effect is real and the analysis must
recover it and the permutation null must reject. In the second there is no arc
effect, only the same distance decay, and the null must NOT reject: without that
the first result would be evidence of nothing, since arcs are short and short
pairs interact more.
"""
import json
import math
import random
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "db"))
import db as DB  # noqa: E402

SWAP = "corpus-swap/pos+len"
SCHEME = "spacy-en_core_web_sm-3.8.0"
PAIR_SCHEME = "tree-vs-linear/all-pairs"
RAWVAR = 0.45467177831665084
DEPS = ["dobj", "nsubj", "conj", "appos", "nmod", "advcl", "ccomp", "attr",
        "poss"]
POSES = ["NOUN", "VERB", "ADJ", "PROPN", "ADP", "DET"]

DRAWS = 8
N_DOCS = 6
N_SPANS = 20
PLANTED_ARC = 0.55      # extra FVE points of interaction on an arc
DECAY_AT_ONE = 0.45     # interaction of a neighbouring pair, before the decay
DECAY_LENGTH = 5.0      # tokens
SD_DRAW = 0.22          # per-draw noise on the interaction


def planted(distance, arc, arc_effect):
    return (arc_effect if arc else 0.0) + DECAY_AT_ONE * math.exp(
        -(distance - 1) / DECAY_LENGTH)


def build(tmp, arc_effect, seed):
    """A store whose interaction is exactly `planted` plus per-draw noise."""
    rng = random.Random(seed)
    path = tmp / "ablation.sqlite"
    conn = DB.connect(path)
    DB.migrate(conn, apply=True)

    docs = [1000 + 7 * i for i in range(N_DOCS)]
    words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta",
             "theta"]
    span_ids, pos_of, base = {}, {}, {}
    heads = {}
    with DB.transaction(conn):
        for d in docs:
            toks = [rng.choice(words) for _ in range(N_SPANS + 10)]
            text = " ".join(toks)
            DB.upsert_doc(conn, d, text, "test")
            keys, off = [], 0
            for w in toks[:N_SPANS]:
                i = text.index(w, off)
                keys.append((d, i, i + len(w)))
                off = i + len(w)
            got = DB.get_or_create_spans(conn, keys)
            span_ids[d] = [got[k] for k in keys]
            for j, s in enumerate(span_ids[d]):
                pos_of[s] = rng.choice(POSES)
                DB.set_label(conn, s, SCHEME, "pos", pos_of[s])
                DB.set_label(conn, s, SCHEME, "tok_i", j)

    # a parse: each token may take one head, at a distance drawn from a mix of
    # short and medium, so the null has strata holding both labels
    with DB.transaction(conn):
        rows = []
        for d in docs:
            ids = span_ids[d]
            for j in range(N_SPANS):
                if rng.random() > 0.55:
                    continue
                step = rng.choice([1, 1, 2, 3, 3, 4, 5, 6, 7, 9])
                k = j + step if rng.random() < 0.5 else j - step
                if not (0 <= k < N_SPANS) or k == j:
                    continue
                if ids[j] in heads:
                    continue
                heads[ids[j]] = ids[k]
                DB.set_label(conn, ids[j], SCHEME, "dep", rng.choice(DEPS))
                rows.append((SCHEME, ids[j], ids[k], "head"))
        DB.add_relations(conn, rows)

    run = DB.new_run(conn, script="06_tree-vs-linear/tree_vs_linear.py",
                     notes="synthetic full matrix. GPU: none.",
                     config=json.dumps({"seed": seed, "draws": DRAWS}))

    def category(u, v, du, dv):
        if heads.get(u) == v or heads.get(v) == u:
            dep = (conn.execute("SELECT value FROM labels WHERE span_id=? AND "
                                "scheme=? AND key='dep'",
                                (u if heads.get(u) == v else v, SCHEME))
                   .fetchone())
            return "arc", (dep[0] if dep else None)
        return ("adjacent" if abs(du - dv) == 1 else "other"), None

    e_single, rels = {}, []
    for d in docs:
        ids = span_ids[d]
        base[d] = 0.80 + 0.02 * rng.random()
        with DB.transaction(conn):
            v = DB.new_variant(conn, d, run)
            DB.record(conn, v, run, "fve", base[d])
            DB.record(conn, v, run, "mse", (1 - base[d]) * RAWVAR)
            DB.record(conn, v, run, "seq_len", 300)
            DB.record(conn, v, run, "dtok", 0)
        with DB.transaction(conn):
            for s in ids:
                mu = abs(rng.gauss(1.0, 0.5))
                for k in range(DRAWS):
                    e = mu + rng.gauss(0, 0.30)
                    e_single[(s, k)] = e
                    v = DB.new_variant(conn, d, run, substitutions=[
                        {"span_id": s, "substitute": rng.choice(words),
                         "source": SWAP, "depth": 1, "draw_idx": k,
                         "prob": 0.01}])
                    DB.record(conn, v, run, "fve", base[d] - e / 100)
                    DB.record(conn, v, run, "mse", 0.1)
                    DB.record(conn, v, run, "seq_len", 300)
                    DB.record(conn, v, run, "dtok", 0)
        with DB.transaction(conn):
            for i in range(N_SPANS):
                for j in range(i + 1, N_SPANS):
                    a, b = ids[i], ids[j]
                    cat, dep = category(a, b, i, j)
                    rels.append((PAIR_SCHEME, a, b,
                                 f"arc:{dep}" if cat == "arc" else cat))
                    want = planted(j - i, cat == "arc", arc_effect)
                    for k in range(DRAWS):
                        both = (e_single[(a, k)] + e_single[(b, k)] - want
                                + rng.gauss(0, SD_DRAW))
                        v = DB.new_variant(conn, d, run, substitutions=[
                            {"span_id": a, "substitute": rng.choice(words),
                             "source": SWAP, "depth": 2, "draw_idx": k,
                             "prob": 0.01},
                            {"span_id": b, "substitute": rng.choice(words),
                             "source": SWAP, "depth": 2, "draw_idx": k,
                             "prob": 0.01}])
                        DB.record(conn, v, run, "fve", base[d] - both / 100)
                        DB.record(conn, v, run, "mse", 0.1)
                        DB.record(conn, v, run, "seq_len", 300)
                        DB.record(conn, v, run, "dtok", 0)
    with DB.transaction(conn):
        DB.add_relations(conn, rels)
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.close()
    return path, run, docs


def field(md, prefix, col, after=None):
    live = after is None
    for line in md.splitlines():
        if after is not None and after in line:
            live = True
            continue
        if live and line.strip().startswith(prefix):
            try:
                return float(line.split()[col])
            except (ValueError, IndexError):
                continue
    return None


def run_case(name, arc_effect, seed, expect_reject):
    tmp = Path(tempfile.mkdtemp(prefix=f"matrixstub_{name}_"))
    path, run, docs = build(tmp, arc_effect, seed)
    out = tmp / "results"
    out.mkdir()
    r = subprocess.run([sys.executable, str(HERE / "tree_vs_linear_analysis.py"),
                        "--db", str(path), "--run", str(run),
                        "--out", str(out), "--permutations", "500"],
                       capture_output=True, text=True)
    print(f"\n=== {name}: planted arc effect {arc_effect}")
    if r.returncode:
        print(r.stdout[-3000:])
        print(r.stderr[-4000:])
        return [f"{name}: the analysis exited {r.returncode}"]

    fails = []
    md = (out / "statistics.md").read_text()
    for want in ["SETUP", "SIGN CONVENTION", "CATEGORIES",
                 "SPLIT-HALF RELIABILITY OF THE INTERACTION",
                 "PRIMARY COMPARISON: TREE PATH LENGTH AGAINST TOKEN DISTANCE",
                 "PERMUTATION NULL ON THE DIFFERENCE",
                 "WHAT EXPLAINS THE INTERACTION",
                 "PERMUTATION NULL ON THE TREE INCREMENT",
                 "MEAN ABSOLUTE INTERACTION PER CELL, BY CATEGORY",
                 "SHARE OF TOTAL ABSOLUTE INTERACTION MASS",
                 "PERMUTATION NULL ON THE PER-CELL ARC INTERACTION",
                 "MEAN ABSOLUTE INTERACTION AGAINST TOKEN DISTANCE",
                 "MEAN ABSOLUTE INTERACTION AGAINST TREE PATH LENGTH",
                 "CONSISTENCY CHECK: MEAN INTERACTION BY ARC DEP TYPE",
                 "WHAT THE DRAWS SEPARATE FROM ZERO", "FIGURES"]:
        if want not in md:
            fails.append(f"{name}: section missing: {want}")
    for d in docs:
        if not (out / f"interaction_matrix_doc{d}.png").exists():
            fails.append(f"{name}: figure missing for doc {d}")
    if "—" in md or "–" in md:
        fails.append(f"{name}: a dash slipped into the report text")

    ARC_NULL = "PERMUTATION NULL ON THE PER-CELL ARC INTERACTION"
    TREE_NULL = "PERMUTATION NULL ON THE TREE INCREMENT"
    CELLS = "MEAN ABSOLUTE INTERACTION PER CELL"
    arc = field(md, "arc ", 3, after=CELLS)
    other = field(md, "other ", 3, after=CELLS)
    excess = field(md, "EXCESS OVER THE NULL", 4, after=ARC_NULL)
    obs = field(md, "observed arc cell mean", 4, after=ARC_NULL)
    rel = field(md, "pooled ", 6, after="SPLIT-HALF RELIABILITY")
    d_tree = field(md, "pooled ", 5, after="WHAT EXPLAINS THE INTERACTION")
    t_excess = field(md, "EXCESS OVER THE NULL", 4, after=TREE_NULL)
    t_obs = field(md, "observed increment", 2, after=TREE_NULL)
    p = t_p = None
    live = False
    for line in md.splitlines():
        if TREE_NULL in line:
            live = True
        if ARC_NULL in line:
            live = False
        if "one-sided p (arc higher)" in line:
            p = float(line.split()[-1])
        if live and line.strip().startswith("one-sided p "):
            t_p = float(line.split()[-1])
    NEAR = "tree path length at most 2"
    c_obs = field(md, NEAR, 6, after="PERMUTATION NULL ON THE DIFFERENCE")
    c_exc = field(md, NEAR, 9, after="PERMUTATION NULL ON THE DIFFERENCE")
    c_p = field(md, NEAR, 11, after="PERMUTATION NULL ON THE DIFFERENCE")
    print(f"  primary comparison, path at most 2: tree minus distance "
          f"{c_obs} of reliable variance, excess over the null {c_exc}, "
          f"one-sided p {c_p}")
    print(f"  per-cell mean abs: arc {arc}, other {other}")
    print(f"  observed arc cell mean {obs}, excess over the null {excess}, "
          f"one-sided p {p}")
    print(f"  split-half reliability {rel}")
    print(f"  tree increment in R2 {d_tree}, observed {t_obs}, excess over "
          f"the null {t_excess}, one-sided p {t_p}")
    if None in (arc, other, excess, p, rel, d_tree, t_excess, t_p, c_obs,
                c_exc, c_p):
        return fails + [f"{name}: the report did not carry the numbers"]

    if not 0.0 < rel <= 1.0:
        fails.append(f"{name}: split-half reliability {rel} is not a fraction")
    if rel < 0.4:
        fails.append(f"{name}: reliability {rel} is far below what the planted "
                     f"signal and noise imply")

    if expect_reject:
        if abs(excess - arc_effect) > 0.12:
            fails.append(f"{name}: arc-cell excess {excess} is not near the "
                         f"planted {arc_effect}")
        if p > 0.01:
            fails.append(f"{name}: the arc-cell null did not reject, p {p}")
        if arc <= other:
            fails.append(f"{name}: the arc cells are not larger than the rest")
        if t_excess <= 0 or t_p > 0.01:
            fails.append(f"{name}: the tree increment null did not reject, "
                         f"excess {t_excess}, p {t_p}")
        if c_exc <= 0 or c_p > 0.05:
            fails.append(f"{name}: the primary comparison null did not reject "
                         f"on pairs close in the tree, excess {c_exc}, "
                         f"p {c_p}")
    else:
        if abs(excess) > 0.10:
            fails.append(f"{name}: arc-cell excess {excess} on a store with no "
                         f"planted arc effect")
        if p < 0.02:
            fails.append(f"{name}: the arc-cell null rejected on a store whose "
                         f"only structure is distance decay, p {p}")
        if arc <= other:
            fails.append(f"{name}: arcs should still look larger before the "
                         f"null, because they are short")
        if t_p < 0.02:
            fails.append(f"{name}: the tree increment null rejected on a store "
                         f"whose only structure is distance decay, p {t_p}")
        if c_p < 0.02:
            fails.append(f"{name}: the primary comparison null rejected on a "
                         f"store whose only structure is distance decay, "
                         f"p {c_p}")
    print(f"  scratch database left at {tmp}")
    return fails


def main():
    fails = run_case("planted", PLANTED_ARC, 0, True)
    fails += run_case("distance-only", 0.0, 1, False)
    print("\nFAILED:" if fails else
          "\nthe analysis recovers a planted on-arc interaction, and its "
          "permutation null\nrejects it while not rejecting distance decay on "
          "its own")
    for f in fails:
        print(" ", f)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
