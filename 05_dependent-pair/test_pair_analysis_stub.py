#!/usr/bin/env python3
"""Drive pair_analysis.py over a synthetic store, so the report is known to run.

    python test_pair_analysis_stub.py

Builds a throwaway database with the same shape as the real one: documents,
spans, a corpus-swap run of singles at eight draws and joint edits over arc and
control pairs, with an interaction planted on the arcs and none on the controls.
The pair table goes into the run's config row and into `relations`, which is
where the analysis reads it from. Then runs the analysis on it and checks that
the planted gap comes back and that every section and figure was produced.
"""
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
PAIR_SCHEME = "pair-ablation/arc+control"
SCHEME = "spacy-en_core_web_sm-3.8.0"
RAWVAR = 0.45467177831665084
DEPS = ["dobj", "nsubj", "conj", "appos", "nmod", "advcl", "ccomp", "attr",
        "poss"]
POSES = ["NOUN", "VERB", "ADJ", "PROPN", "ADP", "DET"]
DRAWS = 8

PLANTED_ARC = 0.60      # FVE points of sub-additivity on an arc
PLANTED_CTRL = 0.10     # and on a control
SD_DRAW = 0.30          # per-draw noise on each of the three effects


def main():
    rng = random.Random(0)
    tmp = Path(tempfile.mkdtemp(prefix="pairstub_"))
    path = tmp / "ablation.sqlite"
    conn = DB.connect(path)
    DB.migrate(conn, apply=True)

    docs = [4635, 2159, 2957, 376, 1934, 8801, 512, 77]
    words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta"]
    span_ids, base, pos_of = {}, {}, {}
    with DB.transaction(conn):
        for d in docs:
            text = " ".join(rng.choice(words) for _ in range(120))
            DB.upsert_doc(conn, d, text, "test")
            keys, off = [], 0
            for w in text.split(" ")[:100]:
                i = text.index(w, off)
                keys.append((d, i, i + len(w)))
                off = i + len(w)
            got = DB.get_or_create_spans(conn, keys)
            span_ids[d] = [got[k] for k in keys]
            for j, s in enumerate(span_ids[d]):
                p = rng.choice(POSES)
                pos_of[s] = p
                DB.set_label(conn, s, SCHEME, "pos", p)
                DB.set_label(conn, s, SCHEME, "tok_i", j)

    # the pair table: one arc and one control per (document, dep type), over
    # spans that no other pair uses, so the sharing story stays simple here
    pairs, pid = [], 0
    e_single = {}
    for d in docs:
        pool = list(span_ids[d])
        rng.shuffle(pool)
        cur = 0
        for dep in DEPS:
            if cur + 4 > len(pool):
                break
            a, b, u, v = pool[cur:cur + 4]
            cur += 4
            dist = 2 + rng.randrange(20)
            pid += 1
            arc_id = pid
            pairs.append({"pair_id": pid, "kind": "arc", "dep": dep,
                          "doc_id": d, "span_a": a, "span_b": b,
                          "span_first": a, "span_second": b,
                          "distance": dist, "pos_first": pos_of[a],
                          "pos_second": pos_of[b], "pos_a": pos_of[a],
                          "pos_b": pos_of[b], "text_a": "x", "text_b": "y",
                          "dep_first": True, "match_of": None,
                          "match_quality": "arc", "planned": True})
            pid += 1
            pairs.append({"pair_id": pid, "kind": "control", "dep": dep,
                          "doc_id": d, "span_a": u, "span_b": v,
                          "span_first": u, "span_second": v,
                          "distance": dist, "pos_first": pos_of[a],
                          "pos_second": pos_of[b], "pos_a": pos_of[a],
                          "pos_b": pos_of[b], "text_a": "x", "text_b": "y",
                          "dep_first": True, "match_of": arc_id,
                          "match_quality": "exact", "planned": True})

    # the store carries the table exactly as the real run leaves it: every
    # column in the run's config, every planned pair as an edge in relations
    COLS = ["pair_id", "kind", "dep", "doc_id", "span_a", "span_b", "distance",
            "pos_a", "pos_b", "match_of", "match_quality", "planned"]
    run = DB.new_run(
        conn, script="05_dependent-pair/pair_ablation.py", notes="synthetic",
        config={"args": {"seed": 0, "draws": DRAWS, "per_type": 150},
                "min_distance": 2, "arc_deps": DEPS,
                "pair_scheme": PAIR_SCHEME, "unmatched_arcs": [],
                "pair_columns": COLS,
                "pairs": [[p[c] for c in COLS] for p in pairs]})
    with DB.transaction(conn):
        DB.add_relations(conn, [
            (PAIR_SCHEME, p["span_a"], p["span_b"],
             f"arc:{p['dep']}" if p["kind"] == "arc" else "control")
            for p in pairs if p["planned"]])

    # baselines, then the singles for every span a pair uses, then the joints
    used = sorted({p[k] for p in pairs for k in ("span_a", "span_b")})
    doc_of = {s: d for d in docs for s in span_ids[d]}
    for d in docs:
        base[d] = 0.80 + 0.02 * rng.random()
        with DB.transaction(conn):
            v = DB.new_variant(conn, d, run)
            DB.record(conn, v, run, "fve", base[d])
            DB.record(conn, v, run, "mse", (1 - base[d]) * RAWVAR)
            DB.record(conn, v, run, "seq_len", 300)

    with DB.transaction(conn):
        for s in used:
            d = doc_of[s]
            mu = abs(rng.gauss(0.9, 0.5))
            for k in range(DRAWS):
                e = mu + rng.gauss(0, SD_DRAW)
                e_single[(s, k)] = e
                v = DB.new_variant(conn, d, run, substitutions=[
                    {"span_id": s, "substitute": rng.choice(words),
                     "source": SWAP, "depth": 1, "draw_idx": k,
                     "prob": 0.01}])
                DB.record(conn, v, run, "fve", base[d] - e / 100)
                DB.record(conn, v, run, "mse", 0.1)
                DB.record(conn, v, run, "seq_len", 300)

    with DB.transaction(conn):
        for p in pairs:
            d, a, b = p["doc_id"], p["span_a"], p["span_b"]
            plant = PLANTED_ARC if p["kind"] == "arc" else PLANTED_CTRL
            for k in range(DRAWS):
                both = (e_single[(a, k)] + e_single[(b, k)] - plant
                        + rng.gauss(0, SD_DRAW))
                v = DB.new_variant(conn, d, run, substitutions=[
                    {"span_id": a, "substitute": rng.choice(words),
                     "source": SWAP, "depth": 2, "draw_idx": k, "prob": 0.01},
                    {"span_id": b, "substitute": rng.choice(words),
                     "source": SWAP, "depth": 2, "draw_idx": k, "prob": 0.01}])
                DB.record(conn, v, run, "fve", base[d] - both / 100)
                DB.record(conn, v, run, "mse", 0.1)
                DB.record(conn, v, run, "seq_len", 300)
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.close()

    out = tmp / "results"
    out.mkdir()

    r = subprocess.run([sys.executable, str(HERE / "pair_analysis.py"),
                        "--db", str(path), "--run", str(run),
                        "--out", str(out), "--permutations", "500"],
                       capture_output=True, text=True)
    print(r.stdout[-7000:])
    if r.returncode:
        print(r.stderr[-4000:])
        print("FAILED: analysis exited", r.returncode)
        return 1

    fails = []
    md = (out / "statistics.md").read_text()
    for want in ["SETUP", "SIGN CONVENTION", "OVERALL, ARC AGAINST CONTROL",
                 "PER DEP TYPE", "REGRESSION, interaction ~ arc + distance",
                 "WITHIN-DOCUMENT PERMUTATION NULL",
                 "INTERACTION AGAINST THE SIZE OF THE SINGLES",
                 "INTERACTION AGAINST TOKEN DISTANCE",
                 "CONTROL MATCH QUALITY", "DRAW-LEVEL SPREAD",
                 "PER DOCUMENT", "FIGURES"]:
        if want not in md:
            fails.append(f"section missing: {want}")
    for f in ["pair_arc_vs_control.png", "pair_interaction_vs_distance.png"]:
        if not (out / f).exists():
            fails.append(f"figure missing: {f}")
    if "—" in md or "–" in md:
        fails.append("a dash slipped into the report text")

    # the planted gap should come back on the arc row and on the arc coefficient
    def field(prefix, col, after=None):
        live = after is None
        for line in md.splitlines():
            if after is not None and after in line:
                live = True
                continue
            if live and line.strip().startswith(prefix):
                try:
                    return float(line.split()[col])
                except ValueError:
                    continue
        return None

    arc = field("arc pairs", 3)
    ctrl = field("control pairs", 3)
    coef = field("arc ", 1, after="REGRESSION, interaction ~ arc")
    print(f"  recovered: arc mean {arc}, control mean {ctrl}, "
          f"regression arc coefficient {coef}")
    if arc is None or abs(arc - PLANTED_ARC) > 0.08:
        fails.append(f"arc mean {arc} is not near the planted {PLANTED_ARC}")
    if ctrl is None or abs(ctrl - PLANTED_CTRL) > 0.08:
        fails.append(f"control mean {ctrl} is not near {PLANTED_CTRL}")
    if coef is None or abs(coef - (PLANTED_ARC - PLANTED_CTRL)) > 0.10:
        fails.append(f"arc coefficient {coef} is not near "
                     f"{PLANTED_ARC - PLANTED_CTRL}")
    p = [l for l in md.splitlines() if "two-sided p" in l]
    print("  permutation:", p[0].strip() if p else "NOT FOUND")
    if p and float(p[0].split()[-1]) > 0.01:
        fails.append("the permutation test did not see the planted gap")

    print("\nFAILED:" if fails else "\nanalysis runs end to end on synthetic "
          "data and recovers what was planted")
    for f in fails:
        print(" ", f)
    print(f"  scratch database left at {tmp}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
