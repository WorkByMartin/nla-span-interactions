#!/usr/bin/env python3
"""Exercise the full-matrix plan on CPU, with no reconstructor.

    python test_tree_vs_linear_stub.py

Everything except the forward pass is real: a throwaway SQLite store built from
spaCy's own parse of nine short documents, the Qwen tokeniser, the swap pool over
that store, the document filters, the complete pair set, the substitute draw,
both splices, the prompt length census and the write back. What this catches is
index and bookkeeping errors, which are otherwise found only after a GPU has
been rented.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "05_dependent-pair"))
sys.path.insert(0, str(REPO / "03_parts-of-speech"))
sys.path.insert(0, str(REPO / "04_ablation-strategy"))
sys.path.insert(0, str(REPO / "db"))

import argparse  # noqa: E402
from collections import Counter  # noqa: E402
from itertools import combinations  # noqa: E402

import db as DB  # noqa: E402
import dbio  # noqa: E402
import textsub as T  # noqa: E402
import swap_ablation as S  # noqa: E402
import pair_ablation as P  # noqa: E402
import tree_vs_linear as M  # noqa: E402

from test_pair_stub import DOCS, DOMAINS, build_store  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(f"  {'ok   ' if cond else 'FAIL '} {name} {detail}")
    if not cond:
        FAILS.append(name)


def args_for(draws, n_docs=2, docs="", seed=0):
    return argparse.Namespace(draws=draws, seed=seed, n_docs=n_docs,
                              docs=docs, batch=16)


def main():
    import spacy
    from transformers import AutoTokenizer

    nlp = spacy.load("en_core_web_sm")
    qtok = AutoTokenizer.from_pretrained(T.QWEN)
    scheme = dbio.spacy_scheme(nlp)
    conn = build_store(nlp, scheme)
    # one document is given a CJK fraction over the cap, and the rest none, so
    # the filter has something real to reject
    with DB.transaction(conn):
        conn.execute("UPDATE docs SET cjk_fraction = 0.0")
        conn.execute("UPDATE docs SET cjk_fraction = 0.42 WHERE doc_id = 109")

    cache = {}
    pool = S.build_pool(conn, qtok, cache)
    index = P.index_pool(pool, qtok, cache)
    syn = P.load_syntax(conn, scheme)
    for r in conn.execute("SELECT doc_id, cjk_fraction FROM docs"):
        syn["docs"][int(r["doc_id"])]["cjk"] = r["cjk_fraction"]
    check("the store came back", len(syn["docs"]) == len(DOCS)
          and len(syn["head"]) > 100,
          f"{len(syn['docs'])} docs, {len(syn['spans'])} spans, "
          f"{len(syn['head'])} head relations")

    # ------------------------------------------------------------- screening
    scr = M.screen(syn, index, qtok, cache)
    check("every document is screened", set(scr) == set(syn["docs"]))
    check("the eligible rate is a fraction of the clean lexical words",
          all(0.0 <= s["rate"] <= 1.0 and s["eligible"] <= s["clean_lexical"]
              for s in scr.values()),
          f"rates {sorted(round(s['rate'], 3) for s in scr.values())}")
    check("the rate is eligible over clean lexical, not over all spans",
          all(abs(s["rate"] * s["clean_lexical"] - s["eligible"]) < 1e-9
              for s in scr.values() if s["clean_lexical"]))

    # the filters, exercised one at a time against known values
    old_cjk, old_elig = M.MAX_CJK, M.MIN_ELIGIBLE
    M.MAX_CJK, M.MIN_ELIGIBLE = 1.0, 0.0
    _, _, exc, ok = M.select_docs(syn, scr, args_for(2))
    check("with the filters wide open nothing is excluded",
          not exc and len(ok) == len(DOCS), f"{len(ok)} pass")
    M.MAX_CJK = 0.05
    _, _, exc, ok = M.select_docs(syn, scr, args_for(2))
    check("the cjk filter excludes the planted document and only that one",
          [d for d, _ in exc] == [109], str(exc))
    M.MAX_CJK, M.MIN_ELIGIBLE = 1.0, 1.01
    _, _, exc, ok = M.select_docs(syn, scr, args_for(1, n_docs=0))
    check("an eligibility floor above one excludes every document",
          len(exc) == len(DOCS) and not ok)
    M.MIN_ELIGIBLE = 0.90
    _, _, exc, ok = M.select_docs(syn, scr, args_for(1, n_docs=0))
    check("the eligibility floor keeps the documents above it and drops the "
          "rest",
          set(ok) == {d for d, s in scr.items() if s["rate"] >= 0.90}
          and len(exc) + len(ok) == len(DOCS),
          f"{len(ok)} pass, {len(exc)} excluded")
    M.MIN_ELIGIBLE = 0.0
    M.MAX_CJK = 1.0

    # the draw itself: uniform, reproducible, and blind to length
    a1, how, _, ok = M.select_docs(syn, scr, args_for(2, n_docs=4, seed=0))
    a2, _, _, _ = M.select_docs(syn, scr, args_for(2, n_docs=4, seed=0))
    a3, _, _, _ = M.select_docs(syn, scr, args_for(2, n_docs=4, seed=1))
    check("the document draw is reproducible from the seed", a1 == a2, str(a1))
    check("a different seed draws differently", a1 != a3, f"{a1} vs {a3}")
    check("the draw takes exactly n documents from those that pass",
          len(a1) == 4 and set(a1) <= set(ok))
    named, how2, _, _ = M.select_docs(syn, scr, args_for(2, docs="103,107"))
    check("--docs names the documents and says the filters were skipped",
          named == [103, 107] and "filters were not applied" in how2)

    # -------------------------------------------------------------- one document
    draws = 3
    args = args_for(draws)
    have = {(d, a, b): s for s, (d, a, b) in syn["spans"].items()}
    doc_id = 101
    pl = M.prepare(doc_id, syn, index, qtok, cache, nlp, args, have)
    check("the document planned", pl["skip"] is None, str(pl.get("skip")))
    n = pl["budget"]["n"]
    n_pairs = n * (n - 1) // 2

    # ------------------------------------------------------ budget arithmetic
    check("the budget is 1 + n x draws + C(n, 2) x draws",
          pl["budget"]["total"] == 1 + n * draws + n_pairs * draws
          and pl["budget"]["pairs"] == n_pairs,
          f"n {n}, pairs {n_pairs}, total {pl['budget']['total']}")
    check("the plan is exactly the budgeted number of passes",
          len(pl["texts"]) == pl["budget"]["total"],
          f"{len(pl['texts'])} texts")
    n_single = sum(1 for m in pl["meta"] if len(m) == 1)
    n_joint = sum(1 for m in pl["meta"] if len(m) == 2)
    check("one single per eligible word per draw", n_single == n * draws,
          f"{n_single} = {n} x {draws}")
    check("one joint edit per pair per draw", n_joint == n_pairs * draws,
          f"{n_joint} = {n_pairs} x {draws}")
    check("texts[0] is the intact document", pl["texts"][0] == pl["text"])
    check("texts and meta are aligned",
          len(pl["texts"]) == len(pl["meta"]) + 1)
    for k in (4, 17, 60):
        b = M.budget_for(k, draws)
        if b["total"] != 1 + (k + k * (k - 1) // 2) * draws:
            check(f"budget_for is wrong at n={k}", False)
    check("budget_for agrees with the closed form at several n", True)

    # ------------------------------------------------------------ the pair set
    elig = pl["eligible"]
    seen = Counter((p["span_a"], p["span_b"]) for p in pl["pairs"])
    check("the pair set is exactly the unordered pairs of the eligible words",
          {frozenset(k) for k in seen}
          == {frozenset(c) for c in combinations(elig, 2)})
    check("no pair is planned twice", len(seen) == n_pairs
          and max(seen.values()) == 1, f"{len(seen)} distinct pairs")
    check("the pair set is every unordered pair of the eligible words",
          len(seen) == len(list(combinations(elig, 2))))
    check("pair ids are dense and start at one",
          sorted(p["pair_id"] for p in pl["pairs"])
          == list(range(1, n_pairs + 1)))

    src = M.doc_eligibility(syn, index, qtok, cache, doc_id)[0]
    bad = []
    for p in pl["pairs"]:
        u, v = p["span_a"], p["span_b"]
        if src[u]["tok_i"] >= src[v]["tok_i"]:
            bad.append(("span_a is not the earlier word", p["pair_id"]))
        if p["distance"] != src[v]["tok_i"] - src[u]["tok_i"]:
            bad.append(("distance disagrees with tok_i", p["pair_id"]))
        if p["adjacent"] != (p["distance"] == 1):
            bad.append(("adjacency disagrees with the distance", p["pair_id"]))
        arc = syn["head"].get(u) == v or syn["head"].get(v) == u
        want = "arc" if arc else ("adjacent" if p["distance"] == 1 else "other")
        if p["category"] != want:
            bad.append(("category", p["pair_id"], p["category"], want))
        if (p["category"] == "arc") != arc:
            bad.append(("arc flag disagrees with the parse", p["pair_id"]))
    check("every pair is in reading order and carries the right distance, "
          "adjacency and category", not bad, str(bad[:4]))
    cats = Counter(p["category"] for p in pl["pairs"])
    check("the three categories partition the matrix",
          sum(cats.values()) == n_pairs
          and set(cats) <= {"arc", "adjacent", "other"}, str(dict(cats)))
    check("adjacent pairs are measured, not filtered out",
          cats["adjacent"] + sum(1 for p in pl["pairs"]
                                 if p["category"] == "arc"
                                 and p["distance"] == 1) > 0,
          f"{cats['adjacent']} adjacent without an arc")

    # ----------------------------------------------------- splice round trip
    fails = P.check_plan(pl["text"], pl["words"], pl["sids"], pl["pairs"],
                         pl["texts"], pl["meta"], pl["subs"], qtok, cache,
                         draws)
    check("every planned edit round-trips against the original, and the "
          "joint splice is its two singles composed", not fails,
          str(fails[:4]))
    check("verify() reports the same, and nothing else", not M.verify(
        pl, qtok, cache, args))

    # a deliberate corruption must be caught, or the check proves nothing
    hurt = dict(pl)
    hurt["texts"] = list(pl["texts"])
    hurt["texts"][3] = hurt["texts"][3] + " x"
    check("a corrupted splice is caught", bool(M.verify(hurt, qtok, cache,
                                                        args)))

    # ------------------------------------------------- common random numbers
    single = {}
    for m in pl["meta"]:
        if len(m) == 1:
            single[(m[0]["span_id"], m[0]["draw_idx"])] = m[0]["substitute"]
    crn = []
    for m in pl["meta"]:
        if len(m) != 2:
            continue
        for s in m:
            got = single.get((s["span_id"], s["draw_idx"]))
            if got != s["substitute"]:
                crn.append((s["span_id"], s["draw_idx"], got, s["substitute"]))
    check("a span splices the same string in its single and in every joint "
          "edit it appears in, at every draw", not crn, str(crn[:3]))
    per_span = Counter()
    for m in pl["meta"]:
        if len(m) == 2:
            for s in m:
                per_span[(s["span_id"], s["draw_idx"])] += 1
    check("each span is used in n - 1 joint edits per draw",
          set(per_span.values()) == {n - 1},
          f"{sorted(set(per_span.values()))}")
    check("draw indices run 0..draws-1 for every span",
          all(sorted(m[0]["draw_idx"] for m in pl["meta"]
                     if len(m) == 1 and m[0]["span_id"] == s)
              == list(range(draws)) for s in elig))
    check("singles carry depth 1 and joint edits depth 2",
          all(m[0]["depth"] == 1 for m in pl["meta"] if len(m) == 1)
          and all(x["depth"] == 2 for m in pl["meta"] if len(m) == 2
                  for x in m))
    check("every substitution names the swap source",
          all(x["source"] == M.SWAP for m in pl["meta"] for x in m))

    pl2 = M.prepare(doc_id, syn, index, qtok, cache, nlp, args, have)
    check("the whole plan is reproducible from the seed",
          pl2["texts"] == pl["texts"] and pl2["meta"] == pl["meta"])
    pl3 = M.prepare(doc_id, syn, index, qtok, cache, nlp,
                    args_for(draws, seed=1), have)
    check("a different seed changes the substitutes",
          pl3["texts"] != pl["texts"])

    # ------------------------------------------------------- prompt lengths
    lens = M.prompt_lengths(qtok, pl["texts"])
    cen = M.delta_census(lens, pl["meta"])
    check("a prompt length is measured for every planned string",
          len(lens) == len(pl["texts"]) and cen["n"] == len(lens))
    check("the length census covers every edit",
          sum(cen["single"].values()) == n_single
          and sum(cen["joint"].values()) == n_joint,
          f"singles {dict(cen['single'])}, joints {dict(cen['joint'])}")
    check("a moved prompt length is recorded, not rejected",
          not M.verify(pl, qtok, cache, args),
          f"{sum(v for d, v in cen['joint'].items() if d)} joint edits moved "
          f"the length")

    # -------------------------------------------------- the store accepts it
    run_id = DB.new_run(conn, script="06_tree-vs-linear/tree_vs_linear.py",
                        notes="stub")
    records = [([], {"mse": 0.10, "fve": 0.70, "seq_len": lens[0], "dtok": 0})]
    for j, subs in enumerate(pl["meta"], start=1):
        records.append((subs, {"mse": 0.10 + j * 1e-6,
                               "fve": 0.70 - j * 1e-6,
                               "seq_len": lens[j],
                               "dtok": lens[j] - lens[0]}))
    dbio.ensure_spans(conn, doc_id, pl["text"], pl["words"], scheme)
    dbio.write_variants(conn, doc_id, run_id, records)
    with DB.transaction(conn):
        DB.add_relations(conn, [
            (M.PAIR_SCHEME, p["span_a"], p["span_b"],
             f"arc:{p['dep']}" if p["category"] == "arc" else p["category"])
            for p in pl["pairs"]])
    one = lambda q, *a: conn.execute(q, a).fetchone()[0]
    check("every planned pass became a variant",
          one("SELECT COUNT(*) FROM variants WHERE created_run_id=?", run_id)
          == len(records))
    check("v_single sees exactly the singles",
          one("SELECT COUNT(*) FROM v_single WHERE run_id=?", run_id)
          == n_single)
    check("v_pair sees exactly the joint edits",
          one("SELECT COUNT(*) FROM v_pair WHERE run_id=?", run_id) == n_joint)
    check("every joint edit finds both of its singles",
          one("SELECT COUNT(*) FROM v_pair WHERE run_id=? AND inter_fve IS "
              "NULL", run_id) == 0)
    check("exactly one baseline for the document",
          one("SELECT COUNT(*) FROM v_baseline WHERE run_id=?", run_id) == 1)
    check("dtok is recorded for every variant",
          one("SELECT COUNT(*) FROM measurements WHERE run_id=? AND "
              "metric='dtok'", run_id) == len(records))
    check("the pair set and its categories are recoverable from relations "
          "alone",
          one("SELECT COUNT(*) FROM relations WHERE scheme=?",
              M.PAIR_SCHEME) == n_pairs)
    kinds = Counter(r[0].split(":")[0] for r in conn.execute(
        "SELECT kind FROM relations WHERE scheme=?", (M.PAIR_SCHEME,)))
    check("the recorded kinds agree with the planned categories",
          kinds == Counter("arc" if p["category"] == "arc" else p["category"]
                           for p in pl["pairs"]), str(dict(kinds)))
    check("the pair scheme did not disturb the spaCy head relations",
          one("SELECT COUNT(*) FROM relations WHERE scheme=? AND kind='head'",
              scheme) == len(syn["head"]))

    # ---------------------------------------------------------- resume skip
    done = S.already_done(conn, run_id)
    check("the committed document is seen as done", done == {doc_id},
          str(done))
    other = 103
    check("an unmeasured document is not skipped", other not in done)
    pl_b = M.prepare(other, syn, index, qtok, cache, nlp, args, have)
    recs = [([], {"mse": 0.1, "fve": 0.7, "seq_len": 10, "dtok": 0})]
    dbio.ensure_spans(conn, other, pl_b["text"], pl_b["words"], scheme)
    dbio.write_variants(conn, other, run_id, recs)
    check("a second committed document joins the skip set",
          S.already_done(conn, run_id) == {doc_id, other},
          str(S.already_done(conn, run_id)))
    check("a document under a different run is not skipped",
          doc_id not in S.already_done(conn, run_id + 1))

    M.MAX_CJK, M.MIN_ELIGIBLE = old_cjk, old_elig
    print(f"\n  doc {doc_id}: {n} eligible words, {n_pairs} pairs, "
          f"{len(pl['texts'])} passes at {draws} draws "
          f"({dict(Counter(p['category'] for p in pl['pairs']))})")
    ex = [m for m in pl["meta"] if len(m) == 2][:2]
    by_sid = {s: k for k, s in enumerate(pl["sids"])}
    for m in ex:
        print("  example joint edit: "
              + ", ".join(f"{pl['words'][by_sid[x['span_id']]]['text']!r} -> "
                          f"{x['substitute']!r}" for x in m))

    print("\nFAILED:" if FAILS else "\nall checks passed")
    for f in FAILS:
        print(" ", f)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
