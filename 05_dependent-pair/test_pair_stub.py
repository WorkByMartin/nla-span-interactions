#!/usr/bin/env python3
"""Exercise the dependent-pair plan on CPU, with no reconstructor.

    python test_pair_stub.py

Everything except the forward pass is real: a throwaway SQLite store built from
spaCy's own parse of nine documents, the Qwen tokeniser, the swap pool over that
store, arc collection, control matching, the substitute draw, both splices and
the write back. What this catches is index and bookkeeping errors, which are
otherwise found only after an A100 has been started.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "03_parts-of-speech"))
sys.path.insert(0, str(REPO / "04_ablation-strategy"))
sys.path.insert(0, str(REPO / "db"))

import random  # noqa: E402
from collections import Counter  # noqa: E402

import db as DB  # noqa: E402
import dbio  # noqa: E402
import textsub as T  # noqa: E402
import swap_ablation as S  # noqa: E402
import pair_ablation as P  # noqa: E402

DOCS = {
    101: "The committee rejected the proposal that the borough council had "
         "drafted, arguing that its funding assumptions were unrealistic and "
         "that the timetable ignored the winter closure of the bridge.",
    102: "Technical documentation describing a database migration, listing the "
         "affected tables and warning the operators that the downtime it "
         "requires will exceed the window the change board approved.",
    103: "The reviewer praised the restaurant near the harbour, noting the "
         "fresh seafood and the unhurried service, and she recommended the "
         "tasting menu to anyone visiting the town in autumn.",
    104: "Financial commentary on a currency devaluation weighed the effect on "
         "importers against the relief it offered exporting manufacturers, and "
         "concluded that the central bank had misjudged the timing.",
    105: "A short account of a hiking accident in the mountains described the "
         "weather, the injury and the slow descent to the road below, where a "
         "farmer who heard the shouting drove the party to hospital.",
    106: "Correspondence arranging a meeting between two departments proposed "
         "three dates and asked the facilities office for a room with a "
         "working projector and a table the whole delegation could sit at.",
    107: "An obituary for a jazz drummer recalled his early recordings, his "
         "long residency at a club near the river, and the students he taught "
         "who now lead bands of their own across the country.",
    108: "Product copy for a waterproof jacket listed the fabric, the seam "
         "construction and the conditions the garment is meant for, promising "
         "that the hood would not obstruct the wearer's vision.",
    109: "A legal summary of a boundary dispute between neighbours set out the "
         "survey evidence and the remedy the court granted, and noted that the "
         "losing party had already lodged an appeal.",
}
DOMAINS = {101: "law", 102: "computer_science_and_technology", 103: "travel",
           104: "finance", 105: "sports", 106: "public_administration",
           107: "music_and_dance", 108: "fashion", 109: "law"}

FAILS = []


def check(name, cond, detail=""):
    print(f"  {'ok   ' if cond else 'FAIL '} {name} {detail}")
    if not cond:
        FAILS.append(name)


def build_store(nlp, scheme):
    """A throwaway store holding the documents, every spaCy token and the parse.

    Shaped like the real one: spans for every token including punctuation, pos,
    tag, dep and tok_i labels, and a `head` relation from dependent to head for
    everything but the root.
    """
    conn = DB.connect(":memory:")
    DB.migrate(conn, apply=True)
    with DB.transaction(conn):
        DB.upsert_docs(conn, [(d, t, "test") for d, t in DOCS.items()])
        conn.executemany("UPDATE docs SET domain = ? WHERE doc_id = ?",
                         [(DOMAINS[d], d) for d in DOCS])
    for doc_id, text in DOCS.items():
        toks = [t for t in nlp(text) if t.text.strip()]
        keys = [(doc_id, t.idx, t.idx + len(t.text)) for t in toks]
        with DB.transaction(conn):
            got = DB.get_or_create_spans(conn, keys)
            DB.set_labels(conn, [(got[k], scheme, key, val)
                                 for t, k in zip(toks, keys)
                                 for key, val in (("pos", t.pos_),
                                                  ("tag", t.tag_),
                                                  ("dep", t.dep_),
                                                  ("tok_i", t.i))])
            sid = {t.i: got[k] for t, k in zip(toks, keys)}
            DB.add_relations(conn, [
                (scheme, sid[t.i], sid[t.head.i], "head")
                for t in toks if t.head.i != t.i and t.head.i in sid])
    return conn


def main():
    import spacy
    from transformers import AutoTokenizer

    nlp = spacy.load("en_core_web_sm")
    qtok = AutoTokenizer.from_pretrained(T.QWEN)
    scheme = dbio.spacy_scheme(nlp)
    conn = build_store(nlp, scheme)

    # ------------------------------------------------------------------ pool
    cache = {}
    pool = S.build_pool(conn, qtok, cache)
    index = P.index_pool(pool, qtok, cache)
    check("pool is non-empty", sum(len(v) for v in pool.values()) > 100,
          f"{sum(len(v) for v in pool.values())} occurrences")

    syn = P.load_syntax(conn, scheme)
    check("the parse came back", len(syn["head"]) > 100 and
          len(syn["pos"]) == len(syn["spans"]),
          f"{len(syn['head'])} head relations, {len(syn['spans'])} spans")

    # the bucketed index must reproduce swap_ablation's own eligibility list
    rng = random.Random(0)
    sample = rng.sample(sorted(syn["spans"]), 60)
    same = True
    for sid in sample:
        doc_id, a, b = syn["spans"][sid]
        text = syn["docs"][doc_id]["text"]
        p = syn["pos"].get(sid)
        if p in T.NON_LEXICAL_POS:
            continue
        word = text[a:b]
        if not T.is_clean_word(word):
            continue
        hs = a > 0 and text[a - 1] == " "
        case = S.case_of(word)
        nq = S.qwen_len(qtok, cache, word, hs)
        mine = P.candidates_for(index, p, hs, case, nq, word, doc_id)
        theirs = S.pool_for(pool, qtok, cache, p, hs, nq, word, case, doc_id)
        if mine != theirs:
            same = False
            break
    check("the bucketed pool index reproduces swap_ablation.pool_for entry for "
          "entry", same)

    # ----------------------------------------------------------- eligibility
    elig, rejected = P.eligible_spans(syn, index, qtok, cache)
    check("eligibility drops every non-lexical span",
          not any(elig[s]["pos"] in T.NON_LEXICAL_POS for s in elig),
          f"{len(elig)}/{len(syn['spans'])} eligible, "
          f"rejected {dict(rejected)}")
    check("every eligible span has at least one substitute",
          all(v["n_cands"] > 0 for v in elig.values()))
    check("every eligible span is a clean word",
          all(T.is_clean_word(v["text"]) for v in elig.values()))

    # ------------------------------------------------------------------ arcs
    arcs = P.collect_arcs(syn, elig)
    n_arc = sum(len(v) for v in arcs.values())
    check("arcs were found", n_arc > 20,
          ", ".join(f"{k} {len(v)}" for k, v in sorted(arcs.items())))
    bad = []
    for d, rows in arcs.items():
        for a in rows:
            if d not in P.ARC_DEPS:
                bad.append(("dep type", d))
            if syn["head"].get(a["span_a"]) != a["span_b"]:
                bad.append(("not a head relation", a["span_a"]))
            if a["distance"] < P.MIN_DISTANCE:
                bad.append(("contiguous", a["distance"]))
            if elig[a["span_a"]]["doc_id"] != elig[a["span_b"]]["doc_id"]:
                bad.append(("crosses documents", a["span_a"]))
            first = a["span_a"] if a["dep_first"] else a["span_b"]
            second = a["span_b"] if a["dep_first"] else a["span_a"]
            if elig[first]["tok_i"] >= elig[second]["tok_i"]:
                bad.append(("position order", a["span_a"]))
            if a["combo"] != (elig[first]["pos"], elig[second]["pos"]):
                bad.append(("combo is not in position order", a["span_a"]))
    check("every arc is a real non-contiguous head relation between two "
          "eligible spans, read in position order", not bad, str(bad[:4]))

    # --------------------------------------------------------- the dep graph
    head = syn["head"]
    some = sorted(head)[0]
    h = head[some]
    check("graph_neighbours sees a direct arc",
          P.graph_neighbours(head, some, h) == "arc")
    sibs = [(u, v) for u in head for v in head
            if u < v and head[u] == head[v]][:1]
    check("graph_neighbours sees a sibling",
          not sibs or P.graph_neighbours(head, *sibs[0]) == "sibling",
          str(sibs[:1]))
    gps = [(u, head[h2]) for u, h2 in head.items()
           if h2 in head and head[h2] != u][:1]
    check("graph_neighbours sees a grandparent",
          not gps or P.graph_neighbours(head, *gps[0]) == "grandparent",
          str(gps[:1]))

    # -------------------------------------------------------------- sampling
    rows, per_type, unmatched = P.sample_pairs(syn, elig, 6, 0)
    n_arcs = sum(1 for r in rows if r["kind"] == "arc")
    n_ctrl = sum(1 for r in rows if r["kind"] == "control")
    check("the per-type cap is respected",
          all(v["sampled"] <= 6 for v in per_type.values()),
          str({k: (v["available"], v["sampled"]) for k, v in per_type.items()}))
    check("every sampled arc is either matched or listed as unmatched",
          n_arcs == n_ctrl + len(unmatched),
          f"{n_arcs} arcs, {n_ctrl} controls, {len(unmatched)} unmatched")
    check("pair ids are unique and dense",
          sorted(r["pair_id"] for r in rows) == list(range(1, len(rows) + 1)))

    rows2, per_type2, unmatched2 = P.sample_pairs(syn, elig, 6, 0)
    check("sampling is reproducible from the seed",
          rows2 == rows and per_type2 == per_type and unmatched2 == unmatched)
    rows3, _, _ = P.sample_pairs(syn, elig, 6, 1)
    check("a different seed gives a different sample", rows3 != rows)

    # ------------------------------------------------------- control quality
    by_id = {r["pair_id"]: r for r in rows}
    cbad = []
    for r in rows:
        if r["kind"] != "control":
            continue
        arc = by_id[r["match_of"]]
        u, v = r["span_first"], r["span_second"]
        if P.graph_neighbours(head, u, v) is not None:
            cbad.append(("control is graph-adjacent",
                         P.graph_neighbours(head, u, v)))
        if (r["pos_first"], r["pos_second"]) != (arc["pos_first"],
                                                 arc["pos_second"]):
            cbad.append(("POS combination", r["pair_id"]))
        if elig[u]["doc_id"] != elig[v]["doc_id"]:
            cbad.append(("control crosses documents", r["pair_id"]))
        if r["distance"] != elig[v]["tok_i"] - elig[u]["tok_i"]:
            cbad.append(("distance disagrees with tok_i", r["pair_id"]))
        if r["match_quality"] in ("exact", "other-doc"):
            if r["distance"] != arc["distance"]:
                cbad.append(("exact match is not the same distance",
                             r["pair_id"]))
        else:
            if abs(r["distance"] - arc["distance"]) != 1:
                cbad.append(("+-1 match is further than one token",
                             r["pair_id"]))
        if r["match_quality"] in ("exact", "dist+-1"):
            if r["doc_id"] != arc["doc_id"]:
                cbad.append(("same-document quality, other document",
                             r["pair_id"]))
        else:
            if syn["docs"][r["doc_id"]]["domain"] != \
                    syn["docs"][arc["doc_id"]]["domain"]:
                cbad.append(("cross-document control is a different domain",
                             r["pair_id"]))
        if r["dep_first"] != arc["dep_first"]:
            cbad.append(("orientation does not mirror the arc", r["pair_id"]))
        if (r["span_a"], r["span_b"]) != ((u, v) if arc["dep_first"]
                                          else (v, u)):
            cbad.append(("span_a does not play the dependent's role",
                         r["pair_id"]))
    check("every control matches its arc on POS combination, distance, "
          "orientation and graph distance", not cbad, str(cbad[:4]))
    q = Counter(r["match_quality"] for r in rows if r["kind"] == "control")
    print(f"       control quality {dict(q)}, unmatched {len(unmatched)}")

    # -------------------------------------------------------------- planning
    doc_id = sorted({r["doc_id"] for r in rows})[0]
    prs = [r for r in rows if r["doc_id"] == doc_id]
    text = DOCS[doc_id]
    words = T.word_spans(nlp, text, lexical_only=True)
    keys = [(doc_id,) + dbio.bare_span(w) for w in words]
    have = {(d, a, b): s for s, (d, a, b) in syn["spans"].items()}
    sids = [have.get(k) for k in keys]
    check("every lexical word resolves to a span in the store",
          all(s is not None for s in sids))

    draws = 4
    texts, meta, sbs = P.plan_document(text, words, sids, prs, elig, index,
                                       0, draws)
    n_single = sum(1 for m in meta if len(m) == 1)
    n_joint = sum(1 for m in meta if len(m) == 2)
    spans_used = {p[k] for p in prs for k in ("span_a", "span_b")}
    check("texts and meta are aligned", len(texts) == len(meta) + 1,
          f"{len(texts)} texts, {len(meta)} edits")
    check("texts[0] is the intact document", texts[0] == text)
    check("one single per span per draw",
          n_single == len(spans_used) * draws,
          f"{n_single} = {len(spans_used)} spans x {draws}")
    check("one joint edit per pair per draw", n_joint == len(prs) * draws,
          f"{n_joint} = {len(prs)} pairs x {draws}")
    check("singles carry depth 1 and joints depth 2",
          all(m[0]["depth"] == 1 for m in meta if len(m) == 1)
          and all(s["depth"] == 2 for m in meta if len(m) == 2 for s in m))
    check("draw indices run 0..n-1 for every span",
          all(sorted(m[0]["draw_idx"] for m in meta
                     if len(m) == 1 and m[0]["span_id"] == s)
              == list(range(draws)) for s in spans_used))
    check("every substitution names the swap source",
          all(x["source"] == P.SWAP for m in meta for x in m))

    fails = P.check_plan(text, words, sids, prs, texts, meta, sbs, qtok, cache,
                         draws)
    check("every planned edit round-trips, and common random numbers hold",
          not fails, str(fails[:4]))

    # sharing is what common random numbers buy, so make sure it happened here
    use = Counter()
    for p in prs:
        use[p["span_a"]] += 1
        use[p["span_b"]] += 1
    shared = [s for s, n in use.items() if n > 1]
    check("at least one span is shared between pairs, and its singles are not "
          "duplicated", not shared or
          all(sum(1 for m in meta
                  if len(m) == 1 and m[0]["span_id"] == s) == draws
              for s in shared),
          f"{len(shared)} shared spans of {len(use)}")

    t2, m2, s2 = P.plan_document(text, words, sids, prs, elig, index, 0, draws)
    check("the whole plan is reproducible from the seed",
          t2 == texts and m2 == meta and s2 == sbs)
    t3, _, _ = P.plan_document(text, words, sids, prs, elig, index, 1, draws)
    check("a different seed changes the substitutes", t3 != texts)

    # a span in two pairs gets the SAME string in both joint edits at a draw
    crn = True
    for s in shared:
        for i in range(draws):
            got = {x["substitute"] for m in meta if len(m) == 2 for x in m
                   if x["span_id"] == s and x["draw_idx"] == i}
            if len(got) > 1:
                crn = False
    check("a shared span splices one string per draw across every pair it is "
          "in", crn)

    # ------------------------------------------------------- the store accepts
    run_id = DB.new_run(conn, script="test_pair_stub.py", notes="pairs")
    records = [([], {"mse": 0.10, "fve": 0.70, "seq_len": 100})]
    for j, subs in enumerate(meta, start=1):
        records.append((subs, {"mse": 0.10 + j * 1e-5,
                               "fve": 0.70 - j * 1e-5, "seq_len": 100}))
    dbio.write_variants(conn, doc_id, run_id, records)
    with DB.transaction(conn):
        DB.add_relations(conn, [
            (P.PAIR_SCHEME, r["span_a"], r["span_b"],
             f"arc:{r['dep']}" if r["kind"] == "arc" else "control")
            for r in prs])
    one = lambda q, *a: conn.execute(q, a).fetchone()[0]
    check("every planned edit became a variant",
          one("SELECT COUNT(*) FROM variants WHERE created_run_id=?", run_id)
          == len(records))
    check("v_single sees exactly the singles",
          one("SELECT COUNT(*) FROM v_single WHERE run_id=?", run_id)
          == n_single)
    check("v_pair sees exactly the joint edits",
          one("SELECT COUNT(*) FROM v_pair WHERE run_id=?", run_id) == n_joint)
    check("every joint edit finds both of its singles, so the interaction is "
          "not NULL",
          one("SELECT COUNT(*) FROM v_pair WHERE run_id=? AND inter_fve IS NULL",
              run_id) == 0)
    check("exactly one baseline for the run",
          one("SELECT COUNT(*) FROM v_baseline WHERE run_id=?", run_id) == 1)
    check("substitutes are stored bare",
          one("SELECT COUNT(*) FROM substitutions WHERE substitute LIKE ' %'")
          == 0)
    check("the pair set is recoverable from relations alone",
          one("SELECT COUNT(*) FROM relations WHERE scheme=?", P.PAIR_SCHEME)
          == len({(r["span_a"], r["span_b"]) for r in prs}))
    check("the pair scheme did not disturb the spaCy head relations",
          one("SELECT COUNT(*) FROM relations WHERE scheme=? AND kind='head'",
              scheme) == len(syn["head"]))

    # the interaction identity the analysis rests on, checked against SQL
    r = conn.execute("SELECT both_fve, a_fve, b_fve, base_fve, inter_fve "
                     "FROM v_pair WHERE run_id=? LIMIT 1", (run_id,)).fetchone()
    want = r["both_fve"] - r["a_fve"] - r["b_fve"] + r["base_fve"]
    check("v_pair's interaction is both - a - b + baseline",
          abs(want - r["inter_fve"]) < 1e-12, f"{r['inter_fve']:.3e}")

    print(f"\n  doc {doc_id}: {len(prs)} pairs over {len(spans_used)} spans, "
          f"{len(texts)} forward passes ({n_single} single, {n_joint} joint, "
          f"1 baseline)")
    ex = [m for m in meta if len(m) == 2][:3]
    by_sid = {s: k for k, s in enumerate(sids)}
    for m in ex:
        print("  example joint edit: "
              + ", ".join(f"{words[by_sid[x['span_id']]]['text']!r} -> "
                          f"{x['substitute']!r}" for x in m))

    print("\nFAILED:" if FAILS else "\nall checks passed")
    for f in FAILS:
        print(" ", f)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
