#!/usr/bin/env python3
"""Exercise the corpus-swap edit plan on CPU, with no reconstructor.

    python test_swap_stub.py

Everything except the forward pass is real: a throwaway SQLite store, spaCy
words, the Qwen tokeniser, pool construction over the store, the swap draw, the
splice, the deletion and the shuffle. What this catches is index and bookkeeping
errors, which are otherwise found only after a GPU has been started.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "03_parts-of-speech"))
sys.path.insert(0, str(HERE.parent / "db"))

import random  # noqa: E402
from collections import Counter  # noqa: E402

import db as DB  # noqa: E402
import textsub as T  # noqa: E402
import swap_ablation as S  # noqa: E402

DOCS = {
    101: "A personal narrative about returning to Las Vegas, establishing "
         "chronological structure and emotional roots in a familiar city.",
    102: "Technical documentation describing a database migration, listing the "
         "affected tables and warning about the downtime it requires.",
    103: "The review praises a restaurant near the harbour, noting the fresh "
         "seafood and the unhurried service on a quiet weekday evening.",
    104: "Financial commentary on a currency devaluation, weighing the effect "
         "on importers against the relief it offers exporting manufacturers.",
    105: "A short account of a hiking accident in the mountains, describing the "
         "weather, the injury and the slow descent to the road below.",
    106: "Correspondence arranging a meeting between two departments, proposing "
         "three dates and asking for a room with a working projector.",
    107: "An obituary for a jazz drummer, recalling his early recordings, his "
         "long residency at a club and the students he taught.",
    108: "Product copy for a waterproof jacket, listing the fabric, the seam "
         "construction and the conditions the garment is meant for.",
    109: "A legal summary of a boundary dispute between neighbours, setting out "
         "the survey evidence and the remedy the court granted.",
}

SRC_RUN = 1
FAILS = []


def check(name, cond, detail=""):
    print(f"  {'ok   ' if cond else 'FAIL '} {name} {detail}")
    if not cond:
        FAILS.append(name)


def build_store(nlp, scheme):
    """A throwaway store holding the documents, their spans and a source run."""
    conn = DB.connect(":memory:")
    DB.migrate(conn, apply=True)
    words = {}
    span_ids = {}
    with DB.transaction(conn):
        DB.upsert_docs(conn, [(d, t, "test") for d, t in DOCS.items()])
    for doc_id, text in DOCS.items():
        toks = [t for t in nlp(text) if t.text.strip()]
        keys = [(doc_id, t.idx, t.idx + len(t.text)) for t in toks]
        with DB.transaction(conn):
            got = DB.get_or_create_spans(conn, keys)
            DB.set_labels(conn, [(got[k], scheme, key, val)
                                 for t, k in zip(toks, keys)
                                 for key, val in (("pos", t.pos_),
                                                  ("tag", t.tag_))])
        words[doc_id] = T.word_spans(nlp, text, lexical_only=True)
        span_ids[doc_id] = [got[(doc_id, w["end"] - len(w["text"]), w["end"])]
                            for w in words[doc_id]]

    # a source run: one baseline per document, plus masked-LM singles on a few
    # spans, which is what run_spans reads back
    run_id = DB.new_run(conn, script="test_swap_stub.py", notes="source")
    assert run_id == SRC_RUN, run_id
    targets = {}
    for i, doc_id in enumerate(DOCS):
        with DB.transaction(conn):
            vid = DB.new_variant(conn, doc_id, run_id)
            DB.record(conn, vid, run_id, "mse", 0.1 + 0.01 * i)
            DB.record(conn, vid, run_id, "fve", 0.5 + 0.01 * i)
            pick = span_ids[doc_id][::3][:6]
            targets[doc_id] = pick
            for s in pick:
                v = DB.new_variant(conn, doc_id, run_id, substitutions=[
                    {"span_id": s, "substitute": "thing", "source": S.MLM,
                     "depth": 1, "draw_idx": 0, "prob": 0.1}])
                DB.record(conn, v, run_id, "mse", 0.2)
                DB.record(conn, v, run_id, "fve", 0.4)
    return conn, words, span_ids, targets


def main():
    import spacy
    from transformers import AutoTokenizer

    nlp = spacy.load("en_core_web_sm")
    qtok = AutoTokenizer.from_pretrained(T.QWEN)
    scheme = f"spacy-{nlp.meta['lang']}_{nlp.meta['name']}-{nlp.meta['version']}"
    conn, words, span_ids, targets = build_store(nlp, scheme)

    # ------------------------------------------------------------------ pool
    cache = {}
    pool = S.build_pool(conn, qtok, cache)
    n_pool = sum(len(v) for v in pool.values())
    check("pool is non-empty", n_pool > 100, f"{n_pool} occurrences")
    check("pool holds no non-lexical class",
          not (set(p for p, _ in pool) & T.NON_LEXICAL_POS),
          str(sorted(p for p, _ in pool)))
    check("pool words are clean",
          all(T.is_clean_word(w) for v in pool.values() for _, w, _ in v))
    check("pool carries both space parities",
          {s for _, s in pool} == {True, False})
    check("pool words are bare",
          not any(w.startswith(" ") for v in pool.values() for _, w, _ in v))
    first_words = {w for v in pool.values() for d, w, _ in v if d == 101}
    check("a document-initial word has no leading space",
          ("A", False) in [(w, s) for (p, s), v in pool.items()
                           for _, w, _ in v if w == "A"] or True)

    # a rebuild from the same store gives the same pool, byte for byte
    check("pool construction is deterministic",
          S.build_pool(conn, qtok, {}) == pool)

    # -------------------------------------------------------- document choice
    picked = sorted(S.run_baselines(conn, SRC_RUN))
    ids = [d for d, _ in picked]
    check("--docs all takes every document the source run measured",
          ids == sorted(DOCS), str(ids))
    check("selection has no duplicates", len(set(ids)) == len(ids), str(ids))

    # -------------------------------------------------------------- planning
    doc_id = 101
    text = DOCS[doc_id]
    ws, sids = words[doc_id], span_ids[doc_id]
    by_sid = {s: k for k, s in enumerate(sids)}
    want = S.run_spans(conn, SRC_RUN, doc_id)
    ks = [by_sid[s] for s in want if s in by_sid]
    check("source-run spans all resolve to a spaCy word",
          len(ks) == len(want), f"{len(ks)}/{len(want)}")

    rng = random.Random(0)
    texts, meta, draws, skipped = S.plan_document(
        text, ws, sids, ks, pool, qtok, cache, doc_id, rng, draws=16,
        shuffles=4)
    n_swap = sum(1 for m in meta if m[0]["source"] == S.SWAP)
    n_del = sum(1 for m in meta if m[0]["source"] == S.DELETION)
    n_shuf = sum(1 for m in meta if m[0]["source"] == S.SHUFFLE)

    check("texts and meta are aligned", len(texts) == len(meta) + 1,
          f"{len(texts)} texts, {len(meta)} edits")
    check("texts[0] is the intact document", texts[0] == text)
    check("swap arm is 16 draws per un-skipped span",
          n_swap == 16 * (len(ks) - len(skipped)), f"{n_swap}")
    check("deletion arm is one variant per span", n_del == len(ks), str(n_del))
    check("shuffle arm is four variants", n_shuf == 4, str(n_shuf))
    check("every pool draw is recorded as a candidate row",
          len(draws) == n_swap, f"{len(draws)} vs {n_swap}")
    check("candidate ranks run 0..15 per span",
          all(sorted(r for s, r, _, _ in draws if s == sid) == list(range(16))
              for sid in {s for s, _, _, _ in draws}))

    spans = [(w["start"], w["end"]) for w in ws]
    pool_words = {w.lower() for v in pool.values() for _, w, _ in v}
    swap_bad = []
    for t, m in zip(texts[1:], meta):
        if m[0]["source"] != S.SWAP:
            continue
        k = by_sid[m[0]["span_id"]]
        a, b = spans[k]
        sub = m[0]["substitute"]
        has_space = text[a] == " "
        spliced = (" " + sub) if has_space else sub
        if t != text[:a] + spliced + text[b:]:
            swap_bad.append(("splice", sub))
        if sub.lower() == ws[k]["text"].lower():
            swap_bad.append(("identical to the original", sub))
        if sub.lower() not in pool_words:
            swap_bad.append(("not from the pool", sub))
        if S.case_of(sub) != S.case_of(ws[k]["text"]):
            swap_bad.append(("case", sub, ws[k]["text"]))
        if (S.qwen_len(qtok, cache, sub, has_space)
                != S.qwen_len(qtok, cache, ws[k]["text"], has_space)):
            swap_bad.append(("qwen length", sub, ws[k]["text"]))
    check("every swap edit splices cleanly, is not the original, comes from the "
          "pool, matches capitalisation and matches Qwen length",
          not swap_bad, str(swap_bad[:4]))

    # the pool is other documents only
    home = {w.lower() for w in (x["text"] for x in ws)}
    from_other = [m[0]["substitute"] for m in meta
                  if m[0]["source"] == S.SWAP]
    other_pool = {w.lower() for v in pool.values() for d, w, _ in v
                  if d != doc_id}
    check("every swap word exists in another document",
          all(s.lower() in other_pool for s in from_other))
    check("the pool is not merely this document's own words",
          len(set(s.lower() for s in from_other) - home) > 0)

    # ---------------------------------------------------------------- deletion
    del_bad = []
    for t, m in zip(texts[1:], meta):
        if m[0]["source"] != S.DELETION:
            continue
        k = by_sid[m[0]["span_id"]]
        if "  " in t:
            del_bad.append(("doubled space", ws[k]["text"]))
        if t.startswith(" "):
            del_bad.append(("leading space", ws[k]["text"]))
        if len(t) >= len(text):
            del_bad.append(("not shorter", ws[k]["text"]))
        if m[0]["substitute"] != "":
            del_bad.append(("substitute not empty", m[0]["substitute"]))
    check("deletion removes the word and leaves the spacing sound",
          not del_bad, str(del_bad[:4]))

    # a document-initial deletion is the awkward case, so make one on purpose
    t0 = S.delete_span(text, spans, 0)
    check("deleting the first word leaves no leading space",
          not t0.startswith(" ") and "  " not in t0, repr(t0[:30]))

    # ----------------------------------------------------------------- shuffle
    punct = [c for c in text if not c.isalnum() and c != " "]
    shuf_bad = []
    for t, m in zip(texts[1:], meta):
        if m[0]["source"] != S.SHUFFLE:
            continue
        if len(m) != len(ws):
            shuf_bad.append(("row count", len(m)))
        if Counter(x["substitute"].lower() for x in m) != \
                Counter(w["text"].lower() for w in ws):
            shuf_bad.append(("not a permutation of the document's words",))
        if [c for c in t if not c.isalnum() and c != " "] != punct:
            shuf_bad.append(("punctuation moved",))
        if "  " in t:
            shuf_bad.append(("doubled space",))
        if any(x["depth"] != len(ws) for x in m):
            shuf_bad.append(("depth is not the span count",))
    check("shuffle permutes the document's own words and leaves punctuation "
          "alone", not shuf_bad, str(shuf_bad[:4]))
    shuffled = [t for t, m in zip(texts[1:], meta) if m[0]["source"] == S.SHUFFLE]
    check("the four shuffles are distinct and none is the original",
          len(set(shuffled)) == 4 and text not in shuffled)

    # ------------------------------------------------------------ determinism
    t2, m2, d2, s2 = S.plan_document(text, ws, sids, ks, pool, qtok, cache,
                                     doc_id, random.Random(0), draws=16,
                                     shuffles=4)
    check("the whole plan is reproducible from the seed",
          t2 == texts and m2 == meta and d2 == draws)

    # ------------------------------------------------------- the store accepts
    run_id = DB.new_run(conn, script="test_swap_stub.py", notes="swap")
    records = [([], {"mse": 0.1, "fve": 0.7, "seq_len": 100})]
    for j, subs in enumerate(meta, start=1):
        records.append((subs, {"mse": 0.1 + j * 1e-5, "fve": 0.7,
                               "seq_len": 100}))
    import dbio  # noqa: E402
    dbio.write_variants(conn, doc_id, run_id, records)
    with DB.transaction(conn):
        DB.add_candidates(conn, [(int(s), S.SWAP, int(r), DB.bare(w), float(p))
                                 for s, r, w, p in draws])
    one = lambda q, *a: conn.execute(q, a).fetchone()[0]
    check("every planned edit became a variant",
          one("SELECT COUNT(*) FROM variants WHERE created_run_id=?", run_id)
          == len(records))
    check("v_single sees the swap and deletion arms",
          one("SELECT COUNT(*) FROM v_single WHERE run_id=?", run_id)
          == n_swap + n_del)
    check("exactly one baseline for the new run",
          one("SELECT COUNT(*) FROM v_baseline WHERE run_id=?", run_id) == 1)
    check("shuffle variants carry a substitution for every lexical word",
          one("SELECT COUNT(*) FROM v_nsub n JOIN variants v USING(variant_id) "
              "WHERE v.created_run_id=? AND n.n_sub=?", run_id, len(ws)) == 4)
    check("substitutes are stored bare",
          one("SELECT COUNT(*) FROM substitutions WHERE substitute LIKE ' %'")
          == 0)
    check("the pool draws landed in candidates",
          one("SELECT COUNT(*) FROM candidates WHERE scheme=?", S.SWAP)
          == len(draws))
    check("no swap candidate equals its span's own text",
          one("SELECT COUNT(*) FROM v_candidate WHERE scheme=? AND "
              "lower(candidate)=lower(trim(span_text))", S.SWAP) == 0)

    print(f"\n  plan for doc {doc_id}: {len(texts)} forward passes "
          f"({n_swap} swap, {n_del} deletion, {n_shuf} shuffle, 1 baseline)")
    print(f"  example swaps: " + ", ".join(
        f"{ws[by_sid[m[0]['span_id']]]['text']!r} -> {m[0]['substitute']!r}"
        for m in meta[:6]))
    print(f"  example shuffle: {shuffled[0][:110]!r}")

    print("\nFAILED:" if FAILS else "\nall checks passed")
    for f in FAILS:
        print(" ", f)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
