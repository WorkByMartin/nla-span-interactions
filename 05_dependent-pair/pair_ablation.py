#!/usr/bin/env python3
"""Dependent-pair interactions: ablate two words together and apart.

    python pair_ablation.py --dry-run --db ../db/ffw_span-ablation_database.sqlite

    python pair_ablation.py --ar "$ASSETS/qwen36-27b_ar-l43-s600_model" \
        --traces ../01_corpus-and-spans/results/ffw_pilot_traces.parquet \
        --db ../db/ffw_span-ablation_database.sqlite \
        --draws 8 --per-type 150 --batch 8 --precision bf16 --threads 8 --seed 0

The edit is the corpus swap of 04_ablation-strategy: a word is replaced by a word
drawn uniformly over occurrences in OTHER documents' verbalisations, matched on
spaCy coarse POS, Qwen token count and leading-space parity, the original
excluded case-blind and the substitute recased to the original. That machinery is
imported from swap_ablation.py rather than restated.

What is new is the unit. Two spans are ablated singly and together, so the
interaction

    interaction = e(a) + e(b) - e(both)

is measurable per draw, where e is FVE points lost against the same document's
unedited baseline. Positive means the pair costs LESS than the sum of its
singles, so the two words carry overlapping information.

Two kinds of pair are measured on the same footing:

  arc      a direct dependency arc between two lexical words, of one of nine
           spaCy `dep` types on the dependent, with the two words at least two
           tokens apart so the pair is never a contiguous bigram.
  control  a pair from the same document with the same ordered POS combination
           and the same token distance, with no arc between them in either
           direction and no path of length two through a shared head or a
           head-of-head, so siblings and grandparent pairs are excluded.

Common random numbers: a span's substitute for draw k is a property of the span
and the draw, not of the pair, so the "a alone" pass at draw k and the "both"
pass at draw k splice the SAME string at a. A span shared by several pairs
therefore contributes one set of singles rather than one per pair.

--dry-run does everything except the forward pass: the pool, the sampling, the
control matching, every splice and its round-trip check, the pass budget and the
pair table. No reconstructor, no GPU.
"""
import argparse
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

HERE = Path(__file__).resolve().parent


def _package(marker, *candidates):
    """Where a sibling directory's code actually is.

    Checked rather than assumed because a GPU host stages this directory on its
    own, so the numbered directories this imports from may sit beside it, or
    their contents may be copied flat next to this file instead.
    """
    for c in candidates:
        if (c / marker).is_file():
            return c
    raise SystemExit(f"cannot find {marker}: not in "
                     + ", ".join(str(c) for c in candidates)
                     + ". Copy it across alongside this directory.")

REPO = HERE.parent

sys.path.insert(0, str(_package("harness.py", REPO / "03_parts-of-speech", HERE)))
sys.path.insert(0, str(_package("swap_ablation.py", REPO / "04_ablation-strategy",
                                HERE)))
sys.path.insert(0, str(_package("extract_traces.py", REPO / "01_corpus-and-spans",
                                HERE)))
sys.path.insert(0, str(_package("dbio.py", REPO / "db", HERE / "db",
                                HERE.parent / "db")))

import textsub as T  # noqa: E402
import swap_ablation as S  # noqa: E402
import db as DB  # noqa: E402
import dbio  # noqa: E402

SWAP = S.SWAP                       # 'corpus-swap/pos+len'
PAIR_SCHEME = "pair-ablation/arc+control"
SPACY = "spacy-en_core_web_sm-3.8.0"

# dep label carried by the DEPENDENT of an arc that this experiment measures
ARC_DEPS = ["dobj", "nsubj", "conj", "appos", "nmod", "advcl", "ccomp",
            "attr", "poss"]
MIN_DISTANCE = 2                    # tokens, so no pair is a contiguous bigram

ASSETS = ["ffw_span-ablation_database", "qwen36-27b_ar-l43-s600_model",
          "qwen36-27b_tokenizer", "spacy_en-core-web-sm_model"]

# the order control fallbacks are tried in, best first
QUALITIES = ["exact", "dist+-1", "other-doc", "other-doc+-1"]


# ------------------------------------------------------------------ the store

def load_syntax(conn, scheme=SPACY):
    """Everything the pair selection reads: spans, labels, heads, documents.

    Returns a dict of plain Python maps rather than leaving the queries to be
    repeated per document, because the control search needs the whole document
    in memory anyway and the whole corpus is 17k spans.
    """
    docs = {}
    for r in conn.execute("SELECT doc_id, text, domain FROM docs"):
        docs[int(r["doc_id"])] = {"text": r["text"], "domain": r["domain"]}
    spans = {}
    for r in conn.execute("SELECT span_id, doc_id, char_start, char_end "
                          "FROM spans ORDER BY span_id"):
        spans[int(r["span_id"])] = (int(r["doc_id"]), int(r["char_start"]),
                                    int(r["char_end"]))
    lab = defaultdict(dict)
    for r in conn.execute("SELECT span_id, key, value FROM labels "
                          "WHERE scheme = ?", (scheme,)):
        lab[r["key"]][int(r["span_id"])] = r["value"]
    head = {}
    for r in conn.execute("SELECT span_a, span_b FROM relations "
                          "WHERE scheme = ? AND kind = 'head'", (scheme,)):
        head[int(r["span_a"])] = int(r["span_b"])
    return {"docs": docs, "spans": spans,
            "pos": dict(lab["pos"]), "dep": dict(lab["dep"]),
            "tok_i": {k: int(v) for k, v in lab["tok_i"].items()},
            "head": head}


# -------------------------------------------------------------- pool indexing

def index_pool(pool, qtok, cache):
    """The swap pool bucketed so eligibility is a lookup rather than a scan.

    swap_ablation.pool_for walks a whole (pos, parity) cell for every span it is
    asked about, recasing each entry to test its Qwen length. Selection here asks
    about every span in the corpus, so the recasing is done once per (cell, case)
    and the result bucketed by Qwen length. Order inside a bucket is the order
    pool_for would have produced, so the two agree entry for entry.
    """
    out = {}
    for (pos, has_space), entries in pool.items():
        for case in ("lower", "title", "upper"):
            b = defaultdict(list)
            for d, w, _ in entries:
                c = S.apply_case(w, case)
                b[S.qwen_len(qtok, cache, c, has_space)].append((d, c))
            out[(pos, has_space, case)] = dict(b)
    return out


def candidates_for(index, pos, has_space, case, n_qwen, orig, doc_id):
    """The eligible substitutes for one span, in the pool's own order."""
    bucket = index.get((pos, has_space, case), {}).get(n_qwen, ())
    lo = orig.lower()
    return [c for d, c in bucket if d != doc_id and c.lower() != lo]


# ------------------------------------------------------------- span eligibility

def eligible_spans(syn, index, qtok, cache):
    """{span_id: info} for every span the swap ablation can actually serve.

    A span qualifies when its POS is lexical, its text passes the clean-word
    test, and at least one pool word matches it on POS, parity, case-corrected
    Qwen length and a different document. `info` carries what both the pair
    search and the plan need, so neither recomputes it.
    """
    out, rejected = {}, Counter()
    for sid, (doc_id, a, b) in syn["spans"].items():
        d = syn["docs"].get(doc_id)
        if d is None:
            rejected["no document"] += 1
            continue
        pos = syn["pos"].get(sid)
        if pos is None or pos in T.NON_LEXICAL_POS:
            rejected["non-lexical POS"] += 1
            continue
        text = d["text"]
        word = text[a:b]
        if not T.is_clean_word(word):
            rejected["not a clean word"] += 1
            continue
        has_space = a > 0 and text[a - 1] == " "
        case = S.case_of(word)
        n_q = S.qwen_len(qtok, cache, word, has_space)
        cands = candidates_for(index, pos, has_space, case, n_q, word, doc_id)
        if not cands:
            rejected["empty swap pool"] += 1
            continue
        out[sid] = {"span_id": sid, "doc_id": doc_id, "start": a, "end": b,
                    "text": word, "pos": pos, "dep": syn["dep"].get(sid),
                    "tok_i": syn["tok_i"].get(sid), "has_space": has_space,
                    "case": case, "n_qwen": n_q, "n_cands": len(cands)}
    return out, rejected


# ------------------------------------------------------------------ pair search

def graph_neighbours(head, u, v):
    """Is there an arc between u and v, or a path of length two through a head?

    Direct arc either way, a shared head (siblings), or one being the other's
    head-of-head (grandparent). A control pair must fail all of these.
    """
    hu, hv = head.get(u), head.get(v)
    if hu == v or hv == u:
        return "arc"
    if hu is not None and hu == hv:
        return "sibling"
    if hu is not None and head.get(hu) == v:
        return "grandparent"
    if hv is not None and head.get(hv) == u:
        return "grandparent"
    return None


def collect_arcs(syn, elig):
    """{dep: [arc dicts]} for every eligible non-contiguous arc in the corpus.

    span_a is the dependent and span_b its head, which is the `relations`
    convention. `first` is whichever of the two comes first in the document, so
    the POS combination a control has to match is read in position order.
    """
    out = defaultdict(list)
    for dep_id, head_id in sorted(syn["head"].items()):
        d = syn["dep"].get(dep_id)
        if d not in ARC_DEPS:
            continue
        a, b = elig.get(dep_id), elig.get(head_id)
        if a is None or b is None:
            continue
        if a["doc_id"] != b["doc_id"]:
            continue
        dist = abs(a["tok_i"] - b["tok_i"])
        if dist < MIN_DISTANCE:
            continue
        dep_first = a["tok_i"] < b["tok_i"]
        combo = ((a["pos"], b["pos"]) if dep_first else (b["pos"], a["pos"]))
        out[d].append({"dep": d, "doc_id": a["doc_id"],
                       "span_a": dep_id, "span_b": head_id,
                       "dep_first": dep_first, "distance": dist,
                       "combo": combo,
                       "text_a": a["text"], "text_b": b["text"],
                       "pos_a": a["pos"], "pos_b": b["pos"]})
    return dict(out)


def arc_census(syn, elig):
    """Why arcs of each type drop out, so the sampled counts can be read."""
    rows = {}
    for d in ARC_DEPS:
        rows[d] = Counter()
    for dep_id, head_id in syn["head"].items():
        d = syn["dep"].get(dep_id)
        if d not in ARC_DEPS:
            continue
        c = rows[d]
        c["arcs"] += 1
        a, b = elig.get(dep_id), elig.get(head_id)
        if a is None or b is None:
            c["endpoint not eligible"] += 1
            continue
        if abs(a["tok_i"] - b["tok_i"]) < MIN_DISTANCE:
            c["contiguous"] += 1
            continue
        c["available"] += 1
    return rows


def doc_candidates(elig):
    """Per document, {(pos_first, pos_second, distance): [(first, second)]}.

    Every ordered-by-position pair of eligible spans in the document, indexed on
    exactly what a control has to match. Built once per document and reused by
    every arc in it.
    """
    by_doc = defaultdict(list)
    for info in elig.values():
        if info["tok_i"] is not None:
            by_doc[info["doc_id"]].append(info)
    out = {}
    for doc_id, infos in by_doc.items():
        infos = sorted(infos, key=lambda x: x["tok_i"])
        idx = defaultdict(list)
        for i in range(len(infos)):
            for j in range(i + 1, len(infos)):
                u, v = infos[i], infos[j]
                dist = v["tok_i"] - u["tok_i"]
                if dist < MIN_DISTANCE:
                    continue
                idx[(u["pos"], v["pos"], dist)].append(
                    (u["span_id"], v["span_id"]))
        out[doc_id] = dict(idx)
    return out


def control_pool(cands, head, doc_id, combo, distance, used):
    """Control pairs in one document matching combo and distance exactly.

    Ordered by span id so the draw is reproducible. A pair already spoken for is
    dropped rather than offered again: two pairs over the same two spans would
    plan identical edits, and the run's variants would then be ambiguous about
    which pair they belong to.
    """
    hits = cands.get(doc_id, {}).get((combo[0], combo[1], distance), ())
    return [(u, v) for u, v in hits
            if (doc_id, u, v) not in used
            and graph_neighbours(head, u, v) is None]


def find_control(arc, cands, head, docs, used, rng):
    """A matched pair for one arc, best available quality.

    exact         same document, same POS combination, same token distance
    dist+-1       same document, distance one token either side
    other-doc     a different document of the same domain, exact distance
    other-doc+-1  a different document of the same domain, distance +-1

    Returns (span_first, span_second, quality, distance, doc_id) or None.
    """
    combo, d0, doc_id = arc["combo"], arc["distance"], arc["doc_id"]
    domain = docs[doc_id]["domain"]
    others = sorted(x for x, v in docs.items()
                    if x != doc_id and v["domain"] == domain)

    def pick(pool_docs, dists, quality):
        take = []
        for d in pool_docs:
            for dist in dists:
                take += [(d, dist, u, v) for u, v in
                         control_pool(cands, head, d, combo, dist, used)]
        if not take:
            return None
        d, dist, u, v = take[rng.randrange(len(take))]
        return u, v, quality, dist, d

    for quality, pool_docs, dists in (
            ("exact", [doc_id], [d0]),
            ("dist+-1", [doc_id], [d0 - 1, d0 + 1]),
            ("other-doc", others, [d0]),
            ("other-doc+-1", others, [d0 - 1, d0 + 1])):
        dists = [x for x in dists if x >= MIN_DISTANCE]
        if not dists or not pool_docs:
            continue
        got = pick(pool_docs, dists, quality)
        if got is not None:
            return got
    return None


def sample_pairs(syn, elig, per_type, seed):
    """The whole pair table: sampled arcs, and one control for each.

    Deterministic in the seed alone. Arc types are drawn independently, so
    raising --per-type extends a type's sample rather than reshuffling it.
    """
    arcs = collect_arcs(syn, elig)
    cands = doc_candidates(elig)
    head, docs = syn["head"], syn["docs"]
    used = set()
    rows, per_type_counts, unmatched = [], {}, []
    pid = 0
    for d in ARC_DEPS:
        pool = sorted(arcs.get(d, []),
                      key=lambda a: (a["doc_id"], a["span_a"], a["span_b"]))
        rng = random.Random(f"arc {seed} {d}")
        take = pool if len(pool) <= per_type else rng.sample(pool, per_type)
        take = sorted(take, key=lambda a: (a["doc_id"], a["span_a"],
                                           a["span_b"]))
        per_type_counts[d] = {"available": len(pool), "sampled": len(take)}
        crng = random.Random(f"control {seed} {d}")
        for arc in take:
            pid += 1
            arc_id = pid
            first = arc["span_a"] if arc["dep_first"] else arc["span_b"]
            second = arc["span_b"] if arc["dep_first"] else arc["span_a"]
            used.add((arc["doc_id"], first, second))
            rows.append({
                "pair_id": arc_id, "kind": "arc", "dep": d,
                "doc_id": arc["doc_id"],
                "span_a": arc["span_a"], "span_b": arc["span_b"],
                "span_first": first, "span_second": second,
                "distance": arc["distance"],
                "pos_first": arc["combo"][0], "pos_second": arc["combo"][1],
                "pos_a": arc["pos_a"], "pos_b": arc["pos_b"],
                "text_a": arc["text_a"], "text_b": arc["text_b"],
                "dep_first": arc["dep_first"],
                "match_of": None, "match_quality": "arc"})
            got = find_control(arc, cands, head, docs, used, crng)
            if got is None:
                unmatched.append(arc_id)
                continue
            u, v, quality, dist, cdoc = got
            used.add((cdoc, u, v))
            pid += 1
            # span_a of a control plays the positional role the arc's dependent
            # played, so the POS combination lines up term for term
            ca, cb = (u, v) if arc["dep_first"] else (v, u)
            rows.append({
                "pair_id": pid, "kind": "control", "dep": d,
                "doc_id": cdoc,
                "span_a": ca, "span_b": cb,
                "span_first": u, "span_second": v,
                "distance": dist,
                "pos_first": elig[u]["pos"], "pos_second": elig[v]["pos"],
                "pos_a": elig[ca]["pos"], "pos_b": elig[cb]["pos"],
                "text_a": elig[ca]["text"], "text_b": elig[cb]["text"],
                "dep_first": arc["dep_first"],
                "match_of": arc_id, "match_quality": quality})
    # the run writes one variant per (pair, draw) and the analysis reads a
    # variant back by the two span ids it substituted, so two pairs over the
    # same two spans would be indistinguishable there
    seen = {(r["doc_id"], r["span_first"], r["span_second"]) for r in rows}
    if len(seen) != len(rows):
        raise SystemExit("two pairs cover the same two spans")
    return rows, per_type_counts, unmatched


# ------------------------------------------------------------------- planning

def draw_substitutes(info, index, seed, draws):
    """This span's substitute for each draw, from a stream seeded on the span.

    Seeding on the span id and not on the document's position in the run is what
    makes a substitute a property of (span, draw): the same string is spliced for
    "a alone" at draw k and for "both" at draw k, whichever pair asked for it.
    """
    cands = candidates_for(index, info["pos"], info["has_space"], info["case"],
                           info["n_qwen"], info["text"], info["doc_id"])
    rng = random.Random(f"pair-sub {seed} {info['span_id']}")
    return [cands[rng.randrange(len(cands))] for _ in range(draws)], \
        (1.0 / len(cands) if cands else None)


def plan_document(text, words, span_ids, pairs, elig, index, seed, draws):
    """Every edited string this document contributes, and what each one is.

    Returns (texts, meta, subs_by_span). texts[0] is the intact document; meta is
    aligned with texts[1:] and each entry is the substitution list
    dbio.write_variants wants. Singles come first, then the joint edits.

    No model and no torch, so the whole plan is checkable on a laptop.
    """
    spans = [(w["start"], w["end"]) for w in words]
    by_sid = {s: k for k, s in enumerate(span_ids)}
    order = sorted({p[key] for p in pairs for key in ("span_a", "span_b")})
    subs_by_span, probs = {}, {}
    for sid in order:
        subs_by_span[sid], probs[sid] = draw_substitutes(elig[sid], index,
                                                         seed, draws)

    texts, meta = [text], []
    for sid in order:
        k = by_sid[sid]
        for i in range(draws):
            w = subs_by_span[sid][i]
            texts.append(T.splice(text, spans, k,
                                  T.fit_space(text, spans[k], w))[0])
            meta.append([{"span_id": sid, "substitute": w, "source": SWAP,
                          "depth": 1, "draw_idx": i, "prob": probs[sid]}])

    for p in sorted(pairs, key=lambda x: x["pair_id"]):
        ka, kb = by_sid[p["span_a"]], by_sid[p["span_b"]]
        for i in range(draws):
            wa, wb = subs_by_span[p["span_a"]][i], subs_by_span[p["span_b"]][i]
            joint = {ka: T.fit_space(text, spans[ka], wa),
                     kb: T.fit_space(text, spans[kb], wb)}
            texts.append(T.splice_many(text, spans, joint)[0])
            meta.append([
                {"span_id": p["span_a"], "substitute": wa, "source": SWAP,
                 "depth": 2, "draw_idx": i, "prob": probs[p["span_a"]]},
                {"span_id": p["span_b"], "substitute": wb, "source": SWAP,
                 "depth": 2, "draw_idx": i, "prob": probs[p["span_b"]]}])
    return texts, meta, subs_by_span


# -------------------------------------------------------------- plan checking

def check_plan(text, words, span_ids, pairs, texts, meta, subs_by_span, qtok,
               cache, draws):
    """Round-trip every planned edit against the original. Returns the failures.

    A single is re-spliced by hand and compared; a joint edit is compared against
    applying its two singles one after the other, which is the property
    splice_many exists to provide. Qwen length parity is checked on the string
    that is actually spliced, spaced form included.
    """
    spans = [(w["start"], w["end"]) for w in words]
    by_sid = {s: k for k, s in enumerate(span_ids)}
    bad = []

    def one(k, word):
        a, b = spans[k]
        has_space = a < len(text) and text[a] == " "
        s = (" " + word) if has_space else word
        return text[:a] + s + text[b:], has_space

    if texts[0] != text:
        bad.append(("baseline is not the intact document", None))
    for t, m in zip(texts[1:], meta):
        if len(m) == 1:
            k = by_sid[m[0]["span_id"]]
            want, has_space = one(k, m[0]["substitute"])
            if t != want:
                bad.append(("single splice", m[0]["substitute"]))
            orig = words[k]["text"]
            if m[0]["substitute"].lower() == orig.lower():
                bad.append(("substitute is the original", orig))
            # the invariant is that the substitute has been put through
            # apply_case under the original's shape, not that reading its case
            # back gives that shape: case_of does not round-trip through
            # apply_case for a digit-initial word or a one-letter substitute
            # for an all-caps original, and that is 04's behaviour, inherited
            sub = m[0]["substitute"]
            if sub != S.apply_case(sub, S.case_of(orig)):
                bad.append(("case", sub, orig))
            if (S.qwen_len(qtok, cache, m[0]["substitute"], has_space)
                    != S.qwen_len(qtok, cache, orig, has_space)):
                bad.append(("qwen length", m[0]["substitute"], orig))
        else:
            ka, kb = by_sid[m[0]["span_id"]], by_sid[m[1]["span_id"]]
            if ka == kb:
                bad.append(("a pair edits one span twice", m[0]["span_id"]))
                continue
            lo, hi = (ka, kb) if ka < kb else (kb, ka)
            wlo = m[0]["substitute"] if ka < kb else m[1]["substitute"]
            whi = m[1]["substitute"] if ka < kb else m[0]["substitute"]
            # apply the later edit first so the earlier span's offsets still hold
            step, _ = one(hi, whi)
            a, b = spans[lo]
            has_space = a < len(step) and step[a] == " "
            want = step[:a] + ((" " + wlo) if has_space else wlo) + step[b:]
            if t != want:
                bad.append(("joint splice", wlo, whi))

    # common random numbers: the string a joint edit splices at a span, at draw
    # k, is the one the single for that span at draw k spliced. Read back off
    # the plan rather than off the generator, so a planning bug cannot hide.
    single = {}
    for m in meta:
        if len(m) == 1:
            single[(m[0]["span_id"], m[0]["draw_idx"])] = m[0]["substitute"]
    for m in meta:
        if len(m) != 2:
            continue
        for s in m:
            got = single.get((s["span_id"], s["draw_idx"]))
            if got is None:
                bad.append(("joint edit with no matching single", s["span_id"]))
            elif got != s["substitute"]:
                bad.append(("common random numbers broken", s["span_id"],
                            got, s["substitute"]))
    for p in pairs:
        for key in ("span_a", "span_b"):
            if len(subs_by_span[p[key]]) != draws:
                bad.append(("wrong number of draws", p[key]))
    return bad


# ------------------------------------------------------------------- reporting

def plan_report(syn, elig, rejected, census, per_type_counts, rows, unmatched,
                budget, dist_by_type, sharing, pool_stats, args):
    """The whole dry-run report as text, printed and written to results/."""
    L = []
    P = L.append
    P("# Dependent-pair interaction plan")
    P("")
    P(f"seed {args.seed}, draws per pair {args.draws}, "
      f"per-type cap {args.per_type}, minimum token distance {MIN_DISTANCE}")
    P(f"documents in the store {len(syn['docs'])}, spans {len(syn['spans'])}, "
      f"head relations {len(syn['head'])}")
    P(f"swap pool {pool_stats['occurrences']} occurrences, "
      f"{pool_stats['types']} word types, {pool_stats['docs']} documents, "
      f"{pool_stats['cells']} (pos, parity) cells")
    P("")
    P("## Span eligibility")
    P("")
    P(f"{len(elig)} of {len(syn['spans'])} spans can be served by the swap pool.")
    for k, v in rejected.most_common():
        P(f"  rejected, {k}: {v}")
    P("")
    P("## Arcs per type")
    P("")
    P(f"  {'dep':8s} {'arcs':>6s} {'endpoint':>9s} {'contig':>7s} "
      f"{'available':>10s} {'sampled':>8s} {'controls':>9s}")
    for d in ARC_DEPS:
        c = census[d]
        n_ctrl = sum(1 for r in rows if r["kind"] == "control" and r["dep"] == d)
        P(f"  {d:8s} {c['arcs']:6d} {c['endpoint not eligible']:9d} "
          f"{c['contiguous']:7d} {per_type_counts[d]['available']:10d} "
          f"{per_type_counts[d]['sampled']:8d} {n_ctrl:9d}")
    n_arc = sum(1 for r in rows if r["kind"] == "arc")
    n_ctrl = sum(1 for r in rows if r["kind"] == "control")
    P(f"  {'total':8s} {'':6s} {'':9s} {'':7s} "
      f"{sum(v['available'] for v in per_type_counts.values()):10d} "
      f"{n_arc:8d} {n_ctrl:9d}")
    P("")
    P("Columns: arcs of this dep type in the store; how many lost an endpoint "
      "the swap pool cannot serve; how many were contiguous; what remained "
      "available; how many were sampled; how many of those got a control.")
    P("")
    P("## Control match quality")
    P("")
    q = Counter(r["match_quality"] for r in rows if r["kind"] == "control")
    for k in QUALITIES:
        if q[k]:
            P(f"  {k:14s} {q[k]:5d}  {100.0 * q[k] / max(1, n_ctrl):5.1f}%")
    P(f"  {'unmatched':14s} {len(unmatched):5d}  "
      f"{100.0 * len(unmatched) / max(1, n_arc):5.1f}% of sampled arcs")
    P("")
    P("  quality by type")
    P(f"  {'dep':8s} " + " ".join(f"{k:>13s}" for k in QUALITIES)
      + f" {'unmatched':>10s}")
    for d in ARC_DEPS:
        cq = Counter(r["match_quality"] for r in rows
                     if r["kind"] == "control" and r["dep"] == d)
        miss = per_type_counts[d]["sampled"] - sum(cq.values())
        P(f"  {d:8s} " + " ".join(f"{cq[k]:13d}" for k in QUALITIES)
          + f" {miss:10d}")
    P("")
    P("## Token distance")
    P("")
    P(f"  {'dep':8s} {'n':>5s} {'min':>5s} {'p25':>5s} {'median':>7s} "
      f"{'p75':>5s} {'p90':>5s} {'max':>5s} {'mean':>6s}")
    for key in list(ARC_DEPS) + ["ARC (all)", "CONTROL (all)"]:
        v = sorted(dist_by_type.get(key, []))
        if not v:
            continue
        pick = lambda f: v[min(len(v) - 1, int(f * len(v)))]
        P(f"  {key:8s} {len(v):5d} {v[0]:5d} {pick(0.25):5d} "
          f"{pick(0.5):7d} {pick(0.75):5d} {pick(0.9):5d} {v[-1]:5d} "
          f"{sum(v) / len(v):6.2f}")
    P("")
    P("## Sharing")
    P("")
    P(f"pairs {len(rows)}, distinct spans in them {sharing['spans']}, "
      f"span slots {2 * len(rows)}")
    P(f"a span appears in {sharing['mean']:.2f} pairs on average, "
      f"at most {sharing['max']}")
    P(f"singles planned {sharing['spans'] * args.draws}, against "
      f"{2 * len(rows) * args.draws} if no single were shared: a saving of "
      f"{2 * len(rows) * args.draws - sharing['spans'] * args.draws} passes")
    P("")
    P("## Pass budget")
    P("")
    P(f"  {'baselines':22s} {budget['baselines']:8d}   one per document")
    P(f"  {'single-span passes':22s} {budget['singles']:8d}   "
      f"{sharing['spans']} spans x {args.draws} draws")
    P(f"  {'joint passes':22s} {budget['pairs']:8d}   "
      f"{len(rows)} pairs x {args.draws} draws")
    P(f"  {'total':22s} {budget['total']:8d}")
    P("")
    P(f"documents planned {budget['docs']}, skipped "
      f"{budget['skipped_docs']} (no trace row, or the prompt is over "
      f"{'the EasyNLA cap' if budget['skipped_docs'] else 'the cap'})")
    P("")
    P("## Sign convention")
    P("")
    P("interaction = e(a) + e(b) - e(both), in FVE points, where e is the drop "
      "in fraction of variance explained against the same document's unedited "
      "baseline, times 100. A POSITIVE interaction means the pair costs LESS "
      "than the sum of its two singles.")
    return "\n".join(L)


# ---------------------------------------------------------------------- main

def open_ro(path):
    import sqlite3
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ar", default=None,
                    help="the AR reconstructor directory. Not needed with "
                         "--dry-run, which uses the Qwen tokeniser alone")
    ap.add_argument("--traces",
                    default="../01_corpus-and-spans/results/"
                            "ffw_pilot_traces.parquet")
    ap.add_argument("--db", default=str(dbio.DEFAULT_DB))
    ap.add_argument("--docs", default="all",
                    help="'all' for every document in the store, or a "
                         "comma-separated list of doc ids")
    ap.add_argument("--draws", type=int, default=8)
    ap.add_argument("--per-type", type=int, default=150)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--precision", default="bf16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / "results"))
    ap.add_argument("--dry-run", action="store_true",
                    help="plan and check every edit, count the passes and "
                         "write the pair table. No reconstructor, no GPU")
    ap.add_argument("--migrate", action="store_true",
                    help="apply outstanding migrations to --db first. Off by "
                         "default: a schema change is Marty's call")
    ap.add_argument("--resume-run", type=int, default=None,
                    help="append to this existing run, skipping documents it "
                         "has already measured")
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        conn = open_ro(args.db)
    else:
        conn = dbio.open_db(args.db, allow_migrate=args.migrate, needs=4)

    # ------------------------------------------------------------- tokeniser
    from transformers import AutoTokenizer
    if args.ar and (Path(args.ar) / "tokenizer_config.json").is_file():
        qtok = AutoTokenizer.from_pretrained(args.ar)
    else:
        if not args.dry_run:
            raise SystemExit("--ar must point at the reconstructor for a real run")
        qtok = AutoTokenizer.from_pretrained(T.QWEN)

    cache = {}
    t0 = time.perf_counter()
    pool = S.build_pool(conn, qtok, cache)
    pool_stats = {
        "occurrences": sum(len(v) for v in pool.values()),
        "types": len({w.lower() for v in pool.values() for _, w, _ in v}),
        "docs": len({d for v in pool.values() for d, _, _ in v}),
        "cells": len(pool)}
    index = index_pool(pool, qtok, cache)
    print(f"pool {pool_stats['occurrences']} occurrences, "
          f"{pool_stats['types']} types, {pool_stats['docs']} documents "
          f"({time.perf_counter() - t0:.1f}s)", flush=True)

    syn = load_syntax(conn)
    elig, rejected = eligible_spans(syn, index, qtok, cache)
    print(f"eligible spans {len(elig)}/{len(syn['spans'])} "
          f"({time.perf_counter() - t0:.1f}s)", flush=True)

    census = arc_census(syn, elig)
    rows, per_type_counts, unmatched = sample_pairs(syn, elig, args.per_type,
                                                    args.seed)
    print(f"pairs {len(rows)} ({time.perf_counter() - t0:.1f}s)", flush=True)

    # ------------------------------------------------------------- documents
    if args.docs == "all":
        want_docs = sorted(syn["docs"])
    else:
        want_docs = [int(d) for d in args.docs.split(",") if d.strip()]
    rows = [r for r in rows if r["doc_id"] in set(want_docs)]

    dist_by_type = defaultdict(list)
    for r in rows:
        if r["kind"] == "arc":
            dist_by_type[r["dep"]].append(r["distance"])
            dist_by_type["ARC (all)"].append(r["distance"])
        else:
            dist_by_type["CONTROL (all)"].append(r["distance"])

    span_use = Counter()
    for r in rows:
        span_use[r["span_a"]] += 1
        span_use[r["span_b"]] += 1
    sharing = {"spans": len(span_use),
               "mean": (sum(span_use.values()) / len(span_use)) if span_use else 0,
               "max": max(span_use.values()) if span_use else 0}

    pairs_by_doc = defaultdict(list)
    for r in rows:
        pairs_by_doc[r["doc_id"]].append(r)

    # ------------------------------------------------------------------ spaCy
    import spacy
    nlp = spacy.load("en_core_web_sm")
    scheme = dbio.spacy_scheme(nlp)
    if scheme != SPACY:
        print(f"WARNING: spaCy here is {scheme}, the store's labels are "
              f"{SPACY}", flush=True)

    # ------------------------------------------------------- the reconstructor
    model = tok = headm = E = None
    by_doc_trace = {}
    rawvar = None
    if not args.dry_run:
        import torch
        import pyarrow.parquet as pq
        from harness import (MAX_PROMPT, make_deterministic, fingerprint,
                             load_ar, mse_of_texts)
        from extract_traces import MSE_SCALE, normalize_activation
        torch.set_num_threads(args.threads)
        make_deterministic(args.seed)
        fp = fingerprint()
        print(json.dumps(fp), flush=True)
        model, tok, headm = load_ar(args.ar, args.device, args.precision)
        E = model.get_input_embeddings().weight
        trace_rows = pq.read_table(args.traces).to_pylist()
        by_doc_trace = {int(r["doc_uid"]): r for r in trace_rows}
        G = torch.tensor([r["activation"] for r in trace_rows],
                         dtype=torch.float64)
        Gn = G / G.norm(dim=1, keepdim=True) * MSE_SCALE
        rawvar = float(((Gn - Gn.mean(0)) ** 2).mean())
        print(f"mse_rawvar {rawvar:.5f}  dFVE = -dMSE / rawvar", flush=True)
    else:
        from textsub import prompt_ids  # noqa: F401
        MAX_PROMPT = 1024
        try:
            import pyarrow.parquet as pq
            by_doc_trace = {int(r["doc_uid"]): True for r in
                            pq.read_table(args.traces,
                                          columns=["doc_uid"]).to_pylist()}
        except Exception as e:
            print(f"note: traces not readable here ({e.__class__.__name__}), "
                  f"the plan assumes every document has a trace row",
                  flush=True)
            by_doc_trace = {d: True for d in syn["docs"]}

    # --------------------------------------------------------------- planning
    plans, skipped_docs, bad = {}, [], []
    n_single = n_joint = 0
    have = {(d, a, b): s for s, (d, a, b) in syn["spans"].items()}
    for doc_id in want_docs:
        prs = pairs_by_doc.get(doc_id)
        if not prs:
            continue
        if doc_id not in by_doc_trace:
            skipped_docs.append((doc_id, "no trace row"))
            continue
        text = syn["docs"][doc_id]["text"]
        if len(T.prompt_ids(qtok, text)) > MAX_PROMPT:
            skipped_docs.append((doc_id, "prompt over the cap"))
            continue
        words = T.word_spans(nlp, text, lexical_only=True)
        keys = [(doc_id,) + dbio.bare_span(w) for w in words]
        sids = [have.get(k) for k in keys]
        missing = [p for p in prs
                   if p["span_a"] not in sids or p["span_b"] not in sids]
        if missing:
            bad.append((doc_id, f"{len(missing)} pairs name a span spaCy did "
                                f"not reproduce here"))
            keep = {p["pair_id"] for p in prs} - {p["pair_id"] for p in missing}
            prs = [p for p in prs if p["pair_id"] in keep]
            if not prs:
                skipped_docs.append((doc_id, "no pair survived re-parsing"))
                continue
        texts, meta, subs_by_span = plan_document(
            text, words, sids, prs, elig, index, args.seed, args.draws)
        plans[doc_id] = (text, words, sids, prs, texts, meta, subs_by_span)
        n_single += sum(1 for m in meta if len(m) == 1)
        n_joint += sum(1 for m in meta if len(m) == 2)

    budget = {"baselines": len(plans), "singles": n_single, "pairs": n_joint,
              "total": len(plans) + n_single + n_joint,
              "docs": len(plans), "skipped_docs": len(skipped_docs)}

    # ------------------------------------------------------------- pair table
    kept = {p["pair_id"] for d in plans for p in plans[d][3]}
    table = [dict(r, planned=(r["pair_id"] in kept)) for r in rows]

    # ---------------------------------------------------------------- dry run
    if args.dry_run:
        t = time.perf_counter()
        fails, shapes = [], []
        for doc_id, (text, words, sids, prs, texts, meta, sbs) in plans.items():
            f = check_plan(text, words, sids, prs, texts, meta, sbs, qtok,
                           cache, args.draws)
            if f:
                fails += [(doc_id,) + tuple(x) for x in f[:4]]
            by_sid = {s: k for k, s in enumerate(sids)}
            for m in meta:
                if len(m) != 1:
                    continue
                orig = words[by_sid[m[0]["span_id"]]]["text"]
                if S.case_of(m[0]["substitute"]) != S.case_of(orig):
                    shapes.append((doc_id, orig, m[0]["substitute"]))
        report = plan_report(syn, elig, rejected, census, per_type_counts,
                             rows, unmatched, budget, dist_by_type, sharing,
                             pool_stats, args)
        report += "\n\n## Plan verification\n\n"
        report += (f"every planned edit round-tripped against the original "
                   f"({budget['singles'] + budget['pairs']} edits over "
                   f"{len(plans)} documents, {time.perf_counter() - t:.1f}s)"
                   if not fails else
                   "FAILURES:\n" + "\n".join(f"  {x}" for x in fails[:20]))
        report += (
            f"\n\ncapitalisation shape read back differs from the original's "
            f"on {len(shapes)} of {budget['singles']} single edits, on "
            f"{len({(d, o) for d, o, _ in shapes})} distinct originals. "
            f"apply_case cannot change the shape of a digit-initial word, and "
            f"a one-letter substitute for an all-caps original reads as title "
            f"case. This is 04's behaviour, inherited unchanged. Examples: "
            + ", ".join(f"{o!r} -> {s!r}" for _, o, s in shapes[:6]))
        if bad:
            report += "\n\nDocuments with a problem:\n" + "\n".join(
                f"  doc {d}: {m}" for d, m in bad)
        if skipped_docs:
            report += "\n\nSkipped documents:\n" + "\n".join(
                f"  doc {d}: {m}" for d, m in skipped_docs)
        report += "\n"
        (out / "plan.md").write_text(report)
        print()
        print(report)
        print(f"plan: {out / 'plan.md'}")
        return 1 if fails else 0

    # ------------------------------------------------------------- the GPU run
    import torch  # noqa: F401
    from harness import mse_of_texts
    from extract_traces import MSE_SCALE, normalize_activation

    skip = set()
    if args.resume_run is not None:
        row = conn.execute("SELECT script FROM runs WHERE run_id = ?",
                           (int(args.resume_run),)).fetchone()
        if row is None:
            raise SystemExit(f"no run {args.resume_run} to resume")
        if "pair_ablation" not in (row["script"] or ""):
            raise SystemExit(f"run {args.resume_run} is {row['script']}, "
                             "not a pair_ablation run")
        run_id = int(args.resume_run)
        skip = S.already_done(conn, run_id)
        print(f"resuming run {run_id}; {len(skip)} documents already measured",
              flush=True)
    else:
        run_id = DB.new_run(
            conn, script="05_dependent-pair/pair_ablation.py", assets=ASSETS,
            notes=args.notes or (
                "Dependent-pair interaction. Two lexical words are corpus-"
                "swapped singly and together, so the interaction e(a) + e(b) - "
                "e(both) is measurable per draw. Pairs are direct dependency "
                "arcs of nine dep types with the two words at least two tokens "
                "apart, each matched to a control pair from the same document "
                "with the same ordered POS combination and the same token "
                "distance, no arc between them either way and no path of length "
                "two through a shared head or head-of-head. Common random "
                "numbers: a span's substitute is a property of the span and the "
                "draw, so the single and the joint edit splice the same string "
                "and a span shared by several pairs contributes one set of "
                "singles. The baseline rides in the same batched call as the "
                "arms."),
            config={"args": vars(args), "fingerprint": fp,
                    "mse_rawvar": rawvar, "swap_scheme": SWAP,
                    "pair_scheme": PAIR_SCHEME, "spacy_scheme": scheme,
                    "arc_deps": ARC_DEPS, "min_distance": MIN_DISTANCE,
                    "pool": pool_stats, "budget": budget, "sharing": sharing,
                    "per_type_counts": per_type_counts,
                    "unmatched_arcs": unmatched,
                    "sign": "interaction = e(a) + e(b) - e(both), FVE points; "
                            "positive means the pair costs less than the sum "
                            "of its singles",
                    "pair_columns": ["pair_id", "kind", "dep", "doc_id",
                                     "span_a", "span_b", "distance",
                                     "pos_a", "pos_b", "match_of",
                                     "match_quality", "planned"],
                    "pairs": [[p["pair_id"], p["kind"], p["dep"], p["doc_id"],
                               p["span_a"], p["span_b"], p["distance"],
                               p["pos_a"], p["pos_b"], p["match_of"],
                               p["match_quality"], p["planned"]]
                              for p in table]})
        conn.commit()
        # the pair set, recoverable from the store alone. relations already means
        # "an edge between two spans under a scheme", so a scheme of its own
        # carries the arc and control pairs without a migration. What it cannot
        # carry is which arc a control was matched to, or the match quality;
        # those are in the runs row's config
        with DB.transaction(conn):
            DB.add_relations(conn, [
                (PAIR_SCHEME, int(p["span_a"]), int(p["span_b"]),
                 f"arc:{p['dep']}" if p["kind"] == "arc" else "control")
                for p in table if p["planned"]])
    print(f"run_id {run_id}", flush=True)
    print(f"planned passes {budget['total']} over {budget['docs']} documents",
          flush=True)

    t0 = time.perf_counter()
    n_docs = n_var = 0
    for doc_id in sorted(plans):
        if doc_id in skip:
            print(f"  doc {doc_id}: already measured under run {run_id}, "
                  f"skipped", flush=True)
            continue
        text, words, sids, prs, texts, meta, _ = plans[doc_id]
        trace = by_doc_trace[doc_id]
        gold = torch.tensor(trace["activation"], dtype=torch.float32,
                            device=args.device)
        gold_n = normalize_activation(gold.unsqueeze(0), MSE_SCALE)[0]
        dbio.ensure_spans(conn, doc_id, text, words, scheme,
                          source="ffw_main_traces")

        vals, lens = mse_of_texts(model, tok, headm, E, gold_n, texts,
                                  args.batch, args.device)
        mse = [float(v) for v in vals]
        records = [([], {"mse": mse[0], "fve": 1 - mse[0] / rawvar,
                         "seq_len": lens[0], "traces_mse": trace["mse"]})]
        for j, subs in enumerate(meta, start=1):
            records.append((subs, {"mse": mse[j], "fve": 1 - mse[j] / rawvar,
                                   "seq_len": lens[j]}))
        dbio.write_variants(conn, doc_id, run_id, records)
        n_docs += 1
        n_var += len(records)
        print(f"  doc {doc_id}  {len(prs)} pairs  {len(records)} variants  "
              f"base FVE {1 - mse[0] / rawvar:.4f}  "
              f"{time.perf_counter() - t0:.0f}s elapsed", flush=True)

    print(f"\nrun {run_id}: {n_docs} docs, {n_var} variants "
          f"({time.perf_counter() - t0:.0f}s)")
    for k, v in DB.counts(conn).items():
        print(f"  {k:16s} {v}")
    conn.execute("PRAGMA optimize")
    conn.close()
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
