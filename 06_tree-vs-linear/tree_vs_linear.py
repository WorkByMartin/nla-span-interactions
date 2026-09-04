#!/usr/bin/env python3
"""Full pairwise interaction matrix over every eligible word of a few documents.

    python tree_vs_linear.py --dry-run --n-docs 5 --seed 0 \
        --db ../db/ffw_span-ablation_database.sqlite

    python tree_vs_linear.py --ar "$AR" \
        --traces ../01_corpus-and-spans/results/ffw_pilot_traces.parquet \
        --db ../db/ffw_span-ablation_database.sqlite \
        --n-docs 5 --seed 0 --draws 8 --batch 16 --precision bf16 --threads 8

The edit, the eligibility test, the pool, the splice, the common random numbers
and the database writes are 05's, imported from pair_ablation rather than
restated. What differs is the unit: 05 sampled dependency arcs and matched
controls, and this measures EVERY unordered pair of eligible words in a
document, so the result is a complete n by n interaction matrix rather than a
sample of cells.

Per document the passes are

    1 baseline + n x draws singles + C(n, 2) x draws joint edits

and a span's substitute at draw k is a property of the span and the draw, so the
string spliced at that span is the same in its single and in all n - 1 joint
edits it takes part in. A document runs to completion at the full draw count and
is committed before the next one starts, so --resume-run skips whole documents.

Every unordered pair is measured, adjacent pairs included, and nothing is
filtered at run time. The prompt length change an edit causes is recorded per
variant as `dtok` so the analysis can separate a pair whose splice moved a token
boundary; only a splice that fails to round-trip against the original is an
error.

--dry-run does everything except the forward pass, including every splice
round-trip check, and writes the budget, the eligibility census and the prompt
length census to results/plan.md.
"""
import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

# pair_ablation finds its own siblings, so importing it is enough to put
# textsub, swap_ablation, db and dbio on the path as well. On the pod every
# module sits flat in one directory, which is the fallback its _package() takes.
for _c in (HERE, REPO / "05_dependent-pair"):
    if (_c / "pair_ablation.py").is_file():
        sys.path.insert(0, str(_c))
        break
else:
    raise SystemExit("cannot find pair_ablation.py beside this file or in "
                     "05_dependent-pair")

import pair_ablation as P  # noqa: E402
import textsub as T  # noqa: E402
import swap_ablation as S  # noqa: E402
import db as DB  # noqa: E402
import dbio  # noqa: E402

SWAP = P.SWAP
SPACY = P.SPACY
ASSETS = P.ASSETS
PAIR_SCHEME = "tree-vs-linear/all-pairs"

# harness.MAX_PROMPT, restated so the dry run needs neither torch nor the
# reconstructor. The real run asserts the two agree.
MAX_PROMPT = 1024

# document filters applied before the random draw
MAX_CJK = 0.05          # a mostly non-Latin verbalisation is a different object
MIN_ELIGIBLE = 0.90     # of the document's clean lexical words

# 05 measured this on an A100 80GB PCIe at batch 8, bf16, over its own
# documents. It is a projection here, not a measurement of this experiment.
REF_PASSES_PER_SEC = 12.1
REF_NOTE = ("projection from 05's measured 12.1 passes/s on an A100 80GB PCIe "
            "at batch 8")

VEC = "<f4"             # docs.activation, float32 little-endian, 5120 floats


# ---------------------------------------------------------------- the gold

def measurable(conn):
    """{doc_id: the MSE recorded at extraction} for documents that carry a gold.

    The store now holds the layer-42 activation of every verbalised document, so
    the gold comes from there rather than from the pilot traces parquet, which
    covers only the first hundred.
    """
    return {int(r["doc_id"]): r["mse"] for r in conn.execute(
        "SELECT doc_id, mse FROM docs WHERE activation IS NOT NULL")}


def gold_vector(conn, doc_id):
    import numpy as np
    row = conn.execute("SELECT activation FROM docs WHERE doc_id = ?",
                       (int(doc_id),)).fetchone()
    return np.frombuffer(row["activation"], dtype=VEC).copy()


def raw_variance(conn, traces, want_source="traces"):
    """The FVE denominator, and where it came from.

    05 computed it over the hundred rows of the pilot traces parquet, and a
    change of denominator would rescale every interaction, so the parquet stays
    the default even though the store now holds far more documents. Both values
    are computed when both are available, and both are recorded.
    """
    import numpy as np
    out = {}
    try:
        import pyarrow.parquet as pq
        rows = pq.read_table(traces, columns=["activation"]).to_pylist()
        G = np.asarray([r["activation"] for r in rows], dtype=np.float64)
        out["traces"] = (G, len(rows))
    except Exception as e:
        print(f"note: the traces parquet is not readable "
              f"({e.__class__.__name__}: {e})", flush=True)
    blobs = [np.frombuffer(r["activation"], dtype=VEC) for r in conn.execute(
        "SELECT activation FROM docs WHERE activation IS NOT NULL")]
    if blobs:
        out["store"] = (np.asarray(blobs, dtype=np.float64), len(blobs))
    vals = {}
    from extract_traces import MSE_SCALE
    for k, (G, n) in out.items():
        Gn = G / np.linalg.norm(G, axis=1, keepdims=True) * MSE_SCALE
        vals[k] = (float(((Gn - Gn.mean(0)) ** 2).mean()), n)
    src = want_source if want_source in vals else next(iter(vals))
    return vals[src][0], src, vals


# ------------------------------------------------------------ document choice

def spans_of(syn, doc_id):
    """This document's spans, from a bucketing built once for the whole store.

    The store holds a span per token of every verbalised document, so grouping
    has to happen once rather than per document: rebuilding the filter for each
    of a thousand documents is quadratic in the corpus.
    """
    if "by_doc" not in syn:
        b = {}
        for sid, v in syn["spans"].items():
            b.setdefault(v[0], {})[sid] = v
        syn["by_doc"] = b
    return syn["by_doc"].get(doc_id, {})


def doc_eligibility(syn, index, qtok, cache, doc_id):
    """(eligible spans, rejection census) for one document.

    P.eligible_spans over a view of the store restricted to this document, so
    the test is the same one 05 applies and the reasons come back per document
    rather than pooled over the corpus.
    """
    return P.eligible_spans(dict(syn, spans=spans_of(syn, doc_id)), index,
                            qtok, cache)


def screen(syn, index, qtok, cache):
    """Every document, its CJK fraction and what share of its words are usable.

    The denominator is the document's CLEAN LEXICAL words, so the ratio measures
    the swap pool's reach over the words this experiment could otherwise edit,
    and is not diluted by punctuation.
    """
    out = {}
    for doc_id in sorted(syn["docs"]):
        elig, rej = doc_eligibility(syn, index, qtok, cache, doc_id)
        clean = len(elig) + rej["empty swap pool"]
        out[doc_id] = {
            "doc_id": doc_id, "eligible": len(elig), "clean_lexical": clean,
            "rate": (len(elig) / clean) if clean else 0.0,
            "cjk": float(syn["docs"][doc_id].get("cjk") or 0.0),
            "rejected": rej}
    return out


def select_docs(syn, scr, args):
    """The documents to measure, and every document excluded, with the reason.

    The draw is uniform over the documents that survive the filters and never
    looks at the text, so nothing about length enters the selection.
    """
    excluded, ok = [], []
    for doc_id, s in sorted(scr.items()):
        why = []
        if s["cjk"] > MAX_CJK:
            why.append(f"cjk fraction {s['cjk']:.3f} over {MAX_CJK}")
        if s["rate"] < MIN_ELIGIBLE:
            why.append(f"only {s['rate']:.3f} of its {s['clean_lexical']} "
                       f"clean lexical words are eligible, under "
                       f"{MIN_ELIGIBLE}")
        (excluded.append((doc_id, "; ".join(why))) if why
         else ok.append(doc_id))
    if args.docs:
        want = [int(d) for d in args.docs.replace(",", " ").split()]
        missing = [d for d in want if d not in syn["docs"]]
        if missing:
            raise SystemExit(f"no such documents in the store: {missing}")
        flagged = [d for d in want if d not in ok]
        how = "named on the command line, so the filters were not applied"
        if flagged:
            how += f"; {flagged} would have been filtered out"
        return want, how, excluded, ok
    if args.n_docs > len(ok):
        raise SystemExit(f"--n-docs {args.n_docs} but only {len(ok)} "
                         f"documents pass the filters")
    rng = random.Random(f"matrix-docs {args.seed}")
    return (sorted(rng.sample(ok, args.n_docs)),
            f"{args.n_docs} of the {len(ok)} documents that pass the filters "
            f"(out of {len(syn['docs'])}), drawn uniformly at random with seed "
            f"{args.seed}, without regard to length",
            excluded, ok)


# ------------------------------------------------------------ the pair set

def categorise(u, v, elig, head, dep):
    """arc, adjacent or other, in that order of precedence.

    A word and its head are often neighbours, so the three would not partition
    the matrix if adjacency were tested first.
    """
    if head.get(u) == v:
        return "arc", dep.get(u)
    if head.get(v) == u:
        return "arc", dep.get(v)
    if abs(elig[u]["tok_i"] - elig[v]["tok_i"]) == 1:
        return "adjacent", None
    return "other", None


def all_pairs(span_ids, elig, head, dep, start_id=1):
    """Every unordered pair of the document's eligible spans, in reading order.

    span_a is whichever word comes first in the document, so a pair is named the
    same way whichever end is asked about, and pair ids are dense and stable
    given the span set. `category` and `adjacent` are carried on the row and
    written to the store, so the analysis never has to re-derive them.
    """
    order = sorted(span_ids, key=lambda s: (elig[s]["tok_i"], s))
    rows, pid = [], start_id
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            u, v = order[i], order[j]
            cat, d = categorise(u, v, elig, head, dep)
            dist = elig[v]["tok_i"] - elig[u]["tok_i"]
            rows.append({"pair_id": pid, "doc_id": elig[u]["doc_id"],
                         "span_a": u, "span_b": v, "distance": dist,
                         "category": cat, "dep": d, "adjacent": dist == 1})
            pid += 1
    return rows


def budget_for(n, draws):
    """1 baseline, n x draws singles, C(n, 2) x draws joint edits."""
    n_pairs = n * (n - 1) // 2
    return {"n": n, "draws": draws, "pairs": n_pairs, "baselines": 1,
            "singles": n * draws, "joints": n_pairs * draws,
            "total": 1 + (n + n_pairs) * draws}


# ------------------------------------------------------------ prompt lengths

def prompt_lengths(qtok, texts, chunk=512):
    """Qwen prompt token count of every planned string, in the plan's order.

    The swap is matched on token count, so these are expected to be flat, but a
    substitute can still shift a boundary. The number is recorded per variant as
    `dtok`, the change against the document's own baseline, rather than being
    turned into a constraint: an adjacent pair or a pair whose splice moves the
    length is a category the analysis can separate afterwards, not a run-time
    filter.
    """
    lens = []
    for i in range(0, len(texts), chunk):
        enc = qtok([T.templated(t) for t in texts[i:i + chunk]],
                   add_special_tokens=False)["input_ids"]
        lens += [len(x) for x in enc]
    return lens


def delta_census(lens, meta):
    """How the prompt length moved, split by single and joint edit."""
    base = lens[0]
    single, joint = Counter(), Counter()
    for j, m in enumerate(meta, start=1):
        (single if len(m) == 1 else joint)[lens[j] - base] += 1
    return {"base": base, "max": max(lens), "n": len(lens),
            "single": single, "joint": joint}


# ------------------------------------------------------------ per document

def prepare(doc_id, syn, index, qtok, cache, nlp, args, have):
    """Plan one document: eligibility, the pair set, and every edited string.

    Returns a dict with the plan and the census, or one carrying `skip`. The
    plan holds one string per forward pass, so it is built when the document is
    reached and dropped once the document is committed.
    """
    elig, rejected = doc_eligibility(syn, index, qtok, cache, doc_id)
    text = syn["docs"][doc_id]["text"]
    n_spans = len(spans_of(syn, doc_id))

    if len(T.prompt_ids(qtok, text)) > MAX_PROMPT:
        return {"doc_id": doc_id, "skip": "prompt over the EasyNLA cap"}

    words = T.word_spans(nlp, text, lexical_only=True)
    keys = [(doc_id,) + dbio.bare_span(w) for w in words]
    sids = [have.get(k) for k in keys]
    present = {s for s in sids if s is not None}

    keep = sorted(s for s in elig
                  if s in present and elig[s]["tok_i"] is not None)
    lost = len(elig) - len(keep)
    if lost:
        rejected["not reproduced by this spaCy parse"] += lost
    if len(keep) < 2:
        return {"doc_id": doc_id, "skip": "fewer than two eligible words"}

    pairs = all_pairs(keep, elig, syn["head"], syn["dep"])
    texts, meta, subs = P.plan_document(text, words, sids, pairs, elig, index,
                                        args.seed, args.draws)
    return {"doc_id": doc_id, "skip": None, "text": text, "words": words,
            "sids": sids, "pairs": pairs, "texts": texts, "meta": meta,
            "subs": subs, "eligible": keep, "rejected": rejected,
            "n_spans": n_spans, "n_words": len(words),
            "categories": Counter(p["category"] for p in pairs),
            "budget": budget_for(len(keep), args.draws)}


def verify(pl, qtok, cache, args):
    """The checks that are errors: the round trip and the size of the plan.

    A splice that does not round-trip against the original is a bug. A prompt
    length that moves is data, so it is counted and recorded, never rejected.
    """
    bad = P.check_plan(pl["text"], pl["words"], pl["sids"], pl["pairs"],
                       pl["texts"], pl["meta"], pl["subs"], qtok, cache,
                       args.draws)
    if len(pl["texts"]) != pl["budget"]["total"]:
        bad.append(("the plan is not the budgeted size", len(pl["texts"]),
                    pl["budget"]["total"]))
    return bad


def census_lines(plan):
    """The per-document eligibility and budget report, as printable lines."""
    b, c = plan["budget"], plan["categories"]
    L = [f"  doc {plan['doc_id']}: {plan['n_spans']} spans in the store, "
         f"{plan['n_words']} lexical words, {b['n']} eligible"]
    for k, v in plan["rejected"].most_common():
        L.append(f"      ineligible, {k}: {v}")
    L.append(f"      pairs {b['pairs']}: {c['arc']} on an arc, "
             f"{c['adjacent']} adjacent without an arc, {c['other']} elsewhere")
    L.append(f"      passes {b['total']} = 1 + {b['n']} x {b['draws']} + "
             f"{b['pairs']} x {b['draws']}")
    return L


# ------------------------------------------------------------ reporting

def projection(total, per_sec=REF_PASSES_PER_SEC):
    s = total / per_sec
    return f"{s / 3600:.2f} h ({s / 60:.0f} min, {total} passes)"


def plan_report(syn, plans, skipped, how, excluded, scr, pool_stats, token,
                n_gold, args):
    L = []
    W = L.append
    W("# Full pairwise interaction matrix, plan")
    W("")
    W(f"seed {args.seed}, draws per pair {args.draws}, batch {args.batch}")
    W(f"documents in the store {len(syn['docs'])}, spans {len(syn['spans'])}, "
      f"head relations {len(syn['head'])}")
    W(f"swap pool {pool_stats['occurrences']} occurrences, "
      f"{pool_stats['types']} word types, {pool_stats['docs']} documents, "
      f"{pool_stats['cells']} (pos, parity) cells")
    W(f"documents carrying a gold activation {n_gold}; the gold is "
      f"docs.activation in the store, not the pilot traces parquet, which "
      f"covers only the first hundred")
    W("")
    W("## Documents")
    W("")
    W(f"selection: {how}")
    W("")
    W(f"  {'doc':>6s} {'store spans':>12s} {'lexical':>8s} {'eligible':>9s} "
      f"{'eligible rate':>14s} {'cjk':>7s} {'pairs':>8s} {'arc':>6s} "
      f"{'adj':>5s} {'other':>7s} {'passes':>9s}")
    for p in plans:
        b, c = p["budget"], p["categories"]
        s = scr[p["doc_id"]]
        W(f"  {p['doc_id']:6d} {p['n_spans']:12d} {p['n_words']:8d} "
          f"{b['n']:9d} {s['rate']:14.4f} {s['cjk']:7.4f} {b['pairs']:8d} "
          f"{c['arc']:6d} {c['adjacent']:5d} {c['other']:7d} {b['total']:9d}")
    total = sum(p["budget"]["total"] for p in plans)
    W(f"  {'total':>6s} {'':12s} {'':8s} "
      f"{sum(p['budget']['n'] for p in plans):9d} {'':14s} {'':7s} "
      f"{sum(p['budget']['pairs'] for p in plans):8d} "
      f"{sum(p['categories']['arc'] for p in plans):6d} "
      f"{sum(p['categories']['adjacent'] for p in plans):5d} "
      f"{sum(p['categories']['other'] for p in plans):7d} {total:9d}")
    W("")
    W("A pair is on an ARC when the store holds a spaCy head relation between "
      "its two words in either direction, ADJACENT when their token indices "
      "differ by one and there is no arc, and elsewhere otherwise. The three "
      "partition the matrix, and each pair's category is written to the store "
      "under the scheme " + PAIR_SCHEME + ".")
    if skipped:
        W("")
        W("skipped after selection:")
        for d, why in skipped:
            W(f"  doc {d}: {why}")
    W("")
    W("## Documents excluded before the draw")
    W("")
    W(f"A document is excluded when its CJK fraction is over {MAX_CJK}, or "
      f"when fewer than {MIN_ELIGIBLE:.0%} of its clean lexical words have a "
      f"swap pool. {len(excluded)} of {len(syn['docs'])} documents were "
      f"excluded.")
    W("")
    if excluded:
        for d, why in excluded:
            W(f"  doc {d}: {why}")
    else:
        W("  none")
    W("")
    rates = sorted(s["rate"] for s in scr.values())
    cjks = sorted(s["cjk"] for s in scr.values())
    W(f"  eligible rate over all {len(rates)} documents: min {rates[0]:.4f}, "
      f"median {rates[len(rates) // 2]:.4f}, max {rates[-1]:.4f}")
    W(f"  cjk fraction over all {len(cjks)} documents: min {cjks[0]:.4f}, "
      f"median {cjks[len(cjks) // 2]:.4f}, max {cjks[-1]:.4f}")
    W("")
    W("## Eligibility, per document")
    W("")
    W("A word is eligible when its spaCy coarse POS is lexical, it passes the "
      "clean-word test, and the swap pool holds at least one word of another "
      "document matching it on POS, leading-space parity and Qwen token count "
      "after recasing. This is 05's test, unchanged.")
    W("")
    for p in plans:
        for line in census_lines(p):
            W(line)
    W("")
    W("## Pass budget")
    W("")
    W(f"  {'baselines':22s} {sum(p['budget']['baselines'] for p in plans):10d}"
      f"   one per document")
    W(f"  {'single-span passes':22s} "
      f"{sum(p['budget']['singles'] for p in plans):10d}   n x draws")
    W(f"  {'joint passes':22s} "
      f"{sum(p['budget']['joints'] for p in plans):10d}   C(n, 2) x draws")
    W(f"  {'total':22s} {total:10d}")
    W("")
    W(f"PROJECTED GPU TIME {projection(total)}")
    W(f"  {REF_NOTE}. Not a measurement of this experiment.")
    W(f"  {'  '.join(str(p['doc_id']) + ': ' + projection(p['budget']['total']) for p in plans)}")
    W("")
    W("A document runs to completion at all "
      f"{args.draws} draws and is committed before the next one starts, so a "
      "run killed in flight leaves whole documents behind and --resume-run "
      "picks up at the next uncommitted document.")
    W("")
    W("## Prompt length")
    W("")
    W("`dtok` is the Qwen prompt token count of an edited document minus the "
      "same document's baseline count, and it is recorded as a metric on every "
      "variant. The swap is matched on token count, so it is expected to be "
      "zero, and where it is not the pair is a category the analysis can "
      "separate rather than a run-time failure. Nothing is filtered on it.")
    W("")
    for d, t in sorted(token.items()):
        W(f"  doc {d}: baseline {t['base']} prompt tokens, longest "
          f"{t['max']}, {t['n']} strings measured")
        W(f"      dtok on singles {dict(sorted(t['single'].items()))}")
        W(f"      dtok on joint edits {dict(sorted(t['joint'].items()))}")
    W("")
    W("## Sign convention")
    W("")
    W("interaction = e(a) + e(b) - e(both), in FVE points, where e is the "
      "drop in fraction of variance explained against the same document's "
      "unedited baseline, times 100. A POSITIVE interaction means the pair "
      "costs LESS than the sum of its two singles, so what the two words "
      "carry overlaps.")
    return "\n".join(L)


# ------------------------------------------------------------ the run

def scored(model, tok, headm, E, gold_n, texts, batch, device, chunk=4096):
    """mse_of_texts in blocks, so a long document reports progress as it goes."""
    from harness import mse_of_texts
    vals, lens = [], []
    t0 = time.perf_counter()
    for i in range(0, len(texts), chunk):
        v, n = mse_of_texts(model, tok, headm, E, gold_n, texts[i:i + chunk],
                            batch, device)
        vals += [float(x) for x in v]
        lens += list(n)
        el = time.perf_counter() - t0
        print(f"      {len(vals)}/{len(texts)} passes, "
              f"{len(vals) / max(el, 1e-9):.1f}/s, {el:.0f}s", flush=True)
    return vals, lens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ar", default=None,
                    help="the AR reconstructor directory. Not needed with "
                         "--dry-run, which uses the Qwen tokeniser alone")
    ap.add_argument("--traces",
                    default="../01_corpus-and-spans/results/"
                            "ffw_pilot_traces.parquet")
    ap.add_argument("--db", default=str(dbio.DEFAULT_DB))
    ap.add_argument("--docs", default="",
                    help="explicit doc ids, comma separated. Overrides "
                         "--n-docs and skips the filters")
    ap.add_argument("--n-docs", type=int, default=5,
                    help="how many documents to draw uniformly at random from "
                         "those that pass the filters")
    ap.add_argument("--draws", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16,
                    help="joint passes are independent, so the batch is a "
                         "memory decision only")
    ap.add_argument("--precision", default="bf16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rawvar-from", default="traces",
                    choices=["traces", "store"],
                    help="the population the FVE denominator is computed over. "
                         "05 used the pilot traces parquet, so that is the "
                         "default and the two runs stay on one scale")
    ap.add_argument("--out", default=str(HERE / "results"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--migrate", action="store_true",
                    help="apply outstanding migrations to --db first. Off by "
                         "default: a schema change is Marty's call")
    ap.add_argument("--resume-run", type=int, default=None,
                    help="append to this existing run, skipping documents it "
                         "has already measured")
    ap.add_argument("--notes", default="")
    ap.add_argument("--extra-notes", default="",
                    help="appended to the notes, whether they are the default "
                         "or given. For what the launcher knows and the run "
                         "does not, such as which GPU was asked for")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    conn = (P.open_ro(args.db) if args.dry_run
            else dbio.open_db(args.db, allow_migrate=args.migrate, needs=4))

    from transformers import AutoTokenizer
    if args.ar and (Path(args.ar) / "tokenizer_config.json").is_file():
        qtok = AutoTokenizer.from_pretrained(args.ar)
    elif args.dry_run:
        qtok = AutoTokenizer.from_pretrained(T.QWEN)
    else:
        raise SystemExit("--ar must point at the reconstructor for a real run")

    cache = {}
    t0 = time.perf_counter()
    pool = S.build_pool(conn, qtok, cache)
    pool_stats = {"occurrences": sum(len(v) for v in pool.values()),
                  "types": len({w.lower() for v in pool.values()
                                for _, w, _ in v}),
                  "docs": len({d for v in pool.values() for d, _, _ in v}),
                  "cells": len(pool)}
    index = P.index_pool(pool, qtok, cache)
    print(f"pool {pool_stats['occurrences']} occurrences, "
          f"{pool_stats['types']} types, {pool_stats['docs']} documents "
          f"({time.perf_counter() - t0:.1f}s)", flush=True)

    syn = P.load_syntax(conn)
    for r in conn.execute("SELECT doc_id, cjk_fraction FROM docs"):
        if int(r["doc_id"]) in syn["docs"]:
            syn["docs"][int(r["doc_id"])]["cjk"] = r["cjk_fraction"]
    scr = screen(syn, index, qtok, cache)
    want_docs, how, excluded, passing = select_docs(syn, scr, args)
    print(f"documents {want_docs}", flush=True)
    print(f"  {how}", flush=True)
    print(f"  {len(excluded)} of {len(syn['docs'])} documents excluded before "
          f"the draw", flush=True)
    for d, why in excluded[:20]:
        print(f"    doc {d}: {why}", flush=True)

    import spacy
    nlp = spacy.load("en_core_web_sm")
    scheme = dbio.spacy_scheme(nlp)
    if scheme != SPACY:
        print(f"WARNING: spaCy here is {scheme}, the store's labels are "
              f"{SPACY}", flush=True)

    model = tok = headm = E = None
    gold_mse = measurable(conn)
    rawvar, rawvar_src, rawvar_all, fp = None, None, {}, None
    print(f"documents carrying a gold activation {len(gold_mse)}/"
          f"{len(syn['docs'])}", flush=True)
    if not args.dry_run:
        import torch
        import harness
        from harness import make_deterministic, fingerprint, load_ar
        from extract_traces import MSE_SCALE, normalize_activation
        assert harness.MAX_PROMPT == MAX_PROMPT, "the prompt cap has drifted"
        torch.set_num_threads(args.threads)
        make_deterministic(args.seed)
        fp = fingerprint()
        print(json.dumps(fp), flush=True)
        model, tok, headm = load_ar(args.ar, args.device, args.precision)
        E = model.get_input_embeddings().weight
        rawvar, rawvar_src, rawvar_all = raw_variance(conn, args.traces,
                                                      args.rawvar_from)
        for k, (v, n) in rawvar_all.items():
            print(f"mse_rawvar over the {k} ({n} documents) {v:.6f}"
                  + ("   <- used" if k == rawvar_src else ""), flush=True)
        print(f"dFVE = -dMSE / rawvar, rawvar {rawvar:.6f} from "
              f"{rawvar_src}", flush=True)

    have = {(d, a, b): s for s, (d, a, b) in syn["spans"].items()}

    # ------------------------------------------------------------- dry run
    if args.dry_run:
        plans, skipped, fails, token = [], [], [], {}
        for doc_id in want_docs:
            if doc_id not in gold_mse:
                skipped.append((doc_id, "no gold activation in the store"))
                continue
            pl = prepare(doc_id, syn, index, qtok, cache, nlp, args, have)
            if pl["skip"]:
                skipped.append((doc_id, pl["skip"]))
                continue
            t = time.perf_counter()
            bad = verify(pl, qtok, cache, args)
            token[doc_id] = delta_census(prompt_lengths(qtok, pl["texts"]),
                                         pl["meta"])
            fails += [(doc_id,) + tuple(x) for x in bad[:4]]
            print(f"  doc {doc_id}: {pl['budget']['total']} passes planned, "
                  f"round-tripped and measured "
                  f"({time.perf_counter() - t:.1f}s)", flush=True)
            pl["texts"] = pl["meta"] = pl["subs"] = None
            plans.append(pl)
        if not plans:
            raise SystemExit("no document survived planning")
        report = plan_report(syn, plans, skipped, how, excluded, scr,
                             pool_stats, token, len(gold_mse), args)
        report += "\n\n## Plan verification\n\n"
        report += ("every planned edit round-tripped against the original, the "
                   "substitute for a span at a draw is one string across its "
                   "single and every joint edit it appears in, and each "
                   "document's plan is exactly its budgeted size. The prompt "
                   "length census above is data, not a check: a splice that "
                   "moves a token boundary is kept and recorded"
                   if not fails else
                   "FAILURES:\n" + "\n".join(f"  {x}" for x in fails[:20]))
        report += "\n"
        (out / "plan.md").write_text(report)
        print()
        print(report)
        print(f"plan: {out / 'plan.md'}")
        return 1 if fails else 0

    # ------------------------------------------------------------- the run
    from extract_traces import MSE_SCALE, normalize_activation
    import torch

    skip = set()
    if args.resume_run is not None:
        row = conn.execute("SELECT script FROM runs WHERE run_id = ?",
                           (int(args.resume_run),)).fetchone()
        if row is None:
            raise SystemExit(f"no run {args.resume_run} to resume")
        if "tree_vs_linear" not in (row["script"] or ""):
            raise SystemExit(f"run {args.resume_run} is {row['script']}, not a "
                             "tree_vs_linear run")
        run_id = int(args.resume_run)
        skip = S.already_done(conn, run_id)
        print(f"resuming run {run_id}; {len(skip)} documents already measured",
              flush=True)
    else:
        gpu = (torch.cuda.get_device_name(0) if torch.cuda.is_available()
               else "no CUDA device")
        notes = (args.notes or (
            "Full pairwise interaction matrix. For each selected document "
            "every eligible word is corpus-swapped alone and every unordered "
            "pair of eligible words is swapped together, at eight draws, so "
            "the interaction e(a) + e(b) - e(both) is measurable for every "
            "cell of the n by n matrix rather than for a sample of cells. "
            "Eligibility, the swap, the pool and the splice are 05's. Common "
            "random numbers: a span's substitute is a property of the span and "
            "the draw, so the string spliced at that span is identical in its "
            "single and in all n - 1 joint edits containing it. A document is "
            "measured at every draw and committed before the next one starts. "
            "Documents were drawn uniformly at random from those with a CJK "
            "fraction at most 0.05 and at least 90 per cent of their clean "
            "lexical words servable by the swap pool."))
        run_id = DB.new_run(
            conn, script="06_tree-vs-linear/tree_vs_linear.py", assets=ASSETS,
            notes=notes + args.extra_notes + f" GPU: {gpu}.",
            config={"args": vars(args), "fingerprint": fp,
                    "mse_rawvar": rawvar, "mse_rawvar_from": rawvar_src,
                    "mse_rawvar_all": {k: v for k, v in rawvar_all.items()},
                    "gold": "docs.activation, float32 little-endian",
                    "swap_scheme": SWAP,
                    "pair_scheme": PAIR_SCHEME, "spacy_scheme": scheme,
                    "gpu": gpu, "pool": pool_stats,
                    "doc_selection": how, "documents": want_docs,
                    "excluded_documents": excluded,
                    "documents_passing_filters": passing,
                    "max_cjk": MAX_CJK, "min_eligible": MIN_ELIGIBLE,
                    "draws": args.draws,
                    "sign": "interaction = e(a) + e(b) - e(both), FVE points; "
                            "positive means the pair costs less than the sum "
                            "of its singles"})
        conn.commit()
    print(f"run_id {run_id}", flush=True)

    t0 = time.perf_counter()
    n_docs = n_var = 0
    for doc_id in want_docs:
        if doc_id in skip:
            print(f"  doc {doc_id}: already measured under run {run_id}, "
                  f"skipped", flush=True)
            continue
        if doc_id not in gold_mse:
            print(f"  doc {doc_id}: no gold activation, skipped", flush=True)
            continue
        pl = prepare(doc_id, syn, index, qtok, cache, nlp, args, have)
        if pl["skip"]:
            print(f"  doc {doc_id}: {pl['skip']}, skipped", flush=True)
            continue
        for line in census_lines(pl):
            print(line, flush=True)
        bad = verify(pl, qtok, cache, args)
        if bad:
            raise SystemExit(f"doc {doc_id}: a planned edit does not "
                             f"round-trip: {bad[:4]}")

        gold = torch.tensor(gold_vector(conn, doc_id), dtype=torch.float32,
                            device=args.device)
        gold_n = normalize_activation(gold.unsqueeze(0), MSE_SCALE)[0]
        dbio.ensure_spans(conn, doc_id, pl["text"], pl["words"], scheme,
                          source="ffw_main_traces")

        mse, lens = scored(model, tok, headm, E, gold_n, pl["texts"],
                           args.batch, args.device)
        # seq_len is the prompt the reconstructor actually read, so dtok is the
        # length change this edit caused, recorded rather than constrained
        cen = delta_census(lens, pl["meta"])
        print(f"      prompt {cen['base']} baseline, {cen['max']} longest; "
              f"dtok singles {dict(sorted(cen['single'].items()))}, "
              f"joints {dict(sorted(cen['joint'].items()))}", flush=True)
        records = [([], {"mse": mse[0], "fve": 1 - mse[0] / rawvar,
                         "seq_len": lens[0], "dtok": 0,
                         "traces_mse": gold_mse[doc_id]})]
        for j, subs in enumerate(pl["meta"], start=1):
            records.append((subs, {"mse": mse[j], "fve": 1 - mse[j] / rawvar,
                                   "seq_len": lens[j],
                                   "dtok": lens[j] - lens[0]}))
        dbio.write_variants(conn, doc_id, run_id, records)
        # the pair set and each pair's category, recoverable from the store
        # alone. `relations` already means an edge between two spans under a
        # scheme, so this needs no migration
        with DB.transaction(conn):
            DB.add_relations(conn, [
                (PAIR_SCHEME, int(p["span_a"]), int(p["span_b"]),
                 f"arc:{p['dep']}" if p["category"] == "arc"
                 else p["category"]) for p in pl["pairs"]])
        n_docs += 1
        n_var += len(records)
        print(f"  doc {doc_id} committed: {len(records)} variants, "
              f"base FVE {1 - mse[0] / rawvar:.4f}, "
              f"{time.perf_counter() - t0:.0f}s elapsed", flush=True)
        pl = None

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
