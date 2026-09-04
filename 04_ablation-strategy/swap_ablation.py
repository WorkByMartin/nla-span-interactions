#!/usr/bin/env python3
"""Corpus-swap ablation: replace a word with a matched word from another document.

    python swap_ablation.py --ar "$ASSETS/qwen36-27b_ar-l43-s600_model" \
        --traces ../01_corpus-and-spans/results/ffw_pilot_traces.parquet \
        --db ../db/ffw_span-ablation_database.sqlite \
        --source-run 3 --docs all --draws 16 --shuffles 4 --shuffle-docs 20 \
        --batch 8 --precision bf16 --threads 8 --seed 0

Needs a GPU. It repeats the documents and the spans of an earlier masked-LM run,
given as --source-run, and writes every forward pass back to the store as a
variant with its substitution and its measurements.

The substitute is drawn from OTHER documents' verbalisations, matched only on
surface properties that keep the edit a one-word swap rather than a change of
shape:

  coarse POS      spaCy `pos`, so a determiner is replaced by a determiner
  Qwen token len  the substitute occupies the same number of Qwen tokens
  space parity    a leading space iff the original had one

The original word is excluded case-insensitively, and the substitute takes the
original's capitalisation. Draws are uniform over pool OCCURRENCES, not types, so
a common word is drawn as often as it occurs. Nothing about the sentence being
edited enters the choice.

Three further arms are measured on the same documents:

  deletion  the word and its space removed outright, one variant per span.
  shuffle   every lexical word in the document permuted among the document's own
            slots, so the bag of words is preserved and the order is destroyed.
  baseline  the intact document, batched with the arms as pos_fve.py does, so it
            comes down the same numerical path as the values it is subtracted
            from.
"""
import argparse, json, os, random, sys, time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import pyarrow.parquet as pq
import torch

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
    raise SystemExit(f"cannot find {marker}: not in " +
                     ", ".join(str(c) for c in candidates) +
                     ". Copy it across alongside this directory.")


sys.path.insert(0, str(_package("harness.py",
                                HERE.parent / "03_parts-of-speech", HERE)))
from harness import (  # noqa: E402
    MAX_PROMPT, make_deterministic, fingerprint, load_ar, mse_of_texts)
import textsub as T  # noqa: E402

sys.path.insert(0, str(_package("extract_traces.py",
                                HERE.parent / "01_corpus-and-spans", HERE)))
from extract_traces import MSE_SCALE, normalize_activation  # noqa: E402

sys.path.insert(0, str(_package("dbio.py", HERE.parent / "db", HERE / "db",
                                HERE.parent.parent / "db")))
import db as DB  # noqa: E402
import dbio  # noqa: E402

SWAP = "corpus-swap/pos+len"
DELETION = "deletion"
SHUFFLE = "shuffle"
MLM = dbio.FILLER

ASSETS = ["ffw_main_traces", "qwen36-27b_ar-l43-s600_model",
          "qwen36-27b_tokenizer"]


# ------------------------------------------------------------- capitalisation

def case_of(word):
    """Which of three capitalisation shapes the original word has."""
    if len(word) > 1 and word.isupper():
        return "upper"
    if word[:1].isupper():
        return "title"
    return "lower"


def apply_case(word, case):
    """Put a pool word into the original's capitalisation shape."""
    w = word.lower()
    if case == "upper":
        return w.upper()
    if case == "title":
        return w[:1].upper() + w[1:]
    return w


# ------------------------------------------------------------------- the pool

def qwen_len(qtok, cache, word, has_space):
    """Qwen token count of the string that would actually be spliced in."""
    key = (word, has_space)
    n = cache.get(key)
    if n is None:
        s = (" " + word) if has_space else word
        n = len(qtok.encode(s, add_special_tokens=False))
        cache[key] = n
    return n


def build_pool(conn, qtok, cache=None):
    """Every lexical span in the store, grouped by (pos, leading-space parity).

    A span's parity is a property of where it sits in its OWN document: the
    character before it is a space, or it starts the document. Nothing here is
    conditioned on any document being edited, so the pool is built once.

    Returns {(pos, has_space): [(doc_id, word, n_qwen), ...]} in span order,
    which makes the draw reproducible from the seed alone.
    """
    cache = {} if cache is None else cache
    texts = {int(r["doc_id"]): r["text"]
             for r in conn.execute("SELECT doc_id, text FROM docs")}
    pool = defaultdict(list)
    q = ("SELECT span_id, doc_id, char_start, char_end, text, pos "
         "FROM v_pos ORDER BY span_id")
    for r in conn.execute(q):
        pos = r["pos"]
        if pos in T.NON_LEXICAL_POS:
            continue
        word = r["text"]
        if not T.is_clean_word(word):
            continue
        doc = int(r["doc_id"])
        text = texts.get(doc)
        if text is None:
            continue
        a = int(r["char_start"])
        has_space = a > 0 and text[a - 1] == " "
        pool[(pos, has_space)].append((doc, word,
                                       qwen_len(qtok, cache, word, has_space)))
    return dict(pool)


def pool_for(pool, qtok, cache, pos, has_space, n_qwen, orig, case, doc_id):
    """The eligible pool occurrences for one span, in a stable order.

    Eligibility is: a different document, not the original word (case blind), and
    the same Qwen token count ONCE the pool word has been recased, since recasing
    is what will actually be spliced.
    """
    lo = orig.lower()
    out = []
    for d, w, _ in pool.get((pos, has_space), ()):
        if d == doc_id or w.lower() == lo:
            continue
        c = apply_case(w, case)
        if qwen_len(qtok, cache, c, has_space) == n_qwen:
            out.append(c)
    return out


# -------------------------------------------------------------------- edits

def delete_span(text, spans, k):
    """Remove unit k, leaving no doubled space and no leading space behind.

    word_spans folds the preceding space into the unit, so the common case
    already deletes ` word` in one piece. The two exceptions are a unit at the
    very start of the document and a unit whose fold did not happen because the
    previous character was not a space.
    """
    a, b = spans[k]
    out = text[:a] + text[b:]
    if 0 < a < len(out) and out[a - 1] == " " and out[a] == " ":
        out = out[:a] + out[a + 1:]
    elif a == 0 and out.startswith(" "):
        out = out[1:]
    return out


def shuffle_text(text, spans, words, perm):
    """Every lexical unit's word replaced by the word of the unit perm maps it to.

    Punctuation is untouched because it is not a unit. Space parity is restored
    per destination slot, so a word that moves to the head of the document loses
    its leading space and one that moves inward gains one.
    """
    subs = {k: T.fit_space(text, spans[k], words[perm[k]]["text"])
            for k in range(len(spans))}
    return T.splice_many(text, spans, subs)[0]


# ------------------------------------------------------------ document choice

def run_baselines(conn, run_id):
    """(doc_id, base_fve) for every document a previous run measured, best last."""
    q = ("SELECT doc_id, base_fve FROM v_baseline WHERE run_id = ? "
         "ORDER BY base_fve")
    return [(int(r["doc_id"]), float(r["base_fve"]))
            for r in conn.execute(q, (int(run_id),))]


def select_shuffle_docs(picked, n, seed):
    """n of the picked documents, one drawn from each stratum of baseline FVE.

    The shuffle arm costs four passes per document, so at 40 documents it is run
    on a subset. Stratifying on baseline FVE keeps the subset spread over the
    range that matters rather than over document ids. n of None, or n at least
    the document count, means every document.
    """
    if n is None or n >= len(picked):
        return {d for d, _ in picked}
    if n <= 0:
        return set()
    order = [d for d, b in sorted(picked, key=lambda x: (x[1], x[0]))]
    rng = random.Random(f"shuffle-docs {seed}")
    m = len(order)
    out = set()
    for i in range(n):
        band = order[i * m // n:(i + 1) * m // n] or order[i * m // n:][:1]
        out.add(band[rng.randrange(len(band))])
    return out


def already_done(conn, run_id):
    """Documents this run has already measured, so a resume can skip them.

    A document is done when it has a baseline variant under this run, which
    write_variants creates in the same transaction as that document's arms. The
    unit of commit is the document, so a run killed mid-flight leaves whole
    documents behind, never half of one.
    """
    q = ("SELECT DISTINCT v.doc_id FROM variants v "
         "WHERE v.created_run_id = ? "
         "  AND NOT EXISTS (SELECT 1 FROM substitutions s "
         "                  WHERE s.variant_id = v.variant_id)")
    return {int(r["doc_id"]) for r in conn.execute(q, (int(run_id),))}


def run_spans(conn, run_id, doc_id, source=MLM):
    """The span_ids a previous run ablated in this document, in span order."""
    q = ("SELECT DISTINCT s.span_id FROM variants v "
         "JOIN substitutions s ON s.variant_id = v.variant_id "
         "WHERE v.created_run_id = ? AND v.doc_id = ? AND s.source = ? "
         "ORDER BY s.span_id")
    return [int(r["span_id"])
            for r in conn.execute(q, (int(run_id), int(doc_id), source))]


# ------------------------------------------------------------------ planning

def plan_document(text, words, span_ids, targets, pool, qtok, cache, doc_id,
                  rng, draws=16, shuffles=4):
    """Every edited string this document contributes, and what each one is.

    Returns (texts, meta, draw_rows). texts[0] is the intact document; meta is
    aligned with texts[1:] and each entry is a list of substitution dicts ready
    for dbio.write_variants. draw_rows carries the swap pool draws for the
    candidates table: (span_id, rank, word, prob).

    No model, no torch: this is the whole edit plan, so it can be checked on a
    laptop before an A100 is started.
    """
    spans = [(w["start"], w["end"]) for w in words]
    texts, meta, draw_rows, skipped = [text], [], [], []

    for k in targets:
        orig = words[k]["text"]
        pos = words[k]["pos"]
        has_space = spans[k][0] < len(text) and text[spans[k][0]] == " "
        case = case_of(orig)
        n_q = qwen_len(qtok, cache, orig, has_space)
        cands = pool_for(pool, qtok, cache, pos, has_space, n_q, orig, case,
                         doc_id)
        if not cands:
            skipped.append((k, orig, pos, n_q))
            continue
        p = 1.0 / len(cands)
        for i in range(draws):
            w = cands[rng.randrange(len(cands))]
            texts.append(T.splice(text, spans, k,
                                  T.fit_space(text, spans[k], w))[0])
            meta.append([{"span_id": span_ids[k], "substitute": w,
                          "source": SWAP, "depth": 1, "draw_idx": i,
                          "prob": p}])
            draw_rows.append((span_ids[k], i, w, p))

    for k in targets:
        texts.append(delete_span(text, spans, k))
        meta.append([{"span_id": span_ids[k], "substitute": "",
                      "source": DELETION, "depth": 1, "draw_idx": 0,
                      "prob": None}])

    n = len(spans)
    for j in range(shuffles):
        perm = list(range(n))
        rng.shuffle(perm)
        texts.append(shuffle_text(text, spans, words, perm))
        meta.append([{"span_id": span_ids[k],
                      "substitute": words[perm[k]]["text"],
                      "source": SHUFFLE, "depth": n, "draw_idx": j,
                      "prob": None} for k in range(n)])

    return texts, meta, draw_rows, skipped


# ---------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ar", required=True)
    ap.add_argument(
        "--traces",
        default="../01_corpus-and-spans/results/ffw_pilot_traces.parquet")
    ap.add_argument("--db", default=str(dbio.DEFAULT_DB))
    ap.add_argument("--source-run", type=int, default=3,
                    help="the run whose documents and spans this one repeats")
    ap.add_argument("--docs", default="all",
                    help="'all' for every document the source run measured, or "
                         "a comma-separated list of doc ids")
    ap.add_argument("--draws", type=int, default=16)
    ap.add_argument("--shuffles", type=int, default=4)
    ap.add_argument("--shuffle-docs", type=int, default=None,
                    help="run the shuffle arm on this many of the chosen "
                         "documents, one drawn per stratum of baseline FVE. "
                         "The default is every document")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--precision", default="bf16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--migrate", action="store_true",
                    help="apply outstanding migrations to --db first. Off by "
                         "default: a schema change is Marty's call")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume-run", type=int, default=None,
                    help="append to this existing run instead of opening a new "
                         "one, skipping documents it has already measured. Draws "
                         "then come from a per-document stream seeded on the "
                         "document id, so a document's draws do not depend on "
                         "how many documents ran before it")
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    make_deterministic(args.seed)
    device = args.device
    fp = fingerprint()
    print(json.dumps(fp), flush=True)

    import spacy
    nlp = spacy.load("en_core_web_sm")
    conn = dbio.open_db(args.db, allow_migrate=args.migrate, needs=4)
    scheme = dbio.spacy_scheme(nlp)

    model, tok, head = load_ar(args.ar, device, args.precision)
    E = model.get_input_embeddings().weight
    rows = pq.read_table(args.traces).to_pylist()
    by_doc = {int(r["doc_uid"]): r for r in rows}
    rng = random.Random(args.seed)

    G = torch.tensor([r["activation"] for r in rows], dtype=torch.float64)
    Gn = G / G.norm(dim=1, keepdim=True) * MSE_SCALE
    rawvar = float(((Gn - Gn.mean(0)) ** 2).mean())
    print(f"mse_rawvar {rawvar:.5f}  dFVE = -dMSE / rawvar", flush=True)

    if args.docs == "auto":
        picked = select_docs(conn, args.source_run, n_extra=args.n_docs - 2)
    elif args.docs == "all":
        # every document the source run measured, in document-id order so the
        # log reads the same way twice
        picked = sorted(run_baselines(conn, args.source_run))
    else:
        base = dict(run_baselines(conn, args.source_run))
        picked = [(int(d), base.get(int(d)))
                  for d in args.docs.split(",") if d.strip()]
    print(f"documents {len(picked)} (source-run baseline FVE): "
          + ", ".join(f"{d} {b:.4f}" for d, b in picked), flush=True)
    shuffle_set = select_shuffle_docs(picked, args.shuffle_docs, args.seed)
    if len(shuffle_set) != len(picked):
        print(f"shuffle arm on {len(shuffle_set)} of {len(picked)} documents: "
              + ", ".join(str(d) for d in sorted(shuffle_set)), flush=True)

    cache = {}
    t_pool = time.perf_counter()
    pool = build_pool(conn, tok, cache)
    n_pool = sum(len(v) for v in pool.values())
    pool_docs = len({d for v in pool.values() for d, _, _ in v})
    pool_types = len({w.lower() for v in pool.values() for _, w, _ in v})
    print(f"pool {n_pool} occurrences ({pool_types} word types) over "
          f"{pool_docs} documents and {len(pool)} (pos, parity) cells "
          f"({time.perf_counter() - t_pool:.1f}s)", flush=True)

    skip = set()
    if args.resume_run is not None:
        row = conn.execute("SELECT script FROM runs WHERE run_id = ?",
                           (int(args.resume_run),)).fetchone()
        if row is None:
            raise SystemExit(f"no run {args.resume_run} to resume")
        if "swap_ablation" not in (row["script"] or ""):
            raise SystemExit(f"run {args.resume_run} is {row['script']}, "
                             "not a swap_ablation run")
        run_id = int(args.resume_run)
        skip = already_done(conn, run_id)
        print(f"resuming run {run_id}; {len(skip)} documents already measured: "
              + ", ".join(str(d) for d in sorted(skip)), flush=True)
    else:
        run_id = DB.new_run(
        conn, script="04_ablation-strategy/swap_ablation.py", assets=ASSETS,
        notes=args.notes or (
            "Corpus-swap ablation. A word is replaced by a word drawn "
            "uniformly over occurrences in OTHER documents' verbalisations, "
            "matched on spaCy coarse POS, Qwen token count and leading-space "
            "parity, excluding the original case-insensitively and recased to "
            "the original. Deletion and within-document word-order shuffle arms "
            "give the swap a scale. Same documents and same spans as the "
            "masked-LM run this is compared against. The baseline rides in the "
            "same batched call as the arms."),
        config={"args": vars(args), "fingerprint": fp, "mse_rawvar": rawvar,
                "swap_scheme": SWAP, "mlm_scheme": MLM,
                "spacy_scheme": scheme, "pool_occurrences": n_pool,
                "pool_documents": pool_docs, "pool_word_types": pool_types,
                "pool_cells": len(pool),
                "pool_source": (
                    "every lexical span of every verbalisation held in the "
                    "store, one occurrence per span, grouped by (spaCy coarse "
                    "POS, leading-space parity); eligibility for a target span "
                    "additionally requires a different document, a different "
                    "word case-blind, and the same Qwen token count after "
                    "recasing to the target's capitalisation"),
                "documents": [[d, b] for d, b in picked],
                "shuffle_documents": sorted(shuffle_set),
                "doc_selection": (
                    "every document the source run measured"
                    if args.docs == "all"
                    else "given explicitly on the command line")})
    conn.commit()
    print(f"run_id {run_id}", flush=True)

    t0 = time.perf_counter()
    n_docs = n_var = n_cand = 0
    for doc_id, src_base in picked:
        if doc_id in skip:
            print(f"  doc {doc_id}: already measured under run {run_id}, "
                  f"skipped", flush=True)
            continue
        row = by_doc.get(doc_id)
        if row is None:
            print(f"  doc {doc_id}: not in the traces file, skipped", flush=True)
            continue
        text = row["explanation"]
        if len(T.prompt_ids(tok, text)) > MAX_PROMPT:
            print(f"  doc {doc_id}: prompt over {MAX_PROMPT}, skipped",
                  flush=True)
            continue

        gold = torch.tensor(row["activation"], dtype=torch.float32, device=device)
        gold_n = normalize_activation(gold.unsqueeze(0), MSE_SCALE)[0]

        words = T.word_spans(nlp, text, lexical_only=True)
        span_ids = dbio.ensure_spans(conn, doc_id, text, words, scheme,
                                     source="ffw_main_traces")
        want = set(run_spans(conn, args.source_run, doc_id))
        targets = [k for k, s in enumerate(span_ids) if s in want]
        if len(targets) != len(want):
            print(f"  doc {doc_id}: WARNING {len(targets)} of {len(want)} "
                  f"source-run spans matched a spaCy word", flush=True)

        # a resumed run draws from a stream seeded on the document, so what a
        # document gets does not depend on how many documents preceded it
        # a tuple is not an accepted seed on Python 3.11 and later, so the
        # per-document stream is seeded on a string of the two numbers
        doc_rng = (random.Random(f"swap {args.seed} {doc_id}")
                   if args.resume_run is not None else rng)
        texts, meta, draws, skipped = plan_document(
            text, words, span_ids, targets, pool, tok, cache, doc_id, doc_rng,
            draws=args.draws,
            shuffles=args.shuffles if doc_id in shuffle_set else 0)
        if skipped:
            print(f"  doc {doc_id}: {len(skipped)} spans had an empty pool: "
                  + ", ".join(f"{w!r}/{p}/{n}q" for _, w, p, n in skipped[:6]),
                  flush=True)

        vals, lens = mse_of_texts(model, tok, head, E, gold_n, texts,
                                  args.batch, device)
        mse = [float(v) for v in vals]
        records = [([], {"mse": mse[0], "fve": 1 - mse[0] / rawvar,
                         "seq_len": lens[0], "traces_mse": row["mse"]})]
        for j, subs in enumerate(meta, start=1):
            records.append((subs, {"mse": mse[j], "fve": 1 - mse[j] / rawvar,
                                   "seq_len": lens[j]}))
        dbio.write_variants(conn, doc_id, run_id, records)
        if draws:
            with DB.transaction(conn):
                DB.add_candidates(conn, [(int(s), SWAP, int(i), DB.bare(w),
                                          float(p)) for s, i, w, p in draws])
            n_cand += len(draws)

        n_docs += 1
        n_var += len(records)
        base_fve = 1 - mse[0] / rawvar
        print(f"  doc {doc_id}  {len(targets)} target spans  {len(words)} "
              f"lexical words  {len(records)} variants  base FVE "
              f"{base_fve:.4f} (source run {src_base:.4f})  "
              f"{time.perf_counter() - t0:.0f}s elapsed", flush=True)

    print(f"\nrun {run_id}: {n_docs} docs, {n_var} variants, {n_cand} pool draws "
          f"recorded ({time.perf_counter() - t0:.0f}s)")
    for k, v in DB.counts(conn).items():
        print(f"  {k:16s} {v}")
    conn.execute("PRAGMA optimize")
    conn.close()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
