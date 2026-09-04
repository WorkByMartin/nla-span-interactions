#!/usr/bin/env python3
"""Sequential removal curves over every eligible word of a document.

    python removal_curve.py --dry-run --with-docs 240,621,1664,2592,4126 --n-docs 15 \
        --db ../db/ffw_span-ablation_database.sqlite

    python removal_curve.py --ar "$AR" --traces ../01_corpus-and-spans/results/ffw_pilot_traces.parquet \
        --db ../db/ffw_span-ablation_database.sqlite \
        --with-docs 240,621,1664,2592,4126 --n-docs 15 --perms 8 --seed 0 --batch 16

Eligibility, the swap pool, the splice and the database writes are 05's and
06's, imported rather than restated. What differs is the unit of measurement:
06 measured every single and every pair; this walks each document from intact
to empty one word at a time and records FVE at every step.

Curves per document, each a sequence of n cumulative states (n eligible words):

    curve 0  random order, deletion       perms permutations
    curve 1  random order, corpus swap    the same perms permutations, one draw per span per permutation
    curve 2  front truncation, deletion   reading order
    curve 3  back truncation, deletion    reverse reading order
    curve 4  random order, filler         the same permutations; the word replaced by the filler
                                          token repeated to the word's Qwen token count
    curve 5  front truncation, filler
    curve 6  back truncation, filler

Passes per document: 1 + n x (3 x perms + 4). A step's variant carries every
removed span as a substitution row (substitute '' under source 'deletion', the
drawn word under the swap scheme, or the filler string under source 'filler'), depth = number of words removed, and the
measurements `curve`, `step`, `perm`, `n_words` beside mse, fve, seq_len, dtok.
Under the swap curve a span's substitute is a property of (span, permutation),
so the same string is spliced at that span at every later step of that curve.

--dry-run plans and checks every string and writes results/plan.md.
"""
import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for _c in (HERE, REPO / "06_tree-vs-linear"):
    if (_c / "tree_vs_linear.py").is_file():
        sys.path.insert(0, str(_c))
        break
else:
    raise SystemExit("cannot find tree_vs_linear.py beside this file or in 06_tree-vs-linear")

import tree_vs_linear as M  # noqa: E402  (also puts pairs/ helpers on the path)
import pair_ablation as P  # noqa: E402
import textsub as T  # noqa: E402
import swap_ablation as S  # noqa: E402
import db as DB  # noqa: E402
import dbio  # noqa: E402

SWAP = P.SWAP
DELETION = S.DELETION
SPACY = P.SPACY
ASSETS = P.ASSETS
MAX_PROMPT = M.MAX_PROMPT
CURVES = {0: "random deletion", 1: "random swap", 2: "front truncation",
          3: "back truncation", 4: "random filler", 5: "front truncation filler",
          6: "back truncation filler"}
FILLER = "filler"


def select_docs(syn, scr, args):
    """--with-docs always in; --n-docs more drawn uniformly from the rest that pass 06's filters."""
    _, _, excluded, ok = M.select_docs(syn, scr, argparse.Namespace(docs="", n_docs=0, seed=args.seed))
    with_docs = [int(d) for d in args.with_docs.replace(",", " ").split()] if args.with_docs else []
    missing = [d for d in with_docs if d not in syn["docs"]]
    if missing:
        raise SystemExit(f"no such documents in the store: {missing}")
    rest = [d for d in ok if d not in with_docs]
    if args.n_docs > len(rest):
        raise SystemExit(f"--n-docs {args.n_docs} but only {len(rest)} candidates")
    rng = random.Random(f"removal-docs {args.seed}")
    drawn = sorted(rng.sample(rest, args.n_docs)) if args.n_docs else []
    how = (f"{len(with_docs)} named ({with_docs}) plus {args.n_docs} of the {len(rest)} other "
           f"documents passing 06's filters (cjk <= {M.MAX_CJK}, eligible >= {M.MIN_ELIGIBLE}), "
           f"drawn uniformly with seed {args.seed}")
    return sorted(with_docs) + drawn, how, excluded, ok


def budget_for(n, perms):
    return {"n": n, "perms": perms, "baselines": 1, "random_del": n * perms,
            "random_swap": n * perms, "random_filler": n * perms, "front": 2 * n, "back": 2 * n,
            "total": 1 + n * (3 * perms + 4)}


def filler_for(qtok, cache, filler, orig, has_space):
    """The filler string whose Qwen token count, spliced with this span's space parity, equals the original's.

    Built as the filler token repeated with spaces between, so each repeat is one token.
    Returns (string, matched); an unmatched string is used anyway and its dtok recorded.
    """
    want = S.qwen_len(qtok, cache, orig, has_space)
    for m in range(1, want + 3):
        cand = " ".join([filler] * m)
        if S.qwen_len(qtok, cache, cand, has_space) == want:
            return cand, True
    return " ".join([filler] * want), False


def plan_document(text, words, sids, keep, elig, index, seed, perms, qtok, cache, filler):
    """(texts, meta, extra, subs_by_span, fill_by_span) for one document. texts[0] is the intact string."""
    spans = [(w["start"], w["end"]) for w in words]
    by_sid = {s: k for k, s in enumerate(sids)}
    order = sorted(keep, key=lambda s: (elig[s]["tok_i"], s))
    n = len(order)
    subs_by_span, probs, fill_by_span = {}, {}, {}
    for sid in order:
        subs_by_span[sid], probs[sid] = P.draw_substitutes(elig[sid], index, seed, perms)
        k = by_sid[sid]
        has_space = spans[k][0] < len(text) and text[spans[k][0]] == " "
        fill_by_span[sid] = filler_for(qtok, cache, filler, words[k]["text"], has_space)

    texts, meta, extra = [text], [], []

    def emit(seq, curve, perm, mode):
        removed = []
        for step, sid in enumerate(seq, start=1):
            removed.append(sid)
            if mode == "swap":
                joint = {by_sid[s]: T.fit_space(text, spans[by_sid[s]], subs_by_span[s][perm])
                         for s in removed}
                texts.append(T.splice_many(text, spans, joint)[0])
                meta.append([{"span_id": s, "substitute": subs_by_span[s][perm], "source": SWAP,
                              "depth": step, "draw_idx": perm, "prob": probs[s]} for s in removed])
            elif mode == "filler":
                joint = {by_sid[s]: T.fit_space(text, spans[by_sid[s]], fill_by_span[s][0])
                         for s in removed}
                texts.append(T.splice_many(text, spans, joint)[0])
                meta.append([{"span_id": s, "substitute": fill_by_span[s][0], "source": FILLER,
                              "depth": step, "draw_idx": perm, "prob": None} for s in removed])
            else:
                texts.append(delete_many(text, spans, [by_sid[s] for s in removed]))
                meta.append([{"span_id": s, "substitute": "", "source": DELETION,
                              "depth": step, "draw_idx": perm, "prob": None} for s in removed])
            extra.append({"curve": curve, "step": step, "perm": perm, "n_words": n})

    for p in range(perms):
        rng = random.Random(f"removal-perm {seed} {elig[order[0]]['doc_id']} {p}")
        seq = list(order)
        rng.shuffle(seq)
        emit(seq, 0, p, "delete")
        emit(seq, 1, p, "swap")
        emit(seq, 4, p, "filler")
    emit(list(order), 2, 0, "delete")
    emit(list(reversed(order)), 3, 0, "delete")
    emit(list(order), 5, 0, "filler")
    emit(list(reversed(order)), 6, 0, "filler")
    return texts, meta, extra, subs_by_span, fill_by_span


def delete_many(text, spans, ks):
    """Delete several units with 04's single-unit rule, applied right to left.

    Right to left, every unit still to be deleted sits before the one just
    removed, so the original offsets stay valid: delete_span only ever touches
    characters at or after its unit's start.
    """
    out = text
    for k in sorted(ks, reverse=True):
        out = S.delete_span(out, spans, k)
    return out


def check_plan(pl, qtok, cache, perms):
    """Round-trip every planned string. Returns the failures."""
    text, words, sids = pl["text"], pl["words"], pl["sids"]
    spans = [(w["start"], w["end"]) for w in words]
    by_sid = {s: k for k, s in enumerate(sids)}
    bad = []
    if pl["texts"][0] != text:
        bad.append(("baseline is not the intact document",))
    if len(pl["texts"]) != pl["budget"]["total"]:
        bad.append(("plan size", len(pl["texts"]), pl["budget"]["total"]))
    seen = {}
    for t, m, x in zip(pl["texts"][1:], pl["meta"], pl["extra"]):
        if len(m) != x["step"] or any(s["depth"] != x["step"] for s in m):
            bad.append(("depth", x))
        ks = [by_sid[s["span_id"]] for s in m]
        if m[0]["source"] == DELETION:
            want = text
            for k in sorted(ks, reverse=True):
                a, b = spans[k]
                want = want[:a] + want[b:]
                if 0 < a < len(want) and want[a - 1] == " " and want[a] == " ":
                    want = want[:a] + want[a + 1:]
                elif a == 0 and want.startswith(" "):
                    want = want[1:]
            if t != want:
                bad.append(("deletion splice", x))
        elif m[0]["source"] == FILLER:
            want = text
            for s, k in sorted(zip(m, ks), key=lambda z: -z[1]):
                a, b = spans[k]
                has_space = a < len(text) and text[a] == " "
                want = want[:a] + ((" " + s["substitute"]) if has_space else s["substitute"]) + want[b:]
            if t != want:
                bad.append(("filler splice", x))
            if any(set(s["substitute"].replace(" ", "")) != {pl["filler"]} for s in m):
                bad.append(("filler string", x))
        else:
            want = text
            for s, k in sorted(zip(m, ks), key=lambda z: -z[1]):
                a, b = spans[k]
                has_space = a < len(text) and text[a] == " "
                want = want[:a] + ((" " + s["substitute"]) if has_space else s["substitute"]) + want[b:]
            if t != want:
                bad.append(("swap splice", x))
            for s, k in zip(m, ks):
                orig = words[k]["text"]
                a, _ = spans[k]
                has_space = a < len(text) and text[a] == " "
                if s["substitute"].lower() == orig.lower():
                    bad.append(("substitute is the original", orig))
                if S.qwen_len(qtok, cache, s["substitute"], has_space) != S.qwen_len(qtok, cache, orig, has_space):
                    bad.append(("qwen length", s["substitute"], orig))
                key = (s["span_id"], s["draw_idx"])
                if seen.setdefault(key, s["substitute"]) != s["substitute"]:
                    bad.append(("common random numbers broken", key))
        if x["curve"] in (2, 3) and x["step"] == x["n_words"] and m[0]["source"] == DELETION:
            pass
    # the fully deleted string is the same whichever order removed it
    ends = {(x["curve"], x["perm"]): t for t, m, x in zip(pl["texts"][1:], pl["meta"], pl["extra"])
            if x["step"] == x["n_words"] and m[0]["source"] == DELETION}
    if len(set(ends.values())) != 1:
        bad.append(("deletion endpoints differ", len(set(ends.values()))))
    return bad[:20]


def prepare(doc_id, syn, index, qtok, cache, nlp, args, have):
    elig, rejected = M.doc_eligibility(syn, index, qtok, cache, doc_id)
    text = syn["docs"][doc_id]["text"]
    if len(T.prompt_ids(qtok, text)) > MAX_PROMPT:
        return {"doc_id": doc_id, "skip": "prompt over the EasyNLA cap"}
    words = T.word_spans(nlp, text, lexical_only=True)
    keys = [(doc_id,) + dbio.bare_span(w) for w in words]
    sids = [have.get(k) for k in keys]
    present = {s for s in sids if s is not None}
    keep = sorted(s for s in elig if s in present and elig[s]["tok_i"] is not None)
    lost = len(elig) - len(keep)
    if lost:
        rejected["not reproduced by this spaCy parse"] += lost
    if len(keep) < 2:
        return {"doc_id": doc_id, "skip": "fewer than two eligible words"}
    texts, meta, extra, subs, fills = plan_document(text, words, sids, keep, elig, index, args.seed,
                                                    args.perms, qtok, cache, args.filler)
    return {"doc_id": doc_id, "skip": None, "text": text, "words": words, "sids": sids,
            "texts": texts, "meta": meta, "extra": extra, "subs": subs, "eligible": keep,
            "rejected": rejected, "n_words": len(words), "budget": budget_for(len(keep), args.perms),
            "filler": args.filler, "filler_unmatched": sum(not ok for _, ok in fills.values())}


def plan_report(plans, skipped, how, excluded, token, args):
    L = ["# Removal curve plan", "", f"documents {[p['doc_id'] for p in plans]}", "", how, "",
         f"{len(excluded)} documents excluded by the filters before the draw", "",
         f"perms {args.perms}, seed {args.seed}, filler {args.filler!r}", "",
         "| doc | lexical words | eligible | passes | baseline prompt | max prompt | filler unmatched |", "|---|---|---|---|---|---|---|"]
    tot = 0
    for p in plans:
        c = token.get(p["doc_id"], {})
        L.append(f"| {p['doc_id']} | {p['n_words']} | {len(p['eligible'])} | {p['budget']['total']} | "
                 f"{c.get('base', '')} | {c.get('max', '')} | {p.get('filler_unmatched', '')} |")
        tot += p["budget"]["total"]
    L += ["", f"total passes {tot}, {tot / M.REF_PASSES_PER_SEC / 60:.0f} min at 05's 12.1/s ({M.REF_NOTE})", ""]
    for d, why in skipped:
        L.append(f"skipped doc {d}: {why}")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ar", default=None)
    ap.add_argument("--traces", default="../01_corpus-and-spans/results/ffw_pilot_traces.parquet")
    ap.add_argument("--db", default=str(dbio.DEFAULT_DB))
    ap.add_argument("--with-docs", default="", help="doc ids always included, comma separated")
    ap.add_argument("--n-docs", type=int, default=15, help="further documents drawn at random")
    ap.add_argument("--perms", type=int, default=8, help="random orderings per document")
    ap.add_argument("--filler", default="_", help="filler token; repeated to match the word's Qwen token count")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--precision", default="bf16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rawvar-from", default="traces", choices=["traces", "store"])
    ap.add_argument("--out", default=str(HERE / "results"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--migrate", action="store_true")
    ap.add_argument("--resume-run", type=int, default=None)
    ap.add_argument("--notes", default="")
    ap.add_argument("--extra-notes", default="")
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
                  "types": len({w.lower() for v in pool.values() for _, w, _ in v}),
                  "docs": len({d for v in pool.values() for d, _, _ in v}), "cells": len(pool)}
    index = P.index_pool(pool, qtok, cache)
    print(f"pool {pool_stats['occurrences']} occurrences ({time.perf_counter() - t0:.1f}s)", flush=True)

    syn = P.load_syntax(conn)
    for r in conn.execute("SELECT doc_id, cjk_fraction FROM docs"):
        if int(r["doc_id"]) in syn["docs"]:
            syn["docs"][int(r["doc_id"])]["cjk"] = r["cjk_fraction"]
    scr = M.screen(syn, index, qtok, cache)
    want_docs, how, excluded, passing = select_docs(syn, scr, args)
    print(f"documents {want_docs}\n  {how}", flush=True)

    import spacy
    nlp = spacy.load("en_core_web_sm")
    scheme = dbio.spacy_scheme(nlp)
    if scheme != SPACY:
        print(f"WARNING: spaCy here is {scheme}, the store's labels are {SPACY}", flush=True)

    model = tok = headm = E = None
    gold_mse = M.measurable(conn)
    rawvar = rawvar_src = fp = None
    rawvar_all = {}
    if not args.dry_run:
        import torch
        import harness
        from harness import make_deterministic, fingerprint, load_ar
        assert harness.MAX_PROMPT == MAX_PROMPT, "the prompt cap has drifted"
        torch.set_num_threads(args.threads)
        make_deterministic(args.seed)
        fp = fingerprint()
        print(json.dumps(fp), flush=True)
        model, tok, headm = load_ar(args.ar, args.device, args.precision)
        E = model.get_input_embeddings().weight
        rawvar, rawvar_src, rawvar_all = M.raw_variance(conn, args.traces, args.rawvar_from)
        print(f"dFVE = -dMSE / rawvar, rawvar {rawvar:.6f} from {rawvar_src}", flush=True)

    have = {(d, a, b): s for s, (d, a, b) in syn["spans"].items()}

    if args.dry_run:
        plans, skipped, fails, token = [], [], [], {}
        for doc_id in want_docs:
            if doc_id not in gold_mse:
                skipped.append((doc_id, "no gold activation in the store")); continue
            pl = prepare(doc_id, syn, index, qtok, cache, nlp, args, have)
            if pl["skip"]:
                skipped.append((doc_id, pl["skip"])); continue
            t = time.perf_counter()
            bad = check_plan(pl, qtok, cache, args.perms)
            lens = M.prompt_lengths(qtok, pl["texts"])
            token[doc_id] = {"base": lens[0], "max": max(lens), "min": min(lens)}
            fails += [(doc_id,) + tuple(x) for x in bad[:4]]
            print(f"  doc {doc_id}: {len(pl['eligible'])} words, {pl['budget']['total']} passes, "
                  f"prompt {lens[0]} -> min {min(lens)} ({time.perf_counter() - t:.1f}s)", flush=True)
            pl["texts"] = pl["meta"] = pl["subs"] = pl["extra"] = None
            plans.append(pl)
        if not plans:
            raise SystemExit("no document survived planning")
        report = plan_report(plans, skipped, how, excluded, token, args)
        report += ("\nevery planned string round-tripped\n" if not fails
                   else "\nFAILURES:\n" + "\n".join(f"  {x}" for x in fails[:20]) + "\n")
        (out / "plan.md").write_text(report)
        print(report)
        return 1 if fails else 0

    from extract_traces import MSE_SCALE, normalize_activation
    import torch
    skip = set()
    if args.resume_run is not None:
        row = conn.execute("SELECT script FROM runs WHERE run_id = ?", (int(args.resume_run),)).fetchone()
        if row is None or "removal_curve" not in (row["script"] or ""):
            raise SystemExit(f"run {args.resume_run} is not a removal_curve run")
        run_id = int(args.resume_run)
        skip = S.already_done(conn, run_id)
        print(f"resuming run {run_id}; {len(skip)} documents already measured", flush=True)
    else:
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no CUDA device"
        notes = args.notes or (
            "Sequential removal curves. For each document every eligible word is removed one at a "
            "time until none remain, and FVE is recorded at every step. Seven curves: random order "
            "with deletion, the same random orders with corpus swap (one draw per span per "
            "permutation, held fixed along the curve), the same random orders with the filler token "
            "repeated to each word's Qwen token count, and front and back truncation by deletion and "
            "by filler. Eligibility, the swap pool and the splice are 05's and 06's; deletion is 04's "
            "rule. Measurements curve (0 random deletion, 1 random swap, 2 front deletion, 3 back "
            "deletion, 4 random filler, 5 front filler, 6 back filler), step (words removed), perm, "
            "n_words are written beside mse, fve, seq_len, dtok; the substitution rows of a variant "
            "list every removed span with depth = step.")
        run_id = DB.new_run(
            conn, script="07_removal-curve/removal_curve.py", assets=ASSETS,
            notes=notes + args.extra_notes + f" GPU: {gpu}.",
            config={"args": vars(args), "fingerprint": fp, "mse_rawvar": rawvar,
                    "mse_rawvar_from": rawvar_src, "mse_rawvar_all": dict(rawvar_all),
                    "swap_scheme": SWAP, "deletion_source": DELETION, "spacy_scheme": scheme,
                    "gpu": gpu, "pool": pool_stats, "doc_selection": how, "documents": want_docs,
                    "curves": CURVES, "perms": args.perms, "filler": args.filler,
                    "filler_source": FILLER})
        conn.commit()
    print(f"run_id {run_id}", flush=True)

    t0 = time.perf_counter()
    n_docs = n_var = 0
    for doc_id in want_docs:
        if doc_id in skip:
            print(f"  doc {doc_id}: already measured, skipped", flush=True); continue
        if doc_id not in gold_mse:
            print(f"  doc {doc_id}: no gold activation, skipped", flush=True); continue
        pl = prepare(doc_id, syn, index, qtok, cache, nlp, args, have)
        if pl["skip"]:
            print(f"  doc {doc_id}: {pl['skip']}, skipped", flush=True); continue
        bad = check_plan(pl, qtok, cache, args.perms)
        if bad:
            raise SystemExit(f"doc {doc_id}: plan check failed: {bad[:4]}")
        print(f"  doc {doc_id}: {len(pl['eligible'])} words, {pl['budget']['total']} passes", flush=True)
        gold = torch.tensor(M.gold_vector(conn, doc_id), dtype=torch.float32, device=args.device)
        gold_n = normalize_activation(gold.unsqueeze(0), MSE_SCALE)[0]
        dbio.ensure_spans(conn, doc_id, pl["text"], pl["words"], scheme, source="ffw_main_traces")
        mse, lens = M.scored(model, tok, headm, E, gold_n, pl["texts"], args.batch, args.device)
        records = [([], {"mse": mse[0], "fve": 1 - mse[0] / rawvar, "seq_len": lens[0], "dtok": 0,
                         "traces_mse": gold_mse[doc_id]})]
        for j, (subs, x) in enumerate(zip(pl["meta"], pl["extra"]), start=1):
            records.append((subs, {"mse": mse[j], "fve": 1 - mse[j] / rawvar, "seq_len": lens[j],
                                   "dtok": lens[j] - lens[0], **x}))
        dbio.write_variants(conn, doc_id, run_id, records)
        n_docs += 1
        n_var += len(records)
        print(f"  doc {doc_id} committed: {len(records)} variants, base FVE {1 - mse[0] / rawvar:.4f}, "
              f"{time.perf_counter() - t0:.0f}s elapsed", flush=True)
        pl = None
    print(f"\nrun {run_id}: {n_docs} docs, {n_var} variants ({time.perf_counter() - t0:.0f}s)")
    for k, v in DB.counts(conn).items():
        print(f"  {k:16s} {v}")
    conn.execute("PRAGMA optimize")
    conn.close()
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
