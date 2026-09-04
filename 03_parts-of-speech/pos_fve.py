#!/usr/bin/env python3
"""Delta FVE of a word-level ablation, by part of speech and by candidate depth.

Spans and candidate distributions come from the SQLite store, not from a JSON
side file, and every forward pass is written back to it as a variant with its
substitution and its measurements. Candidates absent from the store are computed
once with ModernBERT and saved, so the second run over a document is cheap.

The replacement is spliced into the explanation TEXT and the whole templated
prompt is re-tokenised, so what the reconstructor reads is always Qwen's own
canonical tokenisation of the string that is actually there. The sequence length
may change; it is recorded per variant as the seq_len metric.

Depth is the axis of interest. Draws come from the top-k of the masked-fill
distribution renormalised over k, so k controls how far into the tail a
substitute may come from. A closed class runs out of same-class candidates
almost immediately and its reconstruction should fall away as k grows; an open
class should not.

The per-document baseline rides in the same batched call as the arms it is
differenced against, so it comes down the same numerical path. The previous
version scored it alone at batch size 1.

    python pos_fve.py --ar "$ASSETS/qwen36-27b_ar-l43-s600_model" \
        --traces ../01_corpus-and-spans/results/ffw_pilot_traces.parquet \
        --db ../db/ffw_span-ablation_database.sqlite \
        --docs 40 --per-doc 40 --draws 8 --batch 8
"""
import argparse, json, os, random, sys, time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import pyarrow.parquet as pq
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from harness import (  # noqa: E402
    MAX_PROMPT, Sampler, make_deterministic, fingerprint, load_ar, mse_of_texts)
import textsub as T  # noqa: E402

sys.path.insert(0, str(HERE.parent / "01_corpus-and-spans"))
from extract_traces import MSE_SCALE, normalize_activation  # noqa: E402


def _db_package():
    """Where the SQLite store's code lives.

    Checked rather than assumed because a GPU host stages this directory on its
    own, so `db/` may sit beside it or inside it rather than one level up.
    """
    tried = [HERE.parent / "db", HERE / "db", HERE.parent.parent / "db"]
    for c in tried:
        if (c / "dbio.py").is_file():
            return c
    raise SystemExit("cannot find the store: no dbio.py in " +
                     ", ".join(str(c) for c in tried) +
                     ". Copy db/ across alongside this directory.")


sys.path.insert(0, str(_db_package()))
import db as DB  # noqa: E402
import dbio  # noqa: E402

DEPTHS = [1, 3, 5, 10, 25, 50]

ASSETS = ["ffw_main_traces", "qwen36-27b_ar-l43-s600_model",
          "qwen36-27b_tokenizer", "modernbert-large_filler_model"]


def stratified(units, per_doc, rng):
    """Sample words spread across POS classes rather than by frequency.

    Taking a uniform sample would be three quarters nouns, punctuation and
    verbs, and the closed classes this study is about would barely appear.
    """
    by = defaultdict(list)
    for u in units:
        by[u["pos"]].append(u)
    for v in by.values():
        rng.shuffle(v)
    out, i = [], 0
    while len(out) < per_doc and any(len(v) > i for v in by.values()):
        for pos in sorted(by):
            if len(by[pos]) > i and len(out) < per_doc:
                out.append(by[pos][i])
        i += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ar", required=True)
    ap.add_argument(
        "--traces",
        default="../01_corpus-and-spans/results/ffw_pilot_traces.parquet")
    ap.add_argument("--db", default=str(dbio.DEFAULT_DB))
    ap.add_argument("--docs", type=int, default=40)
    ap.add_argument("--per-doc", type=int, default=40)
    ap.add_argument("--draws", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--depths", default=",".join(map(str, DEPTHS)))
    ap.add_argument("--precision", default="bf16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--migrate", action="store_true",
                    help="apply outstanding migrations to --db first. Off by "
                         "default: a schema change is Marty's call")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    depths = [int(x) for x in args.depths.split(",")]
    topk = max(depths)
    torch.set_num_threads(args.threads)
    make_deterministic(args.seed)
    device = args.device
    fp = fingerprint()
    print(json.dumps(fp), flush=True)

    import spacy
    nlp = spacy.load("en_core_web_sm")
    conn = dbio.open_db(args.db, allow_migrate=args.migrate)
    scheme = dbio.spacy_scheme(nlp)

    model, tok, head = load_ar(args.ar, device, args.precision)
    E = model.get_input_embeddings().weight
    rows = pq.read_table(args.traces).to_pylist()
    rng = random.Random(args.seed)

    # FVE needs the raw-variance baseline of the normalised gold distribution,
    # which is a property of the corpus rather than of any one document
    G = torch.tensor([r["activation"] for r in rows], dtype=torch.float64)
    Gn = G / G.norm(dim=1, keepdim=True) * MSE_SCALE
    rawvar = float(((Gn - Gn.mean(0)) ** 2).mean())
    print(f"mse_rawvar {rawvar:.5f}  dFVE = -dMSE / rawvar", flush=True)

    run_id = DB.new_run(
        conn, script="03_parts-of-speech/pos_fve.py", assets=ASSETS,
        notes=args.notes or (
            "Word-level single-span ablation, dFVE by POS class and candidate "
            "depth. Text-space substitution: the substitute string is spliced "
            "into the explanation and the whole templated prompt is "
            "re-tokenised, so sequence length can move and is recorded as "
            "seq_len. The baseline rides in the same batched call as the arms."),
        config={"args": vars(args), "fingerprint": fp, "mse_rawvar": rawvar,
                "depths": depths, "filler_scheme": dbio.FILLER,
                "spacy_scheme": scheme})
    conn.commit()
    print(f"run_id {run_id}", flush=True)

    sampler = None
    t0 = time.perf_counter()
    n_docs = n_var = n_new_cand = 0

    for row in rows[: args.docs]:
        doc_id, text = row["doc_uid"], row["explanation"]
        if len(T.prompt_ids(tok, text)) > MAX_PROMPT:
            continue
        gold = torch.tensor(row["activation"], dtype=torch.float32, device=device)
        gold_n = normalize_activation(gold.unsqueeze(0), MSE_SCALE)[0]

        words = T.word_spans(nlp, text, lexical_only=True)
        spans = [(w["start"], w["end"]) for w in words]
        span_ids = dbio.ensure_spans(conn, doc_id, text, words, scheme,
                                     source="ffw_main_traces")

        cond = dbio.load_conditionals(conn, span_ids, dbio.FILLER, topk)
        need = [i for i, c in enumerate(cond) if c is None]
        if need:
            if sampler is None:
                sampler = Sampler(device, topk=topk, draws=args.draws)
            fresh = sampler.conditionals(text, [spans[i] for i in need])
            n_new_cand += dbio.save_conditionals(
                conn, [span_ids[i] for i in need], fresh, dbio.FILLER)
            for i, c in zip(need, fresh):
                cond[i] = (None if c is None else
                           ([DB.bare(s) for s in c["strs"]],
                            [float(p) for p in c["probs"]]))

        live = [i for i, c in enumerate(cond) if c]
        if not live:
            continue
        picks = stratified([{"k": i, "pos": words[i]["pos"]} for i in live],
                           args.per_doc, rng)

        # one list of texts for the whole document, baseline first, so the
        # baseline is padded and batched exactly like the arms
        texts, meta = [text], []
        for u in picks:
            k = u["k"]
            strs, probs = cond[k]
            for d in depths:
                sub_s, sub_p = strs[:d], probs[:d]
                tot = sum(sub_p)
                if tot <= 0:
                    continue
                w = [p / tot for p in sub_p]
                for i, ci in enumerate(rng.choices(range(len(sub_s)),
                                                   weights=w, k=args.draws)):
                    s = sub_s[ci]
                    texts.append(T.splice(
                        text, spans, k, T.fit_space(text, spans[k], s))[0])
                    meta.append((k, d, i, s, w[ci]))

        vals, lens = mse_of_texts(model, tok, head, E, gold_n, texts,
                                  args.batch, device)
        mse = [float(v) for v in vals]
        records = [([], {"mse": mse[0], "fve": 1 - mse[0] / rawvar,
                         "seq_len": lens[0], "traces_mse": row["mse"]})]
        for j, (k, d, i, s, p) in enumerate(meta, start=1):
            records.append(([{"span_id": span_ids[k], "substitute": s,
                              "source": dbio.FILLER, "depth": d,
                              "draw_idx": i, "prob": p}],
                            {"mse": mse[j], "fve": 1 - mse[j] / rawvar,
                             "seq_len": lens[j]}))
        dbio.write_variants(conn, doc_id, run_id, records)
        n_docs += 1
        n_var += len(records)
        print(f"  doc {doc_id}  {len(picks)} units  {len(records)} variants  "
              f"base FVE {1 - mse[0] / rawvar:.4f}  "
              f"{time.perf_counter() - t0:.0f}s elapsed", flush=True)

    print(f"\nrun {run_id}: {n_docs} docs, {n_var} variants, "
          f"{n_new_cand} candidate rows added ({time.perf_counter() - t0:.0f}s)")
    for k, v in DB.counts(conn).items():
        print(f"  {k:16s} {v}")
    conn.execute("PRAGMA optimize")
    conn.close()


if __name__ == "__main__":
    main()
