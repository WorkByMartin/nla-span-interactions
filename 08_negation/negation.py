#!/usr/bin/env python3
"""Negation flips in the verbalisation, with presence controls.

    python negation.py --dry-run --db ../db/ffw_span-ablation_database.sqlite
    python negation.py --ar "$AR" --traces ... --db ... --draws 4 --batch 16

Instances are every negator spaCy finds in a verbalised document that has a
gold activation: `neg` dependents (not, n't, never), the determiner `no`, and
the preposition `without`. Per instance the conditions are

    0 flip      not/n't deleted (the clause turned affirmative), no -> a or
                deleted by number, without -> with, never -> always
    1 del_neg   the negator deleted (only where the flip is not itself a deletion)
    2 del_gov   the governed word deleted, negator left in place
    3 swap_gov  the governed word corpus-swapped (draws draws), negator left in place

and, for documents with no negator at all, one auxiliary per document gets

    4 ins_not   " not" inserted after the auxiliary
    5 ins_ctrl  " just" inserted after the auxiliary, the presence control

Every condition is one forward pass on the reconstructor against the document's
gold activation; the document's intact string is the baseline. A variant's
substitution rows name the edited units, so the negator span and its character
offsets are recoverable from the store; measurements carry `condition`,
`instance`, `ntype` (0 not, 1 n't, 2 no, 3 without, 4 never, 5 insertion) and
`in_quote` (the negator sits inside a quoted stretch of the verbalisation).
"""
import argparse
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

import tree_vs_linear as M  # noqa: E402
import pair_ablation as P  # noqa: E402
import textsub as T  # noqa: E402
import swap_ablation as S  # noqa: E402
import db as DB  # noqa: E402
import dbio  # noqa: E402

SWAP = P.SWAP
SPACY = P.SPACY
ASSETS = P.ASSETS
MAX_PROMPT = M.MAX_PROMPT
NTYPE = {"not": 0, "n't": 1, "no": 2, "without": 3, "never": 4, "insert": 5}
COND = {0: "flip", 1: "del_neg", 2: "del_gov", 3: "swap_gov", 4: "ins_not", 5: "ins_ctrl"}
SRC = {0: "negation/flip", 1: "negation/del-neg", 2: "negation/del-gov",
       4: "negation/insert", 5: "negation/insert-ctrl"}
CONTRACTION_STEM = {"ca": "can", "wo": "will", "sha": "shall", "ai": None}
AUX = {"is", "are", "was", "were", "will", "would", "can", "could", "should", "has",
       "have", "had", "does", "do", "did", "may", "might", "must"}
QUOTES = '"“”'


def in_quote(text, idx):
    return text[:idx].count('"') % 2 == 1 or (text[:idx].count("“") > text[:idx].count("”"))


def units_of(nlp, text):
    """spaCy tokens as units in textsub's shape, all of them, and the token index map."""
    doc = nlp(text)
    units, tokmap = [], {}
    for t in doc:
        if not t.text.strip():
            continue
        s, e = t.idx, t.idx + len(t.text)
        if s > 0 and text[s - 1] == " ":
            s -= 1
        tokmap[t.i] = len(units)
        units.append({"text": t.text, "pos": t.pos_, "tag": t.tag_, "start": s, "end": e,
                      "span_text": text[s:e]})
    return doc, units, tokmap


def governed(t):
    """The word whose meaning the negator bears on."""
    h = t.head
    if t.lower_ == "without":
        for c in t.children:
            if c.dep_ == "pobj":
                return c
        return None
    if t.dep_ == "det":
        return h
    if h.pos_ == "AUX":
        for c in h.children:
            if c.dep_ in ("attr", "acomp", "prep", "advmod") and c.i > t.i:
                return c
        for c in h.head.children if h.dep_ != "ROOT" else []:
            pass
        return h.head if h.head is not h and h.head.pos_ == "VERB" else h
    return h


def find_instances(doc, text, tokmap):
    out = []
    for t in doc:
        kind = None
        if t.dep_ == "neg" and t.lower_ in ("not",):
            kind = "not"
        elif t.dep_ == "neg" and t.lower_ in ("n't", "n’t", "nt"):
            kind = "n't"
        elif t.dep_ == "neg" and t.lower_ == "never":
            kind = "never"
        elif t.lower_ == "no" and t.dep_ == "det":
            kind = "no"
        elif t.lower_ == "without" and t.pos_ == "ADP":
            kind = "without"
        if kind is None or t.i not in tokmap:
            continue
        g = governed(t)
        if g is None or g.i not in tokmap or g.i == t.i:
            continue
        inst = {"kind": kind, "neg_i": t.i, "neg_k": tokmap[t.i], "neg_text": t.text,
                "gov_i": g.i, "gov_k": tokmap[g.i], "gov_text": g.text, "gov_pos": g.pos_,
                "in_quote": in_quote(text, t.idx), "idx": t.idx}
        if kind == "n't":
            prev = doc[t.i - 1] if t.i > 0 else None
            if prev is None or prev.i not in tokmap:
                continue
            stem = CONTRACTION_STEM.get(prev.lower_, prev.text)
            if stem is None:
                continue
            inst["prev_k"] = tokmap[prev.i]
            inst["prev_fix"] = None if stem == prev.text else stem
        if kind == "no":
            inst["flip_to"] = "a" if g.tag_ in ("NN", "NNP") else None
        out.append(inst)
    return out


def find_insertion(doc, tokmap):
    for t in doc:
        if t.pos_ == "AUX" and t.lower_ in AUX and t.i in tokmap:
            if any(c.dep_ == "neg" for c in t.children) or any(c.dep_ == "neg" for c in t.head.children):
                continue
            nxt = doc[t.i + 1] if t.i + 1 < len(doc) else None
            if nxt is not None and nxt.lower_ in ("not", "n't", "never"):
                continue
            return {"kind": "insert", "aux_i": t.i, "aux_k": tokmap[t.i], "aux_text": t.text,
                    "gov_i": t.head.i, "gov_text": t.head.text, "in_quote": in_quote(doc.text, t.idx),
                    "idx": t.idx}
    return None


def apply_edits(text, spans, edits):
    """edits: {unit index: None to delete, or the bare replacement word}. Right to left."""
    out = text
    for k in sorted(edits, reverse=True):
        if edits[k] is None:
            out = S.delete_span(out, spans, k)
        else:
            out = T.splice(out, spans, k, T.fit_space(text, spans[k], edits[k]))[0]
    return out


def plan_document(doc_id, text, units, insts, ins, sids, elig, index, seed, draws):
    """(texts, meta, extra). texts[0] intact."""
    spans = [(u["start"], u["end"]) for u in units]
    texts, meta, extra = [text], [], []

    def add(edits, cond, inst_id, ntype, inq, subs_extra=None, draw_idx=0, prob=None):
        texts.append(apply_edits(text, spans, edits))
        rows = []
        for k, w in edits.items():
            rows.append({"span_id": sids[k], "substitute": "" if w is None else w,
                         "source": SWAP if cond == 3 else SRC[cond], "depth": len(edits),
                         "draw_idx": draw_idx, "prob": prob})
        meta.append(rows)
        extra.append({"condition": cond, "instance": inst_id, "ntype": ntype, "in_quote": int(inq)})

    for inst_id, x in insts:
        nt = NTYPE[x["kind"]]
        nk, gk = x["neg_k"], x["gov_k"]
        if x["kind"] in ("not", "never") and x["kind"] == "not":
            add({nk: None}, 0, inst_id, nt, x["in_quote"])
        elif x["kind"] == "n't":
            e = {nk: None}
            if x["prev_fix"]:
                e[x["prev_k"]] = x["prev_fix"]
            add(e, 0, inst_id, nt, x["in_quote"])
        elif x["kind"] == "never":
            add({nk: "always"}, 0, inst_id, nt, x["in_quote"])
            add({nk: None}, 1, inst_id, nt, x["in_quote"])
        elif x["kind"] == "no":
            add({nk: x["flip_to"]}, 0, inst_id, nt, x["in_quote"])
            if x["flip_to"] is not None:
                add({nk: None}, 1, inst_id, nt, x["in_quote"])
        elif x["kind"] == "without":
            add({nk: "with"}, 0, inst_id, nt, x["in_quote"])
            add({nk: None}, 1, inst_id, nt, x["in_quote"])
        add({gk: None}, 2, inst_id, nt, x["in_quote"])
        sid = sids[gk]
        if sid in elig:
            cands, prob = P.draw_substitutes(elig[sid], index, seed, draws)
            for i, w in enumerate(cands):
                add({gk: w}, 3, inst_id, nt, x["in_quote"], draw_idx=i, prob=prob)
    if ins is not None:
        inst_id = -doc_id
        k = ins["aux_k"]
        add({k: units[k]["text"] + " not"}, 4, inst_id, NTYPE["insert"], ins["in_quote"])
        add({k: units[k]["text"] + " just"}, 5, inst_id, NTYPE["insert"], ins["in_quote"])
    return texts, meta, extra


def prepare(doc_id, syn, index, qtok, cache, nlp, args, have, next_id):
    text = syn["docs"][doc_id]["text"]
    if len(T.prompt_ids(qtok, text)) > MAX_PROMPT:
        return None
    doc, units, tokmap = units_of(nlp, text)
    insts = find_instances(doc, text, tokmap)
    ins = None if insts else find_insertion(doc, tokmap)
    if not insts and ins is None:
        return None
    if args.max_insertions is not None and ins is not None and next_id["ins"] >= args.max_insertions:
        return None
    elig, _ = M.doc_eligibility(syn, index, qtok, cache, doc_id)
    keys = [(doc_id,) + dbio.bare_span(u) for u in units]
    sids = [have.get(k) for k in keys]          # None until ensure_spans at run time
    numbered = []
    for x in insts:
        numbered.append((next_id["inst"], x))
        next_id["inst"] += 1
    if ins is not None:
        next_id["ins"] += 1
    return {"doc_id": doc_id, "text": text, "units": units, "sids": sids, "insts": numbered,
            "ins": ins, "elig": elig}


def build(pl, index, seed, draws):
    texts, meta, extra = plan_document(pl["doc_id"], pl["text"], pl["units"], pl["insts"],
                                       pl["ins"], pl["sids"], pl["elig"], index, seed, draws)
    pl["texts"], pl["meta"], pl["extra"] = texts, meta, extra


def check(pl):
    bad = []
    if pl["texts"][0] != pl["text"]:
        bad.append("baseline")
    for t, m, x in zip(pl["texts"][1:], pl["meta"], pl["extra"]):
        if t == pl["text"]:
            bad.append(("edit is a no-op", x))
        if any(len(s["substitute"]) > 60 for s in m):
            bad.append(("long substitute", x))
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ar", default=None)
    ap.add_argument("--traces", default="../01_corpus-and-spans/results/ffw_pilot_traces.parquet")
    ap.add_argument("--db", default=str(dbio.DEFAULT_DB))
    ap.add_argument("--draws", type=int, default=4, help="corpus-swap draws of the governed word")
    ap.add_argument("--max-insertions", type=int, default=200)
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
    pool = S.build_pool(conn, qtok, cache)
    index = P.index_pool(pool, qtok, cache)
    syn = P.load_syntax(conn)
    import spacy
    nlp = spacy.load("en_core_web_sm")
    scheme = dbio.spacy_scheme(nlp)
    gold_mse = M.measurable(conn)
    have = {(d, a, b): s for s, (d, a, b) in syn["spans"].items()}

    model = tok = headm = E = None
    rawvar = rawvar_src = fp = None
    rawvar_all = {}
    if not args.dry_run:
        import torch
        import harness
        from harness import make_deterministic, fingerprint, load_ar
        assert harness.MAX_PROMPT == MAX_PROMPT
        torch.set_num_threads(args.threads)
        make_deterministic(args.seed)
        fp = fingerprint()
        model, tok, headm = load_ar(args.ar, args.device, args.precision)
        E = model.get_input_embeddings().weight
        rawvar, rawvar_src, rawvar_all = M.raw_variance(conn, args.traces, args.rawvar_from)
        print(f"rawvar {rawvar:.6f} from {rawvar_src}", flush=True)

    # plan everything first: instance ids are assigned in doc order and must be stable
    t0 = time.perf_counter()
    next_id = {"inst": 1, "ins": 0}
    plans = []
    for doc_id in sorted(syn["docs"]):
        if doc_id not in gold_mse:
            continue
        pl = prepare(doc_id, syn, index, qtok, cache, nlp, args, have, next_id)
        if pl is not None:
            plans.append(pl)
    kinds = Counter(x["kind"] for pl in plans for _, x in pl["insts"])
    n_ins = sum(pl["ins"] is not None for pl in plans)
    print(f"{len(plans)} documents, {sum(kinds.values())} negator instances {dict(kinds)}, "
          f"{n_ins} insertion documents ({time.perf_counter() - t0:.0f}s)", flush=True)

    if args.dry_run:
        total, fails = 0, []
        missing = 0
        for pl in plans:
            build(pl, index, args.seed, args.draws)
            fails += [(pl["doc_id"],) + (b if isinstance(b, tuple) else (b,)) for b in check(pl)]
            total += len(pl["texts"])
            missing += sum(s is None for s in pl["sids"])
        conds = Counter(x["condition"] for pl in plans for x in pl["extra"])
        lines = ["# Negation plan", "", f"{len(plans)} documents, {sum(kinds.values())} negators {dict(kinds)}, "
                 f"{n_ins} insertion documents", f"passes {total} ({len(plans)} baselines)",
                 f"conditions {{{', '.join(f'{COND[c]}: {n}' for c, n in sorted(conds.items()))}}}",
                 f"units without a stored span yet {missing} (created at run time)", "",
                 "every edit is a real change" if not fails else "FAILURES:\n" + "\n".join(map(str, fails[:20]))]
        (out / "plan.md").write_text("\n".join(lines) + "\n")
        print("\n".join(lines))
        return 1 if fails else 0

    from extract_traces import MSE_SCALE, normalize_activation
    import torch
    skip = set()
    if args.resume_run is not None:
        row = conn.execute("SELECT script FROM runs WHERE run_id = ?", (int(args.resume_run),)).fetchone()
        if row is None or "negation" not in (row["script"] or ""):
            raise SystemExit(f"run {args.resume_run} is not a negation run")
        run_id = int(args.resume_run)
        skip = S.already_done(conn, run_id)
    else:
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no CUDA device"
        notes = args.notes or (
            "Negation flips. Every negator in a verbalised document (not, n't, never, determiner no, "
            "without) is flipped to its affirmative form, and compared against deleting the negator, "
            "deleting the governed word and corpus-swapping the governed word. Documents without a "
            "negator get ' not' inserted after one auxiliary, against ' just' as the presence control. "
            "Measurements condition (0 flip, 1 del_neg, 2 del_gov, 3 swap_gov, 4 ins_not, 5 ins_ctrl), "
            "instance, ntype (0 not, 1 n't, 2 no, 3 without, 4 never, 5 insertion), in_quote.")
        run_id = DB.new_run(conn, script="08_negation/negation.py", assets=ASSETS,
                            notes=notes + args.extra_notes + f" GPU: {gpu}.",
                            config={"args": vars(args), "fingerprint": fp, "mse_rawvar": rawvar,
                                    "mse_rawvar_from": rawvar_src, "swap_scheme": SWAP,
                                    "spacy_scheme": scheme, "gpu": gpu, "conditions": COND,
                                    "ntype": NTYPE, "kinds": dict(kinds), "insertion_docs": n_ins})
        conn.commit()
    print(f"run_id {run_id}", flush=True)

    t0 = time.perf_counter()
    n_docs = n_var = 0
    for pl in plans:
        doc_id = pl["doc_id"]
        if doc_id in skip:
            continue
        pl["sids"] = dbio.ensure_spans(conn, doc_id, pl["text"], pl["units"], scheme, source="ffw_main_traces")
        build(pl, index, args.seed, args.draws)
        bad = check(pl)
        if bad:
            raise SystemExit(f"doc {doc_id}: {bad[:4]}")
        gold = torch.tensor(M.gold_vector(conn, doc_id), dtype=torch.float32, device=args.device)
        gold_n = normalize_activation(gold.unsqueeze(0), MSE_SCALE)[0]
        mse, lens = M.scored(model, tok, headm, E, gold_n, pl["texts"], args.batch, args.device)
        records = [([], {"mse": mse[0], "fve": 1 - mse[0] / rawvar, "seq_len": lens[0], "dtok": 0,
                         "traces_mse": gold_mse[doc_id]})]
        for j, (subs, x) in enumerate(zip(pl["meta"], pl["extra"]), start=1):
            records.append((subs, {"mse": mse[j], "fve": 1 - mse[j] / rawvar, "seq_len": lens[j],
                                   "dtok": lens[j] - lens[0], **x}))
        dbio.write_variants(conn, doc_id, run_id, records)
        n_docs += 1
        n_var += len(records)
        if n_docs % 20 == 0:
            print(f"  {n_docs} docs, {n_var} variants, {time.perf_counter() - t0:.0f}s", flush=True)
        pl["texts"] = pl["meta"] = None
    print(f"\nrun {run_id}: {n_docs} docs, {n_var} variants ({time.perf_counter() - t0:.0f}s)")
    for k, v in DB.counts(conn).items():
        print(f"  {k:16s} {v}")
    conn.execute("PRAGMA optimize")
    conn.close()
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
