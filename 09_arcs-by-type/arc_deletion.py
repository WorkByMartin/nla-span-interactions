#!/usr/bin/env python3
"""Arc deletion: delete each word of a dependency arc, singly and together.

    python arc_deletion.py --dry-run --db ../db/ffw_span-ablation_database.sqlite

    python arc_deletion.py --ar "$AR" \
        --traces ../01_corpus-and-spans/results/ffw_pilot_traces.parquet \
        --db ../db/ffw_span-ablation_database.sqlite --per-type 450 \
        --batch 16 --precision bf16 --threads 8 --seed 0

The unit is 05's, the edit is 07's. Two words are ablated singly and together so
the interaction

    interaction = e(a) + e(b) - e(both)

is formed per pair, where e is FVE points lost against the same document's
unedited baseline, and a positive interaction means the pair costs less than the
sum of its singles. The edit is deletion under 04's rule, applied right to left
by 07's `delete_many`, so the word and its folded leading space go and no
doubled space is left behind. Deletion is deterministic, so there are no draws
and no swap pool: a span contributes exactly one single however many pairs use
it, which is what `v_pair` needs to difference a joint against its two singles.

Pairs may be adjacent. `--min-distance` defaults to 1, which overrides 05's
floor of 2, so a head and its dependent standing side by side are measured like
any other arc and their controls are drawn at the same distance. Two adjacent
words are one contiguous region of the document, and the round-trip check tests
that region against its own one-cut oracle as well as against the right-to-left
composition, so the space rule is verified rather than assumed.

Arcs are every `head` relation under `--scheme` whose dependent carries one of
the RELATIONS below, read from the store. `--arcs` replaces them with a JSONL of
one arc per line (`doc_id`, `span_a` the dependent, `span_b` its head,
`deprel`). Each arc is matched to a control pair in
the same document, relaxing in this order:

    exact        no arc, sibling or grandparent link between the two words;
                 same token distance; same Qwen token count word for word; same
                 in-quote status word for word; the pair's mean position within
                 20% of the arc's
    dist+-1      the same, with the distance one token either side, never
                 below the minimum, so an arc at distance 1 relaxes to 1 or 2
    no-position  the position constraint dropped
    no-quote     the in-quote constraint dropped as well
    no-qwen      the token-count constraint dropped as well, leaving the graph
                 and distance constraints alone

The graph constraint is 05's `graph_neighbours` and is never relaxed.
Part of speech is NOT matched on, because it is a covariate of this experiment
rather than a nuisance to be balanced away.

Recorded per pair, for the post-hoc analysis: both words' UPOS, both words'
Qwen token counts, both words' in-quote status, whether either word repeats the
source document's final token, and how many exact copies of each word sit
elsewhere in the verbalisation. The numeric ones ride on the joint variant as
measurements; the strings sit in the run's `config` pair table, keyed by
`pair_id`, and the pair set also goes into `relations` under the scheme
`arc-deletion/arc+control`.

Spans, their offsets and their labels are read from the store under `--scheme`,
so no parser runs here and the script is parser-agnostic: spaCy and Stanza
schemes go down the same path.

--dry-run plans and round-trips every string, writes results/plan.md with the
counts per type, the unmatched arcs and the covariate distributions, and loads
no reconstructor.
"""
import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


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


for _marker, _dir in (("harness.py", "03_parts-of-speech"),
                      ("swap_ablation.py", "04_ablation-strategy"),
                      ("pair_ablation.py", "05_dependent-pair"),
                      ("tree_vs_linear.py", "06_tree-vs-linear"),
                      ("removal_curve.py", "07_removal-curve"),
                      ("negation.py", "08_negation"),
                      ("extract_traces.py", "01_corpus-and-spans"),
                      ("dbio.py", "db")):
    sys.path.insert(0, str(_package(_marker, REPO / _dir, HERE, HERE / "code")))

import textsub as T  # noqa: E402
import swap_ablation as S  # noqa: E402
import pair_ablation as P  # noqa: E402
import tree_vs_linear as M  # noqa: E402
import removal_curve as R  # noqa: E402
import negation as N  # noqa: E402
import db as DB  # noqa: E402
import dbio  # noqa: E402

DELETION = S.DELETION                       # 'deletion'
SPACY = P.SPACY                             # 'spacy-en_core_web_sm-3.8.0'
STANZA = "stanza-en_combined_charlm-1.14.0"
PAIR_SCHEME = "arc-deletion/arc+control"
MAX_PROMPT = M.MAX_PROMPT
# 05 floors the token distance at 2 so no pair is a contiguous bigram. Here the
# floor is 1: an adjacent head and dependent is the commonest shape of several
# arc types, and dropping it would measure det, amod and aux only where they are
# unusually far from their head. 05's module is not touched; this is the
# harness's own floor, and --min-distance overrides it.
MIN_DISTANCE = 1
MAX_POSITION_DELTA = 0.20                   # of the document's words

# the Universal Dependencies relations measured, dependent to head
RELATIONS = ["nsubj", "nsubj:pass", "obj", "iobj", "obl", "nmod", "nmod:poss",
             "amod", "advmod", "acl", "acl:relcl", "advcl", "xcomp", "ccomp",
             "conj", "compound", "appos", "flat", "parataxis", "det", "case",
             "mark", "aux", "aux:pass", "cop", "cc"]

ASSETS = ["ffw_span-ablation_database", "qwen36-27b_ar-l43-s600_model",
          "qwen36-27b_tokenizer"]

# the order the control fallbacks are tried in, best first
QUALITIES = ["exact", "dist+-1", "no-position", "no-quote", "no-qwen"]

# measured throughput of the deletion and swap runs on an H100 80GB HBM3 at
# batch 16: run 9 did 71784 variants in 2437 s, run 8 336301 in 11262 s
REF_PASSES_PER_SEC = 28.0
REF_NOTE = ("run 9's 29.5 passes/s and run 8's 29.9 on an NVIDIA H100 80GB "
            "HBM3 at batch 16, rounded down to 28")


# ------------------------------------------------------------------ the store

def load_syntax(conn, scheme):
    """05's reader, tolerant of a decomposer that names its keys the UD way.

    spaCy's loader writes `pos` and `dep`; a Stanza loader may write `upos` and
    `deprel` instead. Whichever pair is present is read into the same fields, so
    nothing downstream has to know which parser produced the labels.
    """
    syn = P.load_syntax(conn, scheme)
    for want, alt in (("pos", "upos"), ("dep", "deprel")):
        if syn[want]:
            continue
        syn[want] = {int(r["span_id"]): r["value"] for r in conn.execute(
            "SELECT span_id, value FROM labels WHERE scheme = ? AND key = ?",
            (scheme, alt))}
    if not syn["tok_i"]:
        raise SystemExit(f"no tok_i labels under scheme {scheme!r}: the pair "
                         f"search has no token distance to work with")
    return syn


def spans_by_doc(syn):
    """{doc_id: {span_id: (char_start, char_end)}}, built once over the store."""
    out = defaultdict(dict)
    for sid, (doc_id, a, b) in syn["spans"].items():
        out[doc_id][sid] = (a, b)
    return dict(out)


def words_from_store(text, doc_spans):
    """(units, span_ids) in textsub's shape, straight from the store's offsets.

    The store holds bare words and textsub's units carry the preceding space
    folded in, so the fold is redone here rather than by re-running a parser.
    That keeps the splice offsets exact under any label scheme, and removes
    05's and 07's failure mode where a local parse does not reproduce a span.
    """
    words, sids = [], []
    for sid, (a, b) in sorted(doc_spans.items(), key=lambda kv: kv[1]):
        s = a - 1 if a > 0 and text[a - 1] == " " else a
        words.append({"text": text[a:b], "start": s, "end": b,
                      "span_text": text[s:b]})
        sids.append(sid)
    return words, sids


# ----------------------------------------------------------- the final token

# The verbaliser closes many explanations by naming the source document's last
# token and quoting the tail it ends. Both forms are picked up, because a word
# that merely repeats that token is redundant with it and its deletion is a
# different kind of edit from deleting a word that appears once.
FINAL_RE = re.compile(
    r'\bfinal\s+(?:token|word|phrase|fragment|clause|feature|sentence|line|'
    r'segment)s?\b[^"“\n]{0,40}["“]([^"”\n]{1,200})["”]',
    re.I)
QUOTED_RE = re.compile(r'["“]([^"”]{1,4000})["”]')
_WORD_RE = re.compile(r"[^\W_]+(?:['\-][^\W_]+)*", re.UNICODE)


def final_forms(text):
    """Casefolded word forms this verbalisation presents as the source's tail.

    The last `Final token "..."` sentence, and the last quoted stretch, each
    contribute their own final word. Either may be absent; a document with
    neither contributes nothing and every word in it reads as no copy.
    """
    out = set()
    for rx in (FINAL_RE, QUOTED_RE):
        hits = rx.findall(text)
        if not hits:
            continue
        ws = _WORD_RE.findall(hits[-1])
        if ws:
            out.add(ws[-1].lower())
    return out


def n_copies(text_lower, word):
    """Exact copies of this word ELSEWHERE in the verbalisation.

    A whole-word regex, case blind, less the occurrence being counted.
    """
    w = word.strip().lower()
    if not w:
        return 0
    n = len(re.findall(r"(?<!\w)" + re.escape(w) + r"(?!\w)", text_lower))
    return max(0, n - 1)


# ------------------------------------------------------------- span eligibility

def deletable_spans(syn, qtok, cache):
    """{span_id: info} for every span this experiment may delete.

    Lexical POS, a clean word, and a token index, which is 05's test with the
    swap pool dropped: deletion needs no substitute, so pool coverage cannot
    disqualify a word. `info` carries everything both the control search and the
    covariate table need, so neither recomputes it.
    """
    out, rejected = {}, Counter()
    lowered = {}
    for sid, (doc_id, a, b) in syn["spans"].items():
        d = syn["docs"].get(doc_id)
        if d is None:
            rejected["no document"] += 1
            continue
        pos = syn["pos"].get(sid)
        if pos is None or pos in T.NON_LEXICAL_POS:
            rejected["non-lexical POS"] += 1
            continue
        tok_i = syn["tok_i"].get(sid)
        if tok_i is None:
            rejected["no token index"] += 1
            continue
        text = d["text"]
        word = text[a:b]
        if not T.is_clean_word(word):
            rejected["not a clean word"] += 1
            continue
        if doc_id not in lowered:
            lowered[doc_id] = text.lower()
            d["final_forms"] = final_forms(text)
        has_space = a > 0 and text[a - 1] == " "
        out[sid] = {"span_id": sid, "doc_id": doc_id, "start": a, "end": b,
                    "text": word, "pos": pos, "dep": syn["dep"].get(sid),
                    "tok_i": tok_i, "has_space": has_space,
                    "n_qwen": S.qwen_len(qtok, cache, word, has_space),
                    "in_quote": int(N.in_quote(text, a)),
                    "final_copy": int(word.lower() in d["final_forms"]),
                    "copies": n_copies(lowered[doc_id], word)}
    return out, rejected


# ------------------------------------------------------------------ the arcs

def arcs_from_parse(syn, relations=None):
    """Every head relation in the scheme whose dependent carries a wanted label."""
    want = set(relations or RELATIONS)
    return [{"doc_id": syn["spans"][a][0], "span_a": a, "span_b": h,
             "deprel": syn["dep"][a]}
            for a, h in sorted(syn["head"].items())
            if syn["dep"].get(a) in want and a in syn["spans"] and h in syn["spans"]]


def read_arcs(path):
    """An arc list. One JSON object per line, blank lines ignored."""
    rows = []
    for n, line in enumerate(Path(path).read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except ValueError as e:
            raise SystemExit(f"{path}:{n}: not JSON ({e})")
        missing = {"doc_id", "span_a", "span_b", "deprel"} - set(r)
        if missing:
            raise SystemExit(f"{path}:{n}: missing {sorted(missing)}")
        rows.append({"doc_id": int(r["doc_id"]), "span_a": int(r["span_a"]),
                     "span_b": int(r["span_b"]), "deprel": str(r["deprel"])})
    return rows


def screen_arcs(arcs, elig, syn, min_distance=MIN_DISTANCE):
    """Keep the arcs this harness can actually measure, and say why the rest go."""
    keep, why = [], Counter()
    seen = set()
    for r in arcs:
        key = (r["doc_id"], min(r["span_a"], r["span_b"]),
               max(r["span_a"], r["span_b"]))
        if key in seen:
            why["duplicate pair"] += 1
            continue
        a, b = elig.get(r["span_a"]), elig.get(r["span_b"])
        if a is None or b is None:
            why["endpoint not deletable"] += 1
            continue
        if a["doc_id"] != b["doc_id"] or a["doc_id"] != r["doc_id"]:
            why["endpoints in different documents"] += 1
            continue
        if syn["head"].get(r["span_a"]) != r["span_b"]:
            why["not a head arc in this scheme"] += 1
            continue
        dist = abs(a["tok_i"] - b["tok_i"])
        if dist < min_distance:
            why[f"closer than {min_distance} token(s)"] += 1
            continue
        seen.add(key)
        dep_first = a["tok_i"] < b["tok_i"]
        keep.append({**r, "distance": dist, "dep_first": dep_first,
                     "first": r["span_a"] if dep_first else r["span_b"],
                     "second": r["span_b"] if dep_first else r["span_a"]})
    return keep, why


# ------------------------------------------------------------- control search

def doc_words(elig):
    """{doc_id: {tok_i: info}} plus {doc_id: n_words}, over deletable spans."""
    by_tok, n_words = defaultdict(dict), Counter()
    for info in elig.values():
        by_tok[info["doc_id"]][info["tok_i"]] = info
        n_words[info["doc_id"]] += 1
    return dict(by_tok), dict(n_words)


def position(u, v, n_words):
    """Where the pair sits in the document, as a fraction of its words."""
    return (u["tok_i"] + v["tok_i"]) / 2.0 / max(1, n_words)


def find_control(arc, elig, tokmap, n_words, head, used, rng,
                 min_distance=MIN_DISTANCE):
    """A matched control pair for one arc, at the best quality available.

    Candidates are drawn by walking the document's token indices once per
    distance, rather than by enumerating every pair of the document, so the
    search is linear in the document's words and not quadratic. The walk takes
    any distance at or above the floor, so at a floor of 1 it also pairs
    neighbouring words; `d0 - 1` is dropped when it would fall below the floor,
    which is what keeps an arc at distance 1 from asking for a distance of 0.
    """
    a, b = elig[arc["first"]], elig[arc["second"]]
    d0 = arc["distance"]
    doc_id = arc["doc_id"]
    p0 = position(a, b, n_words)
    near = [d for d in (d0, d0 - 1, d0 + 1) if d >= min_distance]

    def pool(dists, qwen, quote, pos):
        take = []
        for dist in dists:
            for t in sorted(tokmap):
                u, v = tokmap.get(t), tokmap.get(t + dist)
                if v is None:
                    continue
                if (doc_id, u["span_id"], v["span_id"]) in used:
                    continue
                if P.graph_neighbours(head, u["span_id"], v["span_id"]) is not None:
                    continue
                if qwen and (u["n_qwen"] != a["n_qwen"]
                             or v["n_qwen"] != b["n_qwen"]):
                    continue
                if quote and (u["in_quote"] != a["in_quote"]
                              or v["in_quote"] != b["in_quote"]):
                    continue
                if pos and abs(position(u, v, n_words) - p0) > MAX_POSITION_DELTA:
                    continue
                take.append((u["span_id"], v["span_id"], dist))
        return take

    for quality, dists, qwen, quote, pos in (
            ("exact", [d0] if d0 >= min_distance else [], True, True, True),
            ("dist+-1", near, True, True, True),
            ("no-position", near, True, True, False),
            ("no-quote", near, True, False, False),
            ("no-qwen", near, False, False, False)):
        if not dists:
            continue
        take = pool(dists, qwen, quote, pos)
        if take:
            u, v, dist = take[rng.randrange(len(take))]
            return u, v, quality, dist
    return None


def covariates(u, v, n_words):
    """The per-word record the post-hoc analysis needs, in position order."""
    return {"nqwen_first": u["n_qwen"], "nqwen_second": v["n_qwen"],
            "copies_first": u["copies"], "copies_second": v["copies"],
            "final_copy_first": u["final_copy"],
            "final_copy_second": v["final_copy"],
            "in_quote_first": u["in_quote"], "in_quote_second": v["in_quote"],
            "tok_i_first": u["tok_i"], "tok_i_second": v["tok_i"],
            "n_words": n_words, "position": position(u, v, n_words)}


def sample_pairs(arcs, elig, syn, per_type, deps, seed,
                 min_distance=MIN_DISTANCE):
    """The whole pair table: sampled arcs, and one control for each.

    Arc types are drawn independently and each from its own stream, so raising
    --per-type extends a type's sample rather than reshuffling it.
    """
    tokmaps, n_words = doc_words(elig)
    head = syn["head"]
    by_dep = defaultdict(list)
    for a in arcs:
        by_dep[a["deprel"]].append(a)
    want = deps or sorted(by_dep)
    used, rows, counts, unmatched = set(), [], {}, defaultdict(list)
    pid = 0
    for d in want:
        pool = sorted(by_dep.get(d, []),
                      key=lambda a: (a["doc_id"], a["span_a"], a["span_b"]))
        rng = random.Random(f"arc-deletion arc {seed} {d}")
        take = pool if len(pool) <= per_type else rng.sample(pool, per_type)
        take = sorted(take, key=lambda a: (a["doc_id"], a["span_a"], a["span_b"]))
        counts[d] = {"available": len(pool), "sampled": len(take)}
        crng = random.Random(f"arc-deletion control {seed} {d}")
        for arc in take:
            doc_id = arc["doc_id"]
            nw = n_words[doc_id]
            u, v = elig[arc["first"]], elig[arc["second"]]
            pid += 1
            arc_id = pid
            used.add((doc_id, arc["first"], arc["second"]))
            rows.append({
                "pair_id": arc_id, "kind": "arc", "dep": d, "doc_id": doc_id,
                "span_a": arc["span_a"], "span_b": arc["span_b"],
                "span_first": arc["first"], "span_second": arc["second"],
                "dep_first": int(arc["dep_first"]),
                "distance": arc["distance"],
                "upos_first": u["pos"], "upos_second": v["pos"],
                "text_first": u["text"], "text_second": v["text"],
                "match_of": None, "match_quality": "arc",
                **covariates(u, v, nw)})
            got = find_control(arc, elig, tokmaps[doc_id], nw, head, used,
                               crng, min_distance)
            if got is None:
                unmatched[d].append(arc_id)
                continue
            cu, cv, quality, dist = got
            used.add((doc_id, cu, cv))
            x, y = elig[cu], elig[cv]
            pid += 1
            # span_a of a control plays the positional role the arc's dependent
            # played, so the two tables line up term for term
            ca, cb = (cu, cv) if arc["dep_first"] else (cv, cu)
            rows.append({
                "pair_id": pid, "kind": "control", "dep": d, "doc_id": doc_id,
                "span_a": ca, "span_b": cb,
                "span_first": cu, "span_second": cv,
                "dep_first": int(arc["dep_first"]), "distance": dist,
                "upos_first": x["pos"], "upos_second": y["pos"],
                "text_first": x["text"], "text_second": y["text"],
                "match_of": arc_id, "match_quality": quality,
                **covariates(x, y, nw)})
    # a joint variant is read back by the two span ids it deleted, so two pairs
    # over the same two spans would be indistinguishable in the store
    seen = {(r["doc_id"], r["span_first"], r["span_second"]) for r in rows}
    if len(seen) != len(rows):
        raise SystemExit("two pairs cover the same two spans")
    return rows, counts, dict(unmatched)


# ------------------------------------------------------------------- planning

def plan_document(text, words, span_ids, pairs):
    """Every deleted string this document contributes, and what each one is.

    Returns (texts, meta, extra). texts[0] is the intact document; meta is
    aligned with texts[1:] and each entry is the substitution list
    dbio.write_variants wants. Singles come first, one per distinct span
    however many pairs use it, then the joint deletions.

    No model and no torch, so the whole plan is checkable on a laptop.
    """
    spans = [(w["start"], w["end"]) for w in words]
    by_sid = {s: k for k, s in enumerate(span_ids)}
    singles = sorted({p[key] for p in pairs for key in ("span_a", "span_b")})

    texts, meta, extra = [text], [], []
    for sid in singles:
        texts.append(S.delete_span(text, spans, by_sid[sid]))
        meta.append([{"span_id": sid, "substitute": "", "source": DELETION,
                      "depth": 1}])
        extra.append({"depth": 1})
    for p in sorted(pairs, key=lambda x: x["pair_id"]):
        ks = [by_sid[p["span_a"]], by_sid[p["span_b"]]]
        texts.append(R.delete_many(text, spans, ks))
        meta.append([{"span_id": p["span_a"], "substitute": "",
                      "source": DELETION, "depth": 2},
                     {"span_id": p["span_b"], "substitute": "",
                      "source": DELETION, "depth": 2}])
        extra.append({"depth": 2, "pair_id": p["pair_id"],
                      "is_arc": 1 if p["kind"] == "arc" else 0,
                      "distance": p["distance"], "dep_first": p["dep_first"],
                      "nqwen_first": p["nqwen_first"],
                      "nqwen_second": p["nqwen_second"],
                      "copies_first": p["copies_first"],
                      "copies_second": p["copies_second"],
                      "final_copy_first": p["final_copy_first"],
                      "final_copy_second": p["final_copy_second"],
                      "in_quote_first": p["in_quote_first"],
                      "in_quote_second": p["in_quote_second"],
                      "tok_i_first": p["tok_i_first"],
                      "tok_i_second": p["tok_i_second"],
                      "n_words": p["n_words"], "position": p["position"]})
    return texts, meta, extra


def space_faults(original, edited):
    """Whitespace this edit introduced that the original did not carry.

    04's deletion rule promises no doubled space and no leading space, so the
    promise is tested by counting rather than by re-deriving the string: a fault
    is caught wherever the splice put it. Adjacent words are the case that can
    break it, because their two folded spans meet and the second deletion lands
    on the space the first one already claimed.
    """
    out = []
    if edited.startswith(" ") and not original.startswith(" "):
        out.append("leading space")
    if edited.count("  ") > original.count("  "):
        out.append("doubled space")
    return out


# (name, document, first word, second word, what deleting both must leave).
# The shapes that make an adjacent pair awkward, written out rather than hoped
# for in the corpus: two plain neighbours, a pair opening the document, a comma
# or a quote mark standing between the two words, no space at all between them,
# and two spaces between them. Offsets are found from the word text and folded
# exactly as words_from_store folds them.
SPACE_CASES = [
    ("plain neighbours", "the quick brown fox", "quick", "brown", "the fox"),
    ("pair at the start", "the quick brown fox", "the", "quick", "brown fox"),
    ("pair at the end", "the quick brown fox", "brown", "fox", "the quick"),
    ("comma between", "a big, red car", "big", "red", "a, car"),
    ("quote mark between", 'he said "hello" there', "said", "hello",
     'he "" there'),
    ("quote mark before the pair", 'he said "hello there" now', "hello",
     "there", 'he said "" now'),
    ("no space between", "it weighs 500lbs total", "500", "lbs",
     "it weighs total"),
    ("two spaces between", "a big  red car", "big", "red", "a car"),
    ("comma glued to the first", "a big, red car", "a", "big", ", red car"),
]


def check_space_rule():
    """Run SPACE_CASES through 07's delete_many. Returns the failures.

    Independent of the store, so a corpus that happens to contain none of these
    shapes cannot hide a regression in the space rule.
    """
    bad = []
    for name, text, w1, w2, want in SPACE_CASES:
        i = text.index(w1)
        j = text.index(w2, i + len(w1))
        spans = []
        for a, b in ((i, i + len(w1)), (j, j + len(w2))):
            spans.append((a - 1 if a > 0 and text[a - 1] == " " else a, b))
        got = R.delete_many(text, spans, [0, 1])
        if got != want:
            bad.append((name, want, got))
        elif space_faults(text, got):
            bad.append((name, "no whitespace fault", space_faults(text, got)))
    return bad


def check_plan(pl):
    """Round-trip every planned string against the original. Returns failures.

    A deletion is recomputed here from the offsets alone, right to left with the
    doubled-space collapse spelled out, so a bug in delete_span cannot verify
    itself. A joint over two adjacent words is checked a second way, against one
    cut across the whole region the two spans cover, which is a different
    computation from the right-to-left composition and so is a real second
    opinion on the case the token-distance floor of 1 introduces. Every string
    is also checked for the whitespace the rule promises never to leave.
    """
    text, words, sids = pl["text"], pl["words"], pl["sids"]
    spans = [(w["start"], w["end"]) for w in words]
    by_sid = {s: k for k, s in enumerate(sids)}
    bad = []

    def want_cut(a, b):
        out = text[:a] + text[b:]
        if 0 < a < len(out) and out[a - 1] == " " and out[a] == " ":
            out = out[:a] + out[a + 1:]
        elif a == 0 and out.startswith(" "):
            out = out[1:]
        return out

    def want_delete(ks):
        out = text
        for k in sorted(ks, reverse=True):
            a, b = spans[k]
            out = out[:a] + out[b:]
            if 0 < a < len(out) and out[a - 1] == " " and out[a] == " ":
                out = out[:a] + out[a + 1:]
            elif a == 0 and out.startswith(" "):
                out = out[1:]
        return out

    if pl["texts"][0] != text:
        bad.append(("baseline is not the intact document",))
    if len(pl["texts"]) != 1 + len(pl["meta"]):
        bad.append(("plan size", len(pl["texts"]), len(pl["meta"])))
    single = {}
    for t, m in zip(pl["texts"][1:], pl["meta"]):
        ks = [by_sid[s["span_id"]] for s in m]
        if len(set(ks)) != len(ks):
            bad.append(("a pair deletes one span twice", m[0]["span_id"]))
            continue
        if any(s["substitute"] != "" or s["source"] != DELETION for s in m):
            bad.append(("not a deletion row", m[0]["span_id"]))
        if any(s["depth"] != len(m) for s in m):
            bad.append(("depth", m[0]["span_id"]))
        if t != want_delete(ks):
            bad.append(("deletion splice", [s["span_id"] for s in m]))
        faults = space_faults(text, t)
        if faults:
            bad.append(("whitespace", [s["span_id"] for s in m], faults))
        if len(m) == 2:
            k1, k2 = sorted(ks)
            # the two folded spans meet, so the pair is one region of the
            # document and one cut has to give the same string as two
            if spans[k1][1] == spans[k2][0] and t != want_cut(spans[k1][0],
                                                              spans[k2][1]):
                bad.append(("adjacent joint is not one contiguous cut",
                            [s["span_id"] for s in m]))
        if len(m) == 1:
            single[m[0]["span_id"]] = t
        elif len(t) >= len(text):
            bad.append(("joint deletion did not shorten the document",
                        [s["span_id"] for s in m]))
    for m in pl["meta"]:
        if len(m) != 2:
            continue
        for s in m:
            if s["span_id"] not in single:
                bad.append(("joint deletion with no matching single",
                            s["span_id"]))
    return bad[:20]


def prepare(doc_id, syn, by_doc, pairs, qtok):
    """One document's plan, or the reason it has none."""
    text = syn["docs"][doc_id]["text"]
    if len(T.prompt_ids(qtok, text)) > MAX_PROMPT:
        return {"doc_id": doc_id, "skip": "prompt over the EasyNLA cap"}
    words, sids = words_from_store(text, by_doc[doc_id])
    have = set(sids)
    lost = [p for p in pairs
            if p["span_a"] not in have or p["span_b"] not in have]
    if lost:
        return {"doc_id": doc_id, "skip": f"{len(lost)} pairs name a span the "
                                          f"store does not hold here"}
    texts, meta, extra = plan_document(text, words, sids, pairs)
    return {"doc_id": doc_id, "skip": None, "text": text, "words": words,
            "sids": sids, "pairs": pairs, "texts": texts, "meta": meta,
            "extra": extra,
            "budget": {"baselines": 1,
                       "singles": sum(1 for m in meta if len(m) == 1),
                       "joints": sum(1 for m in meta if len(m) == 2),
                       "total": 1 + len(meta)}}


# ------------------------------------------------------------------ reporting

def quantiles(v):
    v = sorted(v)
    if not v:
        return None
    pick = lambda f: v[min(len(v) - 1, int(f * len(v)))]
    return (len(v), v[0], pick(0.25), pick(0.5), pick(0.75), pick(0.9), v[-1],
            sum(v) / len(v))


def plan_report(syn, elig, rejected, arc_why, counts, rows, unmatched, plans,
                skipped, budget, token, args):
    """The whole dry-run report as text, printed and written to results/."""
    L = []
    P_ = L.append
    arcs = [r for r in rows if r["kind"] == "arc"]
    ctrl = [r for r in rows if r["kind"] == "control"]
    deps = list(counts)

    P_("# Arc deletion plan")
    P_("")
    P_(f"scheme {args.scheme}, arcs {args.arcs}, per-type cap {args.per_type}, "
       f"seed {args.seed}, minimum token distance {args.min_distance}, position "
       f"tolerance {MAX_POSITION_DELTA:.2f} of the document's words")
    P_(f"documents in the store {len(syn['docs'])}, spans {len(syn['spans'])}, "
       f"head relations {len(syn['head'])}")
    P_("")
    P_("## Span eligibility")
    P_("")
    P_(f"{len(elig)} of {len(syn['spans'])} spans are deletable.")
    for k, v in rejected.most_common():
        P_(f"  rejected, {k}: {v}")
    P_("")
    P_("## Arcs read")
    P_("")
    for k, v in arc_why.most_common():
        P_(f"  dropped, {k}: {v}")
    if not arc_why:
        P_("  every arc in the file survived screening")
    P_("")
    P_("## Arcs per type")
    P_("")
    P_(f"  {'deprel':12s} {'available':>10s} {'sampled':>8s} {'matched':>8s} "
       f"{'unmatched':>10s} {'planned':>8s}")
    planned_pairs = {p["pair_id"] for pl in plans for p in pl["pairs"]}
    for d in deps:
        n_c = sum(1 for r in ctrl if r["dep"] == d)
        n_p = sum(1 for r in rows if r["dep"] == d
                  and r["pair_id"] in planned_pairs)
        P_(f"  {d:12s} {counts[d]['available']:10d} {counts[d]['sampled']:8d} "
           f"{n_c:8d} {len(unmatched.get(d, [])):10d} {n_p:8d}")
    P_(f"  {'total':12s} {sum(c['available'] for c in counts.values()):10d} "
       f"{len(arcs):8d} {len(ctrl):8d} "
       f"{sum(len(v) for v in unmatched.values()):10d} "
       f"{len(planned_pairs):8d}")
    P_("")
    P_("Columns: arcs of this type in the file that survived screening; how "
       "many were sampled; how many found a control; how many did not; how "
       "many pairs survived into a document plan.")
    P_("")
    P_("## Control match quality")
    P_("")
    q = Counter(r["match_quality"] for r in ctrl)
    for k in QUALITIES:
        if q[k]:
            P_(f"  {k:14s} {q[k]:6d}  {100.0 * q[k] / max(1, len(ctrl)):5.1f}%")
    P_(f"  {'unmatched':14s} {sum(len(v) for v in unmatched.values()):6d}  "
       f"{100.0 * sum(len(v) for v in unmatched.values()) / max(1, len(arcs)):5.1f}%"
       f" of sampled arcs")
    P_("")
    P_("  quality by type")
    P_(f"  {'deprel':12s} " + " ".join(f"{k:>12s}" for k in QUALITIES)
       + f" {'unmatched':>10s}")
    for d in deps:
        cq = Counter(r["match_quality"] for r in ctrl if r["dep"] == d)
        P_(f"  {d:12s} " + " ".join(f"{cq[k]:12d}" for k in QUALITIES)
           + f" {len(unmatched.get(d, [])):10d}")
    P_("")
    P_("## Covariates")
    P_("")
    P_("The columns the post-hoc analysis conditions on, arc rows against "
       "control rows. `final copy` is the share of pairs with at least one word "
       "repeating the source document's final token, `copies` the mean number "
       "of exact copies of a word elsewhere in the verbalisation, `in quote` "
       "the share of words inside a quoted stretch.")
    P_("")
    P_(f"  {'set':8s} {'pairs':>6s} {'mean nqwen':>11s} {'mean copies':>12s} "
       f"{'final copy':>11s} {'in quote':>9s} {'mean position':>14s}")
    for name, rs in (("arc", arcs), ("control", ctrl)):
        if not rs:
            continue
        nq = [x for r in rs for x in (r["nqwen_first"], r["nqwen_second"])]
        cp = [x for r in rs for x in (r["copies_first"], r["copies_second"])]
        iq = [x for r in rs for x in (r["in_quote_first"], r["in_quote_second"])]
        fc = [max(r["final_copy_first"], r["final_copy_second"]) for r in rs]
        po = [r["position"] for r in rs]
        P_(f"  {name:8s} {len(rs):6d} {sum(nq) / len(nq):11.3f} "
           f"{sum(cp) / len(cp):12.3f} {sum(fc) / len(fc):11.3f} "
           f"{sum(iq) / len(iq):9.3f} {sum(po) / len(po):14.3f}")
    P_("")
    P_("  token distance")
    P_(f"  {'set':14s} {'n':>6s} {'min':>5s} {'p25':>5s} {'median':>7s} "
       f"{'p75':>5s} {'p90':>5s} {'max':>5s} {'mean':>7s}")
    for name, v in ([(f"arc {d}", [r["distance"] for r in arcs if r["dep"] == d])
                     for d in deps]
                    + [("ARC (all)", [r["distance"] for r in arcs]),
                       ("CONTROL (all)", [r["distance"] for r in ctrl])]):
        qs = quantiles(v)
        if qs:
            P_(f"  {name:14s} {qs[0]:6d} {qs[1]:5d} {qs[2]:5d} {qs[3]:7d} "
               f"{qs[4]:5d} {qs[5]:5d} {qs[6]:5d} {qs[7]:7.2f}")
    P_("")
    P_("  adjacent pairs, token distance 1")
    P_(f"  {'set':8s} {'pairs':>6s} {'at distance 1':>14s} {'share':>7s}")
    for name, rs in (("arc", arcs), ("control", ctrl)):
        if not rs:
            continue
        n1 = sum(1 for r in rs if r["distance"] == 1)
        P_(f"  {name:8s} {len(rs):6d} {n1:14d} {100.0 * n1 / len(rs):6.1f}%")
    sep = Counter()
    for r in rows:
        if r["distance"] != 1:
            continue
        u, v = elig[r["span_first"]], elig[r["span_second"]]
        sep[repr(syn["docs"][r["doc_id"]]["text"][u["end"]:v["start"]])] += 1
    if sep:
        P_("  what stands between the two words at distance 1: "
           + ", ".join(f"{k} {n}" for k, n in sep.most_common(8)))
    P_("")
    P_("  Qwen token count per word")
    for name, rs in (("arc", arcs), ("control", ctrl)):
        c = Counter(x for r in rs
                    for x in (r["nqwen_first"], r["nqwen_second"]))
        P_(f"  {name:8s} " + "  ".join(f"{k}:{c[k]}" for k in sorted(c)))
    P_("")
    P_("  copies elsewhere in the verbalisation, per word")
    for name, rs in (("arc", arcs), ("control", ctrl)):
        c = Counter(min(4, x) for r in rs
                    for x in (r["copies_first"], r["copies_second"]))
        P_(f"  {name:8s} " + "  ".join(
            f"{'4+' if k == 4 else k}:{c[k]}" for k in sorted(c)))
    P_("")
    P_("  UPOS combination, position order, the ten commonest")
    for name, rs in (("arc", arcs), ("control", ctrl)):
        c = Counter(f"{r['upos_first']}-{r['upos_second']}" for r in rs)
        P_(f"  {name:8s} " + "  ".join(f"{k} {n}" for k, n in c.most_common(10)))
    P_("")
    P_("## Pass budget")
    P_("")
    P_(f"  {'baselines':22s} {budget['baselines']:8d}   one per document")
    P_(f"  {'single deletions':22s} {budget['singles']:8d}   one per distinct "
       f"span, however many pairs use it")
    P_(f"  {'joint deletions':22s} {budget['joints']:8d}   one per pair")
    P_(f"  {'total':22s} {budget['total']:8d}")
    P_("")
    P_("## Whitespace rule")
    P_("")
    P_("Constructed adjacent-pair deletions through 07's delete_many, so the "
       "space rule is checked on the awkward shapes whether or not the corpus "
       "holds them.")
    P_("")
    for name, text_, w1, w2, want in SPACE_CASES:
        P_(f"  {name:28s} {text_!r} - {w1!r} {w2!r} -> {want!r}")
    fails = check_space_rule()
    P_("")
    P_("  all cases leave exactly one separator, no doubled space and no "
       "leading space" if not fails
       else "  FAILURES: " + "; ".join(str(x) for x in fails))
    P_("")
    P_(f"documents planned {len(plans)}, skipped {len(skipped)}")
    P_(f"{budget['total'] / REF_PASSES_PER_SEC:.0f} s of forward passes "
       f"({budget['total'] / REF_PASSES_PER_SEC / 60:.1f} min) at "
       f"{REF_PASSES_PER_SEC:.0f}/s ({REF_NOTE})")
    if token:
        base = [t["base"] for t in token.values()]
        P_(f"baseline prompt {min(base)} to {max(base)} Qwen tokens, cap "
           f"{MAX_PROMPT}")
    P_("")
    P_("## Sign convention")
    P_("")
    P_("interaction = e(a) + e(b) - e(both), in FVE points, where e is the drop "
       "in fraction of variance explained against the same document's unedited "
       "baseline, times 100. A POSITIVE interaction means the pair costs LESS "
       "than the sum of its two single deletions.")
    if skipped:
        P_("")
        P_("## Skipped documents")
        P_("")
        for d, why in skipped[:40]:
            P_(f"  doc {d}: {why}")
        if len(skipped) > 40:
            P_(f"  ... and {len(skipped) - 40} more")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ar", default=None,
                    help="the AR reconstructor directory. Not needed with "
                         "--dry-run, which uses the Qwen tokeniser alone")
    ap.add_argument("--traces",
                    default="../01_corpus-and-spans/results/"
                            "ffw_pilot_traces.parquet")
    ap.add_argument("--db", default=str(dbio.DEFAULT_DB))
    ap.add_argument("--scheme", default=STANZA,
                    help="label scheme in `labels` and `relations`")
    ap.add_argument("--arcs", default=None,
                    help="JSONL of arcs: doc_id, span_a (the dependent), "
                         "span_b (its head), deprel. Default is every arc of "
                         "the RELATIONS in --scheme")
    ap.add_argument("--deps", default="",
                    help="comma-separated deprels to measure; default is every "
                         "deprel present in --arcs")
    ap.add_argument("--per-type", type=int, default=450)
    ap.add_argument("--min-distance", type=int, default=MIN_DISTANCE,
                    help="smallest token distance a pair may have. 1, the "
                         "default, admits adjacent words as arcs and as "
                         "controls; 2 reproduces 05's floor")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--precision", default="bf16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rawvar-from", default="traces",
                    choices=["traces", "store"])
    ap.add_argument("--out", default=str(HERE / "results"))
    ap.add_argument("--dry-run", action="store_true",
                    help="plan and round-trip every string, count the passes "
                         "and write the plan report. No reconstructor, no GPU")
    ap.add_argument("--migrate", action="store_true",
                    help="apply outstanding migrations to --db first")
    ap.add_argument("--resume-run", type=int, default=None,
                    help="append to this existing run, skipping documents it "
                         "has already measured")
    ap.add_argument("--notes", default="")
    ap.add_argument("--extra-notes", default="")
    args = ap.parse_args()
    if args.min_distance < 1:
        raise SystemExit("--min-distance must be at least 1: a pair of two "
                         "distinct words is at least one token apart")

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

    t0 = time.perf_counter()
    cache = {}
    syn = load_syntax(conn, args.scheme)
    elig, rejected = deletable_spans(syn, qtok, cache)
    print(f"scheme {args.scheme}: {len(elig)}/{len(syn['spans'])} deletable "
          f"spans ({time.perf_counter() - t0:.1f}s)", flush=True)

    arcs_in = read_arcs(args.arcs) if args.arcs else arcs_from_parse(syn)
    arcs, arc_why = screen_arcs(arcs_in, elig, syn, args.min_distance)
    print(f"arcs {len(arcs)}/{len(arcs_in)} from {args.arcs or args.scheme}", flush=True)
    deps = [d for d in args.deps.replace(",", " ").split() if d]
    rows, counts, unmatched = sample_pairs(arcs, elig, syn, args.per_type,
                                           deps, args.seed, args.min_distance)
    print(f"pairs {len(rows)} ({time.perf_counter() - t0:.1f}s)", flush=True)

    by_doc = spans_by_doc(syn)
    pairs_by_doc = defaultdict(list)
    for r in rows:
        pairs_by_doc[r["doc_id"]].append(r)

    gold_mse = M.measurable(conn)
    rawvar = rawvar_src = fp = None
    rawvar_all = {}
    model = tok = headm = E = None
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
        rawvar, rawvar_src, rawvar_all = M.raw_variance(conn, args.traces,
                                                        args.rawvar_from)
        print(f"dFVE = -dMSE / rawvar, rawvar {rawvar:.6f} from {rawvar_src}",
              flush=True)

    plans, skipped, token = [], [], {}
    for doc_id in sorted(pairs_by_doc):
        if doc_id not in gold_mse:
            skipped.append((doc_id, "no gold activation in the store"))
            continue
        pl = prepare(doc_id, syn, by_doc, pairs_by_doc[doc_id], qtok)
        if pl["skip"]:
            skipped.append((doc_id, pl["skip"]))
            continue
        plans.append(pl)
    if not plans:
        raise SystemExit("no document survived planning")
    budget = {k: sum(p["budget"][k] for p in plans)
              for k in ("baselines", "singles", "joints", "total")}

    # ---------------------------------------------------------------- dry run
    if args.dry_run:
        fails = [("constructed space case",) + tuple(x)
                 for x in check_space_rule()]
        for pl in plans:
            bad = check_plan(pl)
            fails += [(pl["doc_id"],) + tuple(x) for x in bad[:4]]
            lens = M.prompt_lengths(qtok, pl["texts"])
            token[pl["doc_id"]] = {"base": lens[0], "max": max(lens),
                                   "min": min(lens)}
        report = plan_report(syn, elig, rejected, arc_why, counts, rows,
                             unmatched, plans, skipped, budget, token, args)
        report += ("\nevery planned string round-tripped\n" if not fails
                   else "\nFAILURES:\n"
                        + "\n".join(f"  {x}" for x in fails[:20]) + "\n")
        (out / "plan.md").write_text(report)
        print()
        print(report)
        print(f"plan: {out / 'plan.md'}")
        return 1 if fails else 0

    # ------------------------------------------------------------- the GPU run
    import torch
    from extract_traces import MSE_SCALE, normalize_activation

    planned = {p["pair_id"] for pl in plans for p in pl["pairs"]}
    table = [dict(r, planned=(r["pair_id"] in planned)) for r in rows]
    cols = ["pair_id", "kind", "dep", "doc_id", "span_a", "span_b",
            "span_first", "span_second", "dep_first", "distance",
            "upos_first", "upos_second", "text_first", "text_second",
            "nqwen_first", "nqwen_second", "copies_first", "copies_second",
            "final_copy_first", "final_copy_second", "in_quote_first",
            "in_quote_second", "tok_i_first", "tok_i_second", "n_words",
            "position", "match_of", "match_quality", "planned"]

    skip = set()
    if args.resume_run is not None:
        row = conn.execute("SELECT script FROM runs WHERE run_id = ?",
                           (int(args.resume_run),)).fetchone()
        if row is None or "arc_deletion" not in (row["script"] or ""):
            raise SystemExit(f"run {args.resume_run} is not an arc_deletion run")
        run_id = int(args.resume_run)
        skip = S.already_done(conn, run_id)
        print(f"resuming run {run_id}; {len(skip)} documents already measured",
              flush=True)
    else:
        gpu = (torch.cuda.get_device_name(0) if torch.cuda.is_available()
               else "no CUDA device")
        notes = args.notes or (
            "Arc deletion by dependency type. Two lexical words are deleted "
            "singly and together, so the interaction e(a) + e(b) - e(both) is "
            "formed per pair in FVE points; positive means the pair costs less "
            "than the sum of its singles. Pairs are dependency arcs "
            f"at least {args.min_distance} token(s) apart, so an adjacent "
            "head and dependent is included at the default floor of 1, "
            "each matched to a control pair in the same "
            "document with no arc, sibling or grandparent link, the same token "
            "distance, the same Qwen token count word for word, the same "
            "in-quote status word for word and a mean position within 20% of "
            "the document's words, relaxed in that order. Part of speech is a "
            "covariate here, not a matching key. Deletion is 04's rule applied "
            "right to left by 07's delete_many, so a span contributes one "
            "single however many pairs use it. Measurements pair_id, is_arc, "
            "distance, dep_first, nqwen_first/second, copies_first/second, "
            "final_copy_first/second, in_quote_first/second, "
            "tok_i_first/second, n_words and position ride on the joint "
            "variant beside mse, fve, seq_len and dtok; the pair table with "
            "its UPOS, its text and its match quality is in this row's "
            "config.")
        run_id = DB.new_run(
            conn, script="09_arcs-by-type/arc_deletion.py",
            assets=ASSETS, notes=notes + args.extra_notes + f" GPU: {gpu}.",
            config={"args": vars(args), "fingerprint": fp,
                    "mse_rawvar": rawvar, "mse_rawvar_from": rawvar_src,
                    "mse_rawvar_all": dict(rawvar_all),
                    "deletion_source": DELETION, "scheme": args.scheme,
                    "pair_scheme": PAIR_SCHEME, "gpu": gpu,
                    "min_distance": args.min_distance,
                    "max_position_delta": MAX_POSITION_DELTA,
                    "qualities": QUALITIES, "arcs_file": args.arcs,
                    "arc_screening": dict(arc_why),
                    "per_type_counts": counts,
                    "unmatched_arcs": unmatched, "budget": budget,
                    "sign": "interaction = e(a) + e(b) - e(both), FVE points; "
                            "positive means the pair costs less than the sum "
                            "of its singles",
                    "pair_columns": cols,
                    "pairs": [[p[c] for c in cols] for p in table]})
        conn.commit()
        # relations already means "an edge between two spans under a scheme", so
        # the pair set is recoverable from the store without a migration. What
        # it cannot carry is which arc a control was matched to, or the match
        # quality; those are in this run's config
        with DB.transaction(conn):
            DB.add_relations(conn, [
                (PAIR_SCHEME, int(p["span_a"]), int(p["span_b"]),
                 f"arc:{p['dep']}" if p["kind"] == "arc" else "control")
                for p in table if p["planned"]])
    print(f"run_id {run_id}", flush=True)
    print(f"planned passes {budget['total']} over {len(plans)} documents",
          flush=True)

    t0 = time.perf_counter()
    n_docs = n_var = 0
    for pl in plans:
        doc_id = pl["doc_id"]
        if doc_id in skip:
            print(f"  doc {doc_id}: already measured, skipped", flush=True)
            continue
        bad = check_plan(pl)
        if bad:
            raise SystemExit(f"doc {doc_id}: plan check failed: {bad[:4]}")
        gold = torch.tensor(M.gold_vector(conn, doc_id), dtype=torch.float32,
                            device=args.device)
        gold_n = normalize_activation(gold.unsqueeze(0), MSE_SCALE)[0]
        mse, lens = M.scored(model, tok, headm, E, gold_n, pl["texts"],
                             args.batch, args.device)
        records = [([], {"mse": mse[0], "fve": 1 - mse[0] / rawvar,
                         "seq_len": lens[0], "dtok": 0,
                         "traces_mse": gold_mse[doc_id]})]
        for j, (subs, x) in enumerate(zip(pl["meta"], pl["extra"]), start=1):
            records.append((subs, {"mse": mse[j], "fve": 1 - mse[j] / rawvar,
                                   "seq_len": lens[j],
                                   "dtok": lens[j] - lens[0], **x}))
        dbio.write_variants(conn, doc_id, run_id, records)
        n_docs += 1
        n_var += len(records)
        print(f"  doc {doc_id}: {len(pl['pairs'])} pairs, {len(records)} "
              f"variants, base FVE {1 - mse[0] / rawvar:.4f}, "
              f"{time.perf_counter() - t0:.0f}s elapsed", flush=True)
    wall = time.perf_counter() - t0
    conn.execute("UPDATE runs SET config = json_set(config, '$.wall_time_s', ?) "
                 "WHERE run_id = ?", (round(wall), run_id))
    conn.commit()
    print(f"\nrun {run_id}: {n_docs} docs, {n_var} variants ({wall:.0f}s)")
    for k, v in DB.counts(conn).items():
        print(f"  {k:16s} {v}")
    conn.execute("PRAGMA optimize")
    conn.close()
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
