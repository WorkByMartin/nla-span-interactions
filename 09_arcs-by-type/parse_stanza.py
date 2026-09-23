#!/usr/bin/env python3
"""Parse every verbalisation with Stanza and store the parse as its own scheme.

    python parse_stanza.py --db ../db/ffw_span-ablation_database.sqlite \
        --stanza-dir ~/stanza_resources

Every document in `docs` is parsed whole, with Stanza doing its own sentence
splitting, by the local `stanza_en-combined_model` (nothing is downloaded). The
parse goes into the store under the scheme `stanza-en_combined_charlm-1.14.0`:

  spans      one row per word, (doc_id, char_start, char_end) into `docs.text`.
             An existing row at the same offsets is reused, a missing one is
             inserted. Multi-word tokens (doesn't -> does + n't) get one span
             per word, located inside the token's own text.
  labels     `pos` (UPOS), `tag` (xpos), `dep` (the arc label on the dependent)
             and `tok_i` (the document-level 0-based word index).
  relations  (scheme, span_a = dependent, span_b = head, kind = 'head'), one
             per non-root word.

The load is idempotent: this scheme's labels and relations are deleted and
rewritten. Spans, documents and every other scheme are left alone.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "db"))

import db  # noqa: E402
import dbio  # noqa: E402

SCHEME = "stanza-en_combined_charlm-1.14.0"
PROCESSORS = "tokenize,mwt,pos,lemma,depparse"


def load_docs(conn):
    return [(int(r["doc_id"]), r["text"])
            for r in conn.execute("SELECT doc_id, text FROM docs ORDER BY doc_id")]


def mwt_offsets(token_text, words, start, end):
    """(start, end) per word of one token, distinct even for a multi-word token.

    Each word's text is located inside the token text in order. When a word
    cannot be found, the token is split evenly so the words still get distinct
    non-empty spans, and the miss is counted.
    """
    if len(words) == 1:
        return [(start, end)], 0
    spans, cursor, misses = [], 0, 0
    low = token_text.lower()
    for w in words:
        wt = w.text or ""
        j = low.find(wt.lower(), cursor) if wt else -1
        if j < 0:
            spans.append(None)
            misses += 1
        else:
            spans.append((start + j, start + j + len(wt)))
            cursor = j + len(wt)
    if misses:
        n = len(words)
        width = max(1, (end - start) // n)
        spans = [(min(start + i * width, end - 1),
                  min(start + (i + 1) * width, end) if i < n - 1 else end)
                 for i in range(n)]
    return spans, misses


def parse(docs, stanza_dir, batch_size, report_every):
    """({doc_id: [word]}, failed, mwt misses)."""
    import stanza

    nlp = stanza.Pipeline("en", dir=str(stanza_dir), processors=PROCESSORS,
                          download_method=None, verbose=False)
    out, failed, misses = {}, [], 0
    t0, done = time.time(), 0
    for lo in range(0, len(docs), batch_size):
        chunk = docs[lo:lo + batch_size]
        parsed = nlp.bulk_process([stanza.Document([], text=t) for _, t in chunk])
        for (doc_id, _), doc in zip(chunk, parsed):
            words, base = [], 0
            for sent in doc.sentences:
                spans = {}
                for token in sent.tokens:
                    got, miss = mwt_offsets(token.text, token.words,
                                            token.start_char, token.end_char)
                    misses += miss
                    for w, sp in zip(token.words, got):
                        spans[w.id] = sp
                for word in sent.words:
                    a, b = spans[word.id]
                    words.append({"i": base + word.id - 1, "start": a, "end": b,
                                  "text": word.text, "upos": word.upos,
                                  "xpos": word.xpos, "deprel": word.deprel,
                                  "head": -1 if word.head == 0 else base + word.head - 1})
                base += len(sent.words)
            if any(w["i"] != k for k, w in enumerate(words)):
                failed.append(doc_id)
                continue
            out[doc_id] = words
        done += len(chunk)
        if report_every and done % report_every < batch_size:
            print(f"  {done}/{len(docs)} documents, {time.time() - t0:.0f}s", flush=True)
    return out, failed, misses


def store(conn, parsed):
    stats = {"docs": 0, "words": 0, "spans_created": 0, "roots": 0}
    before = conn.execute("SELECT COUNT(*) AS n FROM spans").fetchone()["n"]
    with db.transaction(conn):
        conn.execute("DELETE FROM labels WHERE scheme = ?", (SCHEME,))
        conn.execute("DELETE FROM relations WHERE scheme = ?", (SCHEME,))
        for doc_id, words in parsed.items():
            keys = [(doc_id, w["start"], w["end"]) for w in words]
            got = db.get_or_create_spans(conn, keys)
            sid = [got[k] for k in keys]
            labels = []
            for s, w in zip(sid, words):
                labels += [(s, SCHEME, "pos", w["upos"]), (s, SCHEME, "tag", w["xpos"]),
                           (s, SCHEME, "dep", w["deprel"]), (s, SCHEME, "tok_i", w["i"])]
            db.set_labels(conn, labels)
            rels = [(SCHEME, s, sid[w["head"]], "head")
                    for s, w in zip(sid, words) if w["head"] >= 0]
            db.add_relations(conn, rels)
            stats["docs"] += 1
            stats["words"] += len(words)
            stats["roots"] += len(words) - len(rels)
    stats["spans_created"] = conn.execute(
        "SELECT COUNT(*) AS n FROM spans").fetchone()["n"] - before
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", default=str(dbio.DEFAULT_DB))
    ap.add_argument("--stanza-dir", default=str(Path.home() / "stanza_resources"))
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--report-every", type=int, default=100)
    a = ap.parse_args()

    conn = db.connect(a.db)
    db.require_schema(conn, needs=1)
    docs = load_docs(conn)
    print(f"{len(docs)} documents from {a.db}", flush=True)
    parsed, failed, misses = parse(docs, a.stanza_dir, a.batch_size, a.report_every)
    text_of = dict(docs)
    bad = sum(text_of[d][w["start"]:w["end"]] != w["text"]
              for d, ws in parsed.items() for w in ws)
    stats = store(conn, parsed)
    conn.close()
    print(f"scheme {SCHEME}: {stats['docs']} documents, {stats['words']} words, "
          f"{stats['spans_created']} new spans, {stats['roots']} roots; "
          f"{len(failed)} documents failed, {misses} multi-word offset misses, "
          f"{bad} words whose offsets do not slice back to their text")


if __name__ == "__main__":
    main()
