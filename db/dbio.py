#!/usr/bin/env python3
"""What the ablation scripts need from the store, and nothing else.

Opens the project database, registers documents and their word spans, reads and
writes masked-LM candidate lists, and writes a document's variants and their
measurements in one transaction. An experiment script in a numbered directory
imports it with

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "db"))
    import dbio

Spans here are BARE words. The substitution primitive folds the preceding space
into a unit because both tokenisers carry the space inside the token, but that is
its own internal convention: the store holds the word, and so does every
substitute and candidate string.
"""
from __future__ import annotations

from pathlib import Path

import db

DEFAULT_DB = Path(__file__).resolve().parent / "ffw_span-ablation_database.sqlite"

# ModernBERT masked fill under the text-space regime: space parity plus the
# clean-word filter, over a ModernBERT token RANGE collapsed to one [MASK].
# Distinct from '.../idsplice', the superseded Qwen-token-count regime.
FILLER = "modernbert-large_filler_model/textsub"


NEEDS = 2   # candidates


def open_db(path=None, allow_migrate=False, needs=None):
    """Connect, and refuse to run if the schema is behind the migrations.

    allow_migrate is off by default: a schema change is never a side effect of
    launching a run that will take GPU time and then write into the wrong shape.
    `needs` is the minimum schema version this caller's reads are correct under;
    a run that records several unsubstituted variants per document needs 4,
    where v_baseline stops returning one row per repeat measurement.
    """
    conn = db.connect(path or DEFAULT_DB)
    db.require_schema(conn, needs=needs or NEEDS, apply=allow_migrate,
                      verbose=True)
    return conn


def spacy_scheme(nlp):
    return f"spacy-{nlp.meta['lang']}_{nlp.meta['name']}-{nlp.meta['version']}"


def bare_span(word):
    """(start, end) of the word itself, undoing word_spans' folded-in space."""
    return word["end"] - len(word["text"]), word["end"]


def ensure_spans(conn, doc_id, text, words, scheme, source=None):
    """Register the document and its words. Returns span_ids aligned with words."""
    with db.transaction(conn):
        db.upsert_doc(conn, doc_id, text, source)
        keys = [(doc_id,) + bare_span(w) for w in words]
        got = db.get_or_create_spans(conn, keys)
        ids = [got[k] for k in keys]
        db.set_labels(conn, [(s, scheme, k, w[k])
                             for s, w in zip(ids, words)
                             for k in ("pos", "tag")])
    return ids


def load_conditionals(conn, span_ids, scheme, topk):
    """Cached candidate lists, aligned with span_ids. None where absent."""
    return [db.get_candidates(conn, s, scheme, limit=topk) for s in span_ids]


def save_conditionals(conn, span_ids, conds, scheme):
    """Persist sampler conditionals. `conds` entries may be None."""
    rows = []
    for sid, c in zip(span_ids, conds):
        if c is None:
            continue
        rows += [(sid, scheme, i, db.bare(s), float(p))
                 for i, (s, p) in enumerate(zip(c["strs"], c["probs"]))]
    if rows:
        with db.transaction(conn):
            db.add_candidates(conn, rows)
    return len(rows)


def write_variants(conn, doc_id, run_id, records):
    """One transaction for a whole document.

    records: (subs, metrics) pairs. `subs` is a list of dicts with span_id and
    substitute, optionally source, depth, draw_idx, prob; empty means the
    baseline. `metrics` maps metric name to value, NaN and None dropped.
    Returns the variant_ids, aligned with records.
    """
    with db.transaction(conn):
        vids = db.new_variants(conn, [(doc_id, run_id)] * len(records))
        subs, meas = [], []
        for vid, (ss, mm) in zip(vids, records):
            for s in ss:
                subs.append((vid, s["span_id"], db.bare(s["substitute"]),
                             s.get("source"), s.get("depth"),
                             s.get("draw_idx"), s.get("prob")))
            for k, v in mm.items():
                if v is None or v != v:
                    continue
                meas.append((vid, run_id, k, float(v)))
        db.add_substitutions(conn, subs)
        db.record_many(conn, meas)
    return vids
