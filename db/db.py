#!/usr/bin/env python3
"""SQLite store for span-ablation experiments.

One database per project. Any decomposer or measurement script appends to it;
nothing owns a private file format. Schema changes are migrations under
migrations/, applied in filename order and recorded in schema_version.

Standard library only.
"""
from __future__ import annotations

import contextlib
import json
import re
import sqlite3
import subprocess
import time
from pathlib import Path

MIGRATIONS = Path(__file__).resolve().parent / "migrations"
DEFAULT_PATH = Path(__file__).resolve().parent / "ffw_span-ablation_database.sqlite"


# --------------------------------------------------------------------- setup

def connect(path=DEFAULT_PATH):
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA synchronous = NORMAL")
    return conn


def pending(conn, upto=None):
    """(number, path) of every migration not yet applied, in order."""
    conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER)")
    row = conn.execute("SELECT MAX(version) AS v FROM schema_version").fetchone()
    have = row["v"] or 0
    out = []
    for f in sorted(MIGRATIONS.glob("*.sql")):
        m = re.match(r"(\d+)", f.name)
        if not m:
            continue
        n = int(m.group(1))
        if n > have and (upto is None or n <= upto):
            out.append((n, f))
    return out


def migrate(conn, verbose=False, apply=False, upto=None):
    """Apply migrations newer than schema_version, in filename order.

    apply defaults to False: a schema change is never a side effect of running a
    loader. With apply False this reports what is outstanding and changes
    nothing.
    """
    todo = pending(conn, upto)
    if not apply:
        return [f.name for _, f in todo]
    done = []
    for n, f in todo:
        with conn:
            conn.executescript(f.read_text())
            conn.execute("INSERT INTO schema_version (version) VALUES (?)", (n,))
        done.append(f.name)
        if verbose:
            print(f"applied {f.name}")
    return done


def schema_version(conn):
    """0 for a database that has never had a migration applied."""
    conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER)")
    row = conn.execute("SELECT MAX(version) AS v FROM schema_version").fetchone()
    return row["v"] or 0


def require_schema(conn, needs, apply=False, verbose=False):
    """Refuse to run if the database is older than this caller needs.

    Gated on what the caller needs, not on what happens to be sitting in
    migrations/: an unapplied migration for some later feature must not stop
    today's loader from running. A loader stops and says what is outstanding
    rather than changing the schema as a side effect of being run.
    """
    todo = pending(conn)
    have = schema_version(conn)
    if have >= needs:
        if todo and verbose:
            print("note: outstanding migrations, NOT applied: "
                  + ", ".join(f.name for _, f in todo))
        return []
    if not apply:
        raise SystemExit(
            f"this needs schema version {needs}, the database is at {have}. "
            "Outstanding: " + ", ".join(f.name for _, f in todo) +
            "\nRe-run with --migrate to apply them, or point --db at a "
            "throwaway copy.")
    return migrate(conn, verbose=verbose, apply=True, upto=needs)


@contextlib.contextmanager
def transaction(conn):
    """One commit per load, not one per row."""
    try:
        yield conn
    except Exception:
        conn.rollback()
        raise
    conn.commit()


def git_sha(cwd=None):
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"],
                              cwd=str(cwd or Path(__file__).resolve().parent),
                              capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return None


# ----------------------------------------------------------------- documents

def upsert_doc(conn, doc_id, text, source=None):
    conn.execute(
        "INSERT INTO docs (doc_id, text, source) VALUES (?, ?, ?) "
        "ON CONFLICT(doc_id) DO UPDATE SET text = excluded.text, "
        "source = COALESCE(excluded.source, docs.source)",
        (doc_id, text, source))
    return doc_id


def upsert_docs(conn, rows):
    """rows: (doc_id, text, source)."""
    conn.executemany(
        "INSERT INTO docs (doc_id, text, source) VALUES (?, ?, ?) "
        "ON CONFLICT(doc_id) DO UPDATE SET text = excluded.text, "
        "source = COALESCE(excluded.source, docs.source)", rows)


# --------------------------------------------------------------------- spans

def get_or_create_span(conn, doc_id, char_start, char_end):
    conn.execute(
        "INSERT OR IGNORE INTO spans (doc_id, char_start, char_end) "
        "VALUES (?, ?, ?)", (doc_id, char_start, char_end))
    return conn.execute(
        "SELECT span_id FROM spans WHERE doc_id = ? AND char_start = ? "
        "AND char_end = ?", (doc_id, char_start, char_end)).fetchone()["span_id"]


def get_or_create_spans(conn, rows):
    """rows: (doc_id, char_start, char_end). Returns {(d, s, e): span_id}."""
    rows = list(rows)
    conn.executemany(
        "INSERT OR IGNORE INTO spans (doc_id, char_start, char_end) "
        "VALUES (?, ?, ?)", rows)
    docs = sorted({r[0] for r in rows})
    out = {}
    for i in range(0, len(docs), 500):
        chunk = docs[i:i + 500]
        q = ("SELECT span_id, doc_id, char_start, char_end FROM spans "
             "WHERE doc_id IN (%s)" % ",".join("?" * len(chunk)))
        for r in conn.execute(q, chunk):
            out[(r["doc_id"], r["char_start"], r["char_end"])] = r["span_id"]
    return {k: out[k] for k in rows if k in out}


def span_text(conn, span_id):
    r = conn.execute("SELECT text FROM v_span_text WHERE span_id = ?",
                     (span_id,)).fetchone()
    return None if r is None else r["text"]


# -------------------------------------------------------------------- labels

def set_label(conn, span_id, scheme, key, value):
    conn.execute(
        "INSERT INTO labels (span_id, scheme, key, value) VALUES (?, ?, ?, ?) "
        "ON CONFLICT(span_id, scheme, key) DO UPDATE SET value = excluded.value",
        (span_id, scheme, key, None if value is None else str(value)))


def set_labels(conn, rows):
    """rows: (span_id, scheme, key, value)."""
    conn.executemany(
        "INSERT INTO labels (span_id, scheme, key, value) VALUES (?, ?, ?, ?) "
        "ON CONFLICT(span_id, scheme, key) DO UPDATE SET value = excluded.value",
        [(a, b, c, None if d is None else str(d)) for a, b, c, d in rows])


def add_relation(conn, scheme, span_a, span_b, kind):
    conn.execute(
        "INSERT OR IGNORE INTO relations (scheme, span_a, span_b, kind) "
        "VALUES (?, ?, ?, ?)", (scheme, span_a, span_b, kind))


def add_relations(conn, rows):
    """rows: (scheme, span_a, span_b, kind)."""
    conn.executemany(
        "INSERT OR IGNORE INTO relations (scheme, span_a, span_b, kind) "
        "VALUES (?, ?, ?, ?)", rows)


# ---------------------------------------------------------------------- runs

def new_run(conn, script, assets=None, notes=None, git=None, started_at=None,
            config=None):
    """config: anything json-serialisable. argv, environment fingerprint, and
    every constant the numbers depend on, so a run can be re-read years later."""
    if config is not None and not isinstance(config, str):
        config = json.dumps(config, default=str)
    cur = conn.execute(
        "INSERT INTO runs (script, git_sha, assets, notes, started_at, config) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (script, git if git is not None else git_sha(),
         json.dumps(list(assets or [])), notes,
         started_at or time.strftime("%Y-%m-%dT%H:%M:%S"), config))
    return cur.lastrowid


# ---------------------------------------------------------------- candidates

def bare(s):
    """Substitute and candidate strings are stored without a leading space.

    Space parity is the substitution primitive's internal business: it folds the
    preceding space into the ModernBERT mask range and back in when splicing.
    What the store holds is the word.
    """
    return s[1:] if s.startswith(" ") else s


def set_candidates(conn, span_id, scheme, strs, probs):
    """Replace this span's candidate list under `scheme`. Rank 0 is best."""
    conn.execute("DELETE FROM candidates WHERE span_id = ? AND scheme = ?",
                 (span_id, scheme))
    conn.executemany(
        "INSERT INTO candidates (span_id, scheme, rank, candidate, prob) "
        "VALUES (?, ?, ?, ?, ?)",
        [(span_id, scheme, i, bare(s), float(p))
         for i, (s, p) in enumerate(zip(strs, probs))])


def add_candidates(conn, rows):
    """Bulk. rows: (span_id, scheme, rank, candidate, prob), candidate bare."""
    conn.executemany(
        "INSERT OR REPLACE INTO candidates "
        "(span_id, scheme, rank, candidate, prob) VALUES (?, ?, ?, ?, ?)", rows)


def get_candidates(conn, span_id, scheme, limit=None):
    """(strings, probs) in rank order, or None when the span has no list."""
    q = ("SELECT candidate, prob FROM candidates WHERE span_id = ? "
         "AND scheme = ? ORDER BY rank")
    args = [span_id, scheme]
    if limit is not None:
        q += " LIMIT ?"
        args.append(limit)
    rows = conn.execute(q, args).fetchall()
    if not rows:
        return None
    return [r["candidate"] for r in rows], [r["prob"] for r in rows]


# ------------------------------------------------------------------ variants

SUB_COLS = "(variant_id, span_id, substitute, source, depth, draw_idx, prob)"


def new_variant(conn, doc_id, run_id, substitutions=()):
    """One edited document. substitutions is a list of dicts with span_id and
    substitute, optionally source/depth/draw_idx/prob. Empty = the baseline."""
    cur = conn.execute(
        "INSERT INTO variants (doc_id, created_run_id) VALUES (?, ?)",
        (doc_id, run_id))
    vid = cur.lastrowid
    rows = [(vid, s["span_id"], s["substitute"], s.get("source"),
             s.get("depth"), s.get("draw_idx"), s.get("prob"))
            for s in substitutions]
    if rows:
        conn.executemany(
            f"INSERT INTO substitutions {SUB_COLS} VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows)
    return vid


def new_variants(conn, rows):
    """Bulk. rows: (doc_id, run_id). Returns the allocated variant_ids."""
    rows = list(rows)
    r = conn.execute("SELECT COALESCE(MAX(variant_id), 0) AS m FROM variants")
    base = r.fetchone()["m"]
    ids = list(range(base + 1, base + 1 + len(rows)))
    conn.executemany(
        "INSERT INTO variants (variant_id, doc_id, created_run_id) "
        "VALUES (?, ?, ?)",
        [(i, d, run) for i, (d, run) in zip(ids, rows)])
    return ids


def add_substitutions(conn, rows):
    """rows: (variant_id, span_id, substitute, source, depth, draw_idx, prob)."""
    conn.executemany(
        f"INSERT INTO substitutions {SUB_COLS} VALUES (?, ?, ?, ?, ?, ?, ?)",
        rows)


def baseline_variant(conn, doc_id, run_id):
    """The variant of doc_id with no substitutions that run_id measured."""
    r = conn.execute(
        "SELECT v.variant_id FROM variants v "
        "WHERE v.doc_id = ? "
        "  AND NOT EXISTS (SELECT 1 FROM substitutions s "
        "                  WHERE s.variant_id = v.variant_id) "
        "  AND EXISTS (SELECT 1 FROM measurements m "
        "              WHERE m.variant_id = v.variant_id AND m.run_id = ?) "
        "ORDER BY v.variant_id LIMIT 1", (doc_id, run_id)).fetchone()
    if r:
        return r["variant_id"]
    r = conn.execute(
        "SELECT v.variant_id FROM variants v "
        "WHERE v.doc_id = ? AND v.created_run_id = ? "
        "  AND NOT EXISTS (SELECT 1 FROM substitutions s "
        "                  WHERE s.variant_id = v.variant_id) "
        "ORDER BY v.variant_id LIMIT 1", (doc_id, run_id)).fetchone()
    return None if r is None else r["variant_id"]


# ------------------------------------------------------------- measurements

def record(conn, variant_id, run_id, metric, value):
    conn.execute(
        "INSERT INTO measurements (variant_id, run_id, metric, value) "
        "VALUES (?, ?, ?, ?) "
        "ON CONFLICT(variant_id, run_id, metric) DO UPDATE "
        "SET value = excluded.value", (variant_id, run_id, metric, value))


def record_many(conn, rows):
    """rows: (variant_id, run_id, metric, value)."""
    conn.executemany(
        "INSERT INTO measurements (variant_id, run_id, metric, value) "
        "VALUES (?, ?, ?, ?) "
        "ON CONFLICT(variant_id, run_id, metric) DO UPDATE "
        "SET value = excluded.value", rows)


# --------------------------------------------------------------------- misc

TABLES = ["docs", "doc_tokens", "spans", "labels", "relations", "candidates",
          "variants", "substitutions", "measurements", "runs", "schema_version"]


def counts(conn):
    have = {r["name"] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    return {t: conn.execute(f"SELECT COUNT(*) AS n FROM {t}").fetchone()["n"]
            for t in TABLES if t in have}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="inspect or migrate the store")
    ap.add_argument("path", nargs="?", default=str(DEFAULT_PATH))
    ap.add_argument("--migrate", action="store_true",
                    help="apply outstanding migrations. Off by default, so "
                         "no run changes the schema as a side effect")
    ap.add_argument("--upto", type=int, default=None)
    a = ap.parse_args()
    c = connect(a.path)
    todo = migrate(c, verbose=True, apply=a.migrate, upto=a.upto)
    print(("applied: " if a.migrate else "OUTSTANDING, not applied: ")
          + (", ".join(todo) if todo else "nothing"))
    print("schema_version", schema_version(c))
    for k, v in counts(c).items():
        print(f"  {k:16s} {v}")
