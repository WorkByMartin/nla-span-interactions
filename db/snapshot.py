#!/usr/bin/env python3
"""Point-in-time copies of the store, with a sidecar saying what they are.

    python snapshot.py "what is about to change, or what was just loaded"
    python snapshot.py --list

Copies --db, which defaults to ffw_span-ablation_database.sqlite, into
db/snapshots/ as NNNN_<slug>.sqlite with an NNNN_<slug>.md beside it recording
the schema version, row counts, runs and outstanding migrations. That directory
is gitignored.

Take one immediately BEFORE applying any migration and AFTER any loader run.
The copy goes through sqlite3's backup API rather than a file copy, so it is
consistent under WAL without stopping anything that has the database open.
"""
from __future__ import annotations

import argparse
import re
import sqlite3
import time
from pathlib import Path

import db

HERE = Path(__file__).resolve().parent
SNAPS = HERE / "snapshots"


def slugify(text, cap=50):
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return (s[:cap].rstrip("-") or "snapshot")


def next_seq():
    SNAPS.mkdir(exist_ok=True)
    used = [int(m.group(1)) for f in SNAPS.glob("*.sqlite")
            if (m := re.match(r"(\d{4})_", f.name))]
    return max(used, default=0) + 1


def take(src_path, description):
    src_path = Path(src_path)
    if not src_path.exists():
        raise SystemExit(f"no database at {src_path}")
    seq, slug = next_seq(), slugify(description)
    stem = f"{seq:04d}_{slug}"
    out = SNAPS / f"{stem}.sqlite"

    src = sqlite3.connect(str(src_path))
    dst = sqlite3.connect(str(out))
    with dst:
        src.backup(dst)
    dst.close()
    src.close()

    conn = db.connect(out)
    version = db.schema_version(conn)
    counts = db.counts(conn)
    runs = [(r["run_id"], r["script"]) for r in conn.execute(
        "SELECT run_id, script FROM runs ORDER BY run_id")]
    outstanding = [f.name for _, f in db.pending(conn)]
    conn.close()

    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    size = out.stat().st_size
    lines = [
        f"# {seq:04d} {now} schema v{version} | {description}",
        "",
        f"- file: `{out.name}` ({size / 1e6:.1f} MB)",
        f"- taken: {now}",
        f"- schema_version: {version}",
        f"- outstanding migrations: "
        f"{', '.join(outstanding) if outstanding else 'none'}",
        f"- git sha: {db.git_sha() or 'unknown'}",
        "",
        "## Description",
        "",
        description,
        "",
        "## Rows",
        "",
    ]
    lines += [f"- {k}: {v}" for k, v in counts.items()]
    lines += ["", "## Runs", ""]
    lines += ([f"- {rid}: {script}" for rid, script in runs] or ["- none"])
    (SNAPS / f"{stem}.md").write_text("\n".join(lines) + "\n")
    return out, SNAPS / f"{stem}.md", version, size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("description", nargs="?",
                    help="what is about to change, or what was just loaded")
    ap.add_argument("--db", default=str(db.DEFAULT_PATH))
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        SNAPS.mkdir(exist_ok=True)
        found = sorted(SNAPS.glob("*.md"))
        for f in found:
            print(f.read_text().splitlines()[0].lstrip("# "))
        if not found:
            print("no snapshots")
        return

    if not args.description:
        raise SystemExit("a description is required: snapshot.py \"what changed\"")
    out, side, version, size = take(args.db, args.description)
    print(f"{out.name}  schema v{version}  {size / 1e6:.1f} MB")
    print(f"{side.name}")


if __name__ == "__main__":
    main()
