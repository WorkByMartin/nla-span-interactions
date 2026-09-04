#!/usr/bin/env python3
"""Copy one run out of a database staged on a GPU host into the local store under a new run id.

    python load_pod_run.py PULLED_DB --from-run 6 --to-run 7 --verify
    python load_pod_run.py PULLED_DB --from-run 6 --to-run 7 --scheme pair-ablation/arc+control --write

What moves: the `runs` row, its `variants`, their `substitutions` and
`measurements`, and the `relations` rows the run wrote under `--scheme`.
What does not move: `docs`, `spans`, `labels`, `doc_tokens`, `candidates`.

The run id and every variant id are reassigned. Span ids are kept, and that is
checked rather than assumed: every span the run substituted must exist locally
at the same (doc_id, char_start, char_end), or nothing is written. `--verify`
reports what would move and stops.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOCAL = HERE / "ffw_span-ablation_database.sqlite"
PAIR_SCHEME = "pair-ablation/arc+control"


def ro(path):
    c = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    return c


def main():
    global PAIR_SCHEME
    ap = argparse.ArgumentParser()
    ap.add_argument("pod")
    ap.add_argument("--local", default=str(LOCAL))
    ap.add_argument("--from-run", type=int, default=6)
    ap.add_argument("--to-run", type=int, default=None,
                    help="the local run id to write. Default is one past the "
                         "highest local run id")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--scheme", default=PAIR_SCHEME,
                    help="relations scheme the run wrote its pairs under")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    PAIR_SCHEME = args.scheme

    pod = ro(args.pod)
    src = int(args.from_run)

    row = pod.execute("SELECT * FROM runs WHERE run_id = ?", (src,)).fetchone()
    if row is None:
        raise SystemExit(f"the pulled file has no run {src}")

    loc = ro(args.local) if not args.write else sqlite3.connect(args.local)
    if args.write:
        loc.row_factory = sqlite3.Row
        loc.execute("PRAGMA foreign_keys = ON")

    dst = args.to_run
    if dst is None:
        dst = int(loc.execute("SELECT COALESCE(MAX(run_id), 0) FROM "
                              "runs").fetchone()[0]) + 1
    clash = loc.execute("SELECT 1 FROM runs WHERE run_id = ?", (dst,)).fetchone()
    if clash:
        raise SystemExit(f"local run {dst} already exists; pick another --to-run")

    # ---------------------------------------------------------------- what moves
    variants = [dict(r) for r in pod.execute(
        "SELECT variant_id, doc_id, created_run_id FROM variants "
        "WHERE created_run_id = ? ORDER BY variant_id", (src,))]
    vids = [v["variant_id"] for v in variants]
    if not vids:
        raise SystemExit(f"run {src} has no variants in the pulled file")

    def chunks(xs, n=800):
        for i in range(0, len(xs), n):
            yield xs[i:i + n]

    subs, meas = [], []
    for c in chunks(vids):
        q = "(%s)" % ",".join("?" * len(c))
        subs += [dict(r) for r in pod.execute(
            f"SELECT * FROM substitutions WHERE variant_id IN {q}", c)]
        meas += [dict(r) for r in pod.execute(
            f"SELECT * FROM measurements WHERE variant_id IN {q}", c)]
    off_run = {m["run_id"] for m in meas}
    if off_run - {src}:
        raise SystemExit(f"measurements on these variants name runs {off_run}")

    rels = [dict(r) for r in pod.execute(
        "SELECT * FROM relations WHERE scheme = ?", (PAIR_SCHEME,))]
    local_rels = {tuple(r) for r in loc.execute(
        "SELECT scheme, span_a, span_b, kind FROM relations WHERE scheme = ?",
        (PAIR_SCHEME,))}

    # variant ids are reallocated, because the local file has grown
    base = int(loc.execute("SELECT COALESCE(MAX(variant_id), 0) FROM "
                           "variants").fetchone()[0])
    vmap = {v: base + i + 1 for i, v in enumerate(vids)}

    # ------------------------------------------------------------- span checks
    span_ids = sorted({s["span_id"] for s in subs})
    pod_spans, loc_spans = {}, {}
    for c in chunks(span_ids):
        q = "(%s)" % ",".join("?" * len(c))
        for r in pod.execute(f"SELECT span_id, doc_id, char_start, char_end "
                             f"FROM spans WHERE span_id IN {q}", c):
            pod_spans[r["span_id"]] = (r["doc_id"], r["char_start"],
                                       r["char_end"])
        for r in loc.execute(f"SELECT span_id, doc_id, char_start, char_end "
                             f"FROM spans WHERE span_id IN {q}", c):
            loc_spans[r["span_id"]] = (r["doc_id"], r["char_start"],
                                       r["char_end"])
    missing = [s for s in span_ids if s not in loc_spans]
    moved = [s for s in span_ids
             if s in loc_spans and loc_spans[s] != pod_spans.get(s)]

    doc_ids = sorted({v["doc_id"] for v in variants})
    doc_missing = []
    for c in chunks(doc_ids):
        q = "(%s)" % ",".join("?" * len(c))
        have = {r[0] for r in loc.execute(
            f"SELECT doc_id FROM docs WHERE doc_id IN {q}", c)}
        doc_missing += [d for d in c if d not in have]

    print("WHAT MOVES")
    print(f"  pod run          {src}  ({row['script']})")
    print(f"  local run to be  {dst}")
    print(f"  variants         {len(variants)}  ids {min(vids)}..{max(vids)} "
          f"-> {min(vmap.values())}..{max(vmap.values())}")
    print(f"  substitutions    {len(subs)}")
    print(f"  measurements     {len(meas)}")
    print(f"  relations        {len(rels)} under {PAIR_SCHEME}, "
          f"{len(local_rels)} already local")
    print(f"  documents        {len(doc_ids)}")
    print(f"  distinct spans   {len(span_ids)}")
    print("\nSPAN AND DOCUMENT IDENTITY")
    print(f"  spans absent locally          {len(missing)}")
    print(f"  spans at a different offset   {len(moved)}")
    print(f"  documents absent locally      {len(doc_missing)}")
    if missing[:3]:
        print(f"    first missing {missing[:3]}")
    if moved[:3]:
        print(f"    first moved {[(s, pod_spans[s], loc_spans[s]) for s in moved[:3]]}")
    if missing or moved or doc_missing:
        raise SystemExit("REFUSING: the run's spans do not line up with the "
                         "local file")
    print("  every span the run touched exists locally at the same offsets")

    n_local_spans = loc.execute("SELECT COUNT(*) FROM spans").fetchone()[0]
    n_local_docs = loc.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
    print(f"  local file now holds {n_local_docs} documents and "
          f"{n_local_spans} spans; the run uses {len(doc_ids)} and "
          f"{len(span_ids)} of them")

    if not args.write:
        print("\nnothing written (--verify only). Re-run with --write")
        return 0

    # -------------------------------------------------------------- the write
    cfg = row["config"]
    try:
        c = json.loads(cfg) if cfg else {}
        c["loaded_as_run"] = dst
        c["run_id_on_pod"] = src
        c["variant_id_offset"] = base
        cfg = json.dumps(c, default=str)
    except Exception:
        pass
    note = (row["notes"] or "")
    note += (f" [Loaded from the pod's run {src} as local run {dst}; the local "
             f"file had gained 1,000 further verbalisations under its own run "
             f"{src} while this ran, so both the run id and every variant id "
             f"were reassigned. Span and document ids are unchanged and were "
             f"checked against the local file before the copy.]")

    with loc:
        loc.execute(
            "INSERT INTO runs (run_id, script, git_sha, assets, notes, "
            "started_at, config) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (dst, row["script"], row["git_sha"], row["assets"], note,
             row["started_at"], cfg))
        loc.executemany(
            "INSERT INTO variants (variant_id, doc_id, created_run_id) "
            "VALUES (?, ?, ?)",
            [(vmap[v["variant_id"]], v["doc_id"], dst) for v in variants])
        loc.executemany(
            "INSERT INTO substitutions (variant_id, span_id, substitute, "
            "source, depth, draw_idx, prob) VALUES (?, ?, ?, ?, ?, ?, ?)",
            [(vmap[s["variant_id"]], s["span_id"], s["substitute"],
              s["source"], s["depth"], s["draw_idx"], s["prob"])
             for s in subs])
        loc.executemany(
            "INSERT INTO measurements (variant_id, run_id, metric, value) "
            "VALUES (?, ?, ?, ?)",
            [(vmap[m["variant_id"]], dst, m["metric"], m["value"])
             for m in meas])
        loc.executemany(
            "INSERT OR IGNORE INTO relations (scheme, span_a, span_b, kind) "
            "VALUES (?, ?, ?, ?)",
            [(r["scheme"], r["span_a"], r["span_b"], r["kind"]) for r in rels])

    print(f"\nWROTE run {dst}")
    q = lambda s, *a: loc.execute(s, a).fetchone()[0]
    print(f"  variants      {q('SELECT COUNT(*) FROM variants WHERE created_run_id=?', dst)}")
    print(f"  measurements  {q('SELECT COUNT(*) FROM measurements WHERE run_id=?', dst)}")
    print(f"  substitutions {q('SELECT COUNT(*) FROM substitutions s JOIN variants v USING(variant_id) WHERE v.created_run_id=?', dst)}")
    print(f"  baselines     {q('SELECT COUNT(*) FROM v_baseline WHERE run_id=?', dst)}")
    print(f"  singles       {q('SELECT COUNT(*) FROM v_single WHERE run_id=?', dst)}")
    print(f"  joint edits   {q('SELECT COUNT(*) FROM v_pair WHERE run_id=?', dst)}")
    print(f"  relations     {q('SELECT COUNT(*) FROM relations WHERE scheme=?', PAIR_SCHEME)}")
    loc.execute("PRAGMA optimize")
    loc.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
