#!/usr/bin/env python3
"""Fold the traces parquet into the store: one row per document.

    python load_traces.py [--db PATH] [--traces PATH] [--migrate]

Reads 01_corpus-and-spans/results/ffw_pilot_traces.parquet and writes, per
document, the layer-42 activation, the verbalisation, the reconstruction, the
scored MSE, and the per-token KL between the RL verbaliser and the SFT
reference.

docs.text stays the EXPLANATION, because that is what spans index into. The
verbalisation, which is the whole tagged generation, is its own column. Where a
document is already in the store, its text is asserted to match the parquet byte
for byte before anything is written.

Idempotent. Documents are updated in place, token rows are replaced, and the
extraction run is looked up by the parquet's sha256 rather than added again.
Needs schema version 3.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import db

HERE = Path(__file__).resolve().parent
TRACES = HERE.parent / "01_corpus-and-spans" / "results" / "ffw_pilot_traces.parquet"
ASSET = "ffw_main_traces"
SCRIPT = "01_corpus-and-spans/extract_traces.py"

VEC = "<f4"          # float32 little-endian, 5120 per vector
SCALARS = ["global_id", "domain", "n_tokens", "token_position",
           "cjk_fraction", "mse", "verbalisation"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def existing_run(conn, digest):
    """The extraction run already registered for this parquet, or None.

    Matched on the recorded sha256, over runs whose script is this one under any
    path, so a load made before the script moved is still recognised.
    """
    name = SCRIPT.rsplit("/", 1)[-1]
    for r in conn.execute(
            "SELECT run_id, config FROM runs WHERE script = ? OR script LIKE ?",
            (SCRIPT, "%/" + name)):
        try:
            if json.loads(r["config"] or "{}").get("sha256") == digest:
                return r["run_id"]
        except (ValueError, TypeError):
            continue
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(db.DEFAULT_PATH))
    ap.add_argument("--traces", default=str(TRACES))
    ap.add_argument("--migrate", action="store_true",
                    help="apply outstanding migrations first")
    args = ap.parse_args()

    import numpy as np
    import pyarrow.parquet as pq

    conn = db.connect(args.db)
    db.require_schema(conn, needs=3, apply=args.migrate,
                      verbose=True)

    table = pq.read_table(args.traces)
    meta = {k.decode(): v.decode() for k, v in (table.schema.metadata or {}).items()}
    rows = table.to_pylist()
    digest = sha256(args.traces)

    # ---- the text already in the store must be the text in the parquet
    have = {r["doc_id"]: r["text"] for r in conn.execute("SELECT doc_id, text FROM docs")}
    drift = [r["doc_uid"] for r in rows
             if r["doc_uid"] in have and have[r["doc_uid"]] != r["explanation"]]
    if drift:
        raise SystemExit(
            f"explanation text differs from the store for {len(drift)} "
            f"documents, first {drift[:5]}. Refusing to overwrite; the spans "
            f"already recorded index into the text that is there.")

    cols = ", ".join(f"{c} = ?" for c in SCALARS)
    sql = (f"INSERT INTO docs (doc_id, text, source) VALUES (?, ?, ?) "
           f"ON CONFLICT(doc_id) DO NOTHING")
    n_tok = 0
    with db.transaction(conn):
        conn.executemany(sql, [(r["doc_uid"], r["explanation"], ASSET)
                               for r in rows])
        conn.executemany(
            f"UPDATE docs SET source = ?, {cols}, activation = ?, "
            f"reconstruction = ? WHERE doc_id = ?",
            [tuple([ASSET] + [r[c] for c in SCALARS]
                   + [np.asarray(r["activation"], dtype=VEC).tobytes(),
                      np.asarray(r["reconstruction"], dtype=VEC).tobytes(),
                      r["doc_uid"]])
             for r in rows])
        tok = [(r["doc_uid"], i, int(t), float(k))
               for r in rows
               for i, (t, k) in enumerate(zip(r["gen_token_ids"],
                                              r["kl_per_token"]))]
        conn.executemany(
            "INSERT OR REPLACE INTO doc_tokens (doc_id, position, token_id, kl) "
            "VALUES (?, ?, ?, ?)", tok)
        n_tok = len(tok)

    run_id = existing_run(conn, digest)
    if run_id is None:
        run_id = db.new_run(
            conn, script=SCRIPT, assets=[ASSET],
            notes="Extraction of the 100-document trace set: layer-42 "
                  "activation, verbalisation, reconstruction, scored MSE, and "
                  "per-token KL between the RL verbaliser and the SFT "
                  "reference. Vectors are stored float32; the parquet is "
                  "float64 and remains the precise source.",
            config={"sha256": digest, "parquet": str(args.traces),
                    "n_rows": len(rows), "metadata": meta,
                    "vector_dtype": VEC, "vector_len": len(rows[0]["activation"]),
                    "loaded_by": "db/load_traces.py"})
        conn.commit()
        print(f"registered extraction run {run_id}")
    else:
        print(f"extraction run {run_id} already registered for this parquet")

    print(f"{len(rows)} documents, {n_tok} generated-token rows")
    for k, v in db.counts(conn).items():
        print(f"  {k:16s} {v}")
    r = conn.execute(
        "SELECT COUNT(*) n, SUM(activation IS NOT NULL) a, "
        "SUM(reconstruction IS NOT NULL) b, SUM(length(activation)) sz "
        "FROM docs").fetchone()
    print(f"activation blobs {r['a']}/{r['n']}, reconstruction blobs "
          f"{r['b']}/{r['n']}, {(r['sz'] or 0) / 1e6:.1f} MB of activations")
    conn.execute("PRAGMA optimize")
    conn.close()


if __name__ == "__main__":
    main()
