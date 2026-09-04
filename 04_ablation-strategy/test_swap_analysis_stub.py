#!/usr/bin/env python3
"""Drive swap_analysis.py over a synthetic store, so the report is known to run.

    python test_swap_analysis_stub.py

Builds a throwaway database with the same shape as the real one, 5 documents by
40 spans, a masked-LM run at six depths by eight draws and a swap run at sixteen
draws plus deletion and shuffle arms, with a planted span effect so the variance
decomposition has something to recover. Then runs the analysis on it and checks
that the recovered components are close to what was planted.
"""
import random
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "db"))
import db as DB  # noqa: E402

SWAP = "corpus-swap/pos+len"
MLM = "modernbert-large_filler_model/textsub"
POSES = ["NOUN", "VERB", "ADJ", "ADV", "PROPN", "DET", "ADP", "PRON", "AUX"]
OPEN = {"NOUN", "VERB", "ADJ", "ADV", "PROPN"}
SCHEME = "spacy-en_core_web_sm-3.8.0"
RAWVAR = 0.45467177831665084

SD_BETWEEN, SD_WITHIN = 0.9, 0.5   # FVE points, planted


def main():
    rng = random.Random(0)
    tmp = Path(tempfile.mkdtemp(prefix="swapstub_"))
    path = tmp / "ablation.sqlite"
    conn = DB.connect(path)
    DB.migrate(conn, apply=True)

    docs = [4635, 2159, 2957, 376, 1934]
    words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]
    span_ids, poses, base = {}, {}, {}
    with DB.transaction(conn):
        for d in docs:
            text = " ".join(rng.choice(words) for _ in range(60))
            DB.upsert_doc(conn, d, text, "test")
            keys, off = [], 0
            for w in text.split(" ")[:40]:
                i = text.index(w, off)
                keys.append((d, i, i + len(w)))
                off = i + len(w)
            got = DB.get_or_create_spans(conn, keys)
            span_ids[d] = [got[k] for k in keys]
            for s in span_ids[d]:
                p = rng.choice(POSES)
                poses[s] = p
                DB.set_label(conn, s, SCHEME, "pos", p)

    # masked-LM run: six depths, eight draws, effect grows with depth
    mlm_run = DB.new_run(conn, script="03_parts-of-speech/pos_fve.py", notes="mlm")
    swap_run = DB.new_run(conn, script="04_ablation-strategy/swap_ablation.py",
                          notes="swap", config='{"pool_occurrences": 12854}')
    assert swap_run == 2

    truth = {}
    for d in docs:
        base[d] = 0.8 + 0.01 * rng.random()
        for run in (mlm_run, swap_run):
            with DB.transaction(conn):
                v = DB.new_variant(conn, d, run)
                DB.record(conn, v, run, "fve", base[d])
                DB.record(conn, v, run, "mse", (1 - base[d]) * RAWVAR)
                DB.record(conn, v, run, "seq_len", 200)
        for s in span_ids[d]:
            mu = rng.gauss(0, SD_BETWEEN) * (1.4 if poses[s] in OPEN else 0.6)
            truth[s] = mu
            with DB.transaction(conn):
                for depth in (1, 3, 5, 10, 25, 50):
                    for j in range(8):
                        e = 0.55 * mu + rng.gauss(0, SD_WITHIN)
                        v = DB.new_variant(conn, d, mlm_run, substitutions=[
                            {"span_id": s, "substitute": rng.choice(words),
                             "source": MLM, "depth": depth, "draw_idx": j,
                             "prob": 0.1}])
                        DB.record(conn, v, mlm_run, "fve", base[d] - e / 100)
                        DB.record(conn, v, mlm_run, "mse", 0.1)
                        DB.record(conn, v, mlm_run, "seq_len", 200)
                for j in range(16):
                    e = mu + rng.gauss(0, SD_WITHIN)
                    v = DB.new_variant(conn, d, swap_run, substitutions=[
                        {"span_id": s, "substitute": rng.choice(words),
                         "source": SWAP, "depth": 1, "draw_idx": j,
                         "prob": 0.01}])
                    DB.record(conn, v, swap_run, "fve", base[d] - e / 100)
                    DB.record(conn, v, swap_run, "mse", 0.1)
                    DB.record(conn, v, swap_run, "seq_len", 200)
                e = 1.6 * mu + rng.gauss(0, SD_WITHIN)
                v = DB.new_variant(conn, d, swap_run, substitutions=[
                    {"span_id": s, "substitute": "", "source": "deletion",
                     "depth": 1, "draw_idx": 0, "prob": None}])
                DB.record(conn, v, swap_run, "fve", base[d] - e / 100)
                DB.record(conn, v, swap_run, "mse", 0.1)
                DB.record(conn, v, swap_run, "seq_len", 199)
        with DB.transaction(conn):
            for j in range(4):
                v = DB.new_variant(conn, d, swap_run, substitutions=[
                    {"span_id": s, "substitute": rng.choice(words),
                     "source": "shuffle", "depth": 40, "draw_idx": j,
                     "prob": None} for s in span_ids[d]])
                DB.record(conn, v, swap_run, "fve", base[d] - 0.4)
                DB.record(conn, v, swap_run, "mse", 0.3)
                DB.record(conn, v, swap_run, "seq_len", 200)
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.close()

    out = tmp / "results"
    r = subprocess.run([sys.executable, str(HERE / "swap_analysis.py"),
                        "--db", str(path), "--out", str(out),
                        "--mlm-run", str(mlm_run), "--run", str(swap_run)],
                       capture_output=True, text=True)
    print(r.stdout[-6000:])
    if r.returncode:
        print(r.stderr[-4000:])
        print("FAILED: analysis exited", r.returncode)
        return 1

    fails = []
    md = (out / "statistics.md").read_text()
    for want in ["VARIANCE DECOMPOSITION", "WHERE TO SPEND A FIXED PASS BUDGET",
                 "RANK AGREEMENT", "OPEN AGAINST CLOSED", "WORD-ORDER SHUFFLE",
                 "LEAKAGE", "PRECISION OF A PER-SPAN",
                 "PER-CLASS SPREAD OF THE PER-SPAN SWAP MEAN",
                 "PER-CLASS SIGNED MEAN EFFECT",
                 "DELETION AGAINST SWAP, per class",
                 "PER-CLASS VARIANCE DECOMPOSITION AND PASS BUDGET"]:
        if want not in md:
            fails.append(f"section missing: {want}")
    for f in ["swap_se_vs_draws.png", "swap_vs_mlm_scatter.png",
              "budget_draws_vs_spans.png"]:
        if not (out / f).exists():
            fails.append(f"figure missing: {f}")
    if "—" in md or "–" in md:
        fails.append("a dash slipped into the report text")

    # the planted components should come back
    line = [l for l in md.splitlines() if l.strip().startswith("swap, all spans")]
    print("  recovered:", line[0].strip() if line else "NOT FOUND")
    if line:
        parts = line[0].split()
        between = float(parts[parts.index("between") + 1])
        within = float(parts[parts.index("within") + 1])
        # planted between-span sd varies by class, so check the order of magnitude
        if not 0.3 < between < 2.5:
            fails.append(f"between component {between} is not near what was planted")
        if not abs(within - SD_WITHIN ** 2) < 0.05:
            fails.append(f"within component {within} is not near {SD_WITHIN ** 2}")

    print("\nFAILED:" if fails else "\nanalysis runs end to end on synthetic data")
    for f in fails:
        print(" ", f)
    print(f"  scratch database left at {tmp}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
