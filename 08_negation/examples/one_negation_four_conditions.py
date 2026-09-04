#!/usr/bin/env python3
"""One negator instance under every condition it was run in, with the FVE of each.

    python one_negation_four_conditions.py

The "not" or "n't" instance whose flip effect is nearest the mean flip effect
over all negator instances. Prints the sentence intact, with the negation
flipped, with the governed word deleted, and with the governed word
corpus-swapped (the draw nearest the mean of four), each with its FVE and the
points lost. Writes the same text to one_negation_four_conditions.md. No GPU.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "04_ablation-strategy" / "examples"))
import _common as K  # noqa: E402

DB = HERE.parent.parent / "db" / "ffw_span-ablation_database.sqlite"
RUN = 10
SOURCES = {"negation/flip": "negation flipped", "negation/del-gov": "governed word deleted",
           "corpus-swap/pos+len": "governed word corpus-swapped"}


def edited(con, text, variant_id):
    rows = con.execute("SELECT t.char_start a, t.char_end b, s.substitute w FROM substitutions s "
                       "JOIN v_span_text t ON t.span_id = s.span_id WHERE s.variant_id = ? "
                       "ORDER BY t.char_start DESC", (int(variant_id),)).fetchall()
    out = text
    lo, hi = len(text), 0
    for a, b, w in [tuple(r) for r in rows]:
        out = out[:a] + f"[{w}]" + out[b:]
        lo, hi = min(lo, a), max(hi, b)
    return out, lo, hi


def main():
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    m = pd.read_sql("SELECT v.variant_id, v.doc_id, m.metric, m.value FROM variants v JOIN measurements m "
                    "ON m.variant_id = v.variant_id WHERE v.created_run_id = ?", con, params=(RUN,))
    w = m.pivot(index=["variant_id", "doc_id"], columns="metric", values="value").reset_index()
    base = w[w["condition"].isna()].set_index("doc_id")["fve"]
    d = w[w["condition"].notna()].copy()
    d["base"] = d["doc_id"].map(base)
    d["lost"] = -100.0 * (d["fve"] - d["base"])
    src = {int(r[0]): r[1] for r in con.execute(
        "SELECT s.variant_id, s.source FROM substitutions s JOIN variants v "
        "ON v.variant_id = s.variant_id WHERE v.created_run_id = ?", (RUN,))}
    d["source"] = d["variant_id"].map(src)
    neg = d[(d["instance"] > 0) & (d["ntype"].isin([0, 1]))]
    flips = neg[neg["source"] == "negation/flip"]
    mean_flip = float(d[(d["instance"] > 0) & (d["source"] == "negation/flip")]["lost"].mean())
    have = neg.groupby("instance")["source"].nunique()
    ok = flips[flips["instance"].isin(have[have == 3].index)]
    pick = ok.iloc[int(np.argmin(np.abs(ok["lost"] - mean_flip)))]
    inst, doc = int(pick["instance"]), int(pick["doc_id"])
    rows = neg[neg["instance"] == inst]
    text = K.doc_text(con, doc)
    b = float(base[doc])

    K.tee_to(HERE / "one_negation_four_conditions.md")
    print("# One negation, four conditions")
    print()
    print(f"Document {doc}, baseline FVE {b:.3f}. Effect is FVE points lost, so positive means the")
    print(f"reconstruction got worse. This is the \"not\" or \"n't\" instance whose flip effect is nearest")
    print(f"the mean flip effect over every negator instance ({mean_flip:+.2f} points).")
    print()
    flip_v = int(rows[rows["source"] == "negation/flip"]["variant_id"].iloc[0])
    _, lo, hi = edited(con, text, flip_v)
    s0, s1 = K.sentence_bounds(text, lo, hi)
    print("Intact:")
    print(K.wrap_md(text[s0:s1], 92, indent="    "))
    print(f"    FVE {b:.3f}")
    print()
    for source, name in SOURCES.items():
        r = rows[rows["source"] == source]
        if source.startswith("corpus"):
            r = r.iloc[[int(np.argmin(np.abs(r["lost"] - r["lost"].mean())))]]
            name += f", the draw nearest the mean of {len(rows[rows['source'] == source])} ({rows[rows['source'] == source]['lost'].mean():+.2f} points)"
        v = int(r["variant_id"].iloc[0])
        out, lo2, hi2 = edited(con, text, v)
        e0, e1 = K.sentence_bounds(out, lo2, hi2)
        print(f"{name[0].upper() + name[1:]}:")
        print(K.wrap_md(out[e0:e1], 92, indent="    "))
        print(f"    FVE {float(r['fve'].iloc[0]):.3f}, {float(r['lost'].iloc[0]):+.2f} points")
        print()
    print("Square brackets mark the edited span; empty brackets mark a deletion.")
    K.untee()


if __name__ == "__main__":
    main()
