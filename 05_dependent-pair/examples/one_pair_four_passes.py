#!/usr/bin/env python3
"""One dependency arc pair and its matched control, the four passes each.

    python one_pair_four_passes.py

The arc pair whose eight-draw mean interaction is nearest the arc mean, among
arcs with an exact-quality control (same document, same ordered part-of-speech
pair, same token distance). Prints the sentence, then for one draw the baseline,
A alone, B alone and both, with the interaction arithmetic. Writes the same
text to one_pair_four_passes.md. No GPU.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent / "04_ablation-strategy" / "examples"))
import pair_analysis as PA  # noqa: E402
import _common as K         # noqa: E402
import figstyle             # noqa: E402

RUN = 7


def span_info(conn, span_id):
    r = conn.execute("SELECT doc_id, char_start, char_end, text FROM v_span_text "
                     "WHERE span_id = ?", (int(span_id),)).fetchone()
    return int(r["doc_id"]), int(r["char_start"]), int(r["char_end"]), r["text"]


def mark(text, spans, a, b):
    """Sentence containing spans a and b, each marked [A word] and [B word]."""
    (_, a0, a1, _), (_, b0, b1, _) = spans[a], spans[b]
    s0, _ = K.sentence_bounds(text, min(a0, b0), min(a1, b1))
    _, s1 = K.sentence_bounds(text, max(a0, b0), max(a1, b1))
    edits = sorted([(a0, a1, "A"), (b0, b1, "B")], reverse=True)
    out = text
    for x0, x1, tag in edits:
        out = out[:x0] + f"[{tag}:{out[x0:x1]}]" + out[x1:]
    shift = 2 * len("[A:]")
    return " ".join(out[s0:s1 + shift].split())


def block(name, p, draw_rows, base):
    r = draw_rows[p["pair_id"]]
    print(f"{name}, token distance {p['distance']}, {p['combo']}"
          + (f", {p['dep']} ({figstyle.dep_name(p['dep'])})"
             if p["kind"] == "arc" else ", no dependency"))
    print(f"    baseline FVE {base:.3f}")
    print(f"    A swapped        {base - r['e_a'] / 100:.3f}   e(a)    = {r['e_a']:+.3f} points")
    print(f"    B swapped        {base - r['e_b'] / 100:.3f}   e(b)    = {r['e_b']:+.3f} points")
    print(f"    both swapped     {base - r['e_both'] / 100:.3f}   e(both) = {r['e_both']:+.3f} points")
    print(f"    interaction = e(a) + e(b) - e(both) = {r['inter']:+.3f} points on this draw,")
    print(f"    {p['inter']:+.3f} +- {p['inter_sd'] / np.sqrt(p['n_draws']):.3f} over the {p['n_draws']} draws")
    print()


def main():
    conn = PA.connect(str(K.DB_DEFAULT))
    _, by_span = PA.load_pairs(conn, RUN)
    sing = PA.singles(conn, RUN)
    joint = PA.joints(conn, RUN)
    rows, _ = PA.assemble(by_span, sing, joint)
    P = PA.by_pair(rows)
    arcs = [p for p in P.values() if p["kind"] == "arc"]
    ctrl_of = {p["match_of"]: p for p in P.values()
               if p["kind"] == "control" and p["quality"] == "exact"}
    arc_mean = float(np.mean([p["inter"] for p in arcs]))
    cand = [p for p in arcs if p["pair_id"] in ctrl_of]
    arc = min(cand, key=lambda p: abs(p["inter"] - arc_mean))
    ctrl = ctrl_of[arc["pair_id"]]
    tab = by_span  # span ids per pair
    spans_of = {p["pair_id"]: (p["span_a"], p["span_b"]) for p in tab.values()}

    # the draw nearest the arc pair's own mean
    draws = [r for r in rows if r["pair_id"] == arc["pair_id"]]
    k = min(draws, key=lambda r: abs(r["inter"] - arc["inter"]))["draw"]
    draw_rows = {r["pair_id"]: r for r in rows
                 if r["draw"] == k and r["pair_id"] in (arc["pair_id"], ctrl["pair_id"])}

    doc = arc["doc_id"]
    text = K.doc_text(conn, doc)
    base = K.baseline(conn, RUN, doc)
    spans = {s: span_info(conn, s) for s in spans_of[arc["pair_id"]] + spans_of[ctrl["pair_id"]]}

    K.tee_to(HERE / "one_pair_four_passes.md")
    print("# One pair, four passes")
    print()
    print(f"Document {doc}, baseline FVE {base:.3f}, draw {k} of 8. Effects are FVE points lost,")
    print("so positive means the reconstruction got worse. Interaction = e(a) + e(b) - e(both):")
    print("positive means the two words carry overlapping information, negative means they are")
    print("worth more together than apart.")
    print()
    print(f"The arc pair is the one whose mean interaction is nearest the arc mean of "
          f"{arc_mean:+.3f} points,")
    print(f"among the {len(cand)} arcs with an exact control. Its control is in the same document,")
    print("same ordered part-of-speech pair, same token distance, no dependency between them.")
    print()
    a, b = spans_of[arc["pair_id"]]
    print("Arc pair:")
    print(K.wrap_md(mark(text, spans, a, b), 92, indent="    "))
    print()
    block("arc", arc, draw_rows, base)
    a, b = spans_of[ctrl["pair_id"]]
    print("Control pair:")
    print(K.wrap_md(mark(text, spans, a, b), 92, indent="    "))
    print()
    block("control", ctrl, draw_rows, base)
    K.untee()


if __name__ == "__main__":
    main()
