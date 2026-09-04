#!/usr/bin/env python3
"""The shuffle: the same bag of words in a destroyed order.

    python shuffle.py

Two documents, each with its intact verbalisation and the four variants in which
every lexical word was permuted among the document's own slots. Prints the
report and writes the same text to shuffle.md. No GPU.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as K  # noqa: E402

DOCS = [1925, 3914]
RUN = 5


def shuffle_variants(conn, run_id, doc_id):
    """(variant_id, fve, n_sub) for each shuffle variant of one document."""
    q = ("SELECT vm.variant_id, vm.fve, COUNT(*) AS n_sub "
         "FROM v_variant_metrics vm "
         "JOIN substitutions s ON s.variant_id = vm.variant_id "
         "WHERE vm.run_id = ? AND vm.doc_id = ? AND s.source = ? "
         "GROUP BY vm.variant_id, vm.fve ORDER BY vm.variant_id")
    return [(int(r["variant_id"]), float(r["fve"]), int(r["n_sub"]))
            for r in conn.execute(q, (int(run_id), int(doc_id), K.SHUFFLE))]


def shuffled_text(conn, variant_id, text):
    """The variant's text, rebuilt by splicing its recorded substitutes in.

    Stored spans are the bare word, and the substitute is stored bare too, so
    replacing the span's characters leaves whatever space preceded it in place.
    That is what the ablation's own space-parity rule does, so the string below
    is the string that was measured.
    """
    q = ("SELECT t.char_start AS a, t.char_end AS b, s.substitute AS w "
         "FROM substitutions s JOIN v_span_text t ON t.span_id = s.span_id "
         "WHERE s.variant_id = ? ORDER BY t.char_start DESC")
    out = text
    for r in conn.execute(q, (int(variant_id),)):
        out = out[:r["a"]] + r["w"] + out[r["b"]:]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=str(K.DB_DEFAULT))
    args = ap.parse_args()

    conn = K.connect(args.db)

    panels = []
    for doc in DOCS:
        text = K.doc_text(conn, doc)
        base = K.baseline(conn, RUN, doc)
        var = shuffle_variants(conn, RUN, doc)
        fves = np.array([f for _, f, _ in var])
        mean = float(fves.mean())
        k = int(np.argmin(np.abs(fves - mean)))
        vid, fve, n_sub = var[k]
        panels.append(dict(doc=doc, text=text, base=base, var=var,
                           mean=mean, sd=float(fves.std(ddof=1)),
                           pick=k, vid=vid, fve=fve, n_sub=n_sub,
                           shuf=shuffled_text(conn, vid, text)))

    # ------------------------------------------------------------------ text
    K.tee_to(K.HERE / "shuffle.md")
    print("# The shuffle")
    print()
    print("The whole word order removed, the bag of words kept. Every lexical "
          "word of the document is")
    print("permuted among the document's own slots, four permutations per "
          "document, and the reconstructor")
    print("is run on the result. Punctuation is not a slot and does not move.")
    print()
    print("FVE is the fraction of variance the reconstruction explains, so "
          "1.0 is exact and 0.0 is no")
    print("better than predicting the mean activation. It is unbounded below.")
    print()
    for p in panels:
        print(f"## Document {p['doc']}")
        print()
        print(f"Intact FVE {p['base']:.4f}. {len(p['var'])} shuffles of "
              f"{p['n_sub']} lexical words, mean FVE {p['mean']:.4f}, "
              f"sd {p['sd']:.4f},")
        print(f"so the shuffle costs {-100 * (p['mean'] - p['base']):.3f} FVE "
              f"points on average.")
        print()
        print(K.md_table(
            ["variant", "FVE", "FVE points lost", "shown below"],
            [[v, f"{f:.4f}", f"{-100 * (f - p['base']):+.3f}",
              "yes" if i == p["pick"] else ""]
             for i, (v, f, _) in enumerate(p["var"])],
            ["r", "r", "r", "l"]))
        print()
        print(f"The variant shown is the one whose FVE is closest to the mean "
              f"of the four.")
        print()
        print(f"Intact verbalisation, FVE {p['base']:.4f}:")
        print()
        print(K.wrap_md(" ".join(p["text"].split()), 92, indent="    "))
        print()
        print(f"Shuffled variant {p['vid']}, FVE {p['fve']:.4f}:")
        print()
        print(K.wrap_md(" ".join(p["shuf"].split()), 92, indent="    "))
        print()

    K.untee()
    conn.close()


if __name__ == "__main__":
    main()
