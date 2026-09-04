#!/usr/bin/env python3
"""One sentence, four edits: each ablation applied to one sentence once.

    python one_sentence_four_edits.py

Document 1934, the word "offices". The corpus-swap draw nearest the swap mean,
the masked-LM draw nearest its mean, the deletion, and the shuffle nearest the
shuffle mean, each with its FVE. Prints the text and writes it to
one_sentence_four_edits.md. No GPU.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as K  # noqa: E402
from one_word_all_draws import DOC, WANT, SWAP_RUN, MLM_RUN, pick  # noqa: E402
from shuffle import shuffle_variants, shuffled_text  # noqa: E402

HEAD = 18


def nearest_mean(rows):
    e = np.array([r["effect"] for r in rows])
    return rows[int(np.argmin(np.abs(e - e.mean())))], float(e.mean())


def splice(sent, hl, sub):
    i, j = hl
    return sent[:i] + sub + sent[j:]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=str(K.DB_DEFAULT))
    conn = K.connect(ap.parse_args().db)
    text = K.doc_text(conn, DOC)
    base = K.baseline(conn, SWAP_RUN, DOC)
    span_id, _, (p, word, a, b), _ = pick(conn, WANT)
    s0, s1 = K.sentence_bounds(text, a, b)
    sent, hl = text[s0:s1], (a - s0, b - s0)

    swap, swap_mean = nearest_mean(K.singles(conn, SWAP_RUN, K.SWAP, span_id=span_id))
    mlm_all = K.singles(conn, MLM_RUN, K.MLM, span_id=span_id)
    mlm_nonid = [r for r in mlm_all if r["substitute"].strip().lower() != word.lower()]
    mlm, mlm_mean = nearest_mean(mlm_nonid)
    dele = K.singles(conn, SWAP_RUN, K.DELETION, span_id=span_id)[0]
    var = shuffle_variants(conn, SWAP_RUN, DOC)
    fves = np.array([f for _, f, _ in var])
    vid, sfve, n_sub = var[int(np.argmin(np.abs(fves - fves.mean())))]
    shuf = shuffled_text(conn, vid, text)

    def fve_line(fve):
        return f"FVE {base:.3f} to {fve:.3f}, {-100 * (fve - base):+.2f} points"

    K.tee_to(K.HERE / "one_sentence_four_edits.md")
    print(f"# One sentence, four edits")
    print()
    print(f"Document {DOC}, baseline FVE {base:.3f}. The word is \"{word}\" "
          f"({p}). Effect is FVE points lost,")
    print("so a positive number means the reconstruction got worse.")
    print()
    print("Intact:")
    print()
    print(K.wrap_md(splice(sent, hl, f"[{word}]"), 92, indent="    "))
    print()
    print(f"Corpus swap, the draw nearest the mean of 16 "
          f"(mean {swap_mean:+.2f} points):")
    print()
    print(K.wrap_md(splice(sent, hl, f"[{swap['substitute']}]"), 92, indent="    "))
    print()
    print(f"    {fve_line(swap['fve'])}")
    print()
    print(f"Masked-LM, the draw nearest the mean of the {len(mlm_nonid)} draws "
          f"that changed the word")
    print(f"(mean {mlm_mean:+.2f} points; the other {len(mlm_all) - len(mlm_nonid)} "
          f"of {len(mlm_all)} draws put \"{word}\" back):")
    print()
    print(K.wrap_md(splice(sent, hl, f"[{mlm['substitute']}]"), 92, indent="    "))
    print()
    print(f"    {fve_line(mlm['fve'])}")
    print()
    print("Deletion:")
    print()
    i, j = hl
    print(K.wrap_md(sent[:i].rstrip() + " [] " + sent[j:].lstrip(), 92, indent="    "))
    print()
    print(f"    {fve_line(dele['fve'])}")
    print()
    print(f"Shuffle, every one of the document's {n_sub} lexical words permuted "
          f"among its own slots, the")
    print(f"permutation nearest the mean of 4 (mean FVE {fves.mean():.3f}). "
          f"First {HEAD} words of the document:")
    print()
    print(K.wrap_md(K.head_words(" ".join(text.split()), HEAD)[0], 92, indent="    "))
    print(K.wrap_md(K.head_words(" ".join(shuf.split()), HEAD)[0], 92, indent="    "))
    print()
    print(f"    {fve_line(sfve)}")
    K.untee()
    conn.close()


if __name__ == "__main__":
    main()
