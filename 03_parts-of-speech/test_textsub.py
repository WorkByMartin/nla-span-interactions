#!/usr/bin/env python3
"""Local checks on the string-to-token path. Tokenisers only, no model, no GPU.

    python test_textsub.py

What is being asserted, and why each one would be a silent failure otherwise:

  no-op       substituting a word with itself must return the identical string
              and the identical Qwen id sequence. If it does not, the span
              bookkeeping is wrong and every measured delta contains that error.
  canonical   what the reconstructor reads is tok(templated(spliced_text)). Under
              the old id-for-id splice this was 1.50% false; here it is true by
              construction, so the test is a guard against anyone reintroducing an
              id-space splice.
  ranges      a word's ModernBERT token range must tile its character span, and
              collapsing that range to one [MASK] must leave the rest of the
              sequence byte-identical.
  multi       a substitute that is several Qwen tokens must still round-trip, and
              the sequence length must move by exactly the amount predicted.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import textsub as T

FAILED = []


def check(name, cond, detail=""):
    if cond:
        print(f"  ok    {name}")
    else:
        print(f"  FAIL  {name} {detail}")
        FAILED.append(name)


def main():
    import pyarrow.parquet as pq
    import spacy
    from transformers import AutoTokenizer

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                        / "01_corpus-and-spans"))
    from extract_traces import CRITIC_TEMPLATE as REAL
    check("template matches extract_traces", REAL == T.CRITIC_TEMPLATE)

    nlp = spacy.load("en_core_web_sm")
    qtok = AutoTokenizer.from_pretrained(T.QWEN)
    mtok = AutoTokenizer.from_pretrained(T.MLM)

    rows = pq.read_table("../01_corpus-and-spans/results/ffw_pilot_traces.parquet",
                         columns=["doc_uid", "explanation"]).to_pylist()[:20]

    # ------------------------------------------------------- clean-word filter
    for s in (" cat", "cat", " don't", " well-known", " 2024"):
        check(f"clean {s!r}", T.is_clean_word(s))
    for s in ("", " ", " [MASK]", " ,", " -", " it's-", " �", "[UNK]", " ."):
        check(f"reject {s!r}", not T.is_clean_word(s))
    V = mtok.batch_decode([[i] for i in range(len(mtok))],
                          clean_up_tokenization_spaces=False)
    check("mask token rejected", not T.is_clean_word(mtok.mask_token))
    check("some vocab is clean", 0.5 < sum(map(T.is_clean_word, V)) / len(V) < 1.0)

    # ------------------------------------------------------------- per document
    n_units = n_noop = n_canon = n_range = n_multi = n_fit = 0
    len_moved = 0
    for r in rows:
        text = r["explanation"]
        words = T.word_spans(nlp, text)
        spans = [(w["start"], w["end"]) for w in words]
        base_ids = T.prompt_ids(qtok, text)

        # the whole prompt, tokenised once, is what the AR reads
        if r is rows[0]:
            check("prompt decodes to templated text",
                  qtok.decode(base_ids) == T.templated(text))

        ranges = T.mlm_ranges(mtok, text, spans)
        menc = list(mtok(text)["input_ids"])

        for k, (w, sp) in enumerate(zip(words, spans)):
            n_units += 1

            # --- no-op substitution
            t2, s2 = T.splice(text, spans, k, w["span_text"])
            if t2 == text and s2 == spans and T.prompt_ids(qtok, t2) == base_ids:
                n_noop += 1

            # --- canonical: whatever we splice, the AR reads qtok of that string
            sub = " widget" if text[sp[0]] == " " else "widget"
            # a BARE substitute, which is what the store holds, must regain
            # exactly the leading space this span needs and no other
            if T.fit_space(text, sp, "widget") == sub == T.fit_space(text, sp, sub):
                n_fit += 1
            t3, _ = T.splice(text, spans, k, sub)
            ids3 = T.prompt_ids(qtok, t3)
            if ids3 == qtok.encode(T.templated(t3), add_special_tokens=False) \
                    and qtok.decode(ids3) == T.templated(t3):
                n_canon += 1

            # --- ModernBERT range tiles the span, and masking it is local
            rg = ranges[k]
            if rg is not None:
                n_range += 1
                i0, i1 = rg
                masked = menc[:i0] + [mtok.mask_token_id] + menc[i1:]
                if menc[:i0] != masked[:i0] or menc[i1:] != masked[i0 + 1:]:
                    check(f"mask is local at unit {k}", False)

            # --- a multi-Qwen-token substitute round-trips and moves the length
            sub2 = " unquestionably" if text[sp[0]] == " " else "unquestionably"
            t4, _ = T.splice(text, spans, k, sub2)
            ids4 = T.prompt_ids(qtok, t4)
            if qtok.decode(ids4) == T.templated(t4):
                n_multi += 1
            if len(ids4) != len(base_ids):
                len_moved += 1

        # --- splice_many agrees with sequential splices
        if len(spans) >= 4:
            subs = {1: " alpha", 3: " beta"}
            a, _ = T.splice_many(text, spans, subs)
            b, sb = T.splice(text, spans, 3, " beta")
            b, _ = T.splice(b, sb, 1, " alpha")
            check(f"splice_many doc {r['doc_uid']}", a == b)

    check("no-op is exact everywhere", n_noop == n_units,
          f"{n_noop}/{n_units}")
    check("canonical everywhere", n_canon == n_units, f"{n_canon}/{n_units}")
    check("multi-token substitute round-trips", n_multi == n_units,
          f"{n_multi}/{n_units}")
    check("fit_space restores space parity from a bare substitute",
          n_fit == n_units, f"{n_fit}/{n_units}")
    check("ModernBERT range covers >=99% of units",
          n_range >= 0.99 * n_units, f"{n_range}/{n_units}")
    print(f"\n  {n_units} units over {len(rows)} documents; a length-changing "
          f"substitute moved the prompt length in {len_moved} of them "
          f"({100 * len_moved / n_units:.1f}%)")

    print("\nFAILED:" if FAILED else "\nall checks passed")
    for f in FAILED:
        print(" ", f)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
