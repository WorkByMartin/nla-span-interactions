#!/usr/bin/env python3
"""Text-space substitution: the shared machinery for the marginalisation ablation.

Substitution happens on the explanation STRING, not on Qwen token ids. That is
the whole difference from the first version of this pipeline, and it removes two
of the three constraints the old single-token rule bundled together:

  1. The substitute must come from one masked-LM forward, so it is necessarily
     ONE ModernBERT token. Irreducible, and it binds the substitute only.
  2. The original need NOT be one ModernBERT token. A word's character span maps
     to a ModernBERT token RANGE, and the whole range is replaced by a single
     [MASK]. So `unquestionably` (three ModernBERT tokens) is ablatable.
  3. The substitute need NOT be one Qwen token. The substitute STRING is spliced
     into the explanation and the whole templated prompt is re-tokenised with the
     Qwen tokeniser. What the reconstructor reads is therefore always Qwen's own
     canonical tokenisation of the text that is actually there. The sequence
     length may change; RoPE positions after the splice point may shift. That is
     accepted, and it is the price of not biasing the candidate set.

Two filters remain, and both are properties of the candidate string alone:

  space parity  a candidate carries a leading space iff the original did. Without
                it, ablating ` cat` could yield `catalogue`-style word-internal
                fragments glued onto the preceding word.
  clean word    the candidate decodes to letters and digits, with apostrophes and
                hyphens allowed only word-internally. This drops punctuation-only
                tokens, special tokens, and byte fragments that decode to U+FFFD.

This module holds no torch and loads no model, so the string-to-token path can be
unit-tested with the Qwen tokeniser alone.
"""

import re

QWEN = "Qwen/Qwen3.6-27B"
MLM = "answerdotai/ModernBERT-large"

# Duplicated from ../01_corpus-and-spans/extract_traces.py so this module stays
# torch-free.
# test_textsub.py asserts the two are identical.
CRITIC_TEMPLATE = "Summary of the following text: <text>{explanation}</text> <summary>"

# Parts of speech that are not sensible ablation units. Punctuation carries no
# lexical content and ModernBERT routinely merges adjacent punctuation into one
# token, so a punctuation "word" often has no span of its own to mask.
NON_LEXICAL_POS = {"PUNCT", "SPACE", "SYM", "X"}

_CLEAN = re.compile(r"^[^\W_]+(?:['\-][^\W_]+)*$", re.UNICODE)


# --------------------------------------------------------------- candidates

def has_leading_space(s):
    return s.startswith(" ")


def is_clean_word(s):
    """Does this decoded vocabulary entry look like a word rather than a fragment?

    Letters or digits, optionally with internal apostrophes or hyphens. A leading
    space is stripped before testing, since space parity is a separate filter.
    Rejects `[MASK]`-style special tokens, punctuation-only tokens, tokens that
    decode to a U+FFFD replacement character (an incomplete UTF-8 sequence), and
    anything with a leading or trailing hyphen or apostrophe.
    """
    if not s:
        return False
    w = s[1:] if s.startswith(" ") else s
    if not w or "�" in w:
        return False
    return bool(_CLEAN.match(w))


def spaced_forms(vocab_strings):
    """The set of bare strings that also exist in the vocabulary space-prefixed.

    A word-initial position (start of the explanation, or just after an opening
    quote or bracket) has no leading space, so space parity cannot tell a word
    from a word-internal fragment there. Membership of this set can: `The` is a
    word because ` The` is also a token, whereas `ing` is not because ` ing` is
    not one.
    """
    v = set(vocab_strings)
    return {s for s in v if s and not s.startswith(" ") and " " + s in v}


def candidate_mask(vocab_strings, orig_has_space, spaced=None):
    """Boolean list over the vocabulary: passes clean-word AND space parity.

    `spaced` is the set from spaced_forms(); when supplied it additionally
    requires a no-leading-space candidate to be a word rather than a suffix.
    """
    out = []
    for s in vocab_strings:
        ok = is_clean_word(s) and (has_leading_space(s) == orig_has_space)
        if ok and not orig_has_space and spaced is not None:
            ok = s in spaced
        out.append(ok)
    return out


# --------------------------------------------------------------------- spans

def word_spans(nlp, text, lexical_only=True):
    """spaCy words as character spans, with the preceding space folded in.

    Both tokenisers are byte-level BPE and carry the leading space inside the
    token, so the token covering a word starts one character before spaCy's span.
    Folding the space into the unit is what makes space parity meaningful.
    """
    out = []
    for t in nlp(text):
        if not t.text.strip():
            continue
        if lexical_only and t.pos_ in NON_LEXICAL_POS:
            continue
        s, e = t.idx, t.idx + len(t.text)
        if s > 0 and text[s - 1] == " ":
            s -= 1
        out.append({"text": t.text, "pos": t.pos_, "tag": t.tag_,
                    "start": s, "end": e, "span_text": text[s:e]})
    return out


def covered_exactly(spans, a, b):
    """Indices of tokens whose spans tile [a, b) with no overhang either side."""
    hit = [i for i, (x, y) in enumerate(spans) if x < b and y > a]
    if not hit:
        return None
    if spans[hit[0]][0] != a or spans[hit[-1]][1] != b:
        return None
    return hit


def real_spans(tok, text, **kw):
    """(start, end) of every token that covers real characters, plus their indices.

    Special tokens carry (0, 0) and are dropped, so index i of the returned list
    is not the position in the id sequence; the index is returned alongside.
    """
    offs = tok(text, return_offsets_mapping=True, **kw)["offset_mapping"]
    return [(i, a, b) for i, (a, b) in enumerate(offs) if b > a]


def mlm_ranges(mtok, text, spans):
    """For each (start, end) char span, the ModernBERT token index range covering it.

    Returns a list aligned with `spans` of (i0, i1) half-open id-sequence indices,
    or None where no whole set of ModernBERT tokens tiles the span exactly.
    """
    real = real_spans(mtok, text)
    idx = [i for i, _, _ in real]
    off = [(a, b) for _, a, b in real]
    out = []
    for a, b in spans:
        hit = covered_exactly(off, a, b)
        out.append(None if hit is None else (idx[hit[0]], idx[hit[-1]] + 1))
    return out


# ------------------------------------------------------------------ splicing

def splice(text, spans, k, new_str):
    """Replace unit k's characters with new_str, shifting every later span.

    Unlike the old id-for-id path, a substitution that changes character length
    is fine: the whole prompt is re-tokenised afterwards.
    """
    a, b = spans[k]
    text = text[:a] + new_str + text[b:]
    d = len(new_str) - (b - a)
    spans = [(x, y) if i < k else (a, a + len(new_str)) if i == k
             else (x + d, y + d) for i, (x, y) in enumerate(spans)]
    return text, spans


def fit_space(text, span, word):
    """Give a BARE substitute the leading space this span needs.

    The store holds words, not word-with-space: space parity is this module's
    internal business, folded in by word_spans when it builds the unit and put
    back here when the unit is spliced. A span whose first character is a space
    is one whose space was folded in, so its substitute needs one too.
    """
    a, _ = span
    w = word[1:] if word.startswith(" ") else word
    return " " + w if a < len(text) and text[a] == " " else w


def splice_many(text, spans, subs):
    """Apply several substitutions at once. subs: {unit index: replacement string}.

    Applied right to left so earlier spans are untouched while later ones shift,
    then the shifted span list is rebuilt in one pass so callers can keep working
    on the result.
    """
    out_spans = list(spans)
    for k in sorted(subs, reverse=True):
        text, out_spans = splice(text, out_spans, k, subs[k])
    return text, out_spans


def templated(explanation):
    return CRITIC_TEMPLATE.format(explanation=explanation)


def prompt_ids(qtok, explanation):
    """Qwen ids the reconstructor actually reads for this explanation.

    One encode of the whole templated string, which is what production does. There
    is no splice in id space anywhere in this pipeline, so this is canonical by
    construction.
    """
    return qtok.encode(templated(explanation), add_special_tokens=False)
