#!/usr/bin/env python3
"""Reconstructor forward passes and the masked-LM proposal distribution.

Imported by `pos_fve.py`; not run on its own. Two halves:

  the reconstructor  determinism flags, the environment fingerprint, loading the
                     AR backbone with its separate value head, and scoring a list
                     of explanation strings against a gold activation.
  the sampler        ModernBERT masked fill over a word's ModernBERT token range,
                     filtered on space parity and the clean-word test, giving the
                     top-k conditional a substitute is drawn from.

Numerical policy, and why:

  eager attention.  SDPA's backward is nondeterministic for bf16 (atomicAdd on
  dQ), at roughly the magnitude of the effect being measured. Flash and
  mem-efficient SDPA also cannot double-backward at all. Eager sidesteps both.

  precision is the binding constraint, not determinism.  bf16 has 8 mantissa
  bits and rounds the residual stream once per layer across 43 layers. That
  error is deterministic, so it does not appear as run-to-run jitter, but it
  does not cancel between the baseline and the ablated forward. The only way to
  see it is a higher-precision reference: run --precision fp32 and compare.

  the readout is indexed from the attention mask, never [-1], so right padding
  past the readout is inert and rows of different lengths batch together.
"""
import os
import sys
from pathlib import Path

# must precede any CUDA initialisation
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import textsub as T  # noqa: E402

sys.path.insert(0, str(HERE.parent / "01_corpus-and-spans"))
from extract_traces import (  # noqa: E402
    D_MODEL, MSE_SCALE, normalize_activation, check_load)

MAX_PROMPT = 1024  # EasyNLA drops longer prompts rather than scoring them
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


# --------------------------------------------------------------- determinism

def make_deterministic(seed=0):
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False   # defaults True, so this one matters
    torch.set_float32_matmul_precision("highest")
    # warn_only=False is load-bearing: it is what forces the SDPA backends onto
    # their deterministic backward. With warn_only=True they warn once and carry on.
    torch.use_deterministic_algorithms(True, warn_only=False)


def fingerprint():
    return {"torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "sm_count": torch.cuda.get_device_properties(0).multi_processor_count,
            "cublas_ws": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}


# -------------------------------------------------------------------- model

def load_ar(ar_dir, device, precision):
    from transformers import AutoModel, AutoTokenizer
    ar_dir = Path(ar_dir)
    tok = AutoTokenizer.from_pretrained(ar_dir)
    model, info = AutoModel.from_pretrained(
        ar_dir, dtype=DTYPES[precision], device_map=device,
        attn_implementation="eager", output_loading_info=True)
    check_load(info, ar_dir, allow_missing={"norm.weight"})
    assert isinstance(getattr(model, "norm", None), nn.Module), \
        "no `norm` on the AR backbone: assigning one would leave the real norm in place"
    model.norm = nn.Identity()
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)  # without this, backward allocates a second copy of the weights

    sd = load_file(str(ar_dir / "value_head.safetensors"))
    assert len(sd) == 1, f"expected one value-head tensor, got {list(sd)}"
    w = next(iter(sd.values()))
    assert w.shape == (D_MODEL, D_MODEL), f"value head is {tuple(w.shape)}"
    head = nn.Linear(D_MODEL, D_MODEL, bias=False).to(device=device, dtype=torch.float32)
    head.weight.data.copy_(w.float())
    head.weight.requires_grad_(False)
    return model, tok, head


# ----------------------------------------------------------------- sampler

class Sampler:
    """ModernBERT masked fill over a word's ModernBERT token range.

    A word's character span maps to a contiguous range of ModernBERT tokens; the
    whole range is replaced by ONE [MASK], so a multi-token original is fine. The
    candidate is necessarily a single ModernBERT token, which is the only
    irreducible constraint, and it is filtered on its string alone: space parity
    with the original, and the clean-word test. Nothing here looks at Qwen.
    """

    def __init__(self, device, topk=50, draws=30):
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(T.MLM)
        self.model = AutoModelForMaskedLM.from_pretrained(T.MLM).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device, self.topk, self.draws = device, topk, draws

        self.vocab = self.tok.batch_decode(
            [[i] for i in range(len(self.tok))], clean_up_tokenization_spaces=False)
        spaced = T.spaced_forms(self.vocab)
        B = lambda xs: torch.tensor(xs, dtype=torch.bool, device=device)
        # each filter kept separately so the mass it costs can be attributed to
        # it, per position, rather than only reported jointly
        self.clean_m = B([T.is_clean_word(s) for s in self.vocab])
        self.space_m = {b: B([T.has_leading_space(s) == b for s in self.vocab])
                        for b in (True, False)}
        self.wordform_m = B([s in spaced for s in self.vocab])
        self.keep = {
            b: torch.tensor(T.candidate_mask(self.vocab, b, spaced),
                            dtype=torch.bool, device=device)
            for b in (True, False)}
        for b in (True, False):
            n = int(self.keep[b].sum())
            print(f"  sampler: {n}/{len(self.vocab)} candidates with "
                  f"leading_space={b}", flush=True)

    # ------------------------------------------------------------ conditionals
    def _filtered(self, logits, row, orig_has_space):
        """Top-k over the filtered distribution, plus the mass each filter cost."""
        p = logits[row].float().softmax(-1)
        keep = self.keep[orig_has_space]
        kept = torch.where(keep, p, torch.zeros_like(p))
        mass = float(kept.sum())
        if mass == 0:
            return None
        k = min(self.topk, int(keep.sum()))
        top = kept.topk(k)
        v = top.values[top.values > 0]
        idx = top.indices[: v.shape[0]]
        cm = self.clean_m if orig_has_space else (self.clean_m & self.wordform_m)
        return {"strs": [self.vocab[int(i)] for i in idx],
                "probs": (v / v.sum()).cpu(),
                # mass each filter would keep on its own, and both together, so a
                # position whose candidate set is thin can be traced to a cause
                "mass_parity": float(p[self.space_m[orig_has_space]].sum()),
                "mass_clean": float(p[cm].sum()),
                "mass_kept": mass,
                "top_unfiltered": self.vocab[int(p.argmax())],
                "entropy": float(-(p[p > 0] * p[p > 0].log()).sum())}

    def _forward(self, texts, spans):
        """One masked forward per (text, span), right-padded to the longest row.

        Each span's ModernBERT token range collapses to a single [MASK], so a
        multi-token original shortens its row and rows differ in length.
        Returns (logits or None, live row indices, mask index per input).
        """
        rows, targets = [], []
        for text, (a, b) in zip(texts, spans):
            ids = list(self.tok(text)["input_ids"])
            rng = T.mlm_ranges(self.tok, text, [(a, b)])[0]
            if rng is None:
                rows.append(None)
                targets.append(None)
                continue
            i0, i1 = rng
            rows.append(ids[:i0] + [self.tok.mask_token_id] + ids[i1:])
            targets.append(i0)
        live = [i for i, r in enumerate(rows) if r is not None]
        if not live:
            return None, live, targets
        L = max(len(rows[i]) for i in live)
        pad = self.tok.pad_token_id
        ids = torch.full((len(live), L), pad, dtype=torch.long)
        mask = torch.zeros((len(live), L), dtype=torch.long)
        for r, i in enumerate(live):
            n = len(rows[i])
            ids[r, :n] = torch.tensor(rows[i])
            mask[r, :n] = 1
        with torch.no_grad():
            lg = self.model(input_ids=ids.to(self.device),
                            attention_mask=mask.to(self.device)).logits
        return lg, live, targets

    def conditionals(self, text, char_spans, chunk=16):
        """One conditional per char span of `text`, aligned with char_spans.

        Entries whose span no whole set of ModernBERT tokens tiles come back None.
        """
        out = [None] * len(char_spans)
        for i in range(0, len(char_spans), chunk):
            spans = char_spans[i:i + chunk]
            lg, live, targets = self._forward([text] * len(spans), spans)
            if lg is None:
                continue
            for r, k in enumerate(live):
                out[i + k] = self._filtered(lg[r], targets[k],
                                            text[spans[k][0]] == " ")
        return out

    def draw_ix(self, cond, gen, n=None):
        """Indices into cond['strs'], sampled with replacement from the top-k.

        Callers that record what they drew need the index, because the
        probability of the draw is only recoverable from it.
        """
        n = self.draws if n is None else n
        return [int(i) for i in torch.multinomial(
            cond["probs"], n, replacement=True, generator=gen)]

    def draw(self, cond, gen, n=None):
        """Substitute STRINGS, sampled with replacement from the filtered top-k."""
        return [cond["strs"][i] for i in self.draw_ix(cond, gen, n)]


# ------------------------------------------------------------------- forwards

def mse_from_embeds(model, head, x, attn, gold_n):
    """x: (B, L, D). Returns (B,) fp32 per-row MSE against the gold vector.

    Mirrors reconstruct() exactly: normalise the last hidden state, apply the
    fp32 head, then normalise the prediction again before scoring, which is
    what fve() does. The last real token is indexed from the mask, never [-1].
    """
    h = model(inputs_embeds=x, attention_mask=attn, use_cache=False).last_hidden_state
    last = attn.sum(dim=1) - 1
    last_h = h[torch.arange(x.shape[0], device=x.device), last].float()
    with torch.autocast(device_type=x.device.type, enabled=False):
        pred = head(normalize_activation(last_h, MSE_SCALE)).float()
    pred_n = normalize_activation(pred, MSE_SCALE)
    return ((pred_n - gold_n.unsqueeze(0)) ** 2).mean(dim=1)


def encode_batch(tok, E, explanations, device):
    """Right-padded (embeds, attention) for a batch of whole templated prompts.

    The readout is indexed from the attention mask, never [-1], so right padding
    past the readout is inert.
    """
    seqs = [tok.encode(T.templated(e), add_special_tokens=False)
            for e in explanations]
    L = max(len(s) for s in seqs)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ids = torch.full((len(seqs), L), pad, dtype=torch.long, device=device)
    attn = torch.zeros((len(seqs), L), dtype=torch.long, device=device)
    for r, s in enumerate(seqs):
        ids[r, : len(s)] = torch.tensor(s, device=device)
        attn[r, : len(s)] = 1
    return E[ids].detach(), attn, [len(s) for s in seqs]


MAXLEN = [0]      # longest prompt any substitution produced
OVERLONG = [0]    # how many forwards exceeded the EasyNLA prompt cap


def mse_of_texts(model, tok, head, E, gold_n, explanations, batch, device):
    """Per-explanation MSE. One entry in, one number out, re-tokenised each time.

    Length is not fixed any more, so the EasyNLA prompt cap has to be watched at
    run time rather than argued about once at the top of the document.
    """
    vals, lens = [], []
    for i in range(0, len(explanations), batch):
        chunk = explanations[i:i + batch]
        x, attn, n = encode_batch(tok, E, chunk, device)
        lens.extend(n)
        with torch.no_grad():
            vals.append(mse_from_embeds(model, head, x, attn, gold_n).float().cpu())
        del x
    over = sum(1 for n in lens if n > MAX_PROMPT)
    if over and not OVERLONG[0]:
        print(f"  WARNING: a substitution pushed the prompt to {max(lens)} "
              f"tokens, past the {MAX_PROMPT} EasyNLA drops at. Scored anyway; "
              f"the count is reported per document.", flush=True)
    OVERLONG[0] += over
    MAXLEN[0] = max(MAXLEN[0], max(lens))
    return torch.cat(vals), lens
