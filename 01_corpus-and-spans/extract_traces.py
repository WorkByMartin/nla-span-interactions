"""Run documents through the NLA once and record what the reconstructor sees.

    python extract_traces.py --n 100 --out ffw-5k_pilot_traces.parquet

Per document: the layer-42 activation, the verbalisation the RL verbaliser produced
for it, per-token KL between that verbaliser and the SFT reference, the
reconstruction, and its FVE.

Mechanism follows EasyNLA (github.com/asherps/EasyNLA) exactly. The parts that are
easy to get wrong and silent when wrong are marked WHY.
"""
import argparse, json, math, re, time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from safetensors.torch import load_file

D_MODEL = 5120
LAYER_INDEX = 42            # output of decoder block 42 == HF hidden_states[43]
MSE_SCALE = math.sqrt(D_MODEL)   # sidecar omits mse_scale; absent defaults to sqrt(d_model)
MAX_EXTRACT_LEN = 2048      # EasyNLA's extractor truncation; positions past it are OOD
INJ_ID, LEFT_ID, RIGHT_ID = 158983, 29, 510
INJECT_LAYER = 1            # hook the OUTPUT of block 1, per Karvonen et al.

ACTOR_TEMPLATE = (
    "You are a meticulous AI researcher conducting an important investigation into "
    "activation vectors from a language model. Your overall task is to describe the "
    "semantic content of that activation vector.\n\n"
    "We will pass the vector enclosed in <concept> tags into your context. You must "
    "then produce an explanation for the vector, enclosed within <explanation> tags. "
    "The explanation consists of 2-3 text snippets describing that vector.\n\n"
    "Here is the vector:\n\n"
    "<concept>{injection_char}</concept>\n\n"
    "Please provide an explanation."
)
INJECTION_CHAR = "㈜"
CRITIC_TEMPLATE = "Summary of the following text: <text>{explanation}</text> <summary>"
EXPLANATION_RE = re.compile(r"<explanation>(.*?)</explanation>", re.S)


def normalize_activation(v, scale):
    """Scale to a fixed L2 norm. Norm in fp32, single division. EasyNLA nla/schema.py."""
    if scale is None:
        return v
    n = v.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return v / (n / scale).to(v.dtype)


def cjk_fraction(text):
    """Injection-failure smoke signal: a missed injection makes the verbaliser echo
    the marker glyph and free-associate in Chinese. English explanations are ~0%."""
    import unicodedata
    if not text:
        return 0.0
    return sum(1 for c in text if "CJK" in unicodedata.name(c, "")) / len(text)


# --------------------------------------------------------------- injection ----
def register_karvonen_hook(model, vectors_ref):
    """h'_p = h_p + ||h_p|| * v/||v|| at the marker token, on block INJECT_LAYER's
    output. WHY not an embedding overwrite: the AV was trained with this and only
    this. A wrong site still generates fluent text, so nothing would flag it."""
    state = {"input_ids": None}

    def embed_hook(module, args, kwargs, output):
        ids = kwargs.get("input") if kwargs else None
        if ids is None and args:
            ids = args[0]
        state["input_ids"] = ids
        return output

    def layer_hook(module, args, output):
        resid, rest = (output[0], output[1:]) if isinstance(output, tuple) else (output, None)
        ids = state["input_ids"]
        # seq_len < 2 is a decode step after prefill: the marker is already behind us
        if ids is None or resid.shape[1] < 2:
            return output
        v = vectors_ref[0]
        if v is None or v.shape[0] == 0:
            return output
        ids = ids.to(resid.device)
        out = resid.clone()
        vecs = v.to(resid.device, resid.dtype)
        hits = 0
        for b, p in (ids == INJ_ID).nonzero().tolist():
            if p == 0 or p == ids.shape[-1] - 1:
                continue
            if ids[b, p - 1] != LEFT_ID or ids[b, p + 1] != RIGHT_ID:
                continue
            if hits >= vecs.shape[0]:
                hits += 1          # count it so the assert below reports the truth
                continue
            h_p = out[b, p].clone()
            unit = vecs[hits] / (vecs[hits].norm() + 1e-9)
            out[b, p] = h_p + h_p.norm() * unit
            hits += 1
        assert hits == vecs.shape[0], (
            f"injected {hits} sites, expected {vecs.shape[0]}: template or tokenizer drift"
        )
        return out if rest is None else (out, *rest)

    model.get_input_embeddings().register_forward_hook(embed_hook, with_kwargs=True)
    base = model.get_base_model() if hasattr(model, "peft_config") else model
    base.model.layers[INJECT_LAYER].register_forward_hook(layer_hook)


TIMING = {}


def stamp(label, t):
    """Wall time per stage, so a later run can be sized from measurement rather
    than from a guess about which part is slow."""
    dt = time.time() - t
    TIMING[label] = round(dt, 1)
    peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    print(f"  [{label}] {dt:.1f}s   peak GPU {peak:.1f} GB", flush=True)
    torch.cuda.reset_peak_memory_stats()
    return time.time()


def check_load(info, where, allow_missing=frozenset()):
    """A prefix mismatch loads a randomly initialised 27B backbone and raises
    nothing, so the reconstructions would be garbage and the FVE would read as a
    real negative result. Fail here instead."""
    missing = set(info.get("missing_keys") or []) - set(allow_missing)
    unexpected = [k for k in (info.get("unexpected_keys") or [])
                  if not k.startswith(("mtp", "model.visual", "visual"))]
    assert not missing, f"{where}: {len(missing)} weights did not load, e.g. {sorted(missing)[:3]}"
    assert not unexpected, f"{where}: unexpected keys, e.g. {unexpected[:3]}"


def release(*names):
    """Caller must already have dropped its own references; this just returns the
    blocks to the driver. `del` inside a helper only unbinds the helper's local."""
    torch.cuda.empty_cache()


# ------------------------------------------------------- phase A: activations ----
def extract_activations(base_dir, texts, positions, device, _t0=(0,)):
    """Layer-42 residual stream at each document's read-out position, raw (the
    sidecar says norm: none, and normalisation is training-side)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    _t0 = [time.time()]
    tok = AutoTokenizer.from_pretrained(base_dir)
    model, info = AutoModelForCausalLM.from_pretrained(
        base_dir, dtype=torch.bfloat16, device_map=device,
        attn_implementation="sdpa", output_loading_info=True)
    check_load(info, base_dir)
    model.eval()
    t = stamp("A_load", _t0[0])

    cap = {}

    def hook(mod, args, out):
        cap["h"] = out[0] if isinstance(out, tuple) else out

    model.model.layers[LAYER_INDEX].register_forward_hook(hook)

    acts = torch.empty(len(texts), D_MODEL, dtype=torch.float32)
    for i, (text, pos) in enumerate(zip(texts, positions)):
        # EasyNLA extracts with add_special_tokens=True, but this tokenizer's
        # post-processor is ByteLevel and bos_token is null, so both settings give
        # identical ids. False also matches how draw_corpus.py counted, keeping
        # `pos` valid. If this tokenizer ever gains a BOS, every position shifts.
        ids = tok(text, return_tensors="pt", add_special_tokens=False,
                  truncation=True, max_length=MAX_EXTRACT_LEN).input_ids.to(device)
        assert pos < ids.shape[1], f"row {i}: position {pos} past truncated length {ids.shape[1]}"
        with torch.no_grad():
            model.model(ids, use_cache=False)
        acts[i] = cap["h"][0, pos].float().cpu()
        if (i + 1) % 20 == 0:
            print(f"  activations {i + 1}/{len(texts)}", flush=True)
    stamp("A_forward", t)
    del model, cap
    release()
    return acts


# ---------------------------------------------------- phase B: verbalise + KL ----
def verbalise_and_kl(av_dir, adapter_dir, acts, device, max_new_tokens, temperature):
    """Generate one verbalisation per activation, then teacher-force it through the
    same model with the adapter on and off to get KL(RL || SFT) per generated token.
    Adapter off is the SFT reference: SFT is merged into these base weights."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    t_load = time.time()
    tok = AutoTokenizer.from_pretrained(av_dir)
    model, info = AutoModelForCausalLM.from_pretrained(
        av_dir, dtype=torch.bfloat16, device_map=device,
        attn_implementation="sdpa", output_loading_info=True)
    check_load(info, av_dir)
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()

    vec_ref = [None]
    register_karvonen_hook(model, vec_ref)

    # WHY enable_thinking=False: the default branch of this template ends the prompt
    # with a bare `<think>\n`, which leaves the model inside a reasoning block. It then
    # writes the explanation body without ever emitting the opening <explanation> tag
    # and every extraction fails. The False branch closes the block immediately.
    text = tok.apply_chat_template(
        [{"role": "user", "content": ACTOR_TEMPLATE.format(injection_char=INJECTION_CHAR)}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    prompt_ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

    marks = (prompt_ids[0] == INJ_ID).nonzero().flatten()
    assert len(marks) == 1, f"expected one marker token, found {len(marks)}"
    j = marks.item()
    assert prompt_ids[0, j - 1].item() == LEFT_ID and prompt_ids[0, j + 1].item() == RIGHT_ID, \
        "marker neighbours wrong: the prompt template or tokenizer has drifted"
    n_prompt = prompt_ids.shape[1]
    print(f"  prompt {n_prompt} tokens, marker at {j}", flush=True)
    t = stamp("B_load", t_load)

    out = []
    t_gen = t_kl = 0.0
    for i in range(acts.shape[0]):
        vec_ref[0] = acts[i:i + 1].to(device)
        _tg = time.time()
        with torch.no_grad():
            # WHY top_p/top_k explicitly: av_base/generation_config.json ships
            # top_k=20, top_p=0.95, and generate() merges those in silently. The
            # published rollouts sampled untruncated (SamplingParams top_p=1.0,
            # top_k=-1), so leaving the defaults would sample from a different
            # policy than the one whose KL we are measuring.
            gen = model.generate(
                prompt_ids, do_sample=temperature > 0, temperature=temperature or None,
                top_p=1.0, top_k=0,
                max_new_tokens=max_new_tokens, pad_token_id=tok.eos_token_id)
            verbalisation = tok.decode(gen[0, n_prompt:], skip_special_tokens=True)
            torch.cuda.synchronize(); t_gen += time.time() - _tg; _tk = time.time()

            logits_rl = model(gen).logits
            with model.disable_adapter():
                logits_sft = model(gen).logits
            # position t-1 predicts token t, so generated tokens are scored from n_prompt-1
            sl = slice(n_prompt - 1, gen.shape[1] - 1)
            p = torch.log_softmax(logits_rl[0, sl].float(), dim=-1)
            q = torch.log_softmax(logits_sft[0, sl].float(), dim=-1)
            kl = (p.exp() * (p - q)).sum(-1).cpu()
            torch.cuda.synchronize(); t_kl += time.time() - _tk
        logits_rl = logits_sft = p = q = None

        m = EXPLANATION_RE.search(verbalisation)
        out.append({
            "verbalisation": verbalisation,
            "explanation": m.group(1).strip() if m else None,
            "gen_token_ids": gen[0, n_prompt:].tolist(),
            "kl_per_token": kl.tolist(),
            "cjk_fraction": cjk_fraction(verbalisation),
        })
        if (i + 1) % 10 == 0:
            print(f"  verbalised {i + 1}/{acts.shape[0]}", flush=True)
    n = acts.shape[0]
    TIMING["B_generate"] = round(t_gen, 1)
    TIMING["B_kl"] = round(t_kl, 1)
    print(f"  generate {t_gen:.1f}s ({t_gen / n:.2f}s/doc), "
          f"kl {t_kl:.1f}s ({t_kl / n:.2f}s/doc)", flush=True)
    stamp("B_total", t)
    del model
    release()
    return out


# ------------------------------------------------- phase C: reconstruct + FVE ----
def reconstruct(ar_dir, explanations, device, batch_size=8):
    """pred = value_head(normalize(last_hidden, MSE_SCALE)) at the last real token."""
    from transformers import AutoModel, AutoTokenizer
    ar_dir = Path(ar_dir)
    t_load = time.time()
    tok = AutoTokenizer.from_pretrained(ar_dir)
    model, info = AutoModel.from_pretrained(
        ar_dir, dtype=torch.bfloat16, device_map=device,
        attn_implementation="sdpa", output_loading_info=True)
    # norm.weight is legitimately absent: EasyNLA replaced it with Identity before
    # saving. Anything else missing means the checkpoint's key prefixes did not
    # match, which loads a randomly initialised backbone and raises nothing.
    check_load(info, ar_dir, allow_missing={"norm.weight"})
    # WHY: the value head was trained on the raw layer-K residual. Leaving the norm
    # in the forward path silently shifts every prediction.
    assert isinstance(getattr(model, "norm", None), nn.Module), \
        "no `norm` on the AR backbone: assigning one would leave the real norm in place"
    model.norm = nn.Identity()
    model.eval()

    sd = load_file(str(ar_dir / "value_head.safetensors"))
    assert len(sd) == 1, f"expected one value-head tensor, got {list(sd)}"
    w = next(iter(sd.values()))
    assert w.shape == (D_MODEL, D_MODEL), f"value head is {tuple(w.shape)}"
    head = nn.Linear(D_MODEL, D_MODEL, bias=False).to(device=device, dtype=torch.float32)
    head.weight.data.copy_(w.float())
    t = stamp("C_load", t_load)

    encoded = {i: tok.encode(CRITIC_TEMPLATE.format(explanation=e), add_special_tokens=False)
               for i, e in enumerate(explanations) if e}
    # EasyNLA drops prompts over 1024 tokens rather than scoring them, so which rows
    # enter the FVE mean is a definitional choice, not an implementation detail
    idx = [i for i, ids in encoded.items() if 0 < len(ids) <= 1024]
    preds = torch.full((len(explanations), D_MODEL), float("nan"))
    for cs in range(0, len(idx), batch_size):
        chunk = idx[cs:cs + batch_size]
        seqs = [encoded[i] for i in chunk]
        maxlen = max(len(s) for s in seqs)
        ids = torch.full((len(chunk), maxlen), tok.eos_token_id, dtype=torch.long, device=device)
        attn = torch.zeros((len(chunk), maxlen), dtype=torch.long, device=device)
        for r, s in enumerate(seqs):
            ids[r, :len(s)] = torch.tensor(s, device=device)
            attn[r, :len(s)] = 1
        with torch.no_grad():
            h = model(input_ids=ids, attention_mask=attn, use_cache=False).last_hidden_state
            last = attn.sum(dim=1) - 1
            last_h = h[torch.arange(len(chunk), device=device), last].float()
            with torch.autocast(device_type=device.split(":")[0], enabled=False):
                p = head(normalize_activation(last_h, MSE_SCALE)).float()
        for r, i in enumerate(chunk):
            preds[i] = p[r].cpu()
    stamp("C_forward", t)
    del model, head
    release()
    return preds


def fve(preds, golds, baseline_golds):
    """1 - mse / predict-the-mean baseline, both sides normalised, baseline being the
    raw variance of the normalised distribution. The looser baseline inflates FVE.

    baseline_golds is deliberately a wider sample than the scored rows: EasyNLA
    computes this denominator over thousands of activations, and a variance
    estimated from ~100 rows is both noisy and biased low by the subtracted mean.
    It is still a different draw from the corpus the NLA trained on, so the result
    is not strictly comparable to the published figure."""
    keep = ~torch.isnan(preds[:, 0])
    pred_n = normalize_activation(preds[keep].float(), MSE_SCALE)
    gold_n = normalize_activation(golds[keep].float(), MSE_SCALE)
    mse = ((pred_n - gold_n) ** 2).mean(dim=1)
    base_n = normalize_activation(baseline_golds.float(), MSE_SCALE)
    baseline = ((base_n - base_n.mean(dim=0, keepdim=True)) ** 2).mean().item()
    return 1.0 - mse.mean().item() / baseline, mse, baseline, int(keep.sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", default="ffw-5k_corpus.parquet")
    ap.add_argument("--assets", default=None, help="dir holding the asset directories")
    ap.add_argument("--n", type=int, default=100, help="documents carried through the NLA")
    ap.add_argument("--baseline-n", type=int, default=1000,
                    help="documents used for the predict-the-mean denominator; phase A "
                         "only, so cheap. Must be >= --n")
    ap.add_argument("--resume", action="store_true",
                    help="reuse phase checkpoints next to --out if present")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip", type=int, default=0,
                    help="offset into the shuffled draw before taking --n rows. The "
                         "pilot used skip 0 n 100, so skip 100 is a held-out set from "
                         "the identical permutation, disjoint by construction")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="1.0 reproduces the published eval; eval_temperature was unset")
    ap.add_argument("--out", default="ffw-5k_pilot_traces.parquet")
    args = ap.parse_args()

    import os
    assets = Path(args.assets or os.environ["ASSETS"])
    A = {k: assets / k for k in (
        "qwen36-27b_base_model", "qwen36-27b_av-sft_model",
        "qwen36-27b_av-rl-s600_adapter", "qwen36-27b_ar-l43-s600_model")}
    for k, p in A.items():
        assert p.is_dir(), f"missing asset {k} at {p}"
    device = "cuda:0"
    assert torch.cuda.is_available(), "no GPU visible"

    assert args.baseline_n >= args.skip + args.n, \
        "--baseline-n must cover --skip plus --n"
    ckpt_a = Path(args.out).with_suffix(".phaseA.pt")
    ckpt_b = Path(args.out).with_suffix(".phaseB.json")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    table = pq.read_table(args.corpus)
    n_all = table.num_rows
    # positions past the extractor's truncation were never reachable during
    # training, so those activations would be out of distribution
    table = table.filter(pa.compute.less(table.column("token_position"), MAX_EXTRACT_LEN))
    n_ok = table.num_rows
    g = torch.Generator().manual_seed(args.seed)
    # one shuffle, then the scored rows are a prefix of the baseline rows, so the
    # denominator's sample contains the numerator's
    order = torch.randperm(n_ok, generator=g)[:args.baseline_n].tolist()
    all_rows = table.take(order).to_pylist()
    rows = all_rows[args.skip:args.skip + args.n]
    print(f"corpus {n_all} rows, {n_all - n_ok} dropped for position >= {MAX_EXTRACT_LEN}, "
          f"drew {len(all_rows)} for the baseline and {len(rows)} to score "
          f"at offset {args.skip}, seed {args.seed}", flush=True)

    t0 = time.time()
    if args.resume and ckpt_a.exists():
        all_acts = torch.load(ckpt_a)
        print(f"== phase A: reused {ckpt_a} ==", flush=True)
    else:
        print("== phase A: layer-42 activations ==", flush=True)
        all_acts = extract_activations(A["qwen36-27b_base_model"],
                                       [r["text"] for r in all_rows],
                                       [r["token_position"] for r in all_rows], device)
        torch.save(all_acts, ckpt_a)
    acts = all_acts[args.skip:args.skip + args.n]

    if args.resume and ckpt_b.exists():
        gen = json.loads(ckpt_b.read_text())
        print(f"== phase B: reused {ckpt_b} ==", flush=True)
    else:
        print("== phase B: verbalise and score KL ==", flush=True)
        gen = verbalise_and_kl(A["qwen36-27b_av-sft_model"], A["qwen36-27b_av-rl-s600_adapter"],
                               acts, device, args.max_new_tokens, args.temperature)
        ckpt_b.write_text(json.dumps(gen))

    bad = [i for i, g_ in enumerate(gen) if g_["cjk_fraction"] > 0.2]
    assert not bad, (f"{len(bad)} verbalisations are mostly CJK (rows {bad[:5]}): "
                     "that is the injection-failure signature, not a bad sample")
    n_noexp = sum(1 for g_ in gen if g_["explanation"] is None)
    print(f"  {n_noexp}/{len(gen)} had no <explanation> block", flush=True)

    print("== phase C: reconstruct ==", flush=True)
    preds = reconstruct(A["qwen36-27b_ar-l43-s600_model"],
                        [g_["explanation"] for g_ in gen], device)
    fve_pct, mse, baseline, n_scored = fve(preds, acts, all_acts)
    print(f"\nFVE = {fve_pct * 100:.2f}%  over {n_scored} scored  "
          f"(baseline {baseline:.4f})", flush=True)

    mse_full, k = [], 0
    for i in range(len(rows)):
        if torch.isnan(preds[i, 0]):
            mse_full.append(None)
        else:
            mse_full.append(float(mse[k])); k += 1

    out = pa.table({
        "doc_uid": [r["doc_uid"] for r in rows],
        "global_id": [r["global_id"] for r in rows],
        "domain": [r["domain"] for r in rows],
        "n_tokens": [r["n_tokens"] for r in rows],
        "token_position": [r["token_position"] for r in rows],
        "activation": [acts[i].tolist() for i in range(len(rows))],
        "verbalisation": [g_["verbalisation"] for g_ in gen],
        "explanation": [g_["explanation"] for g_ in gen],
        "gen_token_ids": [g_["gen_token_ids"] for g_ in gen],
        "kl_per_token": [g_["kl_per_token"] for g_ in gen],
        "cjk_fraction": [g_["cjk_fraction"] for g_ in gen],
        "reconstruction": [None if torch.isnan(preds[i, 0]) else preds[i].tolist()
                           for i in range(len(rows))],
        "mse": mse_full,
    }).replace_schema_metadata({
        "consumes": json.dumps(["ffw-5k_corpus"] + sorted(A)),
        "seed": str(args.seed),
        "temperature": str(args.temperature),
        "max_new_tokens": str(args.max_new_tokens),
        "mse_scale": str(MSE_SCALE),
        "layer_index": str(LAYER_INDEX),
        "fve": f"{fve_pct:.6f}",
        "fve_baseline": f"{baseline:.6f}",
        "n_scored": str(n_scored),
        "n_baseline": str(all_acts.shape[0]),
        "n_no_explanation": str(n_noexp),
        "elapsed_s": f"{time.time() - t0:.0f}",
        "timing_s": json.dumps(TIMING),
    })
    pq.write_table(out, args.out)
    print(f"wrote {args.out} with {out.num_rows} rows", flush=True)
    print("timing: " + json.dumps(TIMING), flush=True)
    for c in (ckpt_a, ckpt_b):
        c.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
