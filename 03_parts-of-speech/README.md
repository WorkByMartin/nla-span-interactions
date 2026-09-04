# 03_parts-of-speech

Breaks a word-level ablation of the verbalisation down by the syntactic class of the ablated word, and by how far into the masked-LM candidate list the substitute was drawn from.

```bash
python pos_analysis.py --run 3
```

No GPU. Downloads the `answerdotai/ModernBERT-large` tokeniser files on first run, no weights. It reads run 3 of the `ffw_span-ablation_database` asset and writes every number below to `results/statistics.md`, which it prints and rewrites on every run. Seven figures go to `results/` alongside it.

## What is ablated

The unit is a spaCy word, from `en_core_web_sm` 3.8.0, with the preceding space folded into its character span. Punctuation, whitespace, symbols and the `X` tag are not ablatable units.

The substitute is drawn by marginalisation in text space. The word's ModernBERT token range collapses to one `[MASK]`, ModernBERT-large gives the conditional over its vocabulary at that position, and two filters are applied to the candidate string alone. Space parity requires a candidate to carry a leading space if and only if the original did. The clean-word test requires it to decode to letters and digits, with apostrophes and hyphens allowed only word-internally, which drops punctuation-only tokens, special tokens and byte fragments. Where the original had no leading space, a candidate must also exist in the vocabulary in its space-prefixed form, so that word-internal suffixes are not proposed at a word-initial position.

The surviving distribution is truncated to its top k, renormalised over k, and sampled with replacement. That k is the candidate depth, the second axis of the study, and it takes the values 1, 3, 5, 10, 25 and 50.

The drawn string is spliced into the explanation and the whole templated prompt is re-tokenised with the Qwen tokeniser, so the reconstructor always reads Qwen's own canonical tokenisation of the text that is actually there. Nothing is spliced in token-id space. The prompt length may therefore move, and it is recorded per variant as `seq_len`.

Run 3 took the first 40 documents of the trace set that fit the prompt cap, 40 spans per document chosen round robin across part-of-speech classes rather than by frequency, and 8 draws at each of the 6 depths. That is 76800 single-substitution forward passes. The per-document baseline rides in the same batched call as the arms it is differenced against.

## The measurement

The reconstructor predicts the layer-42 activation the verbalisation came from. Effect is FVE lost, in points, 100 times the drop in fraction of variance explained against the same document's unedited baseline, so a positive number means the reconstruction got worse. Standard errors are clustered on document over 40 clusters, and intervals use t(0.975) on 39 degrees of freedom.

A per-draw absolute MSE change below 2e-4 corresponds to 0.044 FVE points and is treated as the harness reproduction floor.

## The sample

Run 3 holds 76800 single-substitution draws over 40 documents, 1600 spans and 14 word classes. All 1600 spans are tiled exactly by whole ModernBERT tokens, none are dropped. Class purity is scored against a corpus lexicon of 3444 word types, each carrying its modal part of speech. These counts are in the inputs section of `results/statistics.md`.

## Effect by class and depth

Pooled over every class, the mean FVE lost is 0.018 points at depth 1, with a 95% clustered interval of [-0.006, 0.042] that includes zero, and between 0.029 and 0.037 points at depths 3 to 50, with intervals that exclude zero at every one of those depths. Each depth carries 12800 draws. This is the pooled-by-depth section.

Over all 40 documents, ADJ is the only class whose depth-50 interval excludes zero, at 0.073 +/- 0.031 points, interval [0.010, 0.136]. The full class by depth table is the second section of the statistics file.

Document 2159 has a baseline FVE of -0.133 and supplies all five of the largest negative substitutions in the report. The third section repeats the table with that document dropped. Over the remaining 39 documents, three classes have depth-50 intervals excluding zero: ADV at 0.068 +/- 0.027, ADJ at 0.065 +/- 0.031, and PRON at 0.039 +/- 0.019. NOUN moves from -0.069 +/- 0.100 to 0.030 +/- 0.016.

Ranked over the distinct substitute strings drawn for a word, the largest single word is `two` (NUM) in document 1934 at 8.421 points, and the smallest is `rest` (NOUN) in document 2159 at -8.423. 1686 of the 76800 draws, 2.2%, fall outside the plus or minus 1 point range the histogram figures clip to.

## Leakage

Leakage is the share of draws whose substitute is the original word. At depth 50 it runs from 0.031 for INTJ to 0.674 for SCONJ. Summing the class counts in the leakage section, 5279 of the 12800 depth-50 draws reproduce the original word, which is 41% of them. Seven classes leak on more than 40% of their depth-50 draws: SCONJ 0.674, CCONJ 0.661, PRON 0.651, DET 0.627, ADP 0.579, AUX 0.573 and PART 0.441.

The same section recomputes each class mean over the draws that actually changed the word. DET goes from 0.001 to 0.002, CCONJ from 0.000 to 0.001 and ADP from -0.004 to -0.009, so those three closed classes are within 0.01 points of zero whether or not leaked draws are included. PRON moves from 0.080 to 0.231, SCONJ from 0.028 to 0.084 and AUX from 0.027 to 0.063.

The reproduction-floor section gives the complementary count. 44508 of the 76800 draws, 0.5795, sit below the 0.044-point floor. Per class the share below the floor is 0.3328 for PROPN at the lowest and above 0.73 for six classes: SCONJ 0.7789, PRON 0.7710, DET 0.7605, CCONJ 0.7588, AUX 0.7424 and ADP 0.7350.

## Sequence length

71616 of the 76800 substitutions, 93.25%, leave the Qwen token count unchanged. 4.95% shorten it by one and 0.82% by two, 0.47% lengthen it by one and 0.14% by two, and 0.36% move it further. The sequence-length section carries the full table.

Splitting each class by whether the substitution changed the token count, the length-changing draws carry the larger mean in most classes. SCONJ is 0.009 against 0.301, ADJ 0.057 against 0.268, VERB 0.026 against 0.220 and PROPN 0.059 against 0.174. CCONJ and INTJ never changed length in this run.

## Regenerating run 3

The GPU side is `pos_fve.py`, which computes and caches the ModernBERT candidate lists, runs the reconstructor forwards, and writes each variant with its substitution and its measurements back to the store.

```bash
python pos_fve.py --ar "$ASSETS/qwen36-27b_ar-l43-s600_model" \
    --traces ../01_corpus-and-spans/results/ffw_pilot_traces.parquet \
    --db ../db/ffw_span-ablation_database.sqlite \
    --docs 40 --per-doc 40 --draws 8 --batch 8 --precision bf16 --seed 0
```

It needs `en_core_web_sm` installed alongside spaCy, and the assets `qwen36-27b_ar-l43-s600_model`, `qwen36-27b_tokenizer` and `modernbert-large_filler_model`. It appends a new run rather than overwriting run 3, so `pos_analysis.py --run <new id>` is what reads it back.

Determinism is set before any CUDA initialisation: `CUBLAS_WORKSPACE_CONFIG=:4096:8`, TF32 off, `use_deterministic_algorithms` with `warn_only=False`, and eager attention rather than SDPA, whose bf16 backward is nondeterministic. Precision, not determinism, is the binding constraint, since bf16 rounds the residual stream once per layer across 43 layers and that error does not cancel between the baseline and the ablated forward. Run 3 was recorded at bf16 on an A100 80GB PCIe under torch 2.6.0 with CUDA 12.4; the run's fingerprint and every argument it was given are in the `runs` table of the store.

## Files

```
pos_analysis.py                       the tables and the seven figures, read from the store
pos_fve.py                            the GPU run that wrote run 3
harness.py                            reconstructor forwards and the ModernBERT sampler
textsub.py                            candidate filters, word spans, text-space splicing
test_textsub.py                       string-to-token checks, tokenisers only, no model
results/statistics.md                 the printed report, thirteen sections
results/fve_lost_by_class.png         per-draw distribution of FVE lost, one panel per class, by depth
results/fve_lost_all_classes.png      the same distribution pooled over every class
results/mean_by_depth.png             mean FVE lost against candidate depth, per class and pooled
results/leakage_purity_by_depth.png   leakage and candidate class purity against depth
results/effect_vs_leakage.png         depth-50 effect against depth-50 leakage, marker area by draw count
results/multitoken_vs_single.png      depth-50 FVE lost split by the ModernBERT token count of the word
results/per_document.png              per-document mean effect against baseline reconstruction quality
```
