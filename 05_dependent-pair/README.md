# 05_dependent-pair

Measures whether two words joined by a dependency arc interact more, when both are ablated, than two matched words that are not joined. The quantity is the interaction e(a) + e(b) - e(both) in FVE points, where e is the drop in fraction of variance explained against the same document's unedited baseline, times 100. A positive interaction means the pair costs less than the sum of its two singles.

```bash
python pair_analysis.py --run 7
```

No GPU, and no model files. It reads run 7 of the `ffw_span-ablation_database` asset, takes the pair table from that run's row in the `runs` table, checks it against the `pair-ablation/arc+control` edges in `relations` before computing anything, and writes every number below to `results/statistics.md`, which it prints and rewrites on every run. Two figures go to `results/` beside it, and `arc_vs_control_by_type.py` draws a third from the printed per-type table.

## What is ablated

The edit is the corpus swap of 04. A word is replaced by a word drawn uniformly over the lexical span occurrences of other documents' verbalisations, matched on spaCy coarse part of speech, Qwen token count and leading-space parity, with the original excluded case-blind and the substitute recased to the original's shape. The pool holds 12854 lexical occurrences, 3338 word types, over 100 documents.

Two spans are ablated singly and together, eight draws each, so the interaction is formed per draw. A span's substitute is a property of the span and the draw, so the single at draw k and the joint edit at draw k splice the same string at that span, and a span used by several pairs contributes one set of singles rather than one per pair. The baseline is the intact document, batched in the same call as the arms it is differenced against.

An arc pair is a direct dependency arc between two lexical words, carrying one of nine spaCy dep labels on the dependent (dobj, nsubj, conj, appos, nmod, advcl, ccomp, attr, poss), with the two words at least two tokens apart so no pair is a contiguous bigram. A control pair is two words with the same ordered part-of-speech combination and the same token distance, with no arc between them in either direction and no path of length two through a shared head or a head-of-head, so siblings and grandparent pairs are excluded. The control search prefers the same document at the exact distance, then the same document at a distance one token either side, then a different document of the same domain at the exact distance, then a different document at a distance one token either side.

## The sample

Run 7 covers 100 documents and 2455 pairs, 1280 arc and 1175 control, eight draws each. That is 19640 draw-level interactions formed against 29304 single-span measurements and 19640 joint variants. Up to 150 arcs were sampled per dependency type, and attr and poss had 111 and 119 available and took all of them. 105 sampled arcs never found a control, so they appear in the arc rows but not in the paired comparison. The pass budget was 100 baselines, 29304 singles and 19640 joint edits, 49044 forward passes over 3663 distinct spans, each span appearing in 1.34 pairs on average and at most 6. These counts are in the setup header of `results/statistics.md` and in the run's `runs` row.

Standard errors are clustered on document over 100 clusters unless a line says otherwise. The setup header reports a harness floor of 0.044 FVE points.

## Overall

From the overall section, pair means in FVE points with a document-clustered standard error.

| set | pairs | mean | median | mean absolute | share over floor |
| --- | --- | --- | --- | --- | --- |
| arc | 1280 | -0.0171 +- 0.0225 | -0.0028 | 0.1368 | 0.571 |
| control | 1175 | -0.0245 +- 0.0190 | -0.0026 | 0.1207 | 0.514 |
| all | 2455 | -0.0206 +- 0.0193 | -0.0026 | 0.1291 | 0.544 |

Paired inside the match, arc minus its own control over the 1175 matched pairs, the difference is +0.0056 +- 0.0176 points, and the pair is more sub-additive on an arc in 0.504 of matches.

## Per dependency type

From the per-type section, in FVE points. The arc minus control column is paired inside the match, so it uses only arcs that found a control.

| dep | n arc | arc mean | n control | control mean | arc - control |
| --- | --- | --- | --- | --- | --- |
| dobj | 150 | -0.0378 +- 0.0209 | 146 | +0.0016 +- 0.0204 | -0.0388 +- 0.0284 |
| nsubj | 150 | -0.0245 +- 0.0438 | 132 | -0.0384 +- 0.0407 | +0.0014 +- 0.0431 |
| conj | 150 | -0.0098 +- 0.0217 | 148 | +0.0034 +- 0.0180 | -0.0131 +- 0.0208 |
| appos | 150 | +0.0055 +- 0.0168 | 146 | -0.0476 +- 0.0312 | +0.0541 +- 0.0321 |
| nmod | 150 | -0.0987 +- 0.0526 | 144 | -0.0916 +- 0.0910 | -0.0106 +- 0.1086 |
| advcl | 150 | +0.0030 +- 0.0290 | 140 | -0.0757 +- 0.0510 | +0.0762 +- 0.0379 |
| ccomp | 150 | -0.0065 +- 0.0578 | 139 | -0.0134 +- 0.0129 | +0.0097 +- 0.0684 |
| attr | 111 | -0.0739 +- 0.0384 | 87 | +0.0336 +- 0.0218 | -0.1115 +- 0.0550 |
| poss | 119 | +0.0978 +- 0.0596 | 93 | +0.0564 +- 0.0309 | +0.0566 +- 0.0549 |

`results/pair_arc_vs_control.png` draws the arc and control columns of this table, and `results/arc_vs_control_by_type.png` redraws the same rows with each dependency label glossed.

## Regression

The regression section fits interaction on an arc indicator, log2 token distance and a part-of-speech pair block, over all 2455 pairs with 20 regressors and 100 clusters. The reference cell is NOUN-NOUN with 646 pairs, and combinations under 12 pairs are pooled into a single term. The arc coefficient is +0.0023 +- 0.0158, t +0.15, and log2 distance is -0.0130 +- 0.0068, t -1.91. Dropping the part-of-speech block gives arc +0.0066 +- 0.0164 and log2 distance -0.0055 +- 0.0050. The largest part-of-speech terms are PRON-NOUN +0.1528 +- 0.1487, VERB-AUX +0.1077 +- 0.0544, PRON-VERB +0.0627 +- 0.0374 and NOUN-AUX +0.0586 +- 0.0278.

## Permutation null

The permutation section shuffles the arc label 2000 times among the pairs that share a document and a token distance. There are 802 strata, of which 548 hold both labels and so carry information, covering 2150 pairs. The observed arc minus control gap is +0.0074 points, against a null mean of -0.0086 and a null standard deviation of 0.0171, for a two-sided p of 0.3625.

## Size against the singles

From the size section, in FVE points lost.

| set | mean e(a) | mean e(b) | mean sum | mean e(both) | mean interaction | interaction / sum | median abs ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| arc | 0.1906 | 0.1509 | 0.3414 | 0.3585 | -0.0171 | -0.050 | 0.201 |
| control | 0.1623 | 0.2337 | 0.3960 | 0.4205 | -0.0245 | -0.062 | 0.180 |

The interaction is positive on 0.484 of arc pairs and 0.483 of control pairs. Its absolute value clears the floor on 0.571 of arc pairs and 0.514 of control pairs, and exceeds the smaller of the two singles on 0.439 and 0.407.

## Token distance

From the distance section, in FVE points.

| distance | n arc | arc mean | n control | control mean |
| --- | --- | --- | --- | --- |
| 2 | 395 | -0.0239 +- 0.0247 | 203 | +0.0207 +- 0.0195 |
| 3 | 252 | -0.0267 +- 0.0390 | 308 | -0.0251 +- 0.0354 |
| 4 to 5 | 248 | -0.0034 +- 0.0513 | 276 | -0.0396 +- 0.0494 |
| 6 to 9 | 163 | -0.0121 +- 0.0168 | 177 | -0.0099 +- 0.0131 |
| 10 to 17 | 127 | -0.0008 +- 0.0143 | 117 | -0.0262 +- 0.0143 |
| 18+ | 95 | -0.0294 +- 0.0213 | 94 | -0.1008 +- 0.0814 |

`results/pair_interaction_vs_distance.png` draws these bins on a log axis with every pair behind them.

## Control match quality

From the match-quality section, in FVE points.

| quality | n | control mean | arc - control |
| --- | --- | --- | --- |
| exact | 726 | -0.0408 +- 0.0241 | +0.0097 +- 0.0229 |
| dist+-1 | 284 | -0.0060 +- 0.0225 | +0.0311 +- 0.0215 |
| other-doc | 123 | +0.0063 +- 0.0139 | +0.0121 +- 0.0434 |
| other-doc+-1 | 42 | +0.0426 +- 0.0356 | -0.2572 +- 0.2112 |

## Draw-level spread

From the spread section, the within-pair standard deviation of the interaction over draws is 0.1637 mean and 0.0857 median points, so the standard error of one pair's mean at eight draws is about 0.0579 points against the 0.044-point floor. The spread of the pair means themselves is 0.4743 points over 2455 pairs. A per-document table of arc mean, control mean and their difference over the 100 documents closes the report.

## Regenerating run 7

The GPU side is `pair_ablation.py`, which builds the swap pool from the store, samples the arcs, matches each to a control, plans every single and joint edit and round-trips it, runs the reconstructor forwards and writes each variant with its substitutions and its measurements back to the store. It also writes the pair table into the run's `config` and every planned pair into `relations` under the scheme `pair-ablation/arc+control`, with kind `arc:<dep>` or `control`, which is what `pair_analysis.py` reads back.

```bash
python pair_ablation.py --ar "$ASSETS/qwen36-27b_ar-l43-s600_model" \
    --traces ../01_corpus-and-spans/results/ffw_pilot_traces.parquet \
    --db ../db/ffw_span-ablation_database.sqlite \
    --docs all --draws 8 --per-type 150 --batch 8 --precision bf16 \
    --threads 8 --seed 0
```

`--dry-run` does everything except the forward pass, so the pool, the sampling, the control matching, every splice and its round-trip check and the pass budget are all checkable without a GPU.

It needs `en_core_web_sm` installed alongside spaCy, and the assets `ffw_span-ablation_database`, `qwen36-27b_ar-l43-s600_model`, `qwen36-27b_tokenizer` and `spacy_en-core-web-sm_model`. It imports the reconstructor harness and the text-space splicing from `../03_parts-of-speech/`, the swap pool and the substitute machinery from `../04_ablation-strategy/`, the trace normalisation from `../01_corpus-and-spans/`, and the store helpers from `../db/`. Those sibling lookups, and the path the script records for itself in the `runs` table, were pointed at these numbered directories when it was promoted, since the run itself was staged flat on a GPU host, and that is the only change made to it.

The run appends a new run rather than overwriting run 7, so `pair_analysis.py --run <new id>` is what reads it back. Determinism is set before any CUDA initialisation, with `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Run 7 took 1886 s on an NVIDIA H100 80GB HBM3 under torch 2.6.0 with CUDA 12.4, at bf16, measured by the run script from its first document to its last and recorded in the row's config as `wall_time_s`. The run's fingerprint, every argument it was given and the whole pair table are in the `runs` table of the store. That row also records that run 7 was loaded into this store from the GPU host's run 6, with the run id and every variant id reassigned and the span and document ids unchanged.

## Files

```
pair_ablation.py                        the GPU run that wrote run 7
pair_analysis.py                        the tables and two figures, read from the store
test_pair_stub.py                       the whole edit plan on CPU over a throwaway store, no reconstructor
test_pair_analysis_stub.py              the report over a synthetic store with a planted interaction
arc_vs_control_by_type.py               the per-type table redrawn with each dependency label glossed
results/statistics.md                   the printed report, nine sections between the setup header and the figure list
results/pair_arc_vs_control.png         interaction on an arc against its matched control, one row per dep type
results/pair_interaction_vs_distance.png  interaction against token distance, arcs and controls binned separately
results/arc_vs_control_by_type.png      the per-type figure, written by arc_vs_control_by_type.py
```
