# 04_ablation-strategy

Measures the same word-level ablation three more ways, so the masked-LM marginalisation of 03 can be read against them. A word is replaced by a matched word taken from a different document, or deleted outright, and separately every lexical word in a document is permuted among the document's own slots.

```bash
python swap_analysis.py --run 5
```

No GPU, and no model files. It reads run 5 of the `ffw_span-ablation_database` asset, and run 3 alongside it for the masked-LM comparison, and writes every number below to `results/statistics.md`, which it prints and rewrites on every run. Three figures go to `results/` beside it, and `effect_by_word_class.py` draws a fourth from the per-class table, checking every cell of it against `results/statistics.md` first.

## The three arms

**Corpus swap.** The substitute is drawn uniformly over the lexical span occurrences of other documents' verbalisations, matched on spaCy coarse part of speech, Qwen token count and leading-space parity. The original word is excluded case-insensitively and the substitute is recased to the original's shape. The pool holds 12854 lexical occurrences, 3338 word types, over 100 documents, and a target span's own document is excluded from it. Nothing about the sentence being edited enters the choice.

**Deletion.** The word and its folded leading space are removed, one variant per span.

**Shuffle.** Every lexical word of the document is permuted among the document's own slots, four permutations each, on 20 of the 40 documents drawn one per stratum of baseline reconstruction quality. Punctuation is untouched and space parity is restored per destination slot, so the bag of words is preserved exactly and the order is destroyed.

The baseline is the intact document, batched in the same call as the arms it is differenced against.

The unit, the documents and the spans are those of run 3, so the comparison is span for span. Effect is FVE lost in points, 100 times the drop in fraction of variance explained against the same document's unedited baseline, and a positive number means the reconstruction got worse. Standard errors are clustered on document over 40 clusters. A per-draw absolute MSE change below 2e-4 is 0.044 FVE points and is treated as the harness reproduction floor.

## The sample

Run 5 covers 40 documents, 40 spans each, 16 swap draws per span, one deletion variant per span, and 4 shuffles on 20 documents. That is 25568 swap draws and 1598 deletion variants. Two of the 1600 target spans had no eligible pool word and so carry no swap draws, one in document 1379 and one in document 4436, which appear in the per-document table with 624 swap draws rather than 640. Every arm, deletion included, is reported over the 1598 spans the swap covers. The masked-LM comparison is 76704 draws over the same spans. These counts are in the setup header of `results/statistics.md`.

Run 4 in the same database is a five-document pilot of this design. It is not reported here.

## Leakage

The masked-LM arm puts the original word back on 33963 of its 76704 draws, 44.28% exactly and 45.08% case-blind. The swap arm excludes the original by construction and does so on 0 of 25568 draws. This is the leakage section.

## Effect size

Pooled over every span, in FVE points with a document-clustered standard error, from the pooled effect-size section:

| arm | draws | mean absolute | median absolute | signed mean |
| --- | --- | --- | --- | --- |
| corpus swap | 25568 | 0.312 | 0.125 | +0.119 +- 0.020 |
| masked-LM, all draws | 76704 | 0.148 | 0.017 | +0.030 +- 0.011 |
| masked-LM, non-identical draws | 42126 | 0.266 | 0.115 | +0.055 +- 0.020 |
| deletion | 1598 | 0.382 | 0.106 | +0.178 +- 0.052 |

The share of draws whose absolute effect clears the 0.044-point floor is 0.773 for the swap, 0.420 for the masked-LM arm over all draws, 0.753 for its non-identical draws, and 0.745 for deletion. That is the floor section.

## Open against closed class

In the open-against-closed section, the swap is close to flat across the open and closed halves, at 0.302 and 0.321 mean absolute points with signed means of +0.086 +- 0.020 and +0.149 +- 0.029. The masked-LM arm is not, at 0.217 open against 0.088 closed, with 0.587 of open-class draws over the floor against 0.275 of closed-class draws.

The ratio of swap mean absolute effect to masked-LM mean absolute effect, per class, is largest in the closed classes, at SCONJ 5.59, AUX 5.19, DET 5.14, PRON 4.69, CCONJ 3.67, ADP 3.11, PART 2.24. The open classes sit between 1.10 and 1.67, PROPN 1.10, NOUN 1.35, ADJ 1.36, ADV 1.59, VERB 1.67. NUM is 1.99 and INTJ, over four spans, is 0.47.

Per-class signed means with clustered intervals are in the per-class signed-effect section. The largest are SCONJ +0.326 +- 0.094, NUM +0.287 +- 0.159, PRON +0.230 +- 0.115, AUX +0.195 +- 0.097, PART +0.163 +- 0.065 and DET +0.155 +- 0.065.

## Variance and where a pass budget goes

The variance-decomposition section splits a draw's effect into a span mean plus draw noise. Over all 1598 spans the swap's between-span variance is 0.4076 and its within-span variance 0.2168, an intraclass correlation of 0.653. Split by half, the closed class carries more of its spread between spans, 0.5288 against 0.2052 for an ICC of 0.720, and the open class less, 0.2670 against 0.2301 for an ICC of 0.537. The masked-LM arm pooled over its six depths is 0.2132 between and 0.1010 within, ICC 0.679, with the depth effect folded into its within term.

With a fixed budget of N reconstructor passes split as spans by draws, the variance of a class-level mean is (between x d + within) / N, which rises with d whenever the between-span variance is positive. The budget section reports that variance relative to one draw per span. The optimal draw count for a class-level mean is 1 in every row of the table. The 16 draws this run took cost 10.8 times the variance of the same budget spent one draw per span, so the 25568 passes here are worth about 2369 spans at one draw each for a class mean. The per-class table repeats the arithmetic inside each word class and the best draw count is 1 there too.

Per-span means are the other question, and there the draws are what buys the precision. From the precision section, the standard error of one span's own swap mean is 0.086 mean and 0.040 median at 4 draws, 0.066 and 0.033 at 8, 0.055 and 0.028 at 12, and 0.049 and 0.025 at 16, which is 1.1 times the floor.

## Rank agreement

Pooled over the 1598 spans, the Spearman correlation of per-span mean effect is 0.565 between the swap and the masked-LM arm, 0.674 between the swap and deletion, and 0.517 between the masked-LM arm and deletion. The swap's own split half, draws 0 to 7 against draws 8 to 15 of the same spans, is 0.916, which is the ceiling any other correlation with the swap could reach at this draw count. The rank-agreement section gives the same four columns per document.

## Deletion against swap

Paired within the span, deletion is larger than the swap by +0.059 +- 0.044 points signed and by +0.112 +- 0.040 points in absolute value, a ratio of 1.41 over all 1598 spans. The gap is carried by particular classes. NUM is 3.46, AUX 2.21, CCONJ 1.77, PROPN 1.52 and PRON 1.51, while ADJ and ADP are 0.98, DET 0.85 and PART 0.83. The per-span Spearman correlation between the two is 0.674. This is the deletion-against-swap section.

## Shuffle

On the 20 documents that got it, permuting the document's own words among its own slots costs between 31.476 and 161.994 FVE points, document 3914 lowest and document 1925 highest. The section prints the per-document mean and standard deviation over the four permutations, and beside it the sum of that document's 40 single-word deletion effects, which is a different quantity because it covers 40 spans rather than all of them and ignores interaction.

## Regenerating run 5

The GPU side is `swap_ablation.py`, which builds the pool from the store, plans every edit, runs the reconstructor forwards and writes each variant with its substitution and its measurements back to the store.

```bash
python swap_ablation.py --ar "$ASSETS/qwen36-27b_ar-l43-s600_model" \
    --traces ../01_corpus-and-spans/results/ffw_pilot_traces.parquet \
    --db ../db/ffw_span-ablation_database.sqlite \
    --source-run 3 --docs all --draws 16 --shuffles 4 --shuffle-docs 20 \
    --batch 8 --precision bf16 --threads 8 --seed 0
```

It needs `en_core_web_sm` installed alongside spaCy, and the assets `ffw_span-ablation_database`, `qwen36-27b_ar-l43-s600_model`, `qwen36-27b_tokenizer` and `spacy_en-core-web-sm_model`. It imports the reconstructor harness and the text-space splicing from `../03_parts-of-speech/`. It appends a new run rather than overwriting run 5, so `swap_analysis.py --run <new id>` is what reads it back.

Determinism is set before any CUDA initialisation, as in 03: `CUBLAS_WORKSPACE_CONFIG=:4096:8`, TF32 off, `use_deterministic_algorithms` with `warn_only=False`, and eager attention. Run 5 took 2169 s on an A100 80GB PCIe under torch 2.6.0 with CUDA 12.4, at bf16. The run's fingerprint and every argument it was given are in the `runs` table of the store under this script's name.

## Files

```
swap_ablation.py                   the GPU run that wrote run 5
swap_analysis.py                   the tables and the three figures, read from the store
test_swap_stub.py                  the whole edit plan on CPU over a throwaway store, no reconstructor
test_swap_analysis_stub.py         the report over a synthetic store with a planted span effect
effect_by_word_class.py            the per-class signed means recomputed, checked against statistics.md and drawn
results/statistics.md              the printed report, fourteen sections between the setup header and the figure list
results/swap_se_vs_draws.png       standard error of a per-span swap mean against draw count
results/swap_vs_mlm_scatter.png    per-span mean effect, swap against masked-LM, symmetric log axes
results/budget_draws_vs_spans.png  variance of a class-level mean as a fixed pass budget moves onto draws
results/effect_by_word_class.png   signed mean effect by word class, swap and deletion, clustered intervals
```
