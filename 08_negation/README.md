# 08_negation

Measures what turning a negation round costs the reconstruction, against controls that edit the same words without touching polarity. Every negator spaCy finds in a verbalised document is flipped to its affirmative form, and the same instance is also measured with the negator deleted, with the word it governs deleted, and with that governed word corpus-swapped. Documents with no negator at all get one inserted instead.

```bash
python negation_analysis.py --db ../db/ffw_span-ablation_database.sqlite --run 10
```

No GPU, and no model files. It reads run 10 of the `ffw_span-ablation_database` asset and writes every number below to `results/statistics.md`, which it prints and rewrites on every run. One figure goes to `results/` beside it.

## The six conditions

Each condition is one forward pass of the reconstructor on an edited verbalisation, scored against the document's own gold activation, and the document's intact string is the baseline in the same batch. The codes and their names are the run's own, in the `conditions` field of its `runs.config` row, and every variant carries `condition`, `instance`, `ntype` and `in_quote` as measurements beside mse, fve, seq_len and dtok.

**0, flip.** `not` and `n't` are deleted so the clause turns affirmative, with the contraction stem restored where deleting `n't` leaves one (`ca` back to `can`, `wo` to `will`, `sha` to `shall`). `never` becomes `always`, `without` becomes `with`, and the determiner `no` becomes `a` before a singular noun and is deleted otherwise.

**1, del_neg.** The negator is deleted, run only where the flip is not itself a deletion, so it covers `never`, `without` and the `no` instances whose flip was `a`.

**2, del_gov.** The governed word is deleted and the negator is left in place.

**3, swap_gov.** The governed word is replaced by a corpus swap, four draws, and the negator is left in place. The pool, the part-of-speech and token-length matching and the splice are 04's and 05's, imported rather than restated.

**4, ins_not, and 5, ins_ctrl.** In a document with no negator, ` not` is inserted after one auxiliary, against ` just` inserted after the same auxiliary as the presence control.

The governed word is the word the negator bears on, taken off the dependency parse. For a determiner it is the head, for `without` it is the object of the preposition, and for a negator under an auxiliary it is the attribute, complement, prepositional phrase or adverb to the auxiliary's right, falling back to the auxiliary's own verbal head.

Effect is dFVE, a variant's fraction of variance explained minus that document's intact baseline, so a negative number means the reconstruction got worse. Every table works from one value per instance per condition, with the four swap draws averaged first. `ntype` records which negator it was (0 not, 1 n't, 2 no, 3 without, 4 never, 5 insertion) and `in_quote` records whether the negator sits inside a quoted stretch of the verbalisation.

## The sample

Run 10 covers 370 documents with a baseline FVE mean of 0.762, and 509 instances, 309 negators and 200 insertions. Those counts are in the setup header of `results/statistics.md`. By type the negators are 140 `not`, 96 `n't`, 44 `no`, 15 `never` and 14 `without`, from the n column of the per-type section. Every negator instance has a flip and a del_gov variant, 61 have a del_neg variant, and 308 of the 309 have swap_gov draws, the counts in the condition section.

## Mean effect by condition

From the condition section, one value per instance with the swap draws averaged, with a 95 per cent bootstrap interval over instances.

| condition | mean dFVE | 95% bootstrap | n | mean absolute dFVE |
| --- | --- | --- | --- | --- |
| flip | -0.0049 | [-0.0071, -0.0029] | 309 | 0.0077 |
| del_neg | -0.0007 | [-0.0023, +0.0008] | 61 | 0.0035 |
| del_gov | -0.0085 | [-0.0137, -0.0042] | 309 | 0.0115 |
| swap_gov | -0.0040 | [-0.0062, -0.0022] | 308 | 0.0067 |
| ins_not | -0.0016 | [-0.0024, -0.0009] | 200 | 0.0029 |
| ins_ctrl | -0.0006 | [-0.0011, -0.0002] | 200 | 0.0015 |

## Paired contrasts

The paired-contrast section differences the conditions within the instance, over the negator instances.

| contrast | signed difference | share > 0 | absolute difference | n |
| --- | --- | --- | --- | --- |
| flip minus swap_gov | -0.0009 [-0.0036, +0.0018] | 0.46 | +0.0010 [-0.0014, +0.0036] | 308 |
| flip minus del_gov | +0.0036 [-0.0008, +0.0088] | 0.50 | -0.0038 [-0.0090, +0.0004] | 309 |
| flip minus del_neg | -0.0001 [-0.0008, +0.0006] | 0.48 | -0.0003 [-0.0010, +0.0003] | 61 |
| del_gov minus swap_gov | -0.0045 [-0.0080, -0.0015] | 0.48 | +0.0048 [+0.0019, +0.0084] | 308 |

## By negator type

The per-type section gives each condition's mean with its standard error, and the flip minus swap_gov contrast paired within the instance.

| type | n | flip | del_neg | del_gov | swap_gov | flip minus swap_gov |
| --- | --- | --- | --- | --- | --- | --- |
| not | 140 | -0.0063 +- 0.0018 | | -0.0063 +- 0.0024 | -0.0030 +- 0.0009 | -0.0034 [-0.0071, -0.0003] n 139 |
| n't | 96 | -0.0059 +- 0.0023 | | -0.0167 +- 0.0070 | -0.0073 +- 0.0030 | +0.0014 [-0.0058, +0.0086] n 96 |
| no | 44 | -0.0003 +- 0.0009 | +0.0006 +- 0.0009 | -0.0021 +- 0.0006 | -0.0020 +- 0.0007 | +0.0017 [-0.0004, +0.0045] n 44 |
| without | 14 | -0.0010 +- 0.0007 | -0.0017 +- 0.0014 | -0.0011 +- 0.0010 | -0.0003 +- 0.0004 | -0.0007 [-0.0019, +0.0006] n 14 |
| never | 15 | -0.0024 +- 0.0018 | -0.0026 +- 0.0020 | -0.0032 +- 0.0022 | -0.0026 +- 0.0021 | +0.0002 [-0.0016, +0.0019] n 15 |

`not` and `n't` carry no del_neg cell because for those two the flip is the deletion.

## Inside and outside a quoted stretch

From the in-quote section, 244 of the 308 negator instances with a swap sit inside a quoted stretch and 64 outside. Inside, flip is -0.0050 and swap_gov -0.0042, with flip minus swap_gov at -0.0008 [-0.0041, +0.0025]. Outside, flip is -0.0045 and swap_gov -0.0034, with flip minus swap_gov at -0.0012 [-0.0047, +0.0023].

## Insertion

Over the 200 documents that had no negator, from the insertion section, ins_not is -0.0016 [-0.0024, -0.0009] and ins_ctrl is -0.0006 [-0.0011, -0.0002]. Paired within the document, ins_not minus ins_ctrl is -0.0010 [-0.0016, -0.0005] with 0.44 of the pairs above zero, and in absolute value the difference is +0.0014 [+0.0010, +0.0019].

## Scale

The scale section reports the spread of the flip effect on its own. Per instance, the median absolute flip dFVE is 0.0022 and the 90th percentile is 0.0184. No noise floor from an earlier run is recomputed here.

## The largest gaps

The last section lists the twelve instances with the largest absolute flip minus swap_gov, each with its document, its negator type, both effects and forty characters of the verbalisation either side of the negator. The extremes are instance 138 in document 1925, an `n't` at flip -0.003 against swap_gov -0.206, and instance 43 in document 673, an `n't` at flip -0.180 against swap_gov +0.017. The instance, its document and the surrounding text are read back out of the store, from the flip variant's substitution rows and the span offsets they point at, so the section needs nothing beside the database.

## Regenerating run 10

The GPU side is `negation.py`, which parses every verbalisation, finds the negators, plans every edit, runs the reconstructor forwards and writes each variant with its substitutions and its measurements back to the store.

```bash
python negation.py --ar "$ASSETS/qwen36-27b_ar-l43-s600_model" \
    --traces ../01_corpus-and-spans/results/ffw_pilot_traces.parquet \
    --db ../db/ffw_span-ablation_database.sqlite \
    --draws 4 --max-insertions 200 \
    --batch 16 --precision bf16 --threads 8 --seed 0
```

It needs `en_core_web_sm` installed alongside spaCy, and the assets `ffw_span-ablation_database`, `qwen36-27b_ar-l43-s600_model`, `qwen36-27b_tokenizer` and `spacy_en-core-web-sm_model`. It imports `tree_vs_linear` from `../06_tree-vs-linear` and through it the pair helpers from `../05_dependent-pair`, the text-space splicing from `../03_parts-of-speech`, the swap and deletion rules from `../04_ablation-strategy` and the store helpers from `../db`. Three changes have been made to it since it ran, the directory it looks in for `tree_vs_linear.py`, which was 06's scratch workbench and is now `../06_tree-vs-linear`, the script path it records in the `runs` table, and the removal of a side-car `instances.json` the analysis no longer reads.

`--dry-run` plans and checks every string without a model and writes `results/plan.md`. The script appends a new run rather than overwriting run 10, so `negation_analysis.py --run <new id>` is what reads it back, and `--resume-run` continues an existing negation run over the documents it has not measured.

Run 10 ran on an NVIDIA H100 80GB HBM3 with 132 SMs, under torch 2.6.0+cu124 with CUDA 12.4, at bf16, with `CUBLAS_WORKSPACE_CONFIG=:4096:8`. It took 166 s from its first document to its last, measured by the run script and recorded in the row's config as `wall_time_s`. It was measured on a pod and copied into this database, and because the local file had gained further verbalisations under its own run 10 while it ran, the run id and every variant id were reassigned on the copy, with span and document ids unchanged. The run's fingerprint and every argument it was given are in the `runs` table of the store under this script's name.

## Files

```
negation.py               the GPU run that wrote run 10
negation_analysis.py      the tables and the figure, read from the store
results/statistics.md     the printed report, seven sections between the setup header and the figure list
results/negation_by_condition.png   mean FVE points lost per condition with 95% bootstrap intervals
results/negation_flip_vs_swap.png   per-instance flip against governed-word swap, with the correlation
```
