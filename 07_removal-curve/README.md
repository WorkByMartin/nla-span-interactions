# 07_removal-curve

Measures where a document's reconstruction goes as its eligible words are removed one at a time, from intact to nothing left. Each document is walked from intact to empty seven times over, under the seven pairings of removal primitive with removal order set out below, and the fraction of variance explained is recorded at every step.

```bash
python removal_analysis.py --run 9
```

No GPU, and no model files. It reads run 9 of the `ffw_span-ablation_database` asset, and run 8 alongside it for the single-span comparison at step one, and writes every number below to `results/statistics.md`, which it prints and rewrites on every run. One figure goes to `results/` beside it.

## The seven curves

Each document contributes seven curves, indexed by the `curve` measurement written beside every variant. The codes and their names are the run's own, in the `curves` field of its `runs.config` row.

**0, random deletion.** Random order, the word and its folded leading space removed, eight permutations.

**1, random swap.** The same eight permutations, the word replaced by a corpus swap, one draw per span per permutation. A span's substitute is a property of the span and the permutation, so the same string sits at that span at every later step of that curve.

**2, front truncation, and 3, back truncation.** Deletion in reading order and in reverse reading order, one curve each.

**4, random filler.** The same permutations again, the word replaced by the filler token `_` repeated to the word's Qwen token count.

**5, front truncation filler, and 6, back truncation filler.** The same two orders as 2 and 3, with the filler in place of deletion.

Eligibility, the swap pool, the splice and the database writes are 05's and 06's, imported rather than restated. A step's variant carries every removed span as a substitution row, with `depth` set to the number of words removed, and the measurements `curve`, `step`, `perm` and `n_words` beside mse, fve, seq_len and dtok. A document of n eligible words costs 1 + n x (3 x perms + 4) reconstructor passes.

Effect is dFVE, a variant's fraction of variance explained minus that document's intact baseline, so a negative number means the reconstruction got worse. Curves are put on a common grid of 21 points in fraction of eligible words removed before they are averaged. In the tables of `results/statistics.md` curves 2, 3, 5 and 6 are headed front deletion, back deletion, front filler and back filler.

## The sample

Run 9 covers 20 documents, listed in the setup header of `results/statistics.md`. Five were named on the command line (240, 621, 1664, 2592, 4126) and 15 drawn uniformly with seed 0 from the 1095 others that pass 06's filters, a CJK fraction at most 0.05 and at least 90 per cent of clean lexical words servable by the swap pool. That is 160 curves for each of the three random-order curves, 20 documents by 8 permutations, and 20 curves for each of the four truncation curves, the counts in the n-curves column of the concavity section. Baseline FVE over the 20 documents is 0.765 mean, 0.136 lowest and 0.913 highest.

## Endpoints

From the endpoints section, mean FVE over the 20 documents with its standard deviation across them.

| state | mean FVE | sd |
| --- | --- | --- |
| intact | 0.765 | 0.164 |
| every eligible word deleted | -0.871 | 0.296 |
| every eligible word swapped, mean over permutations | -1.649 | 0.448 |
| every eligible word replaced by filler | -0.812 | 0.357 |

## The mean curves

The mean-dFVE section gives all seven curves at every 0.05 of the grid, printed at every 0.10. At half the eligible words removed the random-order curves are at -0.387 +- 0.026 for deletion, -0.840 +- 0.059 for swap and -0.668 +- 0.034 for filler. At every word removed they are at -1.636 +- 0.022, -2.414 +- 0.049 and -1.577 +- 0.020. The truncation curves reach the same endpoints as the random-order curve of their primitive, -1.637 +- 0.062 and -1.637 +- 0.061 for front and back deletion, -1.577 +- 0.056 for both fillers.

The primitive-differences section takes those three random-order curves against each other. Deletion minus swap is -0.022 at 0.20 removed, +0.172 at 0.40, +0.892 at 0.60, +1.154 at 0.80 and +0.779 at 1.00. Deletion minus filler is +0.096, +0.242, +0.336, +0.232 and -0.058 at the same fractions, and swap minus filler is +0.117, +0.070, -0.556, -0.921 and -0.837.

## Concavity

The concavity section reports the signed area between a curve and the straight line from its start to its endpoint, in FVE times fraction, with a positive area meaning the curve sits above the chord.

| curve | mean area | se | n curves | share above the chord | mean endpoint dFVE |
| --- | --- | --- | --- | --- | --- |
| random deletion | +0.2436 | 0.0124 | 160 | 0.91 | -1.636 |
| random swap | +0.1139 | 0.0236 | 160 | 0.66 | -2.414 |
| random filler | +0.0400 | 0.0156 | 160 | 0.60 | -1.577 |
| front deletion | +0.5651 | 0.0394 | 20 | 1.00 | -1.637 |
| back deletion | +0.1435 | 0.0324 | 20 | 0.85 | -1.637 |
| front filler | +0.4328 | 0.0301 | 20 | 1.00 | -1.577 |
| back filler | -0.0298 | 0.0359 | 20 | 0.45 | -1.577 |

## Front against back

The front-against-back section takes the two truncation orders at three fractions removed. Under deletion, front minus back is +0.036 +- 0.049 at 0.25, +0.246 +- 0.100 at 0.50 and +1.052 +- 0.143 at 0.75, and the share of documents whose front curve sits below the back one is 0.55, 0.25 and 0.00. Under filler it is +0.062 +- 0.076, +0.577 +- 0.149 and +0.992 +- 0.111, with shares 0.60, 0.25 and 0.00.

## The first steps

Over the first five steps of the random filler curve the mean dFVE is -0.0021 +- 0.0010, -0.0029 +- 0.0011, -0.0039 +- 0.0012, -0.0057 +- 0.0015 and -0.0081 +- 0.0023, over 160 curves at each step. That is the first-five-steps section.

At the first removal, from the first-removal section over the same 160 curves per primitive, random deletion is -0.0029 mean and 0.0040 mean absolute, random swap -0.0015 and 0.0031, random filler -0.0021 and 0.0036.

## Agreement with the single-span run

76 of the step-one swap variants repeat a run 8 single at the same span with the same substitute. Their correlation is 0.997, the mean absolute difference 0.00022 and the largest 0.00210. That is the step-1 section.

## Whole against sum of parts

The last section puts each document's drop from intact to every eligible word deleted beside the sum of its own first-step deletion effects, the mean step-1 dFVE over that document's random deletion curves times its number of eligible words. The drop runs from -1.086 for document 1929 to -2.248 for document 240. The summed first steps run from -2.904 for document 358 to +0.038 for document 2815, and are positive for five of the 20 documents.

## Regenerating run 9

The GPU side is `removal_curve.py`, which selects the documents, plans every ordering and every string, runs the reconstructor forwards along each curve and writes each step's variant with its substitutions and its measurements back to the store.

```bash
python removal_curve.py --ar "$ASSETS/qwen36-27b_ar-l43-s600_model" \
    --traces ../01_corpus-and-spans/results/ffw_pilot_traces.parquet \
    --db ../db/ffw_span-ablation_database.sqlite \
    --with-docs 240,621,1664,2592,4126 --n-docs 15 --perms 8 --filler _ \
    --batch 16 --precision bf16 --threads 8 --seed 0
```

It needs `en_core_web_sm` installed alongside spaCy, and the assets `ffw_span-ablation_database`, `qwen36-27b_ar-l43-s600_model`, `qwen36-27b_tokenizer` and `spacy_en-core-web-sm_model`. It imports `tree_vs_linear` from `../06_tree-vs-linear` and through it the pair helpers from `../05_dependent-pair`, the text-space splicing from `../03_parts-of-speech`, the swap and deletion rules from `../04_ablation-strategy` and the store helpers from `../db`. The one change made to it since it ran is the directory it looks in for `tree_vs_linear.py`, which was 06's scratch workbench and is now `../06_tree-vs-linear`.

`--dry-run` plans and checks every string without a model and writes `results/plan.md`. The script appends a new run rather than overwriting run 9, so `removal_analysis.py --run <new id>` is what reads it back, and `--resume-run` continues an existing removal-curve run over the documents it has not measured.

Run 9 ran on an NVIDIA H100 80GB HBM3 with 132 SMs, under torch 2.6.0+cu124 with CUDA 12.4, at bf16, with `CUBLAS_WORKSPACE_CONFIG=:4096:8`. It took 2437 s from its first document to its last, measured by the run script and recorded in the row's config as `wall_time_s`. The run's fingerprint and every argument it was given are in the `runs` table of the store under this script's name.

## Files

```
removal_curve.py            the GPU run that wrote run 9
removal_analysis.py         the tables and the figure, read from the store
results/statistics.md       the printed report, nine sections between the setup header and the figure list
results/removal_curves.png  FVE against fraction removed under random order, one mean curve per primitive with a 95% confidence band over the 20 documents
results/truncation_front_vs_back.png  FVE against fraction removed under deletion, front and back truncation, one faint line per document and the mean in bold
results/removal_first_steps.png  the first ten words removed, in FVE points lost, same encoding
```
