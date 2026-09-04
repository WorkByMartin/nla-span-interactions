# 06_tree-vs-linear

Measures every cell of the pairwise interaction matrix over the words of a document, so the interaction between two words can be read against the dependency parse and against the linear distance between them over the whole matrix rather than over a sample of it. The interaction is `e(a) + e(b) - e(both)` in FVE points, where `e` is the drop in fraction of variance explained against the same document's unedited baseline, times 100. A positive interaction means the pair costs less than the sum of its two singles.

```bash
python tree_vs_linear_analysis.py --run 8
```

No GPU, and no model files. It reads run 8 of the `ffw_span-ablation_database` asset and writes every number below to `results/statistics.md`, which it prints and rewrites on every run. Five figures go to `results/` beside it.

## What is ablated

The edit, the eligibility test, the pool, the splice and the common random numbers are 05's, imported from `pair_ablation` rather than restated. A word is replaced by a matched word taken from a different document. For each selected document every eligible word is swapped alone and every unordered pair of eligible words is swapped together, at eight draws. A span's substitute at draw k is a property of the span and the draw, so the string spliced at that span is identical in its single edit and in all n - 1 joint edits containing it. Adjacent pairs are measured and nothing is filtered at run time.

A pair is on an ARC when the store holds a spaCy head relation between its two words in either direction, ADJACENT when their token indices differ by one and there is no arc, and OTHER otherwise. The three partition the matrix. Token distance is the difference of the two token indices. Tree path length is the number of dependency edges between the two words, over every token of the document and not only the editable ones, so an arc is path length one. These definitions and the sign convention are in the setup header of `results/statistics.md`.

## The sample

Run 8 covers 5 documents, 240, 621, 1664, 2592 and 4126, drawn uniformly at random with seed 0 from the 1100 documents whose CJK fraction is at most 0.05 and at least 90 per cent of whose clean lexical words the swap pool can serve. That is 41392 measured pairs at eight draws each, 5 baselines, 5160 singles read and 331136 joint variants. The parse is `spacy-en_core_web_sm-3.8.0` with 180257 head relations, and all 41392 pairs carry a category the run recorded under `tree-vs-linear/all-pairs`. 4536 pairs had at least one draw whose splice moved the prompt length. Tree path length 1 covers 623 pairs, 2 covers 1087, 3 covers 1421, 4 covers 1658, 5 covers 1608, 6 covers 1398, 7 covers 1133 and 8 covers 861, and 29666 pairs are left disconnected by the parse. The harness floor is 0.044 FVE points. These counts are in the setup header.

## Reliability

Even draws against odd draws give two independent estimates of each pair's interaction, and their covariance is the variance of the true pair means. Pooled over the 41392 pairs the observed variance is 0.37992, of which 0.37722 is reliable and 0.00269 is draw noise, a reliability of 0.9929. Per document the reliability is 0.5765 for 240, 0.9951 for 621, 0.9687 for 1664, 0.9960 for 2592 and 0.9993 for 4126. That figure is the largest R squared any model of these pair means could reach, and every R squared below is reported raw and divided by it. This is the split-half section.

## Tree path length against token distance

The registered question. Each of the two is fitted as a saturated categorical block on its own, neither given the other's columns, and each is divided by the split-half ceiling. From the primary-comparison section, pooled with document dummies:

| set | pairs | ceiling | R2 distance | R2 tree | tree minus distance, of reliable |
| --- | --- | --- | --- | --- | --- |
| tree path length at most 2 | 1710 | 0.9668 | 0.0166 | 0.0012 | -0.0159 |
| tree path length over 2 | 39682 | 0.9940 | 0.0005 | 0.0000 | -0.0004 |
| all pairs | 41392 | 0.9929 | 0.0007 | 0.0001 | -0.0006 |

The permutation null shuffles path length, arc status and dep label together among the pairs that share a document and a token distance, at token distance 3 and beyond, which is 40441 of the 41392 pairs. On the difference in R squared as a fraction of reliable variance the excess over the null is +0.0005 at path length at most 2, with z +0.08 and one-sided p 0.5389, and -0.0001 both for path length over 2, z -1.97 and p 0.9980, and for all pairs, z -1.86 and p 0.9920.

## Distance first, then the tree

Nested least squares on the pair means, with token distance entering saturated at one column per exact distance up to 30 and one beyond, then the tree block of path length, then the dep type of the arc and the ordered POS pair. Pooled over 41392 pairs against a ceiling of 0.9929, R squared is 0.0007 for distance, 0.0000 once the tree is added and 0.0044 once the labels are added too. On the same pooled model the document dummies alone reach 0.0033, with distance 0.0040, with the tree 0.0041 and with the labels 0.0084, so the labels add 0.0044 over the tree alone. Per document the labelled model reaches 0.0989 on 240, 0.0049 on 621, 0.0431 on 1664, 0.0150 on 2592 and 0.0020 on 4126.

The permutation null on the tree increment uses the same shuffle over 852 strata, 350 of them informative and covering 25593 pairs, at 500 permutations. The observed increment is 0.00008 and the null mean is 0.00008 with a standard deviation of 0.00004, an excess of -0.00000, z -0.05 and one-sided p 0.5250.

## Per-cell interaction by category

From the per-cell section, with standard errors clustered on document over 5 clusters:

| category | cells | share | mean absolute | signed mean | median absolute | over floor |
| --- | --- | --- | --- | --- | --- | --- |
| arc | 623 | 0.0151 | 0.1156 +- 0.0154 | +0.0194 +- 0.0197 | 0.0552 | 0.583 |
| adjacent | 201 | 0.0049 | 0.1321 +- 0.0295 | +0.0600 +- 0.0454 | 0.0751 | 0.692 |
| other | 40568 | 0.9801 | 0.0661 +- 0.0085 | -0.0284 +- 0.0179 | 0.0350 | 0.396 |
| all | 41392 | 1.0000 | 0.0671 +- 0.0084 | -0.0273 +- 0.0179 | 0.0353 | 0.400 |

The ratio of arc to other per-cell mean absolute interaction is 1.750 pooled, and per document 1.456 on 240, 1.556 on 621, 3.193 on 1664, 1.525 on 2592 and 1.316 on 4126.

Over the whole matrix the total absolute interaction is 2778.59 points. Arcs hold 0.0259 of that mass on 0.0151 of the cells, adjacent pairs 0.0096 on 0.0049, and everything else 0.9645 on 0.9801. This is the mass-share section.

The same shuffle applied to the per-cell arc mean, restricted to token distance 3 and beyond over 852 strata of which 74 are informative, covering 6297 pairs and 207 arcs, gives an observed arc cell mean of 0.0714 points against a null mean of 0.0746 with standard deviation 0.0131. The excess is -0.0033 points, -4.4 per cent, z -0.25, one-sided p 0.5107 and two-sided p 0.8155. Per document the excess is -0.0007 on 240, +0.0065 on 621, +0.0318 on 1664, -0.0503 on 2592 and -0.0040 on 4126.

## Against distance and against path length

Mean absolute interaction falls with token distance over the short bins and then flattens, at 0.1292 points at distance 1 over 476 pairs, 0.1106 at 2 over 475, 0.0795 at 3 over 473, 0.0854 over 4 to 5, 0.0768 over 6 to 9, 0.0566 over 10 to 17, 0.0580 over 18 to 33 and 0.0673 over 27251 pairs at 34 and beyond.

Against tree path length it is 0.1156 at path 1 over 623 pairs, 0.1152 at 2 over 1087, 0.0778 at 3, 0.0756 at 4, 0.0615 at 5, 0.0678 at 6, 0.1017 at 7, 0.0559 at 8, and between 0.0336 and 0.0597 from 9 to 20. The 29666 disconnected pairs sit at 0.0637. The median token distance of a pair rises with path length, from 2.0 at path 1 to 26.0 at path 8 and 71.0 for the disconnected pairs. Both tables are in `results/statistics.md`.

## Dep types, against 05

The consistency-check section reports the same quantity on the same scale as 05, arc by arc, over every arc of these documents. The mean interaction by dep type, in points, is amod +0.0554 +- 0.0273 over 79 arcs, compound -0.0076 +- 0.0144 over 74, prep +0.0411 +- 0.0205 over 62, det +0.0205 +- 0.0231 over 61, pobj +0.0205 +- 0.0211 over 52, conj -0.0348 +- 0.0346 over 43, dobj +0.0310 +- 0.0244 over 42, nsubj -0.0414 +- 0.0313 over 34 and advmod +0.1477 +- 0.1117 over 23. The remaining types carry between 4 and 22 arcs each. Restricted to 05's nine dep types with the two words at least two tokens apart, which is 05's own arc set, 157 arcs give a mean of -0.0142 +- 0.0245 points. 437 of the 623 arcs carry a dep type 05 did not sample.

## What the draws separate from zero

16548 of the 41392 pairs, a fraction of 0.3998, have an absolute mean at least two standard errors from zero, at a median standard error of 0.0210 points. Per document the fraction is 0.3420 on 240, 0.4112 on 621, 0.3269 on 1664, 0.2887 on 2592 and 0.6093 on 4126. Over the cells that survive that test the per-cell means are arc 0.2029, adjacent 0.2223 and other 0.1231, and arcs are 0.0160 of them.

## Figures

Each figure is one document's interaction matrix in reading order, symmetric by construction, with the spaCy head arcs over the same words drawn above it. `interaction_matrix_doc240.png` has 124 words and 7626 cells, 2608 of them outside two standard errors, 120 head arcs and a colour limit of +-0.288 points. `interaction_matrix_doc621.png` has 132 words, 8646 cells, 3555 outside, 128 arcs, limit +-0.562. `interaction_matrix_doc1664.png` has 138 words, 9453 cells, 3090 outside, 132 arcs, limit +-0.401. `interaction_matrix_doc2592.png` has 119 words, 7021 cells, 2027 outside, 115 arcs, limit +-0.394. `interaction_matrix_doc4126.png` has 132 words, 8646 cells, 5268 outside, 128 arcs, limit +-0.110.

`results/three_pairs_one_sentence.png` is written by the worked example in `examples/`, not by the analysis.

## Regenerating run 8

The GPU side is `tree_vs_linear.py`, which selects the documents, plans every single and every joint edit, checks each splice round-trips, runs the reconstructor forwards and writes each variant with its measurements back to the store.

```bash
python tree_vs_linear.py --ar "$ASSETS/qwen36-27b_ar-l43-s600_model" \
    --traces ../01_corpus-and-spans/results/ffw_pilot_traces.parquet \
    --db ../db/ffw_span-ablation_database.sqlite \
    --n-docs 5 --seed 0 --draws 8 --batch 16 --precision bf16 --threads 8
```

It needs `en_core_web_sm` installed alongside spaCy, and the assets `ffw_span-ablation_database`, `qwen36-27b_ar-l43-s600_model`, `qwen36-27b_tokenizer` and `spacy_en-core-web-sm_model`. It imports `pair_ablation` from `../05_dependent-pair`, which in turn puts the reconstructor harness and the text-space splicing from `../03_parts-of-speech`, the swap pool from `../04_ablation-strategy`, and the store bindings from `../db` on the path. Promotion out of the scratch workbench changed the sibling directory these path lists point at, and nothing else in the run script.

Determinism is set before any CUDA initialisation, as in 03 and 05. Run 8 took 11262 s on an NVIDIA H100 80GB HBM3 under torch 2.6.0 with CUDA 12.4, at bf16, with `CUBLAS_WORKSPACE_CONFIG=:4096:8`, measured by the run script from its first document to its last and recorded in the row's config as `wall_time_s`. It was measured on a pod as that machine's run 7 and loaded into this database as run 8, so both the run id and every variant id were reassigned on load while span and document ids were carried across unchanged. The setup header therefore names the script by the path it had when it ran. Every argument the run was given is in the `runs` table of the store.

## Files

```
tree_vs_linear.py                          the GPU run that wrote run 8
tree_vs_linear_analysis.py                 the tables and the five figures, read from the store
test_tree_vs_linear_stub.py                the whole edit plan on CPU over a throwaway store, no reconstructor
test_tree_vs_linear_analysis_stub.py       the report over a synthetic store with a planted on-arc interaction
examples/three_pairs_one_sentence.py       three pairs from one sentence of document 1664, with their interactions
examples/three_pairs_one_sentence.md       what that example prints
examples/three_pairs_one_sentence.png      its figure, copied beside the markdown
results/statistics.md                      the printed report, eleven sections between the setup header and the figure list
results/interaction_matrix_doc240.png      document 240's interaction matrix, head arcs above it
results/interaction_matrix_doc621.png      document 621's interaction matrix, head arcs above it
results/interaction_matrix_doc1664.png     document 1664's interaction matrix, head arcs above it
results/interaction_matrix_doc2592.png     document 2592's interaction matrix, head arcs above it
results/interaction_matrix_doc4126.png     document 4126's interaction matrix, head arcs above it
results/three_pairs_one_sentence.png       the worked example's figure
```
