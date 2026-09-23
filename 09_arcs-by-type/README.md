# 09_arcs-by-type

Asks whether 05's null holds for each dependency relation taken separately: do two words joined by an arc of a given Universal Dependencies relation interact more, when both are deleted, than a matched pair of words with no arc between them? The quantity is the interaction e(a) + e(b) - e(both) in FVE points, where e is the drop in fraction of variance explained against the same document's unedited baseline, times 100. A positive interaction means the pair costs less than the sum of its two single deletions. Per relation, the reported quantity is arc minus matched control.

```bash
python parse_stanza.py --db ../db/ffw_span-ablation_database.sqlite --stanza-dir <stanza_en-combined_model>
python arc_deletion.py --ar <qwen36-27b_ar-l43-s600_model> --db ../db/ffw_span-ablation_database.sqlite --per-type 450 --seed 0
python arc_analysis.py --db ../db/ffw_span-ablation_database.sqlite --run 11
```

`parse_stanza.py` parses every verbalisation with `stanza_en-combined_model` and writes the parse into the store under the scheme `stanza-en_combined_charlm-1.14.0`. `arc_deletion.py` plans the pairs, deletes each word singly and both together, and records every reconstructor pass as run rows in the store; it needs a GPU, and `--dry-run` plans and round-trips every string without one. `arc_analysis.py` needs no GPU: it reads run 11 from the `ffw_span-ablation_database` asset and writes every number below to `results/statistics.md`.

## What is ablated

Deletion under 04's rule, applied right to left by 07's `delete_many`, so a word and its folded leading space go and no doubled space is left. Deletion is deterministic, so each span contributes one single however many pairs use it.

An arc is a `head` relation in the Stanza parse whose dependent carries one of 26 relations. Adjacent words are admitted (token distance at least 1, against 05's floor of 2). Each arc is matched to a control pair in the same document with no arc, sibling or grandparent link between the two words, the same token distance, the same Qwen token count per word, the same in-quote status per word and a mean position within 20% of the arc's. Where no exact match exists the criteria relax in order: distance by one token either side, then position, then quote status, then token count. Part of speech is not matched; it enters the analysis as a covariate.

## The sample, and what is not reproducible

Run 11's arcs were not every Stanza arc. They were first restricted to arcs on which two further annotators agreed with Stanza on both head and relation, and that filter is not published. Run 11 therefore passed its arc list to `arc_deletion.py` with `--arcs`; without it, the published harness takes every Stanza arc of the 26 relations, so re-running it draws a larger, unfiltered sample rather than run 11's. No `flat` arc survived the filter, so run 11 measures 25 relations.

Run 11 covers 857 documents and 9813 arcs, each with its own matched control, up to 450 arcs per relation, seed 0: 857 baselines, 26726 single deletions and 19626 joint deletions, 47209 passes. `iobj` has 31 arcs and `parataxis` 24; both are marked underpowered in the tables. Of the 9813 controls, 77.5% matched exactly. Adjacent pairs are 37.0% of arcs and 22.0% of controls, so distance is a covariate rather than assumed matched. The run's `runs` row names the script by the path it had before publication.

## Analysis

Standard errors are clustered on document, over as many clusters as the header of `results/statistics.md` reports.

- Raw: per relation, the mean over arcs of the arc's interaction minus its own control's.
- Adjusted: one least-squares fit over all 19626 pairs with a fixed effect per relation, an arc indicator per relation, and covariates for token distance (bins 1, 2, 3, 4-6, 7+), Qwen token count, in-quote status, position, copies of each word elsewhere in the verbalisation (capped at 3), copies of the source document's final token, and each word's UPOS. The arc indicator's coefficient is the adjusted arc-minus-control. Part of speech enters as one set of indicators per word rather than per pair, because within a relation the arc's part-of-speech pair is close to constant.
- Winsorised: the adjusted fit with every interaction clipped to the pooled 1st and 99th percentiles (-1.074 and +2.232 points against a range of -43.542 to +49.954). This check was chosen after the unclipped results were seen.

## Results

From `results/statistics.md`:

- Adjusted: the 95% interval excludes zero for 3 of 25 relations (acl, advmod, cc), and lies entirely beyond the 0.044-point floor for 2 of those (advmod, cc). The Bonferroni interval over 25 relations excludes zero for none.
- Winsorised: the 95% interval excludes zero for 8 of 25, and lies entirely beyond the floor for none. The Bonferroni interval excludes zero for 2 (advmod, nsubj).

The per-relation tables, the interaction by token distance and the covariate coefficients are in `results/statistics.md`.
