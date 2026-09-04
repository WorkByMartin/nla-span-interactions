# 02_kl-span-decomposition

Reports where, along a verbalisation, the KL between the RL verbaliser and its SFT reference concentrates, and on what kind of token.

```bash
python kl_analysis.py --db ../db/ffw_span-ablation_database.sqlite --out results
```

No GPU. Downloads the `Qwen/Qwen3.6-27B` tokeniser files on first run, no weights.

The traces it reads were extracted in `01_corpus-and-spans/` and are held in the `ffw_span-ablation_database` asset. Every number below is in `results/statistics.md`, which the script prints and writes on every run.

## Token range

The active range excludes the closing `</explanation>` tag and starts after the opening tag. No further tokens are trimmed from the head. Per-token text is recovered byte-wise, since Qwen is byte-level BPE and a multi-byte character can straddle two tokens; the pieces are asserted to concatenate to the stored verbalisation plus the trailing `<|im_end|>`.

## The sample

100 documents, 18997 generated tokens, 18097 of them inside the active range. Paragraph counts are 3 (35 documents), 4 (45), 5 (17), 6 (2) and 8 (1). Mean FVE is 0.7595 against a predict-the-mean baseline of 0.4545. Pooled per-token KL has mean 0.7387 and maximum 30.574.

## What carries the highest KL

The loud slice pools all 18097 in-range tokens and takes the top 1% by KL, which is 181 tokens with KL at or above 6.435. Lift is a class's share of that slice over its share of all tokens.

| surface class | loud | share of slice | share of all | lift |
| --- | --- | --- | --- | --- |
| other punctuation | 30 | 16.6% | 2.4% | 6.99x |
| quote mark | 43 | 23.8% | 5.0% | 4.72x |
| word continuation | 64 | 35.4% | 17.5% | 2.02x |
| whitespace | 1 | 0.6% | 0.3% | 1.61x |
| newline | 5 | 2.8% | 3.0% | 0.93x |
| sentence punctuation | 5 | 2.8% | 7.3% | 0.38x |
| word start | 33 | 18.2% | 63.1% | 0.29x |

By position in the generation, 55.2% of the loud tokens sit in the fourth quintile (2.78x) and 27.1% in the fifth (1.34x), against 2.8% in the first (0.14x) and 2.2% in the second (0.11x). By paragraph, 60.2% sit in the last one (1.77x) and 2.8% in the first (0.19x).

The three most common loud forms are `>:`, loud on 42 of its 46 occurrences with mean KL 21.56; `"`, loud on 39 of 448 with mean KL 1.63; and `</`, loud on 24 of 45 with mean KL 6.78. The script prints the thirty most common; `results/kl_token_frequency.png` charts the first fifteen, each with its occurrence count, loud rate and mean KL.

Forms tied on loud count and loud rate are ordered by mean KL and then by the form itself, so the reported slice does not depend on document order.

## Files

```
kl_analysis.py                  the loud-token composition and the three figures
results/statistics.md           the printed report
results/kl_token_frequency.png  the most common loud surface forms
results/kl_example_document.png the median-mean-KL document, shaded by per-token KL, with paragraph breaks
results/kl_by_position.png      median and mean KL, and loud-slice share, by relative position in the generation
```
