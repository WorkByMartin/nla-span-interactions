# 01_corpus-and-spans

Draws the document sample the rest of the work runs on.

```bash
python draw_corpus.py --n 5000 --out results/ffw-5k_corpus.parquet
```

No GPU. About 440 MB of transfer for 5000 documents, plus a 40 second listing of the dataset's shards on first run, cached in `.ffw_cache/`.

## The draw

The NLA studied here was trained on 100,000 FineFineWeb documents that were never published, and FineFineWeb has no global document order, so that set cannot be copied. The 63 training documents ceselder shipped as examples span about 40 of the 66 domains with ids spread evenly, so we draw proportionally instead: a domain is picked with probability set by its document count, then a random shard, then a 256 KB window at a random byte offset. Shards are 317 MB and never downloaded whole.

Six domains are excluded (`weapons_science`, `nuclear_science`, `chemistry`, `biology`, `gamble`, `relationship`), being nearest the material that could trigger residual safety behaviour in the model under study. A narrow regex then runs over the text, since domain labels are coarse. Each document gets one read-out position, drawn uniformly from positions with at least 128 tokens before them.

## Output

One row per document: `doc_uid`, `global_id`, `url`, `domain`, `round`, `n_tokens`, `token_position`, `shard`, `window_start`, `text`. Draw parameters, seed and rejection counts are in the file's Parquet metadata.

Seed and filters are constants at the top of `draw_corpus.py`, and positions derive from hashes of document ids rather than process state, so the same command returns the same documents.

## The traces

`extract_traces.py` carries a sub-sample of the draw through the NLA once and records, per document: the layer-42 activation, the verbalisation the RL verbaliser produced for it, the explanation parsed out of that verbalisation, the generated token ids with per-token KL between the verbaliser and the SFT reference, the reconstruction, and its MSE. Domain, token count, read-out position and the CJK fraction of the explanation come across with each row. Run parameters, seed, the scored FVE and its predict-the-mean baseline are in the file's Parquet metadata.

It was run once, as

```bash
python extract_traces.py --corpus ffw-5k_corpus.parquet --assets "$ASSETS" \
    --n 100 --baseline-n 1000 --resume --out ffw-5k_pilot_traces.parquet
```

with the asset directories under `$ASSETS`. Its output is kept at `results/ffw_pilot_traces.parquet`, which `db/load_traces.py` folds into the `ffw_span-ablation_database` store. It is not re-run: generation is sampled at temperature 1.0 from a 27B model, so a second run draws a different sample.
