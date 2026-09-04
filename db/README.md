# Span-ablation store

One SQLite database per project, `ffw_span-ablation_database.sqlite`. Three things are kept apart: the substring, the labels somebody put on it, and the measurement somebody took of an edited document. Decomposers and measurement scripts append to it; nothing owns a private file format.

The file is the asset `ffw_span-ablation_database` in `ASSETS.yaml`. It exceeds GitHub's file size limit, so it is shipped as a GitHub release asset rather than committed. The document rows were loaded from `01_corpus-and-spans/results/ffw_pilot_traces.parquet`, which is float64 while the database stores float32 blobs, so the parquet stays the precise source.

```
python db.py                            # schema version, outstanding migrations, row counts
python load_traces.py                   # the traces parquet (needs schema version 3)
python load_pod_run.py PULLED --from-run 6 --to-run 7 --scheme S --verify   # a run measured on a GPU host, see below
python snapshot.py "why"                # copy the database, with a sidecar saying what changed
```

## Rules

**No migration is applied as a side effect.** `db.migrate` reports by default and applies only with `apply=True`. Every loader gates on the version it needs and stops with a message rather than changing the schema. Pass `--migrate` to apply, or point `--db` at a throwaway copy.

**Snapshot immediately before applying any migration, and after any loader run.** The description says what is about to change, or what was just loaded.

```
python snapshot.py "before 0003 traces"
python db.py ffw_span-ablation_database.sqlite --migrate
python load_traces.py
python snapshot.py "traces loaded, 100 docs 18997 token rows"
python snapshot.py --list
```

**A run measured elsewhere is loaded, not merged by hand.** The GPU scripts of 05 to 08 ran on a host holding a staged copy of this file and wrote their run into that copy. `load_pod_run.py` copies one run back: the `runs` row, its variants, substitutions, measurements and the relations written under the run's scheme. The run id and every variant id are reassigned, span ids are kept and each one is checked to exist locally at the same offsets before anything is written, and `--verify` reports without writing. Snapshot before and after, as for any loader.

```
python snapshot.py "before loading 06, pod run 7 -> local run 8"
python load_pod_run.py pulled.sqlite --from-run 7 --to-run 8 --scheme tree-vs-linear/all-pairs --write
python snapshot.py "after loading 06 as run 8"
```

**Leading spaces are never stored.** Both tokenisers are byte-level BPE and carry the preceding space inside the token, so the substitution primitive folds that space into a unit when it builds the ModernBERT mask range and when it splices. That is the primitive's internal business. What the store holds is the bare word: `spans` are spaCy's own token offsets, and `substitutions.substitute` and `candidates.candidate` have no leading space.

**`[unrecorded]` is a sentinel, not a substitute.** Rows whose `substitute` is `[unrecorded]` carry a `source` ending in ` unrecorded`, meaning the string that was spliced in was never written out. Filter on `source LIKE '% unrecorded'` rather than on the substitute string, because six rows really did substitute a literal `?`.

## Schema

| table | columns | notes |
| --- | --- | --- |
| `docs` | `doc_id` PK, `text`, `source` | `text` is the explanation, which is what spans index into; `source` names where the row came from. Every row holds `ffw_main_traces`, the name the trace set carried before this database absorbed it; the registry entry for the rows is `ffw_span-ablation_database` itself |
| `docs` (0003) | `global_id`, `domain`, `n_tokens`, `token_position`, `cjk_fraction`, `mse`, `verbalisation`, `activation` BLOB, `reconstruction` BLOB | vectors are float32 little-endian, 5120 wide; read with `numpy.frombuffer(b, '<f4')` |
| `doc_tokens` (0003) | `doc_id`, `position`, `token_id`, `kl`, PK(doc,position) | the verbaliser's generation with per-token KL against the SFT reference |
| `spans` | `span_id` PK, `doc_id`, `char_start`, `char_end`, UNIQUE(doc,start,end) | one span per word, bare. No type column, no stored text |
| `labels` | `span_id`, `scheme`, `key`, `value`, PK(span,scheme,key) | long format; `scheme` names decomposer plus version |
| `relations` | `scheme`, `span_a`, `span_b`, `kind`, PK(all four) | dependent to head is `kind='head'` |
| `candidates` (0002) | `span_id`, `scheme`, `rank`, `candidate`, `prob`, PK(span,scheme,rank) | the ranked filler list a substitute is drawn from. Rank 0 is best |
| `variants` | `variant_id` PK, `doc_id`, `created_run_id` | one edited document, one reconstructor forward pass. No substitutions means baseline |
| `substitutions` | `variant_id`, `span_id`, `substitute`, `source`, `depth`, `draw_idx`, `prob`, PK(variant,span) | |
| `measurements` | `variant_id`, `run_id`, `metric`, `value`, PK(variant,run,metric) | `mse`, `fve`, `seq_len`, `traces_mse` on every run; `dtok` from run 7 on; further per-run metrics (curve, step, condition, and so on) are named in the numbered directory that wrote the run |
| `runs` | `run_id` PK, `script`, `git_sha`, `assets` json, `notes`, `started_at`, `config` json (0002) | `config` holds argv, environment fingerprint, and every constant the numbers depend on |
| `schema_version` | `version` | one row per applied migration |

Views: `v_span_text` (span plus derived text), `v_pos` (span, doc, text, POS from any `spacy-%` scheme), `v_candidate`, `v_lexicon` (modal POS per word type, over spans carrying a `tok_i` label), `v_single` (one-substitution variants with metrics pivoted and `dmse`/`dfve` against the same doc and run's baseline), `v_pair` (two-substitution variants with `inter_mse`/`inter_fve` = both - a - b + baseline, NULL when a matching single or the baseline is absent), `v_doc` and `v_doc_kl` (0003). Helpers: `v_nsub`, `v_variant_metrics`, `v_baseline`, `v_baseline_repeats`, `v_single_mean`, `v_word_pos`.

Schemes:

| scheme | keys or content |
| --- | --- |
| `spacy-en_core_web_sm-3.8.0` | `pos`, `tag`, `dep`, `tok_i` labels; `head` relations |
| `qwen36-27b_tokenizer` | `n_tokens_qwen` |
| `modernbert-large_filler_model/idsplice` | masked fill filtered on Qwen token count alone. `entropy`, `mass_kept`, `leak@k` labels, and its candidate lists |
| `modernbert-large_filler_model/textsub` | masked fill filtered on space parity plus a clean-word test, over a ModernBERT token range collapsed to one `[MASK]` |

The two filler schemes are deliberately separate. A candidate list produced under one regime is not interchangeable with the other, and a shared scheme name would silently mix them.

## Runs

The `script` column of `runs` records the path the script had when it ran. The path is the scratch workbench for every run but 5; the directory each script now lives in is given here. The `assets` column of runs 7 to 10 was corrected after the fact to the registry IDs the runs consumed, and their `config` carries `wall_time_s`, the seconds from the first document to the last as measured by the run script on the GPU host; both corrections are recorded in the row's config and bracketed by snapshots 0024 to 0026.

| run | written by | now at |
| --- | --- | --- |
| 1, 3 | `pos_fve.py`, word-level single-span ablation by class and depth | `03_parts-of-speech/` (run 3 is the one reported there) |
| 2 | `extract_traces.py`, the 100-document trace set | `01_corpus-and-spans/` |
| 4 | `swap_pilot.py`, five-document pilot of the corpus swap | superseded by run 5 |
| 5 | `swap_ablation.py` | `04_ablation-strategy/` |
| 6 | `extract_traces.py`, the held-out 1,000-document trace set | `01_corpus-and-spans/` |
| 7 | `pair_ablation.py` | `05_dependent-pair/` |
| 8 | `tree_vs_linear.py` | `06_tree-vs-linear/` |
| 9 | `removal_curve.py` | `07_removal-curve/` |
| 10 | `negation.py` | `08_negation/` |

## Migrations

`migrations/NNNN_name.sql`, applied in filename order, each inside one transaction, with `NNNN` recorded in `schema_version`. Never edit an applied migration; add the next number. `db.pending(conn)` lists what is outstanding, and `python db.py <path> --migrate` applies them.

## Queries

POS histogram data:

```sql
SELECT pos, COUNT(*) AS n FROM v_pos GROUP BY pos ORDER BY n DESC;
```

Leakage per class, the fraction of substitutions that reproduce the span:

```sql
SELECT p.pos, s.depth,
       AVG(CASE WHEN s.substitute = s.span_text THEN 1.0 ELSE 0.0 END) AS leak,
       COUNT(*) AS n
FROM v_single s
JOIN v_pos p ON p.span_id = s.span_id
WHERE s.source NOT LIKE '%unrecorded'
GROUP BY p.pos, s.depth
ORDER BY p.pos, s.depth;
```

Interaction for one document, largest first:

```sql
SELECT text_a, text_b, sub_a, sub_b, both_fve, a_fve, b_fve, base_fve, inter_fve
FROM v_pair
WHERE doc_id = 3696 AND run_id = 1 AND inter_fve IS NOT NULL
ORDER BY abs(inter_fve) DESC
LIMIT 20;
```

Adding a new decomposer's labels. Spans are shared, schemes are not, so a second parser coexists with the first rather than overwriting it:

```python
import db
conn = db.connect(); db.require_schema(conn, needs=2)
with db.transaction(conn):
    sid = db.get_or_create_span(conn, doc_id, start, end)
    db.set_label(conn, sid, "spacy-en_core_web_trf-3.8.0", "dep", "nsubj")
    db.add_relation(conn, "spacy-en_core_web_trf-3.8.0", sid, head_sid, "head")
```

Adding a new metric, no migration needed:

```python
with db.transaction(conn):
    vid = db.new_variant(conn, doc_id, run_id, substitutions=[
        {"span_id": sid, "substitute": "the", "source": "modernbert-large_filler_model/textsub",
         "depth": 10, "draw_idx": 0, "prob": 0.31}])
    db.record(conn, vid, run_id, "kl_mean", 0.0421)
```

```sql
SELECT AVG(value) FROM measurements WHERE run_id = ? AND metric = 'kl_mean';
```

`v_variant_metrics` pivots `mse`, `fve` and `seq_len` only, so a new metric becomes a column on `v_single` and `v_pair` only through a migration that redefines those views.
