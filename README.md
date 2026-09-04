# nla-span-interactions

Word-level ablations of the natural language autoencoder's verbalisation, scored by what the reconstructor recovers of the layer-42 activation the verbalisation describes. Every experiment edits the text of a verbalisation, runs the reconstructor forward, and measures the change in fraction of variance explained (FVE) against the same document's unedited baseline. What varies from one directory to the next is the unit edited, how it is edited, and which pairs or sequences of edits are measured together.

The directories are numbered in the order the work was done. Each one is self-contained: its `README.md` states what it measures, the one command that regenerates its report, the design, the sample, and the results, and every number in it traces to that directory's `results/statistics.md`, which the analysis script prints and rewrites on every run.

| directory | what it does |
| --- | --- |
| `01_corpus-and-spans/` | draws the document sample and extracts the activation, verbalisation, reconstruction and per-token KL for each document |
| `02_kl-span-decomposition/` | where along a verbalisation the KL between the RL verbaliser and its SFT reference concentrates |
| `03_parts-of-speech/` | single-word ablation by masked-LM substitution, broken down by the word's syntactic class and the candidate depth |
| `04_ablation-strategy/` | the same single-word ablation by corpus swap, deletion and word-order shuffle, read against 03 |
| `05_dependent-pair/` | pairwise interaction of two words on a dependency arc against matched pairs that are not |
| `06_tree-vs-linear/` | the full pairwise interaction matrix over the words of a document, read against tree distance and linear distance |
| `07_removal-curve/` | FVE as a document's words are removed one at a time, under three primitives and three orderings |
| `08_negation/` | flipping a negation against controls that edit the same words without touching polarity |
| `db/` | the SQLite store every run writes to and every analysis reads from, with its schema, migrations and loaders |

## Running

The analyses need no GPU and no model weights. Each reads one run of the store and writes `results/statistics.md` and the figures beside it:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
cd 04_ablation-strategy && python swap_analysis.py --run 5
```

The GPU scripts that wrote each run, their arguments, and the assets they consumed are given in the "Regenerating run N" section of each directory's README. Every run appends to the store rather than overwriting an earlier one, and the `runs` table records the script, its arguments and the environment fingerprint for each.

## Assets

`ASSETS.yaml` is the registry of every model, corpus and database the code consumes, with its provenance and hash. Code resolves assets by ID against it. The store, `ffw_span-ablation_database`, exceeds GitHub's file size limit and is shipped as a release asset. The models are named in the registry with their source and the revision each run consumed.

## Layout

```
NN_<slug>/            one piece of work: source .py, tests, results/, README.md
db/                   the store: db.py, dbio.py, migrations/, loaders, snapshot.py
figstyle.py           the one figure style every experiment draws in
store.py              the store's path and the text layout the printed reports share
ASSETS.yaml           the asset registry
requirements.txt
```
