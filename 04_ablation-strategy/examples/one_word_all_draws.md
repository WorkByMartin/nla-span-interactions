# One word, all draws

One ablated word, every draw of it. Document 1934, baseline FVE 0.8749. Effect is FVE points
lost, -100 x (fve - base_fve), so a positive number means the reconstruction got worse.
The harness floor is 0.044 points.

The word is 'offices' (NOUN), the NOUN or PROPN span of the document with the largest mean
swap effect out of 8. Its sentence, the word in capitals between double angle brackets:

    "The scandal began when the finance minister ordered the closure of the Kenya Revenue
    Authority <<OFFICES>> for two" concludes with a time interval, likely "temporary
    closure" or specific date or moratorium implementation

Mean swap effect +1.616, deletion +0.723, masked-LM mean over all 48 draws
+0.366 FVE points.

Corpus swap, 16 draws from other documents matched on part of speech, token count
and space parity:

| substitute  | effect |
|-------------|-------:|
| search      | +3.871 |
| figure      | +1.856 |
| commitments | +1.812 |
| aggregator  | +1.747 |
| style       | +1.711 |
| hand        | +1.685 |
| stitch      | +1.605 |
| easing      | +1.580 |
| article     | +1.534 |
| text        | +1.479 |
| tonight     | +1.399 |
| tourist     | +1.353 |
| exporters   | +1.218 |
| brand       | +1.105 |
| life        | +1.001 |
| website     | +0.898 |

Masked-LM marginalisation, 48 draws over six candidate depths, deduplicated by
substitute. 14 of the 48 draws put the original word back.

| substitute   | draws | effect | note          |
|--------------|------:|-------:|---------------|
| in           |     1 | +1.547 |               |
| Department   |     3 | +1.283 |               |
| agency       |     1 | +1.049 |               |
| here         |     1 | +1.043 |               |
| Limited      |     3 | +1.024 |               |
| operation    |     1 | +0.705 |               |
| Office       |     2 | +0.605 |               |
| operations   |     1 | +0.438 |               |
| service      |     1 | +0.417 |               |
| office       |    18 | +0.231 |               |
| headquarters |     2 | +0.050 |               |
| offices      |    14 | +0.000 | original word |

Deletion, one variant: +0.723 FVE points.

Figure: ../results/one_word_all_draws.png, copied here. Left the corpus-swap draws, right the masked-LM
draws deduplicated by substitute with the draw count in the label, grey where the draw put the
original word back. The dashed line on both panels is the word's single deletion.
