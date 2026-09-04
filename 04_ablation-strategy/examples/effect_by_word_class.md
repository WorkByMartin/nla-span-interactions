# Effect by word class

The signed mean effect of an ablation, by the coarse part of speech of the word ablated,
recomputed from the store over 1598 spans in 40 documents. Effect is FVE points lost,
-100 x (fve - base_fve), so a positive number means the edit made the reconstruction worse and
a negative number means it improved it. Classes holding fewer than 10 spans are pooled
into one row. Every interval is a standard error clustered on the document, over 40 clusters,
which is enough for the asymptotics to be worth something and not enough to lean on hard.

`MLM all` is every masked-LM draw. `MLM non-id` drops the draws that put the original word
back, which is 45 per cent of them, and is the column the chart uses.

| class        | spans | swap (FVE points) | MLM all (FVE points) | MLM non-id (FVE points) | deletion (FVE points) |
|--------------|------:|------------------:|---------------------:|------------------------:|----------------------:|
| SCONJ        |    36 |   +0.326 +- 0.094 |      +0.014 +- 0.019 |         +0.047 +- 0.080 |       +0.341 +- 0.154 |
| NUM          |    46 |   +0.287 +- 0.159 |      +0.098 +- 0.063 |         +0.155 +- 0.097 |       +1.399 +- 1.266 |
| PRON         |   100 |   +0.230 +- 0.115 |      +0.086 +- 0.045 |         +0.274 +- 0.143 |       +0.400 +- 0.231 |
| AUX          |   124 |   +0.195 +- 0.097 |      +0.029 +- 0.012 |         +0.075 +- 0.029 |       +0.559 +- 0.326 |
| PART         |    72 |   +0.163 +- 0.065 |      +0.039 +- 0.034 |         +0.070 +- 0.061 |       +0.112 +- 0.056 |
| DET          |   164 |   +0.155 +- 0.065 |      +0.003 +- 0.007 |         +0.015 +- 0.024 |       +0.056 +- 0.040 |
| PROPN        |   121 |   +0.125 +- 0.046 |      +0.089 +- 0.051 |         +0.114 +- 0.066 |       +0.191 +- 0.085 |
| ADJ          |   180 |   +0.095 +- 0.034 |      +0.073 +- 0.030 |         +0.095 +- 0.040 |       +0.039 +- 0.039 |
| ADV          |   137 |   +0.088 +- 0.042 |      +0.042 +- 0.026 |         +0.057 +- 0.037 |       +0.170 +- 0.079 |
| VERB         |   145 |   +0.076 +- 0.041 |      +0.045 +- 0.032 |         +0.059 +- 0.042 |       +0.048 +- 0.040 |
| other (INTJ) |     4 |   +0.061 +- 0.090 |      +0.096 +- 0.155 |         +0.102 +- 0.163 |       +0.015 +- 0.040 |
| ADP          |   175 |   +0.059 +- 0.037 |      -0.004 +- 0.012 |         -0.011 +- 0.032 |       +0.003 +- 0.029 |
| CCONJ        |   133 |   +0.052 +- 0.047 |      +0.010 +- 0.012 |         +0.029 +- 0.037 |       +0.181 +- 0.203 |
| NOUN         |   161 |   +0.052 +- 0.027 |      -0.057 +- 0.091 |         -0.079 +- 0.126 |       -0.019 +- 0.054 |

Rows are sorted by the swap column, largest first.

## Check against statistics.md: matched

Every one of the 56 cells recomputed here (14 classes x 4 arms), and every
span count, is identical to the table printed in ../results/statistics.md at the three decimal
places that file prints. The chart is drawn from the recomputed values.

Figure: ../results/effect_by_word_class.png, copied here. One row per class, sorted by the swap effect, three markers per row
with a document-clustered standard error either side, a line at zero and the harness floor
shaded.

  NUM / deletion is off the right of the chart at +1.399 +- 1.266 FVE points and is drawn as an arrow at the edge.
  The axis is set from every other point so the rest of the classes stay readable.
