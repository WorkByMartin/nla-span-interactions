# Span ablation by syntactic class, run 3

## Inputs

run 3: 76800 single-substitution draws, 40 documents, 1600 spans, 14 word classes

ModernBERT token counts: 1600 spans tiled exactly, 0 not tiled by whole tokens

corpus lexicon for class purity: 3444 word types

## Class by depth, all 40 documents

mean FVE lost, points, positive = reconstruction got worse; +/- is the per-document clustered standard error

\* marks a cell whose 95% clustered interval excludes zero

| class | n@50 | @1 | @3 | @5 | @10 | @25 | @50 | 95% CI @50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NUM | 376 | 0.066 +/- 0.056 | 0.056 +/- 0.043 | 0.153 +/- 0.097 | 0.090 +/- 0.051 | 0.117 +/- 0.061 | 0.147 +/- 0.128 | [-0.111, 0.405] |
| PROPN | 968 | 0.077 +/- 0.053 | 0.089 +/- 0.056 | 0.092 +/- 0.056 | 0.100 +/- 0.049* | 0.092 +/- 0.052 | 0.086 +/- 0.048 | [-0.011, 0.183] |
| PRON | 800 | 0.041 +/- 0.044 | 0.097 +/- 0.046* | 0.082 +/- 0.052 | 0.110 +/- 0.067 | 0.107 +/- 0.049* | 0.080 +/- 0.045 | [-0.011, 0.172] |
| INTJ | 32 | 0.145 +/- 0.254 | 0.044 +/- 0.165 | 0.059 +/- 0.135 | 0.167 +/- 0.163 | 0.080 +/- 0.062 | 0.080 +/- 0.166 | [-0.256, 0.416] |
| ADJ | 1440 | 0.053 +/- 0.030 | 0.067 +/- 0.032* | 0.080 +/- 0.030* | 0.083 +/- 0.033* | 0.081 +/- 0.032* | 0.073 +/- 0.031* | [0.010, 0.136] |
| ADV | 1096 | 0.020 +/- 0.021 | 0.037 +/- 0.033 | 0.041 +/- 0.028 | 0.041 +/- 0.028 | 0.058 +/- 0.031 | 0.055 +/- 0.029 | [-0.004, 0.114] |
| VERB | 1160 | 0.054 +/- 0.035 | 0.040 +/- 0.033 | 0.047 +/- 0.033 | 0.047 +/- 0.033 | 0.031 +/- 0.029 | 0.052 +/- 0.032 | [-0.013, 0.118] |
| PART | 576 | 0.047 +/- 0.045 | 0.043 +/- 0.040 | 0.039 +/- 0.035 | 0.036 +/- 0.030 | 0.036 +/- 0.028 | 0.034 +/- 0.029 | [-0.025, 0.094] |
| SCONJ | 288 | 0.002 +/- 0.016 | 0.007 +/- 0.019 | 0.002 +/- 0.023 | 0.026 +/- 0.021 | 0.017 +/- 0.020 | 0.028 +/- 0.022 | [-0.017, 0.073] |
| AUX | 992 | 0.028 +/- 0.013* | 0.026 +/- 0.012* | 0.022 +/- 0.010* | 0.040 +/- 0.015* | 0.030 +/- 0.012* | 0.027 +/- 0.014 | [-0.001, 0.054] |
| DET | 1312 | 0.000 +/- 0.008 | 0.011 +/- 0.009 | 0.003 +/- 0.009 | -0.001 +/- 0.009 | 0.005 +/- 0.008 | 0.001 +/- 0.009 | [-0.017, 0.019] |
| CCONJ | 1064 | 0.007 +/- 0.013 | 0.001 +/- 0.012 | 0.008 +/- 0.013 | 0.026 +/- 0.015 | 0.015 +/- 0.012 | 0.000 +/- 0.013 | [-0.026, 0.027] |
| ADP | 1408 | -0.022 +/- 0.011* | 0.006 +/- 0.020 | 0.007 +/- 0.016 | -0.006 +/- 0.010 | -0.006 +/- 0.011 | -0.004 +/- 0.012 | [-0.029, 0.021] |
| NOUN | 1288 | -0.080 +/- 0.107 | -0.056 +/- 0.089 | -0.025 +/- 0.066 | -0.045 +/- 0.085 | -0.065 +/- 0.097 | -0.069 +/- 0.100 | [-0.272, 0.134] |

differ from zero at depth 50: ADJ

## Class by depth, document 2159 excluded (39 documents)

mean FVE lost, points, positive = reconstruction got worse; +/- is the per-document clustered standard error

\* marks a cell whose 95% clustered interval excludes zero

| class | n@50 | @1 | @3 | @5 | @10 | @25 | @50 | 95% CI @50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NUM | 376 | 0.066 +/- 0.056 | 0.056 +/- 0.043 | 0.153 +/- 0.097 | 0.090 +/- 0.051 | 0.117 +/- 0.061 | 0.147 +/- 0.128 | [-0.111, 0.405] |
| PROPN | 968 | 0.077 +/- 0.053 | 0.089 +/- 0.056 | 0.092 +/- 0.056 | 0.100 +/- 0.049* | 0.092 +/- 0.052 | 0.086 +/- 0.048 | [-0.011, 0.183] |
| INTJ | 32 | 0.145 +/- 0.254 | 0.044 +/- 0.165 | 0.059 +/- 0.135 | 0.167 +/- 0.163 | 0.080 +/- 0.062 | 0.080 +/- 0.166 | [-0.256, 0.416] |
| ADV | 1056 | 0.034 +/- 0.017 | 0.062 +/- 0.022* | 0.060 +/- 0.022* | 0.062 +/- 0.020* | 0.073 +/- 0.028* | 0.068 +/- 0.027* | [0.013, 0.123] |
| ADJ | 1400 | 0.057 +/- 0.030 | 0.072 +/- 0.033* | 0.070 +/- 0.029* | 0.072 +/- 0.032* | 0.076 +/- 0.032* | 0.065 +/- 0.031* | [0.002, 0.128] |
| PART | 568 | 0.057 +/- 0.045 | 0.049 +/- 0.040 | 0.052 +/- 0.034 | 0.047 +/- 0.028 | 0.045 +/- 0.027 | 0.045 +/- 0.028 | [-0.011, 0.102] |
| PRON | 768 | -0.002 +/- 0.009 | 0.084 +/- 0.045 | 0.030 +/- 0.010* | 0.043 +/- 0.011* | 0.064 +/- 0.025* | 0.039 +/- 0.019* | [0.000, 0.077] |
| VERB | 1128 | 0.037 +/- 0.031 | 0.024 +/- 0.030 | 0.029 +/- 0.028 | 0.032 +/- 0.031 | 0.024 +/- 0.029 | 0.036 +/- 0.029 | [-0.023, 0.095] |
| NOUN | 1256 | 0.026 +/- 0.014 | 0.032 +/- 0.015* | 0.039 +/- 0.016* | 0.038 +/- 0.017* | 0.031 +/- 0.017 | 0.030 +/- 0.016 | [-0.003, 0.062] |
| SCONJ | 288 | 0.002 +/- 0.016 | 0.007 +/- 0.019 | 0.002 +/- 0.023 | 0.026 +/- 0.021 | 0.017 +/- 0.020 | 0.028 +/- 0.022 | [-0.017, 0.073] |
| AUX | 960 | 0.028 +/- 0.013* | 0.027 +/- 0.012* | 0.022 +/- 0.010* | 0.033 +/- 0.013* | 0.030 +/- 0.012* | 0.024 +/- 0.014 | [-0.004, 0.053] |
| CCONJ | 1040 | 0.008 +/- 0.013 | 0.006 +/- 0.011 | 0.015 +/- 0.011 | 0.027 +/- 0.015 | 0.017 +/- 0.013 | 0.006 +/- 0.012 | [-0.019, 0.030] |
| DET | 1272 | 0.001 +/- 0.008 | 0.006 +/- 0.008 | 0.008 +/- 0.008 | 0.004 +/- 0.008 | 0.007 +/- 0.008 | 0.003 +/- 0.009 | [-0.015, 0.021] |
| ADP | 1368 | -0.023 +/- 0.011* | -0.011 +/- 0.010 | -0.006 +/- 0.010 | -0.009 +/- 0.010 | -0.012 +/- 0.010 | -0.012 +/- 0.010 | [-0.032, 0.008] |

differ from zero at depth 50: ADV, ADJ, PRON

## Pooled mean FVE lost by depth, over every word class

clustered on doc_id over 40 documents; the interval is the 95% document-clustered interval

| depth | n | mean | clustered SE | 95% CI |
| --- | ---: | ---: | ---: | ---: |
| 1 | 12800 | 0.018 | 0.012 | [-0.006, 0.042] |
| 3 | 12800 | 0.029 | 0.013 | [0.004, 0.055] |
| 5 | 12800 | 0.036 | 0.010 | [0.016, 0.056] |
| 10 | 12800 | 0.037 | 0.010 | [0.016, 0.058] |
| 25 | 12800 | 0.033 | 0.013 | [0.007, 0.059] |
| 50 | 12800 | 0.031 | 0.012 | [0.006, 0.056] |

## Draws beyond the plotted range

the histogram figures clip FVE lost to plus or minus 1 points and put the overflow in the end bins

overall: 1686 of 76800 draws overflowed (0.0220)

| class | n | beyond +/- 1 | share |
| --- | ---: | ---: | ---: |
| ADJ | 8640 | 421 | 0.0487 |
| ADP | 8448 | 64 | 0.0076 |
| ADV | 6576 | 218 | 0.0332 |
| AUX | 5952 | 27 | 0.0045 |
| CCONJ | 6384 | 41 | 0.0064 |
| DET | 7872 | 22 | 0.0028 |
| INTJ | 192 | 2 | 0.0104 |
| NOUN | 7728 | 119 | 0.0154 |
| NUM | 2256 | 158 | 0.0700 |
| PART | 3456 | 81 | 0.0234 |
| PRON | 4800 | 102 | 0.0213 |
| PROPN | 5808 | 242 | 0.0417 |
| SCONJ | 1728 | 2 | 0.0012 |
| VERB | 6960 | 187 | 0.0269 |

## Top 20 words by mean FVE lost over distinct substitutes

ranked by the mean over the distinct substitute strings drawn for that word, each counted once however many of its 48 draws produced it and whatever depth they came from, since FVE is a deterministic function of the edited text

a substitute that reproduces the original word counts once, at an FVE lost of exactly zero; n identical is how many of the 48 draws were that substitute

the last two columns are the largest single substitute and its FVE lost, for illustration only; they play no part in the ranking

a word occurring more than once in a document is listed once, at its most extreme occurrence

| doc | word | POS | n distinct | n identical | mean over distinct | mean over distinct non-identical | extreme substitute | its FVE lost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1934 | two | NUM | 5 | 42 | 8.421 | 10.526 | a | 13.831 |
| 1574 | and | CCONJ | 2 | 47 | 5.192 | 10.384 | governments | 10.384 |
| 1336 | her | PRON | 4 | 41 | 4.949 | 6.598 | a | 9.116 |
| 355 | our | PRON | 4 | 44 | 4.574 | 6.099 | a | 13.656 |
| 2159 | we | PRON | 12 | 0 | 3.525 | 3.525 | however | 9.103 |
| 2617 | postpone | VERB | 6 | 0 | 3.508 | 3.508 | withdraw | 6.648 |
| 2159 | for | ADP | 2 | 39 | 3.485 | 6.969 | to | 6.969 |
| 2617 | its | PRON | 3 | 42 | 3.397 | 5.095 | the | 7.952 |
| 2159 | a | DET | 3 | 46 | 3.368 | 5.051 | the | 8.903 |
| 2349 | NXP | PROPN | 11 | 0 | 3.364 | 3.364 | AMD | 4.756 |
| 1664 | well | ADV | 5 | 42 | 3.246 | 4.058 | better | 6.259 |
| 2159 | educational | ADJ | 14 | 0 | 2.893 | 2.893 | continuous | 8.547 |
| 2750 | Program | PROPN | 12 | 3 | 2.841 | 3.099 | Software | 10.069 |
| 2349 | Embedded | ADJ | 7 | 0 | 2.723 | 2.723 | Open | 3.299 |
| 2159 | Long | ADJ | 6 | 2 | 2.649 | 3.179 | Free | 16.804 |
| 3154 | yet | ADV | 7 | 38 | 2.627 | 3.065 | another | 12.143 |
| 2009 | Australian | ADJ | 13 | 0 | 2.615 | 2.615 | distressed | 2.849 |
| 2159 | the | PRON | 8 | 33 | 2.182 | 2.493 | us | 4.294 |
| 3375 | dec | PROPN | 5 | 43 | 2.007 | 2.509 | and | 3.245 |
| 2957 | the | DET | 2 | 47 | 1.935 | 3.870 | these | 3.870 |

## Bottom 20 words by mean FVE lost over distinct substitutes

ranked by the mean over the distinct substitute strings drawn for that word, each counted once however many of its 48 draws produced it and whatever depth they came from, since FVE is a deterministic function of the edited text

a substitute that reproduces the original word counts once, at an FVE lost of exactly zero; n identical is how many of the 48 draws were that substitute

the last two columns are the smallest single substitute and its FVE lost, for illustration only; they play no part in the ranking

a word occurring more than once in a document is listed once, at its most extreme occurrence

| doc | word | POS | n distinct | n identical | mean over distinct | mean over distinct non-identical | extreme substitute | its FVE lost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2159 | rest | NOUN | 6 | 0 | -8.423 | -8.423 | context | -16.088 |
| 2159 | green | ADJ | 2 | 47 | -2.381 | -4.763 | black | -4.763 |
| 2159 | likely | ADV | 15 | 0 | -1.574 | -1.574 | starting | -3.105 |
| 2159 | with | ADP | 8 | 39 | -1.411 | -1.613 | requires | -4.736 |
| 2159 | directly | ADV | 12 | 0 | -1.291 | -1.291 | likely | -2.762 |
| 3914 | fullscreen | NUM | 7 | 0 | -1.143 | -1.143 | several | -2.066 |
| 3696 | 2011 | NUM | 12 | 0 | -1.021 | -1.021 | LA | -1.195 |
| 2118 | today | NOUN | 17 | 20 | -0.958 | -1.018 | you | -2.290 |
| 3696 | likely | ADV | 4 | 0 | -0.938 | -0.938 | was | -0.988 |
| 3914 | SDL | PROPN | 12 | 0 | -0.913 | -0.913 | specific | -1.562 |
| 2118 | helping | VERB | 5 | 31 | -0.889 | -1.111 | allowing | -3.108 |
| 3780 | in | ADP | 4 | 31 | -0.784 | -1.046 | of | -1.078 |
| 3780 | token | ADJ | 10 | 0 | -0.732 | -0.732 | comment | -1.430 |
| 2159 | Each | DET | 10 | 16 | -0.731 | -0.812 | previous | -2.589 |
| 3098 | ASA | PROPN | 12 | 0 | -0.669 | -0.669 | replacement | -1.215 |
| 3780 | Finally | ADV | 17 | 14 | -0.656 | -0.697 | Besides | -1.263 |
| 1301 | throughout | ADP | 11 | 0 | -0.650 | -0.650 | observed | -0.814 |
| 3696 | or | CCONJ | 13 | 4 | -0.642 | -0.696 | before | -1.535 |
| 3780 | - | ADJ | 10 | 0 | -0.624 | -0.624 | ve | -1.277 |
| 2118 | mid | ADJ | 13 | 8 | -0.612 | -0.663 | cross | -0.914 |

## Largest single substitutions, top 5

each substitution shown in place, the original struck through and the substitute in bold, with about 15 words either side; a distinct substitute string appears once however many draws produced it

~~Long~~ **Free** -form educational article structure with consistent section headings and repetitive paragraph patterns describing benefits/groups affected ...

doc 2159, ADJ, depth 50, 1 of 48 draws, FVE lost 16.804, baseline FVE -0.133, word mean over distinct 2.649

... the dispute. Final fragment "The Attorney General issued orders to shut down Revenue Authority for ~~two~~ **the** " ends mid-phrase, requiring a time duration (e.g., "two weeks/days/months") followed by explanation of the ...

doc 1934, NUM, depth 5, 1 of 48 draws, FVE lost 14.391, baseline FVE 0.875, word mean over distinct 6.404

... began when the finance minister ordered the closure of the Kenya Revenue Authority offices for ~~two~~ **a** " concludes with a time interval, likely "temporary closure" or specific date or moratorium implementation

doc 1934, NUM, depth 5, 3 of 48 draws, FVE lost 13.831, baseline FVE 0.875, word mean over distinct 8.421

... your entire website." "When we search on Google for our local \</ex\>: "searching Google for ~~our~~ **a** local internet marketing agency" — requires own named agency\</li\>

doc 355, PRON, depth 3, 1 of 48 draws, FVE lost 13.656, baseline FVE 0.685, word mean over distinct 4.574

... and premiered another song. Finally, at the studio, Pop Smoke used the beat to debut ~~yet~~ **another** "

doc 3154, ADV, depth 3, 4 of 48 draws, FVE lost 12.143, baseline FVE 0.811, word mean over distinct 2.627

## Largest single substitutions, bottom 5

each substitution shown in place, the original struck through and the substitute in bold, with about 15 words either side; a distinct substitute string appears once however many draws produced it

... "green coffee, making it an attractive option for a wide range of people within the ~~rest~~ **context** of the" — directly mirroring the previous section's closing phrase, so "world. However, there may ...

doc 2159, NOUN, depth 1, 36 of 48 draws, FVE lost -16.088, baseline FVE -0.133, word mean over distinct -8.423

... "green coffee, making it an attractive option for a wide range of people within the ~~rest~~ **regions** of the" — directly mirroring the previous section's closing phrase, so "world. However, there may ...

doc 2159, NOUN, depth 10, 1 of 48 draws, FVE lost -11.898, baseline FVE -0.133, word mean over distinct -8.423

... "green coffee, making it an attractive option for a wide range of people within the ~~rest~~ **region** of the" — directly mirroring the previous section's closing phrase, so "world. However, there may ...

doc 2159, NOUN, depth 5, 1 of 48 draws, FVE lost -7.562, baseline FVE -0.133, word mean over distinct -8.423

... "green coffee, making it an attractive option for a wide range of people within the ~~rest~~ **scope** of the" — directly mirroring the previous section's closing phrase, so "world. However, there may ...

doc 2159, NOUN, depth 3, 7 of 48 draws, FVE lost -6.088, baseline FVE -0.133, word mean over distinct -8.423

... article structure with consistent section headings and repetitive paragraph patterns describing benefits/groups affected by pre-ground ~~green~~ **black** coffee. Repetition/continuation pattern: Each section systematically lists specific benefits and applicable context; the current paragraph ...

doc 2159, ADJ, depth 3, 1 of 48 draws, FVE lost -4.763, baseline FVE -0.133, word mean over distinct -2.381

## Sequence length: share of substitutions by change in Qwen token count

| change | n | share |
| --- | ---: | ---: |
| -2 | 632 | 0.0082 |
| -1 | 3804 | 0.0495 |
| +0 | 71616 | 0.9325 |
| +1 | 363 | 0.0047 |
| +2 | 106 | 0.0014 |
| other | 279 | 0.0036 |
| total | 76800 | 1.0000 |

## Mean FVE lost, length-preserving vs length-changing, by class

| class | n same | mean same | n changed | mean changed | difference |
| --- | ---: | ---: | ---: | ---: | ---: |
| ADJ | 7996 | 0.057 | 644 | 0.268 | 0.211 |
| ADP | 8397 | -0.004 | 51 | 0.010 | 0.015 |
| ADV | 6472 | 0.042 | 104 | 0.028 | -0.014 |
| AUX | 5834 | 0.027 | 118 | 0.112 | 0.084 |
| CCONJ | 6384 | 0.010 | 0 | nan | nan |
| DET | 7819 | 0.002 | 53 | 0.119 | 0.117 |
| INTJ | 192 | 0.096 | 0 | nan | nan |
| NOUN | 7186 | -0.063 | 542 | 0.027 | 0.090 |
| NUM | 1632 | 0.084 | 624 | 0.160 | 0.076 |
| PART | 2677 | 0.046 | 779 | 0.017 | -0.029 |
| PRON | 4780 | 0.088 | 20 | -0.205 | -0.292 |
| PROPN | 4285 | 0.059 | 1523 | 0.174 | 0.114 |
| SCONJ | 1699 | 0.009 | 29 | 0.301 | 0.292 |
| VERB | 6263 | 0.026 | 697 | 0.220 | 0.195 |

## Leakage-adjusted effect at depth 50

leakage = share of draws whose substitute is the original word

| class | n | all draws | leakage | n changed | changed only |
| --- | ---: | ---: | ---: | ---: | ---: |
| PRON | 800 | 0.080 | 0.651 | 279 | 0.231 |
| NUM | 376 | 0.147 | 0.290 | 267 | 0.208 |
| PROPN | 968 | 0.086 | 0.204 | 771 | 0.108 |
| ADJ | 1440 | 0.073 | 0.212 | 1135 | 0.092 |
| SCONJ | 288 | 0.028 | 0.674 | 94 | 0.084 |
| INTJ | 32 | 0.080 | 0.031 | 31 | 0.083 |
| ADV | 1096 | 0.055 | 0.242 | 831 | 0.072 |
| VERB | 1160 | 0.052 | 0.200 | 928 | 0.065 |
| AUX | 992 | 0.027 | 0.573 | 424 | 0.063 |
| PART | 576 | 0.034 | 0.441 | 322 | 0.061 |
| DET | 1312 | 0.001 | 0.627 | 490 | 0.002 |
| CCONJ | 1064 | 0.000 | 0.661 | 361 | 0.001 |
| ADP | 1408 | -0.004 | 0.579 | 593 | -0.009 |
| NOUN | 1288 | -0.069 | 0.227 | 995 | -0.090 |

## Reproduction floor

a per-draw |dMSE| below 0.0002 is 0.044 FVE points, which is the harness reproduction floor

overall fraction below the floor: 0.5795 (44508 of 76800 draws)

| class | n | below floor |
| --- | ---: | ---: |
| ADJ | 8640 | 0.4053 |
| ADP | 8448 | 0.7350 |
| ADV | 6576 | 0.4574 |
| AUX | 5952 | 0.7424 |
| CCONJ | 6384 | 0.7588 |
| DET | 7872 | 0.7605 |
| INTJ | 192 | 0.1979 |
| NOUN | 7728 | 0.4085 |
| NUM | 2256 | 0.4676 |
| PART | 3456 | 0.6227 |
| PRON | 4800 | 0.7710 |
| PROPN | 5808 | 0.3328 |
| SCONJ | 1728 | 0.7789 |
| VERB | 6960 | 0.4536 |
