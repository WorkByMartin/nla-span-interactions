```
SAMPLE
  documents 100, generated tokens 18997
  paragraphs per document: 3: 35, 4: 45, 5: 17, 6: 2, 8: 1
  mse mean 0.1093, fve mean 0.7595 (baseline 0.4545)
  kl mean 0.7387, kl max 30.5741

LOUD SLICE: tokens above the 99.0th percentile of KL, 181 of 18097, KL >= 6.4353
  surface class            loud     all   lift   (shares of the loud slice and of all tokens)
  other punctuation       0.166   0.024   6.99
  quote mark              0.238   0.050   4.72
  word continuation       0.354   0.175   2.02
  whitespace              0.006   0.003   1.61
  newline                 0.028   0.030   0.93
  sentence punctuation    0.028   0.073   0.38
  word start              0.182   0.631   0.29
  position quintile        loud     all   lift
    0-20%                 0.028   0.201   0.14
    20-40%                0.022   0.199   0.11
    40-60%                0.127   0.199   0.64
    60-80%                0.552   0.199   2.78
    80-100%               0.271   0.203   1.34
  paragraph                loud     all   lift
    first paragraph       0.028   0.147   0.19
    a middle paragraph    0.370   0.512   0.72
    last paragraph        0.602   0.340   1.77
  surface form             loud    seen   rate  mean kl
  '>:'                       42      46   0.91    21.56
  '"'                        39     448   0.09     1.63
  '</'                       24      45   0.53     6.78
  ' "'                        4     451   0.01     0.65
  '\n'                        3     246   0.01     0.87
  '['                         2       5   0.40     5.19
  'A'                         2      15   0.13     2.49
  ' expects'                  2      21   0.10     1.68
  ' —'                        2     137   0.01     0.83
  '\n\n'                      2     290   0.01     0.56
  '.'                         2     427   0.00     0.53
  ','                         2     635   0.00     0.45
  ':\\'                       1       1   1.00    11.42
  'Spanish'                   1       1   1.00    10.05
  'event'                     1       1   1.00     9.63
  'G'                         1       1   1.00     8.84
  ' end'                      1       1   1.00     8.49
  'Peter'                     1       1   1.00     8.20
  ' pillow'                   1       1   1.00     7.93
  ' fulfillment'              1       1   1.00     7.78
  'Attached'                  1       1   1.00     7.67
  ' farmer'                   1       1   1.00     7.55
  'search'                    1       1   1.00     7.32
  ' grow'                     1       1   1.00     7.30
  ' successfully'             1       1   1.00     7.30
  ' abs'                      1       1   1.00     6.88
  ' environment'              1       1   1.00     6.82
  'private'                   1       1   1.00     6.77
  ' makeup'                   1       1   1.00     6.65
  ' Canadian'                 1       1   1.00     6.48

FIGURES
  kl_example_document.png: document 4635, every token shaded by its KL, paragraph breaks marked; chosen for its mean KL 0.7279, nearest the median over documents of 0.7302
  kl_by_position.png: median KL with the 25th to 75th percentile band, and the mean, over 20 position bins, and the share of the 181 loud tokens falling in each bin
```
