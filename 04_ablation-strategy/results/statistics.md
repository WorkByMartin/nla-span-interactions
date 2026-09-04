```
CORPUS SWAP AGAINST MASKED-LM MARGINALISATION
  swap run 5 (2026-09-03T01:30:52), masked-LM run 3
  documents 40: 267, 355, 376, 477, 886, 1281, 1301, 1336, 1379, 1478, 1574, 1583, 1649, 1664, 1925, 1934, 2009, 2046, 2118, 2159, 2349, 2409, 2435, 2492, 2617, 2632, 2750, 2957, 3098, 3154, 3375, 3696, 3780, 3914, 3916, 4187, 4282, 4420, 4436, 4635
  spans 1598, swap draws 25568, deletion variants 1598, masked-LM draws 76704
  2 of the 1600 target spans had an empty swap pool and so carry no swap draws. Every arm below,
  deletion included, is reported over the 1598 spans the swap covers, so the comparison is like for like
  pool 12854 lexical occurrences, 3338 word types, over 100 documents
  pool construction: every lexical span of every verbalisation held in the store, one occurrence per span, grouped by (spaCy coarse POS, leading-space parity); eligibility for a target span additionally requires a different document, a different word case-blind, and the same Qwen token count after recasing to the target's capitalisation
  effect = FVE points lost = -100 x (fve - base_fve); harness floor 0.044 points
  every confidence interval below is clustered on the document, and there are 40 clusters.
  that is enough for the asymptotics to be worth something, though it is still forty and not four hundred.
  baseline FVE per document, this run against the masked-LM run:
       267  0.8310  0.8310  delta +0.00e+00
       355  0.6850  0.6850  delta +0.00e+00
       376  0.8175  0.8175  delta +0.00e+00
       477  0.7116  0.7116  delta +0.00e+00
       886  0.8728  0.8728  delta +0.00e+00
      1281  0.8657  0.8657  delta +0.00e+00
      1301  0.7323  0.7323  delta +0.00e+00
      1336  0.7707  0.7707  delta +0.00e+00
      1379  0.8135  0.8135  delta +0.00e+00
      1478  0.8121  0.8121  delta +0.00e+00
      1574  0.8535  0.8534  delta +9.10e-05
      1583  0.8154  0.8154  delta +0.00e+00
      1649  0.8385  0.8385  delta +0.00e+00
      1664  0.8771  0.8771  delta +0.00e+00
      1925  0.7289  0.7289  delta +0.00e+00
      1934  0.8749  0.8749  delta +0.00e+00
      2009  0.4999  0.4999  delta +0.00e+00
      2046  0.8892  0.8892  delta +0.00e+00
      2118  0.6671  0.6671  delta +0.00e+00
      2159  -0.1329  -0.1329  delta +0.00e+00
      2349  0.5487  0.5487  delta +0.00e+00
      2409  0.7866  0.7866  delta +0.00e+00
      2435  0.8305  0.8305  delta +0.00e+00
      2492  0.9088  0.9088  delta +0.00e+00
      2617  0.7536  0.7536  delta +0.00e+00
      2632  0.8596  0.8596  delta +0.00e+00
      2750  0.8693  0.8693  delta +0.00e+00
      2957  0.7034  0.7034  delta +0.00e+00
      3098  0.8569  0.8569  delta +1.54e-05
      3154  0.8108  0.8108  delta +0.00e+00
      3375  0.8432  0.8432  delta +0.00e+00
      3696  0.8394  0.8394  delta +0.00e+00
      3780  0.6350  0.6350  delta +0.00e+00
      3914  0.7339  0.7339  delta +0.00e+00
      3916  0.8834  0.8834  delta +0.00e+00
      4187  0.8834  0.8830  delta +4.10e-04
      4282  0.7910  0.7910  delta +0.00e+00
      4420  0.8821  0.8821  delta +0.00e+00
      4436  0.8200  0.8200  delta +0.00e+00
      4635  0.8483  0.8483  delta +0.00e+00

LEAKAGE: draws that put the original word back
  swap       exact 0/25568 (0.00%), case-blind 0/25568 (0.00%)
  masked-LM  exact 33963/76704 (44.28%), case-blind 34578/76704 (45.08%)
  the swap excludes the original by construction, so a non-zero count here is a bug

ABOVE THE HARNESS FLOOR: draws with |effect| > 0.044 points
  swap, all draws                     0.773   n 25568
  masked-LM, all draws                0.420   n 76704
  masked-LM, non-identical draws      0.753   n 42126
  deletion, one per span              0.745   n 1598
  by masked-LM candidate depth:
    depth  1   all  0.374 (n 12784)   non-identical  0.751 (n 6264)
    depth  3   all  0.411 (n 12784)   non-identical  0.749 (n 6919)
    depth  5   all  0.421 (n 12784)   non-identical  0.747 (n 7074)
    depth 10   all  0.436 (n 12784)   non-identical  0.761 (n 7205)
    depth 25   all  0.436 (n 12784)   non-identical  0.756 (n 7264)
    depth 50   all  0.444 (n 12784)   non-identical  0.755 (n 7400)

EFFECT SIZE, POOLED (FVE points; the interval is a document-clustered standard error over 40 clusters)
  swap                               n  25568  mean|e|   0.312  median|e|   0.125  signed mean  +0.119 +- 0.020
  masked-LM, all                     n  76704  mean|e|   0.148  median|e|   0.017  signed mean  +0.030 +- 0.011
  masked-LM, non-identical           n  42126  mean|e|   0.266  median|e|   0.115  signed mean  +0.055 +- 0.020
  deletion                           n   1598  mean|e|   0.382  median|e|   0.106  signed mean  +0.178 +- 0.052
  masked-LM by depth:
  depth 1                            n  12784  mean|e|   0.128  median|e|   0.002  signed mean  +0.017 +- 0.012
  depth 3                            n  12784  mean|e|   0.147  median|e|   0.015  signed mean  +0.029 +- 0.013
  depth 5                            n  12784  mean|e|   0.146  median|e|   0.018  signed mean  +0.036 +- 0.010
  depth 10                           n  12784  mean|e|   0.155  median|e|   0.023  signed mean  +0.037 +- 0.010
  depth 25                           n  12784  mean|e|   0.152  median|e|   0.023  signed mean  +0.033 +- 0.013
  depth 50                           n  12784  mean|e|   0.156  median|e|   0.025  signed mean  +0.030 +- 0.012

EFFECT SIZE, PER DOCUMENT
     doc  base FVE |  swap n  mean|e|   signed | MLM mean|e|   signed | del mean|e|   signed
     267    0.8310 |     640    0.187   +0.126 |       0.036   +0.013 |       0.113   +0.011
     355    0.6850 |     640    0.571   +0.426 |       0.198   +0.094 |       0.601   +0.537
     376    0.8175 |     640    0.141   -0.005 |       0.086   +0.018 |       0.117   -0.031
     477    0.7116 |     640    0.274   +0.224 |       0.129   +0.059 |       0.299   +0.254
     886    0.8728 |     640    0.179   +0.048 |       0.083   +0.017 |       0.145   +0.003
    1281    0.8657 |     640    0.145   +0.013 |       0.123   +0.045 |       0.155   +0.020
    1301    0.7323 |     640    0.187   -0.065 |       0.137   -0.076 |       0.200   -0.148
    1336    0.7707 |     640    0.432   +0.264 |       0.226   +0.067 |       0.394   +0.236
    1379    0.8135 |     624    0.172   +0.068 |       0.093   +0.040 |       0.181   +0.047
    1478    0.8121 |     640    0.164   -0.006 |       0.126   +0.043 |       0.158   -0.007
    1574    0.8535 |     640    0.204   +0.137 |       0.067   +0.036 |       0.735   +0.675
    1583    0.8154 |     640    0.135   +0.020 |       0.073   +0.006 |       0.102   +0.006
    1649    0.8385 |     640    0.095   +0.022 |       0.085   +0.022 |       0.113   +0.040
    1664    0.8771 |     640    0.407   +0.374 |       0.069   +0.050 |       0.669   +0.597
    1925    0.7289 |     640    0.235   +0.115 |       0.081   +0.066 |       0.273   +0.199
    1934    0.8749 |     640    0.307   +0.283 |       0.094   +0.070 |       1.561   +1.520
    2009    0.4999 |     640    0.372   +0.207 |       0.201   +0.114 |       0.354   +0.161
    2046    0.8892 |     640    0.255   +0.198 |       0.053   +0.021 |       0.216   +0.150
    2118    0.6671 |     640    0.651   +0.162 |       0.231   -0.032 |       1.455   +0.818
    2159   -0.1329 |     640    2.029   +0.348 |       1.067   -0.207 |       1.877   -0.421
    2349    0.5487 |     640    0.522   +0.183 |       0.373   +0.306 |       0.479   +0.297
    2409    0.7866 |     640    0.128   +0.030 |       0.068   +0.040 |       0.090   +0.026
    2435    0.8305 |     640    0.145   +0.103 |       0.050   +0.018 |       0.201   +0.173
    2492    0.9088 |     640    0.066   +0.023 |       0.049   +0.019 |       0.041   +0.019
    2617    0.7536 |     640    0.502   +0.255 |       0.130   +0.025 |       0.647   +0.488
    2632    0.8596 |     640    0.160   +0.099 |       0.124   +0.078 |       0.202   +0.101
    2750    0.8693 |     640    0.183   +0.122 |       0.142   +0.121 |       0.181   +0.145
    2957    0.7034 |     640    0.298   +0.114 |       0.110   -0.012 |       0.293   +0.147
    3098    0.8569 |     640    0.189   -0.040 |       0.122   -0.027 |       0.160   -0.093
    3154    0.8108 |     640    0.205   +0.116 |       0.090   +0.070 |       0.182   +0.125
    3375    0.8432 |     640    0.241   +0.156 |       0.062   +0.023 |       0.315   +0.237
    3696    0.8394 |     640    0.452   +0.028 |       0.229   -0.040 |       0.333   -0.037
    3780    0.6350 |     640    0.504   -0.161 |       0.391   +0.005 |       0.479   -0.197
    3914    0.7339 |     640    0.326   -0.004 |       0.234   +0.043 |       0.333   -0.045
    3916    0.8834 |     640    0.269   +0.199 |       0.049   +0.021 |       0.330   +0.243
    4187    0.8834 |     640    0.383   +0.321 |       0.102   +0.047 |       0.636   +0.572
    4282    0.7910 |     640    0.167   +0.008 |       0.095   -0.033 |       0.189   +0.075
    4420    0.8821 |     640    0.123   +0.095 |       0.050   +0.033 |       0.084   +0.062
    4436    0.8200 |     624    0.274   +0.152 |       0.085   +0.022 |       0.222   +0.173
    4635    0.8483 |     640    0.198   +0.022 |       0.087   -0.007 |       0.169   -0.042

VARIANCE DECOMPOSITION of the per-draw effect (FVE points squared)
  one-way random effects: a draw's effect is a span mean plus draw noise. `between` is the
  variance of the true span means, `within` the variance of draws around their own span mean,
  and ICC = between / (between + within), the share of the spread that is real span-to-span
  difference rather than draw noise.
  swap, all spans                    spans 1598  draws/span  16.0  between   0.4076  within   0.2168  ICC  0.653  sd_between  0.638  sd_within  0.466
  swap, open class                   spans  744  draws/span  16.0  between   0.2670  within   0.2301  ICC  0.537  sd_between  0.517  sd_within  0.480
  swap, closed class                 spans  854  draws/span  16.0  between   0.5288  within   0.2052  ICC  0.720  sd_between  0.727  sd_within  0.453
  masked-LM, all spans (all depths)  spans 1598  draws/span  48.0  between   0.2132  within   0.1010  ICC  0.679  sd_between  0.462  sd_within  0.318
  masked-LM, open class              spans  744  draws/span  48.0  between   0.4045  within   0.1199  ICC  0.771  sd_between  0.636  sd_within  0.346
  masked-LM, closed class            spans  854  draws/span  48.0  between   0.0467  within   0.0845  ICC  0.356  sd_between  0.216  sd_within  0.291
  the masked-LM rows pool six candidate depths into the within term, so their `within`
  carries a real depth effect as well as draw noise. Per depth:
  masked-LM, depth 1                 spans 1598  draws/span   8.0  between   0.2850  within   0.0000  ICC  1.000  sd_between  0.534  sd_within  0.007
  masked-LM, depth 3                 spans 1598  draws/span   8.0  between   0.2219  within   0.0970  ICC  0.696  sd_between  0.471  sd_within  0.311
  masked-LM, depth 5                 spans 1598  draws/span   8.0  between   0.1699  within   0.1062  ICC  0.615  sd_between  0.412  sd_within  0.326
  masked-LM, depth 10                spans 1598  draws/span   8.0  between   0.2072  within   0.1023  ICC  0.669  sd_between  0.455  sd_within  0.320
  masked-LM, depth 25                spans 1598  draws/span   8.0  between   0.2224  within   0.1112  ICC  0.667  sd_between  0.472  sd_within  0.334
  masked-LM, depth 50                spans 1598  draws/span   8.0  between   0.2124  within   0.1493  ICC  0.587  sd_between  0.461  sd_within  0.386

WHERE TO SPEND A FIXED PASS BUDGET, for a CLASS-LEVEL mean
  N passes split as n_spans spans by d draws, so n_spans = N / d and
      SE^2 of the class mean = between / n_spans + within / (n_spans x d) = (between x d + within) / N
  which increases with d whenever between > 0. The table is that variance relative to d = 1,
  so a value of 3 means the same budget spent at that draw count gives three times the variance,
  and one draw on many spans is the better buy. This is about class-level means only: per-span
  ranking and pair interactions need the per-span mean itself to be precise, and there the
  draws are what buys the precision.
                                         d=1     d=2     d=4     d=8    d=12    d=16
  swap, all spans                       1.00    1.65    2.96    5.57    8.18   10.79
  swap, open class                      1.00    1.54    2.61    4.76    6.91    9.06
  swap, closed class                    1.00    1.72    3.16    6.04    8.93   11.81
  masked-LM, all spans                  1.00    1.68    3.04    5.75    8.46   11.18
  optimal draws per span for a class-level mean: 1, for every row above
  the 16 draws this run took cost 10.8x the variance of the same budget spent one draw per span
  equivalently, the 1598 spans x 16 draws here (25568 passes) are worth about 2369 spans at one draw each for a class mean

PRECISION OF A PER-SPAN SWAP MEAN against draw count
   draws   mean SE  median SE   spans  x floor
       4     0.086      0.040    1598      2.0
       8     0.066      0.033    1598      1.5
      12     0.055      0.028    1598      1.3
      16     0.049      0.025    1598      1.1
  the standard error is of one span's own mean, computed from that span's first m draws

RANK AGREEMENT over spans (Spearman rho on per-span mean effect)
      doc    n   swap~MLM   swap~del    MLM~del  swap split-half
      267   40      0.236      0.446      0.198            0.884
      355   40      0.577      0.725      0.621            0.897
      376   40      0.506      0.782      0.408            0.822
      477   40      0.395      0.592      0.536            0.865
      886   40      0.525      0.650      0.322            0.905
     1281   40      0.727      0.809      0.760            0.944
     1301   40      0.410      0.771      0.628            0.927
     1336   40      0.644      0.588      0.658            0.954
     1379   39      0.452      0.684      0.285            0.904
     1478   40      0.549      0.794      0.534            0.949
     1574   40      0.707      0.860      0.561            0.966
     1583   40      0.486      0.701      0.625            0.879
     1649   40      0.280      0.628      0.544            0.959
     1664   40      0.182      0.756      0.213            0.907
     1925   40      0.643      0.668      0.427            0.956
     1934   40      0.359      0.643      0.624            0.935
     2009   40      0.654      0.618      0.620            0.886
     2046   40      0.569      0.768      0.473            0.975
     2118   40      0.402      0.543      0.404            0.975
     2159   40      0.662      0.677      0.508            0.889
     2349   40      0.654      0.791      0.602            0.947
     2409   40      0.479      0.682      0.538            0.953
     2435   40      0.525      0.393      0.382            0.871
     2492   40      0.455      0.571      0.367            0.888
     2617   40      0.757      0.753      0.558            0.971
     2632   40      0.311      0.543      0.295            0.906
     2750   40      0.729      0.623      0.332            0.954
     2957   40      0.677      0.768      0.637            0.911
     3098   40      0.550      0.699      0.477            0.868
     3154   40      0.511      0.756      0.482            0.951
     3375   40      0.451      0.643      0.299            0.928
     3696   40      0.300      0.589      0.296            0.813
     3780   40      0.708      0.701      0.683            0.944
     3914   40      0.768      0.725      0.702            0.888
     3916   40      0.519      0.716      0.379            0.955
     4187   40      0.588      0.694      0.420            0.949
     4282   40      0.585      0.622      0.445            0.952
     4420   40      0.487      0.799      0.292            0.963
     4436   39      0.512      0.676      0.658            0.935
     4635   40      0.589      0.674      0.630            0.931
   pooled 1598      0.565      0.674      0.517            0.916
  the split half is draws 0 to 7 against draws 8 to 15 of the same spans, so it is the
  ceiling any other correlation with the swap could reach at this draw count

OPEN AGAINST CLOSED CLASS (open: ADJ ADV NOUN PROPN VERB)
  method                     class  spans   draws   mean|e|    signed  over floor
  swap                        open    744   11904     0.302    +0.086       0.807   (signed +- 0.020)
  swap                      closed    854   13664     0.321    +0.149       0.743   (signed +- 0.029)
  masked-LM                   open    744   35712     0.217    +0.036       0.587   (signed +- 0.025)
  masked-LM                 closed    854   40992     0.088    +0.025       0.275   (signed +- 0.007)
  deletion                    open    744     744     0.299    +0.077       0.769   (signed +- 0.033)
  deletion                  closed    854     854     0.455    +0.267       0.724   (signed +- 0.090)
  per part of speech, swap mean |effect| and masked-LM mean |effect|:
    pos       spans     swap      MLM  deletion  swap/MLM
    PRON        100    0.572    0.122     0.757      4.69
    NUM          46    0.555    0.279     1.626      1.99
    SCONJ        36    0.407    0.073     0.436      5.59
    AUX         124    0.337    0.065     0.696      5.19
    ADJ         180    0.324    0.239     0.251      1.36
    NOUN        161    0.317    0.235     0.294      1.35
    ADV         137    0.300    0.189     0.346      1.59
    VERB        145    0.288    0.173     0.264      1.67
    DET         164    0.279    0.054     0.210      5.14
    PROPN       121    0.266    0.242     0.366      1.10
    PART         72    0.256    0.114     0.211      2.24
    CCONJ       133    0.246    0.067     0.422      3.67
    ADP         175    0.214    0.069     0.171      3.11
    INTJ          4    0.114    0.240     0.065      0.47

PER-CLASS SPREAD OF THE PER-SPAN SWAP MEAN, against the number of draws it was built from
  a span's value at m is the mean of its first m draws. The mean column is over spans and barely
  moves with m; what moves is the spread, which is draw noise leaking into the per-span estimate.
  classes with fewer than 10 spans are pooled into one `other` row. sd is over spans, in FVE points.
  class                   spans             m=1             m=4             m=8            m=16
  ADJ                       180   +0.119/ 1.017   +0.128/ 0.731   +0.106/ 0.676   +0.095/ 0.605
  ADP                       175   +0.076/ 0.713   +0.056/ 0.500   +0.064/ 0.493   +0.059/ 0.484
  DET                       164   +0.151/ 0.896   +0.157/ 0.783   +0.156/ 0.771   +0.155/ 0.749
  NOUN                      161   +0.082/ 0.535   +0.042/ 0.640   +0.032/ 0.614   +0.052/ 0.567
  VERB                      145   +0.107/ 0.691   +0.081/ 0.548   +0.067/ 0.462   +0.076/ 0.496
  ADV                       137   +0.066/ 0.657   +0.084/ 0.535   +0.085/ 0.511   +0.088/ 0.464
  CCONJ                     133   +0.047/ 0.437   +0.052/ 0.428   +0.052/ 0.433   +0.052/ 0.427
  AUX                       124   +0.254/ 1.387   +0.214/ 1.019   +0.205/ 0.960   +0.195/ 0.932
  PROPN                     121   +0.137/ 0.517   +0.119/ 0.478   +0.117/ 0.471   +0.125/ 0.474
  PRON                      100   +0.095/ 1.213   +0.144/ 1.088   +0.196/ 1.062   +0.230/ 1.141
  PART                       72   +0.159/ 0.496   +0.161/ 0.498   +0.162/ 0.497   +0.163/ 0.494
  NUM                        46   +0.245/ 0.750   +0.286/ 0.971   +0.295/ 0.952   +0.287/ 0.917
  SCONJ                      36   +0.292/ 0.521   +0.325/ 0.511   +0.344/ 0.509   +0.326/ 0.505
  other (INTJ)                4   +0.069/ 0.178   +0.069/ 0.178   +0.065/ 0.178   +0.061/ 0.180
  each cell is mean/sd over the class's spans

PER-CLASS SIGNED MEAN EFFECT, swap against masked-LM against deletion
  FVE points lost, signed, so a negative number means the edit IMPROVED the reconstruction.
  Every interval is a standard error clustered on the document, over 40 clusters.
  class                   spans                  swap               MLM all            MLM non-id              deletion
  ADJ                       180      +0.095 +-  0.034      +0.073 +-  0.030      +0.095 +-  0.040      +0.039 +-  0.039
  ADP                       175      +0.059 +-  0.037      -0.004 +-  0.012      -0.011 +-  0.032      +0.003 +-  0.029
  DET                       164      +0.155 +-  0.065      +0.003 +-  0.007      +0.015 +-  0.024      +0.056 +-  0.040
  NOUN                      161      +0.052 +-  0.027      -0.057 +-  0.091      -0.079 +-  0.126      -0.019 +-  0.054
  VERB                      145      +0.076 +-  0.041      +0.045 +-  0.032      +0.059 +-  0.042      +0.048 +-  0.040
  ADV                       137      +0.088 +-  0.042      +0.042 +-  0.026      +0.057 +-  0.037      +0.170 +-  0.079
  CCONJ                     133      +0.052 +-  0.047      +0.010 +-  0.012      +0.029 +-  0.037      +0.181 +-  0.203
  AUX                       124      +0.195 +-  0.097      +0.029 +-  0.012      +0.075 +-  0.029      +0.559 +-  0.326
  PROPN                     121      +0.125 +-  0.046      +0.089 +-  0.051      +0.114 +-  0.066      +0.191 +-  0.085
  PRON                      100      +0.230 +-  0.115      +0.086 +-  0.045      +0.274 +-  0.143      +0.400 +-  0.231
  PART                       72      +0.163 +-  0.065      +0.039 +-  0.034      +0.070 +-  0.061      +0.112 +-  0.056
  NUM                        46      +0.287 +-  0.159      +0.098 +-  0.063      +0.155 +-  0.097      +1.399 +-  1.266
  SCONJ                      36      +0.326 +-  0.094      +0.014 +-  0.019      +0.047 +-  0.080      +0.341 +-  0.154
  other (INTJ)                4      +0.061 +-  0.090      +0.096 +-  0.155      +0.102 +-  0.163      +0.015 +-  0.040
  MLM non-id drops the masked-LM draws that put the original word back, which is 45% of them

DELETION AGAINST SWAP, per class
  the swap column is the per-span mean over its draws, the deletion column that span's single
  deletion variant, and the difference is paired within the span before it is averaged, so it
  is not the difference of the two columns' clustered intervals.
  class                   spans      swap  deletion         del - swap |   |swap| |deletion|       |del| - |swap|   ratio
  ADJ                       180    +0.095    +0.039     -0.056 +- 0.022 |    0.256      0.251       -0.005 +- 0.011    0.98
  ADP                       175    +0.059    +0.003     -0.056 +- 0.044 |    0.174      0.171       -0.003 +- 0.030    0.98
  DET                       164    +0.155    +0.056     -0.099 +- 0.056 |    0.248      0.210       -0.038 +- 0.029    0.85
  NOUN                      161    +0.052    -0.019     -0.071 +- 0.038 |    0.258      0.294       +0.037 +- 0.035    1.14
  VERB                      145    +0.076    +0.048     -0.028 +- 0.030 |    0.232      0.264       +0.032 +- 0.014    1.14
  ADV                       137    +0.088    +0.170     +0.082 +- 0.057 |    0.266      0.346       +0.080 +- 0.053    1.30
  CCONJ                     133    +0.052    +0.181     +0.128 +- 0.173 |    0.238      0.422       +0.184 +- 0.170    1.77
  AUX                       124    +0.195    +0.559     +0.364 +- 0.257 |    0.315      0.696       +0.381 +- 0.257    2.21
  PROPN                     121    +0.125    +0.191     +0.065 +- 0.051 |    0.241      0.366       +0.125 +- 0.043    1.52
  PRON                      100    +0.230    +0.400     +0.170 +- 0.140 |    0.500      0.757       +0.256 +- 0.123    1.51
  PART                       72    +0.163    +0.112     -0.052 +- 0.037 |    0.253      0.211       -0.042 +- 0.030    0.83
  NUM                        46    +0.287    +1.399     +1.112 +- 1.130 |    0.470      1.626       +1.156 +- 1.130    3.46
  SCONJ                      36    +0.326    +0.341     +0.014 +- 0.098 |    0.384      0.436       +0.053 +- 0.085    1.14
  other (INTJ)                4    +0.061    +0.015     -0.046 +- 0.076 |    0.111      0.065       -0.046 +- 0.073    0.59
  ALL                      1598    +0.119    +0.178     +0.059 +- 0.044 |    0.271      0.382       +0.112 +- 0.040    1.41
  Spearman rho between the per-span swap mean and the per-span deletion effect, pooled: 0.674

PER-CLASS VARIANCE DECOMPOSITION AND PASS BUDGET
  the same one-way random-effects split as above, and the same budget arithmetic, computed
  inside each word class. `between` is the variance of the true per-span means in that class,
  `within` the variance of draws around their own span mean, ICC the share of the spread that
  is real span-to-span difference. The d= columns are the variance of that class's mean at a
  fixed budget of reconstructor passes, relative to spending the same budget one draw per span,
  so a value of 3 means three times the variance and one draw on three times as many spans is
  the better buy. best d is the draw count that minimises it, which is 1 whenever between > 0.
  swap:
    class                   spans  draws   between   within    ICC    d=1    d=2    d=4    d=8   d=12   d=16  best d
    ADJ                       180   16.0    0.3514   0.2379  0.596   1.00   1.60   2.79   5.17   7.56   9.94       1
    ADP                       175   16.0    0.2288   0.0849  0.729   1.00   1.73   3.19   6.11   9.02  11.94       1
    DET                       164   16.0    0.5498   0.1860  0.747   1.00   1.75   3.24   6.23   9.22  12.21       1
    NOUN                      161   16.0    0.2987   0.3565  0.456   1.00   1.46   2.37   4.19   6.01   7.84       1
    VERB                      145   16.0    0.2228   0.3775  0.371   1.00   1.37   2.11   3.60   5.08   6.57       1
    ADV                       137   16.0    0.2098   0.0937  0.691   1.00   1.69   3.07   5.84   8.61  11.37       1
    CCONJ                     133   16.0    0.1818   0.0120  0.938   1.00   1.94   3.81   7.56  11.32  15.07       1
    AUX                       124   16.0    0.8486   0.3272  0.722   1.00   1.72   3.17   6.05   8.94  11.83       1
    PROPN                     121   16.0    0.2231   0.0283  0.888   1.00   1.89   3.66   7.21  10.76  14.31       1
    PRON                      100   16.0    1.2642   0.6059  0.676   1.00   1.68   3.03   5.73   8.44  11.14       1
    PART                       72   16.0    0.2441   0.0017  0.993   1.00   1.99   3.98   7.95  11.92  15.90       1
    NUM                        46   16.0    0.8077   0.5286  0.604   1.00   1.60   2.81   5.23   7.65  10.07       1
    SCONJ                      36   16.0    0.2508   0.0736  0.773   1.00   1.77   3.32   6.41   9.50  12.60       1
    other (INTJ)                4   16.0    0.0322   0.0003  0.992   1.00   1.99   3.97   7.94  11.91  15.87       1
    ALL                      1598   16.0    0.4076   0.2168  0.653   1.00   1.65   2.96   5.57   8.18  10.79       1
  masked-LM, all depths:
    class                   spans  draws   between   within    ICC    d=1    d=2    d=4    d=8   d=12   d=16  best d
    ADJ                       180   48.0    0.2856   0.1402  0.671   1.00   1.67   3.01   5.70   8.38  11.06       1
    ADP                       175   48.0    0.0221   0.0595  0.271   1.00   1.27   1.81   2.90   3.98   5.06       1
    DET                       164   48.0    0.0085   0.0245  0.257   1.00   1.26   1.77   2.80   3.83   4.86       1
    NOUN                      161   48.0    1.1972   0.1549  0.885   1.00   1.89   3.66   7.20  10.74  14.28       1
    VERB                      145   48.0    0.1066   0.0610  0.636   1.00   1.64   2.91   5.45   8.00  10.54       1
    ADV                       137   48.0    0.1168   0.1411  0.453   1.00   1.45   2.36   4.17   5.98   7.79       1
    CCONJ                     133   48.0    0.0163   0.0377  0.301   1.00   1.30   1.90   3.11   4.32   5.52       1
    AUX                       124   48.0    0.0182   0.0237  0.434   1.00   1.43   2.30   4.04   5.78   7.51       1
    PROPN                     121   48.0    0.2052   0.0901  0.695   1.00   1.69   3.08   5.86   8.64  11.42       1
    PRON                      100   48.0    0.1402   0.2295  0.379   1.00   1.38   2.14   3.65   5.17   6.69       1
    PART                       72   48.0    0.0848   0.0261  0.765   1.00   1.76   3.29   6.35   9.41  12.47       1
    NUM                        46   48.0    0.1914   0.5257  0.267   1.00   1.27   1.80   2.87   3.94   5.00       1
    SCONJ                      36   48.0    0.0159   0.0143  0.527   1.00   1.53   2.58   4.69   6.80   8.91       1
    other (INTJ)                4   48.0    0.0947   0.0627  0.602   1.00   1.60   2.80   5.21   7.62  10.02       1
    ALL                      1598   48.0    0.2132   0.1010  0.679   1.00   1.68   3.04   5.75   8.46  11.18       1

WORD-ORDER SHUFFLE, the document's own words permuted among its own slots
  run on 20 of the 40 documents
      doc  base FVE  shuffles  mean FVE      sd  points lost  vs deletion sum
      355    0.6850         4   -0.5927  0.1374      127.762           21.486
      376    0.8175         4   -0.0693  0.0490       88.683           -1.226
     1336    0.7707         4   -0.6567  0.1875      142.737            9.440
     1379    0.8135         4    0.0083  0.0351       80.520            1.818
     1649    0.8385         4   -0.0786  0.0825       91.708            1.599
     1664    0.8771         4   -0.2095  0.0643      108.653           23.881
     1925    0.7289         4   -0.8910  0.1244      161.994            7.949
     1934    0.8749         4   -0.3326  0.0504      120.750           60.785
     2159   -0.1329         4   -0.5209  0.2327       38.808          -16.837
     2349    0.5487         4   -0.0490  0.2494       59.767           11.877
     2435    0.8305         4   -0.4527  0.0645      128.321            6.904
     2492    0.9088         4   -0.0405  0.1046       94.930            0.742
     2750    0.8693         4   -0.7496  0.3139      161.895            5.812
     2957    0.7034         4   -0.1456  0.0921       84.898            5.865
     3098    0.8569         4   -0.3046  0.0983      116.153           -3.720
     3154    0.8108         4   -0.2014  0.0750      101.220            5.011
     3696    0.8394         4   -0.5318  0.0612      137.123           -1.474
     3914    0.7339         4    0.4192  0.0713       31.476           -1.812
     4187    0.8834         4   -0.3884  0.2198      127.184           22.899
     4635    0.8483         4   -0.7226  0.4492      157.095           -1.662
  the last column is the sum of the 40 single-word deletion effects in that document, which
  is not the same quantity: it is 40 spans, not all of them, and it ignores interaction

FIGURES
  swap_se_vs_draws.png: the standard error of one span's swap mean against how many draws
    it was built from, mean and median over the 1598 spans, with the 1/sqrt(m) reference
  swap_vs_mlm_scatter.png: per-span mean effect, corpus swap against masked-LM
    marginalisation on symmetric log axes, linear within +-0.1. Open class and closed class marked,
    y = x drawn
  budget_draws_vs_spans.png: variance of a class-level mean at a fixed pass budget as the
    budget is moved from spans onto draws
```
