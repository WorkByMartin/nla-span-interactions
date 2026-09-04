```
SETUP
  database          /home/martin/phd/primary/projects/04_nla-span-interactions/public-repo/db/ffw_span-ablation_database.sqlite
  run               7, _scratch/pairs/pair_ablation.py, started 2026-09-03T06:36:40
  pair table        the run's config row, 2455 pairs, checked against pair-ablation/arc+control in relations
  documents         100
  pairs measured    2455  (1280 arc, 1175 control)
  draws per pair    8, [(8, 2455)]
  draw-level rows   19640
  singles read      29304
  joint variants    19640
  arc dep types     dobj, nsubj, conj, appos, nmod, advcl, ccomp, attr, poss
  minimum distance  2 tokens
  edit              corpus-swap/pos+len, the corpus swap of 04, matched on spaCy coarse POS,
                    Qwen token count and leading-space parity
  harness floor     0.044 FVE points

  SIGN CONVENTION. interaction = e(a) + e(b) - e(both) in FVE points, where e is
  the drop in fraction of variance explained against the same document's unedited
  baseline, times 100. A POSITIVE interaction means the pair costs LESS than the
  sum of its two singles, so the two words carry overlapping information. A
  NEGATIVE interaction means the pair costs MORE than the sum, so the two words
  are worth more together than apart.

  Every interaction is formed per draw against the singles that spliced the same
  strings at the same draw, then averaged inside the pair. Standard errors are
  clustered on document over 100 clusters unless the line says otherwise.


OVERALL, ARC AGAINST CONTROL
  set                      pairs      mean      se   median  mean abs  over floor
  arc pairs                 1280   -0.0171  0.0225  -0.0028    0.1368       0.571
  control pairs             1175   -0.0245  0.0190  -0.0026    0.1207       0.514
  all pairs                 2455   -0.0206  0.0193  -0.0026    0.1291       0.544

  paired within the match, arc minus its own control, over 1175 matched pairs
    +0.0056 +- 0.0176 points
    the pair is more sub-additive on an arc in 0.504 of matches

  105 sampled arcs never found a control and are in the arc rows but not in
  the paired comparison.


PER DEP TYPE
  dep       n arc  arc mean      se n ctrl  ctrl mean      se  arc - ctrl      se
  dobj        150   -0.0378  0.0209    146    +0.0016  0.0204     -0.0388  0.0284
  nsubj       150   -0.0245  0.0438    132    -0.0384  0.0407     +0.0014  0.0431
  conj        150   -0.0098  0.0217    148    +0.0034  0.0180     -0.0131  0.0208
  appos       150   +0.0055  0.0168    146    -0.0476  0.0312     +0.0541  0.0321
  nmod        150   -0.0987  0.0526    144    -0.0916  0.0910     -0.0106  0.1086
  advcl       150   +0.0030  0.0290    140    -0.0757  0.0510     +0.0762  0.0379
  ccomp       150   -0.0065  0.0578    139    -0.0134  0.0129     +0.0097  0.0684
  attr        111   -0.0739  0.0384     87    +0.0336  0.0218     -0.1115  0.0550
  poss        119   +0.0978  0.0596     93    +0.0564  0.0309     +0.0566  0.0549

  arc - ctrl is paired inside the match, so it uses only arcs that found a control.


REGRESSION, interaction ~ arc + distance + POS pair
  2455 pairs, 20 regressors, 100 clusters
  POS-pair reference cell NOUN-NOUN (646 pairs); combinations under 12 pairs are pooled into 'pos other'

  term                              beta       se       t
  intercept                      -0.0153   0.0286   -0.53
  arc                            +0.0023   0.0158   +0.15
  log2 distance                  -0.0130   0.0068   -1.91
  pos ADJ-VERB                   -0.0001   0.0658   -0.00
  pos AUX-NOUN                   +0.0247   0.0253   +0.98
  pos AUX-VERB                   +0.0393   0.0335   +1.17
  pos NOUN-ADJ                   -0.0031   0.0460   -0.07
  pos NOUN-AUX                   +0.0586   0.0278   +2.11
  pos NOUN-PROPN                 +0.0320   0.0310   +1.03
  pos NOUN-VERB                  +0.0072   0.0443   +0.16
  pos PRON-NOUN                  +0.1528   0.1487   +1.03
  pos PRON-VERB                  +0.0627   0.0374   +1.68
  pos PROPN-NOUN                 +0.0409   0.0321   +1.27
  pos PROPN-PROPN                -0.0877   0.0910   -0.96
  pos PROPN-VERB                 +0.0152   0.0356   +0.43
  pos VERB-AUX                   +0.1077   0.0544   +1.98
  pos VERB-NOUN                  -0.0021   0.0426   -0.05
  pos VERB-PROPN                 +0.0033   0.0523   +0.06
  pos VERB-VERB                  +0.0485   0.0349   +1.39
  pos other                      +0.0215   0.0564   +0.38

  'arc' is the coefficient the experiment is about: the extra sub-additivity of a
  pair on a dependency arc, holding token distance and the POS combination fixed.

  without the POS block: arc +0.0066 +- 0.0164, log2 distance -0.0055 +- 0.0050


WITHIN-DOCUMENT PERMUTATION NULL
  the arc label is shuffled among the pairs that share a document and a token
  distance, 2000 times. Strata 802, of which 548 hold both labels and so carry information
  (2150 pairs).

  observed arc minus control   +0.0074 points
  null mean                    -0.0086
  null sd                      0.0171
  two-sided p                  0.3625


INTERACTION AGAINST THE SIZE OF THE SINGLES
  A large interaction on two words that barely matter is not the same finding as
  the same number on two words that matter a lot, so the ratio is reported beside
  the difference. sum is e(a) + e(b), both in FVE points lost.

  set               mean e(a)  mean e(b)  mean sum  mean e(both)  mean inter  inter/sum  median |i|/|sum|
  arc                  0.1906     0.1509    0.3414        0.3585     -0.0171     -0.050             0.201
  control              0.1623     0.2337    0.3960        0.4205     -0.0245     -0.062             0.180

  set               inter > 0  |inter| over floor  |inter| > smaller single
  arc                   0.484               0.571                     0.439
  control               0.483               0.514                     0.407

  A positive fraction near 0.5 with a mean near zero would mean the pair is simply
  additive. The direction of the mean is the claim; the ratio says how much of the
  singles' cost the overlap accounts for.


INTERACTION AGAINST TOKEN DISTANCE
  distance      n arc  arc mean      se n ctrl  ctrl mean      se
  2               395   -0.0239  0.0247    203    +0.0207  0.0195
  3               252   -0.0267  0.0390    308    -0.0251  0.0354
  4 to 5          248   -0.0034  0.0513    276    -0.0396  0.0494
  6 to 9          163   -0.0121  0.0168    177    -0.0099  0.0131
  10 to 17        127   -0.0008  0.0143    117    -0.0262  0.0143
  18+              95   -0.0294  0.0213     94    -0.1008  0.0814


CONTROL MATCH QUALITY
  quality               n  ctrl mean      se  arc - ctrl      se
  exact               726    -0.0408  0.0241     +0.0097  0.0229
  dist+-1             284    -0.0060  0.0225     +0.0311  0.0215
  other-doc           123    +0.0063  0.0139     +0.0121  0.0434
  other-doc+-1         42    +0.0426  0.0356     -0.2572  0.2112

  Exact means the same document, the same ordered POS combination and the same
  token distance. The other rows relaxed the distance by one token, or moved to a
  different document of the same domain.


DRAW-LEVEL SPREAD
  within-pair sd of the interaction over draws: mean 0.1637, median 0.0857 points
  so the standard error of one pair's mean at 8 draws is about 0.0579 points, against a harness floor of 0.044
  spread of the pair means: sd 0.4743 points over 2455 pairs


PER DOCUMENT
     doc  pairs  arc mean  ctrl mean  arc - ctrl
      11     37   +0.0552    +0.0444     +0.0107
      82     28   +0.0607    +0.0510     +0.0097
     240     21   +0.0406    +0.0795     -0.0389
     267     42   +0.1039    +0.0117     +0.0922
     323     43   +0.1084    -0.0075     +0.1159
     324     20   +0.0393    +0.0268     +0.0124
     342     22   -0.0663    -0.0488     -0.0175
     355     39   +0.5331    +0.2747     +0.2585
     362     25   +0.0027    +0.0369     -0.0342
     376     25   -0.0412    +0.0000     -0.0412
     414     26   -0.0174    -0.0310     +0.0135
     442     23   +0.1015    +0.1135     -0.0121
     447     25   -0.1821    -0.2753     +0.0932
     452     25   +0.0321    +0.0638     -0.0317
     477     28   +0.0718    +0.0711     +0.0007
     612     29   -0.0468    -0.0312     -0.0156
     886     31   +0.0037    -0.0081     +0.0118
     920     18   +0.0444    +0.0330     +0.0114
     965     22   +0.0290    -0.0432     +0.0722
    1045     12   -0.0026    -0.0113     +0.0087
    1281     25   -0.0100    +0.0055     -0.0155
    1301     19   -0.0354    -0.0060     -0.0295
    1336     27   -0.0116    +0.0603     -0.0719
    1342     27   -0.0373    -0.0557     +0.0184
    1349     11   -0.0289    -0.0082     -0.0207
    1379     15   -0.1449    +0.0031     -0.1479
    1390     20   -0.0777    -0.0193     -0.0585
    1478     25   +0.0547    +0.0698     -0.0150
    1486     12   +0.0130    -0.0872     +0.1002
    1574     26   +0.0022    +0.0999     -0.0977
    1583     29   -0.0187    -0.0013     -0.0174
    1649     25   -0.0104    -0.0047     -0.0057
    1664     46   +0.0061    -0.0139     +0.0201
    1717     17   -0.0195    +0.0319     -0.0513
    1738     23   -0.0508    -0.0313     -0.0195
    1780     20   -0.0017    +0.0117     -0.0135
    1854     29   +0.0890    +0.0150     +0.0740
    1874     32   -0.0239    -0.0331     +0.0092
    1925     26   +0.2034    +0.0247     +0.1787
    1934     28   -0.0047    -0.0070     +0.0023
    1954     12   -0.0070    -0.0029     -0.0041
    2009     36   -0.0309    -0.0776     +0.0467
    2046     31   +0.0345    -0.0090     +0.0435
    2068     23   -0.0826    -0.1725     +0.0900
    2118     29   -0.0590    +0.1474     -0.2064
    2159     36   -1.1225    -0.9186     -0.2039
    2349     37   -0.3799    -0.2075     -0.1724
    2409     14   -0.0007    +0.0105     -0.0111
    2416     20   +0.0517    +0.0639     -0.0122
    2428     24   +0.0144    -0.0321     +0.0465
    2435     33   -0.0268    -0.0072     -0.0195
    2492     32   +0.1528    +0.0049     +0.1480
    2533     11   -0.0124    +0.0392     -0.0517
    2592     21   +0.1374    +0.1070     +0.0304
    2617     23   +0.0406    -0.0133     +0.0539
    2632     23   +0.0454    +0.0550     -0.0096
    2748     27   -0.0016    -0.0206     +0.0190
    2750     28   -0.0411    -0.0448     +0.0036
    2765     14   -0.2480    -0.0092     -0.2389
    2810     35   -0.0630    -0.1002     +0.0372
    2852     17   -0.1191    -0.1220     +0.0029
    2957     18   -0.0524    -0.0282     -0.0242
    2994     20   -0.0333    -0.0537     +0.0204
    3098     32   -0.0730    -0.0668     -0.0062
    3129     20   +0.0977    +0.0966     +0.0011
    3154     17   +0.0447    +0.0825     -0.0378
    3205     25   +0.0747    +0.0258     +0.0488
    3207     15   +0.0049    -0.0187     +0.0236
    3257     32   +0.1808    +0.0473     +0.1334
    3259     31   +0.0483    +0.0530     -0.0047
    3268     26   +0.0202    -0.0307     +0.0510
    3325     19   -0.0749    -0.0279     -0.0470
    3375     30   +0.0469    +0.0538     -0.0070
    3468     24   -0.7264    +0.0413     -0.7677
    3665     32   +0.0777    +0.0506     +0.0270
    3696     27   +0.0290    +0.0246     +0.0044
    3726     15   -0.0495    -0.0271     -0.0224
    3780     20   +0.1130    -0.1039     +0.2169
    3914     19   +0.0417    +0.0425     -0.0008
    3916     17   +0.0310    +0.1922     -0.1612
    3952     36   +0.0254    +0.0122     +0.0132
    3973     15   -0.0017    -1.8467     +1.8450
    4000     13   +0.0015    +0.2119     -0.2103
    4040     14   -0.0485    -0.0425     -0.0060
    4187     24   +0.0368    -0.0300     +0.0669
    4197     25   +0.0070    +0.0791     -0.0721
    4261     22   +0.0096    -0.0267     +0.0363
    4282     22   -0.0184    -0.0179     -0.0005
    4420     27   +0.0100    -0.0062     +0.0162
    4436     20   -0.0600    -0.1424     +0.0824
    4449     18   +0.0196    +0.0627     -0.0430
    4538     24   +0.0191    -0.0430     +0.0621
    4569     28   +0.0068    +0.1097     -0.1029
    4613     24   -0.1681    -0.2865     +0.1184
    4635     33   -0.0003    -0.0609     +0.0606
    4725     25   -0.0109    -0.0666     +0.0557
    4733     29   -0.2706    -0.1726     -0.0980
    4863     25   +0.0151    -0.0322     +0.0473
    4899     29   +0.1100    -0.0282     +0.1382
    4929     14   +0.0346    +0.0186     +0.0160


FIGURES
  pair_arc_vs_control.png: interaction on an arc against its matched control, one
    row per dep type, standard errors clustered on document, harness floor marked
  pair_interaction_vs_distance.png: interaction against token distance on a log
    axis, arcs and controls binned separately, every pair drawn behind
```
