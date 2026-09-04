```
SETUP
  database          /home/martin/phd/primary/projects/04_nla-span-interactions/public-repo/db/ffw_span-ablation_database.sqlite
  run               8, _scratch/matrix/tree_vs_linear.py, started 2026-09-03T08:19:33
  notes             Full pairwise interaction matrix. For each selected document every eligible word is corpus-swapped alone and every unordered pair of eligible words is swapped together, at eight draws, so the interaction e(a) + e(b) - e(both) is measurable for every cell of the n by n matrix rather than for a sample
  documents         5: 240, 621, 1664, 2592, 4126
  measured pairs    41392
  draws per pair    [(8, 41392)]
  baselines         5
  singles read      5160
  joint variants    331136
  parse             spacy-en_core_web_sm-3.8.0, 180257 head relations
  categories        41392 pairs carry a category recorded by the run under tree-vs-linear/all-pairs
  prompt shifts     4536 pairs had at least one draw whose splice moved the prompt length
  tree path lengths {1: 623, 2: 1087, 3: 1421, 4: 1658, 5: 1608, 6: 1398, 7: 1133, 8: 861} ... disconnected 29666
  harness floor     0.044 FVE points

  SIGN CONVENTION. interaction = e(a) + e(b) - e(both) in FVE points, where e is
  the drop in fraction of variance explained against the same document's unedited
  baseline, times 100. A POSITIVE interaction means the pair costs LESS than the
  sum of its two singles, so the two words carry overlapping information.

  CATEGORIES. A pair is on an ARC when the store holds a spaCy head relation between
  its two words in either direction, ADJACENT when their token indices differ by one
  and there is no arc, and OTHER otherwise. The three partition the matrix.

  TOKEN DISTANCE is the difference of the two token indices. TREE PATH LENGTH is the
  number of dependency edges between the two words, over every token of the document
  and not only the editable ones, so an arc is path length one.


SPLIT-HALF RELIABILITY OF THE INTERACTION
  The even draws and the odd draws give two independent estimates of each pair's
  interaction. Their covariance is the variance of the true pair means, which is the
  ceiling on any model of them; the rest of the observed spread is draw noise.

      doc   pairs  sd observed  var observed  var reliable   var noise  reliability   half r     S-B
      240    7626       0.0905       0.00819       0.00472     0.00347       0.5765   0.4050  0.5765
      621    8646       0.9499       0.90230       0.89792     0.00438       0.9951   0.9909  0.9954
     1664    9453       0.3303       0.10907       0.10566     0.00341       0.9687   0.9404  0.9693
     2592    7021       0.6573       0.43211       0.43036     0.00175       0.9960   0.9920  0.9960
     4126    8646       0.6582       0.43327       0.43296     0.00030       0.9993   0.9987  0.9993
   pooled   41392       0.6164       0.37992       0.37722     0.00269       0.9929   0.9860  0.9929

  reliability is the reliable variance as a fraction of the observed variance, and
  it is the largest R squared any model of these pair means could reach.


PRIMARY COMPARISON: TREE PATH LENGTH AGAINST TOKEN DISTANCE
  The registered question. Variance in the SIGNED interaction explained by the
  dependency path length between the two words, against the variance explained by
  their token distance. Each is a saturated categorical block fitted on its own, so
  neither is given the other's columns, and each is divided by the split-half
  ceiling because neither can explain the draw noise. A pair the parse leaves
  disconnected counts as a long tree path, not as a missing one.

  tree path length at most 2
        set   pairs  ceiling  R2 distance  of reliable   R2 tree  of reliable  tree - distance  of reliable
        240     325   0.7463       0.0417       0.0558    0.0202       0.0270          -0.0215      -0.0288
        621     344   0.2539       0.0393       0.1548    0.0052       0.0205          -0.0341      -0.1343
       1664     374   0.9191       0.0448       0.0488    0.0013       0.0014          -0.0435      -0.0474
       2592     320   0.9822       0.0646       0.0658    0.0005       0.0005          -0.0641      -0.0653
       4126     347   0.9996       0.0666       0.0666    0.0015       0.0015          -0.0651      -0.0651
     pooled    1710   0.9668       0.0166       0.0171    0.0012       0.0012          -0.0154      -0.0159
            29 distance columns, 1 tree columns, document dummies in the pooled row only

  tree path length over 2
        set   pairs  ceiling  R2 distance  of reliable   R2 tree  of reliable  tree - distance  of reliable
        240    7301   0.5574       0.0033       0.0059    0.0024       0.0043          -0.0009      -0.0016
        621    8302   0.9973       0.0031       0.0031    0.0002       0.0002          -0.0029      -0.0029
       1664    9079   0.9696       0.0020       0.0021    0.0012       0.0013          -0.0008      -0.0008
       2592    6701   0.9962       0.0011       0.0011    0.0006       0.0006          -0.0005      -0.0005
       4126    8299   0.9992       0.0001       0.0001    0.0001       0.0001          -0.0000      -0.0000
     pooled   39682   0.9940       0.0005       0.0005    0.0000       0.0000          -0.0004      -0.0004
            29 distance columns, 5 tree columns, document dummies in the pooled row only

  all pairs
        set   pairs  ceiling  R2 distance  of reliable   R2 tree  of reliable  tree - distance  of reliable
        240    7626   0.5765       0.0031       0.0053    0.0045       0.0078           0.0015       0.0026
        621    8646   0.9951       0.0032       0.0033    0.0003       0.0003          -0.0029      -0.0030
       1664    9453   0.9687       0.0048       0.0050    0.0036       0.0037          -0.0013      -0.0013
       2592    7021   0.9960       0.0018       0.0018    0.0007       0.0007          -0.0012      -0.0012
       4126    8646   0.9993       0.0002       0.0002    0.0006       0.0006           0.0004       0.0004
     pooled   41392   0.9929       0.0007       0.0007    0.0001       0.0001          -0.0006      -0.0006
            29 distance columns, 7 tree columns, document dummies in the pooled row only

  PERMUTATION NULL ON THE DIFFERENCE, AS A FRACTION OF RELIABLE VARIANCE
  Path length, arc status and dep label are shuffled together among the pairs that
  share a document and a token distance, at token distance 3 and beyond. The stratum
  is re-selected from the shuffled path each time, so a subset defined by path length
  is tested against the same procedure rather than against a fixed set of pairs.
  40441 of 41392 pairs qualify.

  stratum                         observed  null mean  null sd   excess       z  one-sided p
  tree path length at most 2       -0.0136    -0.0141   0.0061  +0.0005   +0.08       0.5389
  tree path length over 2          -0.0004    -0.0003   0.0000  -0.0001   -1.97       0.9980
  all pairs                        -0.0004    -0.0003   0.0000  -0.0001   -1.86       0.9920

  A positive excess means the tree beats token distance by more than a tree drawn at
  random from the pairs at the same distance would.


WHAT EXPLAINS THE INTERACTION: LINEAR DISTANCE, THEN THE TREE
  Nested least squares on the pair means. Token distance enters SATURATED, one column
  per exact distance up to 30 and one for everything beyond, so the linear model is
  given every chance before the tree is asked to add anything. The tree block is the
  dependency path length, one column per length up to 6, one beyond and one for
  a pair the parse leaves disconnected. The second variant adds the dep type of the
  arc and the ordered POS pair.

      doc   pairs  ceiling  R2 distance  of reliable    +tree  of reliable  +tree+labels  of reliable
      240    7626   0.5765       0.0031       0.0053   0.0047       0.0081        0.0989       0.1716
      621    8646   0.9951       0.0032       0.0033   0.0003       0.0004        0.0049       0.0049
     1664    9453   0.9687       0.0048       0.0050   0.0004       0.0004        0.0431       0.0445
     2592    7021   0.9960       0.0018       0.0018   0.0005       0.0005        0.0150       0.0151
     4126    8646   0.9993       0.0002       0.0002   0.0010       0.0010        0.0020       0.0020
   pooled   41392   0.9929       0.0007       0.0007   0.0000       0.0000        0.0044       0.0044

  pooled model: document dummies, then 29 distance columns, then 7 tree columns,
  then 15 dep columns and 126 POS pair columns.
  R squared of the document dummies alone 0.0033; with distance 0.0040;
  with the tree 0.0041; with the labels too 0.0084.
  The labels add 0.0044 over the tree alone, 0.0044 of the reliable variance.

  'of reliable' divides the R squared by the ceiling above, so it is the share of the
  variance that could be explained at all, rather than of the variance including noise.


PERMUTATION NULL ON THE TREE INCREMENT
  The statistic is the increase in R squared from adding the tree block to the
  saturated distance model. Path length, arc status and dep label are shuffled
  together among the pairs that share a document and a token distance, so the
  permuted tree is one the linear model could not tell from the real one.
  Restricted to token distance 3 and beyond: 40441 of 41392 pairs.

  strata 852, informative 350 (25593 pairs), 500 permutations

  observed increment           0.00008
  null mean                    0.00008
  null sd                      0.00004
  EXCESS OVER THE NULL         -0.00000
  excess as a share of the reliable variance   -0.00000
  z                            -0.05
  one-sided p                  0.5250

  A permutation always explains a little by chance, which is why the null mean is
  above zero. The excess is the part of the tree's contribution that survives holding
  document and token distance fixed.


MEAN ABSOLUTE INTERACTION PER CELL, BY CATEGORY
  The per-cell mean is the quantity the categories can be compared on: a mass share
  mostly reports how many cells a category has.

  set          cells   share  mean abs      se      mean      se  median abs  over floor
  arc            623  0.0151    0.1156  0.0154   +0.0194  0.0197      0.0552       0.583
  adjacent       201  0.0049    0.1321  0.0295   +0.0600  0.0454      0.0751       0.692
  other        40568  0.9801    0.0661  0.0085   -0.0284  0.0179      0.0350       0.396
  all          41392  1.0000    0.0671  0.0084   -0.0273  0.0179      0.0353       0.400

  Standard errors are clustered on document over 5 clusters, which is few.

  per document, mean absolute interaction per cell
      doc  words   cells        arc   adjacent      other  arc / other
      240    124    7626     0.1323     0.1358     0.0909        1.456
      621    132    8646     0.1256     0.0841     0.0807        1.556
     1664    138    9453     0.1533     0.2094     0.0480        3.193
     2592    119    7021     0.0985     0.1485     0.0646        1.525
     4126    132    8646     0.0664     0.0624     0.0505        1.316
   pooled          41392     0.1156     0.1321     0.0661        1.750


SHARE OF TOTAL ABSOLUTE INTERACTION MASS
  Fractions of the total absolute interaction over all C(n, 2) pairs of the document,
  beside the share of the cells each category holds. The two are equal when a
  category carries nothing the rest of the matrix does not.

      doc  total |i|      mass arc     cells arc mass adjacent cells adjacent    mass other   cells other
      240     699.19        0.0227        0.0157        0.0054        0.0037        0.9719        0.9806
      621     703.93        0.0228        0.0148        0.0053        0.0051        0.9719        0.9801
     1664     476.03        0.0425        0.0140        0.0229        0.0055        0.9346        0.9805
     2592     460.63        0.0246        0.0164        0.0126        0.0056        0.9628        0.9781
     4126     438.82        0.0194        0.0148        0.0054        0.0044        0.9752        0.9808
   pooled    2778.59        0.0259        0.0151        0.0096        0.0049        0.9645        0.9801


PERMUTATION NULL ON THE PER-CELL ARC INTERACTION
  The same shuffle, with a simpler statistic: the mean absolute interaction of an arc
  cell. Restricted to token distance 3 and beyond, 40441 of 41392 pairs.

  strata 852, informative 74 (6297 pairs, 207 of them arcs)

  observed arc cell mean       0.0714 FVE points
  null mean                    0.0746
  null sd                      0.0131
  EXCESS OVER THE NULL         -0.0033 points  (-4.4%)
  z                            -0.25
  one-sided p (arc higher)     0.5107
  two-sided p                  0.8155

      doc   pairs  arcs  observed  null mean  null sd   excess       z  one-sided p
      240    7456    37    0.0960     0.0967   0.0118  -0.0007   -0.06       0.4963
      621    8440    37    0.0863     0.0797   0.0248  +0.0065   +0.26       0.2939
     1664    9235    45    0.1060     0.0743   0.0275  +0.0318   +1.15       0.0735
     2592    6854    41    0.0358     0.0861   0.0518  -0.0503   -0.97       0.9745
     4126    8456    47    0.0381     0.0421   0.0045  -0.0040   -0.88       0.8046


MEAN ABSOLUTE INTERACTION AGAINST TOKEN DISTANCE
  distance       pairs  mean abs      se      mean   arcs  arc mean abs  rest mean abs   excess
  1                476    0.1292  0.0205   +0.0473    275        0.1271         0.1321  -0.0050
  2                475    0.1106  0.0147   +0.0148    141        0.1581         0.0905  +0.0676
  3                473    0.0795  0.0093   -0.0023     57        0.0732         0.0803  -0.0071
  4 to 5           930    0.0854  0.0112   -0.0254     60        0.0828         0.0856  -0.0028
  6 to 9          1823    0.0768  0.0149   -0.0235     44        0.0706         0.0769  -0.0063
  10 to 17        3510    0.0566  0.0100   -0.0117     20        0.0782         0.0565  +0.0217
  18 to 33        6454    0.0580  0.0134   -0.0219     17        0.0405         0.0580  -0.0175
  34+            27251    0.0673  0.0079   -0.0334      9        0.0308         0.0673  -0.0366


MEAN ABSOLUTE INTERACTION AGAINST TREE PATH LENGTH
  path           pairs  mean abs      se      mean  median token distance
  1                623    0.1156  0.0154   +0.0194                    2.0
  2               1087    0.1152  0.0095   -0.0223                    4.0
  3               1421    0.0778  0.0094   -0.0168                    8.0
  4               1658    0.0756  0.0120   -0.0248                   12.0
  5               1608    0.0615  0.0110   -0.0192                   16.0
  6               1398    0.0678  0.0129   -0.0285                   19.0
  7               1133    0.1017  0.0431   -0.0656                   23.0
  8                861    0.0559  0.0117   -0.0161                   26.0
  9                607    0.0500  0.0110   -0.0135                   29.0
  10               406    0.0581  0.0147   -0.0122                   29.0
  11               241    0.0597  0.0176   -0.0178                   31.0
  12               164    0.0563  0.0239   -0.0037                   30.0
  13               127    0.0465  0.0084   +0.0279                   32.0
  14               101    0.0484  0.0177   +0.0367                   36.0
  15                86    0.0428  0.0193   +0.0323                   35.0
  16                75    0.0404  0.0080   +0.0340                   35.0
  17                64    0.0376  0.0092   +0.0367                   33.0
  18                40    0.0399  0.0068   +0.0399                   29.5
  19                19    0.0338     nan   +0.0322                   28.0
  20                 6    0.0336     nan   +0.0336                   24.0
  disconnected   29666    0.0637  0.0082   -0.0301                   71.0


CONSISTENCY CHECK: MEAN INTERACTION BY ARC DEP TYPE
  05 measured a sample of arcs of nine dep types with the two words at least two
  tokens apart, against matched controls. These are the same quantity on the same
  scale over every arc of these documents, so the two are comparable type by type.

  dep         in 05   arcs      mean      se  mean abs  median dist
  amod           no     79   +0.0554  0.0273    0.1185          1.0
  compound       no     74   -0.0076  0.0144    0.0888          1.0
  prep           no     62   +0.0411  0.0205    0.1510          1.0
  det            no     61   +0.0205  0.0231    0.1645          2.0
  pobj           no     52   +0.0205  0.0211    0.1157          2.0
  conj          yes     43   -0.0348  0.0346    0.0763          4.0
  dobj          yes     42   +0.0310  0.0244    0.0932          2.5
  nsubj         yes     34   -0.0414  0.0313    0.1120          2.0
  advmod         no     23   +0.1477  0.1117    0.2310          1.0
  cc             no     22   +0.0339  0.0562    0.1196          1.0
  appos         yes     16   -0.0259  0.0358    0.0616          5.0
  nmod          yes     16   -0.0416  0.0383    0.0776          2.0
  acl            no     15   +0.0155  0.0325    0.0968          1.0
  attr          yes     12   +0.0078  0.0122    0.0551          2.5
  ccomp         yes     10   -0.0304  0.0124    0.0603          9.5
  advcl         yes      9   +0.0355  0.0299    0.0676         11.0
  poss          yes      4   +0.0766  0.0044    0.0984          2.0

  restricted to 05's nine types at distance two or more, which is 05's own arc set:
    157 arcs, mean -0.0142 +- 0.0245 points
  437 of 623 arcs carry a dep type 05 did not sample.


WHAT THE DRAWS SEPARATE FROM ZERO
      doc   pairs  |mean| >= 2 se  fraction  median se
      240    7626            2608    0.3420     0.0566
      621    8646            3555    0.4112     0.0207
     1664    9453            3090    0.3269     0.0163
     2592    7021            2027    0.2887     0.0251
     4126    8646            5268    0.6093     0.0154
   pooled   41392           16548    0.3998     0.0210

  over the cells that survive the two standard error test, the per-cell means are
    arc 0.2029, adjacent 0.2223, other 0.1231, and arcs are 0.0160 of them.


FIGURES
  interaction_matrix_doc240.png: 124 words, 7626 cells, 2608 outside two standard errors, 120 head arcs drawn above the axis, colour limit +-0.288 FVE points
  interaction_matrix_doc621.png: 132 words, 8646 cells, 3555 outside two standard errors, 128 head arcs drawn above the axis, colour limit +-0.562 FVE points
  interaction_matrix_doc1664.png: 138 words, 9453 cells, 3090 outside two standard errors, 132 head arcs drawn above the axis, colour limit +-0.401 FVE points
  interaction_matrix_doc2592.png: 119 words, 7021 cells, 2027 outside two standard errors, 115 head arcs drawn above the axis, colour limit +-0.394 FVE points
  interaction_matrix_doc4126.png: 132 words, 8646 cells, 5268 outside two standard errors, 128 head arcs drawn above the axis, colour limit +-0.110 FVE points
  Each figure is one document's interaction matrix in reading order, symmetric by
  construction, with the spaCy head arcs over the same words drawn above it.
```
