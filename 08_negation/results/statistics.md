# Negation, run 10

documents 370, baseline FVE mean 0.762
instances 509 (309 negators, 200 insertions)

## Mean dFVE by condition (one value per instance, swap draws averaged)

| condition | mean dFVE [95% bootstrap] | mean abs dFVE |
|---|---|---|
| flip | -0.0049 [-0.0071, -0.0029] n 309 | 0.0077 |
| del_neg | -0.0007 [-0.0023, +0.0008] n 61 | 0.0035 |
| del_gov | -0.0085 [-0.0137, -0.0042] n 309 | 0.0115 |
| swap_gov | -0.0040 [-0.0062, -0.0022] n 308 | 0.0067 |
| ins_not | -0.0016 [-0.0024, -0.0009] n 200 | 0.0029 |
| ins_ctrl | -0.0006 [-0.0011, -0.0002] n 200 | 0.0015 |

## Paired contrasts over negator instances

| contrast | signed dFVE difference [95% bootstrap] | share > 0 | abs dFVE difference |
|---|---|---|---|
| flip − swap_gov | -0.0009 [-0.0036, +0.0018] n 308 | 0.46 | +0.0010 [-0.0014, +0.0036] n 308 |
| flip − del_gov | +0.0036 [-0.0008, +0.0088] n 309 | 0.50 | -0.0038 [-0.0090, +0.0004] n 309 |
| flip − del_neg | -0.0001 [-0.0008, +0.0006] n 61 | 0.48 | -0.0003 [-0.0010, +0.0003] n 61 |
| del_gov − swap_gov | -0.0045 [-0.0080, -0.0015] n 308 | 0.48 | +0.0048 [+0.0019, +0.0084] n 308 |

## Correlation of flip with swap_gov across negator instances

Pearson r +0.137, R squared 0.019, Spearman rho +0.383, n 308. The paired contrast above compares the two conditions in level; this compares them instance by instance.

## By negator type: mean dFVE per condition, and flip − swap_gov paired

| type | n | flip | del_neg | del_gov | swap_gov | flip − swap_gov |
|---|---|---|---|---|---|---|
| not | 140 | -0.0063 ± 0.0018 |  | -0.0063 ± 0.0024 | -0.0030 ± 0.0009 | -0.0034 [-0.0071, -0.0003] n 139 |
| n't | 96 | -0.0059 ± 0.0023 |  | -0.0167 ± 0.0070 | -0.0073 ± 0.0030 | +0.0014 [-0.0058, +0.0086] n 96 |
| no | 44 | -0.0003 ± 0.0009 | +0.0006 ± 0.0009 | -0.0021 ± 0.0006 | -0.0020 ± 0.0007 | +0.0017 [-0.0004, +0.0045] n 44 |
| without | 14 | -0.0010 ± 0.0007 | -0.0017 ± 0.0014 | -0.0011 ± 0.0010 | -0.0003 ± 0.0004 | -0.0007 [-0.0019, +0.0006] n 14 |
| never | 15 | -0.0024 ± 0.0018 | -0.0026 ± 0.0020 | -0.0032 ± 0.0022 | -0.0026 ± 0.0021 | +0.0002 [-0.0016, +0.0019] n 15 |

## Inside vs outside a quoted stretch, flip − swap_gov

| in_quote | n | flip | swap_gov | flip − swap_gov |
|---|---|---|---|---|
| 0 | 64 | -0.0045 | -0.0034 | -0.0012 [-0.0047, +0.0023] n 64 |
| 1 | 244 | -0.0050 | -0.0042 | -0.0008 [-0.0041, +0.0025] n 244 |

## Insertion: ' not' vs ' just' after the same auxiliary

ins_not -0.0016 [-0.0024, -0.0009] n 200
ins_ctrl -0.0006 [-0.0011, -0.0002] n 200
ins_not − ins_ctrl -0.0010 [-0.0016, -0.0005] n 200, share > 0 0.44
abs: |ins_not| − |ins_ctrl| +0.0014 [+0.0010, +0.0019] n 200

## Scale reference

harness floor from earlier runs is not recomputed here; per-instance flip |dFVE| median 0.0022, 90th percentile 0.0184

## Largest flip − swap_gov instances

| instance | doc | type | flip | swap_gov | context |
|---|---|---|---|---|---|
| 138 | 1925 | n't | -0.003 | -0.206 | nal phrase "If you can't see it, it doesn't need" is mid-sentence, ending on an  |
| 43 | 673 | n't | -0.180 | +0.017 | oise was in a set with a ring, but I wasn't compelled" implies a design decision |
| 272 | 4574 | not | -0.164 | +0.004 | "Super hero cover dait is iconic and cannot be passed on" implies a closing phra |
| 20 | 359 | n't | -0.007 | -0.126 |  similar described here: http://xxx (didn't read" — a forum citation with an ope |
| 9 | 197 | n't | +0.008 | -0.096 |  third parties... Health line, Inc. doesn't warrant or represent which" — the cl |
| 237 | 3928 | not | -0.075 | -0.022 | may not redistribute data. Such data is not warranted to be" ends with an incomp |
| 42 | 673 | n't | -0.018 | -0.068 |  belonged with the pawn chain, but I didn't feel compelled"), requiring a verb p |
| 114 | 1588 | not | +0.044 | -0.006 |  pay for overtime worked, e.g., "you cannot earn extra for overtime hours."  The |
| 19 | 334 | not | -0.083 | -0.036 | t factor. And I say this because you cannot talk about" — demands a direct objec |
| 251 | 4206 | not | +0.005 | -0.042 | ype and state insurance. </ex>  "you're not on the hook for" strongly signals th |
| 113 | 1588 | no | +0.027 | -0.019 | oyee distinction typically follows with no additional pay for overtime worked, e |
| 73 | 1030 | n't | -0.034 | +0.010 |  or override the system data. They couldn't change the name of the book" — a |

## Figures

negation_by_condition.png: mean FVE points lost per condition with a 95% bootstrap interval and the harness floor shaded
negation_flip_vs_swap.png: flip against corpus swap of the governed word, one point per negator instance
Both figures are in FVE points lost (100 x the drop in FVE), while the tables above are in raw dFVE
