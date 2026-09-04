# Removal curves, run 9

documents 20: [240, 358, 621, 1455, 1535, 1633, 1664, 1929, 1965, 1983, 2112, 2193, 2432, 2592, 2617, 2815, 3616, 3664, 4126, 4979]

baseline FVE mean 0.765 (min 0.136, max 0.913)

## Endpoints

| | mean FVE | sd |
|---|---|---|
| intact | 0.765 | 0.164 |
| every eligible word deleted | -0.871 | 0.296 |
| every eligible word swapped (mean over perms) | -1.649 | 0.448 |
| every eligible word replaced by filler | -0.812 | 0.357 |

## Mean dFVE by fraction of eligible words removed

| fraction | random deletion | random swap | random filler | front deletion | back deletion | front filler | back filler |
|---|---|---|---|---|---|---|---|
| 0.00 | +0.000 ± 0.000 | +0.000 ± 0.000 | +0.000 ± 0.000 | +0.000 ± 0.000 | +0.000 ± 0.000 | +0.000 ± 0.000 | +0.000 ± 0.000 |
| 0.10 | -0.028 ± 0.005 | -0.019 ± 0.003 | -0.047 ± 0.010 | -0.010 ± 0.004 | -0.042 ± 0.010 | -0.010 ± 0.004 | -0.029 ± 0.006 |
| 0.20 | -0.083 ± 0.011 | -0.062 ± 0.006 | -0.179 ± 0.021 | -0.029 ± 0.009 | -0.037 ± 0.007 | -0.029 ± 0.008 | -0.038 ± 0.007 |
| 0.30 | -0.151 ± 0.015 | -0.170 ± 0.018 | -0.322 ± 0.028 | -0.053 ± 0.007 | -0.099 ± 0.044 | -0.060 ± 0.010 | -0.128 ± 0.069 |
| 0.40 | -0.249 ± 0.019 | -0.422 ± 0.035 | -0.492 ± 0.031 | -0.072 ± 0.009 | -0.230 ± 0.060 | -0.090 ± 0.015 | -0.527 ± 0.131 |
| 0.50 | -0.387 ± 0.026 | -0.840 ± 0.059 | -0.668 ± 0.034 | -0.139 ± 0.044 | -0.385 ± 0.081 | -0.188 ± 0.049 | -0.765 ± 0.120 |
| 0.60 | -0.585 ± 0.033 | -1.477 ± 0.071 | -0.921 ± 0.034 | -0.180 ± 0.043 | -0.998 ± 0.098 | -0.314 ± 0.062 | -1.354 ± 0.095 |
| 0.70 | -0.871 ± 0.033 | -2.009 ± 0.065 | -1.177 ± 0.030 | -0.243 ± 0.061 | -1.265 ± 0.109 | -0.426 ± 0.080 | -1.453 ± 0.068 |
| 0.80 | -1.143 ± 0.030 | -2.297 ± 0.055 | -1.375 ± 0.025 | -0.319 ± 0.065 | -1.362 ± 0.113 | -0.609 ± 0.081 | -1.470 ± 0.070 |
| 0.90 | -1.437 ± 0.024 | -2.413 ± 0.051 | -1.517 ± 0.021 | -0.769 ± 0.085 | -1.497 ± 0.097 | -1.069 ± 0.075 | -1.520 ± 0.067 |
| 1.00 | -1.636 ± 0.022 | -2.414 ± 0.049 | -1.577 ± 0.020 | -1.637 ± 0.062 | -1.637 ± 0.061 | -1.577 ± 0.056 | -1.577 ± 0.056 |

## Concavity: signed area between curve and chord (positive = early removals cheaper than proportional)

| curve | mean area | se | n curves | docs with area > 0 | mean endpoint dFVE |
|---|---|---|---|---|---|
| random deletion | +0.2436 | 0.0124 | 160 | 0.91 | -1.636 |
| random swap | +0.1139 | 0.0236 | 160 | 0.66 | -2.414 |
| random filler | +0.0400 | 0.0156 | 160 | 0.60 | -1.577 |
| front deletion | +0.5651 | 0.0394 | 20 | 1.00 | -1.637 |
| back deletion | +0.1435 | 0.0324 | 20 | 0.85 | -1.637 |
| front filler | +0.4328 | 0.0301 | 20 | 1.00 | -1.577 |
| back filler | -0.0298 | 0.0359 | 20 | 0.45 | -1.577 |

## Front vs back truncation, dFVE at fixed fractions removed

| primitive | fraction | front | back | front − back | se | docs with front < back |
|---|---|---|---|---|---|---|
| deletion | 0.25 | -0.052 | -0.088 | +0.036 | 0.049 | 0.55 |
| deletion | 0.50 | -0.139 | -0.385 | +0.246 | 0.100 | 0.25 |
| deletion | 0.75 | -0.264 | -1.316 | +1.052 | 0.143 | 0.00 |
| filler | 0.25 | -0.055 | -0.117 | +0.062 | 0.076 | 0.60 |
| filler | 0.50 | -0.188 | -0.765 | +0.577 | 0.149 | 0.25 |
| filler | 0.75 | -0.467 | -1.459 | +0.992 | 0.111 | 0.00 |

## Random order, primitive differences in mean dFVE

| fraction | deletion − swap | deletion − filler | swap − filler |
|---|---|---|---|
| 0.00 | +0.000 | +0.000 | +0.000 |
| 0.20 | -0.022 | +0.096 | +0.117 |
| 0.40 | +0.172 | +0.242 | +0.070 |
| 0.60 | +0.892 | +0.336 | -0.556 |
| 0.80 | +1.154 | +0.232 | -0.921 |
| 1.00 | +0.779 | -0.058 | -0.837 |

## Random filler, first five steps (does FVE rise at first?)

| step | mean dFVE | se | n |
|---|---|---|---|
| 1 | -0.0021 | 0.0010 | 160 |
| 2 | -0.0029 | 0.0011 | 160 |
| 3 | -0.0039 | 0.0012 | 160 |
| 4 | -0.0057 | 0.0015 | 160 |
| 5 | -0.0081 | 0.0023 | 160 |

## Step-1 swap variants that repeat a run 8 single (same span, same substitute)

n 76, correlation 0.997, mean |difference| 0.00022, max |difference| 0.00210

## First removal (one word), by primitive

random deletion: mean dFVE -0.0029, mean |dFVE| 0.0040, n 160
random swap: mean dFVE -0.0015, mean |dFVE| 0.0031, n 160
random filler: mean dFVE -0.0021, mean |dFVE| 0.0036, n 160

## Whole vs sum of parts

| doc | base FVE | end FVE (deleted) | drop | sum of step-1 deletion dFVE over the doc's random curves ÷ perms x n |
|---|---|---|---|---|
| 240 | 0.761 | -1.488 | -2.248 | -0.024 |
| 358 | 0.818 | -0.662 | -1.479 | -2.904 |
| 621 | 0.813 | -0.865 | -1.678 | -0.618 |
| 1455 | 0.715 | -0.652 | -1.367 | +0.037 |
| 1535 | 0.701 | -0.702 | -1.403 | +0.018 |
| 1633 | 0.912 | -0.880 | -1.792 | -0.055 |
| 1664 | 0.877 | -0.815 | -1.692 | -0.337 |
| 1929 | 0.803 | -0.283 | -1.086 | +0.027 |
| 1965 | 0.913 | -0.828 | -1.741 | -0.400 |
| 1983 | 0.809 | -0.743 | -1.551 | -0.063 |
| 2112 | 0.671 | -0.440 | -1.111 | -0.130 |
| 2193 | 0.723 | -1.132 | -1.855 | -0.260 |
| 2432 | 0.136 | -1.366 | -1.502 | -0.253 |
| 2592 | 0.774 | -1.085 | -1.859 | -0.762 |
| 2617 | 0.754 | -1.001 | -1.755 | -0.008 |
| 2815 | 0.834 | -0.672 | -1.506 | +0.038 |
| 3616 | 0.729 | -1.230 | -1.960 | -0.651 |
| 3664 | 0.798 | -0.654 | -1.452 | -0.333 |
| 4126 | 0.864 | -0.996 | -1.860 | +0.003 |
| 4979 | 0.895 | -0.923 | -1.817 | -0.416 |

## Figures

removal_curves.png: FVE against fraction removed under random order, one mean curve per primitive with a 95% confidence band for the mean over the 20 documents (each document's permutations averaged first); the dotted horizontal line is the mean FVE with every eligible word deleted
truncation_front_vs_back.png: FVE against fraction removed under deletion, truncating from the front and from the back, one faint line per document and the mean over the 20 documents in bold
removal_first_steps.png: the same seven curve types over the first ten words removed, in FVE points lost (100 x the drop in FVE), mean over the 20 documents
