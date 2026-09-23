# run 11: arc minus matched control, per relation type

19626 pairs (9813 arcs, 9813 controls) over 857 documents; 857 baselines. interaction = e(a) + e(b) - e(both), FVE points; positive means the pair costs less than the sum of its singles. Standard errors clustered on document. Pooled fit: 19626 rows, 89 columns. Floor 0.044 FVE points. 95% intervals are per type, unadjusted for 25 comparisons; the `bonf` column says whether the Bonferroni interval (z 3.0233) excludes zero.

Control match quality: exact 7607, dist+-1 1544, no-position 368, no-quote 150, no-qwen 144; exact 77.5% of 9813 controls. Adjacent pairs (distance 1): 37.0% of arcs, 22.0% of controls.

```
type            n docs     arc    ctrl     raw     se     adj     se        adj 95% CI bonf exact n raw exact adjacent% arc/ctrl
iobj           31   28  -0.056  +0.345  -0.402  0.375  -0.358  0.369 [ -1.082, +0.366]   no      25    -0.481    90/74     underpowered
cop           435  331  -0.063  +0.071  -0.134  0.117  -0.182  0.135 [ -0.448, +0.083]   no     349    -0.002    40/23   
det           450  337  +0.033  +0.063  -0.029  0.097  -0.137  0.100 [ -0.333, +0.059]   no     329    +0.028    43/22   
parataxis      24   24  -0.042  +0.181  -0.222  0.166  -0.117  0.141 [ -0.394, +0.160]   no      22    -0.242     0/0      underpowered
mark          450  315  -0.134  +0.009  -0.143  0.116  -0.096  0.127 [ -0.345, +0.153]   no     297    -0.102    64/33   
nmod          450  331  +0.051  +0.061  -0.010  0.035  -0.062  0.048 [ -0.156, +0.031]   no     401    -0.028     0/0    
obj           450  326  +0.010  +0.022  -0.012  0.020  -0.061  0.050 [ -0.159, +0.037]   no     347    -0.015    22/8    
appos         326  268  +0.049  -0.001  +0.050  0.029  -0.048  0.046 [ -0.139, +0.043]   no     212    +0.056     1/2    
nmod:poss     450  283  +0.041  +0.005  +0.036  0.052  -0.040  0.066 [ -0.170, +0.090]   no     334    +0.073    37/18   
compound      450  321  -0.049  -0.031  -0.018  0.065  -0.037  0.069 [ -0.173, +0.099]   no     237    -0.004    91/47   
amod          450  321  +0.020  -0.024  +0.044  0.130  -0.002  0.122 [ -0.241, +0.237]   no     305    +0.130    78/49   
conj          450  343  +0.071  +0.031  +0.040  0.047  +0.016  0.044 [ -0.071, +0.103]   no     381    +0.029     0/0    
advcl         379  305  +0.027  +0.021  +0.005  0.032  +0.069  0.074 [ -0.075, +0.214]   no     360    +0.010     0/0    
aux           450  302  -0.002  +0.012  -0.015  0.077  +0.071  0.103 [ -0.130, +0.273]   no     369    -0.047    60/46   
nsubj:pass    424  301  +0.097  +0.073  +0.024  0.044  +0.080  0.066 [ -0.050, +0.210]   no     364    +0.029     0/0    
acl:relcl     275  202  +0.095  +0.054  +0.040  0.061  +0.091  0.083 [ -0.072, +0.255]   no     260    +0.031     0/0    
acl           398  303  +0.018  +0.010  +0.008  0.022  +0.117  0.056 [ +0.007, +0.228]   no     282    +0.021    74/48   
ccomp         383  284  +0.150  +0.039  +0.111  0.080  +0.124  0.109 [ -0.089, +0.337]   no     366    +0.111     0/0    
case          450  329  +0.134  -0.079  +0.213  0.072  +0.154  0.081 [ -0.004, +0.313]   no     367    +0.240    23/12   
obl           450  315  +0.305  +0.027  +0.278  0.160  +0.155  0.143 [ -0.125, +0.435]   no     418    +0.275     0/0    
aux:pass      450  321  +0.196  +0.138  +0.058  0.102  +0.163  0.141 [ -0.113, +0.439]   no     328    +0.044    87/62   
nsubj         450  330  +0.178  +0.078  +0.100  0.102  +0.174  0.112 [ -0.046, +0.394]   no     319    +0.009    48/25   
xcomp         388  274  +0.192  +0.020  +0.172  0.087  +0.180  0.103 [ -0.023, +0.382]   no     288    +0.174    22/8    
advmod        450  335  +0.106  -0.022  +0.128  0.061  +0.249  0.086 [ +0.081, +0.418]   no     328    +0.114    76/53   
cc            450  332  +0.386  -0.118  +0.505  0.179  +0.488  0.165 [ +0.164, +0.811]   no     319    +0.575    47/26   
```

Adjusted: 95% interval excludes zero for 3 of 25 (acl, advmod, cc); of those, interval entirely beyond the floor for 2 (advmod, cc); Bonferroni interval excludes zero for 0 (none).

Adjusted, winsorised: 95% interval excludes zero for 8 of 25 (advcl, advmod, amod, aux:pass, case, cc, nsubj, xcomp); of those, interval entirely beyond the floor for 0 (none); Bonferroni interval excludes zero for 2 (advmod, nsubj).

## robustness to the tails

Pooled interaction 1st and 99th percentiles: -1.074 and +2.232 FVE points, range -43.542 to +49.954. Winsorised columns clip every pair's interaction to that band before differencing or fitting.

```
type         raw mean  median    wins     se     adj adj wins     se   adj wins 95% CI
iobj           -0.402  -0.057  -0.100  0.082  -0.358   -0.047  0.084 [ -0.212, +0.119]  underpowered
cop            -0.134  -0.007  -0.012  0.019  -0.182   -0.034  0.024 [ -0.082, +0.014]
det            -0.029  +0.004  +0.024  0.023  -0.137   -0.002  0.025 [ -0.052, +0.048]
parataxis      -0.222  -0.039  -0.148  0.095  -0.117   -0.081  0.081 [ -0.240, +0.079]  underpowered
mark           -0.143  +0.013  +0.010  0.026  -0.096   +0.012  0.039 [ -0.065, +0.089]
nmod           -0.010  -0.000  +0.002  0.016  -0.062   -0.015  0.018 [ -0.049, +0.020]
obj            -0.012  -0.009  -0.012  0.016  -0.061   -0.007  0.020 [ -0.045, +0.032]
appos          +0.050  +0.008  +0.030  0.018  -0.048   -0.009  0.020 [ -0.048, +0.031]
nmod:poss      +0.036  -0.011  +0.009  0.023  -0.040   -0.017  0.024 [ -0.064, +0.029]
compound       -0.018  +0.013  +0.007  0.020  -0.037   +0.004  0.022 [ -0.039, +0.046]
amod           +0.044  +0.008  +0.032  0.020  -0.002   +0.041  0.021 [ +0.000, +0.082]
conj           +0.040  -0.011  -0.008  0.020  +0.016   -0.007  0.020 [ -0.046, +0.032]
advcl          +0.005  -0.006  -0.006  0.014  +0.069   +0.039  0.019 [ +0.002, +0.075]
aux            -0.015  +0.005  +0.014  0.027  +0.071   +0.042  0.034 [ -0.024, +0.109]
nsubj:pass     +0.024  -0.001  +0.020  0.021  +0.080   +0.047  0.025 [ -0.002, +0.096]
acl:relcl      +0.040  -0.001  +0.029  0.034  +0.091   +0.053  0.039 [ -0.022, +0.129]
acl            +0.008  -0.002  -0.000  0.018  +0.117   +0.043  0.023 [ -0.001, +0.088]
ccomp          +0.111  -0.005  +0.012  0.024  +0.124   +0.041  0.027 [ -0.012, +0.094]
case           +0.213  +0.015  +0.084  0.023  +0.154   +0.068  0.024 [ +0.021, +0.115]
obl            +0.278  +0.012  +0.062  0.025  +0.155   +0.040  0.023 [ -0.006, +0.085]
aux:pass       +0.058  +0.013  +0.044  0.027  +0.163   +0.080  0.033 [ +0.015, +0.145]
nsubj          +0.100  +0.022  +0.072  0.026  +0.174   +0.098  0.030 [ +0.039, +0.157]
xcomp          +0.172  +0.013  +0.041  0.035  +0.180   +0.071  0.036 [ +0.001, +0.141]
advmod         +0.128  -0.002  +0.036  0.022  +0.249   +0.095  0.028 [ +0.039, +0.150]
cc             +0.505  -0.005  +0.040  0.023  +0.488   +0.054  0.024 [ +0.007, +0.102]
```

## interaction by token distance, all types pooled

```
bin    kind    n   mean     se
  1     arc 3635 +0.055 +0.038
  1 control 2162 -0.003 +0.033
  2     arc 2539 +0.132 +0.026
  2 control 3710 +0.037 +0.010
  3     arc 1413 +0.091 +0.031
  3 control 1641 +0.024 +0.012
4-6     arc 1567 +0.092 +0.040
4-6 control 1638 +0.012 +0.008
 7+     arc  659 +0.003 +0.015
 7+ control  662 +0.021 +0.013
```

## pooled fit, covariate coefficients

```
intercept                      -0.099   0.061
dist:2                         +0.043   0.031
dist:3                         +0.014   0.035
dist:4-6                       +0.002   0.035
dist:7+                        -0.027   0.034
nqwen_first                    +0.010   0.027
nqwen_second                   -0.003   0.028
in_quote_first                 +0.159   0.042
in_quote_second                -0.156   0.034
position                       +0.084   0.048
final_copy_first               -0.072   0.044
final_copy_second              +0.959   0.244
copies_first(cap3)             +0.030   0.012
copies_second(cap3)            +0.031   0.012
upos_first:ADJ                 +0.042   0.031
upos_first:ADP                 -0.002   0.019
upos_first:ADV                 -0.056   0.037
upos_first:AUX                 -0.021   0.051
upos_first:CCONJ               -0.013   0.035
upos_first:DET                 +0.012   0.033
upos_first:INTJ                +0.091   0.070
upos_first:NUM                 +0.026   0.115
upos_first:PART                -0.072   0.082
upos_first:PRON                +0.003   0.043
upos_first:PROPN               -0.053   0.043
upos_first:SCONJ               +0.082   0.059
upos_first:VERB                +0.060   0.041
upos_second:ADJ                +0.019   0.045
upos_second:ADP                -0.062   0.036
upos_second:ADV                -0.041   0.046
upos_second:AUX                -0.119   0.080
upos_second:CCONJ              -0.021   0.054
upos_second:DET                -0.104   0.060
upos_second:INTJ               -0.028   0.065
upos_second:NUM                -0.160   0.113
upos_second:PART               -0.108   0.060
upos_second:PRON               -0.079   0.038
upos_second:PROPN              -0.018   0.055
upos_second:SCONJ              -0.034   0.053
upos_second:VERB               -0.111   0.057
```
