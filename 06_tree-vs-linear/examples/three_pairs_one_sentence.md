# Three pairs, one sentence

Document 1664, baseline FVE 0.877, the full pairwise run at 8 draws per pair.
Interaction = e(a) + e(b) - e(both) in FVE points lost: positive means the two words carry
overlapping information, negative means they are worth more together than apart.
Each pair is the one of its kind, in this sentence, nearest that kind's mean over the
whole document (adjacent +0.136, arc at distance 3 or more +0.043, no arc at distance
3 or more -0.014), among pairs whose standard error over the draws is
at or under the harness floor of 0.044 points.

    [A3:Narrative] momentum: [B3:article] has covered bankruptcy [A2:filing], debt
    [B2:restructuring], costs frozen operations, and now focuses on the CEO's quoted
    optimism about long-term financial strength [A1:and] [B1:strategic] recapitalization.

Pair 1, adjacent: "and .. strategic", CCONJ-ADJ, token distance 1, tree path 3
    interaction +0.105 +- 0.043 points over 8 draws

Pair 2, arc: "filing .. restructuring", NOUN-NOUN, token distance 3, tree path 1, dep conj
    interaction +0.047 +- 0.021 points over 8 draws

Pair 3, unrelated: "Narrative .. article", NOUN-NOUN, token distance 3, tree path 3
    interaction -0.020 +- 0.021 points over 8 draws

Figure: ../results/three_pairs_one_sentence.png, copied here.
