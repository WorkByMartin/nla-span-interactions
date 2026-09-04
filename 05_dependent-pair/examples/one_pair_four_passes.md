# One pair, four passes

Document 1301, baseline FVE 0.733, draw 0 of 8. Effects are FVE points lost,
so positive means the reconstruction got worse. Interaction = e(a) + e(b) - e(both):
positive means the two words carry overlapping information, negative means they are
worth more together than apart.

The arc pair is the one whose mean interaction is nearest the arc mean of -0.017 points,
among the 726 arcs with an exact control. Its control is in the same document,
same ordered part-of-speech pair, same token distance, no dependency between them.

Arc pair:
    Final token ends the equipment/signature gear paragraph with "and a couple of effects
    pedals.\n\nMy studio is currently my lounge and I am working on a used JB-1 EG guitar,
    still [B:has] fantastic [A:tone].

arc, token distance 2, VERB-NOUN, dobj (object noun to verb)
    baseline FVE 0.733
    A swapped        0.734   e(a)    = -0.130 points
    B swapped        0.731   e(b)    = +0.171 points
    both swapped     0.732   e(both) = +0.053 points
    interaction = e(a) + e(b) - e(both) = -0.012 points on this draw,
    -0.017 +- 0.024 over the 8 draws

Control pair:
    " — this [B:completes] a [A:gear]-answer block, signaling the next block will introduce
    another interview question, likely about influences, favourite artists, or writing
    goals.

control, token distance 2, VERB-NOUN, no dependency
    baseline FVE 0.733
    A swapped        0.729   e(a)    = +0.324 points
    B swapped        0.733   e(b)    = +0.013 points
    both swapped     0.731   e(both) = +0.158 points
    interaction = e(a) + e(b) - e(both) = +0.178 points on this draw,
    +0.007 +- 0.041 over the 8 draws

