# One sentence, four edits

Document 1934, baseline FVE 0.875. The word is "offices" (NOUN). Effect is FVE points lost,
so a positive number means the reconstruction got worse.

Intact:

    "The scandal began when the finance minister ordered the closure of the Kenya Revenue
    Authority [offices] for two" concludes with a time interval, likely "temporary closure"
    or specific date or moratorium implementation

Corpus swap, the draw nearest the mean of 16 (mean +1.62 points):

    "The scandal began when the finance minister ordered the closure of the Kenya Revenue
    Authority [stitch] for two" concludes with a time interval, likely "temporary closure"
    or specific date or moratorium implementation

    FVE 0.875 to 0.859, +1.61 points

Masked-LM, the draw nearest the mean of the 34 draws that changed the word
(mean +0.52 points; the other 14 of 48 draws put "offices" back):

    "The scandal began when the finance minister ordered the closure of the Kenya Revenue
    Authority [operations] for two" concludes with a time interval, likely "temporary
    closure" or specific date or moratorium implementation

    FVE 0.875 to 0.871, +0.44 points

Deletion:

    "The scandal began when the finance minister ordered the closure of the Kenya Revenue
    Authority [] for two" concludes with a time interval, likely "temporary closure" or
    specific date or moratorium implementation

    FVE 0.875 to 0.868, +0.72 points

Shuffle, every one of the document's 138 lexical words permuted among its own slots, the
permutation nearest the mean of 4 (mean FVE -0.333). First 18 words of the document:

    Kenyan legal constitutional journalism context, documenting court proceedings and
    allegations against PRESUAA concerning Barack Obama. Narrative momentum building ...
    context by two The duration, Kenya Obama Authority catalyst date Revenue shutdown Barack
    continued He. Kenya against or ...

    FVE 0.875 to -0.339, +121.44 points
