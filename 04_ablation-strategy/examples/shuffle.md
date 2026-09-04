# The shuffle

The whole word order removed, the bag of words kept. Every lexical word of the document is
permuted among the document's own slots, four permutations per document, and the reconstructor
is run on the result. Punctuation is not a slot and does not move.

FVE is the fraction of variance the reconstruction explains, so 1.0 is exact and 0.0 is no
better than predicting the mean activation. It is unbounded below.

## Document 1925

Intact FVE 0.7289. 4 shuffles of 142 lexical words, mean FVE -0.8910, sd 0.1244,
so the shuffle costs 161.994 FVE points on average.

| variant |     FVE | FVE points lost | shown below |
|--------:|--------:|----------------:|-------------|
|  167281 | -0.8604 |        +158.934 | yes         |
|  167282 | -1.0067 |        +173.564 |             |
|  167283 | -0.9678 |        +169.673 |             |
|  167284 | -0.7291 |        +145.806 |             |

The variant shown is the one whose FVE is closest to the mean of the four.

Intact verbalisation, FVE 0.7289:

    Persuasive academic essay by a high school student, consistently advocating against gang
    membership with informal register and personal anecdotes. The essay's argumentative
    momentum builds toward a closing inspirational call-to-action, with the final sentence
    beginning "Just keep the world clean of gangs and your problems." The final phrase "If
    you can't see it, it doesn't need" is mid-sentence, ending on an incomplete predicate;
    the next token must complete this thought, likely with "to be" or "to worry about,"
    following the axiomatic, proverb-like pattern of the conclusion. "races can be separated
    by flooding it out... flooding it... done by keeping your eyes open. There ain't no such
    things as A gang free America, gang problems are forever taking over the world...keep it
    out your thougt. If it caan't been seen it doesn't need"

Shuffled variant 167281, FVE -0.8604:

    can taking been predicate next a gang the, as complete n't like need proverb to
    following America mid to. conclusion onpattern is it There be need the seen the-with-
    done, clean essay A and anecdotes "be out a personal the your - by thought it."
    advocating informal likely "of closing tokenschool If or, the itca toward" momentum
    separatedovern't, races your academic problems final; membership flooding flooding
    ending forever student keeping, keep sentence "the it" eyes "open this worry," axiomatic
    argumentative problems, no-register must gangs to inspirational. "does free consistently
    by essay The Just beginning... out things... your gang see it you such. final gangworld
    The call action 's against n't thougt by, world caan't it of keep with sentence
    it...high If about ai are. builds an Persuasive with does incomplete n'tand phrase"

## Document 3914

Intact FVE 0.7339. 4 shuffles of 123 lexical words, mean FVE 0.4192, sd 0.0713,
so the shuffle costs 31.476 FVE points on average.

| variant |    FVE | FVE points lost | shown below |
|--------:|-------:|----------------:|-------------|
|  180264 | 0.4757 |         +25.824 |             |
|  180265 | 0.3912 |         +34.273 | yes         |
|  180266 | 0.3313 |         +40.268 |             |
|  180267 | 0.4786 |         +25.537 |             |

The variant shown is the one whose FVE is closest to the mean of the four.

Intact verbalisation, FVE 0.7339:

    Email thread/mechanics context: technical VGA programming mailing list discussion about
    VGA modes, SDL, VESA BIOS, screen modes. Continuation of the topic about setting up
    graphics environments: previous sentences covered VESA Video Modes and resolution
    details, establishing domain of mode-setting programming. Final fragment "setwscreen of
    size 800 x " is mid-word ("width," truncated), and ends with "in the other " suggesting
    the same phrase as the opening — indicating a screen mode/resolution context, likely
    completing with "mode" or "resolution" referencing a specific VGA/VESA mode like mode
    13h or a linear memory mode (0x4110). The email concerns SDL VGA modes — the answer
    would reference fullscreen mode settings, e.g., screen resolution modes on VGA, e.g.,
    "Switch" or VGA resolution modes, the well-established framework.

Shuffled variant 180265, FVE 0.3912:

    details with/Final establishing: screen answer 13h or resolution resolution well Video
    mode, the, VGA and, resolution -. resolution the modes Email VGA mid is the mechanics:
    linear a covered e.g. mode modes VGA setwscreen SDL, about 800 VESA of-a 0x4110.
    programming Switch "setting list established mailing x " Modes screensettingprevious
    ("with," context), VGA the same "Continuation as likely " up word truncated memory of
    concerns mode — resolution like referencing VGA/on mode, a the VGA "discussion" modes
    "width" fragment completing or SDL/settings context VESA indicating mode e.g. sentences
    fullscreen domain framework (screen). suggesting would mode size other opening — BIOS or
    mode environments about programming ends, The, modes topic VESA in and, phrase, "thread"
    graphics specific reference email, technical modes-the of.

