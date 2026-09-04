-- 0002_candidates: the ranked filler distribution a substitute is drawn from,
-- plus structured run configuration.
--
-- Holds the sampler's top-k proposals per span. A candidate is a proposal: not a measurement, and not a
-- label, because it belongs to no variant until some draw turns it into a
-- substitution.
--
-- Candidate strings are stored BARE, with no leading space, the same rule
-- substitutions follow. Space parity is the substitution primitive's business.

CREATE TABLE candidates (
    span_id   INTEGER NOT NULL REFERENCES spans(span_id),
    scheme    TEXT NOT NULL,                   -- proposer + the filter regime
    rank      INTEGER NOT NULL,                -- 0 = most probable
    candidate TEXT NOT NULL,                   -- bare, no leading space
    prob      REAL NOT NULL,                   -- normalised over the stored list
    PRIMARY KEY (span_id, scheme, rank)
);

CREATE INDEX ix_cand_scheme_text ON candidates(scheme, candidate);

-- Everything a run was configured with, as json: argv, environment
-- fingerprint, and any constant the numbers depend on (mse_rawvar).
ALTER TABLE runs ADD COLUMN config TEXT;

CREATE VIEW v_candidate AS
SELECT c.span_id, t.doc_id, t.text AS span_text,
       c.scheme, c.rank, c.candidate, c.prob
FROM candidates c
JOIN v_span_text t ON t.span_id = c.span_id;

-- Word types with their POS counted in context. `tok_i` is present only on
-- spans that are a decomposer's own word tokens, which keeps derived or
-- phrase-level spans out of the lexicon.
CREATE VIEW v_word_pos AS
SELECT lower(trim(t.text)) AS word, l.value AS pos,
       COUNT(*) AS n, MIN(t.span_id) AS first_span_id
FROM v_span_text t
JOIN labels l  ON l.span_id = t.span_id AND l.key = 'pos'
              AND l.scheme LIKE 'spacy-%'
JOIN labels tk ON tk.span_id = t.span_id AND tk.key = 'tok_i'
              AND tk.scheme = l.scheme
WHERE trim(t.text) <> ''
GROUP BY word, pos;

-- Modal POS per word type. Ties go to whichever was seen first in the corpus,
-- which is what collections.Counter.most_common does.
CREATE VIEW v_lexicon AS
SELECT word, pos, n FROM (
    SELECT word, pos, n,
           ROW_NUMBER() OVER (PARTITION BY word
                              ORDER BY n DESC, first_span_id) AS r
    FROM v_word_pos)
WHERE r = 1;
