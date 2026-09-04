-- 0001_init: substrings, the labels put on them, the measurements taken of
-- edited documents. Three separable things, three sets of tables.

CREATE TABLE docs (
    doc_id INTEGER PRIMARY KEY,
    text   TEXT NOT NULL,
    source TEXT                                  -- asset id, e.g. ffw_main_traces
);

CREATE TABLE runs (
    run_id     INTEGER PRIMARY KEY,
    script     TEXT,
    git_sha    TEXT,
    assets     TEXT,                             -- json list of asset ids
    notes      TEXT,
    started_at TEXT
);

-- A span is a substring. No type column: what it is belongs in labels.
CREATE TABLE spans (
    span_id    INTEGER PRIMARY KEY,
    doc_id     INTEGER NOT NULL REFERENCES docs(doc_id),
    char_start INTEGER NOT NULL,
    char_end   INTEGER NOT NULL,
    UNIQUE (doc_id, char_start, char_end)
);

-- Long format. One decomposer writes one scheme and never collides with another.
CREATE TABLE labels (
    span_id INTEGER NOT NULL REFERENCES spans(span_id),
    scheme  TEXT NOT NULL,                       -- decomposer + version
    key     TEXT NOT NULL,
    value   TEXT,
    PRIMARY KEY (span_id, scheme, key)
);

CREATE TABLE relations (
    scheme TEXT NOT NULL,
    span_a INTEGER NOT NULL REFERENCES spans(span_id),
    span_b INTEGER NOT NULL REFERENCES spans(span_id),
    kind   TEXT NOT NULL,                        -- e.g. 'head'
    PRIMARY KEY (scheme, span_a, span_b, kind)
);

-- One edited version of one document. A variant with no substitutions is the
-- baseline. Every reconstructor forward pass is one variant.
CREATE TABLE variants (
    variant_id     INTEGER PRIMARY KEY,
    doc_id         INTEGER NOT NULL REFERENCES docs(doc_id),
    created_run_id INTEGER REFERENCES runs(run_id)
);

CREATE TABLE substitutions (
    variant_id INTEGER NOT NULL REFERENCES variants(variant_id),
    span_id    INTEGER NOT NULL REFERENCES spans(span_id),
    substitute TEXT NOT NULL,
    source     TEXT,                             -- where the substitute came from
    depth      INTEGER,
    draw_idx   INTEGER,
    prob       REAL,
    PRIMARY KEY (variant_id, span_id)
);

CREATE TABLE measurements (
    variant_id INTEGER NOT NULL REFERENCES variants(variant_id),
    run_id     INTEGER NOT NULL REFERENCES runs(run_id),
    metric     TEXT NOT NULL,                    -- mse | fve | seq_len
    value      REAL,
    PRIMARY KEY (variant_id, run_id, metric)
);

CREATE INDEX ix_spans_doc            ON spans(doc_id);
CREATE INDEX ix_labels_scheme_key    ON labels(scheme, key, value);
CREATE INDEX ix_relations_a          ON relations(span_a);
CREATE INDEX ix_relations_b          ON relations(span_b);
CREATE INDEX ix_variants_doc         ON variants(doc_id);
CREATE INDEX ix_variants_run         ON variants(created_run_id);
CREATE INDEX ix_subs_span            ON substitutions(span_id);
CREATE INDEX ix_subs_depth           ON substitutions(depth);
CREATE INDEX ix_meas_run_metric      ON measurements(run_id, metric);

-- Span text is derived, never stored.
CREATE VIEW v_span_text AS
SELECT s.span_id, s.doc_id, s.char_start, s.char_end,
       substr(d.text, s.char_start + 1, s.char_end - s.char_start) AS text
FROM spans s
JOIN docs d ON d.doc_id = s.doc_id;

CREATE VIEW v_pos AS
SELECT t.span_id, t.doc_id, t.char_start, t.char_end, t.text,
       l.value AS pos, l.scheme
FROM v_span_text t
JOIN labels l ON l.span_id = t.span_id
            AND l.key = 'pos'
            AND l.scheme LIKE 'spacy-%';

-- Helper: how many spans a variant edits. 0 = baseline.
CREATE VIEW v_nsub AS
SELECT v.variant_id, COUNT(s.span_id) AS n_sub
FROM variants v
LEFT JOIN substitutions s ON s.variant_id = v.variant_id
GROUP BY v.variant_id;

-- Helper: metrics pivoted to columns, one row per (variant, run).
CREATE VIEW v_variant_metrics AS
SELECT v.variant_id, v.doc_id, m.run_id,
       MAX(CASE WHEN m.metric = 'mse'     THEN m.value END) AS mse,
       MAX(CASE WHEN m.metric = 'fve'     THEN m.value END) AS fve,
       MAX(CASE WHEN m.metric = 'seq_len' THEN m.value END) AS seq_len
FROM variants v
JOIN measurements m ON m.variant_id = v.variant_id
GROUP BY v.variant_id, v.doc_id, m.run_id;

CREATE VIEW v_baseline AS
SELECT vm.variant_id, vm.doc_id, vm.run_id,
       vm.mse AS base_mse, vm.fve AS base_fve, vm.seq_len AS base_seq_len
FROM v_variant_metrics vm
JOIN v_nsub n ON n.variant_id = vm.variant_id AND n.n_sub = 0;

-- One substituted span, its measurement, and the same doc's baseline in the
-- same run, so dmse/dfve are columns.
CREATE VIEW v_single AS
SELECT vm.variant_id, vm.doc_id, vm.run_id,
       su.span_id, t.char_start, t.char_end, t.text AS span_text,
       su.substitute, su.source, su.depth, su.draw_idx, su.prob,
       vm.mse, vm.fve, vm.seq_len,
       b.variant_id AS baseline_variant_id, b.base_mse, b.base_fve,
       vm.mse - b.base_mse AS dmse,
       vm.fve - b.base_fve AS dfve
FROM v_variant_metrics vm
JOIN v_nsub n         ON n.variant_id = vm.variant_id AND n.n_sub = 1
JOIN substitutions su ON su.variant_id = vm.variant_id
JOIN v_span_text t    ON t.span_id = su.span_id
LEFT JOIN v_baseline b ON b.doc_id = vm.doc_id AND b.run_id = vm.run_id;

-- Helper: the two singles a pair is compared against, collapsed to one row per
-- (run, doc, span, substitute) so repeat draws average rather than fan out.
CREATE VIEW v_single_mean AS
SELECT run_id, doc_id, span_id, substitute,
       AVG(mse) AS mse, AVG(fve) AS fve, COUNT(*) AS n_draws
FROM v_single
GROUP BY run_id, doc_id, span_id, substitute;

-- Two substituted spans with the four-way interaction both - a - b + baseline.
-- NULL wherever a matching single or the baseline is missing from the run.
CREATE VIEW v_pair AS
SELECT p.variant_id, p.doc_id, p.run_id,
       p.span_a, p.span_b,
       ta.text AS text_a, tb.text AS text_b,
       sa.substitute AS sub_a, sb.substitute AS sub_b,
       sa.depth AS depth_a, sb.depth AS depth_b,
       p.mse AS both_mse, p.fve AS both_fve,
       ma.mse AS a_mse, ma.fve AS a_fve,
       mb.mse AS b_mse, mb.fve AS b_fve,
       bl.base_mse, bl.base_fve,
       p.mse - ma.mse - mb.mse + bl.base_mse AS inter_mse,
       p.fve - ma.fve - mb.fve + bl.base_fve AS inter_fve
FROM (
    SELECT vm.variant_id, vm.doc_id, vm.run_id, vm.mse, vm.fve,
           MIN(su.span_id) AS span_a, MAX(su.span_id) AS span_b
    FROM v_variant_metrics vm
    JOIN v_nsub n         ON n.variant_id = vm.variant_id AND n.n_sub = 2
    JOIN substitutions su ON su.variant_id = vm.variant_id
    GROUP BY vm.variant_id, vm.doc_id, vm.run_id, vm.mse, vm.fve
) p
JOIN substitutions sa ON sa.variant_id = p.variant_id AND sa.span_id = p.span_a
JOIN substitutions sb ON sb.variant_id = p.variant_id AND sb.span_id = p.span_b
JOIN v_span_text ta   ON ta.span_id = p.span_a
JOIN v_span_text tb   ON tb.span_id = p.span_b
LEFT JOIN v_baseline bl ON bl.doc_id = p.doc_id AND bl.run_id = p.run_id
LEFT JOIN v_single_mean ma ON ma.run_id = p.run_id AND ma.doc_id = p.doc_id
                          AND ma.span_id = p.span_a AND ma.substitute = sa.substitute
LEFT JOIN v_single_mean mb ON mb.run_id = p.run_id AND mb.doc_id = p.doc_id
                          AND mb.span_id = p.span_b AND mb.substitute = sb.substitute;
