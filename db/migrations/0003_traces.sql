-- 0003_traces: fold ffw_pilot_traces.parquet into the store.
--
-- The parquet is the one row per document that everything else is a delta of:
-- the layer-42 activation, the verbalisation it was turned into, the
-- reconstruction, and the per-token KL between the RL verbaliser and the SFT
-- reference. Keeping it in a side file meant every script re-read a 4.5 MB
-- parquet to get the text it was already asking the store for.
--
-- docs.text stays the EXPLANATION, which is what spans index into. The
-- verbalisation is the whole tagged generation and is its own column.
--
-- Vectors are float32 little-endian BLOBs: 5120 floats, 20480 bytes each.
-- That is half the parquet's float64 and lossy against it, so the parquet
-- remains the precise source and is not deleted. Read one back with
--   numpy.frombuffer(blob, dtype='<f4')

ALTER TABLE docs ADD COLUMN global_id TEXT;
ALTER TABLE docs ADD COLUMN domain TEXT;
ALTER TABLE docs ADD COLUMN n_tokens INTEGER;        -- length of the source doc
ALTER TABLE docs ADD COLUMN token_position INTEGER;  -- readout position in it
ALTER TABLE docs ADD COLUMN cjk_fraction REAL;
ALTER TABLE docs ADD COLUMN mse REAL;                -- as scored at extraction
ALTER TABLE docs ADD COLUMN verbalisation TEXT;
ALTER TABLE docs ADD COLUMN activation BLOB;         -- float32 LE, 5120
ALTER TABLE docs ADD COLUMN reconstruction BLOB;     -- float32 LE, 5120

-- The verbaliser's generation, token by token, with the KL against the SFT
-- reference at each step. One row per generated token.
CREATE TABLE doc_tokens (
    doc_id   INTEGER NOT NULL REFERENCES docs(doc_id),
    position INTEGER NOT NULL,
    token_id INTEGER NOT NULL,
    kl       REAL,
    PRIMARY KEY (doc_id, position)
);

CREATE INDEX ix_docs_domain ON docs(domain);

-- Everything about a document except the two big vectors, so `SELECT *` on a
-- hundred documents does not pull 4 MB of activations.
CREATE VIEW v_doc AS
SELECT doc_id, source, global_id, domain, n_tokens, token_position,
       cjk_fraction, mse, length(text) AS n_chars, text, verbalisation
FROM docs;

-- Per-document KL summary, the shape most analyses want.
CREATE VIEW v_doc_kl AS
SELECT doc_id, COUNT(*) AS n_gen_tokens, AVG(kl) AS kl_mean,
       MAX(kl) AS kl_max, SUM(kl) AS kl_total
FROM doc_tokens
GROUP BY doc_id;
