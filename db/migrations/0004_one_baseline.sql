-- 0004_one_baseline: a (document, run) has exactly one baseline row.
--
-- 0001 assumed one unsubstituted variant per document per run. A real run
-- records several: the greedy stage re-measures the intact string each round so
-- that every delta is against a baseline from its own batched call, and the
-- interaction stage measures it again. Left as it was, v_baseline returned one
-- row per measurement and every join through it fanned v_single out by that
-- many copies.
--
-- The baseline is the FIRST unsubstituted variant the run measured. The others
-- stay in the store as what they are, repeat measurements of the same object,
-- reachable through v_variant_metrics.

DROP VIEW v_baseline;

CREATE VIEW v_baseline AS
SELECT vm.variant_id, vm.doc_id, vm.run_id,
       vm.mse AS base_mse, vm.fve AS base_fve, vm.seq_len AS base_seq_len
FROM v_variant_metrics vm
JOIN v_nsub n ON n.variant_id = vm.variant_id AND n.n_sub = 0
WHERE vm.variant_id = (
    SELECT MIN(vm2.variant_id)
    FROM v_variant_metrics vm2
    JOIN v_nsub n2 ON n2.variant_id = vm2.variant_id AND n2.n_sub = 0
    WHERE vm2.doc_id = vm.doc_id AND vm2.run_id = vm.run_id);

-- Every unsubstituted measurement, so baseline drift within a run is visible.
CREATE VIEW v_baseline_repeats AS
SELECT vm.variant_id, vm.doc_id, vm.run_id, vm.mse, vm.fve, vm.seq_len,
       b.variant_id AS baseline_variant_id,
       vm.mse - b.base_mse AS drift_mse
FROM v_variant_metrics vm
JOIN v_nsub n ON n.variant_id = vm.variant_id AND n.n_sub = 0
JOIN v_baseline b ON b.doc_id = vm.doc_id AND b.run_id = vm.run_id;
