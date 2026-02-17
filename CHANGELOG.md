# Changelog

## v3.3.1 (2026-02-17)

### Documentation
- **README.md**: Complete rewrite for reproducibility. Added step-by-step
  reviewer guide, Paper–Code correspondence map (Table/Figure → command/output),
  expected results directory structure, time estimates per phase, and
  environment setup instructions referencing `env_setup/`.
- **readme_CL.txt**: Full rewrite. All commands now grouped by experimental
  phase with Paper Table/Figure cross-references, time estimates, expected
  result.json counts, and output file → paper item mapping.
- **data/README_data.md**: Restructured with per-dataset details, sample
  counts, target column names, source URLs, and clear ethics statement.
- **CHANGELOG.md**: Added v3.3.1 documentation release entry.

---

## v3.3 (2026-02-12)

### Fixes
- **Task 1 — run_kao.py**: `hparams_effective` in result.json now records the per-seed `seed_hparams` (includes `seed` and `cv_seed`), matching the parameters actually passed to `KAO_Core`.
- **Task 2 — run_baselines.py**: gplearn and Operon branches now forward user-supplied hparams (from `--hparams_json` or CLI) to `baseline.fit()` via `**method_kwargs`, so `hparams_effective` in result.json reflects what was actually executed.
- **Task 3 — result.json backward compat**: All `write_result_json()` calls now emit both `budget_seconds` and `time_budget` with identical values, preventing downstream breakage from the v3.2.1 field rename.

### Files changed
- `scripts/run_kao.py` — renamed `default_hparams` to `seed_hparams`, used it in result.json
- `scripts/run_baselines.py` — added `method_kwargs` forwarding for gplearn/Operon; added `time_budget` field to all 4 result.json writes
- `README.md` — updated repo structure and changelog
- `CHANGELOG.md` — new file

---

## v3.2.1 (2026-02-12)

### Fixes
- **P0-1**: Fixed `UnboundLocalError` in `run_kao.py` — `time_budget` variable was used before assignment. Introduced `utils/hparams.py` with `merge_hparams(defaults, json_overrides, cli_overrides)` for 3-tier priority merge. Changed `--time_budget` default from `60.0` to `None`.
- **P0-2**: `PySRSR.fit()` now correctly forwards `time_budget=time_budget` to `run_pysr_unified()`.
- **P1-1**: Removed fragile double-parse of `hparams_json` in RILS-ROLS branch. `max_time` now defaults to `time_budget` when not user-specified.
- **P2-1**: All result.json writes include `hparams_effective` and `budget_seconds`.

### New files
- `utils/hparams.py`
- `tests/test_v321_regressions.py`

---

## v3.2 (2026-02-11)

### Features
- Time-budget end-to-end propagation (early stop in `run_evolution()`)
- Directory normalisation: method-specific subdirectories
- Unified `result.json` schema across all 5 methods (KAO, PySR, RILS-ROLS, gplearn, Operon)
- `n_jobs` configurable (default 1 for fair comparison)
- Statistics: inner-join on `(dataset, seed)` for pairwise tests
- Summary table: one p-value column per dataset

### New files
- `utils/result_io.py` — `write_result_json()` with numpy/NaN safety
- `utils/cv_eval.py` — `kfold_eval_fixed_model()` for fold-based evaluation without retraining
- `tests/test_v32_upgrades.py`

---

## v3.1 (2026-02)
Full restructure. Added gplearn/Operon baselines, ablation/Pareto/SRBench experiment scripts, statistical analysis pipeline (Wilcoxon/Friedman/bootstrap), 11 visualization functions, master runner, pytest suite, eICU preprocessing, configs/default.yaml.

## v2.1 (2025-11)
Initial release.
