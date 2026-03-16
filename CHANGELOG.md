# Changelog

## [v4.1] - 2026-03-17 (post-review patches)

### Fixed
- `analysis/visualization.py`: Plot title "Convergence" → "Runtime vs. Final Test R²" for Figs S11–S13 (scatter plots, not convergence curves)

### Documentation
- `README.md`, `readme_CL.txt`, `CHANGELOG.md`: Full rewrite for v4 (argparse flags verified, file structure scanned)

---

## [v4] - 2026-03-16

### Paper
- Target: Swarm and Evolutionary Computation (SWEVO)
- Title: KAO: Budget-Reachable Symbolic Regression via Typed Quadratic Operators and Knee-Point Selection

### Fixed
- **cv_seed propagation bug** in `run_ablation.py`: ablation experiments now pass `cv_seed=seed` to `run_single()`, matching the main experiment convention in `run_kao.py`. Previously, ablation used a fixed `cv_seed=2025` (the `KAO_Core` default), causing different cross-validation fold assignments and resulting in Table 2 vs Table 7 numerical discrepancies.
- **`run_single()` default cv_seed**: Added automatic `cv_seed=seed` propagation in `KAO_v3_1.py:run_single()` when caller does not specify `cv_seed`, preventing future callers from hitting the same bug.
- **Section 2.2 size bound**: Corrected representational efficiency proof to use Horner factorisation as the arithmetic-only baseline. Size saving corrected from `2|X|+3` to `|X|+2`.

### Changed
- **Narrative**: Restructured paper around "zero-configuration delivery" as primary value proposition (was: accuracy competitiveness under strict budgets)
- **Contributions reordered**: (i) budget reachability formulation, (ii) zero-config NSGA-II + knee-point protocol, (iii) typed quadratic operator as optional inductive bias
- **"strict 60s"** changed to **"nominal 60s"** (wall-clock time); "strict" retained only for complexity budget (<=8 nodes)
- **"exact symbolic recovery"** changed to **"exact functional recovery"** (SRBench AST checker cannot parse KAO/makeTP primitives)
- **"complete-case analysis"** changed to **"two-stage preprocessing pipeline"** (extraction-stage imputation + row-wise deletion)
- **Section 4.4**: Hydraulic reframed as boundary case (was: "comparably on Hydraulic")
- **Section 4.6**: Ablation reframed -- NSGA-II + knee-point identified as primary driver; KAO operator repositioned as optional inductive bias
- **Section 4.7**: SRBench reframed around operator-coverage boundaries
- **Section 5 Discussion**: Added No-Free-Lunch perspective paragraph; added "structural parsimony does not equal semantic interpretability" paragraph
- **Limitations**: PS-Tree (SWEVO paper) explicitly mentioned as important future comparison
- Healthcare language cooled throughout ("clinical analytics" changed to "healthcare risk-score approximation")

### Added
- `scripts/rebuild_ablation_v4.py` -- hybrid data loader (KAO from main experiment, No_KAO from v4_ablation) with Holm-Bonferroni correction
- `--outdir` CLI parameter for `run_ablation.py`
- `Wolpert1997` reference (No Free Lunch Theorems)
- `idTP` identity wrapper definition in Section 2.1
- Cover letter (`cover_letter_v4.tex`)
- TRIPOD+AI checklist v4

### Data
- `results/v4_ablation/`: 450 new ablation result files (3 healthcare datasets x 5 conditions x 30 seeds)
- Regenerated: `ablation_summary.csv`, `ablation_pairwise.csv`, `ablation_table.tex`, `ablation_bars.pdf`
- Original `results/ablation/` preserved (v3.12 data, do not use for v4 paper)

### Git
- Commit: `881f79d`

---

## v3.3.1 (2026-02-17)

### Documentation
- **README.md**: Complete rewrite for reproducibility. Added step-by-step
  reviewer guide, Paper-Code correspondence map (Table/Figure -> command/output),
  expected results directory structure, time estimates per phase, and
  environment setup instructions referencing `env_setup/`.
- **readme_CL.txt**: Full rewrite. All commands now grouped by experimental
  phase with Paper Table/Figure cross-references, time estimates, expected
  result.json counts, and output file -> paper item mapping.
- **data/README_data.md**: Restructured with per-dataset details, sample
  counts, target column names, source URLs, and clear ethics statement.
- **CHANGELOG.md**: Added v3.3.1 documentation release entry.

---

## v3.3 (2026-02-12)

### Fixes
- **Task 1 -- run_kao.py**: `hparams_effective` in result.json now records the per-seed `seed_hparams` (includes `seed` and `cv_seed`), matching the parameters actually passed to `KAO_Core`.
- **Task 2 -- run_baselines.py**: gplearn and Operon branches now forward user-supplied hparams (from `--hparams_json` or CLI) to `baseline.fit()` via `**method_kwargs`, so `hparams_effective` in result.json reflects what was actually executed.
- **Task 3 -- result.json backward compat**: All `write_result_json()` calls now emit both `budget_seconds` and `time_budget` with identical values, preventing downstream breakage from the v3.2.1 field rename.

### Files changed
- `scripts/run_kao.py` -- renamed `default_hparams` to `seed_hparams`, used it in result.json
- `scripts/run_baselines.py` -- added `method_kwargs` forwarding for gplearn/Operon; added `time_budget` field to all 4 result.json writes
- `README.md` -- updated repo structure and changelog
- `CHANGELOG.md` -- new file

---

## v3.2.1 (2026-02-12)

### Fixes
- **P0-1**: Fixed `UnboundLocalError` in `run_kao.py` -- `time_budget` variable was used before assignment. Introduced `utils/hparams.py` with `merge_hparams(defaults, json_overrides, cli_overrides)` for 3-tier priority merge. Changed `--time_budget` default from `60.0` to `None`.
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
- `utils/result_io.py` -- `write_result_json()` with numpy/NaN safety
- `utils/cv_eval.py` -- `kfold_eval_fixed_model()` for fold-based evaluation without retraining
- `tests/test_v32_upgrades.py`

---

## v3.1 (2026-02)
Full restructure. Added gplearn/Operon baselines, ablation/Pareto/SRBench experiment scripts, statistical analysis pipeline (Wilcoxon/Friedman/bootstrap), 11 visualization functions, master runner, pytest suite, eICU preprocessing, configs/default.yaml.

## v2.1 (2025-11)
Initial release.
