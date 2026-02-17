# ============================================================
# KAO Experiment Command Line Reference
# Updated: 2026-02-17
# Paper: "Parsimonious Symbolic Regression via Typed Quadratic
#         Operators and Knee-Point Selection"
# ============================================================
#
# This file lists every command needed to reproduce all paper
# results. Commands are grouped by experimental phase and map
# to specific tables/figures in the manuscript.
#
# Notation:
#   SEEDS30 = 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30
#
# Prerequisite: activate the kao310 Conda environment
#   conda activate kao310

# ============================================================
# ENVIRONMENT SETUP
# ============================================================

# Option A: Conda (recommended)
conda env create -f env_setup/env_kao310.yml
conda activate kao310
pip install rils-rols>=1.6.0 --no-build-isolation

# Option B: pip only
pip install -r requirements.txt

# Verify installation
python -c "from kao.KAO_v3_1 import run_single; print('KAO OK')"
python -c "from baselines.registry import get_all_baselines; print([b.name for b in get_all_baselines()])"
python -c "from data.srbench_synthetic.generate import BENCHMARKS; print(list(BENCHMARKS.keys()))"
python -c "from analysis.statistics import StatisticalAnalyzer; print('Statistics OK')"

# ============================================================
# QUICK SMOKE TEST (~5 min)
# ============================================================

python scripts/run_all.py --dry-run                          # Preview all commands
python scripts/run_all.py --seeds 5 --time-budget 10         # Execute quick test

# ============================================================
# FULL REPRODUCTION via MASTER RUNNER (~24-48 h)
# ============================================================
# Runs all 7 phases: kao → baselines → ablation → pareto →
# srbench → eicu → analysis. Uses configs/default.yaml.

python scripts/run_all.py                                    # 30 seeds, 60s budget
python scripts/run_all.py --seeds 30 --time-budget 60        # Explicit equivalent

# Phase-by-phase alternative:
python scripts/run_all.py --phase kao                        # ~3 h
python scripts/run_all.py --phase baselines                  # ~8 h
python scripts/run_all.py --phase ablation                   # ~5 h
python scripts/run_all.py --phase pareto                     # ~3 h
python scripts/run_all.py --phase srbench                    # ~3 h
python scripts/run_all.py --phase eicu                       # ~1 h (included in kao/baselines too)
python scripts/run_all.py --phase analysis                   # ~2 min

# ============================================================
# PHASE 1: KAO on all datasets (30 seeds × 60s)
# → Contributes to Table 1–3 (main comparison, reachability,
#   representative expressions)
# Estimated time: ~25 min per dataset × 7 = ~3 h
# ============================================================

SEEDS30=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30

# MIMIC-IV ICU
python scripts/run_kao.py \
  --csv data/mimic_iv/ICU_composite_risk_score.csv \
  --target composite_risk_score \
  --seeds $SEEDS30 --time_budget 60 --dataset_name mimic_iv \
  --outdir results/mimic_iv/60s/kao

# NHANES
python scripts/run_kao.py \
  --csv data/nhanes/NHANES_metabolic_score.csv \
  --target metabolic_score \
  --seeds $SEEDS30 --time_budget 60 --dataset_name nhanes \
  --outdir results/nhanes/60s/kao

# eICU
python scripts/run_kao.py \
  --csv data/eicu/eICU_composite_risk_score.csv \
  --target composite_risk_score \
  --seeds $SEEDS30 --time_budget 60 --dataset_name eicu \
  --outdir results/eicu/60s/kao

# UCI Airfoil
python scripts/run_kao.py \
  --csv data/uci/airfoil_self_noise.csv \
  --target SSPL \
  --seeds $SEEDS30 --time_budget 60 --dataset_name airfoil \
  --outdir results/airfoil/60s/kao

# UCI Auto-MPG
python scripts/run_kao.py \
  --csv data/uci/auto-mpg_sel.csv \
  --target mpg \
  --seeds $SEEDS30 --time_budget 60 --dataset_name auto_mpg \
  --outdir results/auto_mpg/60s/kao

# UCI Communities & Crime
python scripts/run_kao.py \
  --csv data/uci/ComCri_ViolentCrimesPerPop.csv \
  --target ViolentCrimesPerPop \
  --seeds $SEEDS30 --time_budget 60 --dataset_name communities \
  --outdir results/communities/60s/kao

# UCI Hydraulic Systems
python scripts/run_kao.py \
  --csv data/uci/HydraulicSys_fault_score.csv \
  --target fault_score \
  --seeds $SEEDS30 --time_budget 60 --dataset_name hydraulic \
  --outdir results/hydraulic/60s/kao

# ============================================================
# PHASE 2: Baselines on all datasets (30 seeds × 60s)
# → Contributes to Table 1–3
# Estimated time: ~1 h per dataset × 7 = ~8 h
# ============================================================
# --experiments all  runs PySR, RILS-ROLS, gplearn, Operon

# MIMIC-IV ICU
python scripts/run_baselines.py \
  --csv data/mimic_iv/ICU_composite_risk_score.csv \
  --target composite_risk_score \
  --experiments all --seeds $SEEDS30 --time_budget 60 \
  --dataset_name mimic_iv --outdir results/mimic_iv/60s

# NHANES
python scripts/run_baselines.py \
  --csv data/nhanes/NHANES_metabolic_score.csv \
  --target metabolic_score \
  --experiments all --seeds $SEEDS30 --time_budget 60 \
  --dataset_name nhanes --outdir results/nhanes/60s

# eICU
python scripts/run_baselines.py \
  --csv data/eicu/eICU_composite_risk_score.csv \
  --target composite_risk_score \
  --experiments all --seeds $SEEDS30 --time_budget 60 \
  --dataset_name eicu --outdir results/eicu/60s

# UCI Airfoil
python scripts/run_baselines.py \
  --csv data/uci/airfoil_self_noise.csv \
  --target SSPL \
  --experiments all --seeds $SEEDS30 --time_budget 60 \
  --dataset_name airfoil --outdir results/airfoil/60s

# UCI Auto-MPG
python scripts/run_baselines.py \
  --csv data/uci/auto-mpg_sel.csv \
  --target mpg \
  --experiments all --seeds $SEEDS30 --time_budget 60 \
  --dataset_name auto_mpg --outdir results/auto_mpg/60s

# UCI Communities & Crime
python scripts/run_baselines.py \
  --csv data/uci/ComCri_ViolentCrimesPerPop.csv \
  --target ViolentCrimesPerPop \
  --experiments all --seeds $SEEDS30 --time_budget 60 \
  --dataset_name communities --outdir results/communities/60s

# UCI Hydraulic Systems
python scripts/run_baselines.py \
  --csv data/uci/HydraulicSys_fault_score.csv \
  --target fault_score \
  --experiments all --seeds $SEEDS30 --time_budget 60 \
  --dataset_name hydraulic --outdir results/hydraulic/60s

# ============================================================
# PHASE 3: Ablation study (30 seeds × 60s)
# → Table 4, Fig. 3 (ablation bars), Suppl. Fig. S11-S13
# Estimated time: ~1.5 h per dataset × 3 = ~5 h
# ============================================================

# MIMIC-IV
python scripts/run_ablation.py \
  --csv data/mimic_iv/ICU_composite_risk_score.csv \
  --target composite_risk_score \
  --seeds 30 --time_budget 60 --dataset_name mimic_iv

# NHANES
python scripts/run_ablation.py \
  --csv data/nhanes/NHANES_metabolic_score.csv \
  --target metabolic_score \
  --seeds 30 --time_budget 60 --dataset_name nhanes

# eICU
python scripts/run_ablation.py \
  --csv data/eicu/eICU_composite_risk_score.csv \
  --target composite_risk_score \
  --seeds 30 --time_budget 60 --dataset_name eicu

# ============================================================
# PHASE 4: Time-resolved Pareto front (30 seeds × 60s)
# → Suppl. Fig. S14 (pareto panels), HV trend
# Estimated time: ~1 h per dataset × 3 = ~3 h
# ============================================================

# MIMIC-IV
python scripts/run_pareto_time.py \
  --csv data/mimic_iv/ICU_composite_risk_score.csv \
  --target composite_risk_score \
  --seeds 30 --time_budget 60 --dataset_name mimic_iv

# NHANES
python scripts/run_pareto_time.py \
  --csv data/nhanes/NHANES_metabolic_score.csv \
  --target metabolic_score \
  --seeds 30 --time_budget 60 --dataset_name nhanes

# eICU
python scripts/run_pareto_time.py \
  --csv data/eicu/eICU_composite_risk_score.csv \
  --target composite_risk_score \
  --seeds 30 --time_budget 60 --dataset_name eicu

# ============================================================
# PHASE 5: SRBench synthetic benchmarks (30 seeds × 60s)
# → Table 5, Fig. 4 (SRBench heatmap)
# Estimated time: ~3 h
# ============================================================

python scripts/run_srbench.py --seeds 30 --time_budget 60

# Noiseless only:
# python scripts/run_srbench.py --noise_levels 0.0 --seeds 30 --time_budget 60

# KAO only (skip baselines):
# python scripts/run_srbench.py --skip_baselines --seeds 30 --time_budget 60

# ============================================================
# PHASE 6: eICU preprocessing (optional — CSV already provided)
# ============================================================

python data/eicu/preprocess.py --target composite_risk_score --also_csv

# ============================================================
# PHASE 7: Analysis — generate all figures + LaTeX tables
# → ALL tables and figures in the paper
# Estimated time: ~2 min
# ============================================================

python analysis/generate_paper_figures.py --results-dir results/

# Output file → Paper item mapping:
#   results/tables/summary_main.tex              → Table 1
#   results/tables/summary_main.csv              → Table 1 (CSV)
#   results/tables/Table_reachability.tex         → Table 2
#   results/tables/Table_expressions.tex          → Table 3
#   results/tables/ablation_table.tex             → Table 4
#   results/tables/Table_srbench.tex              → Table 5
#   results/tables/Table_pairwise_full.tex        → Suppl. Table S1
#   results/tables/Table_reachability_thresholds.tex → Suppl. Table S2
#   results/tables/Table_runtime.tex              → Suppl. Table S3
#   results/figures/Fig_r2_vs_complexity_*.pdf     → Fig. 2
#   results/figures/ablation_bars.pdf              → Fig. 3
#   results/figures/srbench_heatmap.pdf            → Fig. 4
#   results/figures/cd_diagram.pdf                 → Suppl. Fig. S1
#   results/figures/pairwise_heatmap.pdf           → Suppl. Fig. S2
#   results/figures/forest_plot.pdf                → Suppl. Fig. S3
#   results/figures/violin_*.pdf                   → Suppl. Fig. S4–S10
#   results/figures/convergence_*.pdf              → Suppl. Fig. S11–S13
#   results/figures/pareto_evolution_panels.pdf     → Suppl. Fig. S14
#   results/figures/Fig_runtime_by_dataset.pdf      → Suppl. Fig. S15
#   results/figures/Fig_reachability_thresholds.pdf → Suppl. Fig. S16

# ============================================================
# INDIVIDUAL SCRIPT USAGE REFERENCE
# ============================================================

# --- run_kao.py ---
# Args: --csv (required) --target (required) --outdir (required)
#       --seeds (comma-separated) --time_budget (float, default from config)
#       --dataset_name --test_size (default 0.2) --hparams_json
#       --verbose --standardize --standardize-y

# --- run_baselines.py ---
# Args: --csv (required) --target (required) --experiments (required: pysr|rils_rols|all)
#       --outdir (required) --seeds --time_budget --dataset_name
#       --test_size --hparams_json --verbose --standardize --standardize-y

# --- run_ablation.py ---
# Args: --csv (required) --target (required)
#       --seeds (int, default 30) --time_budget (default 60.0)
#       --dataset_name --test_size --conditions --skip_plots

# --- run_pareto_time.py ---
# Args: --csv (required) --target (required)
#       --seeds (int, default 30) --time_budget (default 60.0)
#       --dataset_name --test_size --checkpoints --skip_plots --save_snapshots

# --- run_srbench.py ---
# Args: --benchmark (list) --methods (list) --seeds (int, default 30)
#       --time_budget (default 60.0) --noise_levels (list) --skip_plots --skip_baselines

# --- run_all.py ---
# Args: --config (default configs/default.yaml) --phase (all|kao|baselines|...)
#       --seeds (int) --time-budget (float) --dry-run --skip-plots

# --- analysis/generate_paper_figures.py ---
# Args: --results-dir (default results/) --skip-plots

# ============================================================
# TESTING
# ============================================================

pytest tests/ -v
pytest tests/test_kao.py -v          # KAO core (10s budget per test)
pytest tests/test_baselines.py -v    # Baseline wrappers
pytest tests/test_srbench.py -v      # SRBench data generation

# ============================================================
# UTILITIES
# ============================================================

# Verify imports
python -c "from kao.KAO_v3_1 import run_single, KAOResult; print('KAO OK')"
python -c "from baselines.registry import get_all_baselines; print([b.name for b in get_all_baselines()])"
python -c "from data.srbench_synthetic.generate import BENCHMARKS; print(list(BENCHMARKS.keys()))"

# Count experiment results
find results/ -name "result.json" | wc -l

# Expected result.json files per phase:
#   Phase 1 (KAO):       7 datasets × 30 seeds = 210
#   Phase 2 (Baselines): 7 datasets × 4 methods × 30 seeds = 840
#   Phase 3 (Ablation):  3 datasets × 5 conditions × 30 seeds = 450
#   Phase 4 (Pareto):    3 datasets × 2 conditions × 30 seeds = 180
#   Phase 5 (SRBench):   5 benchmarks × 2 noises × ~5 methods × 30 seeds = ~1500
#   Total:               ~3180 result.json files
