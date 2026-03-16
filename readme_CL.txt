# ============================================================
# KAO v4 — Complete Command-Line Reference
# Paper: KAO: Budget-Reachable Symbolic Regression via Typed
#        Quadratic Operators and Knee-Point Selection
# Target: Swarm and Evolutionary Computation (SWEVO)
# Server: eis06 (/home/dengou/KAO_v4)
# Env:    conda activate kao310
# Git:    commit 881f79d
# ============================================================
#
# This file lists every command needed to reproduce all paper
# results. Commands are grouped by experimental phase and map
# to specific tables/figures in the manuscript.
#
# Notation:
#   SEEDS30 = 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30
#
# Working directory: /home/dengou/KAO_v4/KAO  (unless noted)
#
# Prerequisite: activate the kao310 Conda environment
#   eval "$(/home/dengou/anaconda3/bin/conda shell.bash hook)"
#   conda activate kao310

# ============================================================
# 0. ENVIRONMENT SETUP
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

# ============================================================
# QUICK SMOKE TEST (~5 min)
# ============================================================

python scripts/run_all.py --dry-run                          # Preview all commands
python scripts/run_all.py --seeds 5 --time-budget 10         # Execute quick test

# ============================================================
# FULL REPRODUCTION via MASTER RUNNER (~24-48 h)
# ============================================================
# Runs all 7 phases: kao -> baselines -> ablation -> pareto ->
# srbench -> eicu -> analysis.  Uses configs/default.yaml.

python scripts/run_all.py                                    # 30 seeds, 60s budget
python scripts/run_all.py --seeds 30 --time-budget 60        # Explicit equivalent

# Phase-by-phase alternative:
python scripts/run_all.py --phase kao                        # ~3 h
python scripts/run_all.py --phase baselines                  # ~8 h
python scripts/run_all.py --phase ablation                   # ~5 h
python scripts/run_all.py --phase pareto                     # ~3 h
python scripts/run_all.py --phase srbench                    # ~3 h
python scripts/run_all.py --phase eicu                       # preprocessing
python scripts/run_all.py --phase analysis                   # ~2 min

# ============================================================
# PHASE 1: KAO on all datasets (30 seeds x 60s)
# -> Table 1 (main R2/nodes), Table 3 (reachability),
#    Table 4 (expressions), Figure 2 (scatter)
# Estimated time: ~25 min per dataset x 7 = ~3 h
# ============================================================
# run_kao.py args:
#   --csv (required)      Path to CSV data file
#   --target (required)   Target column name
#   --outdir (required)   Output directory
#   --seeds               Comma-separated seed list (default: 1,2,3,5,8)
#   --time_budget         Seconds per run (default: 60 from config)
#   --dataset_name        Label for dataset (default: CSV stem)
#   --test_size           Train/test split ratio (default: 0.2)
#   --hparams_json        JSON string of KAO hyperparameter overrides
#   --standardize         Standardize features
#   --standardize-y       Also standardize target
#   --verbose             Verbose output

SEEDS30=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30

# MIMIC-IV ICU
python scripts/run_kao.py \
  --csv data/mimic_iv/ICU_composite_risk_score.csv \
  --target composite_risk_score \
  --seeds $SEEDS30 --time_budget 60 --dataset_name mimic_iv \
  --outdir results/mimic_iv/60s/kao

# eICU
python scripts/run_kao.py \
  --csv data/eicu/eICU_composite_risk_score.csv \
  --target composite_risk_score \
  --seeds $SEEDS30 --time_budget 60 --dataset_name eicu \
  --outdir results/eicu/60s/kao

# NHANES
python scripts/run_kao.py \
  --csv data/nhanes/NHANES_metabolic_score.csv \
  --target metabolic_score \
  --seeds $SEEDS30 --time_budget 60 --dataset_name nhanes \
  --outdir results/nhanes/60s/kao

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
# PHASE 2: Baselines on all datasets (30 seeds x 60s)
# -> Table 1, Table 3, Figure 2
# Estimated time: ~1 h per dataset x 7 = ~8 h
# ============================================================
# run_baselines.py args:
#   --csv (required)          Path to CSV data file
#   --target (required)       Target column name
#   --experiments (required)  pysr|rils_rols|gplearn|operon|all
#   --outdir (required)       Output directory
#   --seeds                   Comma-separated seed list (default: 1,2,3,5,8)
#   --time_budget             Seconds per run (default: 60 from config)
#   --dataset_name            Label for dataset (default: CSV stem)
#   --test_size               Train/test split ratio (default: 0.2)
#   --hparams_json            JSON string of hyperparameter overrides
#   --standardize             Standardize features
#   --standardize-y           Also standardize target
#   --verbose                 Verbose output

# MIMIC-IV ICU
python scripts/run_baselines.py \
  --csv data/mimic_iv/ICU_composite_risk_score.csv \
  --target composite_risk_score \
  --experiments all --seeds $SEEDS30 --time_budget 60 \
  --dataset_name mimic_iv --outdir results/mimic_iv/60s

# eICU
python scripts/run_baselines.py \
  --csv data/eicu/eICU_composite_risk_score.csv \
  --target composite_risk_score \
  --experiments all --seeds $SEEDS30 --time_budget 60 \
  --dataset_name eicu --outdir results/eicu/60s

# NHANES
python scripts/run_baselines.py \
  --csv data/nhanes/NHANES_metabolic_score.csv \
  --target metabolic_score \
  --experiments all --seeds $SEEDS30 --time_budget 60 \
  --dataset_name nhanes --outdir results/nhanes/60s

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
# PHASE 3: Ablation study (30 seeds x 60s)
# -> Table 5 (ablation), Figure 5 (ablation bars)
# NOTE: v4 fix -- cv_seed=seed is now passed automatically
# Estimated time: ~1.5 h per dataset x 3 = ~5 h
# ============================================================
# run_ablation.py args:
#   --csv (required)     Path to CSV data file
#   --target (required)  Target column name
#   --seeds              Number of seeds 1..N (int, default: 30)
#   --time_budget        Seconds per run (default: 60.0)
#   --dataset_name       Label for dataset (default: CSV stem)
#   --test_size          Train/test split ratio (default: 0.2)
#   --conditions         Subset of condition names (default: all 5)
#   --outdir             Output directory (default: results/ablation)
#   --skip_plots         Skip figure generation

# MIMIC-IV  (v4: use --outdir results/v4_ablation)
python scripts/run_ablation.py \
  --csv data/mimic_iv/ICU_composite_risk_score.csv \
  --target composite_risk_score \
  --seeds 30 --time_budget 60 --dataset_name mimic_iv \
  --outdir results/v4_ablation

# eICU
python scripts/run_ablation.py \
  --csv data/eicu/eICU_composite_risk_score.csv \
  --target composite_risk_score \
  --seeds 30 --time_budget 60 --dataset_name eicu \
  --outdir results/v4_ablation

# NHANES
python scripts/run_ablation.py \
  --csv data/nhanes/NHANES_metabolic_score.csv \
  --target metabolic_score \
  --seeds 30 --time_budget 60 --dataset_name nhanes \
  --outdir results/v4_ablation

# ============================================================
# PHASE 4: Time-resolved Pareto front (30 seeds x 60s)
# -> SM Figure S12 (HV trend), SM Figure S13 (panels)
# Estimated time: ~1 h per dataset x 3 = ~3 h
# ============================================================
# run_pareto_time.py args:
#   --csv (required)     Path to CSV data file
#   --target (required)  Target column name
#   --seeds              Number of seeds 1..N (int, default: 30)
#   --time_budget        Seconds per run (default: 60.0)
#   --dataset_name       Label for dataset (default: CSV stem)
#   --test_size          Train/test split ratio (default: 0.2)
#   --checkpoints        Time checkpoints in seconds (default: 10 20 30 40 50 60)
#   --skip_plots         Skip figure generation
#   --save_snapshots     Save raw Pareto snapshots as JSON

# MIMIC-IV
python scripts/run_pareto_time.py \
  --csv data/mimic_iv/ICU_composite_risk_score.csv \
  --target composite_risk_score \
  --seeds 30 --time_budget 60 --dataset_name mimic_iv

# eICU
python scripts/run_pareto_time.py \
  --csv data/eicu/eICU_composite_risk_score.csv \
  --target composite_risk_score \
  --seeds 30 --time_budget 60 --dataset_name eicu

# NHANES
python scripts/run_pareto_time.py \
  --csv data/nhanes/NHANES_metabolic_score.csv \
  --target metabolic_score \
  --seeds 30 --time_budget 60 --dataset_name nhanes

# ============================================================
# PHASE 5: SRBench synthetic benchmarks (30 seeds x 60s)
# -> Table 6 (SRBench 10 tasks), SM Figure S17 (heatmap)
# Estimated time: ~3 h (sequential), ~30 min (parallel)
# ============================================================
# run_srbench.py args (sequential):
#   --benchmark          Benchmark keys (default: all in BENCHMARKS)
#   --methods            Method names (default: KAO + all baselines)
#   --seeds              Number of seeds 1..N (int, default: 30)
#   --time_budget        Seconds per run (default: 60.0)
#   --noise_levels       Noise std levels (default: 0.0 0.1)
#   --skip_plots         Skip figure generation
#   --skip_baselines     Run KAO only
#
# run_srbench_parallel.py args (parallel):
#   --workers            Number of parallel workers (default: 25)
#   --benchmarks         Benchmark keys or 'NEW'/'ALL' (default: NEW)
#   --seeds              Number of seeds 1..N (int, default: 30)
#   --time_budget        Seconds per run (default: 60.0)
#   --dry_run            Print tasks without running

# Sequential (all benchmarks, both noise levels)
python scripts/run_srbench.py --seeds 30 --time_budget 60

# Parallel (25 workers, all 10 benchmarks)
python scripts/run_srbench_parallel.py --workers 25 --benchmarks ALL --seeds 30 --time_budget 60

# Parallel (new 5 benchmarks only)
python scripts/run_srbench_parallel.py --workers 25 --benchmarks NEW --seeds 30 --time_budget 60

# ============================================================
# PHASE 6: Complexity-matched baseline tuning
# -> Table 2 (MAE/RMSE), SM Table S7, Figure 4, SM Figure S24
# ============================================================
# run_complexity_matched.py args:
#   --step               tuning|eval|both (default: both)
#   --datasets           Dataset keys or 'all' (default: all)
#   --methods            Method names or 'all' (default: all)
#   --grid_config        Path to grid config JSON

python scripts/run_complexity_matched.py --step both --datasets all --methods all

# ============================================================
# PHASE 7: Weight sensitivity analysis
# -> SM Table S6
# ============================================================
# run_weight_sensitivity.py: NO CLI args (hardcoded)
#   Datasets: mimic_iv, eicu
#   Variants: original, uniform, perturbed
#   Seeds: 1-10
#   Budget: 60s

python scripts/run_weight_sensitivity.py

# ============================================================
# PHASE 8: Generate paper outputs
# ============================================================
# IMPORTANT: Run from /home/dengou/KAO_v4 (project root), NOT KAO/
# All analysis scripts below have HARDCODED paths and take no CLI args
# (except generate_paper_figures.py)

# --- 8a. Master analysis command (from within KAO/) ---
python analysis/generate_paper_figures.py --results-dir results/

# --- 8b. Individual top-level analysis scripts ---
# Run from: /home/dengou/KAO_v4
cd /home/dengou/KAO_v4

# Main results -> Table 1 (summary_main.csv -> summary_main.tex)
python scripts/01_summary_main.py

# R2 vs complexity scatter -> Figure 2
python scripts/02b_scatter_r2_complexity_v2.py

# Reachability -> Table 3 (reachability.csv -> Table_reachability.tex)
python scripts/03_reachability.py

# Representative expressions -> Table 4 (Table_expressions.tex)
python scripts/04_expressions.py

# SRBench table -> (srbench_formatted.csv -> Table_srbench.tex)
python scripts/05_srbench_table.py

# Pareto HV trend -> SM Figure S12
python scripts/06_pareto_time.py

# Complexity violin plots -> SM Figures S4-S10
python scripts/07_complexity_violin.py

# Pairwise statistical tests -> SM Table S1 (cd_diagram, heatmap, forest plot)
python scripts/08a_pairwise_table.py

# Runtime table -> SM Table S2
python scripts/08b_runtime_table.py

# SRBench numerical equivalence check
python scripts/09_srbench_numerical_equiv.py

# Accuracy-parsimony tradeoff -> Figure 3
python scripts/plot_tradeoff_r2loss_vs_nodesaving.py

# Ablation bars (v4 hybrid: KAO from main, No_KAO from v4_ablation) -> Figure 5
python scripts/r4a_ablation_bars.py

# Runtime figure -> SM Figure S18
python scripts/r4b_runtime_figure.py

# Reachability thresholds -> SM Table S3, SM Figure S19
python scripts/r4c_reachability_thresholds.py

# Expressions audit
python scripts/r4d_expressions_audit.py

# eICU tradeoff -> SM Figure S20
python scripts/r4e_eicu_tradeoff.py

# gplearn sensitivity
python scripts/r4f_gplearn_sensitivity.py

# Pareto evolution panels -> SM Figure S13
python scripts/reflow_pareto_panels.py

# --- 8c. Ablation rebuild (v4 hybrid data) ---
# Reads KAO from main experiment, No_KAO from results/v4_ablation/
# -> ablation_summary.csv, ablation_pairwise.csv, ablation_table.tex
python scripts/rebuild_ablation_v4.py

# --- 8d. KAO/analysis scripts (hardcoded, run from /home/dengou/KAO_v4) ---

# Complexity-matched outputs -> SM Table S7, Figure 4, SM Figure S24
python KAO/analysis/generate_complexity_matched_outputs.py

# MAE/RMSE + calibration -> Table 2, SM Table S8, SM Figures S21-S23
python KAO/analysis/generate_mae_rmse_calibration.py

# Missingness table -> SM Table S5
python KAO/analysis/generate_missingness_table.py

# SRBench 10-task outputs -> Table 6, SM Figure S17
python KAO/analysis/generate_srbench_10task_outputs.py

# SRBench task descriptors -> SM Table S4
python KAO/analysis/generate_srbench_task_descriptors.py

# Weight sensitivity -> SM Table S6
python KAO/analysis/generate_weight_sensitivity_output.py

# SRBench aggregation
python KAO/analysis/aggregate_srbench_results.py

# ============================================================
# 9. VALIDATION
# ============================================================

python scripts/99_final_check.py

# ============================================================
# 10. COMPILE PAPER
# ============================================================
cd /home/dengou/KAO_v4/paper_KAO_SWEVO

# Main manuscript (pdflatex -> bibtex -> pdflatex -> pdflatex)
pdflatex -interaction=nonstopmode KAO_SWEVO_v4.tex
bibtex KAO_SWEVO_v4
pdflatex -interaction=nonstopmode KAO_SWEVO_v4.tex
pdflatex -interaction=nonstopmode KAO_SWEVO_v4.tex

# Supplementary material
pdflatex -interaction=nonstopmode KAO_SM_v4.tex
pdflatex -interaction=nonstopmode KAO_SM_v4.tex

# TRIPOD+AI checklist
pdflatex -interaction=nonstopmode TRIPOD_AI_checklist_v4.tex

# Cover letter
pdflatex -interaction=nonstopmode cover_letter_v4.tex

# ============================================================
# INDIVIDUAL SCRIPT CLI REFERENCE
# ============================================================

# --- run_kao.py ---
# --csv (str, required)        Path to CSV data file
# --target (str, required)     Target column name
# --outdir (str, required)     Output directory for results
# --seeds (str, default: "1,2,3,5,8")  Comma-separated seed list
# --time_budget (float)        Seconds per run (default 60 from config)
# --dataset_name (str)         Label for dataset (default: CSV stem)
# --test_size (float, 0.2)     Train/test split ratio
# --hparams_json (str)         JSON string of KAO hyperparameter overrides
# --standardize                Standardize features
# --standardize-y              Also standardize target
# --verbose                    Verbose output

# --- run_baselines.py ---
# --csv (str, required)        Path to CSV data file
# --target (str, required)     Target column name
# --experiments (str, required) pysr|rils_rols|gplearn|operon|all
# --outdir (str, required)     Output directory
# --seeds (str, default: "1,2,3,5,8")  Comma-separated seed list
# --time_budget (float)        Seconds per run
# --dataset_name (str)         Label for dataset
# --test_size (float, 0.2)
# --hparams_json (str)         JSON string of hyperparameter overrides
# --standardize / --standardize-y / --verbose

# --- run_ablation.py ---
# --csv (str, required)        Path to CSV data file
# --target (str, required)     Target column name
# --seeds (int, default: 30)   Number of seeds (1..N)
# --time_budget (float, 60.0)  Seconds per run
# --dataset_name (str)         Label for dataset
# --test_size (float, 0.2)
# --conditions (list)          Subset of condition names (default: all 5)
# --outdir (str)               Output directory (default: results/ablation)
# --skip_plots                 Skip figure generation

# --- run_pareto_time.py ---
# --csv (str, required)        Path to CSV data file
# --target (str, required)     Target column name
# --seeds (int, default: 30)   Number of seeds (1..N)
# --time_budget (float, 60.0)  Seconds per run
# --dataset_name (str)
# --test_size (float, 0.2)
# --checkpoints (list of int)  Time checkpoints (default: 10 20 30 40 50 60)
# --skip_plots / --save_snapshots

# --- run_srbench.py ---
# --benchmark (list)           Benchmark keys (default: all)
# --methods (list)             Method names (default: KAO + all baselines)
# --seeds (int, default: 30)   Number of seeds (1..N)
# --time_budget (float, 60.0)  Seconds per run
# --noise_levels (list float)  Noise std levels (default: 0.0 0.1)
# --skip_plots / --skip_baselines

# --- run_srbench_parallel.py ---
# --workers (int, default: 25) Number of parallel workers
# --benchmarks (list)          Benchmark keys or 'NEW'/'ALL' (default: NEW)
# --seeds (int, default: 30)   Number of seeds (1..N)
# --time_budget (float, 60.0)
# --dry_run                    Print task list without running

# --- run_complexity_matched.py ---
# --step (str)                 tuning|eval|both (default: both)
# --datasets (list)            Dataset keys or 'all' (default: all)
# --methods (list)             Method names or 'all' (default: all)
# --grid_config (str)          Path to grid config JSON

# --- run_weight_sensitivity.py ---
# NO CLI ARGUMENTS. Hardcoded: mimic_iv + eicu, 3 variants, 10 seeds, 60s.

# --- run_all.py ---
# --config (str, default: "configs/default.yaml")
# --phase (str)                all|kao|baselines|ablation|pareto|srbench|eicu|analysis
# --dry-run                    Show commands without executing
# --seeds (int)                Override seeds from config
# --time-budget (float)        Override time budget from config
# --skip-plots

# --- analysis/generate_paper_figures.py ---
# --results-dir (str, default: "results/")
# --skip-plots

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
#   Phase 1 (KAO):       7 datasets x 30 seeds = 210
#   Phase 2 (Baselines): 7 datasets x 4 methods x 30 seeds = 840
#   Phase 3 (Ablation):  3 datasets x 5 conditions x 30 seeds = 450
#   Phase 4 (Pareto):    3 datasets x 2 conditions x 30 seeds = 180
#   Phase 5 (SRBench):   10 benchmarks x 2 noises x ~5 methods x 30 seeds = ~3000
#   Total:               ~4680 result files
