# KAO: Budget-Reachable Symbolic Regression via Typed Quadratic Operators and Knee-Point Selection

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> KAO provides zero-configuration delivery of compact symbolic models; under nominal time budgets it reliably reaches single-digit expressions without per-dataset parsimony tuning.

## Overview

KAO is a strongly typed symbolic regression (SR) framework designed for budget-constrained settings where both wall-clock time and expression complexity are bounded. A univariate quadratic operator `KAO(x; a,b,c) = ax^2 + bx + c` serves as a typed primitive within strongly typed genetic programming, localizing nonlinearity while maintaining semantic validity. Combined with bi-objective Pareto search (NSGA-II) and automatic knee-point selection, KAO discovers compact, interpretable expressions without per-dataset parsimony tuning.

## Key Features

- **Budget reachability** -- Reliably produces models within a specified node
  budget (67--100% reachability for <=8 nodes across 7 datasets)
- **Zero-configuration** -- NSGA-II + knee-point selection automatically selects
  compact models; no per-dataset parsimony weight tuning required
- **Typed quadratic operator** -- Structural compression for polynomial
  interaction blocks via the strongly typed `KAO` leaf primitive
- **Multi-objective** -- Bi-objective search jointly optimizing cross-validated
  error and symbolic complexity via NSGA-II with epsilon-pruned Pareto archives
- **Reproducible** -- Fixed seeds, documented configs, all results stored as
  per-seed JSON files with full hyperparameter provenance

## Installation

### Prerequisites

| Item | Requirement |
|------|-------------|
| OS | Linux (Ubuntu 22.04 tested), macOS 13+ |
| Python | 3.10+ (tested on 3.10.19 and 3.12.3) |
| RAM | >= 16 GB recommended |
| Disk | ~2 GB for full results |

### Setup (Conda -- recommended)

```bash
git clone https://github.com/oudeng/KAO.git
cd KAO
conda env create -f env_setup/env_kao310.yml
conda activate kao310

# rils-rols requires separate install (pybind11 build)
pip install rils-rols>=1.6.0 --no-build-isolation
```

### Setup (pip only)

```bash
pip install -r requirements.txt
```

See `env_setup/README_env.md` for detailed instructions and troubleshooting.

### Key dependencies

`deap`, `numpy`, `scipy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`,
`kneed`, `sympy`, `tqdm`, `pyyaml`, `pysr`, `gplearn`, `pyoperon` (0.4.0)

### Verify installation

```bash
python -c "from kao.KAO_v3_1 import run_single; print('KAO OK')"
python -c "from baselines.registry import get_all_baselines; print([b.name for b in get_all_baselines()])"
```

## Project Structure

```
KAO/
├── kao/                          # Core KAO implementation
│   ├── KAO_v3_1.py              # Main algorithm: GP evolution, run_single(), KAOResult
│   └── shared_classes.py        # StandardizedIndividual, Fitness classes
│
├── baselines/                    # Baseline SR method wrappers
│   ├── pysr_wrapper.py          # PySR (Julia-based)
│   ├── rils_rols_wrapper.py     # RILS-ROLS
│   ├── gplearn_wrapper.py       # gplearn (scikit-learn-compatible GP)
│   ├── operon_wrapper.py        # Operon C++ (via pyoperon)
│   └── registry.py              # get_all_baselines(), get_baseline_by_name()
│
├── scripts/                      # Experiment runners
│   ├── run_kao.py               # KAO on any CSV dataset
│   ├── run_baselines.py         # All 4 baselines on any CSV dataset
│   ├── run_ablation.py          # Ablation study (KAO vs No_KAO variants)
│   ├── run_pareto_time.py       # Time-resolved Pareto front (HV analysis)
│   ├── run_srbench.py           # SRBench synthetic benchmarks (sequential)
│   ├── run_srbench_parallel.py  # SRBench synthetic benchmarks (parallel)
│   ├── run_complexity_matched.py# Complexity-matched baseline grid search
│   ├── run_weight_sensitivity.py# Weight-sensitivity analysis (hardcoded)
│   └── run_all.py               # Master runner (all phases via YAML config)
│
├── analysis/                     # Paper figure/table generation
│   ├── generate_paper_figures.py # One-command figure + table generation
│   ├── generate_mae_rmse_calibration.py    # MAE/RMSE tables + calibration plots
│   ├── generate_complexity_matched_outputs.py # Complexity-matched tables/figures
│   ├── generate_srbench_10task_outputs.py   # SRBench 10-task table/heatmap
│   ├── generate_srbench_task_descriptors.py # SRBench benchmark metadata table
│   ├── generate_missingness_table.py        # Preprocessing missingness table
│   ├── generate_weight_sensitivity_output.py# Weight-sensitivity table
│   ├── aggregate_srbench_results.py         # SRBench results aggregator
│   ├── statistics.py            # StatisticalAnalyzer (Wilcoxon, Friedman, etc.)
│   └── visualization.py         # Plot functions (scatter, heatmap, CD, violin)
│
├── configs/                      # Experiment configurations
│   ├── default.yaml             # Central config (seeds, budget, datasets, conditions)
│   ├── complexity_matched_grid.json # Grid-search space for complexity-matched tuning
│   └── {dataset}_{method}_{budget}.json # Per-dataset/method overrides
│
├── data/                         # Datasets and extraction scripts
│   ├── ~~mimic_iv/~~            ~~# MIMIC-IV ICU (2939 x 20, credentialed)~~
│   ├── ~~eicu/~~                ~~# eICU ICU (4536 x 9, credentialed)~~
│   ├── nhanes/                  # NHANES metabolic (2281 x 11, public)
│   ├── uci/                     # UCI benchmarks (4 CSVs, public)
│   ├── srbench_synthetic/       # SRBench benchmark generator
│   └── README_data.md           # Data sources, ethics, access instructions
│
├── utils/                        # Shared utilities
│   ├── hparams.py               # merge_hparams(defaults, json, cli)
│   ├── result_io.py             # write_result_json() with numpy/NaN safety
│   └── cv_eval.py               # kfold_eval_fixed_model()
│
├── tests/                        # pytest test suite
│   ├── test_kao.py              # KAO core functionality
│   ├── test_baselines.py        # Baseline wrapper tests
│   ├── test_srbench.py          # SRBench data generation
│   ├── test_v32_upgrades.py     # v3.2 regression tests
│   └── test_v321_regressions.py # v3.2.1 regression tests
│
├── env_setup/                    # Environment configuration
│   ├── env_kao310.yml           # Conda environment spec (Python 3.10)
│   └── README_env.md            # Conda setup guide
│
├── results/                      # Experiment outputs (generated at runtime)
│   ├── {dataset}/{budget}s/{method}/seed_{N}/result.json
│   ├── v4_ablation/             # v4 ablation re-run (450 JSON files)
│   ├── ablation/                # Original ablation (legacy, do not use)
│   ├── pareto_time/             # Time-resolved Pareto snapshots
│   ├── srbench/                 # SRBench results
│   ├── weight_sensitivity/      # Weight-sensitivity results
│   ├── figures/                 # Generated PDF figures
│   └── tables/                  # Generated CSV + LaTeX tables
│
├── requirements.txt
├── README.md                     # This file
├── readme_CL.txt                 # Complete command-line reference
├── CHANGELOG.md                  # Version history
└── LICENSE                       # MIT License
```

## Quick Start

### Run KAO on a single dataset

```bash
python scripts/run_kao.py \
  --csv data/nhanes/NHANES_metabolic_score.csv \
  --target metabolic_score \
  --seeds 1,2,3,4,5 --time_budget 60 \
  --dataset_name nhanes --outdir results/nhanes/60s/kao
```

### Run baselines on the same dataset

```bash
python scripts/run_baselines.py \
  --csv data/nhanes/NHANES_metabolic_score.csv \
  --target metabolic_score \
  --experiments all --seeds 1,2,3,4,5 --time_budget 60 \
  --dataset_name nhanes --outdir results/nhanes/60s
```

### Run ablation study

```bash
python scripts/run_ablation.py \
  --csv data/nhanes/NHANES_metabolic_score.csv \
  --target metabolic_score \
  --seeds 5 --time_budget 60 --dataset_name nhanes \
  --outdir results/v4_ablation
```

### Quick smoke test (all phases, ~5 min)

```bash
python scripts/run_all.py --seeds 5 --time-budget 10 --dry-run   # preview
python scripts/run_all.py --seeds 5 --time-budget 10              # execute
```

### Full paper reproduction (~24-48 h)

```bash
python scripts/run_all.py --seeds 30 --time-budget 60
```

Or run individual phases:

```bash
python scripts/run_all.py --phase kao          # KAO on all 7 datasets
python scripts/run_all.py --phase baselines    # All 4 baselines
python scripts/run_all.py --phase ablation     # Ablation (3 healthcare datasets)
python scripts/run_all.py --phase pareto       # Time-resolved Pareto front
python scripts/run_all.py --phase srbench      # SRBench synthetic
python scripts/run_all.py --phase analysis     # Generate all figures + tables
```

## Datasets

| Dataset | Type | Target | Samples | Features | Source | Access |
|---------|------|--------|---------|----------|--------|--------|
| MIMIC-IV ICU | Healthcare | composite_risk_score | 2939 | 20 | PhysioNet | Credentialed |
| eICU | Healthcare | composite_risk_score | 4536 | 9 | PhysioNet | Credentialed |
| NHANES | Healthcare | metabolic_score | 2281 | 11 | CDC | Public |
| Airfoil | UCI benchmark | SSPL | 1503 | 5 | UCI ML Repository | Public |
| Auto-MPG | UCI benchmark | mpg | 392 | 7 | UCI ML Repository | Public |
| Communities | UCI benchmark | ViolentCrimesPerPop | 1994 | 14 | UCI ML Repository | Public |
| Hydraulic | UCI benchmark | fault_score | 2205 | 44 | UCI ML Repository | Public |
| SRBench (x10) | Synthetic | Known ground truth | varies | 1-5 | Generated | Public |

MIMIC-IV and eICU require PhysioNet credentialed access. Extraction scripts are
provided for transparency. NHANES, UCI, and SRBench datasets are included in the
repository. See `data/README_data.md` for details.

## Experiment Reproduction

### Full reproduction (7 datasets x 5 methods x 30 seeds)

```bash
# Step 1: Run KAO on all datasets (~3 h)
for ds in mimic_iv eicu nhanes airfoil auto_mpg communities hydraulic; do
  python scripts/run_kao.py \
    --csv $(python -c "import yaml; c=yaml.safe_load(open('configs/default.yaml')); ds=[d for g in ['healthcare','uci'] for d in c['datasets'][g] if d['name']=='$ds'][0]; print(ds['csv'])") \
    --target $(python -c "import yaml; c=yaml.safe_load(open('configs/default.yaml')); ds=[d for g in ['healthcare','uci'] for d in c['datasets'][g] if d['name']=='$ds'][0]; print(ds['target'])") \
    --seeds 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30 \
    --time_budget 60 --dataset_name $ds --outdir results/$ds/60s/kao
done

# Step 2: Run baselines (~8 h)
for ds in mimic_iv eicu nhanes airfoil auto_mpg communities hydraulic; do
  python scripts/run_baselines.py \
    --csv ... --target ... \
    --experiments all --seeds ... --time_budget 60 \
    --dataset_name $ds --outdir results/$ds/60s
done

# Or use the master runner:
python scripts/run_all.py --phase kao --seeds 30 --time-budget 60
python scripts/run_all.py --phase baselines --seeds 30 --time-budget 60
```

### Ablation reproduction (3 healthcare x 5 conditions x 30 seeds)

```bash
# v4 fix: cv_seed=seed is now passed automatically
for ds_csv_target in \
  "mimic_iv data/mimic_iv/ICU_composite_risk_score.csv composite_risk_score" \
  "eicu data/eicu/eICU_composite_risk_score.csv composite_risk_score" \
  "nhanes data/nhanes/NHANES_metabolic_score.csv metabolic_score"; do
  set -- $ds_csv_target
  python scripts/run_ablation.py \
    --csv $2 --target $3 \
    --seeds 30 --time_budget 60 --dataset_name $1 \
    --outdir results/v4_ablation
done
```

### Complexity-matched baseline tuning

```bash
python scripts/run_complexity_matched.py --step both --datasets all --methods all
```

### SRBench benchmarks

```bash
# Sequential (all 10 benchmarks, 2 noise levels)
python scripts/run_srbench.py --seeds 30 --time_budget 60

# Parallel (25 workers)
python scripts/run_srbench_parallel.py --workers 25 --benchmarks ALL --seeds 30 --time_budget 60
```

## Configuration

Central configuration via `configs/default.yaml`:

```yaml
experiment:
  seeds: 30
  time_budget: 60          # nominal wall-clock seconds per run
  n_jobs: 1                # single-threaded for fair comparison

kao:
  use_kao_leaf: true       # enable typed quadratic operator
  max_complexity: null     # no explicit cap (NSGA-II handles trade-off)
  time_checkpoints: [10, 20, 30, 40, 50, 60]

ablation:
  conditions:
    - {name: "KAO",          use_kao_leaf: true,  max_complexity: null}
    - {name: "No_KAO",       use_kao_leaf: false, max_complexity: null}
    - {name: "No_KAO_cap7",  use_kao_leaf: false, max_complexity: 7}
    - {name: "No_KAO_cap9",  use_kao_leaf: false, max_complexity: 9}
    - {name: "No_KAO_cap11", use_kao_leaf: false, max_complexity: 11}

baselines:
  enabled: ["PySR", "RILS-ROLS", "gplearn", "Operon"]
```

Per-dataset/method overrides in `configs/*.json` control population size,
mutation rates, and other hyperparameters.

## Analysis and Paper Figures

Generate all tables and figures from experiment results:

```bash
# One-command generation (requires all experiments completed)
python analysis/generate_paper_figures.py --results-dir results/

# Or run individual analysis scripts (hardcoded paths, no CLI args):
cd /path/to/KAO_v4
python scripts/01_summary_main.py           # Table 1: main comparison
python scripts/03_reachability.py            # Table 3: reachability
python scripts/r4a_ablation_bars.py          # Figure 5: ablation bars
python scripts/rebuild_ablation_v4.py        # Table 5: ablation (v4 hybrid data)
python KAO/analysis/generate_mae_rmse_calibration.py   # Table 2: MAE/RMSE
python KAO/analysis/generate_srbench_10task_outputs.py  # Table 6: SRBench
```

See `readme_CL.txt` for the complete list of analysis commands and their
output file mappings.

## Key Implementation Details

- **Complexity counting**: Weighted mode; each KAO node costs 2 (configurable).
  Other primitives cost 1 per node.
- **cv_seed alignment**: Cross-validation fold assignment uses `cv_seed = seed`
  (fixed in v4; previously ablation used a fixed `cv_seed = 2025`).
- **Knee-point selection**: Kneedle algorithm on the normalized NSGA-II Pareto
  front, with a window size of 3.
- **Time budget**: Checked at generation boundaries (nominal, not hard-stop).
  Actual runtimes may be 60-65 s due to evaluation overhead.
- **Model selection**: NSGA-II with epsilon-pruned archive (epsilon_MSE = 1e-3,
  epsilon_comp = 1), then local coefficient refitting on top-K=25 elites.

## Testing

```bash
pytest tests/ -v
pytest tests/test_kao.py -v          # KAO core (10s budget per test)
pytest tests/test_baselines.py -v    # Baseline wrappers
pytest tests/test_srbench.py -v      # SRBench data generation
```

## Expected Results

Under matched 60-second, single-threaded wall-clock budgets with 30 random seeds:

| Dataset | KAO R^2 (mean +/- std) | KAO nodes (mean +/- std) | Reachability (<=8 nodes) |
|---------|------------------------|--------------------------|--------------------------|
| MIMIC-IV | 0.644 +/- 0.026 | 6.4 +/- 1.0 | 30/30 (100%) |
| eICU | 0.507 +/- 0.050 | 7.4 +/- 1.7 | 24/30 (80%) |
| NHANES | 0.514 +/- 0.020 | 6.6 +/- 1.9 | 22/30 (73%) |
| Airfoil | 0.502 +/- 0.031 | (varies) | 20/30 (67%) |
| Auto MPG | 0.839 +/- 0.004 | (varies) | 27/30 (90%) |
| Communities | 0.567 +/- 0.013 | (varies) | 26/30 (87%) |
| Hydraulic | 0.548 +/- 0.031 | (varies) | 23/30 (77%) |

## Citation

If you use KAO in your research, please cite:

```bibtex
@article{deng2026kao,
  title   = {KAO: Budget-Reachable Symbolic Regression via Typed Quadratic
             Operators and Knee-Point Selection},
  author  = {Deng, Ou and Nishimura, Shoji and Ogihara, Atsushi and Jin, Qun},
  journal = {TBD},
  year    = {2026},
  note    = {Under review}
}
```

## License

MIT License. See [LICENSE](LICENSE).

## Change Log

See [CHANGELOG.md](CHANGELOG.md) for full version history.
