# KAO: Parsimonious Symbolic Regression via Typed Quadratic Operators and Knee-Point Selection

[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Overview

KAO is an operator-centric symbolic regression framework that achieves
interpretability through parsimony. A univariate quadratic Kolmogorov–Arnold
operator `KAO(x; a,b,c) = ax² + bx + c` serves as a typed primitive within
strongly-typed genetic programming, localizing nonlinearity while maintaining
semantic validity. Combined with bi-objective Pareto search (NSGA-II) and
knee-point selection, KAO discovers compact, interpretable mathematical
expressions.

**Key Features:**
- Typed Genetic Programming — Two base types (Real and TripleParam) with the quadratic KAO leaf node
- Multi-objective Optimization — Jointly optimizes cross-validated error and symbolic complexity via NSGA-II with epsilon-pruned Pareto archives
- Knee-point Selection — Automatically selects models at the optimal accuracy–parsimony trade-off
- Interpretable Output — Produces short, auditable formulas (typically 6–20 nodes) aligned with domain knowledge
- Time-checkpoint Snapshots — Captures Pareto front evolution at configurable intervals for search-efficiency analysis

---

## Reproducing the Paper Results (Reviewer Guide)

This section provides a step-by-step guide for reviewers to reproduce all
tables and figures reported in the manuscript.

### Prerequisites

| Item | Requirement |
|------|-------------|
| OS | Linux (Ubuntu 22.04 tested), macOS 13+ |
| Python | 3.10 (via Conda; see `env_setup/`) |
| RAM | ≥ 16 GB recommended |
| Disk | ~2 GB for results |
| Wall-clock | ~24–48 h for full 30-seed reproduction (single-threaded) |

### Step 1 — Clone and enter the repository

```bash
git clone https://github.com/oudeng/KAO.git
cd KAO
```

### Step 2 — Create the Conda environment

```bash
# Option A: Conda (recommended; includes Julia for PySR)
conda env create -f env_setup/env_kao310.yml
conda activate kao310
# rils-rols requires separate install (pybind11 build issue)
pip install rils-rols>=1.6.0 --no-build-isolation

# Option B: pip only (no Operon/rils-rols)
pip install -r requirements.txt
```

See `env_setup/README_env.md` for detailed instructions and troubleshooting.

### Step 3 — Verify the installation

```bash
python -c "from kao.KAO_v3_1 import run_single; print('KAO OK')"
python -c "from baselines.registry import get_all_baselines; print([b.name for b in get_all_baselines()])"
python -c "from data.srbench_synthetic.generate import BENCHMARKS; print(list(BENCHMARKS.keys()))"
python -c "from analysis.statistics import StatisticalAnalyzer; print('Statistics OK')"
```

### Step 4 — Quick smoke test (≈ 5 min)

Runs all phases with 5 seeds and 10-second budgets to verify the pipeline:

```bash
python scripts/run_all.py --seeds 5 --time-budget 10 --dry-run   # preview commands
python scripts/run_all.py --seeds 5 --time-budget 10              # execute
```

### Step 5 — Full reproduction (≈ 24–48 h)

Reproduces the paper's 30-seed, 60-second experiments across all 7 datasets,
ablation, Pareto, and SRBench phases, then generates all figures and tables:

```bash
python scripts/run_all.py --seeds 30 --time-budget 60
```

Alternatively, run individual phases (see "Running Individual Phases" below):

```bash
python scripts/run_all.py --phase kao          # Phase 1: KAO on all datasets
python scripts/run_all.py --phase baselines    # Phase 2: All baselines
python scripts/run_all.py --phase ablation     # Phase 3: Ablation study
python scripts/run_all.py --phase pareto       # Phase 4: Time-resolved Pareto
python scripts/run_all.py --phase srbench      # Phase 5: SRBench synthetic
python scripts/run_all.py --phase eicu         # Phase 6: eICU preprocessing + runs
python scripts/run_all.py --phase analysis     # Phase 7: Figures + tables
```

### Step 6 — Verify outputs

After completion, check the generated outputs:

```bash
ls results/figures/*.pdf       # All paper figures
ls results/tables/*.tex        # LaTeX tables
ls results/tables/*.csv        # CSV summaries
```

### Paper–Code Correspondence Map

| Paper Item | Script / Command | Output |
|---|---|---|
| **Table 1** (Main comparison: R², nodes) | `run_all.py --phase kao` + `--phase baselines` + `--phase analysis` | `results/tables/summary_main.tex`, `results/tables/summary_main.csv` |
| **Table 2** (Reachability ≤ 8 nodes) | Same as Table 1 → analysis | `results/tables/Table_reachability.tex` |
| **Table 3** (Representative expressions) | Same as Table 1 → analysis | `results/tables/Table_expressions.tex` |
| **Table 4** (Ablation) | `run_all.py --phase ablation` + `--phase analysis` | `results/tables/ablation_table.tex` |
| **Table 5** (SRBench) | `run_all.py --phase srbench` + `--phase analysis` | `results/tables/Table_srbench.tex` |
| **Fig. 1** (KAO framework) | Static diagram (not generated) | `outputs/figures/KAO_framework_cropped.pdf` |
| **Fig. 2** (R² vs complexity scatter) | `--phase analysis` | `results/figures/Fig_r2_vs_complexity_healthcare.pdf`, `Fig_r2_vs_complexity_uci.pdf` |
| **Fig. 3** (Ablation bars) | `--phase analysis` | `results/figures/ablation_bars.pdf` |
| **Fig. 4** (SRBench heatmap) | `--phase analysis` | `results/figures/srbench_heatmap.pdf` |
| **Suppl. Table S1** (Pairwise tests) | `--phase analysis` | `results/tables/Table_pairwise_full.tex` |
| **Suppl. Table S2** (Reachability thresholds) | `--phase analysis` | `results/tables/Table_reachability_thresholds.tex` |
| **Suppl. Table S3** (Runtime summary) | `--phase analysis` | `results/tables/Table_runtime.tex` |
| **Suppl. Fig. S1** (CD diagram) | `--phase analysis` | `results/figures/cd_diagram.pdf` |
| **Suppl. Fig. S2–S3** (Pairwise heatmap, forest plot) | `--phase analysis` | `results/figures/pairwise_heatmap.pdf`, `forest_plot.pdf` |
| **Suppl. Fig. S4–S10** (Violin plots × 7 datasets) | `--phase analysis` | `results/figures/violin_*.pdf` |
| **Suppl. Fig. S11–S13** (Convergence) | `--phase analysis` | `results/figures/convergence_*.pdf` |
| **Suppl. Fig. S14** (Pareto panels) | `--phase analysis` | `results/figures/pareto_evolution_panels.pdf` |
| **Suppl. Fig. S15** (Runtime distribution) | `--phase analysis` | `results/figures/Fig_runtime_by_dataset.pdf` |
| **Suppl. Fig. S16** (Reachability heatmap) | `--phase analysis` | `results/figures/Fig_reachability_thresholds.pdf` |

---

## Repository Structure

```
KAO/
├── kao/                          # Core KAO implementation
│   ├── __init__.py
│   ├── KAO_v3_1.py              # KAO algorithm (GP, evolution, run_single)
│   └── shared_classes.py        # StandardizedIndividual / Fitness
│
├── baselines/                    # Baseline SR methods
│   ├── __init__.py              # BaselineSR abstract base class
│   ├── pysr_wrapper.py          # PySR wrapper
│   ├── rils_rols_wrapper.py     # RILS-ROLS wrapper
│   ├── gplearn_wrapper.py       # gplearn wrapper
│   ├── operon_wrapper.py        # Operon C++ wrapper (optional)
│   └── registry.py              # get_all_baselines(), get_baseline_by_name()
│
├── scripts/                      # Experiment runners
│   ├── run_kao.py               # KAO on any CSV dataset
│   ├── run_baselines.py         # PySR / RILS-ROLS / gplearn / Operon
│   ├── run_ablation.py          # Complexity-capped No-KAO ablation
│   ├── run_pareto_time.py       # Time-resolved Pareto front (HV analysis)
│   ├── run_srbench.py           # SRBench synthetic benchmarks
│   └── run_all.py               # Master runner (all phases)
│
├── analysis/                     # Statistics & visualization
│   ├── __init__.py
│   ├── statistics.py            # StatisticalAnalyzer (Wilcoxon, Friedman, etc.)
│   ├── visualization.py         # Plot functions (scatter, heatmap, CD, violin, etc.)
│   └── generate_paper_figures.py # One-click figure + table generation
│
├── data/                         # Datasets & preprocessing scripts
│   ├── README_data.md           # Data sources & ethics statement
│   ├── mimic_iv/                # MIMIC-IV ICU composite risk score (2939 × 20)
│   ├── nhanes/                  # NHANES metabolic score (2281 × 11)
│   ├── eicu/                    # eICU composite risk score (4536 × 9)
│   ├── uci/                     # UCI benchmarks (airfoil, auto-mpg, communities, hydraulic)
│   └── srbench_synthetic/       # SRBench standard benchmark generator
│
├── configs/                      # Experiment configurations
│   ├── default.yaml             # Central config for run_all.py
│   └── *.json                   # Per-dataset/method hyperparameter overrides
│
├── utils/                        # Shared utilities
│   ├── hparams.py               # merge_hparams(defaults, json, cli)
│   ├── result_io.py             # write_result_json() with numpy safety
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
│   ├── README_env.md            # Conda setup guide (Ubuntu 22.04)
│   └── env_kao310.yml           # Conda environment spec (Python 3.10)
│
├── results/                      # Experiment outputs (generated at runtime)
│   ├── {dataset}/{budget}s/{method}/seed_{N}/
│   │   └── result.json          # Per-seed metrics: r2_test, complexity_nodes, etc.
│   ├── ablation/                # Ablation study JSON logs
│   ├── pareto_time/             # Pareto front snapshots & HV data
│   ├── srbench/                 # SRBench results
│   ├── figures/                 # PDF figures (generated by analysis phase)
│   └── tables/                  # CSV + LaTeX tables (generated by analysis phase)
│
├── requirements.txt
├── README.md                     # This file
├── readme_CL.txt                 # Command-line quick reference (all commands)
├── CHANGELOG.md                  # Version history
└── LICENSE                       # MIT License
```

## Datasets

| Dataset | Type | Target | Samples | Features | Source |
|---|---|---|---|---|---|
| MIMIC-IV ICU | Healthcare | composite_risk_score | 2939 | 20 | PhysioNet (credentialed) |
| NHANES | Healthcare | metabolic_score | 2281 | 11 | CDC (public) |
| eICU | Healthcare | composite_risk_score | 4536 | 9 | PhysioNet (credentialed) |
| UCI Airfoil | Benchmark | SSPL | 1503 | 5 | UCI ML Repository |
| UCI Auto-MPG | Benchmark | mpg | 392 | 7 | UCI ML Repository |
| UCI Communities | Benchmark | ViolentCrimesPerPop | 1994 | 14 | UCI ML Repository |
| UCI Hydraulic | Benchmark | fault_score | 2205 | 44 | UCI ML Repository |
| SRBench (×5) | Synthetic | Known ground truth | varies | 1–5 | Generated |

All preprocessed CSV files are included in the repository under `data/`.
MIMIC-IV and eICU extraction scripts are provided for transparency (require
credentialed PhysioNet access). See `data/README_data.md` for details.

## Configuration

Central configuration via `configs/default.yaml`:

```yaml
experiment:
  seeds: 30
  time_budget: 60
  n_jobs: 1        # single-threaded for fair comparison

kao:
  use_kao_leaf: true
  time_checkpoints: [10, 20, 30, 40, 50, 60]

ablation:
  conditions:
    - {name: "KAO",          use_kao_leaf: true,  max_complexity: null}
    - {name: "No_KAO",       use_kao_leaf: false, max_complexity: null}
    - {name: "No_KAO_cap7",  use_kao_leaf: false, max_complexity: 7}
    - {name: "No_KAO_cap9",  use_kao_leaf: false, max_complexity: 9}
    - {name: "No_KAO_cap11", use_kao_leaf: false, max_complexity: 11}
```

Per-dataset/method JSON overrides in `configs/*.json` control population size,
mutation rates, and other hyperparameters.

## Testing

```bash
pytest tests/ -v
pytest tests/test_kao.py -v          # KAO core (10s budget per test)
pytest tests/test_baselines.py -v    # Baseline wrappers
pytest tests/test_srbench.py -v      # SRBench data generation
```

## Expected Results

Under matched 60-second, single-threaded wall-clock budgets with 30 random seeds:

| Dataset | KAO R² (mean ± std) | KAO nodes (mean ± std) |
|---|---|---|
| MIMIC-IV ICU | 0.644 ± 0.026 | 6.4 ± 1.0 |
| NHANES | 0.515 ± 0.014 | 7.2 ± 2.4 |
| eICU | 0.695 ± 0.022 | 6.0 ± 0.7 |

KAO retains ~80–85% of PySR's accuracy while using only ~40–50% of the
symbolic length, and produces expressions 4–5× shorter than RILS-ROLS.

## Citation

If you use KAO in your research, please cite:

```bibtex
@article{deng2026kao,
  title   = {KAO: Auditability-Oriented Symbolic Regression via Typed Quadratic Operators and Knee-Point Selection under Strict Time and Complexity Budgets},
  author  = {Deng, Ou and Nishimura, Shouji and Ogihara, Atsushi and Jin, Qun},
  journal = {arXiv},
  year    = {2026}
}
```

## License

MIT License. See [LICENSE](LICENSE).

## Change Log

See [CHANGELOG.md](CHANGELOG.md) for full version history.
