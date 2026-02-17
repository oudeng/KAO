#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Complexity-Capped No-KAO Ablation
==================================
Responds to the reviewer concern about Table 4: "No KAO R² > KAO R² —
what is the value of KAO?" by showing that the R² advantage of No KAO
disappears when both methods are constrained to the same complexity budget.

Experiment matrix
-----------------
Conditions : KAO / No_KAO / No_KAO_cap7 / No_KAO_cap9 / No_KAO_cap11
Datasets   : specified via --csv / --target (or built-in presets)
Seeds      : 30 (default)
Time budget: 60 s (default)

Outputs
-------
results/ablation/{dataset}_{condition}_seed{seed}.json   per-run logs
results/tables/ablation_summary.csv                      mean ± std
results/tables/ablation_pairwise.csv                     stat tests
results/figures/ablation_barplot.pdf                      grouped bars
results/figures/ablation_complexity_scatter.pdf           R² vs complexity

Usage
-----
  python scripts/run_ablation.py --csv data/mimic_iv/ICU_composite_risk_score.csv \\
         --target composite_risk_score --seeds 30 --time_budget 60
  python scripts/run_ablation.py --csv data/mimic_iv/ICU_composite_risk_score.csv \\
         --target composite_risk_score --seeds 3 --time_budget 10  # smoke test
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from kao.KAO_v3_1 import run_single, KAOResult  # noqa: E402
from utils.result_io import write_result_json  # noqa: E402

# ---------------------------------------------------------------------------
# Ablation condition definitions
# ---------------------------------------------------------------------------
ABLATION_CONFIGS: list[dict] = [
    {"name": "KAO",          "use_kao_leaf": True,  "max_complexity": None},
    {"name": "No_KAO",       "use_kao_leaf": False, "max_complexity": None},
    {"name": "No_KAO_cap7",  "use_kao_leaf": False, "max_complexity": 7},
    {"name": "No_KAO_cap9",  "use_kao_leaf": False, "max_complexity": 9},
    {"name": "No_KAO_cap11", "use_kao_leaf": False, "max_complexity": 11},
]

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Complexity-capped No-KAO ablation experiment",
    )
    p.add_argument("--csv", type=str, required=True,
                   help="Path to CSV data file")
    p.add_argument("--target", type=str, required=True,
                   help="Target column name")
    p.add_argument("--dataset_name", type=str, default=None,
                   help="Label for this dataset (default: stem of CSV path)")
    p.add_argument("--seeds", type=int, default=30,
                   help="Number of seeds (1..N)")
    p.add_argument("--time_budget", type=float, default=60.0,
                   help="Wall-clock seconds per run")
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--conditions", nargs="*", default=None,
                   help="Subset of condition names to run (default: all)")
    p.add_argument("--skip_plots", action="store_true",
                   help="Skip figure generation")
    return p.parse_args()

# ---------------------------------------------------------------------------
# Data loading (mirrors scripts/run_kao.py convention)
# ---------------------------------------------------------------------------

def load_data(csv_path: str, target: str, test_size: float = 0.2):
    df = pd.read_csv(csv_path).dropna()
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not in columns: {list(df.columns)}")
    feature_cols = [c for c in df.columns if c != target]
    X = df[feature_cols].values.astype(float)
    y = df[target].values.astype(float)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=2025,
    )
    return X_tr, X_te, y_tr, y_te, feature_cols

# ---------------------------------------------------------------------------
# Core experiment loop
# ---------------------------------------------------------------------------

def run_ablation(
    X_train, y_train, X_test, y_test,
    feature_names: list[str],
    dataset_name: str,
    seeds: list[int],
    time_budget: float,
    conditions: list[dict],
) -> pd.DataFrame:
    """Run all conditions × seeds and return a tidy DataFrame."""
    rows: list[dict] = []
    out_dir = Path("results") / "ablation"
    out_dir.mkdir(parents=True, exist_ok=True)

    total = len(conditions) * len(seeds)
    done = 0

    for cond in conditions:
        cname = cond["name"]
        for seed in seeds:
            done += 1
            print(f"[{done}/{total}] {cname}  seed={seed}", end=" … ", flush=True)

            result: KAOResult = run_single(
                X_train, y_train, X_test, y_test,
                dataset_name=dataset_name,
                seed=seed,
                time_budget=time_budget,
                use_kao_leaf=cond["use_kao_leaf"],
                max_complexity=cond["max_complexity"],
                time_checkpoints=[10, 20, 30, 40, 50, 60],
                feature_names=feature_names,
            )
            print(f"R²={result.r2_test:.4f}  nodes={result.complexity_nodes}")

            row = {
                "dataset": dataset_name,
                "condition": cname,
                "seed": seed,
                "r2_test": result.r2_test,
                "r2_cv": result.r2_cv,
                "complexity_nodes": result.complexity_nodes,
                "complexity_chars": result.complexity_chars,
                "runtime": result.runtime,
                "expression": result.expression,
            }
            rows.append(row)

            # Per-run JSON log (NaN/Inf → null, numpy → native)
            log_path = out_dir / f"{dataset_name}_{cname}_seed{seed}.json"
            write_result_json(log_path, row)

    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def make_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean ± std per condition per dataset."""
    metrics = ["r2_test", "r2_cv", "complexity_nodes", "complexity_chars", "runtime"]
    agg = {}
    for m in metrics:
        agg[f"{m}_mean"] = (m, "mean")
        agg[f"{m}_std"] = (m, "std")
        agg[f"{m}_median"] = (m, "median")
    summary = df.groupby(["dataset", "condition"]).agg(**agg).reset_index()
    return summary

# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def pairwise_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Wilcoxon signed-rank + Cliff's delta for KAO vs each No_KAO variant."""
    from scipy.stats import wilcoxon

    def cliffs_delta(x, y):
        """Cliff's delta effect size."""
        nx, ny = len(x), len(y)
        if nx == 0 or ny == 0:
            return 0.0
        more = sum(1 for xi in x for yi in y if xi > yi)
        less = sum(1 for xi in x for yi in y if xi < yi)
        return (more - less) / (nx * ny)

    results = []
    for dataset in df["dataset"].unique():
        d = df[df["dataset"] == dataset]
        kao_vals = d.loc[d["condition"] == "KAO", "r2_test"].values
        for cond in d["condition"].unique():
            if cond == "KAO":
                continue
            other_vals = d.loc[d["condition"] == cond, "r2_test"].values
            n_pairs = min(len(kao_vals), len(other_vals))
            if n_pairs < 5:
                continue
            a, b = kao_vals[:n_pairs], other_vals[:n_pairs]
            try:
                stat, pval = wilcoxon(a, b)
            except Exception:
                stat, pval = float("nan"), float("nan")
            cd = cliffs_delta(a, b)

            stars = ""
            if pval < 0.001:
                stars = "***"
            elif pval < 0.01:
                stars = "**"
            elif pval < 0.05:
                stars = "*"

            results.append({
                "dataset": dataset,
                "comparison": f"KAO vs {cond}",
                "wilcoxon_stat": stat,
                "p_value": pval,
                "significance": stars,
                "cliffs_delta": cd,
                "n_pairs": n_pairs,
            })

    return pd.DataFrame(results)

# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_ablation_barplot(df: pd.DataFrame, out_path: str):
    """Grouped barplot: R² per condition, with 95 % CI error bars."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
        "font.size": 10,
    })

    datasets = df["dataset"].unique()
    conditions = [c["name"] for c in ABLATION_CONFIGS]
    # Okabe-Ito colorblind-safe palette
    colors = ["#0072B2", "#D55E00", "#CC79A7", "#009E73", "#F0E442"]

    n_ds = len(datasets)
    fig, axes = plt.subplots(1, max(n_ds, 1), figsize=(5 * max(n_ds, 1), 4),
                             squeeze=False)

    for idx, ds in enumerate(datasets):
        ax = axes[0, idx]
        sub = df[df["dataset"] == ds]
        means, cis = [], []
        present_conditions = []
        for cond in conditions:
            vals = sub.loc[sub["condition"] == cond, "r2_test"].values
            if len(vals) == 0:
                continue
            present_conditions.append(cond)
            means.append(np.mean(vals))
            cis.append(1.96 * np.std(vals) / max(np.sqrt(len(vals)), 1))

        x = np.arange(len(present_conditions))
        bars = ax.bar(x, means, yerr=cis, capsize=3,
                      color=colors[:len(present_conditions)], edgecolor="black",
                      linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(present_conditions, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("R² (test)")
        ax.set_title(ds)
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_complexity_scatter(df: pd.DataFrame, out_path: str):
    """Scatter plot: R² vs complexity_nodes, coloured by condition."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
        "font.size": 10,
    })

    colors_map = {
        "KAO": "#0072B2",
        "No_KAO": "#D55E00",
        "No_KAO_cap7": "#CC79A7",
        "No_KAO_cap9": "#009E73",
        "No_KAO_cap11": "#F0E442",
    }

    datasets = df["dataset"].unique()
    n_ds = len(datasets)
    fig, axes = plt.subplots(1, max(n_ds, 1), figsize=(5 * max(n_ds, 1), 4),
                             squeeze=False)

    for idx, ds in enumerate(datasets):
        ax = axes[0, idx]
        sub = df[df["dataset"] == ds]
        for cond in sub["condition"].unique():
            c = sub[sub["condition"] == cond]
            ax.scatter(c["complexity_nodes"], c["r2_test"],
                       label=cond, alpha=0.6, s=20,
                       color=colors_map.get(cond, "gray"))
        ax.set_xlabel("Complexity (nodes)")
        ax.set_ylabel("R² (test)")
        ax.set_title(ds)
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    dataset_name = args.dataset_name or Path(args.csv).stem
    seeds = list(range(1, args.seeds + 1))

    # Select conditions
    conditions = ABLATION_CONFIGS
    if args.conditions:
        allowed = set(args.conditions)
        conditions = [c for c in conditions if c["name"] in allowed]
        if not conditions:
            print(f"ERROR: no matching conditions in {args.conditions}")
            sys.exit(1)

    print(f"=== Ablation Experiment ===")
    print(f"  Dataset   : {dataset_name} ({args.csv})")
    print(f"  Target    : {args.target}")
    print(f"  Conditions: {[c['name'] for c in conditions]}")
    print(f"  Seeds     : {len(seeds)}")
    print(f"  Budget    : {args.time_budget}s")
    print()

    X_tr, X_te, y_tr, y_te, feat_names = load_data(
        args.csv, args.target, args.test_size,
    )
    print(f"  Train: {X_tr.shape}, Test: {X_te.shape}")
    print()

    # Run experiments
    df = run_ablation(
        X_tr, y_tr, X_te, y_te,
        feature_names=feat_names,
        dataset_name=dataset_name,
        seeds=seeds,
        time_budget=args.time_budget,
        conditions=conditions,
    )

    # Summary table
    summary = make_summary(df)
    tables_dir = Path("results") / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(tables_dir / "ablation_summary.csv", index=False)
    print(f"\n  Summary saved → {tables_dir / 'ablation_summary.csv'}")
    print(summary.to_string(index=False))

    # Pairwise tests
    try:
        pw = pairwise_tests(df)
        if len(pw) > 0:
            pw.to_csv(tables_dir / "ablation_pairwise.csv", index=False)
            print(f"\n  Pairwise tests → {tables_dir / 'ablation_pairwise.csv'}")
            print(pw.to_string(index=False))
    except ImportError:
        print("  (scipy not available — skipping statistical tests)")

    # Plots
    if not args.skip_plots:
        fig_dir = "results/figures"
        try:
            plot_ablation_barplot(df, f"{fig_dir}/ablation_barplot.pdf")
            plot_complexity_scatter(df, f"{fig_dir}/ablation_complexity_scatter.pdf")
        except Exception as e:
            print(f"  Warning: plotting failed ({e})")

    print("\n=== Ablation experiment complete ===")


if __name__ == "__main__":
    main()
