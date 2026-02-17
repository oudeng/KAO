#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SRBench Synthetic Benchmark Runner
====================================
Runs each benchmark x each method (KAO + baselines) x N seeds x time_budget.

Two noise settings are run by default:
  - noise_std = 0.0  (noiseless)
  - noise_std = 0.1  (moderate noise)

Evaluation metrics per run:
  - R^2 test
  - complexity_nodes, complexity_chars
  - runtime
  - symbolic_recovery  (sympy equivalence check against ground truth)

Outputs
-------
results/tables/srbench_results.csv            full per-run table
results/tables/srbench_summary.csv            mean +/- std aggregation
results/figures/srbench_heatmap.pdf           recovery-rate heatmap
results/figures/srbench_r2_comparison.pdf      R^2 grouped bar chart

Usage
-----
  python scripts/run_srbench.py --seeds 30 --time_budget 60          # full run
  python scripts/run_srbench.py --benchmark nguyen_1 --seeds 3 \\
         --time_budget 10                                             # smoke test
  python scripts/run_srbench.py --seeds 5 --methods KAO gplearn      # subset
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
from sklearn.metrics import r2_score

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from data.srbench_synthetic.generate import (          # noqa: E402
    BENCHMARKS,
    SRBenchmark,
    generate_dataset,
    check_symbolic_recovery,
)
from kao.KAO_v3_1 import run_single, KAOResult         # noqa: E402
from baselines.registry import get_all_baselines        # noqa: E402

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SRBench synthetic benchmark runner",
    )
    p.add_argument("--benchmark", nargs="*", default=None,
                   help="Benchmark keys to run (default: all). "
                        "Choices: " + ", ".join(BENCHMARKS.keys()))
    p.add_argument("--methods", nargs="*", default=None,
                   help="Method names to run (default: KAO + all available baselines)")
    p.add_argument("--seeds", type=int, default=30,
                   help="Number of seeds (1..N)")
    p.add_argument("--time_budget", type=float, default=60.0,
                   help="Wall-clock seconds per run")
    p.add_argument("--noise_levels", nargs="*", type=float, default=[0.0, 0.1],
                   help="Noise std levels (default: 0.0 0.1)")
    p.add_argument("--skip_plots", action="store_true")
    p.add_argument("--skip_baselines", action="store_true",
                   help="Run KAO only (skip all baselines)")
    return p.parse_args()

# ---------------------------------------------------------------------------
# KAO runner adapter
# ---------------------------------------------------------------------------

def run_kao_on_benchmark(
    bench: SRBenchmark,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    time_budget: float,
) -> dict:
    """Run KAO and return a standardised result dict."""
    feature_names = [f"x{i}" for i in range(bench.n_vars)]
    result: KAOResult = run_single(
        X_train, y_train, X_test, y_test,
        dataset_name=bench.name,
        seed=seed,
        time_budget=time_budget,
        use_kao_leaf=True,
        feature_names=feature_names,
    )
    y_pred_test = result.y_pred_test
    r2 = r2_score(y_test, y_pred_test) if y_pred_test is not None else float("nan")
    return {
        "expression": result.expression,
        "r2_test": r2,
        "complexity_nodes": result.complexity_nodes,
        "complexity_chars": result.complexity_chars,
        "runtime": result.runtime,
    }

# ---------------------------------------------------------------------------
# Baseline runner adapter
# ---------------------------------------------------------------------------

def run_baseline_on_benchmark(
    baseline,
    bench: SRBenchmark,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    time_budget: float,
) -> dict:
    """Run a BaselineSR and return a standardised result dict."""
    feature_names = [f"x{i}" for i in range(bench.n_vars)]
    try:
        res = baseline.fit(
            X_train, y_train,
            X_test=X_test, y_test=y_test,
            time_budget=time_budget,
            random_state=seed,
            feature_names=feature_names,
        )
    except Exception as exc:
        warnings.warn(f"{baseline.name} failed: {exc}")
        return {
            "expression": "FAILED",
            "r2_test": float("nan"),
            "complexity_nodes": 0,
            "complexity_chars": 0,
            "runtime": 0.0,
        }

    expr_str = res.get("expression", "FAILED")
    y_pred_test = res.get("y_pred_test")
    if y_pred_test is not None:
        y_pred_test = np.where(np.isfinite(y_pred_test), y_pred_test, 0.0)
        r2 = r2_score(y_test, y_pred_test)
    else:
        r2 = float("nan")

    return {
        "expression": expr_str,
        "r2_test": r2,
        "complexity_nodes": int(res.get("complexity", 0)),
        "complexity_chars": int(res.get("complexity_chars", 0)),
        "runtime": float(res.get("runtime", 0.0)),
    }

# ---------------------------------------------------------------------------
# Core experiment loop
# ---------------------------------------------------------------------------

def run_srbench(
    benchmarks: dict[str, SRBenchmark],
    methods: list[dict],       # [{"name": ..., "runner": callable}, ...]
    seeds: list[int],
    time_budget: float,
    noise_levels: list[float],
) -> pd.DataFrame:
    """Run the full experiment matrix and return a tidy DataFrame."""
    rows: list[dict] = []
    out_dir = Path("results") / "srbench"
    out_dir.mkdir(parents=True, exist_ok=True)

    total = len(benchmarks) * len(methods) * len(seeds) * len(noise_levels)
    done = 0

    for noise_std in noise_levels:
        noise_label = f"noise{noise_std}"
        for bkey, bench in benchmarks.items():
            for method in methods:
                mname = method["name"]
                for seed in seeds:
                    done += 1
                    print(
                        f"[{done}/{total}] {bench.name} | {mname} | "
                        f"seed={seed} | {noise_label}",
                        end=" … ",
                        flush=True,
                    )

                    # Generate data for this seed + noise level
                    X_tr, y_tr, X_te, y_te = generate_dataset(
                        bench, seed=seed, noise_std=noise_std,
                    )

                    res = method["runner"](
                        bench, X_tr, y_tr, X_te, y_te, seed, time_budget,
                    )

                    # Symbolic recovery check
                    recovered = check_symbolic_recovery(
                        res["expression"], bench.ground_truth,
                    )

                    print(
                        f"R²={res['r2_test']:.4f}  "
                        f"nodes={res['complexity_nodes']}  "
                        f"recovered={recovered}"
                    )

                    row = {
                        "benchmark": bench.name,
                        "benchmark_key": bkey,
                        "method": mname,
                        "seed": seed,
                        "noise_std": noise_std,
                        "r2_test": res["r2_test"],
                        "complexity_nodes": res["complexity_nodes"],
                        "complexity_chars": res["complexity_chars"],
                        "runtime": res["runtime"],
                        "expression": res["expression"],
                        "ground_truth": bench.ground_truth,
                        "symbolic_recovery": recovered,
                    }
                    rows.append(row)

                    # Per-run JSON log
                    log_path = (
                        out_dir
                        / f"{bkey}_{mname}_{noise_label}_seed{seed}.json"
                    )
                    with open(log_path, "w") as f:
                        json.dump(row, f, indent=2, default=str)

    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def make_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate mean +/- std per benchmark x method x noise level."""
    metrics = ["r2_test", "complexity_nodes", "complexity_chars", "runtime"]
    agg = {}
    for m in metrics:
        agg[f"{m}_mean"] = (m, "mean")
        agg[f"{m}_std"] = (m, "std")
    agg["recovery_rate"] = ("symbolic_recovery", "mean")
    agg["n_seeds"] = ("seed", "count")

    summary = (
        df.groupby(["benchmark", "method", "noise_std"])
        .agg(**agg)
        .reset_index()
    )
    return summary

# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _setup_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
        "font.size": 10,
    })
    return plt


def plot_recovery_heatmap(summary: pd.DataFrame, out_path: str):
    """Heatmap of symbolic recovery rate: benchmarks (rows) x methods (cols)."""
    plt = _setup_plt()

    # Use noiseless only for the heatmap
    sub = summary[summary["noise_std"] == 0.0]
    if sub.empty:
        sub = summary

    pivot = sub.pivot_table(
        index="benchmark", columns="method", values="recovery_rate",
        aggfunc="first",
    )

    fig, ax = plt.subplots(figsize=(max(3, 1.2 * len(pivot.columns)), max(3, 0.8 * len(pivot.index))))
    im = ax.imshow(pivot.values, cmap="YlGn", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                        fontsize=8, color="black" if val < 0.7 else "white")

    fig.colorbar(im, ax=ax, label="Recovery Rate")
    ax.set_title("Symbolic Recovery Rate (noiseless)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_r2_comparison(summary: pd.DataFrame, out_path: str):
    """Grouped bar chart: R^2 per method per benchmark."""
    plt = _setup_plt()

    # Okabe-Ito palette
    palette = ["#0072B2", "#D55E00", "#CC79A7", "#009E73", "#F0E442",
               "#56B4E9", "#E69F00"]

    noise_levels = sorted(summary["noise_std"].unique())
    n_panels = len(noise_levels)
    fig, axes = plt.subplots(1, max(n_panels, 1),
                             figsize=(6 * max(n_panels, 1), 5), squeeze=False)

    for pidx, noise in enumerate(noise_levels):
        ax = axes[0, pidx]
        sub = summary[summary["noise_std"] == noise]
        benchmarks = sub["benchmark"].unique()
        methods = sub["method"].unique()

        n_bench = len(benchmarks)
        n_meth = len(methods)
        bar_w = 0.8 / max(n_meth, 1)

        for midx, method in enumerate(methods):
            ms = sub[sub["method"] == method]
            means = []
            for b in benchmarks:
                row = ms[ms["benchmark"] == b]
                means.append(row["r2_test_mean"].values[0] if len(row) else 0)
            x = np.arange(n_bench) + midx * bar_w
            color = palette[midx % len(palette)]
            ax.bar(x, means, width=bar_w, label=method, color=color,
                   edgecolor="black", linewidth=0.3)

        ax.set_xticks(np.arange(n_bench) + 0.4)
        ax.set_xticklabels(benchmarks, rotation=30, ha="right", fontsize=7)
        ax.set_ylabel("R² (test)")
        ax.set_title(f"noise_std = {noise}")
        ax.legend(fontsize=6, loc="lower right")
        ax.grid(axis="y", alpha=0.3)

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
    seeds = list(range(1, args.seeds + 1))

    # Select benchmarks
    if args.benchmark:
        selected = {}
        for bk in args.benchmark:
            if bk not in BENCHMARKS:
                print(f"WARNING: unknown benchmark '{bk}', skipping")
                continue
            selected[bk] = BENCHMARKS[bk]
        if not selected:
            print("ERROR: no valid benchmarks selected")
            sys.exit(1)
        benchmarks = selected
    else:
        benchmarks = BENCHMARKS

    # Build method list
    methods: list[dict] = []

    # KAO is always included
    kao_runner = lambda bench, X_tr, y_tr, X_te, y_te, seed, tb: \
        run_kao_on_benchmark(bench, X_tr, y_tr, X_te, y_te, seed, tb)
    methods.append({"name": "KAO", "runner": kao_runner})

    # Baselines
    if not args.skip_baselines:
        try:
            baselines = get_all_baselines(include_optional=True)
        except Exception as exc:
            warnings.warn(f"Could not load baselines: {exc}")
            baselines = []

        for bl in baselines:
            # Filter by --methods if specified
            if args.methods and bl.name not in args.methods:
                continue
            # Capture bl in closure
            def _make_runner(b):
                return lambda bench, X_tr, y_tr, X_te, y_te, seed, tb: \
                    run_baseline_on_benchmark(b, bench, X_tr, y_tr, X_te, y_te, seed, tb)
            methods.append({"name": bl.name, "runner": _make_runner(bl)})

    # If --methods filters were given, also filter KAO
    if args.methods and "KAO" not in args.methods:
        methods = [m for m in methods if m["name"] != "KAO"]

    if not methods:
        print("ERROR: no methods selected")
        sys.exit(1)

    print("=" * 60)
    print("SRBench Synthetic Benchmark Runner")
    print("=" * 60)
    print(f"  Benchmarks : {[b.name for b in benchmarks.values()]}")
    print(f"  Methods    : {[m['name'] for m in methods]}")
    print(f"  Seeds      : {len(seeds)}")
    print(f"  Budget     : {args.time_budget}s")
    print(f"  Noise      : {args.noise_levels}")
    print()

    # Run experiments
    df = run_srbench(
        benchmarks=benchmarks,
        methods=methods,
        seeds=seeds,
        time_budget=args.time_budget,
        noise_levels=args.noise_levels,
    )

    # Save full results
    tables_dir = Path("results") / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(tables_dir / "srbench_results.csv", index=False)
    print(f"\n  Full results → {tables_dir / 'srbench_results.csv'}")

    # Summary
    summary = make_summary(df)
    summary.to_csv(tables_dir / "srbench_summary.csv", index=False)
    print(f"  Summary     → {tables_dir / 'srbench_summary.csv'}")
    print(summary.to_string(index=False))

    # Plots
    if not args.skip_plots:
        fig_dir = "results/figures"
        try:
            plot_recovery_heatmap(summary, f"{fig_dir}/srbench_heatmap.pdf")
        except Exception as e:
            print(f"  Warning: heatmap plot failed ({e})")
        try:
            plot_r2_comparison(summary, f"{fig_dir}/srbench_r2_comparison.pdf")
        except Exception as e:
            print(f"  Warning: R² comparison plot failed ({e})")

    print("\n=== SRBench experiment complete ===")


if __name__ == "__main__":
    main()
