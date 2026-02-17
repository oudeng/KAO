#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Time-Resolved Pareto Front Comparison
======================================
Uses time_checkpoints in KAO_v3_1 to capture Pareto front snapshots at
t = 10, 20, 30, 40, 50, 60 s. Computes the 2-D hypervolume indicator at
each snapshot and compares KAO vs No_KAO across multiple seeds.

Outputs
-------
results/pareto_time/{dataset}_hv.csv                per-seed HV at each checkpoint
results/pareto_time/{dataset}_snapshots.json        raw Pareto fronts (optional)
results/tables/pareto_time_summary.csv              mean ± std HV
results/figures/pareto_hv_vs_time.pdf               HV vs Time curves + CI bands
results/figures/pareto_evolution_panels.pdf          2×3 Pareto front panels
results/figures/pareto_delta_hv.pdf                  ΔHV bar chart with significance

Usage
-----
  python scripts/run_pareto_time.py --csv data/mimic_iv/ICU_composite_risk_score.csv \\
         --target composite_risk_score --seeds 30 --time_budget 60
  python scripts/run_pareto_time.py --csv data/mimic_iv/ICU_composite_risk_score.csv \\
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
from utils.result_io import ensure_json_serializable  # noqa: E402

# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------
PARETO_CONDITIONS: list[dict] = [
    {"name": "KAO",    "use_kao_leaf": True,  "max_complexity": None},
    {"name": "No_KAO", "use_kao_leaf": False, "max_complexity": None},
]

DEFAULT_CHECKPOINTS = [10, 20, 30, 40, 50, 60]

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Time-resolved Pareto front comparison (KAO vs No_KAO)",
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
    p.add_argument("--checkpoints", nargs="*", type=int, default=None,
                   help="Time checkpoints in seconds (default: 10 20 30 40 50 60)")
    p.add_argument("--skip_plots", action="store_true",
                   help="Skip figure generation")
    p.add_argument("--save_snapshots", action="store_true",
                   help="Save raw Pareto snapshots as JSON")
    return p.parse_args()

# ---------------------------------------------------------------------------
# Data loading
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
# Hypervolume computation (2-D: error × complexity)
# ---------------------------------------------------------------------------

def hypervolume_2d(front: list[tuple], ref_point: tuple[float, float]) -> float:
    """Compute exact 2-D hypervolume dominated by *front* w.r.t. *ref_point*.

    Parameters
    ----------
    front : list of (error, complexity, ...) tuples
        Only the first two elements are used.
    ref_point : (ref_error, ref_complexity)
        Reference point; must dominate no member of the front.

    Returns
    -------
    float  Non-negative hypervolume value.
    """
    if not front:
        return 0.0

    ref_e, ref_c = ref_point

    # Extract (error, complexity), discard dominated & points beyond ref
    pts = []
    for p in front:
        e, c = float(p[0]), float(p[1])
        if e < ref_e and c < ref_c:
            pts.append((e, c))

    if not pts:
        return 0.0

    # Sort by error ascending
    pts.sort(key=lambda x: x[0])

    # Remove dominated points (keep non-dominated only)
    nd = [pts[0]]
    for e, c in pts[1:]:
        if c < nd[-1][1]:
            nd.append((e, c))

    # Sweep-line HV computation
    hv = 0.0
    prev_c = ref_c
    for e, c in nd:
        width = ref_e - e
        height = prev_c - c
        hv += width * height
        prev_c = c

    return hv

# ---------------------------------------------------------------------------
# Reference point computation
# ---------------------------------------------------------------------------

def compute_ref_point(all_snapshots: dict, margin: float = 1.1) -> tuple[float, float]:
    """Compute reference point as margin × worst observed values across all runs."""
    max_error = 0.0
    max_complexity = 0.0
    for key, snapshots in all_snapshots.items():
        for cp, front in snapshots.items():
            for pt in front:
                max_error = max(max_error, float(pt[0]))
                max_complexity = max(max_complexity, float(pt[1]))
    # Ensure non-zero
    max_error = max(max_error, 1.0)
    max_complexity = max(max_complexity, 1.0)
    return (max_error * margin, max_complexity * margin)

# ---------------------------------------------------------------------------
# Core experiment loop
# ---------------------------------------------------------------------------

def run_pareto_time(
    X_train, y_train, X_test, y_test,
    feature_names: list[str],
    dataset_name: str,
    seeds: list[int],
    time_budget: float,
    checkpoints: list[int],
    conditions: list[dict],
) -> tuple[pd.DataFrame, dict]:
    """Run all conditions × seeds and return HV DataFrame + raw snapshots."""
    out_dir = Path("results") / "pareto_time"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect all snapshots: {(condition, seed): {cp: [(e,c,expr), ...]}}
    all_snapshots: dict[tuple[str, int], dict] = {}

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
                time_checkpoints=checkpoints,
                feature_names=feature_names,
            )
            print(f"R²={result.r2_test:.4f}  nodes={result.complexity_nodes}")

            all_snapshots[(cname, seed)] = result.pareto_snapshots

    # Compute reference point across all runs
    ref_point = compute_ref_point(all_snapshots)
    print(f"\n  Reference point: error={ref_point[0]:.4f}, complexity={ref_point[1]:.1f}")

    # Build HV table
    rows: list[dict] = []
    for (cname, seed), snapshots in all_snapshots.items():
        for cp in checkpoints:
            front = snapshots.get(cp, [])
            hv = hypervolume_2d(front, ref_point)
            rows.append({
                "dataset": dataset_name,
                "condition": cname,
                "seed": seed,
                "checkpoint": cp,
                "hypervolume": hv,
                "front_size": len(front),
            })

    df = pd.DataFrame(rows)

    # Save per-seed HV CSV
    hv_path = out_dir / f"{dataset_name}_hv.csv"
    df.to_csv(hv_path, index=False)
    print(f"  HV data saved → {hv_path}")

    return df, all_snapshots

# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def make_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Mean ± std HV per condition per checkpoint."""
    summary = (
        df.groupby(["dataset", "condition", "checkpoint"])["hypervolume"]
        .agg(["mean", "std", "median", "count"])
        .reset_index()
    )
    summary.columns = [
        "dataset", "condition", "checkpoint",
        "hv_mean", "hv_std", "hv_median", "n_seeds",
    ]
    return summary

# ---------------------------------------------------------------------------
# Statistical tests at each checkpoint
# ---------------------------------------------------------------------------

def pairwise_tests_per_checkpoint(df: pd.DataFrame) -> pd.DataFrame:
    """Wilcoxon signed-rank + Cliff's delta for KAO vs No_KAO at each checkpoint."""
    from scipy.stats import wilcoxon

    def cliffs_delta(x, y):
        nx, ny = len(x), len(y)
        if nx == 0 or ny == 0:
            return 0.0
        more = sum(1 for xi in x for yi in y if xi > yi)
        less = sum(1 for xi in x for yi in y if xi < yi)
        return (more - less) / (nx * ny)

    results = []
    for dataset in df["dataset"].unique():
        d = df[df["dataset"] == dataset]
        for cp in sorted(d["checkpoint"].unique()):
            dc = d[d["checkpoint"] == cp]
            kao_vals = dc.loc[dc["condition"] == "KAO", "hypervolume"].values
            nokao_vals = dc.loc[dc["condition"] == "No_KAO", "hypervolume"].values

            n_pairs = min(len(kao_vals), len(nokao_vals))
            if n_pairs < 5:
                continue

            a, b = kao_vals[:n_pairs], nokao_vals[:n_pairs]

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

            delta_hv = np.mean(a) - np.mean(b)

            results.append({
                "dataset": dataset,
                "checkpoint": cp,
                "comparison": "KAO vs No_KAO",
                "delta_hv": delta_hv,
                "wilcoxon_stat": stat,
                "p_value": pval,
                "significance": stars,
                "cliffs_delta": cd,
                "n_pairs": n_pairs,
            })

    return pd.DataFrame(results)

# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def _setup_rcparams():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
        "font.size": 10,
    })
    return plt


# Okabe-Ito colorblind-safe
_COLORS = {"KAO": "#0072B2", "No_KAO": "#D55E00"}

# ---------------------------------------------------------------------------
# Plot 1: HV vs Time curves with 95% CI
# ---------------------------------------------------------------------------

def plot_hv_vs_time(df: pd.DataFrame, out_path: str):
    """Line plot: HV vs checkpoint with 95% CI shaded bands."""
    plt = _setup_rcparams()

    datasets = df["dataset"].unique()
    n_ds = len(datasets)
    fig, axes = plt.subplots(1, max(n_ds, 1), figsize=(5 * max(n_ds, 1), 4),
                             squeeze=False)

    for idx, ds in enumerate(datasets):
        ax = axes[0, idx]
        sub = df[df["dataset"] == ds]
        for cond in ["KAO", "No_KAO"]:
            c = sub[sub["condition"] == cond]
            if c.empty:
                continue
            grouped = c.groupby("checkpoint")["hypervolume"]
            means = grouped.mean()
            stds = grouped.std()
            counts = grouped.count()
            ci = 1.96 * stds / np.sqrt(counts.clip(lower=1))

            x = means.index.values
            y = means.values
            lo = y - ci.values
            hi = y + ci.values

            color = _COLORS.get(cond, "gray")
            ax.plot(x, y, marker="o", markersize=4, label=cond,
                    color=color, linewidth=1.5)
            ax.fill_between(x, lo, hi, alpha=0.2, color=color)

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Hypervolume")
        ax.set_title(ds)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Plot 2: 2×3 Pareto front evolution panels
# ---------------------------------------------------------------------------

def plot_pareto_evolution(
    all_snapshots: dict,
    checkpoints: list[int],
    dataset_name: str,
    out_path: str,
):
    """4-row × 3-col panels showing Pareto fronts at each checkpoint.

    Row 0-1 = KAO (t=10..60), Row 2-3 = No_KAO (t=10..60).
    Points are overlaid across seeds with low alpha.
    """
    plt = _setup_rcparams()

    selected_cps = checkpoints[:6]
    conditions = ["KAO", "No_KAO"]

    fig, axes = plt.subplots(4, 3, figsize=(12, 14), squeeze=False)

    for cond_idx, cond in enumerate(conditions):
        color = _COLORS.get(cond, "gray")
        for t_idx, cp in enumerate(selected_cps):
            row = cond_idx * 2 + t_idx // 3
            col = t_idx % 3
            ax = axes[row, col]

            # Gather fronts from all seeds
            for (cn, seed), snapshots in all_snapshots.items():
                if cn != cond:
                    continue
                front = snapshots.get(cp, [])
                if not front:
                    continue
                errors = [float(p[0]) for p in front]
                complexities = [float(p[1]) for p in front]
                ax.scatter(complexities, errors, s=8, alpha=0.3, color=color)

            ax.set_title(f"{cond}  t={cp}s", fontsize=10)
            ax.grid(alpha=0.3)

    # Axis labels: y only on left, x only on bottom row
    for row in range(4):
        axes[row, 0].set_ylabel("CV-MSE")
        for col in range(1, 3):
            axes[row, col].set_ylabel("")
    for col in range(3):
        axes[3, col].set_xlabel("Complexity (nodes)")
        for row in range(3):
            axes[row, col].set_xlabel("")

    fig.suptitle(f"Pareto Front Evolution — {dataset_name}", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.4)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Plot 3: ΔHV bar chart with significance
# ---------------------------------------------------------------------------

def plot_delta_hv(pw: pd.DataFrame, out_path: str):
    """Bar chart of ΔHV (KAO − No_KAO) at each checkpoint with significance stars."""
    plt = _setup_rcparams()

    datasets = pw["dataset"].unique()
    n_ds = len(datasets)
    fig, axes = plt.subplots(1, max(n_ds, 1), figsize=(5 * max(n_ds, 1), 4),
                             squeeze=False)

    for idx, ds in enumerate(datasets):
        ax = axes[0, idx]
        sub = pw[pw["dataset"] == ds].sort_values("checkpoint")
        x = np.arange(len(sub))
        bars = ax.bar(x, sub["delta_hv"].values, color="#0072B2",
                      edgecolor="black", linewidth=0.5)

        # Add significance stars above bars
        for i, (_, row) in enumerate(sub.iterrows()):
            if row["significance"]:
                y_pos = row["delta_hv"]
                offset = abs(y_pos) * 0.05 + 0.01
                y_text = y_pos + offset if y_pos >= 0 else y_pos - offset
                ax.text(i, y_text, row["significance"],
                        ha="center", va="bottom" if y_pos >= 0 else "top",
                        fontsize=9, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels([f"{int(c)}s" for c in sub["checkpoint"].values],
                           fontsize=8)
        ax.set_xlabel("Checkpoint")
        ax.set_ylabel("ΔHV (KAO − No_KAO)")
        ax.set_title(ds)
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
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
    dataset_name = args.dataset_name or Path(args.csv).stem
    seeds = list(range(1, args.seeds + 1))
    checkpoints = args.checkpoints or DEFAULT_CHECKPOINTS

    print(f"=== Time-Resolved Pareto Front Experiment ===")
    print(f"  Dataset     : {dataset_name} ({args.csv})")
    print(f"  Target      : {args.target}")
    print(f"  Conditions  : {[c['name'] for c in PARETO_CONDITIONS]}")
    print(f"  Seeds       : {len(seeds)}")
    print(f"  Budget      : {args.time_budget}s")
    print(f"  Checkpoints : {checkpoints}")
    print()

    X_tr, X_te, y_tr, y_te, feat_names = load_data(
        args.csv, args.target, args.test_size,
    )
    print(f"  Train: {X_tr.shape}, Test: {X_te.shape}")
    print()

    # Run experiments
    df, all_snapshots = run_pareto_time(
        X_tr, y_tr, X_te, y_te,
        feature_names=feat_names,
        dataset_name=dataset_name,
        seeds=seeds,
        time_budget=args.time_budget,
        checkpoints=checkpoints,
        conditions=PARETO_CONDITIONS,
    )

    # Summary table
    summary = make_summary(df)
    tables_dir = Path("results") / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(tables_dir / "pareto_time_summary.csv", index=False)
    print(f"\n  Summary saved → {tables_dir / 'pareto_time_summary.csv'}")
    print(summary.to_string(index=False))

    # Pairwise tests per checkpoint
    pw = pd.DataFrame()
    try:
        pw = pairwise_tests_per_checkpoint(df)
        if len(pw) > 0:
            pw.to_csv(tables_dir / "pareto_time_pairwise.csv", index=False)
            print(f"\n  Pairwise tests → {tables_dir / 'pareto_time_pairwise.csv'}")
            print(pw.to_string(index=False))
    except ImportError:
        print("  (scipy not available — skipping statistical tests)")

    # Save raw snapshots
    if args.save_snapshots:
        snap_dir = Path("results") / "pareto_time"
        snap_path = snap_dir / f"{dataset_name}_snapshots.json"
        # Convert tuple keys to strings for JSON
        serialisable = {}
        for (cname, seed), snaps in all_snapshots.items():
            key = f"{cname}_seed{seed}"
            serialisable[key] = {
                str(cp): [(float(e), int(c), s) for e, c, s in front]
                for cp, front in snaps.items()
            }
        # NaN/Inf → null, numpy → native
        serialisable = ensure_json_serializable(serialisable)
        with open(snap_path, "w") as f:
            json.dump(serialisable, f, indent=2)
        print(f"  Snapshots saved → {snap_path}")

    # Plots
    if not args.skip_plots:
        fig_dir = "results/figures"
        try:
            plot_hv_vs_time(df, f"{fig_dir}/pareto_hv_vs_time.pdf")
        except Exception as e:
            print(f"  Warning: HV vs Time plot failed ({e})")
        try:
            plot_pareto_evolution(
                all_snapshots, checkpoints, dataset_name,
                f"{fig_dir}/pareto_evolution_panels.pdf",
            )
        except Exception as e:
            print(f"  Warning: Pareto evolution plot failed ({e})")
        try:
            if len(pw) > 0:
                plot_delta_hv(pw, f"{fig_dir}/pareto_delta_hv.pdf")
        except Exception as e:
            print(f"  Warning: ΔHV bar plot failed ({e})")

    print("\n=== Time-resolved Pareto experiment complete ===")


if __name__ == "__main__":
    main()
