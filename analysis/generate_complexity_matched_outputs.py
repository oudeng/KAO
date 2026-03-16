#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 3 — Generate complexity-matched summary table and Pareto plots.

Reads:
  results/{dataset}/complexity_matched/{method}/eval_30seeds/seed_{N}/result.json
  results/{dataset}/60s/kao/seed_{N}/result.json   (KAO original results)

Outputs:
  outputs/tables/Table_complexity_matched_summary.tex
  outputs/figures/Fig_complexity_matched_pareto.pdf     (SM: all 7 datasets)
  outputs/figures/Fig_complexity_matched_main.pdf        (main text: 3 healthcare)
  outputs/csv/complexity_matched_summary.csv
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─── Paths ───────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent

DATASETS_ORDER = [
    ("mimic_iv",    "MIMIC-IV"),
    ("eicu",        "eICU"),
    ("nhanes",      "NHANES"),
    ("airfoil",     "Airfoil"),
    ("auto_mpg",    "Auto-MPG"),
    ("communities", "Communities"),
    ("hydraulic",   "Hydraulic"),
]

HEALTHCARE = {"mimic_iv", "eicu", "nhanes"}

METHODS_BASELINE = ["PySR", "RILS-ROLS", "gplearn", "Operon"]
METHOD_DIRS = {
    "PySR": "pysr", "RILS-ROLS": "rils_rols",
    "gplearn": "gplearn", "Operon": "operon",
    "KAO": "kao",
}

NODE_BUDGET = 8

METHOD_COLORS = {
    "KAO":       "#0072B2",
    "PySR":      "#D55E00",
    "RILS-ROLS": "#009E73",
    "gplearn":   "#CC79A7",
    "Operon":    "#E69F00",
}
METHOD_MARKERS = {
    "KAO": "o", "PySR": "s", "RILS-ROLS": "^",
    "gplearn": "D", "Operon": "v",
}

SEEDS = list(range(1, 31))


# ─── Data collection ─────────────────────────────────────────────────
def _load_seed_data(base_dir: Path, seeds: list) -> List[dict]:
    """Load result.json from each seed directory."""
    rows = []
    for seed in seeds:
        rpath = base_dir / f"seed_{seed}" / "result.json"
        if not rpath.exists():
            continue
        with open(rpath) as f:
            d = json.load(f)
        rows.append({
            "seed": seed,
            "r2_test": d.get("r2_test"),
            "complexity_nodes": d.get("complexity_nodes", 0),
            "expression": d.get("expression", ""),
        })
    return rows


def collect_all_data() -> pd.DataFrame:
    """Collect all complexity-matched results + KAO original results."""
    all_rows = []

    for ds_key, ds_display in DATASETS_ORDER:
        # ── KAO original results ──
        kao_dir = ROOT / "results" / ds_key / "60s" / "kao"
        kao_seeds = _load_seed_data(kao_dir, SEEDS)
        for row in kao_seeds:
            all_rows.append({
                "dataset": ds_display,
                "dataset_key": ds_key,
                "method": "KAO",
                "source": "original",
                "config": "NSGA-II knee",
                **row,
            })

        # ── Complexity-matched baselines ──
        for method in METHODS_BASELINE:
            mdir = METHOD_DIRS[method]
            eval_dir = ROOT / "results" / ds_key / "complexity_matched" / mdir / "eval_30seeds"
            if not eval_dir.exists():
                continue

            # Read tuning report for config
            report_path = ROOT / "results" / ds_key / "complexity_matched" / mdir / "tuning_report.json"
            config_str = ""
            if report_path.exists():
                with open(report_path) as f:
                    report = json.load(f)
                config_str = json.dumps(report.get("best_config", {}))

            cm_seeds = _load_seed_data(eval_dir, SEEDS)
            for row in cm_seeds:
                all_rows.append({
                    "dataset": ds_display,
                    "dataset_key": ds_key,
                    "method": method,
                    "source": "complexity_matched",
                    "config": config_str,
                    **row,
                })

    return pd.DataFrame(all_rows)


# ─── Summary statistics ──────────────────────────────────────────────
def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-(dataset, method, source) aggregated statistics."""
    rows = []
    for (ds, method, source), grp in df.groupby(["dataset", "method", "source"]):
        ok = grp.dropna(subset=["r2_test"])
        n_ok = len(ok)
        n_reach = int((ok["complexity_nodes"] <= NODE_BUDGET).sum()) if n_ok > 0 else 0
        rows.append({
            "dataset": ds,
            "method": method,
            "source": source,
            "config": grp["config"].iloc[0] if len(grp) > 0 else "",
            "n_seeds": n_ok,
            "n_reached": n_reach,
            "reachability": f"{n_reach}/{n_ok}",
            "reach_pct": n_reach / n_ok * 100 if n_ok > 0 else 0,
            "r2_mean": float(ok["r2_test"].mean()) if n_ok > 0 else float("nan"),
            "r2_std": float(ok["r2_test"].std()) if n_ok > 0 else float("nan"),
            "nodes_mean": float(ok["complexity_nodes"].mean()) if n_ok > 0 else float("nan"),
            "nodes_std": float(ok["complexity_nodes"].std()) if n_ok > 0 else float("nan"),
        })
    return pd.DataFrame(rows)


# ─── LaTeX table ──────────────────────────────────────────────────────
def write_summary_table(summary: pd.DataFrame, path: Path):
    """Write the complexity-matched summary LaTeX table."""
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\caption{Complexity-matched baseline comparison (30 seeds, mean $\pm$ std). "
        r"Each baseline is tuned on training data to target a strict $\le 8$-node budget "
        r"under the same 60\,s cap. "
        r"``Reach.'' is the fraction of seeds producing expressions with $\le 8$ nodes. "
        r"KAO rows show the original NSGA-II knee-point results for reference. "
        r"Bold: highest reachability per dataset; "
        r"underline: highest $R^2$ among methods with $\ge 50\%$ reachability.}",
        r"\label{tab:complexity_matched_summary}",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Dataset & Method & Config & Reach.\,$\le 8$ & Test $R^2$ & Nodes \\",
        r"\midrule",
    ]

    ds_order = [d[1] for d in DATASETS_ORDER]
    for i, ds in enumerate(ds_order):
        sub = summary[summary["dataset"] == ds].copy()
        if sub.empty:
            continue

        # Find best reachability and best R² among high-reach methods
        best_reach = sub["reach_pct"].max()
        high_reach = sub[sub["reach_pct"] >= 50]
        best_r2_val = high_reach["r2_mean"].max() if len(high_reach) > 0 else float("nan")

        for _, row in sub.iterrows():
            method = row["method"]
            source = row["source"]

            # Format config string
            if source == "original":
                cfg_str = "NSGA-II knee"
            else:
                try:
                    cfg_dict = json.loads(row["config"])
                    parts = [f"{k}={v}" for k, v in cfg_dict.items()]
                    cfg_str = ", ".join(parts)
                    if len(cfg_str) > 30:
                        cfg_str = cfg_str[:27] + "..."
                except (json.JSONDecodeError, TypeError):
                    cfg_str = str(row["config"])[:30]

            reach_s = row["reachability"]
            r2_s = f"${row['r2_mean']:.3f} \\pm {row['r2_std']:.3f}$"
            nodes_s = f"${row['nodes_mean']:.1f} \\pm {row['nodes_std']:.1f}$"

            # Bold best reachability
            if abs(row["reach_pct"] - best_reach) < 0.01:
                reach_s = r"\textbf{" + reach_s + "}"

            # Underline best R² among high-reach methods
            if (row["reach_pct"] >= 50 and
                    np.isfinite(best_r2_val) and
                    abs(row["r2_mean"] - best_r2_val) < 1e-6):
                r2_s = r"\underline{" + r2_s + "}"

            method_disp = f"{method}" if source == "original" else f"{method} (tuned)"

            lines.append(
                f"{ds} & {method_disp} & {cfg_str} & {reach_s} & {r2_s} & {nodes_s} \\\\"
            )

        if i < len(ds_order) - 1:
            lines.append(r"\midrule")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {path}")


# ─── Pareto plots ─────────────────────────────────────────────────────
def _plot_pareto_panel(ax, ds_key: str, ds_display: str, df: pd.DataFrame):
    """Draw a single Pareto panel: nodes vs R² for one dataset."""
    sub = df[df["dataset_key"] == ds_key]

    # Green auditable zone
    ax.axvspan(0, NODE_BUDGET + 0.5, alpha=0.10, color="#009E73",
               zorder=0, label="_nolegend_")
    # Dashed line at 8 nodes
    ax.axvline(NODE_BUDGET + 0.5, color="#009E73", ls="--", alpha=0.5,
               lw=0.8, zorder=1)

    methods_present = []
    for method in ["KAO"] + METHODS_BASELINE:
        msub = sub[sub["method"] == method]
        if msub.empty:
            continue
        methods_present.append(method)

        nodes = msub["complexity_nodes"].values
        r2 = msub["r2_test"].values

        # Scatter individual seeds (small, transparent)
        ax.scatter(nodes, r2, s=18, alpha=0.35,
                   c=METHOD_COLORS[method],
                   marker=METHOD_MARKERS[method],
                   edgecolors="none",
                   zorder=2, label="_nolegend_")

        # Mean marker (larger, opaque)
        mn = np.mean(nodes)
        mr2 = np.nanmean(r2)
        ax.scatter([mn], [mr2], s=90, alpha=0.95,
                   c=METHOD_COLORS[method],
                   marker=METHOD_MARKERS[method],
                   edgecolors="white", linewidth=0.6,
                   zorder=3, label=method)

    ax.set_xlabel("Nodes", fontsize=9)
    ax.set_ylabel("Test $R^2$", fontsize=9)
    ax.set_title(ds_display, fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=8)

    # Set reasonable x limits
    if len(sub) > 0:
        max_nodes = sub["complexity_nodes"].max()
        ax.set_xlim(0, min(max_nodes * 1.1 + 2, 60))


def plot_pareto_sm(df: pd.DataFrame, path: Path):
    """SM figure: 7 panels (2 rows × 4 cols, last panel = legend)."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes_flat = axes.flatten()

    for idx, (ds_key, ds_display) in enumerate(DATASETS_ORDER):
        _plot_pareto_panel(axes_flat[idx], ds_key, ds_display, df)

    # Last panel: legend only
    ax_leg = axes_flat[7]
    ax_leg.axis("off")
    handles = []
    for method in ["KAO"] + METHODS_BASELINE:
        handles.append(plt.scatter([], [], s=60,
                                    c=METHOD_COLORS[method],
                                    marker=METHOD_MARKERS[method],
                                    label=method))
    handles.append(mpatches.Patch(color="#009E73", alpha=0.15,
                                   label=r"Auditable zone ($\leq 8$ nodes)"))
    ax_leg.legend(handles=handles, loc="center", fontsize=10,
                  frameon=True, framealpha=0.9, ncol=1)

    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {path}")


def plot_pareto_main(df: pd.DataFrame, path: Path):
    """Main-text figure: 3 healthcare panels (1 row × 3 cols)."""
    healthcare_ds = [(k, d) for k, d in DATASETS_ORDER if k in HEALTHCARE]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for idx, (ds_key, ds_display) in enumerate(healthcare_ds):
        _plot_pareto_panel(axes[idx], ds_key, ds_display, df)
        if idx == 0:
            axes[idx].legend(fontsize=7, loc="lower right",
                             framealpha=0.8)

    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {path}")


# ─── Main ─────────────────────────────────────────────────────────────
def main():
    out_tab = ROOT / "outputs" / "tables"
    out_fig = ROOT / "outputs" / "figures"
    out_csv = ROOT / "outputs" / "csv"
    for d in (out_tab, out_fig, out_csv):
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Phase 3: Complexity-matched Outputs")
    print("=" * 70)

    # Collect data
    df = collect_all_data()
    if df.empty:
        print("ERROR: No data found! Check that evaluation has completed.")
        return

    print(f"Collected {len(df)} rows across "
          f"{df['dataset'].nunique()} datasets × {df['method'].nunique()} methods")

    # Summary
    summary = compute_summary(df)
    summary.to_csv(out_csv / "complexity_matched_summary.csv", index=False)
    print(f"Saved {out_csv / 'complexity_matched_summary.csv'}")

    # LaTeX table
    write_summary_table(summary, out_tab / "Table_complexity_matched_summary.tex")

    # Pareto plots
    plot_pareto_sm(df, out_fig / "Fig_complexity_matched_pareto.pdf")
    plot_pareto_main(df, out_fig / "Fig_complexity_matched_main.pdf")

    # Print summary for quick review
    print("\n" + "─" * 70)
    print("SUMMARY")
    print("─" * 70)
    for ds_display in [d[1] for d in DATASETS_ORDER]:
        sub = summary[summary["dataset"] == ds_display]
        if sub.empty:
            continue
        print(f"\n  {ds_display}:")
        for _, row in sub.iterrows():
            tag = "(orig)" if row["source"] == "original" else "(tuned)"
            print(f"    {row['method']:12s} {tag:8s}  "
                  f"reach={row['reachability']:6s}  "
                  f"R²={row['r2_mean']:.4f}±{row['r2_std']:.4f}  "
                  f"nodes={row['nodes_mean']:.1f}±{row['nodes_std']:.1f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
