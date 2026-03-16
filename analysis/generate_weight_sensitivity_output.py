#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 4 — Generate weight-sensitivity LaTeX table.

Reads  results/weight_sensitivity/{ds}/{variant}/{method}/summary.json
Writes outputs/tables/Table_weight_sensitivity.tex
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent  # KAO_v3/

DATASETS  = ["mimic_iv", "eicu"]
VARIANTS  = ["original", "uniform", "perturbed"]
METHODS   = ["KAO", "PySR"]
DISPLAY   = {
    "mimic_iv": "MIMIC-IV", "eicu": "eICU",
    "original": "Original", "uniform": "Uniform", "perturbed": "Perturbed",
}


def load_summaries() -> pd.DataFrame:
    rows = []
    base = ROOT / "results" / "weight_sensitivity"
    for ds in DATASETS:
        for var in VARIANTS:
            for m in METHODS:
                p = base / ds / var / m.lower() / "summary.json"
                if not p.exists():
                    print(f"  MISSING: {p}")
                    continue
                s = json.loads(p.read_text())
                rows.append({
                    "dataset": ds,
                    "variant": var,
                    "method": m,
                    "r2_mean": s["r2_mean"],
                    "r2_std": s["r2_std"],
                    "nodes_mean": s["nodes_mean"],
                    "nodes_std": s["nodes_std"],
                    "reachability": s["reachability"],
                })
    return pd.DataFrame(rows)


def generate_latex(df: pd.DataFrame) -> str:
    """Generate a compact LaTeX table."""
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Weight-sensitivity analysis for the two healthcare composite-risk-score datasets.",
        r"  \textit{Original}: default heuristic weights; \textit{Uniform}: all indicators weighted equally;",
        r"  \textit{Perturbed}: high-severity indicators up-weighted ($\times 3$).",
        r"  KAO uses the standard NSGA-II knee configuration; PySR uses the complexity-matched",
        r"  tuned configuration (Table~\ref{tab:complexity_matched_summary}).",
        r"  All runs use 10 seeds, 60\,s budget, and the same train/test split.}",
        r"\label{tab:weight_sensitivity}",
        r"\small",
        r"\begin{tabular}{ll ll rr c}",
        r"\toprule",
        r"Dataset & Variant & Method & $R^2$ (mean$\pm$std) & Nodes (mean$\pm$std) & Reach.\,$\le 8$ \\",
        r"\midrule",
    ]

    prev_ds = None
    prev_var = None
    for _, row in df.iterrows():
        ds_disp = DISPLAY[row["dataset"]]
        var_disp = DISPLAY[row["variant"]]

        # Add midrule between datasets
        if prev_ds is not None and row["dataset"] != prev_ds:
            lines.append(r"\midrule")
        # Add cline between variants within same dataset
        elif prev_var is not None and row["variant"] != prev_var and row["dataset"] == prev_ds:
            lines.append(r"\addlinespace[2pt]")

        # Show dataset name only on first row
        ds_cell = ds_disp if row["dataset"] != prev_ds else ""
        # Show variant name only on first row of variant block
        var_cell = var_disp if (row["variant"] != prev_var or row["dataset"] != prev_ds) else ""

        r2m = row["r2_mean"]
        r2s = row["r2_std"]
        nm  = row["nodes_mean"]
        ns  = row["nodes_std"]

        r2_str = f"${r2m:.3f} \\pm {r2s:.3f}$" if np.isfinite(r2m) else "N/A"
        n_str  = f"${nm:.1f} \\pm {ns:.1f}$"

        line = (
            f"  {ds_cell} & {var_cell} & {row['method']} "
            f"& {r2_str} & {n_str} & {row['reachability']} \\\\"
        )
        lines.append(line)

        prev_ds = row["dataset"]
        prev_var = row["variant"]

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def main():
    print("Loading summaries...")
    df = load_summaries()
    print(f"  Loaded {len(df)} rows")

    if len(df) == 0:
        print("ERROR: No summary files found!")
        return

    # Sort for consistent ordering
    ds_order = {d: i for i, d in enumerate(DATASETS)}
    var_order = {v: i for i, v in enumerate(VARIANTS)}
    m_order = {m: i for i, m in enumerate(METHODS)}
    df["_ds"] = df["dataset"].map(ds_order)
    df["_var"] = df["variant"].map(var_order)
    df["_m"] = df["method"].map(m_order)
    df = df.sort_values(["_ds", "_var", "_m"]).drop(columns=["_ds", "_var", "_m"])

    # Generate LaTeX table
    tex = generate_latex(df)
    out_path = ROOT / "outputs" / "tables" / "Table_weight_sensitivity.tex"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(tex)
    print(f"✓ {out_path}")

    # Also save as CSV
    csv_path = ROOT / "outputs" / "csv" / "weight_sensitivity.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"✓ {csv_path}")

    # Print summary
    print("\n" + "=" * 70)
    for _, row in df.iterrows():
        print(f"  {DISPLAY[row['dataset']]:10s} {DISPLAY[row['variant']]:12s} "
              f"{row['method']:6s}  "
              f"R²={row['r2_mean']:.4f}±{row['r2_std']:.4f}  "
              f"nodes={row['nodes_mean']:.1f}±{row['nodes_std']:.1f}  "
              f"reach={row['reachability']}")


if __name__ == "__main__":
    main()
