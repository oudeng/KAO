#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SRBench 10-Task Output Generator — Phase 6C
==============================================
Generates:
  1. outputs/tables/Table_srbench_10tasks.tex  — main text Table 7 replacement
  2. outputs/tables/Table_srbench_descriptors.tex — SM benchmark descriptor table
  3. outputs/figures/Fig_srbench_heatmap.pdf — SM heatmap (R² + nodes, 2 noise levels)

Reads: outputs/csv/srbench_10tasks_results.csv (3000 rows from Phase 6B)
       outputs/csv/srbench_task_descriptors.csv (Phase 6A)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent  # KAO_v3

BENCHMARKS_ORDER = [
    # Group 1: Polynomial
    "nguyen_1",
    # Group 2: Trigonometric / Exp-Log
    "nguyen_7", "nguyen_9", "nguyen_10", "keijzer_4",
    # Group 3: Rational / Complex
    "keijzer_6", "vladislavleva_1", "pagie_1", "vladislavleva_4",
    # Group 4: Challenging High-dim
    "korns_12",
]

BENCH_DISPLAY = {
    "nguyen_1": "Nguyen-1",
    "nguyen_7": "Nguyen-7",
    "nguyen_9": "Nguyen-9",
    "nguyen_10": "Nguyen-10",
    "keijzer_4": "Keijzer-4",
    "keijzer_6": "Keijzer-6",
    "vladislavleva_1": "Vladislavleva-1",
    "pagie_1": "Pagie-1",
    "vladislavleva_4": "Vladislavleva-4",
    "korns_12": "Korns-12",
}

FAMILY_TAG = {
    "nguyen_1": "poly",
    "nguyen_7": "exp/log",
    "nguyen_9": "trig",
    "nguyen_10": "trig",
    "keijzer_4": "exp+trig",
    "keijzer_6": "rational",
    "vladislavleva_1": "exp+rational",
    "pagie_1": "rational",
    "vladislavleva_4": "rational",
    "korns_12": "trig",
}

METHODS_ORDER = ["KAO", "Operon", "PySR", "RILS-ROLS", "gplearn"]

# Group separators for Table (insert \addlinespace after these benchmark_keys)
GROUP_BREAKS = {"nguyen_1", "keijzer_4", "vladislavleva_4"}


def load_data():
    results_path = ROOT / "outputs" / "csv" / "srbench_10tasks_results.csv"
    df = pd.read_csv(results_path)
    df["r2_test"] = pd.to_numeric(df["r2_test"], errors="coerce")
    # Fill status for old results
    if "status" in df.columns:
        mask = df["status"].isna()
        df.loc[mask, "status"] = np.where(
            np.isfinite(df.loc[mask, "r2_test"]), "ok", "div"
        )
    else:
        df["status"] = np.where(np.isfinite(df["r2_test"]), "ok", "div")
    return df


def compute_summary(df):
    """Per (benchmark_key, noise_std, method) summary.

    Divergence policy (matching original Table 7):
    - Per-seed: if R² < -10 or NaN/inf → treat as diverged for that seed
    - Per-group: if ALL seeds diverged → show "div."
    - Per-group: if SOME diverged → compute mean/std over valid seeds only
    """
    rows = []
    for (bkey, noise, method), grp in df.groupby(["benchmark_key", "noise_std", "method"]):
        n_total = len(grp)
        r2_vals = pd.to_numeric(grp["r2_test"], errors="coerce").values
        nodes_vals = grp["complexity_nodes"].values

        # Mark valid (non-diverged) seeds: finite AND not extremely negative
        valid_mask = np.isfinite(r2_vals) & (r2_vals >= -10)
        n_ok = int(valid_mask.sum())
        n_div = int(n_total - n_ok)
        n_timeout = int((grp["status"] == "timeout").sum()) if "status" in grp.columns else 0

        if n_ok > 0:
            valid_r2 = r2_vals[valid_mask]
            valid_nodes = nodes_vals[valid_mask]
            mean_r2 = float(np.mean(valid_r2))
            std_r2 = float(np.std(valid_r2))
            mean_nodes = float(np.mean(valid_nodes))
            std_nodes = float(np.std(valid_nodes))
        else:
            mean_r2 = std_r2 = mean_nodes = std_nodes = float("nan")

        rows.append({
            "benchmark_key": bkey,
            "noise_std": noise,
            "method": method,
            "mean_r2": mean_r2,
            "std_r2": std_r2,
            "mean_nodes": mean_nodes,
            "std_nodes": std_nodes,
            "n_ok": n_ok,
            "n_div": n_div,
            "n_timeout": n_timeout,
            "n_total": n_total,
        })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────
# TABLE 7: 10-task LaTeX table
# ──────────────────────────────────────────────────────────────

def format_r2(mean, std, n_ok, n_total, is_best=False):
    """Format R² value for LaTeX."""
    if n_ok == 0:
        return "div."

    # Check for high variance / extreme negative
    if mean < -10:
        val = f"{mean:.1f}"
        if std > 10:
            val += r"$^\S$"
        if is_best:
            return r"\textbf{" + val + "}"
        return val

    if std < 0.0005:
        val = f"{mean:.3f}"
    else:
        val = f"{mean:.3f}" + r"\scriptsize{$\pm$" + f"{std:.3f}" + "}"

    if n_ok < n_total and n_ok > 0:
        # Partial divergence footnote
        val += r"\textsuperscript{\textdagger}"

    if is_best:
        return r"\textbf{" + val + "}"
    return val


def format_nodes(mean, std, is_best=False):
    """Format nodes value for LaTeX."""
    if np.isnan(mean):
        return "--"
    if std < 0.05:
        val = f"{mean:.1f}"
    else:
        val = f"{mean:.1f}" + r"\scriptsize{$\pm$" + f"{std:.1f}" + "}"
    if is_best:
        return r"\textbf{" + val + "}"
    return val


def generate_table_srbench(summary):
    """Generate Table_srbench_10tasks.tex."""
    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{SRBench synthetic benchmark results on ten tasks selected via stratified coverage "
                 r"(Supplementary Table~\ref{tab:srbench_descriptors}). "
                 r"30 seeds, 60\,s single-thread budget per run. "
                 r"``div.''~indicates numerical divergence across all seeds. "
                 r"\textbf{Bold}: highest $R^2$ and lowest nodes per row (diverged values excluded). "
                 r"$^\S$\,High variance across seeds. "
                 r"\textsuperscript{\textdagger}\,Partial divergence (mean computed over valid seeds only). "
                 r"Tasks are grouped by function family.}")
    lines.append(r"\label{tab:srbench}")
    lines.append(r"\small")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{llcccccccccc}")
    lines.append(r"\toprule")

    # Header
    header = r"Benchmark & $\sigma$"
    for m in METHODS_ORDER:
        mdisp = m.replace("-", "{-}")
        header += rf" & \multicolumn{{2}}{{c}}{{{mdisp}}}"
    header += r" \\"
    lines.append(header)

    cmidrule = ""
    for i, m in enumerate(METHODS_ORDER):
        col_start = 3 + i * 2
        col_end = col_start + 1
        cmidrule += rf"\cmidrule(lr){{{col_start}-{col_end}}} "
    lines.append(cmidrule)

    subheader = r" &  "
    for _ in METHODS_ORDER:
        subheader += r" & $R^2$ & Nodes"
    subheader += r" \\"
    lines.append(subheader)
    lines.append(r"\midrule")

    # Data rows
    for bkey in BENCHMARKS_ORDER:
        bname = BENCH_DISPLAY[bkey]
        for noise_idx, noise in enumerate([0.0, 0.1]):
            # Get data for this row
            row_data = {}
            for m in METHODS_ORDER:
                s = summary[(summary["benchmark_key"] == bkey) &
                            (summary["noise_std"] == noise) &
                            (summary["method"] == m)]
                if len(s) > 0:
                    row_data[m] = s.iloc[0].to_dict()
                else:
                    row_data[m] = None

            # Find best R² (highest among methods with n_ok > 0)
            best_r2 = -np.inf
            best_r2_method = None
            for m in METHODS_ORDER:
                if row_data[m] is not None and row_data[m]["n_ok"] > 0:
                    r2 = row_data[m]["mean_r2"]
                    if np.isfinite(r2) and r2 > best_r2:
                        best_r2 = r2
                        best_r2_method = m

            # Find best nodes (lowest among methods with n_ok > 0 and R² > -1)
            best_nodes = np.inf
            best_nodes_method = None
            for m in METHODS_ORDER:
                if row_data[m] is not None and row_data[m]["n_ok"] > 0:
                    r2 = row_data[m]["mean_r2"]
                    nodes = row_data[m]["mean_nodes"]
                    if np.isfinite(nodes) and np.isfinite(r2) and r2 > -1:
                        if nodes < best_nodes:
                            best_nodes = nodes
                            best_nodes_method = m

            # Build row
            if noise_idx == 0:
                row = f"{bname} & {noise}"
            else:
                row = f" & {noise}"

            for m in METHODS_ORDER:
                d = row_data[m]
                if d is None:
                    row += " & -- & --"
                else:
                    r2_str = format_r2(
                        d["mean_r2"], d["std_r2"],
                        d["n_ok"], d["n_total"],
                        is_best=(m == best_r2_method),
                    )
                    nodes_str = format_nodes(
                        d["mean_nodes"], d["std_nodes"],
                        is_best=(m == best_nodes_method),
                    )
                    row += f" & {r2_str} & {nodes_str}"

            row += r" \\"
            lines.append(row)

        # Group separator
        if bkey in GROUP_BREAKS:
            lines.append(r"\addlinespace")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# SM DESCRIPTORS TABLE
# ──────────────────────────────────────────────────────────────

def generate_descriptors_table():
    """Generate Table_srbench_descriptors.tex."""
    desc_path = ROOT / "outputs" / "csv" / "srbench_task_descriptors.csv"
    desc = pd.read_csv(desc_path, index_col=0)

    ground_truths = {
        "nguyen_1": r"$x^3 + x^2 + x$",
        "nguyen_7": r"$\log(x+1) + \log(x^2+1)$",
        "nguyen_9": r"$\sin(x_0) + \sin(x_1^2)$",
        "nguyen_10": r"$2\sin(x_0)\cos(x_1)$",
        "keijzer_4": r"$x^3 e^{-x} \cos x \sin x (\sin^2 x \cos x - 1)$",
        "keijzer_6": r"$\sum_{i=1}^{x} 1/i$",
        "vladislavleva_1": r"$e^{-(x_0-1)^2} / (1.2 + (x_1-2.5)^2)$",
        "pagie_1": r"$\frac{1}{1+x_0^{-4}} + \frac{1}{1+x_1^{-4}}$",
        "vladislavleva_4": r"$10/(5+\sum(x_i-3)^2)$",
        "korns_12": r"$2 - 2.1\cos(9.8x_0)\sin(1.3x_4)$",
    }

    kao_coverage = {
        "nguyen_1": "Full",
        "nguyen_7": r"Partial (no $\log$)",
        "nguyen_9": r"Partial (no $\sin$)",
        "nguyen_10": r"Partial (no $\sin$/$\cos$)",
        "keijzer_4": r"Partial (no $\exp$/$\sin$/$\cos$)",
        "keijzer_6": "Full",
        "vladislavleva_1": r"Partial (no $\exp$)",
        "pagie_1": "Full",
        "vladislavleva_4": "Full",
        "korns_12": r"Partial (no $\sin$/$\cos$)",
    }

    selection_rationale = {
        "nguyen_1": "Original (poly reference)",
        "nguyen_7": r"Original ($\log$ boundary)",
        "nguyen_9": "Added (2D trig)",
        "nguyen_10": "Added (2D trig)",
        "keijzer_4": "Added (complex mixed)",
        "keijzer_6": "Original (rational)",
        "vladislavleva_1": r"Added (2D exp+rational)",
        "pagie_1": "Added (2D rational)",
        "vladislavleva_4": "Original (5D rational)",
        "korns_12": "Original (5D trig, hard)",
    }

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{Task descriptors for the ten SRBench benchmarks selected via stratified coverage. "
                 r"``KAO coverage'' indicates whether KAO's typed quadratic operator set natively "
                 r"includes all operators needed by the ground-truth expression.}")
    lines.append(r"\label{tab:srbench_descriptors}")
    lines.append(r"\small")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{lclclll}")
    lines.append(r"\toprule")
    lines.append(r"Benchmark & $n_\mathrm{vars}$ & Ground truth & Family & KAO coverage & Selection \\")
    lines.append(r"\midrule")

    for bkey in BENCHMARKS_ORDER:
        bname = BENCH_DISPLAY[bkey]
        if bkey in desc.index:
            nvars = int(desc.loc[bkey, "n_vars"])
        else:
            nvars = "?"
        gt = ground_truths.get(bkey, "?")
        family = FAMILY_TAG.get(bkey, "?")
        coverage = kao_coverage.get(bkey, "?")
        rationale = selection_rationale.get(bkey, "?")

        line = f"{bname} & {nvars} & {gt} & {family} & {coverage} & {rationale}"
        line += r" \\"
        lines.append(line)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# HEATMAP FIGURE
# ──────────────────────────────────────────────────────────────

def generate_heatmap(summary):
    """Generate Fig_srbench_heatmap.pdf — 4-panel: R² and Nodes × 2 noise."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 9,
    })

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    bench_labels = [BENCH_DISPLAY[k] for k in BENCHMARKS_ORDER]
    family_labels = [f"[{FAMILY_TAG[k]}]" for k in BENCHMARKS_ORDER]
    method_labels = METHODS_ORDER

    for col_idx, noise in enumerate([0.0, 0.1]):
        # R² matrix
        r2_mat = np.full((len(BENCHMARKS_ORDER), len(METHODS_ORDER)), np.nan)
        nodes_mat = np.full((len(BENCHMARKS_ORDER), len(METHODS_ORDER)), np.nan)

        for bi, bkey in enumerate(BENCHMARKS_ORDER):
            for mi, method in enumerate(METHODS_ORDER):
                s = summary[(summary["benchmark_key"] == bkey) &
                            (summary["noise_std"] == noise) &
                            (summary["method"] == method)]
                if len(s) > 0:
                    r = s.iloc[0]
                    if r["n_ok"] > 0:
                        r2_val = r["mean_r2"]
                        # Clip for display
                        r2_mat[bi, mi] = np.clip(r2_val, -1, 1)
                        nodes_mat[bi, mi] = r["mean_nodes"]

        # R² heatmap
        ax = axes[0, col_idx]
        norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
        im = ax.imshow(r2_mat, cmap="RdYlGn", norm=norm, aspect="auto")

        ax.set_xticks(range(len(method_labels)))
        ax.set_xticklabels(method_labels, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(len(bench_labels)))
        y_labels = [f"{bl}  {fl}" for bl, fl in zip(bench_labels, family_labels)]
        ax.set_yticklabels(y_labels, fontsize=7)
        ax.set_title(f"$R^2$ (mean, $\\sigma$={noise})", fontsize=10)

        # Annotate
        for bi in range(len(BENCHMARKS_ORDER)):
            for mi in range(len(METHODS_ORDER)):
                val = r2_mat[bi, mi]
                if np.isnan(val):
                    ax.text(mi, bi, "div.", ha="center", va="center",
                            fontsize=6, color="gray")
                else:
                    color = "white" if abs(val) > 0.6 else "black"
                    txt = f"{val:.2f}" if val > -1 else f"{val:.1f}"
                    ax.text(mi, bi, txt, ha="center", va="center",
                            fontsize=6, color=color)

        # Nodes heatmap
        ax = axes[1, col_idx]
        nodes_display = np.where(np.isnan(nodes_mat), 0, nodes_mat)
        im2 = ax.imshow(nodes_display, cmap="YlOrRd", vmin=0, vmax=50, aspect="auto")

        ax.set_xticks(range(len(method_labels)))
        ax.set_xticklabels(method_labels, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(len(bench_labels)))
        ax.set_yticklabels(y_labels, fontsize=7)
        ax.set_title(f"Complexity (nodes, $\\sigma$={noise})", fontsize=10)

        # Annotate
        for bi in range(len(BENCHMARKS_ORDER)):
            for mi in range(len(METHODS_ORDER)):
                val = nodes_mat[bi, mi]
                if np.isnan(val):
                    ax.text(mi, bi, "div.", ha="center", va="center",
                            fontsize=6, color="gray")
                else:
                    color = "white" if val > 30 else "black"
                    ax.text(mi, bi, f"{val:.0f}", ha="center", va="center",
                            fontsize=6, color=color)

    fig.tight_layout(rect=[0, 0, 0.92, 1])

    # Colorbars
    cbar_ax1 = fig.add_axes([0.93, 0.55, 0.015, 0.35])
    fig.colorbar(im, cax=cbar_ax1, label="$R^2$")
    cbar_ax2 = fig.add_axes([0.93, 0.08, 0.015, 0.35])
    fig.colorbar(im2, cax=cbar_ax2, label="Nodes")

    out_path = ROOT / "outputs" / "figures" / "Fig_srbench_heatmap.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")
    return str(out_path)


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Phase 6C: SRBench 10-Task Output Generator")
    print("=" * 70)

    df = load_data()
    print(f"  Loaded {len(df)} results")

    summary = compute_summary(df)
    print(f"  Computed summary ({len(summary)} rows)")

    tables_dir = ROOT / "outputs" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # 1. Main text table
    table_tex = generate_table_srbench(summary)
    table_path = tables_dir / "Table_srbench_10tasks.tex"
    with open(table_path, "w") as f:
        f.write(table_tex + "\n")
    print(f"  ✓ {table_path}")

    # 2. SM descriptors table
    desc_tex = generate_descriptors_table()
    desc_path = tables_dir / "Table_srbench_descriptors.tex"
    with open(desc_path, "w") as f:
        f.write(desc_tex + "\n")
    print(f"  ✓ {desc_path}")

    # 3. Heatmap
    heatmap_path = generate_heatmap(summary)
    print(f"  ✓ {heatmap_path}")

    # 4. Print summary for report
    print(f"\n{'='*80}")
    print("SUMMARY (σ=0.0)")
    print(f"{'='*80}")
    for bkey in BENCHMARKS_ORDER:
        for noise in [0.0]:
            s = summary[(summary["benchmark_key"] == bkey) & (summary["noise_std"] == noise)]
            bname = BENCH_DISPLAY[bkey]
            line = f"  {bname:20s}"
            for m in METHODS_ORDER:
                ms = s[s["method"] == m]
                if len(ms) > 0:
                    r = ms.iloc[0]
                    if r["n_ok"] > 0:
                        line += f"  {r['mean_r2']:>8.3f}"
                    else:
                        line += f"  {'div.':>8s}"
                else:
                    line += f"  {'N/A':>8s}"
            print(line)


if __name__ == "__main__":
    main()
