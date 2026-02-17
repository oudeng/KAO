# -*- coding: utf-8 -*-
"""
Paper-Quality Visualisation Functions
======================================
Every public function saves a PDF to *save_path* (creating parent dirs
as needed).  All figures use the Okabe–Ito colorblind-safe palette,
Arial sans-serif font, and 300 DPI.

Main-text figures
-----------------
1. plot_accuracy_complexity_scatter
2. plot_statistical_heatmap
3. plot_critical_difference_diagram
4. plot_ablation_bars
5. plot_hypervolume_curves
6. plot_pareto_snapshots
7. plot_srbench_heatmap
8. plot_violin_strip

Appendix figures
----------------
A1. plot_residual_diagnostics
A2. plot_convergence_curves
A3. plot_bootstrap_forest
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════════════════
# Global style
# ═══════════════════════════════════════════════════════════════════════════

METHOD_COLORS = {
    "KAO":       "#0072B2",  # blue
    "PySR":      "#D55E00",  # vermilion
    "RILS-ROLS": "#009E73",  # green
    "gplearn":   "#CC79A7",  # pink
    "Operon":    "#E69F00",  # orange
}

METHOD_MARKERS = {
    "KAO": "o",
    "PySR": "s",
    "RILS-ROLS": "^",
    "gplearn": "D",
    "Operon": "v",
}

# Ablation condition colours (blue-to-red gradient logic)
ABLATION_COLORS = {
    "KAO":          "#0072B2",
    "No_KAO":       "#D55E00",
    "No_KAO_cap7":  "#CC79A7",
    "No_KAO_cap9":  "#009E73",
    "No_KAO_cap11": "#F0E442",
}

# Dataset display-name mapping (CMD 8: title unification)
DATASET_DISPLAY = {
    "hydraulic":   "Hydraulic",
    "eicu":        "eICU",
    "nhanes":      "NHANES",
    "mimic_iv":    "MIMIC-IV",
    "mimic-iv":    "MIMIC-IV",
    "mimic":       "MIMIC-IV",
    "airfoil":     "AirFoil",
    "communities": "Crime & Communities",
    "auto_mpg":    "AutoMPG",
    "auto-mpg":    "AutoMPG",
}


def set_paper_style():
    """Apply journal-quality rcParams globally."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 300,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })
    return plt


def _ensure_dir(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def _color(method: str) -> str:
    return METHOD_COLORS.get(method, "#999999")


def _marker(method: str) -> str:
    return METHOD_MARKERS.get(method, "o")


# ═══════════════════════════════════════════════════════════════════════════
# 1. R² vs Complexity scatter with Pareto front overlay
# ═══════════════════════════════════════════════════════════════════════════

def plot_accuracy_complexity_scatter(
    df: pd.DataFrame,
    dataset: str,
    save_path: str,
    metric: str = "r2_test",
    complexity: str = "complexity_nodes",
):
    """Scatter: *metric* vs *complexity*, one colour / marker per method.

    30-seed individual points (small, translucent) + method-mean large point.
    A Pareto-front envelope is drawn across all methods' means.
    """
    plt = set_paper_style()

    sub = df[df["dataset"] == dataset]
    methods = sub["method"].unique()

    fig, ax = plt.subplots(figsize=(3.5, 3.0))

    mean_pts = []
    for m in methods:
        ms = sub[sub["method"] == m]
        c, mk = _color(m), _marker(m)
        ax.scatter(
            ms[complexity], ms[metric],
            s=12, alpha=0.3, color=c, marker=mk,
        )
        mx = ms[complexity].mean()
        my = ms[metric].mean()
        ax.scatter(
            mx, my, s=60, color=c, marker=mk,
            edgecolors="black", linewidths=0.6, label=m, zorder=5,
        )
        mean_pts.append((mx, my, m))

    # Pareto front of means (higher R² better, lower complexity better)
    mean_pts.sort(key=lambda t: t[0])  # sort by complexity ascending
    pareto = []
    best_r2 = -np.inf
    for cx, r2, m in mean_pts:
        if r2 > best_r2:
            pareto.append((cx, r2))
            best_r2 = r2
    if len(pareto) >= 2:
        px, py = zip(*pareto)
        ax.plot(px, py, "k--", linewidth=0.8, alpha=0.5, label="Pareto front")

    ax.set_xlabel(complexity.replace("_", " ").title())
    ax.set_ylabel("R² (test)")
    ax.set_title(dataset)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"  Saved {save_path}")


# ═══════════════════════════════════════════════════════════════════════════
# 2. Pairwise p-value / effect-size heatmap
# ═══════════════════════════════════════════════════════════════════════════

def plot_statistical_heatmap(
    pairwise_df: pd.DataFrame,
    save_path: str,
):
    """Method × method heatmap.

    *pairwise_df* must contain: comparison, p_value, cliffs_delta, dataset.
    Uses first dataset found; for multi-dataset combine beforehand.
    Colour encodes absolute Cliff's delta; cell text shows p-value + stars.
    """
    plt = set_paper_style()

    if "dataset" in pairwise_df.columns:
        ds = pairwise_df["dataset"].iloc[0]
        pw = pairwise_df[pairwise_df["dataset"] == ds]
    else:
        pw = pairwise_df

    # Extract unique methods
    method_set = set()
    for comp in pw["comparison"]:
        parts = comp.split(" vs ")
        method_set.update(parts)
    methods = sorted(method_set)
    n = len(methods)

    mat_p = np.ones((n, n))
    mat_d = np.zeros((n, n))
    idx = {m: i for i, m in enumerate(methods)}

    for _, row in pw.iterrows():
        parts = row["comparison"].split(" vs ")
        if len(parts) != 2:
            continue
        i, j = idx.get(parts[0]), idx.get(parts[1])
        if i is None or j is None:
            continue
        mat_p[i, j] = row["p_value"]
        mat_p[j, i] = row["p_value"]
        mat_d[i, j] = abs(row["cliffs_delta"])
        mat_d[j, i] = abs(row["cliffs_delta"])

    fig, ax = plt.subplots(figsize=(1.1 * n + 1.5, 1.1 * n + 0.5))
    im = ax.imshow(mat_d, cmap="YlOrRd", vmin=0, vmax=1, aspect="equal")

    ax.set_xticks(range(n))
    ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(methods, fontsize=8)

    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(j, i, "—", ha="center", va="center", fontsize=10)
            else:
                p = mat_p[i, j]
                stars = ""
                if p < 0.001:
                    stars = "***"
                elif p < 0.01:
                    stars = "**"
                elif p < 0.05:
                    stars = "*"
                txt = f"{p:.3f}{stars}"
                color = "white" if mat_d[i, j] > 0.55 else "black"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=9, color=color)

    fig.colorbar(im, ax=ax, label="|Cliff's delta|", shrink=0.8)
    ax.set_title("Pairwise comparisons (p-value / effect size)",
                 fontsize=14, fontweight='bold')

    fig.tight_layout()
    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"  Saved {save_path}")


# ═══════════════════════════════════════════════════════════════════════════
# 3. Critical Difference diagram (Nemenyi)
# ═══════════════════════════════════════════════════════════════════════════

def plot_critical_difference_diagram(
    friedman_result: Dict[str, Any],
    save_path: str,
):
    """Horizontal CD diagram: methods on axis by average rank, CD bar on top.

    *friedman_result* is the dict returned by
    ``StatisticalAnalyzer.friedman_nemenyi()``.
    """
    plt = set_paper_style()

    avg_ranks = friedman_result["avg_ranks"]
    cd = friedman_result["cd"]
    groups = friedman_result.get("groups", [])

    # --- Label shortening (修正A) ---
    label_map = {
        "No_KAO":       "No KAO",
        "No_KAO_cap7":  "Cap7",
        "No_KAO_cap9":  "Cap9",
        "No_KAO_cap11": "Cap11",
    }

    methods = sorted(avg_ranks, key=avg_ranks.get)
    n = len(methods)
    ranks = [avg_ranks[m] for m in methods]

    fig, ax = plt.subplots(figsize=(8.5, max(2.5, 0.45 * n + 1.5)))

    # Horizontal line
    ax.plot([min(ranks) - 0.5, max(ranks) + 0.5], [0, 0],
            "k-", linewidth=0.5)

    # --- Stagger labels to avoid overlap (修正B) ---
    # Split into upper (even index) and lower (odd index) groups
    upper = [(i, methods[i], ranks[i]) for i in range(n) if i % 2 == 0]
    lower = [(i, methods[i], ranks[i]) for i in range(n) if i % 2 != 0]

    # Sort each group by rank for stagger offset
    upper.sort(key=lambda x: x[2])
    lower.sort(key=lambda x: x[2])

    offset_step = 0.15
    base_y = 0.25

    # Compute staggered y for upper group
    y_offsets = {}
    for k, (idx, m, r) in enumerate(upper):
        y_offsets[idx] = base_y + offset_step * k
    for k, (idx, m, r) in enumerate(lower):
        y_offsets[idx] = -(base_y + offset_step * k)

    # Method labels
    for i, (m, r) in enumerate(zip(methods, ranks)):
        y_off = y_offsets[i]
        side = 1 if y_off > 0 else -1
        display_name = label_map.get(m, m)
        ax.plot(r, 0, "ko", markersize=5, zorder=5)
        ax.plot([r, r], [0, y_off], "k-", linewidth=0.5)
        ax.text(
            r, y_off + 0.05 * side, f"{display_name}\n({r:.2f})",
            ha="center", va="bottom" if side > 0 else "top",
            fontsize=7,
        )

    # CD bar — position above the tallest upper label
    max_upper_y = max(v for v in y_offsets.values() if v > 0) if y_offsets else 0.25
    y_cd = max_upper_y + 0.35
    mid = (min(ranks) + max(ranks)) / 2
    ax.plot([mid - cd / 2, mid + cd / 2], [y_cd, y_cd], "k-", linewidth=2)
    ax.plot([mid - cd / 2, mid - cd / 2], [y_cd - 0.04, y_cd + 0.04], "k-", linewidth=2)
    ax.plot([mid + cd / 2, mid + cd / 2], [y_cd - 0.04, y_cd + 0.04], "k-", linewidth=2)
    ax.text(mid, y_cd + 0.06, f"CD = {cd:.2f}", ha="center", va="bottom",
            fontsize=9, fontweight="bold")

    # Group brackets — position below the lowest lower label
    min_lower_y = min(v for v in y_offsets.values() if v < 0) if y_offsets else -0.25
    bracket_y = min_lower_y - 0.30
    for grp in groups:
        grp_ranks = [avg_ranks[m] for m in grp]
        lo, hi = min(grp_ranks), max(grp_ranks)
        ax.plot([lo, hi], [bracket_y, bracket_y], "-", linewidth=3,
                color="#0072B2", alpha=0.5)
        bracket_y -= 0.12

    ax.set_xlim(min(ranks) - 1, max(ranks) + 1)
    ax.set_ylim(bracket_y - 0.2, y_cd + 0.25)
    ax.set_xlabel("Average Rank")
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"  Saved {save_path}")


# ═══════════════════════════════════════════════════════════════════════════
# 4. Ablation bars
# ═══════════════════════════════════════════════════════════════════════════

def plot_ablation_bars(
    df: pd.DataFrame,
    save_path: str,
    metric: str = "r2_test",
):
    """Grouped bar chart for ablation conditions with 95 % CI + significance."""
    plt = set_paper_style()

    datasets = df["dataset"].unique()
    conditions = sorted(df["condition"].unique(),
                        key=lambda c: list(ABLATION_COLORS.keys()).index(c)
                        if c in ABLATION_COLORS else 99)

    n_ds = len(datasets)
    fig, axes = plt.subplots(1, max(n_ds, 1), figsize=(5 * max(n_ds, 1), 4),
                             squeeze=False)

    for didx, ds in enumerate(datasets):
        ax = axes[0, didx]
        sub = df[df["dataset"] == ds]
        means, cis, colors_list, labels = [], [], [], []
        kao_vals = sub.loc[sub["condition"] == "KAO", metric].values

        for cond in conditions:
            vals = sub.loc[sub["condition"] == cond, metric].values
            if len(vals) == 0:
                continue
            labels.append(cond)
            means.append(np.mean(vals))
            cis.append(1.96 * np.std(vals) / max(np.sqrt(len(vals)), 1))
            colors_list.append(ABLATION_COLORS.get(cond, "#999999"))

        x = np.arange(len(labels))
        bars = ax.bar(x, means, yerr=cis, capsize=3,
                      color=colors_list, edgecolor="black", linewidth=0.5)

        # Significance stars vs KAO
        for i, cond in enumerate(labels):
            if cond == "KAO":
                continue
            other = sub.loc[sub["condition"] == cond, metric].values
            n = min(len(kao_vals), len(other))
            if n < 5:
                continue
            from scipy.stats import wilcoxon
            try:
                _, p = wilcoxon(kao_vals[:n], other[:n])
            except Exception:
                continue
            stars = ""
            if p < 0.001:
                stars = "***"
            elif p < 0.01:
                stars = "**"
            elif p < 0.05:
                stars = "*"
            if stars:
                y_pos = means[i] + cis[i] + 0.005
                ax.text(i, y_pos, stars, ha="center", va="bottom",
                        fontsize=8, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("R² (test)")
        ax.set_title(ds)
        ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"  Saved {save_path}")


# ═══════════════════════════════════════════════════════════════════════════
# 5. Hypervolume vs Time curves
# ═══════════════════════════════════════════════════════════════════════════

def plot_hypervolume_curves(
    hv_df: pd.DataFrame,
    save_path: str,
):
    """HV vs Time with 95 % CI bands.  KAO blue, No_KAO red."""
    plt = set_paper_style()

    cond_colors = {"KAO": "#0072B2", "No_KAO": "#D55E00"}
    datasets = hv_df["dataset"].unique()
    n = len(datasets)
    fig, axes = plt.subplots(1, max(n, 1), figsize=(5 * max(n, 1), 3.5),
                             squeeze=False)

    for idx, ds in enumerate(datasets):
        ax = axes[0, idx]
        sub = hv_df[hv_df["dataset"] == ds]
        for cond in ["KAO", "No_KAO"]:
            c = sub[sub["condition"] == cond]
            if c.empty:
                continue
            g = c.groupby("checkpoint")["hypervolume"]
            m = g.mean()
            s = g.std()
            cnt = g.count()
            ci = 1.96 * s / np.sqrt(cnt.clip(lower=1))
            color = cond_colors.get(cond, "gray")
            ax.plot(m.index, m.values, "o-", markersize=4, color=color,
                    label=cond, linewidth=1.5)
            ax.fill_between(m.index, m.values - ci.values,
                            m.values + ci.values, alpha=0.2, color=color)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Hypervolume")
        ax.set_title(ds)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)

    fig.tight_layout()
    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"  Saved {save_path}")


# ═══════════════════════════════════════════════════════════════════════════
# 6. Pareto front snapshots (2 × 3 panels)
# ═══════════════════════════════════════════════════════════════════════════

def plot_pareto_snapshots(
    snapshots: Dict[str, Any],
    dataset: str,
    save_path: str,
    checkpoints: Optional[List[int]] = None,
):
    """2-row × 3-col panels.  Row 0 = KAO, Row 1 = No_KAO.

    *snapshots* : ``{f"{cond}_seed{s}": {cp: [(e, c, expr), ...], ...}}``
    (the JSON-serialisable format from ``run_pareto_time.py``).
    """
    plt = set_paper_style()

    if checkpoints is None:
        checkpoints = [10, 20, 30, 40, 50, 60]
    n_cols = min(len(checkpoints), 6)
    selected = checkpoints[:n_cols]

    cond_colors = {"KAO": "#0072B2", "No_KAO": "#D55E00"}
    fig, axes = plt.subplots(2, n_cols, figsize=(3.5 * n_cols, 6), squeeze=False)

    for row, cond in enumerate(["KAO", "No_KAO"]):
        color = cond_colors.get(cond, "gray")
        for col, cp in enumerate(selected):
            ax = axes[row, col]
            cp_key = str(cp)
            for key, snap_dict in snapshots.items():
                if not key.startswith(cond + "_seed"):
                    continue
                front = snap_dict.get(cp_key, [])
                if not front:
                    continue
                errors = [float(p[0]) for p in front]
                complexities = [float(p[1]) for p in front]
                ax.scatter(complexities, errors, s=8, alpha=0.3, color=color)
            ax.set_xlabel("Complexity")
            ax.set_ylabel("CV-MSE")
            ax.set_title(f"{cond}  t={cp}s")
            ax.grid(alpha=0.25)

    fig.suptitle(f"Pareto Front Evolution — {dataset}", fontsize=12, y=1.02)
    fig.tight_layout()
    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"  Saved {save_path}")


# ═══════════════════════════════════════════════════════════════════════════
# 7. SRBench recovery-rate heatmap
# ═══════════════════════════════════════════════════════════════════════════

def plot_srbench_heatmap(
    summary_df: pd.DataFrame,
    save_path: str,
    noise_std: float = 0.0,
):
    """Benchmark × method heatmap of symbolic recovery rate."""
    plt = set_paper_style()

    sub = summary_df
    if "noise_std" in sub.columns:
        sub = sub[sub["noise_std"] == noise_std]

    pivot = sub.pivot_table(
        index="benchmark", columns="method", values="recovery_rate",
        aggfunc="first",
    )

    nr, nc = pivot.shape
    fig, ax = plt.subplots(figsize=(max(3, 1.2 * nc), max(2.5, 0.7 * nr)))
    im = ax.imshow(pivot.values, cmap="YlGn", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(nc))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(nr))
    ax.set_yticklabels(pivot.index, fontsize=8)

    for i in range(nr):
        for j in range(nc):
            val = pivot.values[i, j]
            if np.isfinite(val):
                color = "white" if val >= 0.7 else "black"
                ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                        fontsize=8, color=color)

    fig.colorbar(im, ax=ax, label="Recovery Rate", shrink=0.85)
    ax.set_title(f"Symbolic Recovery (noise={noise_std})",
                 fontsize=14, fontweight='bold')

    fig.tight_layout()
    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"  Saved {save_path}")


# ═══════════════════════════════════════════════════════════════════════════
# 8. Violin + strip (jitter) plot
# ═══════════════════════════════════════════════════════════════════════════

def plot_violin_strip(
    df: pd.DataFrame,
    metric: str,
    save_path: str,
    dataset: Optional[str] = None,
):
    """Violin + jitter strip showing the full 30-seed distribution per method."""
    plt = set_paper_style()

    sub = df.copy()
    if dataset is not None:
        sub = sub[sub["dataset"] == dataset]

    methods = sorted(sub["method"].unique())
    n = len(methods)

    fig, ax = plt.subplots(figsize=(max(3.5, 0.9 * n), 3.5))

    positions = np.arange(n)
    for i, m in enumerate(methods):
        vals = sub.loc[sub["method"] == m, metric].dropna().values
        if len(vals) == 0:
            continue
        color = _color(m)

        # Violin
        parts = ax.violinplot(vals, positions=[i], showmeans=False,
                              showextrema=False, widths=0.7)
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_alpha(0.3)
            pc.set_edgecolor(color)

        # Jitter strip
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(vals))
        ax.scatter(i + jitter, vals, s=10, alpha=0.6, color=color,
                   edgecolors="none", zorder=4)

        # Mean + 95 % CI
        mean = vals.mean()
        ci = 1.96 * vals.std() / max(np.sqrt(len(vals)), 1)
        ax.plot(i, mean, "D", markersize=6, color=color,
                markeredgecolor="black", markeredgewidth=0.5, zorder=5)
        ax.plot([i, i], [mean - ci, mean + ci], "-", color="black",
                linewidth=1.2, zorder=5)

    ax.set_xticks(positions)
    ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(metric.replace("_", " ").title())
    if dataset:
        ax.set_title(DATASET_DISPLAY.get(dataset, dataset),
                     fontsize=14, fontweight='bold')
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"  Saved {save_path}")


# ═══════════════════════════════════════════════════════════════════════════
# A1. Residual diagnostics
# ═══════════════════════════════════════════════════════════════════════════

def plot_residual_diagnostics(
    df: pd.DataFrame,
    method: str,
    dataset: str,
    save_path: str,
):
    """2-panel: (a) predicted vs actual, (b) residual histogram.

    *df* must contain columns: method, dataset, seed, y_actual, y_predicted.
    Uses seed=1 by default.
    """
    plt = set_paper_style()

    sub = df[(df["method"] == method) & (df["dataset"] == dataset)]
    if "seed" in sub.columns:
        seed_val = sub["seed"].min()
        sub = sub[sub["seed"] == seed_val]

    if "y_actual" not in sub.columns or "y_predicted" not in sub.columns:
        print(f"  Skipping residual diagnostics: missing y_actual/y_predicted columns")
        return

    y_act = sub["y_actual"].values
    y_pred = sub["y_predicted"].values
    residuals = y_act - y_pred

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.0))

    # (a) Predicted vs Actual
    ax1.scatter(y_act, y_pred, s=8, alpha=0.5, color=_color(method))
    lo = min(y_act.min(), y_pred.min())
    hi = max(y_act.max(), y_pred.max())
    ax1.plot([lo, hi], [lo, hi], "k--", linewidth=0.8)
    ax1.set_xlabel("Actual")
    ax1.set_ylabel("Predicted")
    ax1.set_title(f"{method} — {dataset}")
    ax1.grid(alpha=0.25)

    # (b) Residual histogram
    ax2.hist(residuals, bins=30, color=_color(method), alpha=0.7,
             edgecolor="black", linewidth=0.3)
    ax2.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.set_xlabel("Residual")
    ax2.set_ylabel("Count")
    ax2.set_title("Residual Distribution")
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"  Saved {save_path}")


# ═══════════════════════════════════════════════════════════════════════════
# A2. Convergence curves (from JSON logs)
# ═══════════════════════════════════════════════════════════════════════════

def plot_convergence_curves(
    logs_dir: str,
    dataset: str,
    save_path: str,
):
    """Plot best-so-far R² over generations/time from per-run JSON logs.

    Expects JSON files in *logs_dir* matching ``{dataset}_*_seed*.json``
    with a ``pareto_snapshots`` or ``time_checkpoints`` field.
    Falls back to a simple runtime-vs-R² scatter when snapshots are absent.
    """
    import glob
    import json

    plt = set_paper_style()

    pattern = os.path.join(logs_dir, f"{dataset}_*.json")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"  No log files matching {pattern}")
        return

    fig, ax = plt.subplots(figsize=(5, 3.5))

    for fpath in files:
        try:
            with open(fpath) as f:
                log = json.load(f)
        except Exception:
            continue
        r2 = log.get("r2_test", None)
        rt = log.get("runtime", None)
        cond = log.get("condition", log.get("method", "unknown"))
        if r2 is not None and rt is not None:
            color = ABLATION_COLORS.get(cond, _color(cond))
            ax.scatter(rt, r2, s=15, alpha=0.5, color=color)

    ax.set_xlabel("Runtime (s)")
    ax.set_ylabel("R² (test)")
    display = DATASET_DISPLAY.get(dataset, dataset)
    ax.set_title(f"Convergence — {display}",
                 fontsize=14, fontweight='bold')
    ax.grid(alpha=0.25)

    fig.tight_layout()
    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"  Saved {save_path}")


# ═══════════════════════════════════════════════════════════════════════════
# A3. Bootstrap forest plot
# ═══════════════════════════════════════════════════════════════════════════

def plot_bootstrap_forest(
    pairwise_df: pd.DataFrame,
    save_path: str,
):
    """Forest plot of bootstrap mean-difference CIs — 2-column layout.

    Each row is a comparison (e.g. KAO vs PySR); horizontal error bar
    shows [ci_lower, ci_upper], diamond at mean_diff.
    Left column: MIMIC-IV, eICU, NHANES, Airfoil
    Right column: Auto MPG, Communities, Hydraulic
    """
    plt = set_paper_style()

    pw = pairwise_df.copy()
    if "comparison" not in pw.columns:
        print("  Skipping forest plot: missing 'comparison' column")
        return

    pw = pw.dropna(subset=["ci_lower", "ci_upper", "mean_diff"])
    if pw.empty:
        print("  Skipping forest plot: no valid CI data")
        return

    DS_DISPLAY = {
        "mimic_iv": "MIMIC-IV", "eicu": "eICU", "nhanes": "NHANES",
        "airfoil": "Airfoil", "auto_mpg": "Auto MPG",
        "communities": "Communities", "hydraulic": "Hydraulic",
    }

    datasets_left  = ["mimic_iv", "eicu", "nhanes", "airfoil"]
    datasets_right = ["auto_mpg", "communities", "hydraulic"]

    # Determine shared x-limits
    xmin = pw["ci_lower"].min()
    xmax = pw["ci_upper"].max()
    x_margin = (xmax - xmin) * 0.08
    xlim = (xmin - x_margin, xmax + x_margin)

    def _draw_column(ax, ds_list):
        y_pos = 0
        tick_positions = []
        tick_labels = []
        boundary_positions = []

        for ds in ds_list:
            sub = pw[pw["dataset"] == ds] if "dataset" in pw.columns else pw
            if sub.empty:
                continue

            # Dataset header
            ax.text(xlim[0] + 0.01 * (xlim[1] - xlim[0]), y_pos,
                    DS_DISPLAY.get(ds, ds),
                    fontsize=8, fontweight="bold", va="center")
            y_pos += 1

            for _, row in sub.iterrows():
                color = "#0072B2" if row["mean_diff"] > 0 else "#D55E00"
                ax.plot([row["ci_lower"], row["ci_upper"]], [y_pos, y_pos],
                        "-", color=color, linewidth=1.5)
                ax.plot(row["mean_diff"], y_pos, "D", color=color,
                        markersize=5, markeredgecolor="black",
                        markeredgewidth=0.4)
                tick_positions.append(y_pos)
                # Shorten label: remove "KAO vs " prefix for compactness
                lbl = row["comparison"]
                if lbl.startswith("KAO vs "):
                    lbl = "vs " + lbl[7:]
                tick_labels.append(lbl)
                y_pos += 1

            boundary_positions.append(y_pos - 0.5)
            y_pos += 0.5  # spacing between datasets

        # Draw dataset boundary lines
        for by in boundary_positions[:-1]:
            ax.axhline(y=by, color="lightgrey", linestyle="-", linewidth=0.5)

        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlim(xlim)
        ax.set_ylim(y_pos - 0.5, -0.5)
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_labels, fontsize=7)
        ax.set_xlabel("Mean Difference (R²)")
        ax.grid(axis="x", alpha=0.25)

        return y_pos

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 9),
                                             sharey=False)

    _draw_column(ax_left, datasets_left)
    _draw_column(ax_right, datasets_right)

    fig.suptitle("Bootstrap 95% CI of Pairwise Differences",
                 fontsize=14, fontweight='bold')

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"  Saved {save_path}")
