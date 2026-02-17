# -*- coding: utf-8 -*-
"""
Comprehensive Statistical Analysis Toolkit
============================================
Provides ``StatisticalAnalyzer`` — the single entry-point for all
pairwise tests, multiple-comparison corrections, Friedman–Nemenyi
analysis, and LaTeX / CSV table generation used in the KAO paper.

Expected input DataFrame columns
---------------------------------
dataset, method, seed, r2_test, complexity_nodes, complexity_chars,
runtime, expression

Dependencies: numpy, pandas, scipy.  (Optional: scikit-posthocs for
Nemenyi; a pure-Python fallback is included.)
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Effect-size thresholds (Cliff's delta)
# ---------------------------------------------------------------------------
_CLIFF_THRESHOLDS = [
    (0.147, "negligible"),
    (0.330, "small"),
    (0.474, "medium"),
]


def _cliff_category(d: float) -> str:
    ad = abs(d)
    for thresh, label in _CLIFF_THRESHOLDS:
        if ad < thresh:
            return label
    return "large"


# ═══════════════════════════════════════════════════════════════════════════
# StatisticalAnalyzer
# ═══════════════════════════════════════════════════════════════════════════

class StatisticalAnalyzer:
    """Stateless helper — every method is a classmethod or staticmethod."""

    # ------------------------------------------------------------------
    # 1. Pairwise comparison
    # ------------------------------------------------------------------
    @staticmethod
    def pairwise_comparison(
        df: pd.DataFrame,
        method_a: str,
        method_b: str,
        dataset: str,
        metric: str = "r2_test",
        n_bootstrap: int = 10_000,
        bootstrap_seed: int = 0,
    ) -> Dict[str, Any]:
        """Wilcoxon signed-rank + Cliff's delta + bootstrap CI.

        Parameters
        ----------
        df : DataFrame with columns [dataset, method, seed, <metric>].
        method_a, method_b : names to compare.
        dataset : dataset label to filter on.
        metric : column to compare (default ``r2_test``).
        n_bootstrap : bootstrap resamples for CI of mean difference.
        bootstrap_seed : RNG seed for reproducibility.

        Returns
        -------
        dict with keys: stat, p_value, cliffs_delta, effect_category,
        ci_lower, ci_upper, mean_diff, n_pairs.
        """
        from scipy.stats import wilcoxon

        # Type safety: ensure metric column is numeric
        if metric in df.columns:
            df = df.copy()
            df[metric] = pd.to_numeric(df[metric], errors="coerce")

        # v3.2: inner-join on (dataset, seed) for proper alignment
        sub = df[df["dataset"] == dataset]
        A = sub.loc[sub["method"] == method_a, ["seed", metric]].rename(columns={metric: "a"})
        B = sub.loc[sub["method"] == method_b, ["seed", metric]].rename(columns={metric: "b"})
        M = A.merge(B, on="seed", how="inner").dropna()

        n = len(M)
        if n < 5:
            return {
                "stat": float("nan"),
                "p_value": float("nan"),
                "cliffs_delta": float("nan"),
                "effect_category": "insufficient_data",
                "ci_lower": float("nan"),
                "ci_upper": float("nan"),
                "mean_diff": float("nan"),
                "n_pairs": n,
            }

        a = M["a"].to_numpy()
        b = M["b"].to_numpy()

        # Wilcoxon
        try:
            stat, pval = wilcoxon(a, b)
        except Exception:
            stat, pval = float("nan"), float("nan")

        # Cliff's delta
        cd = StatisticalAnalyzer._cliffs_delta(a, b)
        cat = _cliff_category(cd)

        # Bootstrap 95 % CI of mean difference
        rng = np.random.RandomState(bootstrap_seed)
        diffs = a - b
        boot_means = np.empty(n_bootstrap)
        for i in range(n_bootstrap):
            idx = rng.randint(0, n, size=n)
            boot_means[i] = diffs[idx].mean()
        ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])

        return {
            "stat": float(stat),
            "p_value": float(pval),
            "cliffs_delta": float(cd),
            "effect_category": cat,
            "ci_lower": float(ci_lo),
            "ci_upper": float(ci_hi),
            "mean_diff": float(diffs.mean()),
            "n_pairs": n,
        }

    @staticmethod
    def _cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
        nx, ny = len(x), len(y)
        if nx == 0 or ny == 0:
            return 0.0
        more = sum(1 for xi in x for yi in y if xi > yi)
        less = sum(1 for xi in x for yi in y if xi < yi)
        return (more - less) / (nx * ny)

    # ------------------------------------------------------------------
    # 2. Holm-Bonferroni correction
    # ------------------------------------------------------------------
    @staticmethod
    def holm_bonferroni_correction(
        p_values: Dict[Tuple, float],
    ) -> Dict[Tuple, float]:
        """Apply Holm–Bonferroni step-down correction.

        Parameters
        ----------
        p_values : mapping  (method_a, method_b, dataset) -> raw p-value.

        Returns
        -------
        Same structure with corrected p-values (clipped at 1.0).
        """
        keys = list(p_values.keys())
        raw = np.array([p_values[k] for k in keys])
        m = len(raw)

        order = np.argsort(raw)
        corrected = np.empty(m)
        cummax = 0.0
        for rank, idx in enumerate(order):
            adj = raw[idx] * (m - rank)
            cummax = max(cummax, adj)
            corrected[idx] = min(cummax, 1.0)

        return {k: float(corrected[i]) for i, k in enumerate(keys)}

    # ------------------------------------------------------------------
    # 3. Friedman + Nemenyi
    # ------------------------------------------------------------------
    @staticmethod
    def friedman_nemenyi(
        df: pd.DataFrame,
        metric: str = "r2_test",
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        """Friedman rank test across datasets + Nemenyi post-hoc CD.

        The input *df* must contain columns: dataset, method, seed, <metric>.
        For each (dataset, seed) we get one value per method; the Friedman
        test operates over seeds-within-datasets as blocks.

        Returns
        -------
        dict with keys: friedman_stat, friedman_p, avg_ranks (dict),
        cd (critical difference), n_blocks, n_methods, groups.
        """
        from scipy.stats import friedmanchisquare

        methods = sorted(df["method"].unique())
        k = len(methods)

        # Build a matrix: rows = blocks (dataset × seed), cols = methods
        blocks = []
        for (ds, seed), grp in df.groupby(["dataset", "seed"]):
            row = {}
            for m in methods:
                vals = grp.loc[grp["method"] == m, metric].values
                if len(vals) == 0:
                    break
                row[m] = vals[0]
            if len(row) == k:
                blocks.append(row)

        if len(blocks) < 3:
            return {
                "friedman_stat": float("nan"),
                "friedman_p": float("nan"),
                "avg_ranks": {},
                "cd": float("nan"),
                "n_blocks": len(blocks),
                "n_methods": k,
                "groups": [],
            }

        mat = np.array([[b[m] for m in methods] for b in blocks])
        n = len(blocks)

        # Friedman test
        try:
            stat, p = friedmanchisquare(*[mat[:, j] for j in range(k)])
        except Exception:
            stat, p = float("nan"), float("nan")

        # Average ranks (higher metric = rank 1)
        ranks = np.zeros_like(mat)
        for i in range(n):
            order = np.argsort(-mat[i])  # descending
            for r, j in enumerate(order):
                ranks[i, j] = r + 1
        avg_ranks = {m: float(ranks[:, j].mean()) for j, m in enumerate(methods)}

        # Nemenyi critical difference
        from scipy.stats import studentized_range
        q_alpha = studentized_range.ppf(1 - alpha, k, np.inf)
        cd = q_alpha / np.sqrt(2) * np.sqrt(k * (k + 1) / (6 * n))

        # Identify groups (methods whose rank difference < CD)
        groups = StatisticalAnalyzer._nemenyi_groups(avg_ranks, cd)

        return {
            "friedman_stat": float(stat),
            "friedman_p": float(p),
            "avg_ranks": avg_ranks,
            "cd": float(cd),
            "n_blocks": n,
            "n_methods": k,
            "groups": groups,
        }

    @staticmethod
    def _nemenyi_groups(
        avg_ranks: Dict[str, float], cd: float,
    ) -> List[List[str]]:
        """Find groups of methods not significantly different."""
        methods = sorted(avg_ranks, key=avg_ranks.get)
        groups: List[List[str]] = []
        for i, mi in enumerate(methods):
            group = [mi]
            for mj in methods[i + 1:]:
                if abs(avg_ranks[mi] - avg_ranks[mj]) < cd:
                    group.append(mj)
            if len(group) > 1 and group not in groups:
                groups.append(group)
        return groups

    # ------------------------------------------------------------------
    # 4. Summary table (LaTeX booktabs)
    # ------------------------------------------------------------------
    @staticmethod
    def generate_summary_table(
        df: pd.DataFrame,
        reference_method: str = "KAO",
        metric: str = "r2_test",
    ) -> str:
        r"""Generate a LaTeX booktabs table.

        Rows = methods, columns = datasets.  Each cell shows
        ``mean +/- std``.  The best value per dataset is wrapped in
        ``\textbf{}``.  An extra column shows the Wilcoxon p-value
        vs *reference_method* with significance stars.

        Returns
        -------
        str : LaTeX source.
        """
        # Type safety: ensure metric column is numeric
        if metric in df.columns:
            df = df.copy()
            df[metric] = pd.to_numeric(df[metric], errors="coerce")

        datasets = sorted(df["dataset"].dropna().unique())
        methods = sorted(df["method"].dropna().unique())

        # Pre-compute means / stds / p-values
        cells: Dict[Tuple[str, str], str] = {}
        pvals: Dict[Tuple[str, str], str] = {}
        best_per_ds: Dict[str, str] = {}

        for ds in datasets:
            best_val = -np.inf
            best_m = ""
            for m in methods:
                vals = df.loc[
                    (df["dataset"] == ds) & (df["method"] == m), metric
                ].values
                if len(vals) == 0:
                    cells[(m, ds)] = "---"
                    continue
                mean = vals.mean()
                std = vals.std()
                cells[(m, ds)] = f"{mean:.4f} $\\pm$ {std:.4f}"
                if mean > best_val:
                    best_val = mean
                    best_m = m
            best_per_ds[ds] = best_m

        # p-values vs reference
        for ds in datasets:
            for m in methods:
                if m == reference_method:
                    pvals[(m, ds)] = ""
                    continue
                res = StatisticalAnalyzer.pairwise_comparison(
                    df, reference_method, m, ds, metric=metric,
                )
                p = res["p_value"]
                if np.isnan(p):
                    pvals[(m, ds)] = "---"
                else:
                    stars = ""
                    if p < 0.001:
                        stars = "***"
                    elif p < 0.01:
                        stars = "**"
                    elif p < 0.05:
                        stars = "*"
                    pvals[(m, ds)] = f"{p:.4f}{stars}"

        # Bold best
        for ds in datasets:
            m = best_per_ds[ds]
            if (m, ds) in cells and cells[(m, ds)] != "---":
                cells[(m, ds)] = "\\textbf{" + cells[(m, ds)] + "}"

        # Build LaTeX — v3.2: one p-value column per dataset
        n_ds = len(datasets)
        col_spec = "l" + "c" * n_ds + "c" * n_ds
        lines = [
            "\\begin{table}[t]",
            "\\centering",
            "\\caption{Mean $\\pm$ std of " + metric.replace("_", "\\_") + " across 30 seeds.}",
            "\\label{tab:summary_" + metric + "}",
            "\\begin{tabular}{" + col_spec + "}",
            "\\toprule",
        ]

        # Header row: metrics columns then p-value columns
        header_parts = ["Method"]
        for ds in datasets:
            header_parts.append(ds.replace("_", "\\_"))
        for ds in datasets:
            header_parts.append("$p$ (" + ds.replace("_", "\\_") + ")")
        header = " & ".join(header_parts) + " \\\\"
        lines.append(header)
        lines.append("\\midrule")

        for m in methods:
            row_parts = [m.replace("_", "\\_")]
            for ds in datasets:
                row_parts.append(cells.get((m, ds), "---"))
            for ds in datasets:
                row_parts.append(pvals.get((m, ds), ""))
            lines.append(" & ".join(row_parts) + " \\\\")

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 5. Ablation table
    # ------------------------------------------------------------------
    @staticmethod
    def generate_ablation_table(
        df: pd.DataFrame,
        reference: str = "KAO",
    ) -> str:
        r"""LaTeX booktabs table for ablation conditions.

        Columns: condition, mean R^2 +/- std, mean complexity +/- std,
        p-value vs reference, Cliff's delta.
        """
        datasets = sorted(df["dataset"].unique())
        conditions = sorted(df["condition"].unique())

        lines = [
            "\\begin{table}[t]",
            "\\centering",
            "\\caption{Ablation results (mean $\\pm$ std, 30 seeds).}",
            "\\label{tab:ablation}",
            "\\begin{tabular}{lcccc}",
            "\\toprule",
            "Condition & R$^2$ (test) & Complexity & $p$-value & Cliff's $\\delta$ \\\\",
            "\\midrule",
        ]

        for ds in datasets:
            if len(datasets) > 1:
                lines.append("\\multicolumn{5}{l}{\\textit{" + ds.replace("_", "\\_") + "}} \\\\")
            sub = df[df["dataset"] == ds]
            ref_vals = sub.loc[sub["condition"] == reference, "r2_test"].values

            best_r2 = -np.inf
            best_cond = ""
            cond_data = {}
            for cond in conditions:
                vals = sub.loc[sub["condition"] == cond, "r2_test"].values
                comp = sub.loc[sub["condition"] == cond, "complexity_nodes"].values
                if len(vals) == 0:
                    continue
                cond_data[cond] = (vals, comp)
                if vals.mean() > best_r2:
                    best_r2 = vals.mean()
                    best_cond = cond

            for cond in conditions:
                if cond not in cond_data:
                    continue
                vals, comp = cond_data[cond]
                r2_str = f"{vals.mean():.4f} $\\pm$ {vals.std():.4f}"
                comp_str = f"{comp.mean():.1f} $\\pm$ {comp.std():.1f}"

                if cond == best_cond:
                    r2_str = "\\textbf{" + r2_str + "}"

                if cond == reference:
                    p_str = "---"
                    d_str = "---"
                else:
                    n = min(len(ref_vals), len(vals))
                    if n >= 5:
                        from scipy.stats import wilcoxon
                        try:
                            _, pval = wilcoxon(ref_vals[:n], vals[:n])
                        except Exception:
                            pval = float("nan")
                        cd = StatisticalAnalyzer._cliffs_delta(ref_vals[:n], vals[:n])
                        stars = ""
                        if pval < 0.001:
                            stars = "***"
                        elif pval < 0.01:
                            stars = "**"
                        elif pval < 0.05:
                            stars = "*"
                        p_str = f"{pval:.4f}{stars}"
                        cat = _cliff_category(cd)
                        d_str = f"{cd:.3f} ({cat})"
                    else:
                        p_str = "---"
                        d_str = "---"

                cond_label = cond.replace("_", "\\_")
                lines.append(
                    f"{cond_label} & {r2_str} & {comp_str} & {p_str} & {d_str} \\\\"
                )

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Convenience: run all pairwise comparisons
    # ------------------------------------------------------------------
    @staticmethod
    def all_pairwise(
        df: pd.DataFrame,
        reference: str = "KAO",
        metric: str = "r2_test",
        correct: bool = True,
    ) -> pd.DataFrame:
        """Return a DataFrame of pairwise results for *reference* vs every
        other method, across every dataset.  Optionally applies
        Holm–Bonferroni correction.
        """
        # Type safety: ensure metric column is numeric
        if metric in df.columns:
            df = df.copy()
            df[metric] = pd.to_numeric(df[metric], errors="coerce")

        datasets = sorted(df["dataset"].dropna().unique())
        methods = sorted(df["method"].dropna().unique())
        others = [m for m in methods if m != reference]

        raw_p: Dict[Tuple, float] = {}
        records: List[Dict] = []

        for ds in datasets:
            for m in others:
                res = StatisticalAnalyzer.pairwise_comparison(
                    df, reference, m, ds, metric=metric,
                )
                raw_p[(reference, m, ds)] = res["p_value"]
                records.append({
                    "dataset": ds,
                    "comparison": f"{reference} vs {m}",
                    **res,
                })

        if correct and records:
            corrected = StatisticalAnalyzer.holm_bonferroni_correction(raw_p)
            for rec in records:
                ds = rec["dataset"]
                comp = rec["comparison"]
                m = comp.split(" vs ")[1]
                key = (reference, m, ds)
                rec["p_value_corrected"] = corrected.get(key, rec["p_value"])

        return pd.DataFrame(records)
