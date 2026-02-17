#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
One-Click Paper Figure & Table Generator
==========================================
Scans ``results/`` for experiment outputs, runs the full statistical
analysis pipeline, and produces every figure + LaTeX table needed for
the KAO paper.

Usage:
  python analysis/generate_paper_figures.py --results-dir results/
  python analysis/generate_paper_figures.py --results-dir results/ --skip-plots

Outputs:
  results/figures/*.pdf   all figures
  results/tables/*.tex    LaTeX booktabs tables
  results/tables/*.csv    CSV summaries
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Ensure project root on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from analysis.statistics import StatisticalAnalyzer   # noqa: E402
from analysis import visualization as viz             # noqa: E402

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate all paper figures & tables")
    p.add_argument("--results-dir", type=str, default="results/")
    p.add_argument("--skip-plots", action="store_true")
    return p.parse_args()

# ---------------------------------------------------------------------------
# Result loaders
# ---------------------------------------------------------------------------

def load_json_logs(directory: str, pattern: str = "*.json") -> pd.DataFrame:
    """Load all JSON log files from *directory* into a DataFrame."""
    files = sorted(glob.glob(os.path.join(directory, pattern)))
    rows = []
    for f in files:
        try:
            with open(f) as fh:
                rows.append(json.load(fh))
        except Exception:
            continue
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def load_csv_safe(path: str) -> pd.DataFrame:
    """Load a CSV; return empty DataFrame on failure."""
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def collect_main_results(results_dir: str) -> pd.DataFrame:
    """Merge per-dataset/per-method results into a single tidy DataFrame.

    v3.2: Primary scan — recursively find all ``result.json`` files.
    Falls back to legacy layout if no result.json files are found.

    Also loads:
      results/ablation/*.json
      results/tables/srbench_results.csv
    """
    frames = []

    # v3.2: Primary — recursive result.json scan
    result_jsons = list(Path(results_dir).rglob('result.json'))
    if result_jsons:
        rows = []
        for p in result_jsons:
            try:
                rows.append(json.loads(p.read_text(encoding='utf-8')))
            except Exception:
                pass
        if rows:
            frames.append(pd.DataFrame(rows))
    else:
        # Legacy layout fallback: results/{dataset}/*s/{method}/seed_*/*.json
        for budget_dir in glob.glob(os.path.join(results_dir, "*", "*s")):
            dataset = Path(budget_dir).parent.name
            for method_dir in glob.glob(os.path.join(budget_dir, "*")):
                method = Path(method_dir).name
                df = load_json_logs(method_dir)
                if not df.empty:
                    if "dataset" not in df.columns:
                        df["dataset"] = dataset
                    if "method" not in df.columns:
                        df["method"] = method
                    frames.append(df)

    # Ablation logs
    abl_df = load_json_logs(os.path.join(results_dir, "ablation"))
    if not abl_df.empty:
        if "method" not in abl_df.columns and "condition" in abl_df.columns:
            abl_df["method"] = abl_df["condition"]
        frames.append(abl_df)

    # SRBench CSV
    srb = load_csv_safe(os.path.join(results_dir, "tables", "srbench_results.csv"))
    if not srb.empty:
        frames.append(srb)

    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    rdir = args.results_dir
    fig_dir = os.path.join(rdir, "figures")
    tab_dir = os.path.join(rdir, "tables")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tab_dir, exist_ok=True)

    n_figs = 0
    n_tabs = 0

    print("=" * 60)
    print("Paper Figures & Tables Generator")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load all results
    # ------------------------------------------------------------------
    print("\n[1] Loading experiment results ...")
    main_df = collect_main_results(rdir)
    ablation_csv = load_csv_safe(os.path.join(tab_dir, "ablation_summary.csv"))
    hv_csv = load_csv_safe(os.path.join(tab_dir, "pareto_time_summary.csv"))
    hv_raw = load_csv_safe(
        os.path.join(rdir, "pareto_time",
                     next(iter(glob.glob(os.path.join(rdir, "pareto_time", "*_hv.csv"))), ""))
    )
    srbench_summary = load_csv_safe(os.path.join(tab_dir, "srbench_summary.csv"))
    ablation_raw = load_json_logs(os.path.join(rdir, "ablation"))

    # Type cleansing: coerce known numeric columns to float
    _NUMERIC_COLS = [
        "r2_test", "r2_cv", "complexity_nodes", "complexity_chars",
        "runtime", "train_mse", "test_mse", "test_loss", "cv_loss",
    ]
    for col in _NUMERIC_COLS:
        if not main_df.empty and col in main_df.columns:
            main_df[col] = pd.to_numeric(main_df[col], errors="coerce")

    print(f"  main results : {len(main_df)} rows")
    print(f"  ablation     : {len(ablation_csv)} rows")
    print(f"  hv summary   : {len(hv_csv)} rows")
    print(f"  srbench      : {len(srbench_summary)} rows")

    # ------------------------------------------------------------------
    # 2. Statistical analysis
    # ------------------------------------------------------------------
    print("\n[2] Running statistical analysis ...")
    pairwise_df = pd.DataFrame()
    friedman_result = {}

    if not main_df.empty and "method" in main_df.columns:
        # Pairwise comparisons (KAO vs each other method)
        methods = main_df["method"].unique()
        if "KAO" in methods and len(methods) > 1:
            try:
                pairwise_df = StatisticalAnalyzer.all_pairwise(
                    main_df, reference="KAO", metric="r2_test",
                )
                pairwise_df.to_csv(os.path.join(tab_dir, "pairwise_tests.csv"), index=False)
                n_tabs += 1
                print(f"  pairwise tests: {len(pairwise_df)} comparisons")
            except Exception as exc:
                print(f"  pairwise tests failed: {exc}")

        # Friedman–Nemenyi
        if len(methods) >= 3:
            try:
                friedman_result = StatisticalAnalyzer.friedman_nemenyi(
                    main_df, metric="r2_test",
                )
                if friedman_result.get("friedman_p") is not None:
                    print(f"  Friedman p={friedman_result['friedman_p']:.4f}")
            except Exception as exc:
                print(f"  Friedman test failed: {exc}")

        # LaTeX summary table
        try:
            latex = StatisticalAnalyzer.generate_summary_table(main_df)
            with open(os.path.join(tab_dir, "summary_r2.tex"), "w") as f:
                f.write(latex)
            n_tabs += 1
            print("  summary table saved")
        except Exception as exc:
            print(f"  summary table failed: {exc}")

    # Ablation LaTeX table
    if not ablation_raw.empty and "condition" in ablation_raw.columns:
        try:
            abl_tex = StatisticalAnalyzer.generate_ablation_table(ablation_raw)
            with open(os.path.join(tab_dir, "ablation_table.tex"), "w") as f:
                f.write(abl_tex)
            n_tabs += 1
            print("  ablation table saved")
        except Exception as exc:
            print(f"  ablation table failed: {exc}")

    # ------------------------------------------------------------------
    # 3. Figures
    # ------------------------------------------------------------------
    if args.skip_plots:
        print("\n[3] Skipping plots (--skip-plots)")
    else:
        print("\n[3] Generating figures ...")

        # 3a. Accuracy-complexity scatter (per dataset)
        if not main_df.empty and "dataset" in main_df.columns:
            _datasets = main_df["dataset"].dropna().unique()
            _datasets = [d for d in _datasets if d and str(d) != "nan"]
            for ds in _datasets:
                try:
                    viz.plot_accuracy_complexity_scatter(
                        main_df, ds,
                        os.path.join(fig_dir, f"scatter_{ds}.pdf"),
                    )
                    n_figs += 1
                except Exception as exc:
                    print(f"  scatter {ds} failed: {exc}")

        # 3b. Statistical heatmap
        if not pairwise_df.empty:
            try:
                viz.plot_statistical_heatmap(
                    pairwise_df,
                    os.path.join(fig_dir, "pairwise_heatmap.pdf"),
                )
                n_figs += 1
            except Exception as exc:
                print(f"  heatmap failed: {exc}")

        # 3c. Critical difference diagram
        if friedman_result and "avg_ranks" in friedman_result:
            try:
                viz.plot_critical_difference_diagram(
                    friedman_result,
                    os.path.join(fig_dir, "cd_diagram.pdf"),
                )
                n_figs += 1
            except Exception as exc:
                print(f"  CD diagram failed: {exc}")

        # 3d. Ablation bars
        if not ablation_raw.empty and "condition" in ablation_raw.columns:
            try:
                viz.plot_ablation_bars(
                    ablation_raw,
                    os.path.join(fig_dir, "ablation_bars.pdf"),
                )
                n_figs += 1
            except Exception as exc:
                print(f"  ablation bars failed: {exc}")

        # 3e. HV curves
        if not hv_raw.empty:
            try:
                viz.plot_hypervolume_curves(
                    hv_raw,
                    os.path.join(fig_dir, "hv_vs_time.pdf"),
                )
                n_figs += 1
            except Exception as exc:
                print(f"  HV curves failed: {exc}")

        # 3f. Pareto snapshots
        for snap_file in glob.glob(os.path.join(rdir, "pareto_time", "*_snapshots.json")):
            ds_name = Path(snap_file).stem.replace("_snapshots", "")
            try:
                with open(snap_file) as f:
                    snaps = json.load(f)
                viz.plot_pareto_snapshots(
                    snaps, ds_name,
                    os.path.join(fig_dir, f"pareto_panels_{ds_name}.pdf"),
                )
                n_figs += 1
            except Exception as exc:
                print(f"  pareto panels {ds_name} failed: {exc}")

        # 3g. SRBench heatmap
        if not srbench_summary.empty:
            try:
                viz.plot_srbench_heatmap(
                    srbench_summary,
                    os.path.join(fig_dir, "srbench_heatmap.pdf"),
                )
                n_figs += 1
            except Exception as exc:
                print(f"  srbench heatmap failed: {exc}")

        # 3h. Violin-strip (per dataset)
        if not main_df.empty and "method" in main_df.columns:
            _datasets_v = main_df["dataset"].dropna().unique()
            _datasets_v = [d for d in _datasets_v if d and str(d) != "nan"]
            for ds in _datasets_v:
                try:
                    viz.plot_violin_strip(
                        main_df, "r2_test",
                        os.path.join(fig_dir, f"violin_{ds}.pdf"),
                        dataset=ds,
                    )
                    n_figs += 1
                except Exception as exc:
                    print(f"  violin {ds} failed: {exc}")

        # Appendix: bootstrap forest
        if not pairwise_df.empty:
            try:
                viz.plot_bootstrap_forest(
                    pairwise_df,
                    os.path.join(fig_dir, "forest_plot.pdf"),
                )
                n_figs += 1
            except Exception as exc:
                print(f"  forest plot failed: {exc}")

        # Appendix: convergence
        logs_dir = os.path.join(rdir, "ablation")
        if os.path.isdir(logs_dir):
            ds_names = set()
            for f in glob.glob(os.path.join(logs_dir, "*.json")):
                parts = Path(f).stem.split("_")
                if len(parts) >= 2:
                    ds_names.add(parts[0])
            for ds in ds_names:
                try:
                    viz.plot_convergence_curves(
                        logs_dir, ds,
                        os.path.join(fig_dir, f"convergence_{ds}.pdf"),
                    )
                    n_figs += 1
                except Exception as exc:
                    print(f"  convergence {ds} failed: {exc}")

    # ------------------------------------------------------------------
    # 4. Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Paper Figures & Tables Generated")
    print("=" * 60)
    print(f"Figures saved: {fig_dir}/ ({n_figs} PDF files)")
    print(f"Tables saved:  {tab_dir}/ ({n_tabs} files)")

    # Key findings excerpt
    if not pairwise_df.empty:
        kao_vs = pairwise_df[
            pairwise_df["comparison"].str.contains("No_KAO_cap7|No_KAO", na=False)
        ]
        if not kao_vs.empty:
            row = kao_vs.iloc[0]
            p = row.get("p_value", float("nan"))
            d = row.get("cliffs_delta", float("nan"))
            cat = row.get("effect_category", "?")
            print(f"\nKey findings:")
            print(f"  {row['comparison']}: p={p:.4f}, Cliff's d={d:.3f} ({cat})")

    if not hv_csv.empty and "hv_mean" in hv_csv.columns:
        try:
            hv30 = hv_csv[hv_csv["checkpoint"] == 30]
            kao_hv = hv30.loc[hv30["condition"] == "KAO", "hv_mean"].values
            nk_hv = hv30.loc[hv30["condition"] == "No_KAO", "hv_mean"].values
            if len(kao_hv) and len(nk_hv):
                delta = kao_hv[0] - nk_hv[0]
                print(f"  KAO HV advantage at t=30s: {delta:.4f}")
        except Exception:
            pass

    if not srbench_summary.empty and "recovery_rate" in srbench_summary.columns:
        try:
            noise0 = srbench_summary
            if "noise_std" in noise0.columns:
                noise0 = noise0[noise0["noise_std"] == 0.0]
            for m in ["KAO", "PySR"]:
                mr = noise0.loc[noise0["method"] == m, "recovery_rate"]
                if not mr.empty:
                    print(f"  SRBench recovery ({m}): {mr.mean():.0%}")
        except Exception:
            pass

    print()


if __name__ == "__main__":
    main()
