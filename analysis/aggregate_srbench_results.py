#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Aggregate SRBench Results — Phase 6B Task 6B-4
================================================
Reads all result.json files from results/srbench/ (all 10 benchmarks),
merges with Phase 6A task descriptors, and outputs:
  - outputs/csv/srbench_10tasks_results.csv   (per-run, ~3000 rows)
  - outputs/csv/srbench_10tasks_summary.csv   (aggregated per benchmark×method×noise)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent.parent  # → KAO_v3
RESULTS_DIR = ROOT / "results" / "srbench"
OUTPUTS_DIR = ROOT / "outputs" / "csv"

BENCHMARKS_ORDER = [
    "nguyen_1", "nguyen_7", "keijzer_6", "vladislavleva_4", "korns_12",
    "vladislavleva_1", "nguyen_9", "nguyen_10", "keijzer_4", "pagie_1",
]

METHODS_ORDER = ["KAO", "PySR", "Operon", "gplearn", "RILS-ROLS"]


def load_all_results() -> pd.DataFrame:
    """Read every result.json from RESULTS_DIR and return a DataFrame."""
    rows = []
    for fname in sorted(RESULTS_DIR.glob("*.json")):
        with open(fname) as f:
            try:
                r = json.load(f)
            except json.JSONDecodeError:
                print(f"  WARNING: corrupt JSON: {fname}")
                continue
        rows.append(r)

    df = pd.DataFrame(rows)
    print(f"  Loaded {len(df)} result files from {RESULTS_DIR}")
    return df


def merge_task_descriptors(df: pd.DataFrame) -> pd.DataFrame:
    """Merge task descriptor info (family_tags, n_vars, kao_has_ops)."""
    desc_path = OUTPUTS_DIR / "srbench_task_descriptors.csv"
    if desc_path.exists():
        desc = pd.read_csv(desc_path, index_col=0)
        # Select only the 10 benchmarks
        desc = desc.loc[desc.index.isin(BENCHMARKS_ORDER)]
        desc = desc[["benchmark", "n_vars", "family_tags", "opset_required", "kao_has_ops"]]
        desc = desc.rename(columns={"benchmark": "benchmark_name_desc"})

        df = df.merge(desc, left_on="benchmark_key", right_index=True, how="left")
        print(f"  Merged task descriptors (n_vars, family_tags, kao_has_ops)")
    else:
        print(f"  WARNING: {desc_path} not found, skipping task descriptor merge")
    return df


def make_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean ± std per (benchmark, method, noise_std)."""
    # Clip extreme R² to avoid inf/nan in summary stats
    df = df.copy()
    # Replace nan and inf with nan for proper aggregation
    df["r2_test"] = pd.to_numeric(df["r2_test"], errors="coerce")

    # Group and aggregate
    agg = (
        df.groupby(["benchmark", "benchmark_key", "method", "noise_std"])
        .agg(
            r2_mean=("r2_test", lambda x: np.nanmean(x)),
            r2_std=("r2_test", lambda x: np.nanstd(x)),
            r2_median=("r2_test", lambda x: np.nanmedian(x)),
            nodes_mean=("complexity_nodes", "mean"),
            nodes_std=("complexity_nodes", "std"),
            runtime_mean=("runtime", "mean"),
            recovery_rate=("symbolic_recovery", "mean"),
            n_seeds=("seed", "count"),
            n_ok=("status", lambda x: (x == "ok").sum()),
            n_div=("status", lambda x: (x == "div").sum()),
            n_timeout=("status", lambda x: (x == "timeout").sum()),
            n_exception=("status", lambda x: (x == "exception").sum()),
        )
        .reset_index()
    )

    # Merge task descriptor info
    desc_path = OUTPUTS_DIR / "srbench_task_descriptors.csv"
    if desc_path.exists():
        desc = pd.read_csv(desc_path, index_col=0)
        desc = desc.loc[desc.index.isin(BENCHMARKS_ORDER)]
        desc = desc[["n_vars", "family_tags", "kao_has_ops"]]
        agg = agg.merge(desc, left_on="benchmark_key", right_index=True, how="left")

    return agg


def main():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SRBench Results Aggregation — Phase 6B")
    print("=" * 70)

    # 1. Load all results
    df = load_all_results()

    # 2. Validate completeness
    expected = 10 * 5 * 2 * 30  # 10 benchmarks × 5 methods × 2 noise × 30 seeds = 3000
    print(f"  Expected: {expected}, Got: {len(df)}")
    if len(df) != expected:
        # Find missing
        for bkey in BENCHMARKS_ORDER:
            count = len(df[df["benchmark_key"] == bkey])
            if count != 300:
                print(f"    ⚠ {bkey}: {count}/300")

    # 2b. Fill missing 'status' field for older results
    if "status" not in df.columns:
        df["status"] = "ok"
    else:
        # Older results without status field → infer from R²
        mask_no_status = df["status"].isna()
        r2_vals = pd.to_numeric(df.loc[mask_no_status, "r2_test"], errors="coerce")
        df.loc[mask_no_status, "status"] = np.where(
            np.isfinite(r2_vals), "ok", "div"
        )
    print(f"  Status counts: {df['status'].value_counts().to_dict()}")

    # 3. Merge task descriptors
    df = merge_task_descriptors(df)

    # 4. Sort for readability
    bkey_order = {k: i for i, k in enumerate(BENCHMARKS_ORDER)}
    meth_order = {k: i for i, k in enumerate(METHODS_ORDER)}
    df["_bkey_sort"] = df["benchmark_key"].map(bkey_order)
    df["_meth_sort"] = df["method"].map(meth_order)
    df = df.sort_values(["_bkey_sort", "noise_std", "_meth_sort", "seed"])
    df = df.drop(columns=["_bkey_sort", "_meth_sort"])

    # 5. Save per-run CSV
    results_path = OUTPUTS_DIR / "srbench_10tasks_results.csv"
    df.to_csv(results_path, index=False)
    print(f"\n  ✓ Per-run results → {results_path} ({len(df)} rows)")

    # 6. Summary
    summary = make_summary(df)
    summary_path = OUTPUTS_DIR / "srbench_10tasks_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"  ✓ Summary → {summary_path} ({len(summary)} rows)")

    # 7. Print summary table
    print(f"\n{'='*100}")
    print("RESULTS SUMMARY (mean R² across 30 seeds)")
    print(f"{'='*100}")

    for noise in [0.0, 0.1]:
        print(f"\n  --- noise_std = {noise} ---")
        sub = summary[summary["noise_std"] == noise]
        header = f"  {'Benchmark':<20s}"
        for m in METHODS_ORDER:
            header += f"  {m:>12s}"
        header += "  n_vars  family_tags"
        print(header)
        print("  " + "-" * len(header))

        for bkey in BENCHMARKS_ORDER:
            row_data = sub[sub["benchmark_key"] == bkey]
            if row_data.empty:
                continue
            bench_name = row_data.iloc[0]["benchmark"]
            line = f"  {bench_name:<20s}"
            for m in METHODS_ORDER:
                mr = row_data[row_data["method"] == m]
                if not mr.empty:
                    r2 = mr.iloc[0]["r2_mean"]
                    n_div = mr.iloc[0]["n_div"]
                    if np.isfinite(r2):
                        if n_div > 0:
                            line += f"  {r2:>10.4f}*{int(n_div)}"
                        else:
                            line += f"  {r2:>12.4f}"
                    else:
                        line += f"  {'div':>12s}"
                else:
                    line += f"  {'N/A':>12s}"

            # Add task descriptor info
            nvars = row_data.iloc[0].get("n_vars", "?")
            tags = row_data.iloc[0].get("family_tags", "?")
            line += f"  {nvars:>5}  {tags}"
            print(line)

    # 8. Failure statistics
    print(f"\n{'='*70}")
    print("FAILURE STATISTICS")
    print(f"{'='*70}")
    failures = summary[(summary["n_div"] > 0) | (summary["n_timeout"] > 0) | (summary["n_exception"] > 0)]
    if failures.empty:
        print("  No failures!")
    else:
        for _, row in failures.iterrows():
            print(f"  {row['benchmark']:20s} | noise={row['noise_std']} | "
                  f"{row['method']:12s} | div={int(row['n_div'])} timeout={int(row['n_timeout'])} "
                  f"exc={int(row['n_exception'])}")

    print(f"\n{'='*70}")
    print("AGGREGATION COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
