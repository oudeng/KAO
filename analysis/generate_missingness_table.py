#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate missingness summary table for Phase 1 (Experiment C).

Produces:
  - outputs/csv/missingness_summary.csv   (pipeline-level NaN statistics)
  - outputs/tables/Table_missingness.tex  (two-layer LaTeX table for SM)

The LaTeX table contains:
  - Extraction-stage handling (qualitative, hardcoded from script analysis)
  - Pipeline-stage dropna (quantitative, computed from CSV files)
"""

import os
import sys
import pandas as pd
import numpy as np

# ── Project root ──────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Dataset registry (matches CC_Master_Instructions.md) ─────────────────
DATASETS = {
    "MIMIC-IV":     ("KAO/data/mimic_iv/ICU_composite_risk_score.csv",    "composite_risk_score"),
    "eICU":         ("KAO/data/eicu/eICU_composite_risk_score.csv",       "composite_risk_score"),
    "NHANES":       ("KAO/data/nhanes/NHANES_metabolic_score.csv",        "metabolic_score"),
    "Airfoil":      ("KAO/data/uci/airfoil_self_noise.csv",              "SSPL"),
    "Auto-MPG":     ("KAO/data/uci/auto-mpg_sel.csv",                    "mpg"),
    "Communities":   ("KAO/data/uci/ComCri_ViolentCrimesPerPop.csv",     "ViolentCrimesPerPop"),
    "Hydraulic":    ("KAO/data/uci/HydraulicSys_fault_score.csv",        "fault_score"),
}

# Expected final row counts from paper tab:datasets
EXPECTED_FINAL = {
    "MIMIC-IV": 2939,
    "eICU": 4536,
    "NHANES": 2281,
    "Airfoil": 1503,
    "Auto-MPG": 392,
    "Communities": 1994,
    "Hydraulic": 2205,
}


def compute_pipeline_stats():
    """Compute pipeline-level (dropna) statistics for all 7 CSVs."""
    rows = []
    for name, (relpath, target) in DATASETS.items():
        csv_path = os.path.join(ROOT, relpath)
        if not os.path.exists(csv_path):
            print(f"  WARNING: {csv_path} not found — skipping {name}")
            continue

        df = pd.read_csv(csv_path)
        n_csv = len(df)
        n_nan_rows = int(df.isnull().any(axis=1).sum())
        n_final = len(df.dropna())
        pct = f"{n_nan_rows / n_csv * 100:.1f}" if n_csv > 0 else "0.0"

        rows.append({
            "Dataset": name,
            "Rows_CSV": n_csv,
            "Pipeline_dropna": n_nan_rows,
            "Pct_removed": pct,
            "Rows_final": n_final,
        })

        # Detail NaN columns if any
        if n_nan_rows > 0:
            miss_cols = df.columns[df.isnull().any()].tolist()
            miss_rates = {c: f"{df[c].isnull().mean() * 100:.1f}%" for c in miss_cols}
            print(f"  {name} NaN columns: {miss_rates}")

    return pd.DataFrame(rows)


def generate_latex_table(stats_df):
    """Generate the two-layer LaTeX table with hardcoded extraction descriptions."""

    # Build a lookup from stats_df
    lookup = {}
    for _, row in stats_df.iterrows():
        lookup[row["Dataset"]] = row

    # ── Extraction-stage descriptions (from script analysis) ─────────────
    extraction_desc = {
        "MIMIC-IV": (
            r"Zero-fill: vasopressor/ventilation/Charlson; "
            r"drop rows missing any of MAP, lactate, GCS, creatinine"
        ),
        "eICU": (
            r"Zero-fill: treatment flags; drop cols ${>}50\%$ missing; "
            r"drop rows with remaining NaN"
        ),
        "NHANES": (
            r"Drop rows missing 7 core features; "
            r"median impute: BMI, HbA1c; zero-fill: gate cols"
        ),
        "Airfoil": r"None (complete dataset)",
        "Auto-MPG": r"None",
        "Communities": r"Pre-selected subset$^{a}$",
        "Hydraulic": r"None (complete dataset)",
    }

    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(
        r"\caption{Missing data handling summary. "
        r"``Extraction-stage'' describes preprocessing "
        r"applied during dataset construction (before the SR pipeline); "
        r"``Pipeline-stage'' reports the effect of the complete-case filter "
        r"(\texttt{dropna}) applied uniformly across all methods within the SR evaluation pipeline.}"
    )
    lines.append(r"\label{tab:missingness}")
    lines.append(r"\begin{tabular}{lp{5.5cm}rrr}")
    lines.append(r"\toprule")
    lines.append(
        r"Dataset & Extraction-stage handling & Rows (CSV) "
        r"& Pipeline \texttt{dropna} & Rows (final) \\"
    )
    lines.append(r"\midrule")

    # Healthcare datasets
    for name in ["MIMIC-IV", "eICU", "NHANES"]:
        r = lookup[name]
        lines.append(
            f"{name} & {extraction_desc[name]} "
            f"& {int(r['Rows_CSV'])} "
            f"& {int(r['Pipeline_dropna'])} ({r['Pct_removed']}\\%) "
            f"& {int(r['Rows_final'])} \\\\"
        )

    lines.append(r"\midrule")

    # UCI datasets
    for name in ["Airfoil", "Auto-MPG", "Communities", "Hydraulic"]:
        r = lookup[name]
        lines.append(
            f"{name} & {extraction_desc[name]} "
            f"& {int(r['Rows_CSV'])} "
            f"& {int(r['Pipeline_dropna'])} ({r['Pct_removed']}\\%) "
            f"& {int(r['Rows_final'])} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append("")
    lines.append(r"\vspace{0.3em}")
    lines.append(r"\footnotesize")
    lines.append(
        r"$^{a}$The Communities \& Crime dataset was obtained as a pre-processed subset "
        r"(14 features selected from the original 128 predictors); the original UCI dataset "
        r"contains substantial missingness, but the features used here are complete."
    )
    lines.append(r"\medskip")
    lines.append("")
    lines.append(
        r"\noindent Extraction scripts are provided in the code repository "
        r"(\texttt{data/mimic\_iv/mimic\_extract\_v7.py}, "
        r"\texttt{data/eicu/eicu\_extract\_v2\_1.py} + \texttt{preprocess.py}, "
        r"\texttt{data/nhanes/fm\_XPT\_toCSV\_v4\_3.py}). "
        r"For MIMIC-IV and eICU, the original databases contain substantially more "
        r"ICU stays/patients than the final cohorts shown here; extraction-stage "
        r"filtering (eligibility criteria and missingness-driven row removal) accounts "
        r"for the reduction. Authorized users can reproduce the full pipeline using "
        r"the provided scripts and credentialed PhysioNet access."
    )
    lines.append(r"\end{table}")

    return "\n".join(lines)


def main():
    print("=" * 60)
    print("Phase 1: Missingness Summary Table Generator")
    print("=" * 60)
    print(f"Project root: {ROOT}")
    print()

    # ── Ensure output directories exist ──
    os.makedirs(os.path.join(ROOT, "outputs", "csv"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "outputs", "tables"), exist_ok=True)

    # ── 1. Compute pipeline-level stats ──
    print("Computing pipeline-level NaN statistics...")
    stats_df = compute_pipeline_stats()
    print()
    print(stats_df.to_string(index=False))
    print()

    # ── 2. Cross-validate with paper ──
    print("Cross-validating with paper tab:datasets...")
    all_match = True
    for _, row in stats_df.iterrows():
        name = row["Dataset"]
        expected = EXPECTED_FINAL.get(name)
        actual = int(row["Rows_final"])
        match = "OK" if actual == expected else "MISMATCH"
        if actual != expected:
            all_match = False
        print(f"  {name:15s}  expected={expected}  actual={actual}  {match}")
    print()

    if not all_match:
        print("WARNING: Some row counts do not match paper expectations!")
    else:
        print("All row counts match paper tab:datasets.")
    print()

    # ── 3. Save CSV ──
    csv_path = os.path.join(ROOT, "outputs", "csv", "missingness_summary.csv")
    stats_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # ── 4. Generate and save LaTeX ──
    latex = generate_latex_table(stats_df)
    tex_path = os.path.join(ROOT, "outputs", "tables", "Table_missingness.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex + "\n")
    print(f"Saved: {tex_path}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
