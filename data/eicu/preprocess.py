#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eICU Preprocessing Pipeline
============================
Loads the eICU composite-risk-score CSV (produced by eicu_extract_v2_1.py)
and prepares it for KAO / baseline symbolic regression experiments.

Target variables (by priority):
  1. composite_risk_score  — always available in the CSV
  2. los_hours             — ICU length of stay (if column present, e.g. from
                             a future extraction that retains it)

The reviewer concern is that the composite score is a self-constructed proxy.
When los_hours is available it provides a *real* clinical outcome, eliminating
that bias. The script supports both targets via --target.

Pipeline:
  1. Load CSV
  2. Drop columns with > 50 % missing
  3. Drop rows with any remaining NaN
  4. Optionally standardise features (StandardScaler)
  5. 80/20 train/test split (random_state=42)
  6. Save to .npz  (X_train, X_test, y_train, y_test, feature_names)
  7. Print summary statistics

Usage:
  python data/eicu/preprocess.py                                   # defaults
  python data/eicu/preprocess.py --target composite_risk_score     # explicit
  python data/eicu/preprocess.py --target los_hours --standardize  # LOS target
  python data/eicu/preprocess.py --csv path/to/custom.csv          # custom CSV
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_DEFAULT_CSV = _THIS_DIR / "eICU_composite_risk_score.csv"
_DEFAULT_TARGET = "composite_risk_score"
_DEFAULT_OUT = _THIS_DIR / "eicu_processed.npz"

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="eICU preprocessing for SR experiments")
    p.add_argument("--csv", type=str, default=str(_DEFAULT_CSV),
                   help="Path to raw eICU CSV")
    p.add_argument("--target", type=str, default=_DEFAULT_TARGET,
                   help="Target column name")
    p.add_argument("--out", type=str, default=str(_DEFAULT_OUT),
                   help="Output .npz path")
    p.add_argument("--missing_threshold", type=float, default=0.5,
                   help="Drop columns with missing fraction > this (default 0.5)")
    p.add_argument("--standardize", action="store_true",
                   help="StandardScaler on features (default: no)")
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--also_csv", action="store_true",
                   help="Also save a cleaned CSV alongside the .npz")
    return p.parse_args()

# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def preprocess(
    csv_path: str,
    target: str,
    missing_threshold: float = 0.5,
    standardize: bool = False,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """Load, clean, split, and return arrays + metadata."""

    # 1. Load
    df = pd.read_csv(csv_path)
    print(f"Loaded {csv_path}: {df.shape[0]} rows, {df.shape[1]} columns")

    if target not in df.columns:
        available = list(df.columns)
        raise ValueError(
            f"Target '{target}' not found. Available columns:\n  {available}"
        )

    # 2. Drop columns with too much missing data
    frac_missing = df.isnull().mean()
    drop_cols = frac_missing[frac_missing > missing_threshold].index.tolist()
    if drop_cols:
        print(f"Dropping {len(drop_cols)} columns (>{missing_threshold*100:.0f}% missing): {drop_cols}")
        df = df.drop(columns=drop_cols)

    # Ensure target survived
    if target not in df.columns:
        raise ValueError(f"Target '{target}' was dropped due to missing data.")

    # 3. Drop rows with any NaN
    n_before = len(df)
    df = df.dropna()
    n_after = len(df)
    if n_before != n_after:
        print(f"Dropped {n_before - n_after} rows with NaN ({n_after} remaining)")

    if n_after == 0:
        raise RuntimeError("No rows left after dropping NaN.")

    # 4. Separate features / target
    feature_cols = [c for c in df.columns if c != target]
    X = df[feature_cols].values.astype(np.float64)
    y = df[target].values.astype(np.float64)

    print(f"Features: {len(feature_cols)}  Samples: {len(y)}")
    print(f"Target '{target}': mean={y.mean():.3f}, std={y.std():.3f}, "
          f"range=[{y.min():.3f}, {y.max():.3f}]")

    # 5. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
    )

    # 6. Optional standardisation
    scaler = None
    if standardize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        print("Applied StandardScaler to features")

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": np.array(feature_cols),
        "target_name": target,
        "scaler": scaler,
        "df_clean": df,
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    print("=" * 60)
    print("eICU Preprocessing Pipeline")
    print("=" * 60)

    result = preprocess(
        csv_path=args.csv,
        target=args.target,
        missing_threshold=args.missing_threshold,
        standardize=args.standardize,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    # Save .npz
    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez(
        out_path,
        X_train=result["X_train"],
        X_test=result["X_test"],
        y_train=result["y_train"],
        y_test=result["y_test"],
        feature_names=result["feature_names"],
        target_name=result["target_name"],
    )
    print(f"\nSaved {out_path}")

    # Optionally save cleaned CSV
    if args.also_csv:
        csv_out = str(out_path).replace(".npz", ".csv")
        result["df_clean"].to_csv(csv_out, index=False)
        print(f"Saved {csv_out}")

    # Summary
    print("\n" + "-" * 40)
    print("Summary")
    print("-" * 40)
    print(f"  Source   : {args.csv}")
    print(f"  Target   : {args.target}")
    print(f"  Features : {len(result['feature_names'])}")
    print(f"  Train    : {result['X_train'].shape[0]}")
    print(f"  Test     : {result['X_test'].shape[0]}")
    print(f"  Scaled   : {args.standardize}")
    print(f"  Output   : {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
