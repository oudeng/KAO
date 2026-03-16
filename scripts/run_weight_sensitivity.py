#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 4 — Target-weight Sensitivity Analysis

Varies heuristic weights in the composite_risk_score for MIMIC-IV and eICU,
then re-runs KAO (knee) + PySR (tuned) on each variant with 10 seeds.

Usage:
  python run_weight_sensitivity.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")

# ─── Project paths ────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent   # KAO_v3/
sys.path.insert(0, str(ROOT / "KAO"))

from utils.result_io import write_result_json

# ─── Constants ────────────────────────────────────────────────────────
SPLIT_SEED  = 2025
TEST_SIZE   = 0.2
TIME_BUDGET = 60.0
SEEDS       = list(range(1, 11))  # 10 seeds
NODE_BUDGET = 8


# ═══════════════════════════════════════════════════════════════════════
# Weight variant definitions
# ═══════════════════════════════════════════════════════════════════════

# MIMIC-IV composite_risk_score components:
#   MAP < 65:         indicator from map_mmhg
#   Lactate > 2:      indicator from lactate_mmol_l
#   ARDS (PF < 300):  NOT available in CSV — infer as residual
#   GCS < 8:          indicator from gcs
#   AKI (Cr > 1.5):   indicator from creatinine_mg_dl
#   Vasopressor:       vasopressor_use_std

MIMIC_WEIGHTS = {
    "original": {
        # indicator                weight
        "map_below_65":           1,
        "lactate_elevated":       2,
        # ards_mild:              1   (residual, always weight 1)
        "gcs_severe":             2,
        "aki_stage1":             1,
        "vasopressor_use":        2,
    },
    "uniform": {
        "map_below_65":           1,
        "lactate_elevated":       1,
        "gcs_severe":             1,
        "aki_stage1":             1,
        "vasopressor_use":        1,
    },
    "perturbed": {
        "map_below_65":           1,
        "lactate_elevated":       3,
        "gcs_severe":             1,
        "aki_stage1":             1,
        "vasopressor_use":        3,
    },
}

# eICU composite_risk_score components:
#   MAP < 65:                    map_mmhg
#   Lactate > 2:                 lactate_mmol_l
#   Lactate > 4 (severely):      lactate_mmol_l
#   AKI stage 1 (Cr > 1.5):     creatinine_mg_dl
#   AKI stage 2 (Cr > 2.0):     creatinine_mg_dl
#   Hypoxemia (SpO2 < 92):      spo2_min
#   Mech ventilation:            mechanical_ventilation_std
#   Vasopressor:                 vasopressor_use_std
#   GCS < 8:                    gcs
#   Tachycardia (HR > 100):     hr_max
#   Tachypnea (RR > 24):        resprate_max

EICU_WEIGHTS = {
    "original": {
        "map_below_65":                1,
        "lactate_elevated":            2,
        "lactate_severely_elevated":   1,
        "aki_stage1":                  1,
        "aki_stage2":                  1,
        "hypoxemia":                   1,
        "mechanical_ventilation":      2,
        "vasopressor_use":             2,
        "gcs_severe":                  2,
        "tachycardia":                 1,
        "tachypnea":                   1,
    },
    "uniform": {
        "map_below_65":                1,
        "lactate_elevated":            1,
        "lactate_severely_elevated":   1,
        "aki_stage1":                  1,
        "aki_stage2":                  1,
        "hypoxemia":                   1,
        "mechanical_ventilation":      1,
        "vasopressor_use":             1,
        "gcs_severe":                  1,
        "tachycardia":                 1,
        "tachypnea":                   1,
    },
    "perturbed": {
        "map_below_65":                1,
        "lactate_elevated":            3,
        "lactate_severely_elevated":   1,
        "aki_stage1":                  1,
        "aki_stage2":                  1,
        "hypoxemia":                   1,
        "mechanical_ventilation":      3,
        "vasopressor_use":             3,
        "gcs_severe":                  1,
        "tachycardia":                 1,
        "tachypnea":                   1,
    },
}


# ═══════════════════════════════════════════════════════════════════════
# Target recomputation
# ═══════════════════════════════════════════════════════════════════════

def _compute_mimic_target(df: pd.DataFrame, weights: dict) -> pd.Series:
    """Recompute MIMIC-IV composite_risk_score with given weights.

    The ARDS_mild component (PF_ratio < 300) is not directly available in the
    CSV.  We back-infer it from the original score:
        ards_mild = original_score - sum(other indicators * original_weights)
    Since ARDS always carries weight 1 in all variants, we can add it back.
    """
    original_weights = MIMIC_WEIGHTS["original"]

    # Compute binary indicators from raw features
    indicators = {
        "map_below_65":     (df["map_mmhg"] < 65).astype(int),
        "lactate_elevated": (df["lactate_mmol_l"] > 2).astype(int),
        "gcs_severe":       (df["gcs"] < 8).astype(int),
        "aki_stage1":       (df["creatinine_mg_dl"] > 1.5).astype(int),
        "vasopressor_use":  (df["vasopressor_use_std"] == 1).astype(int),
    }

    # Back-infer ARDS component from original score
    original_non_ards = sum(
        indicators[k] * original_weights[k] for k in indicators
    )
    ards_mild = df["composite_risk_score"] - original_non_ards
    # Sanity: ards_mild should be 0 or 1
    ards_mild = ards_mild.clip(0, 1).astype(int)

    # Recompute with new weights
    score = sum(indicators[k] * weights[k] for k in indicators)
    score += ards_mild * 1  # ARDS always weight 1
    return score


def _compute_eicu_target(df: pd.DataFrame, weights: dict) -> pd.Series:
    """Recompute eICU composite_risk_score with given weights."""
    indicators = {
        "map_below_65":              (df["map_mmhg"] < 65).astype(int),
        "lactate_elevated":          (df["lactate_mmol_l"] > 2).astype(int),
        "lactate_severely_elevated": (df["lactate_mmol_l"] > 4).astype(int),
        "aki_stage1":                (df["creatinine_mg_dl"] > 1.5).astype(int),
        "aki_stage2":                (df["creatinine_mg_dl"] > 2.0).astype(int),
        "hypoxemia":                 (df["spo2_min"] < 92).astype(int),
        "mechanical_ventilation":    (df["mechanical_ventilation_std"] == 1).astype(int),
        "vasopressor_use":           (df["vasopressor_use_std"] == 1).astype(int),
        "gcs_severe":                (df["gcs"] < 8).astype(int),
        "tachycardia":               (df["hr_max"] > 100).astype(int),
        "tachypnea":                 (df["resprate_max"] > 24).astype(int),
    }

    score = sum(indicators[k] * weights[k] for k in indicators)
    return score


def load_data_with_variant(ds_key: str, variant: str):
    """Load dataset and recompute target with the specified weight variant."""
    csv_paths = {
        "mimic_iv": ROOT / "KAO/data/mimic_iv/ICU_composite_risk_score.csv",
        "eicu":     ROOT / "KAO/data/eicu/eICU_composite_risk_score.csv",
    }
    df = pd.read_csv(csv_paths[ds_key]).dropna()

    # Recompute target
    if ds_key == "mimic_iv":
        weights = MIMIC_WEIGHTS[variant]
        new_target = _compute_mimic_target(df, weights)
    else:
        weights = EICU_WEIGHTS[variant]
        new_target = _compute_eicu_target(df, weights)

    target_col = "composite_risk_score"
    feature_names = [c for c in df.columns if c != target_col]
    X = df[feature_names].values.astype(float)
    y = new_target.values.astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SPLIT_SEED
    )
    return X_train, X_test, y_train, y_test, feature_names


# ═══════════════════════════════════════════════════════════════════════
# Feature-name substitution
# ═══════════════════════════════════════════════════════════════════════

def _substitute_feature_names(expr_str: str, feature_names: list) -> str:
    if not feature_names or not expr_str:
        return expr_str
    result = expr_str
    for i in sorted(range(len(feature_names)), reverse=True):
        name = feature_names[i]
        for pattern in [rf'\bX{i}\b', rf'\bX_{i}\b', rf'\bx{i}\b', rf'\bx_{i}\b']:
            result = re.sub(pattern, name, result)
    return result


# ═══════════════════════════════════════════════════════════════════════
# Runners
# ═══════════════════════════════════════════════════════════════════════

def run_kao_single(X_train, y_train, X_test, y_test, feature_names, seed):
    """Run KAO with default knee-point selection."""
    from kao.KAO_v3_1 import run_single as kao_run_single

    result = kao_run_single(
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        dataset_name="weight_sensitivity",
        seed=seed,
        time_budget=TIME_BUDGET,
        use_kao_leaf=True,
        feature_names=feature_names,
    )
    return {
        "expression": result.expression,
        "r2_test": result.r2_test,
        "complexity_nodes": result.complexity_nodes,
        "runtime": result.runtime,
    }


# PySR best configs from Phase 3 complexity-matched tuning
PYSR_CONFIGS = {
    "mimic_iv": {"maxsize": 10, "parsimony": 0.1},
    "eicu":     {"maxsize": 8,  "parsimony": 0.1},
}


def run_pysr_single(X_train, y_train, X_test, y_test, feature_names, seed,
                    ds_key: str):
    """Run PySR with the tuned config from Phase 3."""
    from baselines.pysr_wrapper import PySRSR

    cfg = PYSR_CONFIGS[ds_key]
    baseline = PySRSR()
    res = baseline.fit(
        X_train, y_train, X_test=X_test, y_test=y_test,
        time_budget=TIME_BUDGET, random_state=seed,
        feature_names=feature_names,
        niterations=40,
        population_size=33,
        maxsize=cfg["maxsize"],
        parsimony=cfg["parsimony"],
        maxdepth=10,
    )

    expr_str = _substitute_feature_names(
        res.get("expression", "FAILED"), feature_names
    )
    complexity = int(res.get("complexity", 0))

    y_pred = res.get("y_pred_test")
    if y_pred is not None:
        y_pred = np.asarray(y_pred, dtype=float)
        bad = ~np.isfinite(y_pred)
        if bad.any():
            y_pred[bad] = np.mean(y_train)
        r2 = float(r2_score(y_test, y_pred))
    else:
        r2 = float("nan")

    return {
        "expression": expr_str,
        "r2_test": r2,
        "complexity_nodes": complexity,
        "runtime": res.get("runtime", 0.0),
    }


# ═══════════════════════════════════════════════════════════════════════
# Main experiment
# ═══════════════════════════════════════════════════════════════════════

def run_experiment():
    datasets = ["mimic_iv", "eicu"]
    variants = ["original", "uniform", "perturbed"]
    methods  = ["KAO", "PySR"]

    display_ds = {"mimic_iv": "MIMIC-IV", "eicu": "eICU"}
    display_var = {"original": "Original", "uniform": "Uniform", "perturbed": "Perturbed"}

    out_dir = ROOT / "results" / "weight_sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []

    total = len(datasets) * len(variants) * len(methods) * len(SEEDS)
    done = 0

    for ds in datasets:
        for var in variants:
            print(f"\n{'='*60}")
            print(f"  {display_ds[ds]} / {display_var[var]}")
            print(f"{'='*60}")

            X_train, X_test, y_train, y_test, fnames = \
                load_data_with_variant(ds, var)

            # Sanity: print target stats
            print(f"  Target: mean={y_train.mean():.2f}, "
                  f"std={y_train.std():.2f}, "
                  f"range=[{y_train.min():.0f}, {y_train.max():.0f}]")

            for method in methods:
                seed_results = []
                for seed in SEEDS:
                    done += 1
                    print(f"  [{done}/{total}] {method} seed={seed} ... ",
                          end="", flush=True)

                    try:
                        if method == "KAO":
                            res = run_kao_single(
                                X_train, y_train, X_test, y_test, fnames, seed
                            )
                        else:
                            res = run_pysr_single(
                                X_train, y_train, X_test, y_test, fnames, seed,
                                ds_key=ds
                            )
                    except Exception as e:
                        print(f"ERROR: {e}")
                        res = {
                            "expression": "FAILED",
                            "r2_test": float("nan"),
                            "complexity_nodes": 0,
                            "runtime": 0.0,
                        }

                    seed_results.append(res)
                    nodes = res["complexity_nodes"]
                    r2 = res["r2_test"]
                    rt = res["runtime"]
                    print(f"R²={r2:.4f}  nodes={nodes}  "
                          f"time={rt:.1f}s")

                    # Save per-seed result
                    seed_dir = out_dir / ds / var / method.lower() / f"seed_{seed}"
                    seed_dir.mkdir(parents=True, exist_ok=True)
                    result_json = {
                        "dataset": ds,
                        "variant": var,
                        "method": method,
                        "seed": seed,
                        "expression": res["expression"],
                        "r2_test": res["r2_test"],
                        "complexity_nodes": res["complexity_nodes"],
                        "runtime": res["runtime"],
                    }
                    with open(seed_dir / "result.json", "w") as f:
                        json.dump(result_json, f, indent=2)

                # Aggregate
                r2_vals = [r["r2_test"] for r in seed_results
                           if np.isfinite(r["r2_test"])]
                node_vals = [r["complexity_nodes"] for r in seed_results]
                reach = sum(1 for r in seed_results
                            if r["complexity_nodes"] <= NODE_BUDGET
                            and r["expression"] != "FAILED")

                r2_mean = float(np.mean(r2_vals)) if r2_vals else float("nan")
                r2_std  = float(np.std(r2_vals))  if r2_vals else float("nan")
                n_mean  = float(np.mean(node_vals))
                n_std   = float(np.std(node_vals))

                row = {
                    "dataset": ds,
                    "dataset_display": display_ds[ds],
                    "variant": var,
                    "variant_display": display_var[var],
                    "method": method,
                    "r2_mean": r2_mean,
                    "r2_std": r2_std,
                    "nodes_mean": n_mean,
                    "nodes_std": n_std,
                    "reachability": f"{reach}/{len(SEEDS)}",
                    "reach_int": reach,
                }
                all_rows.append(row)

                print(f"  → {method}: R²={r2_mean:.4f}±{r2_std:.4f}  "
                      f"nodes={n_mean:.1f}±{n_std:.1f}  "
                      f"reach={reach}/{len(SEEDS)}")

                # Save summary
                summary = {
                    "dataset": ds, "variant": var, "method": method,
                    "r2_mean": r2_mean, "r2_std": r2_std,
                    "nodes_mean": n_mean, "nodes_std": n_std,
                    "reachability": f"{reach}/{len(SEEDS)}",
                    "n_seeds": len(SEEDS),
                }
                summary_path = out_dir / ds / var / method.lower() / "summary.json"
                with open(summary_path, "w") as f:
                    json.dump(summary, f, indent=2)

    # Save master CSV
    df_out = pd.DataFrame(all_rows)
    csv_path = ROOT / "outputs" / "csv" / "weight_sensitivity.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(csv_path, index=False)
    print(f"\n✓ Master CSV: {csv_path}")

    return df_out


if __name__ == "__main__":
    run_experiment()
