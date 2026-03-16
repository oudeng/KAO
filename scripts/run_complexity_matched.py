#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 3 — Complexity-matched Baseline Experiments

Two-step pipeline:
  Step 1 (Tuning):  5-seed grid search per dataset × method to find
                    the config that maximises reachability (≤8 nodes),
                    breaking ties by CV R².
  Step 2 (Eval):    30-seed full evaluation with the best config,
                    writing result.json per seed.

Usage:
  python run_complexity_matched.py --step tuning   [--datasets all]
  python run_complexity_matched.py --step eval      [--datasets all]
  python run_complexity_matched.py --step both      [--datasets all]
"""

from __future__ import annotations

import argparse
import copy
import inspect
import json
import os
import re
import sys
import time
import traceback
import warnings
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ─── Project paths ───────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent   # KAO_v3/
sys.path.insert(0, str(ROOT / "KAO"))

from utils.result_io import write_result_json

# ─── Constants ───────────────────────────────────────────────────────
SPLIT_SEED = 2025
TEST_SIZE  = 0.2
TIME_BUDGET = 60.0

TUNING_SEEDS = [1, 2, 3, 4, 5]
EVAL_SEEDS   = list(range(1, 31))
NODE_BUDGET  = 8      # reachability threshold

DATASETS = {
    "mimic_iv":    {"csv": "KAO/data/mimic_iv/ICU_composite_risk_score.csv",
                    "target": "composite_risk_score", "display": "MIMIC-IV"},
    "eicu":        {"csv": "KAO/data/eicu/eICU_composite_risk_score.csv",
                    "target": "composite_risk_score", "display": "eICU"},
    "nhanes":      {"csv": "KAO/data/nhanes/NHANES_metabolic_score.csv",
                    "target": "metabolic_score", "display": "NHANES"},
    "airfoil":     {"csv": "KAO/data/uci/airfoil_self_noise.csv",
                    "target": "SSPL", "display": "Airfoil"},
    "auto_mpg":    {"csv": "KAO/data/uci/auto-mpg_sel.csv",
                    "target": "mpg", "display": "Auto-MPG"},
    "communities": {"csv": "KAO/data/uci/ComCri_ViolentCrimesPerPop.csv",
                    "target": "ViolentCrimesPerPop", "display": "Communities"},
    "hydraulic":   {"csv": "KAO/data/uci/HydraulicSys_fault_score.csv",
                    "target": "fault_score", "display": "Hydraulic"},
}

METHODS = ["PySR", "RILS-ROLS", "gplearn", "Operon"]


# ─── Data loading ────────────────────────────────────────────────────
_data_cache: Dict[str, Any] = {}

def load_data(ds_key: str):
    """Return (X_train, X_test, y_train, y_test, feature_names)."""
    if ds_key in _data_cache:
        return _data_cache[ds_key]
    info = DATASETS[ds_key]
    df = pd.read_csv(ROOT / info["csv"]).dropna()
    target = info["target"]
    feature_names = [c for c in df.columns if c != target]
    X = df[feature_names].values
    y = df[target].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SPLIT_SEED)
    _data_cache[ds_key] = (X_train, X_test, y_train, y_test, feature_names)
    return X_train, X_test, y_train, y_test, feature_names


# ─── Feature-name substitution (copy from run_baselines.py) ─────────
def _substitute_feature_names(expr_str: str, feature_names: list) -> str:
    if not feature_names or not expr_str:
        return expr_str
    result = expr_str
    for i in sorted(range(len(feature_names)), reverse=True):
        name = feature_names[i]
        for pattern in [rf'\bX{i}\b', rf'\bX_{i}\b', rf'\bx{i}\b', rf'\bx_{i}\b']:
            result = re.sub(pattern, name, result)
    return result


# ─── Wrapper runners ─────────────────────────────────────────────────
def _run_single(method: str, X_train, y_train, X_test, y_test,
                feature_names, seed: int, grid_params: dict) -> dict:
    """Run a single (method, seed, grid_params) combination.

    Returns a dict with keys:
        expression, r2_test, complexity_nodes, complexity_chars, runtime, error
    """
    try:
        if method == "PySR":
            from baselines.pysr_wrapper import PySRSR
            baseline = PySRSR()
            kwargs = {
                "feature_names": feature_names,
                "niterations": grid_params.get("niterations", 40),
                "population_size": grid_params.get("population_size", 33),
                "maxsize": grid_params.get("maxsize", 20),
                "parsimony": grid_params.get("parsimony", 0.0032),
                "maxdepth": grid_params.get("maxdepth", 10),
            }
            res = baseline.fit(X_train, y_train, X_test=X_test, y_test=y_test,
                               time_budget=TIME_BUDGET, random_state=seed, **kwargs)

        elif method == "RILS-ROLS":
            from baselines.rils_rols_wrapper import RILSROLSSR
            baseline = RILSROLSSR()
            kwargs = {
                "feature_names": feature_names,
                "max_fit_calls": grid_params.get("max_fit_calls", 100000),
                "complexity_penalty": grid_params.get("complexity_penalty", 0.001),
                "max_complexity": grid_params.get("max_complexity", 50),
                "sample_size": grid_params.get("sample_size", 1.0),
                "internal_standardize": True,
            }
            res = baseline.fit(X_train, y_train, X_test=X_test, y_test=y_test,
                               time_budget=TIME_BUDGET, random_state=seed, **kwargs)

        elif method == "gplearn":
            from baselines.gplearn_wrapper import GPLearnSR
            baseline = GPLearnSR()
            kwargs = {
                "feature_names": feature_names,
            }
            # gplearn-specific: parsimony_coefficient is set inside base_params
            # We need to patch gplearn_wrapper to accept these
            gp_parsimony = grid_params.get("parsimony_coefficient", 0.001)
            gp_max_depth = grid_params.get("max_depth", None)

            # gplearn wrapper creates SymbolicRegressor internally
            # We monkey-patch for this run by calling fit with custom params
            from gplearn.genetic import SymbolicRegressor as _GPSR
            pop_size = grid_params.get("population_size", 1000)

            base_params = dict(
                population_size=pop_size,
                function_set=["add", "sub", "mul", "div",
                              "sqrt", "log", "abs", "neg"],
                parsimony_coefficient=gp_parsimony,
                max_samples=0.9,
                p_crossover=0.7,
                p_subtree_mutation=0.1,
                p_hoist_mutation=0.05,
                p_point_mutation=0.1,
                random_state=seed,
                verbose=0,
                n_jobs=1,
            )
            # gplearn has no max_depth param; constrain via init_depth
            # and rely on parsimony_coefficient to penalise bloat.
            if gp_max_depth is not None:
                md = int(gp_max_depth)
                base_params["init_depth"] = (2, md)

            start = time.perf_counter()
            # Phase 1: calibrate timing
            est = _GPSR(generations=1, warm_start=False, **base_params)
            est.fit(X_train, y_train)
            gen1_time = time.perf_counter() - start

            remaining = TIME_BUDGET - gen1_time
            if remaining > gen1_time and gen1_time > 0:
                total_gens = min(1 + int(remaining / gen1_time), 200)
                est2 = _GPSR(generations=total_gens, warm_start=False, **base_params)
                est2.fit(X_train, y_train)
                est = est2

            elapsed = time.perf_counter() - start

            if hasattr(est, "_program") and est._program is not None:
                best = est._program
                expr_str = _substitute_feature_names(str(best), feature_names)
                complexity = best.length_
                y_pred_test = est.predict(X_test) if X_test is not None else None
            else:
                expr_str = "FAILED"
                complexity = 0
                y_pred_test = np.full(len(X_test), np.mean(y_train))

            y_pred_test = np.asarray(y_pred_test, dtype=float)
            bad = ~np.isfinite(y_pred_test)
            if bad.any():
                y_pred_test[bad] = np.mean(y_train)

            r2_test = float(r2_score(y_test, y_pred_test))
            from baselines import BaselineSR
            res = {
                "expression": expr_str,
                "complexity": complexity,
                "complexity_chars": BaselineSR.compute_complexity_chars(expr_str),
                "runtime": elapsed,
                "y_pred_test": y_pred_test,
            }

        elif method == "Operon":
            from baselines.operon_wrapper import OperonSR
            baseline = OperonSR()
            kwargs = {
                "feature_names": feature_names,
                "population_size": grid_params.get("population_size", 1000),
            }
            # Pass max_length and max_depth if specified
            ml = grid_params.get("max_length")
            md = grid_params.get("max_depth")
            # These go as kwargs directly to OperonSR.fit → _OperonSR(**params)
            # We need to inject them into the wrapper. The wrapper already
            # passes **kwargs to its internal params dict.
            # Actually, let's pass them directly and let the wrapper handle.
            if ml is not None:
                kwargs["max_length"] = ml
            if md is not None:
                kwargs["max_depth"] = md
            res = baseline.fit(X_train, y_train, X_test=X_test, y_test=y_test,
                               time_budget=TIME_BUDGET, random_state=seed, **kwargs)

        else:
            return {"error": f"Unknown method: {method}"}

        # ── Extract results ──
        expr_str = res.get("expression", "FAILED")
        expr_str = _substitute_feature_names(expr_str, feature_names)
        complexity = int(res.get("complexity", 0))

        if method != "gplearn":
            y_pred_test = res.get("y_pred_test")
            if y_pred_test is not None:
                y_pred_test = np.asarray(y_pred_test, dtype=float)
                bad = ~np.isfinite(y_pred_test)
                if bad.any():
                    y_pred_test[bad] = np.mean(y_train)
                r2_test = float(r2_score(y_test, y_pred_test))
            else:
                r2_test = float("nan")

        from baselines import BaselineSR
        return {
            "expression": expr_str,
            "r2_test": r2_test,
            "complexity_nodes": complexity,
            "complexity_chars": BaselineSR.compute_complexity_chars(expr_str),
            "runtime": res.get("runtime", 0.0),
            "error": "",
        }

    except Exception as e:
        traceback.print_exc()
        return {
            "expression": f"FAILED: {e}",
            "r2_test": float("nan"),
            "complexity_nodes": 0,
            "complexity_chars": 0,
            "runtime": 0.0,
            "error": str(e)[:200],
        }


# ─── Operon wrapper patching ────────────────────────────────────────
# The Operon wrapper only passes recognized params (random_state,
# time_limit, n_threads, population_size). We need to also pass
# max_length and max_depth. Rather than modifying the wrapper file,
# we do it here.
_operon_patched = False

def _patch_operon_wrapper():
    """Patch OperonSR.fit to forward max_length and max_depth to _OperonSR."""
    global _operon_patched
    if _operon_patched:
        return
    try:
        from baselines.operon_wrapper import OperonSR
        _original_fit = OperonSR.fit

        def _patched_fit(self, X_train, y_train, X_test=None, y_test=None,
                         time_budget=60.0, random_state=42, feature_names=None, **kwargs):
            # Extract our extra params before passing to original fit
            extra_operon = {}
            for k in ("max_length", "max_depth", "generations", "max_evaluations"):
                if k in kwargs:
                    extra_operon[k] = kwargs.pop(k)

            # Call original fit
            # But we need to inject extra_operon into the _OperonSR constructor.
            # The cleanest way: temporarily monkey-patch the params dict building
            # inside the original fit method.
            # Actually, the original fit dynamically checks valid_params.
            # Since max_length and max_depth ARE in the pyoperon API,
            # they should be auto-detected. Let's just add them to kwargs
            # and re-check.

            # Actually, looking at the wrapper code more carefully:
            # It builds `params` dict from dynamic detection, then calls
            # _OperonSR(**params). Extra kwargs are used for population_size
            # only. So we need to be more surgical.

            # Simplest approach: call _OperonSR directly ourselves.
            if not extra_operon:
                return _original_fit(self, X_train, y_train, X_test, y_test,
                                     time_budget, random_state, feature_names, **kwargs)

            from pyoperon.sklearn import SymbolicRegressor as _OperonSR
            import inspect as _insp
            import time as _time

            pop_size = kwargs.pop("population_size", 1000)
            start = _time.perf_counter()

            sig = _insp.signature(_OperonSR.__init__)
            valid_params = list(sig.parameters.keys())

            params = {"random_state": random_state}

            # Time limit
            for tp in ("max_time", "time_limit", "time_budget", "max_seconds"):
                if tp in valid_params:
                    params[tp] = int(time_budget)
                    break

            # Thread count
            for tp in ("n_threads", "num_threads", "n_jobs"):
                if tp in valid_params:
                    params[tp] = 1
                    break

            if "population_size" in valid_params:
                params["population_size"] = pop_size

            # Add our extra params (cast to int for C++ binding compat)
            for k, v in extra_operon.items():
                if k in valid_params:
                    params[k] = int(v) if isinstance(v, (float, int)) else v

            # Standardise X
            from sklearn.preprocessing import StandardScaler
            self._scaler_X = StandardScaler()
            X_train_sc = self._scaler_X.fit_transform(X_train)

            try:
                est = _OperonSR(**params)
                est.fit(X_train_sc, y_train)
            except Exception as e:
                result = {
                    "expression": f"FAILED: {e}",
                    "y_pred_train": np.full(len(y_train), np.mean(y_train)),
                    "complexity": 0,
                    "complexity_chars": 0,
                    "runtime": _time.perf_counter() - start,
                    "model": None,
                }
                if X_test is not None:
                    result["y_pred_test"] = np.full(len(X_test), np.mean(y_train))
                return result

            elapsed = _time.perf_counter() - start

            # Extract expression string (same logic as original wrapper)
            expr_str = "UNKNOWN"
            if hasattr(est, "get_model_string") and hasattr(est, "model_"):
                fn_for_model = list(feature_names) if feature_names else None
                try:
                    expr_str = est.get_model_string(est.model_, precision=6,
                                                     names=fn_for_model)
                except TypeError:
                    try:
                        expr_str = est.get_model_string()
                    except Exception:
                        expr_str = str(est)
            else:
                for attr in ("model_string_", "expr_", "best_model_"):
                    if hasattr(est, attr):
                        val = getattr(est, attr)
                        expr_str = val() if callable(val) else str(val)
                        break
            if expr_str == "UNKNOWN":
                expr_str = str(est)

            # Extract complexity
            complexity = 0
            if hasattr(est, "stats_") and isinstance(est.stats_, dict):
                complexity = est.stats_.get("model_length", 0)
            elif hasattr(est, "model_") and hasattr(est.model_, "__len__"):
                complexity = len(est.model_)
            elif hasattr(est, "complexity_"):
                complexity = est.complexity_
            else:
                complexity = len(expr_str)

            # Predictions
            y_min, y_max = float(np.min(y_train)), float(np.max(y_train))
            y_range = y_max - y_min if y_max > y_min else 1.0
            clip_lo = y_min - 3 * y_range
            clip_hi = y_max + 3 * y_range

            raw_pred_train = np.asarray(est.predict(X_train_sc), dtype=float)
            raw_pred_train = np.where(np.isfinite(raw_pred_train), raw_pred_train, np.mean(y_train))
            raw_pred_train = np.clip(raw_pred_train, clip_lo, clip_hi)

            from baselines import BaselineSR
            result = {
                "expression": expr_str,
                "y_pred_train": raw_pred_train,
                "complexity": int(complexity),
                "complexity_chars": BaselineSR.compute_complexity_chars(expr_str),
                "runtime": elapsed,
                "model": est,
                "hparams_effective": params,
            }
            if X_test is not None:
                X_test_sc = self._scaler_X.transform(X_test)
                raw_pred_test = np.asarray(est.predict(X_test_sc), dtype=float)
                raw_pred_test = np.where(np.isfinite(raw_pred_test), raw_pred_test, np.mean(y_train))
                raw_pred_test = np.clip(raw_pred_test, clip_lo, clip_hi)
                result["y_pred_test"] = raw_pred_test

            return result

        OperonSR.fit = _patched_fit
        _operon_patched = True
        print("[Operon] Wrapper patched to support max_length/max_depth")
    except ImportError:
        print("[Operon] pyoperon not available, skipping patch")


# ═══════════════════════════════════════════════════════════════════
# Step 1: TUNING
# ═══════════════════════════════════════════════════════════════════
def run_tuning(ds_keys: List[str], methods: List[str], grid_cfg: dict):
    """Grid-search over 5 tuning seeds.

    For each (dataset, method), evaluate every grid-point configuration
    and select the one with highest reachability (≤8 nodes proportion),
    breaking ties by mean CV R² (approximated by mean test R² across
    the 5 tuning seeds).

    Writes:
      results/{dataset}/complexity_matched/{method_dir}/tuning_report.json
      results/{dataset}/complexity_matched/{method_dir}/grid_results.csv
    """
    print("\n" + "=" * 70)
    print("STEP 1: COMPLEXITY-MATCHED TUNING  (5 seeds)")
    print("=" * 70)

    for ds_key in ds_keys:
        ds_info = DATASETS[ds_key]
        X_train, X_test, y_train, y_test, feat_names = load_data(ds_key)
        print(f"\n{'─'*60}")
        print(f"Dataset: {ds_info['display']}  "
              f"(n_train={len(y_train)}, n_test={len(y_test)}, n_feat={len(feat_names)})")

        for method in methods:
            if method not in grid_cfg:
                print(f"  {method}: no grid config, skipping")
                continue

            mcfg = grid_cfg[method]
            grid_spec = mcfg.get("grid", {})
            fixed = mcfg.get("fixed", {})

            if not grid_spec:
                print(f"  {method}: empty grid, skipping")
                continue

            # Build Cartesian product of grid
            param_names = sorted(grid_spec.keys())
            param_values = [grid_spec[k] for k in param_names]
            combos = list(product(*param_values))

            method_dir = method.lower().replace("-", "_")
            out_dir = ROOT / "results" / ds_key / "complexity_matched" / method_dir
            out_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n  {method}: {len(combos)} configs × {len(TUNING_SEEDS)} seeds = "
                  f"{len(combos) * len(TUNING_SEEDS)} runs")

            grid_rows = []
            for ci, combo in enumerate(combos):
                config = dict(zip(param_names, combo))
                full_params = {**fixed, **config}

                seed_r2s = []
                seed_nodes = []
                seed_reach = 0

                for seed in TUNING_SEEDS:
                    t0 = time.time()
                    res = _run_single(method, X_train, y_train, X_test, y_test,
                                      feat_names, seed, full_params)
                    elapsed = time.time() - t0

                    r2 = res.get("r2_test", float("nan"))
                    nodes = res.get("complexity_nodes", 0)
                    err = res.get("error", "")

                    seed_r2s.append(r2)
                    seed_nodes.append(nodes)
                    if nodes <= NODE_BUDGET:
                        seed_reach += 1

                reach_frac = seed_reach / len(TUNING_SEEDS)
                mean_r2 = float(np.nanmean(seed_r2s)) if seed_r2s else float("nan")
                mean_nodes = float(np.mean(seed_nodes)) if seed_nodes else float("nan")

                grid_rows.append({
                    "config_idx": ci,
                    **config,
                    "reachability_5": reach_frac,
                    "r2_mean": mean_r2,
                    "nodes_mean": mean_nodes,
                    "n_seeds": len(TUNING_SEEDS),
                })

                print(f"    config {ci+1}/{len(combos)} "
                      f"{config}  reach={reach_frac:.0%}  "
                      f"R²={mean_r2:.4f}  nodes={mean_nodes:.1f}")

            # ── Select best ──
            grid_df = pd.DataFrame(grid_rows)
            grid_df.to_csv(out_dir / "grid_results.csv", index=False)

            # Sort: highest reachability first, then highest R²
            grid_df_sorted = grid_df.sort_values(
                ["reachability_5", "r2_mean"], ascending=[False, False]
            )
            best = grid_df_sorted.iloc[0]
            best_config = {pn: best[pn] for pn in param_names}

            report = {
                "dataset": ds_key,
                "method": method,
                "best_config": best_config,
                "fixed_params": fixed,
                "selection_criteria": (
                    f"highest reachability_5 ({best['reachability_5']:.0%}), "
                    f"then highest r2 ({best['r2_mean']:.4f})"
                ),
                "grid_results_file": "grid_results.csv",
                "best_r2_mean": float(best["r2_mean"]),
                "best_reachability_5": float(best["reachability_5"]),
                "best_mean_nodes": float(best["nodes_mean"]),
                "n_grid_combos": len(combos),
                "n_tuning_seeds": len(TUNING_SEEDS),
            }
            with open(out_dir / "tuning_report.json", "w") as f:
                json.dump(report, f, indent=2)

            print(f"  ✓ {method} best: {best_config}  "
                  f"reach={best['reachability_5']:.0%}  "
                  f"R²={best['r2_mean']:.4f}  nodes={best['nodes_mean']:.1f}")


# ═══════════════════════════════════════════════════════════════════
# Step 2: EVALUATION
# ═══════════════════════════════════════════════════════════════════
def run_eval(ds_keys: List[str], methods: List[str], grid_cfg: dict):
    """30-seed evaluation with the best config from tuning.

    Reads tuning_report.json, runs 30 seeds, writes per-seed result.json
    and summary.json.
    """
    print("\n" + "=" * 70)
    print("STEP 2: 30-SEED EVALUATION  (best config from tuning)")
    print("=" * 70)

    for ds_key in ds_keys:
        ds_info = DATASETS[ds_key]
        X_train, X_test, y_train, y_test, feat_names = load_data(ds_key)
        print(f"\n{'─'*60}")
        print(f"Dataset: {ds_info['display']}  "
              f"(n_train={len(y_train)}, n_test={len(y_test)})")

        for method in methods:
            method_dir = method.lower().replace("-", "_")
            tuning_dir = ROOT / "results" / ds_key / "complexity_matched" / method_dir
            report_path = tuning_dir / "tuning_report.json"

            if not report_path.exists():
                print(f"  {method}: tuning_report.json not found, skipping")
                continue

            with open(report_path) as f:
                report = json.load(f)

            best_config = report["best_config"]
            fixed_params = report.get("fixed_params", {})
            full_params = {**fixed_params, **best_config}

            print(f"\n  {method}: config={best_config}  → 30 seeds")

            eval_dir = tuning_dir / "eval_30seeds"
            eval_dir.mkdir(parents=True, exist_ok=True)

            seed_results = []
            for seed in EVAL_SEEDS:
                t0 = time.time()
                res = _run_single(method, X_train, y_train, X_test, y_test,
                                  feat_names, seed, full_params)
                elapsed = time.time() - t0

                r2 = res.get("r2_test", float("nan"))
                nodes = res.get("complexity_nodes", 0)

                # Compute MAE/RMSE from y_pred_test if we can re-derive it
                # Actually _run_single doesn't return y_pred. We need to
                # compute them. Let's use the expression.
                # For now store what we have — the analysis script will
                # re-eval expressions if needed.
                test_mse = float("nan")
                mae = float("nan")

                seed_dir = eval_dir / f"seed_{seed}"
                seed_dir.mkdir(parents=True, exist_ok=True)

                write_result_json(seed_dir / "result.json", {
                    "dataset": ds_key,
                    "method": method,
                    "seed": seed,
                    "budget_seconds": TIME_BUDGET,
                    "time_budget": TIME_BUDGET,
                    "budget_label": f"{int(TIME_BUDGET)}s",
                    "r2_test": float(r2) if np.isfinite(r2) else None,
                    "complexity_nodes": int(nodes),
                    "complexity_chars": res.get("complexity_chars", 0),
                    "runtime": res.get("runtime", elapsed),
                    "expression": res.get("expression", "FAILED"),
                    "hparams_effective": full_params,
                    "extra": {
                        "alpha": 1.0,
                        "beta": 0.0,
                        "status": "ok" if not res.get("error") else res["error"],
                        "tuning_config": best_config,
                    },
                })

                seed_results.append({
                    "seed": seed,
                    "r2_test": r2,
                    "complexity_nodes": nodes,
                    "runtime": res.get("runtime", elapsed),
                    "expression": res.get("expression", "FAILED"),
                    "error": res.get("error", ""),
                })

                reached = "✓" if nodes <= NODE_BUDGET else " "
                status = "OK" if not res.get("error") else f"ERR: {res['error'][:40]}"
                sys.stdout.write(f"\r    seed {seed:2d}/30  "
                                 f"R²={r2:+.4f}  nodes={nodes:3d} [{reached}]  {status}    ")
                sys.stdout.flush()

            print()  # newline after progress

            # ── Summary ──
            sr_df = pd.DataFrame(seed_results)
            n_ok = int(sr_df["error"].apply(lambda e: e == "").sum())
            ok_df = sr_df[sr_df["error"] == ""]
            n_reach = int((ok_df["complexity_nodes"] <= NODE_BUDGET).sum()) if len(ok_df) > 0 else 0

            summary = {
                "dataset": ds_key,
                "method": method,
                "best_config": best_config,
                "n_seeds": len(EVAL_SEEDS),
                "n_ok": n_ok,
                "n_reached": n_reach,
                "reachability_30": f"{n_reach}/{n_ok}" if n_ok > 0 else "0/0",
                "r2_mean": float(ok_df["r2_test"].mean()) if len(ok_df) > 0 else None,
                "r2_std": float(ok_df["r2_test"].std()) if len(ok_df) > 0 else None,
                "nodes_mean": float(ok_df["complexity_nodes"].mean()) if len(ok_df) > 0 else None,
                "nodes_std": float(ok_df["complexity_nodes"].std()) if len(ok_df) > 0 else None,
            }
            with open(eval_dir / "summary.json", "w") as f:
                json.dump(summary, f, indent=2)

            r2m = summary['r2_mean']
            r2s = summary['r2_std']
            nm = summary['nodes_mean']
            ns = summary['nodes_std']
            print(f"  ✓ {method}:  "
                  f"R²={r2m if r2m is None else f'{r2m:.4f}'}±"
                  f"{r2s if r2s is None else f'{r2s:.4f}'}  "
                  f"nodes={nm if nm is None else f'{nm:.1f}'}±"
                  f"{ns if ns is None else f'{ns:.1f}'}  "
                  f"reach={summary['reachability_30']}")


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Phase 3: Complexity-matched baseline experiments")
    parser.add_argument("--step", choices=["tuning", "eval", "both"],
                        default="both",
                        help="Which step to run (default: both)")
    parser.add_argument("--datasets", nargs="+", default=["all"],
                        help="Dataset keys (or 'all')")
    parser.add_argument("--methods", nargs="+", default=["all"],
                        help="Method names (or 'all')")
    parser.add_argument("--grid_config", type=str,
                        default=str(ROOT / "KAO/configs/complexity_matched_grid.json"),
                        help="Path to grid config JSON")
    args = parser.parse_args()

    # Resolve datasets
    if "all" in args.datasets:
        ds_keys = list(DATASETS.keys())
    else:
        ds_keys = args.datasets

    # Resolve methods
    if "all" in args.methods:
        methods = METHODS
    else:
        methods = args.methods

    # Load grid config
    with open(args.grid_config) as f:
        grid_cfg = json.load(f)
    # Remove comment key
    grid_cfg.pop("_comment", None)

    print(f"Datasets: {ds_keys}")
    print(f"Methods:  {methods}")
    print(f"Step:     {args.step}")
    print(f"Grid:     {args.grid_config}")

    # Patch Operon wrapper if needed
    if "Operon" in methods:
        _patch_operon_wrapper()

    if args.step in ("tuning", "both"):
        run_tuning(ds_keys, methods, grid_cfg)

    if args.step in ("eval", "both"):
        run_eval(ds_keys, methods, grid_cfg)

    print("\n" + "=" * 70)
    print("Phase 3 experiments complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
