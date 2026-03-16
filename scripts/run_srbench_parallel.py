#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SRBench Parallel Runner — Phase 6B
====================================
Runs new benchmark experiments using multiprocessing.Pool for parallelism.
Each worker process is forced single-threaded via environment variables.

Usage:
  python scripts/run_srbench_parallel.py --workers 25 --benchmarks NEW
  python scripts/run_srbench_parallel.py --workers 25 --benchmarks ALL
  python scripts/run_srbench_parallel.py --workers 4 --benchmarks pagie_1 --seeds 3  # smoke

Environment:
  Requires kao310 conda environment.
  Each worker sets OMP/MKL/OPENBLAS/NUMEXPR/JULIA_NUM_THREADS=1.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
import warnings
from pathlib import Path
from multiprocessing import Pool, current_process

import numpy as np

# ---------------------------------------------------------------------------
# Force single-thread BEFORE any library import
# ---------------------------------------------------------------------------
for var in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "JULIA_NUM_THREADS"]:
    os.environ[var] = "1"

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
_KAO_ROOT = str(Path(__file__).resolve().parent.parent)       # KAO/
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)  # KAO_v3/
if _KAO_ROOT not in sys.path:
    sys.path.insert(0, _KAO_ROOT)

from data.srbench_synthetic.generate import (
    BENCHMARKS, SRBenchmark, generate_dataset, check_symbolic_recovery,
)

# New benchmark keys (Phase 6A selection)
NEW_KEYS = ["vladislavleva_1", "nguyen_9", "nguyen_10", "keijzer_4", "pagie_1"]
EXISTING_KEYS = ["nguyen_1", "nguyen_7", "keijzer_6", "vladislavleva_4", "korns_12"]
ALL_KEYS = EXISTING_KEYS + NEW_KEYS

METHODS = ["KAO", "PySR", "RILS-ROLS", "gplearn", "Operon"]
NOISE_LEVELS = [0.0, 0.1]
DEFAULT_SEEDS = 30
TIME_BUDGET = 60.0

# ---------------------------------------------------------------------------
# Worker initializer — force single-thread in each subprocess
# ---------------------------------------------------------------------------

def _worker_init():
    for var in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "JULIA_NUM_THREADS"]:
        os.environ[var] = "1"


# ---------------------------------------------------------------------------
# Single run function (called by each worker)
# ---------------------------------------------------------------------------

def run_single_experiment(task: dict) -> dict:
    """Execute a single (benchmark, method, noise, seed) experiment.

    Includes a hard timeout of 3× time_budget to catch hung processes.
    """
    import signal

    def _timeout_handler(signum, frame):
        raise TimeoutError("Hard timeout exceeded")

    # Set hard timeout (3× budget to allow generous overhead)
    hard_timeout = int(task["time_budget"] * 3) + 30
    try:
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(hard_timeout)
    except (ValueError, AttributeError):
        pass  # SIGALRM not available on non-Unix or in non-main thread

    # Re-force single-thread (belt-and-suspenders)
    for var in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "JULIA_NUM_THREADS"]:
        os.environ[var] = "1"

    bkey = task["benchmark_key"]
    method = task["method"]
    seed = task["seed"]
    noise_std = task["noise_std"]
    time_budget = task["time_budget"]
    out_dir = task["out_dir"]

    bench = BENCHMARKS[bkey]
    noise_label = f"noise{noise_std}"

    # Check if result already exists (skip if done)
    result_path = os.path.join(out_dir, f"{bkey}_{method}_{noise_label}_seed{seed}.json")
    if os.path.exists(result_path):
        return {"status": "skipped", "benchmark_key": bkey, "method": method,
                "seed": seed, "noise_std": noise_std}

    # Generate data
    X_train, y_train, X_test, y_test = generate_dataset(bench, seed=seed, noise_std=noise_std)
    feature_names = [f"x{i}" for i in range(bench.n_vars)]

    result = {
        "benchmark": bench.name,
        "benchmark_key": bkey,
        "method": method,
        "seed": seed,
        "noise_std": noise_std,
        "r2_test": float("nan"),
        "complexity_nodes": 0,
        "complexity_chars": 0,
        "runtime": 0.0,
        "expression": "FAILED",
        "ground_truth": bench.ground_truth,
        "symbolic_recovery": False,
        "status": "ok",
    }

    try:
        if method == "KAO":
            from kao.KAO_v3_1 import run_single as kao_run_single, KAOResult
            from sklearn.metrics import r2_score

            res = kao_run_single(
                X_train, y_train, X_test, y_test,
                dataset_name=bench.name,
                seed=seed,
                time_budget=time_budget,
                use_kao_leaf=True,
                feature_names=feature_names,
            )
            y_pred = res.y_pred_test
            r2 = r2_score(y_test, y_pred) if y_pred is not None else float("nan")
            result.update({
                "expression": res.expression,
                "r2_test": r2,
                "complexity_nodes": res.complexity_nodes,
                "complexity_chars": res.complexity_chars,
                "runtime": res.runtime,
            })
        else:
            from baselines.registry import get_all_baselines
            from sklearn.metrics import r2_score

            baselines = get_all_baselines(include_optional=True)
            bl = None
            for b in baselines:
                if b.name == method:
                    bl = b
                    break
            if bl is None:
                result["status"] = "method_not_found"
                result["expression"] = f"Method {method} not found"
            else:
                res = bl.fit(
                    X_train, y_train,
                    X_test=X_test, y_test=y_test,
                    time_budget=time_budget,
                    random_state=seed,
                    feature_names=feature_names,
                )
                expr_str = res.get("expression", "FAILED")
                y_pred = res.get("y_pred_test")
                if y_pred is not None:
                    y_pred = np.where(np.isfinite(y_pred), y_pred, 0.0)
                    r2 = r2_score(y_test, y_pred)
                else:
                    r2 = float("nan")

                result.update({
                    "expression": expr_str,
                    "r2_test": r2,
                    "complexity_nodes": int(res.get("complexity", 0)),
                    "complexity_chars": int(res.get("complexity_chars", 0)),
                    "runtime": float(res.get("runtime", 0.0)),
                })

        # Check for divergence
        if np.isnan(result["r2_test"]) or np.isinf(result["r2_test"]):
            result["status"] = "div"
        elif result["r2_test"] < -1e6:
            result["status"] = "div"

    except TimeoutError:
        result["status"] = "timeout"
        result["expression"] = "TIMEOUT: hard timeout exceeded"
        result["runtime"] = 0.0
    except Exception as exc:
        result["status"] = "exception"
        result["expression"] = f"EXCEPTION: {str(exc)[:200]}"
        result["runtime"] = 0.0
    finally:
        # Cancel the alarm
        try:
            signal.alarm(0)
        except Exception:
            pass

    # Symbolic recovery check
    try:
        result["symbolic_recovery"] = check_symbolic_recovery(
            result["expression"], bench.ground_truth
        )
    except Exception:
        result["symbolic_recovery"] = False

    # Write result JSON
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    return {
        "status": result["status"],
        "benchmark_key": bkey,
        "method": method,
        "seed": seed,
        "noise_std": noise_std,
        "r2_test": result["r2_test"],
    }


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------

def print_benchmark_progress(benchmark_key: str, results: list, total_per_bm: int):
    """Print progress summary for a completed benchmark."""
    completed = [r for r in results if r["status"] != "skipped"]
    skipped = [r for r in results if r["status"] == "skipped"]
    ok = [r for r in completed if r["status"] == "ok"]
    div = [r for r in completed if r["status"] == "div"]
    exc = [r for r in completed if r["status"] == "exception"]
    not_found = [r for r in completed if r["status"] == "method_not_found"]

    bench_name = BENCHMARKS[benchmark_key].name
    print(f"\n{'='*70}")
    print(f"[Progress] {bench_name}: {len(completed) + len(skipped)}/{total_per_bm} runs")
    print(f"  OK: {len(ok)}, Diverged: {len(div)}, Exceptions: {len(exc)}, "
          f"Skipped (existing): {len(skipped)}, Method not found: {len(not_found)}")

    if div:
        div_methods = {}
        for r in div:
            m = r["method"]
            div_methods[m] = div_methods.get(m, 0) + 1
        print(f"  Diverged by method: {div_methods}")

    if exc:
        for r in exc[:3]:
            print(f"  Exception: {r['benchmark_key']} / {r['method']} / seed={r['seed']}")

    # R² summary per method (non-div only)
    ok_results = [r for r in results if r["status"] in ("ok", "skipped") and not np.isnan(r.get("r2_test", float("nan")))]
    if ok_results:
        from collections import defaultdict
        by_method = defaultdict(list)
        for r in ok_results:
            by_method[r["method"]].append(r.get("r2_test", 0))
        print(f"  R² mean by method:")
        for m in METHODS:
            vals = by_method.get(m, [])
            if vals:
                print(f"    {m:12s}: {np.mean(vals):.4f} ± {np.std(vals):.4f} ({len(vals)} runs)")
    print(f"{'='*70}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SRBench Parallel Runner (Phase 6B)")
    parser.add_argument("--workers", type=int, default=25,
                        help="Number of parallel worker processes")
    parser.add_argument("--benchmarks", nargs="*", default=["NEW"],
                        help="Benchmark keys or 'NEW' for 5 new, 'ALL' for all 10")
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS,
                        help="Number of seeds (1..N)")
    parser.add_argument("--time_budget", type=float, default=TIME_BUDGET)
    parser.add_argument("--dry_run", action="store_true",
                        help="Print task list without running")
    args = parser.parse_args()

    # Determine benchmark keys
    bm_keys = []
    for bm in args.benchmarks:
        if bm.upper() == "NEW":
            bm_keys.extend(NEW_KEYS)
        elif bm.upper() == "ALL":
            bm_keys.extend(ALL_KEYS)
        elif bm in BENCHMARKS:
            bm_keys.append(bm)
        else:
            print(f"WARNING: unknown benchmark '{bm}', skipping")

    bm_keys = list(dict.fromkeys(bm_keys))  # deduplicate preserving order

    if not bm_keys:
        print("ERROR: no valid benchmarks selected")
        sys.exit(1)

    seeds = list(range(1, args.seeds + 1))
    # Results go to KAO_v3/results/srbench/ (same as existing run_srbench.py)
    out_dir = os.path.join(_PROJECT_ROOT, "results", "srbench")
    os.makedirs(out_dir, exist_ok=True)

    # Build task list
    tasks = []
    for bkey in bm_keys:
        for noise_std in NOISE_LEVELS:
            for method in METHODS:
                for seed in seeds:
                    tasks.append({
                        "benchmark_key": bkey,
                        "method": method,
                        "seed": seed,
                        "noise_std": noise_std,
                        "time_budget": args.time_budget,
                        "out_dir": out_dir,
                    })

    # Check how many already exist
    existing_count = 0
    for t in tasks:
        noise_label = f"noise{t['noise_std']}"
        path = os.path.join(out_dir, f"{t['benchmark_key']}_{t['method']}_{noise_label}_seed{t['seed']}.json")
        if os.path.exists(path):
            existing_count += 1

    total = len(tasks)
    new_runs = total - existing_count

    print("=" * 70)
    print("SRBench Parallel Runner — Phase 6B")
    print("=" * 70)
    print(f"  Benchmarks : {[BENCHMARKS[k].name for k in bm_keys]}")
    print(f"  Methods    : {METHODS}")
    print(f"  Seeds      : {len(seeds)}")
    print(f"  Noise      : {NOISE_LEVELS}")
    print(f"  Budget     : {args.time_budget}s")
    print(f"  Workers    : {args.workers}")
    print(f"  Total tasks: {total}")
    print(f"  Existing   : {existing_count} (will skip)")
    print(f"  New runs   : {new_runs}")
    print()

    if args.dry_run:
        print("[DRY RUN] Would run {new_runs} experiments. Exiting.")
        return

    if new_runs == 0:
        print("All experiments already completed! Nothing to do.")
        # Still print progress summaries
        for bkey in bm_keys:
            bm_tasks = [t for t in tasks if t["benchmark_key"] == bkey]
            # Read existing results
            bm_results = []
            for t in bm_tasks:
                noise_label = f"noise{t['noise_std']}"
                path = os.path.join(out_dir, f"{t['benchmark_key']}_{t['method']}_{noise_label}_seed{t['seed']}.json")
                if os.path.exists(path):
                    with open(path) as f:
                        r = json.load(f)
                    bm_results.append({
                        "status": r.get("status", "ok"),
                        "benchmark_key": bkey,
                        "method": t["method"],
                        "seed": t["seed"],
                        "noise_std": t["noise_std"],
                        "r2_test": r.get("r2_test", float("nan")),
                    })
            print_benchmark_progress(bkey, bm_results, len(bm_tasks))
        return

    # Run experiments with Pool
    start_time = time.time()

    # Group tasks by benchmark for progress reporting
    tasks_by_bm = {}
    for t in tasks:
        bkey = t["benchmark_key"]
        if bkey not in tasks_by_bm:
            tasks_by_bm[bkey] = []
        tasks_by_bm[bkey].append(t)

    # Process benchmark by benchmark for cleaner progress output
    all_results = []
    for bkey in bm_keys:
        bm_tasks = tasks_by_bm[bkey]
        bench_name = BENCHMARKS[bkey].name
        bm_new = sum(1 for t in bm_tasks
                     if not os.path.exists(
                         os.path.join(out_dir,
                                      f"{t['benchmark_key']}_{t['method']}_noise{t['noise_std']}_seed{t['seed']}.json")))

        print(f"\n--- Starting {bench_name}: {len(bm_tasks)} tasks ({bm_new} new) ---")

        # Use apply_async with per-task timeout to avoid hung RILS-ROLS
        hard_timeout = int(args.time_budget * 3) + 60  # generous timeout
        bm_results = []
        with Pool(processes=args.workers, initializer=_worker_init) as pool:
            async_results = []
            for t in bm_tasks:
                ar = pool.apply_async(run_single_experiment, (t,))
                async_results.append((ar, t))

            for ar, t in async_results:
                try:
                    result = ar.get(timeout=hard_timeout)
                    bm_results.append(result)
                except Exception as e:
                    # Timeout or crash — write a timeout result
                    noise_label = f"noise{t['noise_std']}"
                    result_path = os.path.join(
                        out_dir,
                        f"{t['benchmark_key']}_{t['method']}_{noise_label}_seed{t['seed']}.json"
                    )
                    if not os.path.exists(result_path):
                        timeout_result = {
                            "benchmark": BENCHMARKS[t['benchmark_key']].name,
                            "benchmark_key": t['benchmark_key'],
                            "method": t['method'],
                            "seed": t['seed'],
                            "noise_std": t['noise_std'],
                            "r2_test": float('nan'),
                            "complexity_nodes": 0,
                            "complexity_chars": 0,
                            "runtime": 0.0,
                            "expression": f"TIMEOUT: {str(e)[:100]}",
                            "ground_truth": BENCHMARKS[t['benchmark_key']].ground_truth,
                            "symbolic_recovery": False,
                            "status": "timeout",
                        }
                        with open(result_path, "w") as f:
                            json.dump(timeout_result, f, indent=2, default=str)
                    bm_results.append({
                        "status": "timeout",
                        "benchmark_key": t['benchmark_key'],
                        "method": t['method'],
                        "seed": t['seed'],
                        "noise_std": t['noise_std'],
                        "r2_test": float('nan'),
                    })

            pool.terminate()  # Kill any remaining hung workers

        all_results.extend(bm_results)
        print_benchmark_progress(bkey, bm_results, len(bm_tasks))

    elapsed = time.time() - start_time

    # Final summary
    print(f"\n{'='*70}")
    print(f"PHASE 6B COMPLETE")
    print(f"{'='*70}")
    print(f"  Total time: {elapsed/3600:.2f} hours ({elapsed:.0f} seconds)")
    completed = [r for r in all_results if r["status"] != "skipped"]
    skipped = [r for r in all_results if r["status"] == "skipped"]
    ok = [r for r in all_results if r["status"] == "ok"]
    div = [r for r in all_results if r["status"] == "div"]
    exc = [r for r in all_results if r["status"] == "exception"]
    print(f"  Completed: {len(completed)}, Skipped: {len(skipped)}")
    print(f"  OK: {len(ok)}, Diverged: {len(div)}, Exceptions: {len(exc)}")


if __name__ == "__main__":
    main()
