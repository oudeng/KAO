#!/usr/bin/env python
# run_v2_4a_baselines_fixed.py - Unified runner for baseline experiments with complete metrics
# Modified to include train_mse and test_mse for PySR and RILS-ROLS
# Created on Oct 23, 2025

import os
import sys
import time
import json
import pickle
import argparse
import warnings
warnings.filterwarnings("ignore")

import re
import random
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Import shared standardized classes for HOF compatibility
from kao.shared_classes import StandardizedIndividual
from utils.result_io import write_result_json
from utils.cv_eval import kfold_eval_fixed_model

def parse_hparams(s: str) -> dict:
    import json, os
    if not s:
        return {}
    # accept @file or plain existing path
    path = None
    if s.startswith("@"):
        path = s[1:]
    elif os.path.exists(s):
        path = s
    if path:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    # otherwise treat as inline JSON
    return json.loads(s)

def parse_seeds_arg(seeds_arg):
    """Parse seeds from CLI: accepts '1,2,3', '1 2 3', '1;2;3', list or int."""
    if isinstance(seeds_arg, (list, tuple)):
        return [int(s) for s in seeds_arg]
    tokens = re.split(r"[,;\s]+", str(seeds_arg).strip(",; \t"))
    seeds = [int(t) for t in tokens if t != ""]
    if not seeds:
        raise ValueError(f"--seeds='{seeds_arg}' parsed to empty/invalid list. Example: --seeds 1,2,3")
    return seeds

def substitute_feature_names(expr_str, feature_names):
    """Replace X0, X1, ... or X_0, X_1, ... with actual feature names."""
    if not feature_names or not expr_str:
        return expr_str
    result = expr_str
    # Sort by index descending to avoid X1 replacing X10's prefix
    for i in sorted(range(len(feature_names)), reverse=True):
        name = feature_names[i]
        for pattern in [rf'\bX{i}\b', rf'\bX_{i}\b', rf'\bx{i}\b', rf'\bx_{i}\b']:
            result = re.sub(pattern, name, result)
    return result

def parse_arguments():
    """Parse command line arguments for baseline experiments"""
    parser = argparse.ArgumentParser(
        description='Baseline Symbolic Regression Methods (PySR/RILS-ROLS/gplearn/Operon)'
    )
    
    parser.add_argument('--csv', type=str, required=True,
                       help='Path to CSV data file')
    parser.add_argument('--target', type=str, required=True,
                       help='Target column name in CSV')
    parser.add_argument('--experiments', type=str, required=True,
                       choices=['pysr', 'rils_rols', 'gplearn', 'operon', 'all'],
                       help='Which experiments to run')
    parser.add_argument('--seeds', type=str, default='1,2,3,5,8',
                       help='Comma-separated list of random seeds')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Test set size ratio (default: 0.2)')
    parser.add_argument('--outdir', type=str, required=True,
                       help='Output directory for results')
    parser.add_argument('--hparams_json', type=str, default='{}',
                       help='JSON string of hyperparameters')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    parser.add_argument('--standardize', action='store_true', default=False,
                       help='Standardize features (default: False)')
    parser.add_argument('--standardize-y', action='store_true', default=False,
                       help='Also standardize target variable Y (default: False)')
    parser.add_argument('--time_budget', type=float, default=None,
                       help='Time budget per seed (seconds); default 60 if unset')
    parser.add_argument('--dataset_name', type=str, default=None,
                       help='Dataset name label (default: stem of CSV)')

    return parser.parse_args()

def load_and_prepare_data(csv_path, target_col, test_size=0.2, random_state=2025):
    """Load CSV and prepare train/test split"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data file not found: {csv_path}")
    
    data = pd.read_csv(csv_path)
    
    if target_col not in data.columns:
        raise ValueError(f"Target column '{target_col}' not found in data")
    
    # Clean data
    data = data.dropna()
    
    # Separate features and target
    feature_cols = [col for col in data.columns if col != target_col]
    X = data[feature_cols].values
    Y = data[target_col].values
    
    # Train/test split
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, Y_train, Y_test, feature_cols

def evaluate_expression_pysr(expr_str, X_train, Y_train, X_test, Y_test, feature_names):
    """Evaluate PySR expression and compute train/test MSE"""
    from baselines.pysr_wrapper import evaluate_pysr_expr
    
    try:
        # Compute predictions
        yhat_train = evaluate_pysr_expr(expr_str, X_train, feature_names)
        yhat_test = evaluate_pysr_expr(expr_str, X_test, feature_names)
        
        # Clip and sanitize
        yhat_train = np.clip(yhat_train, -1e15, 1e15)
        yhat_train = np.where(np.isfinite(yhat_train), yhat_train, 0.0)
        yhat_test = np.clip(yhat_test, -1e15, 1e15)
        yhat_test = np.where(np.isfinite(yhat_test), yhat_test, 0.0)
        
        # Compute MSE
        train_mse = mean_squared_error(Y_train, yhat_train)
        test_mse = mean_squared_error(Y_test, yhat_test)
        
        return float(train_mse), float(test_mse)
    except Exception as e:
        if args.verbose:
            print(f"    Error evaluating PySR expression: {e}")
        return np.nan, np.nan

def evaluate_expression_rils_rols(expr_str, X_train, Y_train, X_test, Y_test):
    """Evaluate RILS-ROLS expression and compute train/test MSE"""
    try:
        # RILS-ROLS expressions are Python-evaluable
        # Create namespace with numpy functions
        namespace = {
            'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
            'exp': np.exp, 'log': lambda x: np.log(np.abs(x) + 1e-10),
            'sqrt': np.sqrt, 'abs': np.abs, 'sign': np.sign,
            'square': lambda x: x**2
        }
        
        def eval_expr(X, expr):
            """Evaluate expression on data"""
            n_samples = X.shape[0]
            results = []
            
            for i in range(n_samples):
                # Create local namespace with variable values
                local_ns = namespace.copy()
                for j in range(X.shape[1]):
                    local_ns[f'x{j}'] = X[i, j]
                
                # Parse and replace variable names in expression
                expr_parsed = str(expr)
                # RILS-ROLS uses variable names from features
                # We need to handle this more carefully
                
                try:
                    result = eval(expr_parsed, {"__builtins__": {}}, local_ns)
                    results.append(float(result))
                except:
                    results.append(0.0)
            
            return np.array(results)
        
        # For RILS-ROLS, we need a different approach since it outputs fitted expressions
        # The expression already includes coefficients
        # We just evaluate it as-is
        
        # This is a simplified evaluation - RILS-ROLS expressions are complex
        # For now, return the test_loss that was already computed
        return np.nan, np.nan
        
    except Exception as e:
        if args.verbose:
            print(f"    Error evaluating RILS-ROLS expression: {e}")
        return np.nan, np.nan

def run_pysr_experiment(args, X_train, X_test, Y_train, Y_test, feature_names):
    """Run PySR experiments with complete metrics"""
    from baselines.pysr_wrapper import run_pysr_unified, extract_best_from_pysr
    from utils.hparams import merge_hparams

    seeds = parse_seeds_arg(args.seeds)

    # ---- v3.2.1: unified hparams merge (defaults < json < CLI) ----
    defaults = {
        "niterations": 40,
        "population_size": 33,
        "maxsize": 20,
        "maxdepth": 10,
        "parsimony": 0.0032,
        "topk": 12,
        "time_budget": 60.0,
    }
    json_overrides = parse_hparams(args.hparams_json) if args.hparams_json else {}
    cli_overrides = {"time_budget": args.time_budget}
    effective_hparams = merge_hparams(defaults, json_overrides, cli_overrides)

    time_budget = effective_hparams["time_budget"]
    dataset_name = args.dataset_name if hasattr(args, 'dataset_name') and args.dataset_name else Path(args.csv).stem
    budget_label = f"{int(time_budget)}s"

    # Working copy of hparams for PySR (remove time_budget — passed separately)
    default_hparams = {k: v for k, v in effective_hparams.items() if k != "time_budget"}

    # v3.2: PySR to its own subdirectory
    output_dir = Path(args.outdir) / "pysr"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config = {
        'csv': args.csv,
        'target': args.target,
        'seeds': seeds,
        'test_size': args.test_size,
        'hyperparameters': default_hparams,
        'method': 'pysr',
        'standardize': args.standardize,
        'standardize_y': args.standardize_y
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    results = []
    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"Running PySR with seed {seed}")
        print(f"{'='*60}")
        
        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        
        t_start = time.time()
        
        # Run PySR with unified interface (v3.2: pass time_budget)
        df_results = run_pysr_unified(
            X_train, Y_train,
            feature_names=feature_names,
            random_state=seed,
            X_test=X_test,
            y_test=Y_test,
            verbose=args.verbose,
            time_budget=time_budget,
            **default_hparams
        )
        
        runtime = time.time() - t_start
        
        # Extract HOF from results
        hof = extract_best_from_pysr(df_results, feature_names)
        
        # Save HOF
        with open(seed_dir / "hof.pkl", "wb") as f:
            pickle.dump(hof, f)
        
        # Save feature names
        with open(seed_dir / "feature_names.json", "w") as f:
            json.dump(list(feature_names), f)
        
        # Save detailed results
        df_results.to_csv(seed_dir / "pysr_results.csv", index=False)
        
        if len(df_results) > 0:
            best_row = df_results.iloc[0]  # Best by CV loss
            
            # Save best individual
            best_ind = hof[0] if hof else None
            with open(seed_dir / "best_individual.pkl", "wb") as f:
                pickle.dump(best_ind, f)
            
            # Compute train_mse and test_mse
            train_mse, test_mse = evaluate_expression_pysr(
                best_row['expr'], X_train, Y_train, X_test, Y_test, feature_names
            )

            # Map generic variable names to real feature names
            pysr_expr = substitute_feature_names(best_row['expr'], list(feature_names))

            # Unified results format
            from kao.KAO_v3_1 import compute_complexity_chars
            results.append({
                'seed': seed,
                'method': 'pysr',
                'expr': pysr_expr,
                'train_mse': train_mse,
                'test_mse': test_mse,
                'cv_loss': best_row['cv_loss'],
                'cv_r2': best_row['cv_r2'],
                'test_loss': best_row['test_loss'] if 'test_loss' in best_row else test_mse,
                'test_r2': best_row['test_r2'] if 'test_r2' in best_row else np.nan,
                'complexity': best_row['complexity'],
                'instability': best_row.get('instability', 0.0),
                'runtime_sec': best_row.get('runtime_sec', runtime),
                'hof_size': len(hof),
                'alpha': 1.0,
                'beta': 0.0
            })

            # v3.3: write unified result.json
            _r2_test = best_row['test_r2'] if 'test_r2' in best_row else np.nan
            write_result_json(seed_dir / "result.json", {
                "dataset": dataset_name,
                "method": "PySR",
                "seed": seed,
                "budget_seconds": time_budget,
                "time_budget": time_budget,
                "budget_label": budget_label,
                "r2_test": float(_r2_test) if np.isfinite(_r2_test) else None,
                "r2_cv": float(best_row['cv_r2']),
                "test_mse": float(test_mse) if np.isfinite(test_mse) else None,
                "train_mse": float(train_mse) if np.isfinite(train_mse) else None,
                "cv_loss": float(best_row['cv_loss']),
                "instability": float(best_row.get('instability', 0.0)),
                "complexity_nodes": int(best_row['complexity']),
                "complexity_chars": compute_complexity_chars(pysr_expr),
                "runtime": best_row.get('runtime_sec', runtime),
                "expression": pysr_expr,
                "hparams_effective": effective_hparams,
                "extra": {"status": "ok"}
            })
    
    # Save summary with unified column names
    summary_df = pd.DataFrame(results)
    summary_df.to_csv(output_dir / "summary_results.csv", index=False)
    
    print("\n" + "="*60)
    print("PySR Experiment Summary:")
    print("="*60)
    if len(summary_df) > 0:
        print(f"Average Test R2: {summary_df['test_r2'].mean():.4f} ± {summary_df['test_r2'].std():.4f}")
        print(f"Average Test Loss: {summary_df['test_loss'].mean():.4f} ± {summary_df['test_loss'].std():.4f}")
        print(f"Average Train MSE: {summary_df['train_mse'].dropna().mean():.4f}")
        print(f"Average Test MSE: {summary_df['test_mse'].dropna().mean():.4f}")
        print(f"Average CV Loss: {summary_df['cv_loss'].mean():.4f}")
        print(f"Average Complexity: {summary_df['complexity'].mean():.1f} ± {summary_df['complexity'].std():.1f}")
    
    return results

def run_rils_rols_experiment(args, X_train, X_test, Y_train, Y_test, feature_names):
    """Run RILS-ROLS experiments with complete metrics"""
    from baselines.rils_rols_wrapper import run_rils_rols_unified, RILSROLSConfig
    from utils.hparams import merge_hparams

    seeds = parse_seeds_arg(args.seeds)

    # ---- v3.2.1: unified hparams merge (defaults < json < CLI) ----
    # Note: max_time defaults to time_budget (=60.0 unless overridden)
    defaults = {
        "max_fit_calls": 100000,
        "max_time": 60,           # will be overridden by time_budget below
        "complexity_penalty": 0.001,
        "max_complexity": 50,
        "sample_size": 1.0,
        "internal_standardize": not args.standardize,
        "time_budget": 60.0,
    }
    json_overrides = parse_hparams(args.hparams_json) if args.hparams_json else {}
    cli_overrides = {"time_budget": args.time_budget}
    effective_hparams = merge_hparams(defaults, json_overrides, cli_overrides)

    # v3.2.1: max_time defaults to time_budget // 5 (internal 5-fold CV)
    if "max_time" not in json_overrides:
        effective_hparams["max_time"] = int(effective_hparams["time_budget"]) // 5

    time_budget = effective_hparams["time_budget"]
    dataset_name = args.dataset_name if hasattr(args, 'dataset_name') and args.dataset_name else Path(args.csv).stem
    budget_label = f"{int(time_budget)}s"

    # Working copy for RILS-ROLS (remove time_budget — not a RILS param)
    default_hparams = {k: v for k, v in effective_hparams.items() if k != "time_budget"}

    # v3.2: RILS-ROLS to its own subdirectory
    output_dir = Path(args.outdir) / "rils_rols"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config = {
        'csv': args.csv,
        'target': args.target,
        'seeds': seeds,
        'test_size': args.test_size,
        'hyperparameters': default_hparams,
        'method': 'rils_rols',
        'cli_standardize': args.standardize,
        'standardize_y': args.standardize_y
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    results = []
    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"Running RILS-ROLS with seed {seed}")
        if not default_hparams.get('internal_standardize', True):
            print("Note: Internal standardization disabled (using CLI-standardized data)")
        print(f"{'='*60}")
        
        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        
        t_start = time.time()
        
        # Run RILS-ROLS
        cfg = RILSROLSConfig(
            seed=seed,
            verbose=args.verbose,
            **default_hparams
        )
        
        # Debug: Check if import will work
        if cfg.verbose:
            try:
                from rils_rols.rils_rols import RILSROLSRegressor
                print(f"    [Seed {seed}] RILSROLSRegressor imported successfully")
            except ImportError as e:
                print(f"    [Seed {seed}] Failed to import RILSROLSRegressor: {e}")
        
        result = run_rils_rols_unified(
            X_train, Y_train, X_test, Y_test,
            seed=seed,
            cfg=cfg,
            feature_names=feature_names
        )
        
        runtime = time.time() - t_start
        
        # Debug: Print result to see what's happening
        if cfg.verbose:
            print(f"    [Seed {seed}] Result: expr_str={result.get('expr_str')}, error={result.get('error')}")
        
        # Check if we have a valid result
        if result and result.get('expr_str'):
            # Create HOF with single best expression
            best_expr = substitute_feature_names(result['expr_str'], list(feature_names))
            fitness_values = (result.get('test_loss', np.nan), result.get('complexity', 1))
            hof = [StandardizedIndividual(best_expr, fitness_values)]
            
            # Save HOF
            with open(seed_dir / "hof.pkl", "wb") as f:
                pickle.dump(hof, f)
            
            # Save best individual
            with open(seed_dir / "best_individual.pkl", "wb") as f:
                pickle.dump(hof[0], f)
            
            # Save feature names
            with open(seed_dir / "feature_names.json", "w") as f:
                json.dump(list(feature_names), f)
            
            # For RILS-ROLS, train_mse is approximated from CV loss
            # and test_mse is the same as test_loss
            train_mse_approx = result.get('cv_loss', np.nan)
            test_mse_val = result.get('test_loss', np.nan)
            
            # Unified results format
            from kao.KAO_v3_1 import compute_complexity_chars
            results.append({
                'seed': seed,
                'method': 'rils_rols',
                'expr': best_expr,
                'train_mse': train_mse_approx,
                'test_mse': test_mse_val,
                'cv_loss': result.get('cv_loss', np.nan),
                'cv_r2': result.get('cv_r2', np.nan),
                'test_loss': result.get('test_loss', np.nan),
                'test_r2': result.get('test_r2', np.nan),
                'complexity': result.get('complexity', np.nan),
                'instability': result.get('instability', 0.0),
                'runtime_sec': runtime,
                'hof_size': 1,
                'alpha': result.get('alpha', 1.0),
                'beta': result.get('beta', 0.0)
            })

            # v3.3: write unified result.json
            _r2_test_rils = result.get('test_r2', np.nan)
            write_result_json(seed_dir / "result.json", {
                "dataset": dataset_name,
                "method": "RILS-ROLS",
                "seed": seed,
                "budget_seconds": time_budget,
                "time_budget": time_budget,
                "budget_label": budget_label,
                "r2_test": float(_r2_test_rils) if np.isfinite(_r2_test_rils) else None,
                "r2_cv": float(result.get('cv_r2', np.nan)) if np.isfinite(result.get('cv_r2', np.nan)) else None,
                "test_mse": float(test_mse_val) if np.isfinite(test_mse_val) else None,
                "train_mse": float(train_mse_approx) if np.isfinite(train_mse_approx) else None,
                "cv_loss": float(result.get('cv_loss', np.nan)) if np.isfinite(result.get('cv_loss', np.nan)) else None,
                "instability": float(result.get('instability', 0.0)),
                "complexity_nodes": int(result.get('complexity', 0)) if np.isfinite(result.get('complexity', 0)) else 0,
                "complexity_chars": compute_complexity_chars(best_expr),
                "runtime": runtime,
                "expression": best_expr,
                "hparams_effective": effective_hparams,
                "extra": {"status": "ok"}
            })
        else:
            # Handle error case
            print(f"    [Seed {seed}] Warning: RILS-ROLS failed to produce result")
            results.append({
                'seed': seed,
                'method': 'rils_rols',
                'expr': f"ERROR: {result.get('error', 'Unknown error')}",
                'train_mse': np.nan,
                'test_mse': np.nan,
                'cv_loss': np.nan,
                'cv_r2': np.nan,
                'test_loss': np.nan,
                'test_r2': np.nan,
                'complexity': np.nan,
                'instability': np.nan,
                'runtime_sec': runtime,
                'hof_size': 0,
                'alpha': np.nan,
                'beta': np.nan
            })
    
    # Save summary
    summary_df = pd.DataFrame(results)
    summary_df.to_csv(output_dir / "summary_results.csv", index=False)
    
    print("\n" + "="*60)
    print("RILS-ROLS Experiment Summary:")
    print("="*60)
    
    # Filter valid results
    valid_df = summary_df[~summary_df['expr'].str.startswith('ERROR:', na=False)]
    
    if len(valid_df) > 0:
        print(f"Successful runs: {len(valid_df)}/{len(summary_df)}")
        print(f"Average Test R2: {valid_df['test_r2'].mean():.4f} ± {valid_df['test_r2'].std():.4f}")
        print(f"Average Test Loss: {valid_df['test_loss'].mean():.4f} ± {valid_df['test_loss'].std():.4f}")
        print(f"Average Train MSE: {valid_df['train_mse'].dropna().mean():.4f}")
        print(f"Average Test MSE: {valid_df['test_mse'].dropna().mean():.4f}")
        print(f"Average CV Loss: {valid_df['cv_loss'].mean():.4f}")
        print(f"Average Complexity: {valid_df['complexity'].mean():.1f} ± {valid_df['complexity'].std():.1f}")
    else:
        print("All runs failed!")
    
    return results

def run_gplearn_experiment(args, X_train, X_test, Y_train, Y_test, feature_names):
    """Run gplearn experiments via unified BaselineSR interface"""
    from baselines.gplearn_wrapper import GPLearnSR
    from utils.hparams import merge_hparams

    seeds = parse_seeds_arg(args.seeds)

    # ---- v3.2.1: unified hparams merge ----
    defaults = {"time_budget": 60.0}
    json_overrides = parse_hparams(args.hparams_json) if args.hparams_json else {}
    cli_overrides = {"time_budget": args.time_budget}
    effective_hparams = merge_hparams(defaults, json_overrides, cli_overrides)

    time_budget = effective_hparams["time_budget"]
    dataset_name = args.dataset_name if hasattr(args, 'dataset_name') and args.dataset_name else Path(args.csv).stem
    budget_label = f"{int(time_budget)}s"

    # Create output directory
    output_dir = Path(args.outdir) / "gplearn"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    config = {
        'csv': args.csv,
        'target': args.target,
        'seeds': seeds,
        'test_size': args.test_size,
        'method': 'gplearn',
        'standardize': args.standardize,
        'standardize_y': args.standardize_y,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    baseline = GPLearnSR()
    results = []
    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"Running gplearn with seed {seed}")
        print(f"{'='*60}")

        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        t_start = time.time()

        # v3.3: pass method-specific kwargs so hparams_effective matches execution
        _blocked = {"time_budget", "dataset_name", "method", "seed",
                     "random_state", "feature_names", "outdir"}
        method_kwargs = {k: v for k, v in effective_hparams.items()
                         if k not in _blocked}
        try:
            res = baseline.fit(
                X_train, Y_train,
                X_test=X_test, y_test=Y_test,
                time_budget=time_budget,
                random_state=seed,
                feature_names=feature_names,
                **method_kwargs,
            )
        except Exception as e:
            print(f"    [Seed {seed}] gplearn failed: {e}")
            results.append({
                'seed': seed, 'method': 'gplearn',
                'expr': f"ERROR: {e}",
                'train_mse': np.nan, 'test_mse': np.nan,
                'cv_loss': np.nan, 'cv_r2': np.nan,
                'test_loss': np.nan, 'test_r2': np.nan,
                'complexity': np.nan, 'instability': np.nan,
                'runtime_sec': time.time() - t_start,
                'hof_size': 0, 'alpha': np.nan, 'beta': np.nan,
            })
            continue

        runtime = time.time() - t_start
        expr_str = substitute_feature_names(res.get('expression', ''), list(feature_names))

        # Compute metrics (sanitise NaN/Inf in predictions)
        y_pred_train = res.get('y_pred_train', None)
        y_pred_test = res.get('y_pred_test', None)

        if y_pred_train is not None:
            y_pred_train = np.asarray(y_pred_train, dtype=float)
            bad = ~np.isfinite(y_pred_train)
            if bad.any():
                y_pred_train[bad] = np.mean(Y_train)
            train_mse = float(mean_squared_error(Y_train, y_pred_train))
        else:
            train_mse = np.nan

        if y_pred_test is not None:
            y_pred_test = np.asarray(y_pred_test, dtype=float)
            bad = ~np.isfinite(y_pred_test)
            if bad.any():
                y_pred_test[bad] = np.mean(Y_train)
            test_mse = float(mean_squared_error(Y_test, y_pred_test))
        else:
            test_mse = np.nan

        # R² from MSE
        ss_tot_test = np.sum((Y_test - np.mean(Y_test)) ** 2)
        test_r2 = 1.0 - (test_mse * len(Y_test)) / ss_tot_test if ss_tot_test > 0 else np.nan

        # Save HOF
        fitness_values = (test_mse, res.get('complexity', 0))
        hof = [StandardizedIndividual(expr_str, fitness_values)]
        with open(seed_dir / "hof.pkl", "wb") as f:
            pickle.dump(hof, f)
        with open(seed_dir / "best_individual.pkl", "wb") as f:
            pickle.dump(hof[0], f)
        with open(seed_dir / "feature_names.json", "w") as f:
            json.dump(list(feature_names), f)

        # v3.2: fold-based CV evaluation for gplearn
        cv_loss_gp, cv_r2_gp, instability_gp = np.nan, np.nan, 0.0
        try:
            if res.get('model') is not None:
                predict_fn = lambda X_: res['model'].predict(X_)
                cv_loss_gp, cv_r2_gp, instability_gp = kfold_eval_fixed_model(
                    predict_fn, X_train, Y_train
                )
        except Exception:
            pass

        from kao.KAO_v3_1 import compute_complexity_chars
        results.append({
            'seed': seed,
            'method': 'gplearn',
            'expr': expr_str,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'cv_loss': cv_loss_gp if np.isfinite(cv_loss_gp) else train_mse,
            'cv_r2': cv_r2_gp if np.isfinite(cv_r2_gp) else np.nan,
            'test_loss': test_mse,
            'test_r2': test_r2,
            'complexity': res.get('complexity', np.nan),
            'instability': instability_gp,
            'runtime_sec': res.get('runtime', runtime),
            'hof_size': 1,
            'alpha': 1.0,
            'beta': 0.0,
        })

        # v3.3: write unified result.json
        write_result_json(seed_dir / "result.json", {
            "dataset": dataset_name,
            "method": "gplearn",
            "seed": seed,
            "budget_seconds": time_budget,
            "time_budget": time_budget,
            "budget_label": budget_label,
            "r2_test": float(test_r2) if np.isfinite(test_r2) else None,
            "r2_cv": float(cv_r2_gp) if np.isfinite(cv_r2_gp) else None,
            "test_mse": float(test_mse) if np.isfinite(test_mse) else None,
            "train_mse": float(train_mse) if np.isfinite(train_mse) else None,
            "cv_loss": float(cv_loss_gp) if np.isfinite(cv_loss_gp) else None,
            "instability": float(instability_gp),
            "complexity_nodes": int(res.get('complexity', 0)),
            "complexity_chars": compute_complexity_chars(expr_str),
            "runtime": res.get('runtime', runtime),
            "expression": expr_str,
            "hparams_effective": effective_hparams,
            "extra": {"status": "ok"}
        })

    # Save summary
    summary_df = pd.DataFrame(results)
    summary_df.to_csv(output_dir / "summary_results.csv", index=False)

    print("\n" + "="*60)
    print("gplearn Experiment Summary:")
    print("="*60)
    valid_df = summary_df[~summary_df['expr'].str.startswith('ERROR:', na=False)]
    if len(valid_df) > 0:
        print(f"Successful runs: {len(valid_df)}/{len(summary_df)}")
        print(f"Average Test R2: {valid_df['test_r2'].mean():.4f} ± {valid_df['test_r2'].std():.4f}")
        print(f"Average Test MSE: {valid_df['test_mse'].dropna().mean():.4f}")
        print(f"Average Train MSE: {valid_df['train_mse'].dropna().mean():.4f}")
        print(f"Average Complexity: {valid_df['complexity'].mean():.1f} ± {valid_df['complexity'].std():.1f}")
    else:
        print("All runs failed!")

    return results


def run_operon_experiment(args, X_train, X_test, Y_train, Y_test, feature_names):
    """Run Operon experiments via unified BaselineSR interface"""
    try:
        from baselines.operon_wrapper import OperonSR, OPERON_AVAILABLE
    except ImportError:
        print("WARNING: baselines.operon_wrapper could not be imported. Skipping Operon.")
        return []

    if not OPERON_AVAILABLE:
        print("WARNING: pyoperon is not installed. Skipping Operon experiment.")
        print("  Install with: pip install pyoperon")
        return []

    seeds = parse_seeds_arg(args.seeds)

    # ---- v3.2.1: unified hparams merge ----
    from utils.hparams import merge_hparams
    defaults = {"time_budget": 60.0}
    json_overrides = parse_hparams(args.hparams_json) if args.hparams_json else {}
    cli_overrides = {"time_budget": args.time_budget}
    effective_hparams = merge_hparams(defaults, json_overrides, cli_overrides)

    time_budget = effective_hparams["time_budget"]
    dataset_name = args.dataset_name if hasattr(args, 'dataset_name') and args.dataset_name else Path(args.csv).stem
    budget_label = f"{int(time_budget)}s"

    # Create output directory
    output_dir = Path(args.outdir) / "operon"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    config = {
        'csv': args.csv,
        'target': args.target,
        'seeds': seeds,
        'test_size': args.test_size,
        'method': 'operon',
        'standardize': args.standardize,
        'standardize_y': args.standardize_y,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    baseline = OperonSR()
    results = []
    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"Running Operon with seed {seed}")
        print(f"{'='*60}")

        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        t_start = time.time()

        # v3.3: pass method-specific kwargs so hparams_effective matches execution
        _blocked = {"time_budget", "dataset_name", "method", "seed",
                     "random_state", "feature_names", "outdir"}
        method_kwargs = {k: v for k, v in effective_hparams.items()
                         if k not in _blocked}
        try:
            res = baseline.fit(
                X_train, Y_train,
                X_test=X_test, y_test=Y_test,
                time_budget=time_budget,
                random_state=seed,
                feature_names=feature_names,
                **method_kwargs,
            )
        except Exception as e:
            print(f"    [Seed {seed}] Operon failed: {e}")
            results.append({
                'seed': seed, 'method': 'operon',
                'expr': f"ERROR: {e}",
                'train_mse': np.nan, 'test_mse': np.nan,
                'cv_loss': np.nan, 'cv_r2': np.nan,
                'test_loss': np.nan, 'test_r2': np.nan,
                'complexity': np.nan, 'instability': np.nan,
                'runtime_sec': time.time() - t_start,
                'hof_size': 0, 'alpha': np.nan, 'beta': np.nan,
            })
            continue

        runtime = time.time() - t_start
        expr_str = substitute_feature_names(res.get('expression', ''), list(feature_names))

        # Compute metrics (sanitise NaN/Inf in predictions)
        y_pred_train = res.get('y_pred_train', None)
        y_pred_test = res.get('y_pred_test', None)

        if y_pred_train is not None:
            y_pred_train = np.asarray(y_pred_train, dtype=float)
            bad = ~np.isfinite(y_pred_train)
            if bad.any():
                y_pred_train[bad] = np.mean(Y_train)
            train_mse = float(mean_squared_error(Y_train, y_pred_train))
        else:
            train_mse = np.nan

        if y_pred_test is not None:
            y_pred_test = np.asarray(y_pred_test, dtype=float)
            bad = ~np.isfinite(y_pred_test)
            if bad.any():
                y_pred_test[bad] = np.mean(Y_train)
            test_mse = float(mean_squared_error(Y_test, y_pred_test))
        else:
            test_mse = np.nan

        # R² from MSE
        ss_tot_test = np.sum((Y_test - np.mean(Y_test)) ** 2)
        test_r2 = 1.0 - (test_mse * len(Y_test)) / ss_tot_test if ss_tot_test > 0 else np.nan

        # Save HOF
        fitness_values = (test_mse, res.get('complexity', 0))
        hof = [StandardizedIndividual(expr_str, fitness_values)]
        with open(seed_dir / "hof.pkl", "wb") as f:
            pickle.dump(hof, f)
        with open(seed_dir / "best_individual.pkl", "wb") as f:
            pickle.dump(hof[0], f)
        with open(seed_dir / "feature_names.json", "w") as f:
            json.dump(list(feature_names), f)

        # v3.2: fold-based CV evaluation for Operon
        cv_loss_op, cv_r2_op, instability_op = np.nan, np.nan, 0.0
        try:
            if res.get('model') is not None:
                _operon_model = res['model']
                _operon_scaler = getattr(baseline, '_scaler_X', None)
                if _operon_scaler is not None:
                    predict_fn = lambda X_: _operon_model.predict(_operon_scaler.transform(X_))
                else:
                    predict_fn = lambda X_: _operon_model.predict(X_)
                cv_loss_op, cv_r2_op, instability_op = kfold_eval_fixed_model(
                    predict_fn, X_train, Y_train
                )
        except Exception:
            pass

        from kao.KAO_v3_1 import compute_complexity_chars
        results.append({
            'seed': seed,
            'method': 'operon',
            'expr': expr_str,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'cv_loss': cv_loss_op if np.isfinite(cv_loss_op) else train_mse,
            'cv_r2': cv_r2_op if np.isfinite(cv_r2_op) else np.nan,
            'test_loss': test_mse,
            'test_r2': test_r2,
            'complexity': res.get('complexity', np.nan),
            'instability': instability_op,
            'runtime_sec': res.get('runtime', runtime),
            'hof_size': 1,
            'alpha': 1.0,
            'beta': 0.0,
        })

        # v3.3: write unified result.json
        # Merge wrapper-level actual params into hparams_effective
        _operon_hparams = {**effective_hparams, **res.get("hparams_effective", {})}
        write_result_json(seed_dir / "result.json", {
            "dataset": dataset_name,
            "method": "Operon",
            "seed": seed,
            "budget_seconds": time_budget,
            "time_budget": time_budget,
            "budget_label": budget_label,
            "r2_test": float(test_r2) if np.isfinite(test_r2) else None,
            "r2_cv": float(cv_r2_op) if np.isfinite(cv_r2_op) else None,
            "test_mse": float(test_mse) if np.isfinite(test_mse) else None,
            "train_mse": float(train_mse) if np.isfinite(train_mse) else None,
            "cv_loss": float(cv_loss_op) if np.isfinite(cv_loss_op) else None,
            "instability": float(instability_op),
            "complexity_nodes": int(res.get('complexity', 0)),
            "complexity_chars": compute_complexity_chars(expr_str),
            "runtime": res.get('runtime', runtime),
            "expression": expr_str,
            "hparams_effective": _operon_hparams,
            "extra": {"status": "ok"}
        })

    # Save summary
    summary_df = pd.DataFrame(results)
    summary_df.to_csv(output_dir / "summary_results.csv", index=False)

    print("\n" + "="*60)
    print("Operon Experiment Summary:")
    print("="*60)
    valid_df = summary_df[~summary_df['expr'].str.startswith('ERROR:', na=False)]
    if len(valid_df) > 0:
        print(f"Successful runs: {len(valid_df)}/{len(summary_df)}")
        print(f"Average Test R2: {valid_df['test_r2'].mean():.4f} ± {valid_df['test_r2'].std():.4f}")
        print(f"Average Test MSE: {valid_df['test_mse'].dropna().mean():.4f}")
        print(f"Average Train MSE: {valid_df['train_mse'].dropna().mean():.4f}")
        print(f"Average Complexity: {valid_df['complexity'].mean():.1f} ± {valid_df['complexity'].std():.1f}")
    else:
        print("All runs failed!")

    return results


def main():
    """Main entry point for baseline experiments"""
    global args
    args = parse_arguments()
    
    print(f"{'='*60}")
    print("Baseline Symbolic Regression Experiments")
    print(f"{'='*60}")
    print(f"Method: {args.experiments}")
    print(f"Data: {args.csv}")
    print(f"Target: {args.target}")
    print(f"Seeds: {args.seeds}")
    print(f"Output: {args.outdir}")
    
    # Load and prepare data
    X_train, X_test, Y_train, Y_test, feature_names = load_and_prepare_data(
        args.csv, args.target, args.test_size, random_state=2025
    )
    
    print(f"\nData shape:")
    print(f"  Training: {X_train.shape[0]} samples")
    print(f"  Testing: {X_test.shape[0]} samples")
    print(f"  Features: {len(feature_names)}")
    
    # Standardize if requested
    if args.standardize:
        scaler_X = StandardScaler()
        X_train = scaler_X.fit_transform(X_train)
        X_test = scaler_X.transform(X_test)
        print("  Features standardized")
    
    if args.standardize_y:
        scaler_Y = StandardScaler()
        Y_train = scaler_Y.fit_transform(Y_train.reshape(-1, 1)).ravel()
        Y_test = scaler_Y.transform(Y_test.reshape(-1, 1)).ravel()
        print("  Target standardized")
    
    # Run experiments
    all_results = []
    
    if args.experiments in ['pysr', 'all']:
        results = run_pysr_experiment(args, X_train, X_test, Y_train, Y_test, feature_names)
        all_results.extend(results)
    
    if args.experiments in ['rils_rols', 'all']:
        results = run_rils_rols_experiment(args, X_train, X_test, Y_train, Y_test, feature_names)
        all_results.extend(results)

    if args.experiments in ['gplearn', 'all']:
        results = run_gplearn_experiment(args, X_train, X_test, Y_train, Y_test, feature_names)
        all_results.extend(results)

    if args.experiments in ['operon', 'all']:
        results = run_operon_experiment(args, X_train, X_test, Y_train, Y_test, feature_names)
        all_results.extend(results)

    print(f"\n{'='*60}")
    print("Baseline experiments completed!")
    print(f"Results saved to: {args.outdir}")
    
    return all_results

if __name__ == "__main__":
    main()