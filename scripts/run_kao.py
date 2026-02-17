#!/usr/bin/env python
# run_v2_4c_kao_fixed.py - KAO-only symbolic regression experiments with complete metrics
# Modified to include cv_loss, cv_r2, test_loss, instability, runtime_sec
# Created on Oct 23, 2025
# Rev1: add def parse_hparams with the json files in overrides sub-fold for eqtime tests on Nov 2, 2025

import os
import sys
import json
import pickle
import argparse
import random
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
import time
import re

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Import shared standardized classes for HOF compatibility
from kao.shared_classes import StandardizedIndividual, StandardizedFitness, create_standardized_individual
from utils.result_io import write_result_json

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

def parse_arguments():
    """Parse command line arguments for KAO experiments"""
    parser = argparse.ArgumentParser(description='KAO Symbolic Regression Experiments')
    
    parser.add_argument('--csv', type=str, required=True,
                       help='Path to CSV data file')
    parser.add_argument('--target', type=str, required=True,
                       help='Target column name in CSV')
    parser.add_argument('--seeds', type=str, default='1,2,3,5,8',
                       help='Comma-separated list of random seeds')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Test set size ratio')
    parser.add_argument('--outdir', type=str, required=True,
                       help='Output directory for results')
    parser.add_argument('--hparams_json', type=str, default='{}',
                       help='JSON string of KAO hyperparameters')
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
    """Load and prepare data with train/test split"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data file not found: {csv_path}")
    
    data = pd.read_csv(csv_path)
    
    if target_col not in data.columns:
        raise ValueError(f"Target column '{target_col}' not found in data")
    
    # Clean data - remove rows with NaN
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

def compute_cv_metrics(ind, X_train, Y_train, pset, n_splits=5, random_state=0):
    """Compute cross-validation metrics for an individual"""
    from deap import gp
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_squared_error, r2_score
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv_losses = []
    cv_r2s = []
    
    for train_idx, val_idx in kf.split(X_train):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        Y_tr, Y_val = Y_train[train_idx], Y_train[val_idx]
        
        try:
            # Compile individual
            func = gp.compile(ind, pset)
            
            # Evaluate on train fold to get alpha/beta
            yhat_tr = np.array([func(*X_tr[i]) for i in range(len(X_tr))])
            
            # Fit linear scaling
            coef = np.polyfit(yhat_tr, Y_tr, 1)
            alpha, beta = coef[0], coef[1]
            
            # Evaluate on validation fold
            yhat_val = np.array([func(*X_val[i]) for i in range(len(X_val))])
            yhat_val_scaled = alpha * yhat_val + beta
            
            # Clip predictions
            yhat_val_scaled = np.clip(yhat_val_scaled, -1e15, 1e15)
            yhat_val_scaled = np.where(np.isfinite(yhat_val_scaled), yhat_val_scaled, 0.0)
            
            cv_loss = mean_squared_error(Y_val, yhat_val_scaled)
            cv_r2 = r2_score(Y_val, yhat_val_scaled)
            
            cv_losses.append(cv_loss)
            cv_r2s.append(cv_r2)
            
        except Exception:
            cv_losses.append(1e9)
            cv_r2s.append(-1.0)
    
    # Calculate metrics
    mean_cv_loss = float(np.mean(cv_losses))
    mean_cv_r2 = float(np.mean(cv_r2s))
    instability = float(np.std(cv_losses) / max(np.mean(cv_losses), 1e-12))
    
    return mean_cv_loss, mean_cv_r2, instability

def run_kao_experiment(args, X_train, X_test, Y_train, Y_test, feature_names):
    """Run KAO experiments with optimized settings and complete metrics"""
    import json, random, pickle
    import numpy as np
    import pandas as pd
    from pathlib import Path
    from kao.KAO_v3_1 import KAO_Core  # Use the optimized v3c version
    from kneed import KneeLocator
    from utils.hparams import merge_hparams

    seeds = parse_seeds_arg(args.seeds)

    # ---- v3.2.1: unified hparams merge (defaults < json < CLI) ----
    # Step 1: code defaults
    default_hparams_base = {
        # Evolution parameters
        "mu": 2000,
        "lambda": 2000,
        "cxpb": 0.6,
        "mutpb": 0.4,
        "ngen": 80,
        "max_depth": 11,

        # Pareto pruning
        "eps_mse": 1e-3,
        "eps_comp": 1,

        # Linear scaling
        "enable_linear_scaling": True,

        # Local refit
        "enable_local_refit": True,
        "local_refit_topK": 25,

        # Knee detection window
        "knee_window": 3,

        # KAO seeding
        "kao_inject_ratio": 0.15,

        # CV settings
        "cv_folds": 5,
        "cv_shuffle": True,
        "cv_seed": 2025,  # Default CV seed

        # Time budget default
        "time_budget": 60.0,
    }

    # Step 2: json overrides (parsed once)
    json_overrides = parse_hparams(args.hparams_json) if args.hparams_json else {}

    # Step 3: CLI overrides (None = user did not specify)
    cli_overrides = {
        "time_budget": args.time_budget,  # None when unset
    }

    # Step 4: merge → single source of truth
    effective_hparams = merge_hparams(default_hparams_base, json_overrides, cli_overrides)

    # Convenience aliases (always from effective_hparams)
    effective_time_budget = effective_hparams["time_budget"]
    dataset_name = args.dataset_name if (hasattr(args, 'dataset_name') and args.dataset_name) else Path(args.csv).stem
    budget_label = f"{int(effective_time_budget)}s"

    # v3.2: use outdir directly (run_all already appends /kao)
    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration (uses effective values)
    config = {
        'csv': args.csv,
        'target': args.target,
        'seeds': seeds,
        'test_size': args.test_size,
        'hyperparameters': effective_hparams,
        'method': 'kao',
        'standardize': args.standardize,
        'standardize_y': args.standardize_y,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)
    
    results = []
    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"Running KAO with seed {seed}")
        print(f"{'='*60}")
        
        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        
        # Set random seeds
        random.seed(seed)
        np.random.seed(seed)
        
        t_start = time.time()
        
        # v3.3: seed-specific hparams (single source of truth for this seed)
        seed_hparams = effective_hparams.copy()
        seed_hparams['seed'] = seed
        seed_hparams['cv_seed'] = seed

        # Initialize KAO with required positional arguments and hyperparameters
        kao = KAO_Core(feature_names, X_train, Y_train, **seed_hparams)
        
        # Call fit for optional internal standardization
        kao.fit()
        
        # Run evolution to get results
        pop, logbook, hof, pset = kao.run_evolution()
        
        runtime = time.time() - t_start
        
        # Save HOF
        with open(seed_dir / "hof.pkl", "wb") as f:
            pickle.dump(hof, f)
        
        # Save feature names
        with open(seed_dir / "feature_names.json", "w") as f:
            json.dump(list(feature_names), f)
        
        # Select best individuals using KAO's built-in methods
        hof_df = kao.build_hof_dataframe(hof)
        pruned_df = kao.epsilon_prune(hof_df)
        
        # Create Pareto front DataFrame for saving
        pareto_data = []
        for idx, ind in enumerate(hof):
            pareto_data.append({
                'idx': idx,
                'mse': ind.fitness.values[0],
                'complexity': ind.fitness.values[1],
                'expression': str(ind)
            })
        
        pareto_df = pd.DataFrame(pareto_data)
        pareto_df.to_csv(seed_dir / "pareto_front.csv", index=False)
        
        # Knee detection using pruned DataFrame
        best_knee_ind = None
        knee_row = None
        
        if len(pruned_df) > 3:
            try:
                kl = KneeLocator(
                    pruned_df.Complexity.tolist(),
                    pruned_df.mse.tolist(),
                    curve="convex", 
                    direction="decreasing"
                )
                knee = kl.knee
                
                if knee is None:
                    knee = float(pruned_df["Complexity"].median())
                    
                complexities = pruned_df["Complexity"].values
                knee_val = int(complexities[np.argmin(np.abs(complexities - knee))])
                
                window = int(seed_hparams.get('knee_window', 3))
                candidates = pruned_df[pruned_df.Complexity.between(knee_val - window, knee_val + window)].reset_index(drop=True)
                
                # knee-window best
                if len(candidates) == 0:
                    knee_row = pruned_df.iloc[pruned_df['mse'].argmin()]
                else:
                    knee_row = candidates.iloc[candidates['mse'].argmin()]
                    
                best_knee_ind = hof[int(knee_row.idx)] if len(hof) else None
                
                if args.verbose:
                    print(f"  Knee found at complexity {knee_val}, MSE {knee_row.mse:.6f}")
                    
            except Exception as e:
                if args.verbose:
                    print(f"  Knee detection failed: {e}")
        
        # Also get best by MSE from pruned frontier
        if len(pruned_df) > 0:
            mse_row = pruned_df.iloc[pruned_df['mse'].argmin()]
            best_mse_ind = hof[int(mse_row.idx)] if len(hof) else None
        else:
            best_mse_ind = hof[0] if len(hof) > 0 else None
        
        # Fallback if no knee found
        if best_knee_ind is None and len(hof) > 0:
            best_knee_ind = hof[0]
            if knee_row is None:
                knee_row = pruned_df.iloc[0] if len(pruned_df) > 0 else pareto_df.iloc[0]
        
        # Helper functions for evaluation
        from deap import gp
        
        def fit_alpha_beta(ind, X, Y, pset):
            """Fit linear scaling parameters"""
            try:
                func = gp.compile(ind, pset)
                yhat = np.array([func(*X[i]) for i in range(len(X))])
                coef = np.polyfit(yhat, Y, 1)
                return float(coef[0]), float(coef[1])
            except:
                return 1.0, 0.0
        
        def eval_on_test(ind, X_tr, Y_tr, X_te, Y_te, pset, alpha=None, beta=None):
            """Evaluate individual on test set with linear scaling"""
            try:
                func = gp.compile(ind, pset)
                
                if alpha is None or beta is None:
                    yhat_tr = np.array([func(*X_tr[i]) for i in range(len(X_tr))])
                    coef = np.polyfit(yhat_tr, Y_tr, 1)
                    alpha, beta = coef[0], coef[1]
                
                yhat_te = np.array([func(*X_te[i]) for i in range(len(X_te))])
                yhat_te = alpha * yhat_te + beta
                
                # Clip predictions
                yhat_te = np.clip(yhat_te, -1e15, 1e15)
                yhat_te = np.where(np.isfinite(yhat_te), yhat_te, 0.0)
                
                from sklearn.metrics import mean_squared_error, r2_score
                mse_te = mean_squared_error(Y_te, yhat_te)
                r2_te = r2_score(Y_te, yhat_te)
                
                # Also compute on training set
                yhat_tr = np.array([func(*X_tr[i]) for i in range(len(X_tr))])
                yhat_tr = alpha * yhat_tr + beta
                yhat_tr = np.clip(yhat_tr, -1e15, 1e15)
                yhat_tr = np.where(np.isfinite(yhat_tr), yhat_tr, 0.0)
                mse_tr = mean_squared_error(Y_tr, yhat_tr)
                    
            except Exception:
                return np.nan, -1e6, float(alpha), float(beta), np.nan
            
            return float(mse_te), float(r2_te), float(alpha), float(beta), float(mse_tr)

        # Evaluate best individuals
        for label, ind in [("knee", best_knee_ind), ("mse", best_mse_ind)]:
            if ind is None:
                continue
                
            alpha, beta = fit_alpha_beta(ind, X_train, Y_train, pset)
            mse_te, r2_te, _, _, mse_tr = eval_on_test(ind, X_train, Y_train, X_test, Y_test, pset, alpha, beta)
            
            # Compute CV metrics
            cv_loss, cv_r2, instability = compute_cv_metrics(ind, X_train, Y_train, pset, n_splits=5, random_state=seed)
            
            # Save individual with metrics
            save_data = {
                "individual": ind,
                "alpha": float(alpha),
                "beta": float(beta),
                "test_mse": float(mse_te),
                "test_r2": float(r2_te),
                "train_mse": float(mse_tr),
                "cv_loss": float(cv_loss),
                "cv_r2": float(cv_r2),
                "instability": float(instability),
                "complexity": len(ind),
                "expression": str(ind),
                "runtime_sec": runtime
            }
            
            with open(seed_dir / f"best_{label}_individual.pkl", "wb") as f:
                pickle.dump(save_data, f)
            
            # Also save metrics as JSON
            metrics = {k: v for k, v in save_data.items() if k != "individual"}
            with open(seed_dir / f"best_{label}_metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)
            
            if label == "knee":
                # Handle potential inf values
                test_loss_val = float(mse_te)
                if not np.isfinite(test_loss_val):
                    test_loss_val = np.nan
                
                # Use the MSE from the individual's fitness if knee_row not available
                train_mse_for_results = float(mse_tr) if np.isfinite(mse_tr) else np.nan
                if knee_row is not None and hasattr(knee_row, 'mse'):
                    individual_mse = float(knee_row.mse)
                else:
                    individual_mse = ind.fitness.values[0] if ind else np.nan
                
                results.append({
                    'seed': seed,
                    'method': 'kao',
                    'expr': str(ind),
                    'train_mse': train_mse_for_results,
                    'test_mse': float(mse_te) if np.isfinite(mse_te) else np.nan,
                    'cv_loss': float(cv_loss),
                    'cv_r2': float(cv_r2),
                    'test_loss': test_loss_val,
                    'test_r2': float(r2_te),
                    'complexity': len(ind),
                    'instability': float(instability),
                    'runtime_sec': runtime,
                    'hof_size': len(hof),
                    'alpha': float(alpha),
                    'beta': float(beta)
                })

                # v3.3: write unified result.json (seed_hparams as hparams_effective)
                from kao.KAO_v3_1 import compute_complexity_chars
                write_result_json(seed_dir / "result.json", {
                    "dataset": dataset_name,
                    "method": "KAO",
                    "seed": seed,
                    "budget_seconds": effective_time_budget,
                    "time_budget": effective_time_budget,
                    "budget_label": budget_label,
                    "r2_test": float(r2_te) if np.isfinite(r2_te) else None,
                    "r2_cv": float(cv_r2),
                    "test_mse": float(mse_te) if np.isfinite(mse_te) else None,
                    "train_mse": train_mse_for_results,
                    "cv_loss": float(cv_loss),
                    "instability": float(instability),
                    "complexity_nodes": int(ind.fitness.values[1]) if hasattr(ind, 'fitness') and ind.fitness.valid else len(ind),
                    "complexity_chars": compute_complexity_chars(str(ind)),
                    "runtime": runtime,
                    "expression": str(ind),
                    "hparams_effective": seed_hparams,
                    "extra": {
                        "status": "ok",
                        "alpha": float(alpha),
                        "beta": float(beta),
                        "use_kao_leaf": seed_hparams.get("use_kao_leaf", True),
                    }
                })
    
    # Save summary with unified column names
    summary_df = pd.DataFrame(results)
    summary_df.to_csv(output_dir / "summary_results.csv", index=False)
    
    print("\n" + "="*60)
    print("KAO Experiment Summary:")
    print("="*60)
    if len(summary_df) > 0:
        print(f"Average Test R2: {summary_df['test_r2'].mean():.4f} ± {summary_df['test_r2'].std():.4f}")
        print(f"Average Test MSE: {summary_df['test_loss'].mean():.4f} ± {summary_df['test_loss'].std():.4f}")
        print(f"Average CV Loss: {summary_df['cv_loss'].mean():.4f}")
        print(f"Average CV R2: {summary_df['cv_r2'].mean():.4f}")
        print(f"Average Complexity: {summary_df['complexity'].mean():.1f} ± {summary_df['complexity'].std():.1f}")
        print(f"Average Runtime: {summary_df['runtime_sec'].mean():.1f}s")
    
    return results

def main():
    """Main entry point for KAO experiments"""
    args = parse_arguments()
    
    print(f"{'='*60}")
    print("KAO Symbolic Regression Experiments")
    print(f"{'='*60}")
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
    
    # Run KAO experiment
    results = run_kao_experiment(args, X_train, X_test, Y_train, Y_test, feature_names)
    
    print(f"\n{'='*60}")
    print("KAO experiments completed successfully!")
    print(f"Results saved to: {args.outdir}")
    
    return results

if __name__ == "__main__":
    main()