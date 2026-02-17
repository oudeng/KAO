# -*- coding: utf-8 -*-
"""
rils_rols_wrapper.py - RILS-ROLS baseline with unified BaselineSR interface
Original: RILS_ROLS_v2.py by Ou Deng (Oct 2025)
Upgraded: Feb 2026 — added RILSROLSSR(BaselineSR) class
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List, Set
import time
import re
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Import shared standardized classes for HOF compatibility
from kao.shared_classes import StandardizedIndividual, StandardizedFitness, create_standardized_individual
from baselines import BaselineSR

# Import RILS-ROLS
def _import_rils_rols_regressor():
    """Import RILSROLSRegressor from the correct location"""
    try:
        from rils_rols.rils_rols import RILSROLSRegressor
        return RILSROLSRegressor, "rils_rols.rils_rols"
    except ImportError:
        try:
            from rils_rols import RILSROLSRegressor
            return RILSROLSRegressor, "rils_rols"
        except ImportError as e:
            return None, f"import_failed: {str(e)}"

# Gate/binary features that should not be filtered even if imbalanced
GATE_WHITELIST = {
    "gender_std", "age_band", "mechanical_ventilation_std", 
    "vasopressor_use_std", "DS", "sex", "is_male", "is_female",
    "gender", "sex_male", "sex_female", "binary_*"
}

@dataclass
class RILSROLSConfig:
    """Configuration for RILS-ROLS experiments"""
    seed: int = 0
    max_fit_calls: int = 100000
    max_time: int = 100
    complexity_penalty: float = 0.001
    max_complexity: int = 50
    sample_size: float = 1.0
    verbose: bool = False
    timeout_sec: Optional[int] = None
    internal_standardize: bool = True  # NEW: Flag to control internal standardization

def calc_weighted_complexity(expr_str):
    """Calculate weighted complexity for consistency with other methods"""
    if not expr_str or expr_str in ["N/A", "Error extracting", None]:
        return 0.0
    
    # Check if constant
    try:
        float(expr_str)
        return 1.0
    except:
        pass
    
    expr_lower = str(expr_str).lower()
    
    weights = {
        '+': 1.0, '-': 1.0, '*': 1.0, '/': 1.5,
        'sqrt': 1.5, 'log': 1.5, 'exp': 1.5,
        'sin': 1.5, 'cos': 1.5, 'tan': 1.5,
        'pow': 2.0, 'abs': 1.5, 'sign': 1.0
    }
    
    complexity = 0.0
    for op, weight in weights.items():
        if op in ['+', '-', '*', '/']:
            complexity += expr_lower.count(op) * weight
        else:
            pattern = r'\b' + re.escape(op) + r'\b'
            complexity += len(re.findall(pattern, expr_lower)) * weight
    
    # Add complexity for variables
    variables = set(re.findall(r'\b[A-Za-z_]\w*\b', expr_str))
    func_names = {'exp', 'log', 'sqrt', 'sin', 'cos', 'tan', 'abs', 'sign', 'pow'}
    variables = {v for v in variables if v.lower() not in func_names}
    complexity += len(variables) * 0.5
    
    # Add complexity for constants
    constants = len(re.findall(r'\b\d+\.?\d*\b', expr_str))
    complexity += constants * 0.25
    
    return max(1.0, complexity)

def safe_eval_expr(expr_str, X, feature_names=None):
    """
    Safely evaluate expression with NaN/Inf protection
    Returns: (predictions, is_valid)
    """
    try:
        if not expr_str or expr_str in ["N/A", "Error extracting", None]:
            return np.full(X.shape[0], np.nan), False
        
        # If expression is just a constant
        try:
            const_val = float(expr_str)
            return np.full(X.shape[0], const_val), True
        except:
            pass
        
        # Map feature names to X columns
        if feature_names:
            # Create a safe namespace with numpy functions
            namespace = {
                'exp': np.exp,
                'log': lambda x: np.log(np.abs(x) + 1e-9),
                'sqrt': lambda x: np.sqrt(np.abs(x) + 1e-9),
                'abs': np.abs,
                'sin': np.sin,
                'cos': np.cos,
                'tan': np.tan,
            }
            
            # Add feature columns to namespace
            for i, fname in enumerate(feature_names):
                if i < X.shape[1]:
                    namespace[fname] = X[:, i]
            
            # Evaluate expression
            result = eval(expr_str, namespace)
            
            # Convert to array and check for NaN/Inf
            result = np.asarray(result)
            if np.any(~np.isfinite(result)):
                return result, False
            
            return result, True
        else:
            return np.full(X.shape[0], np.nan), False
            
    except Exception as e:
        return np.full(X.shape[0], np.nan), False

def linear_refit(y_true, f_x):
    """
    Refit linear scaling: y = alpha * f(x) + beta
    Returns: alpha, beta, y_pred
    """
    try:
        if np.all(~np.isfinite(f_x)) or np.std(f_x) < 1e-10:
            # Constant or invalid predictions
            return 0.0, np.mean(y_true), np.full_like(y_true, np.mean(y_true))
        
        # Use LinearRegression for robust fitting
        lr = LinearRegression()
        f_x_reshaped = f_x.reshape(-1, 1)
        lr.fit(f_x_reshaped, y_true)
        
        alpha = lr.coef_[0]
        beta = lr.intercept_
        y_pred = lr.predict(f_x_reshaped)
        
        return alpha, beta, y_pred
        
    except Exception as e:
        return 0.0, np.mean(y_true), np.full_like(y_true, np.mean(y_true))

def _complexity_from_expression(expr_str):
    """Extract complexity metrics from RILS-ROLS expression"""
    try:
        if not expr_str or expr_str in ["N/A", "Error extracting", None]:
            return np.nan, np.nan
        
        # Check if constant
        try:
            float(expr_str)
            return 1, 0  # Constant: size=1, height=0
        except:
            pass
        
        # Count operations and terms
        operations = len(re.findall(r'[+\-*/^]|exp|log|sin|cos|tan|sqrt|abs', expr_str))
        
        # Count unique variables
        variables = set(re.findall(r'\bx\d+\b|[A-Za-z_]\w*(?<!exp)(?<!log)(?<!sin)(?<!cos)(?<!sqrt)(?<!abs)', expr_str))
        variables = {v for v in variables if v not in ['exp', 'log', 'sin', 'cos', 'tan', 'sqrt', 'abs']}
        n_vars = len(variables)
        
        # Estimate height from parenthesis nesting
        max_height = 0
        current_depth = 0
        for char in expr_str:
            if char == '(':
                current_depth += 1
                max_height = max(max_height, current_depth)
            elif char == ')':
                current_depth -= 1
        
        # Count constants
        constants = len(re.findall(r'\b\d+\.?\d*\b', expr_str))
        
        # Size is roughly operations + variables + constants
        size = max(1, operations + n_vars + constants)
        
        return size, max(0, max_height)
        
    except Exception:
        return np.nan, np.nan

def _filter_near_constant_features_safe(X_tr, X_te, names, var_eps=1e-12, uniq_ratio=0.01, whitelist=None):
    """Filter near-constant features with whitelist support"""
    if whitelist is None:
        whitelist = GATE_WHITELIST
    
    X_tr = np.asarray(X_tr)
    X_te = np.asarray(X_te)
    keep = []
    
    for j in range(X_tr.shape[1]):
        name = names[j] if j < len(names) else f'X{j}'
        
        col = X_tr[:, j]
        col_clean = col[~np.isnan(col)]
        var = np.nanvar(col_clean) if col_clean.size > 0 else 0
        
        # Whitelist check
        is_whitelisted = False
        for white_pattern in whitelist:
            if white_pattern.endswith('*'):
                if name.startswith(white_pattern[:-1]):
                    is_whitelisted = True
                    break
            elif name == white_pattern:
                is_whitelisted = True
                break
        
        if is_whitelisted:
            keep.append(j)
            continue
        
        if col_clean.size == 0:
            continue
        
        unique = np.unique(col_clean)
        if unique.size <= 1:
            continue
        
        std = np.nanstd(col_clean)
        uniq_frac = unique.size / col_clean.size
        
        if std <= var_eps or uniq_frac <= uniq_ratio:
            continue
        
        keep.append(j)
    
    keep = np.array(keep, dtype=int)
    
    if keep.size == 0:
        # Keep at least some features with highest variance
        variances = []
        for j in range(X_tr.shape[1]):
            col = X_tr[:, j]
            col_clean = col[~np.isnan(col)]
            var = np.nanvar(col_clean) if col_clean.size > 0 else 0
            variances.append(var)
        n_keep = min(3, len(variances))
        keep = np.argsort(variances)[-n_keep:][::-1]
    
    X_tr_filtered = X_tr[:, keep]
    X_te_filtered = X_te[:, keep]
    names_filtered = [names[i] for i in keep]
    
    drop_mask = [1 if i not in keep else 0 for i in range(len(names))]
    
    return X_tr_filtered, X_te_filtered, names_filtered, keep, drop_mask

def _map_expression_to_features(expr_str, feature_names):
    """Map x0, x1, etc. to actual feature names"""
    if not expr_str or not feature_names:
        return expr_str
    
    mapped_expr = expr_str
    
    # Sort by index in descending order to avoid x1 being replaced before x10
    indices = []
    for i, fname in enumerate(feature_names):
        indices.append((i, fname))
    indices.sort(key=lambda x: x[0], reverse=True)
    
    for i, fname in indices:
        mapped_expr = re.sub(f'\\bx{i}\\b', fname, mapped_expr)
    
    return mapped_expr

def _kfold_cv_with_expressions(X, y, build_fn, seed: int, feature_names, k: int = 5, internal_standardize: bool = True):
    """K-fold cross validation with standardized metrics
    
    Parameters:
    -----------
    internal_standardize : bool
        If True, standardize within each fold. If False, use data as-is.
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    cv_results = []
    mses = []
    r2s = []
    
    fold_idx = 0
    for tr, va in kf.split(X):
        fold_idx += 1
        
        # Conditionally standardize within each fold
        if internal_standardize:
            scaler = StandardScaler()
            Xt = scaler.fit_transform(X[tr])
            Xv = scaler.transform(X[va])
        else:
            scaler = None
            Xt = X[tr]
            Xv = X[va]
        
        yt, yv = y[tr], y[va]
        
        m = build_fn()
        try:
            # Train model
            m.fit(Xt, yt)
            
            # Get expression
            expr_raw = m.model_string()
            
            # Handle different return types
            if isinstance(expr_raw, (float, int, np.number)):
                expr = f"{float(expr_raw):.6f}"
                is_constant = True
            else:
                expr = str(expr_raw)
                is_constant = False
            
            # Map to feature names
            expr = _map_expression_to_features(expr, feature_names)
            
            # Direct prediction for validation
            yh = m.predict(Xv)
            mse = mean_squared_error(yv, yh)
            r2 = r2_score(yv, yh)
            
            mses.append(mse)
            r2s.append(r2)
            
            cv_results.append({
                'fold': fold_idx,
                'r2': r2,
                'mse': mse,
                'expr': expr,
                'is_constant': is_constant,
                'scaler': scaler,
                'model': m
            })
            
        except Exception as e:
            cv_results.append({
                'fold': fold_idx,
                'r2': np.nan,
                'mse': np.nan,
                'expr': None,
                'is_constant': True,
                'scaler': scaler,
                'model': None
            })
            mses.append(np.nan)
            r2s.append(np.nan)
    
    # Calculate instability
    EPS = 1e-12
    valid_mses = [m for m in mses if not np.isnan(m)]
    if valid_mses:
        instability = float(np.std(valid_mses) / max(np.mean(valid_mses), EPS))
    else:
        instability = 1.0
    
    # Add instability to results
    for result in cv_results:
        result['instability'] = instability
    
    return cv_results

def run_rils_rols_unified(X_tr, y_tr, X_te, y_te, seed, cfg: RILSROLSConfig, feature_names: List[str] = None):
    """
    Run RILS-ROLS experiment with unified interface
    
    Returns dictionary with standardized metrics
    """
    
    if feature_names is None:
        feature_names = [f'x{i}' for i in range(X_tr.shape[1])]
    
    # Filter near-constant features
    original_n_features = X_tr.shape[1]
    X_tr, X_te, feature_names, keep_idx, drop_mask = _filter_near_constant_features_safe(
        X_tr, X_te, feature_names, whitelist=GATE_WHITELIST
    )
    
    if cfg.verbose:
        print(f"    [Seed {seed}] Features: {original_n_features} -> {X_tr.shape[1]} after filtering")
        if not cfg.internal_standardize:
            print(f"    [Seed {seed}] Internal standardization disabled (data already standardized)")
    
    assert X_tr.shape[1] == X_te.shape[1] == len(feature_names), \
        f"Feature count mismatch after filtering"
    assert X_tr.shape[1] > 0, "No features remaining after filtering"
    
    RILSROLSRegressor, backend = _import_rils_rols_regressor()
    t0 = time.time()

    row = {
        "seed": seed,
        "runtime_sec": None,
        "cv_loss": np.nan,
        "cv_r2": np.nan,
        "test_loss": np.nan,
        "test_r2": np.nan,
        "complexity": np.nan,
        "instability": np.nan,
        "expr_str": None,
        "error": "",
    }

    if RILSROLSRegressor is None:
        row["error"] = f"rils_rols_import_failed: {backend}"
        row["runtime_sec"] = round(time.time() - t0, 4)
        return row

    def build():
        """Build RILS-ROLS model"""
        return RILSROLSRegressor(
            max_fit_calls=cfg.max_fit_calls,
            max_time=cfg.max_time,
            complexity_penalty=cfg.complexity_penalty,
            max_complexity=cfg.max_complexity,
            sample_size=cfg.sample_size,
            verbose=cfg.verbose,
            random_state=seed if seed != 0 else None
        )

    # Run K-fold CV with conditional standardization
    if cfg.verbose:
        print(f"    [Seed {seed}] Running 5-fold CV...")
    cv_results = _kfold_cv_with_expressions(
        X_tr, y_tr, build, seed, feature_names, 
        k=5, internal_standardize=cfg.internal_standardize  # Pass flag
    )
    
    # Calculate CV statistics
    valid_mses = [r['mse'] for r in cv_results if not np.isnan(r['mse'])]
    valid_r2s = [r['r2'] for r in cv_results if not np.isnan(r['r2'])]
    
    if valid_mses and valid_r2s:
        row["cv_loss"] = float(np.median(valid_mses))
        row["cv_r2"] = float(np.mean(valid_r2s))
        row["instability"] = cv_results[0]['instability']
    
    # Select best CV expression
    best_fold = None
    best_r2 = -np.inf
    for result in cv_results:
        if not np.isnan(result['r2']) and result['r2'] > best_r2:
            if not result['is_constant']:
                best_r2 = result['r2']
                best_fold = result
    
    # Fallback to best constant if no non-constant found
    if best_fold is None:
        for result in cv_results:
            if not np.isnan(result['r2']) and result['r2'] > best_r2:
                best_r2 = result['r2']
                best_fold = result
    
    if best_fold is None:
        row["error"] = "cv_all_failed"
        row["runtime_sec"] = round(time.time() - t0, 4)
        return row
    
    # Use best CV expression for final evaluation
    if cfg.verbose:
        print(f"    [Seed {seed}] Using best CV expression (R² = {best_fold['r2']:.4f})")
    
    try:
        # Conditionally standardize full training and test sets
        if cfg.internal_standardize:
            scaler_final = StandardScaler()
            Xtr_s = scaler_final.fit_transform(X_tr)
            Xte_s = scaler_final.transform(X_te)
        else:
            Xtr_s = X_tr
            Xte_s = X_te
        
        # Get expression
        expr = best_fold['expr']
        row["expr_str"] = expr
        
        # Calculate weighted complexity
        row["complexity"] = calc_weighted_complexity(expr)
        
        # Evaluate on training set with linear refit
        yh_train, train_ok = safe_eval_expr(expr, Xtr_s, feature_names)
        
        if train_ok and not np.all(np.isnan(yh_train)):
            # Linear refit on training data
            alpha, beta, yh_train_fitted = linear_refit(y_tr, yh_train)
            
            # Apply to test set
            yh_test, test_ok = safe_eval_expr(expr, Xte_s, feature_names)
            
            if test_ok and not np.all(np.isnan(yh_test)):
                # Apply linear scaling
                yh_test_fitted = alpha * yh_test + beta
                
                row["test_loss"] = float(mean_squared_error(y_te, yh_test_fitted))
                row["test_r2"] = float(r2_score(y_te, yh_test_fitted))
                
                if cfg.verbose:
                    print(f"    [Seed {seed}] Test R² = {row['test_r2']:.4f}")
            else:
                row["error"] = "test_eval_failed"
        else:
            row["error"] = "train_eval_failed"
            
    except Exception as e:
        row["error"] = f"evaluation_failed: {str(e)}"

    row["runtime_sec"] = round(time.time() - t0, 4)

    if cfg.verbose:
        print(f"    [Seed {seed}] Completed in {row['runtime_sec']}s")

    return row


# ==========================================================
# Unified BaselineSR interface
# ==========================================================
class RILSROLSSR(BaselineSR):
    """RILS-ROLS wrapped in the unified BaselineSR interface."""

    @property
    def name(self) -> str:
        return "RILS-ROLS"

    def fit(self, X_train, y_train, X_test=None, y_test=None,
            time_budget=60.0, random_state=42, **kwargs):
        feature_names = kwargs.pop(
            "feature_names",
            [f"x{i}" for i in range(X_train.shape[1])],
        )

        # Internal 5-fold CV: each fold gets max_time, so divide budget by 5
        cfg = RILSROLSConfig(
            seed=random_state,
            max_time=int(time_budget) // 5,
            max_fit_calls=kwargs.pop("max_fit_calls", 100000),
            complexity_penalty=kwargs.pop("complexity_penalty", 0.001),
            max_complexity=kwargs.pop("max_complexity", 50),
            sample_size=kwargs.pop("sample_size", 1.0),
            verbose=kwargs.pop("verbose", False),
            internal_standardize=kwargs.pop("internal_standardize", True),
        )

        row = run_rils_rols_unified(
            X_train, y_train,
            X_test if X_test is not None else X_train[:0],
            y_test if y_test is not None else y_train[:0],
            seed=random_state, cfg=cfg,
            feature_names=feature_names,
        )

        expr_str = row.get("expr_str") or "FAILED"
        runtime = row.get("runtime_sec", 0.0)

        # Compute predictions via safe_eval
        y_pred_train, _ = safe_eval_expr(expr_str, X_train, feature_names)
        y_pred_train = np.where(np.isfinite(y_pred_train), y_pred_train, 0.0)

        result = {
            "expression": expr_str,
            "y_pred_train": y_pred_train,
            "complexity": int(row.get("complexity", 0)) if np.isfinite(row.get("complexity", 0)) else 0,
            "complexity_chars": self.compute_complexity_chars(expr_str),
            "runtime": float(runtime),
            "model": row,
        }

        if X_test is not None:
            y_pred_test, _ = safe_eval_expr(expr_str, X_test, feature_names)
            result["y_pred_test"] = np.where(np.isfinite(y_pred_test), y_pred_test, 0.0)

        return result