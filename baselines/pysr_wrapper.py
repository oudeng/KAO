# -*- coding: utf-8 -*-
"""
pysr_wrapper.py - PySR baseline with unified BaselineSR interface
Original: PySR_v1.py by Ou Deng (Oct 2025)
Upgraded: Feb 2026 — added PySRSR(BaselineSR) class
"""
from __future__ import annotations

import time
import json
import re
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

# Import shared standardized classes for HOF compatibility
from kao.shared_classes import StandardizedIndividual, StandardizedFitness, create_standardized_individual
from baselines import BaselineSR

try:
    from pysr import PySRRegressor
except Exception:
    PySRRegressor = None

# ==========================================================
# Safe math and metrics 
# ==========================================================
EPS = 1e-12

def mse(y, yhat):
    """Calculate MSE with numerical safety"""
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    yhat_clipped = np.clip(yhat, -1e15, 1e15)
    with np.errstate(all="ignore"):
        diff = y - yhat_clipped
        diff_clipped = np.clip(diff, -1e7, 1e7)
        mse_val = np.mean(diff_clipped ** 2)
    return float(mse_val) if np.isfinite(mse_val) else 1e9

def r2(y, yhat):
    """Calculate R2 score with numerical safety"""
    try:
        return float(r2_score(y, yhat))
    except:
        return 0.0

def clip_and_sanitize(yhat, y_clip=None):
    """Clip and sanitize predictions"""
    yhat = np.asarray(yhat, dtype=float)
    if y_clip is not None and len(y_clip) == 2:
        yhat = np.clip(yhat, y_clip[0], y_clip[1])
    return np.where(np.isfinite(yhat), yhat, 0.0)

# ==========================================================
# Expression evaluation
# ==========================================================
def evaluate_pysr_expr(expr_str: str, X: np.ndarray, feature_names: List[str]) -> np.ndarray:
    """Evaluate a PySR expression string on data X"""
    try:
        # Create namespace with feature values
        namespace = {}
        for i, fname in enumerate(feature_names):
            namespace[fname] = X[:, i]
            namespace[f'x{i}'] = X[:, i]
        
        # Add numpy functions
        namespace.update({
            'sqrt': np.sqrt,
            'square': lambda x: x**2,
            'exp': np.exp,
            'log': lambda x: np.log(np.abs(x) + 1e-10),
            'sin': np.sin,
            'cos': np.cos,
            'abs': np.abs,
            'sign': np.sign,
        })
        
        # Clean expression
        expr_clean = str(expr_str).strip()
        
        # Handle subscript notation
        subscripts = {'₀':'0', '₁':'1', '₂':'2', '₃':'3', '₄':'4', 
                     '₅':'5', '₆':'6', '₇':'7', '₈':'8', '₉':'9'}
        for sub, digit in subscripts.items():
            expr_clean = expr_clean.replace(sub, digit)
        
        # Evaluate expression
        with np.errstate(all='ignore'):
            result = eval(expr_clean, {"__builtins__": {}}, namespace)
        
        return np.asarray(result, dtype=float)
    except Exception as e:
        return np.zeros(X.shape[0])

def calc_weighted_complexity(expr_str: str) -> float:
    """Calculate weighted complexity matching LGO's approach"""
    if not expr_str or expr_str in ["N/A", "Error extracting", None]:
        return 0.0
    
    expr_lower = str(expr_str).lower()
    
    # Weights matching unified complexity calculation
    weights = {
        '+': 1.0, '-': 1.0, '*': 1.0, '/': 1.5,
        'sqrt': 1.5, 'square': 1.5, 'log': 1.5, 'exp': 1.5,
        'sin': 1.5, 'cos': 1.5, 'tan': 1.5,
        'abs': 1.5, 'sign': 1.0, 'pow': 2.0
    }
    
    complexity = 0.0
    
    # Count operators
    for op, weight in weights.items():
        if op in ['+', '-', '*', '/']:
            complexity += expr_lower.count(op) * weight
        else:
            pattern = r'\b' + re.escape(op) + r'\b'
            complexity += len(re.findall(pattern, expr_lower)) * weight
    
    # Add complexity for variables
    variables = set(re.findall(r'\b[A-Za-z_]\w*\b', expr_str))
    func_names = set(weights.keys())
    variables = {v for v in variables if v.lower() not in func_names}
    complexity += len(variables) * 0.5
    
    # Add minimal complexity for constants
    constants = len(re.findall(r'\b\d+\.?\d*\b', expr_str))
    complexity += constants * 0.25
    
    return max(1.0, complexity)

def crossval_pysr(expr_str: str, X: np.ndarray, y: np.ndarray, 
                  feature_names: List[str], n_splits: int = 5, 
                  random_state: int = 0) -> tuple:
    """Compute CV metrics for a PySR expression"""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    mses, r2s = [], []
    
    for tr, va in kf.split(X):
        yhat_va = evaluate_pysr_expr(expr_str, X[va], feature_names)
        yhat_va = clip_and_sanitize(yhat_va)
        mses.append(mse(y[va], yhat_va))
        r2s.append(r2(y[va], yhat_va))
    
    cv_loss = float(np.median(mses)) if mses else 1e9
    instability = float(np.std(mses) / max(np.mean(mses), EPS)) if mses else 1.0
    cv_r2 = float(np.mean(r2s)) if r2s else 0.0
    
    return cv_loss, instability, cv_r2

# ==========================================================
# Main unified PySR runner 
# ==========================================================
def run_pysr_unified(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Optional[Sequence[str]] = None,
    niterations: int = 40,
    population_size: int = 33,
    maxsize: int = 20,
    maxdepth: int = 10,
    parsimony: float = 0.0032,
    topk: int = 12,
    random_state: int = 0,
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    verbose: bool = True,
    time_budget: Optional[float] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Run PySR with unified interface
    
    Returns DataFrame with columns:
    - expr: expression string
    - cv_loss: CV MSE
    - cv_r2: CV R2
    - test_loss: test MSE
    - test_r2: test R2
    - complexity: weighted complexity
    - instability: CV instability metric
    """
    
    if PySRRegressor is None:
        raise RuntimeError("PySR is not installed. Install with: pip install pysr")
    
    t0 = time.time()
    
    # Setup feature names
    if feature_names is None:
        feature_names = [f"x{i}" for i in range(X.shape[1])]
    
    # Configure PySR
    binary_ops = ["+", "-", "*", "/"]
    unary_ops = ["sqrt", "exp", "log"]
    
    # v3.2: runtime-detect timeout parameter name
    import inspect
    timeout_kwargs = {}
    if time_budget is not None:
        sig = inspect.signature(PySRRegressor.__init__)
        if 'timeout_in_seconds' in sig.parameters:
            timeout_kwargs['timeout_in_seconds'] = float(time_budget)
        elif 'timeout' in sig.parameters:
            timeout_kwargs['timeout'] = float(time_budget)
        else:
            if verbose:
                print('[PySR] WARNING: no native timeout param; will rely on niterations/pop_size settings')

    model = PySRRegressor(
        niterations=niterations,
        population_size=population_size,
        binary_operators=binary_ops,
        unary_operators=unary_ops,
        maxsize=maxsize,
        maxdepth=maxdepth,
        parsimony=parsimony,
        complexity_of_operators={
            "+": 1, "-": 1, "*": 1, "/": 1.5,
            "sqrt": 1.5, "exp": 1.5, "log": 1.5
        },
        elementwise_loss="(x, y) -> (x - y)^2",
        batching=False,
        turbo=False,
        progress=verbose,
        random_state=random_state if random_state != 0 else None,
        procs=1,
        temp_equation_file=False,
        verbosity=2 if verbose else 0,
        **timeout_kwargs,
    )
    
    if verbose:
        print(f"[PySR] Starting with population={population_size}, iterations={niterations}")
    
    # Fit model
    model.fit(X, y, variable_names=list(feature_names))
    
    # Extract equations
    rows = []
    
    try:
        eq_df = model.equations_
        if eq_df is None or len(eq_df) == 0:
            print("[PySR] Warning: No equations found")
            return pd.DataFrame()
        
        # Sort by loss and complexity
        eq_df = eq_df.sort_values(["loss", "complexity"], ascending=[True, True])
        
        # Process top K equations
        K = min(topk, len(eq_df))
        
        if verbose:
            print(f"[PySR] Evaluating top {K} candidates...")
        
        for i in range(K):
            row_data = eq_df.iloc[i]
            
            # Get expression string
            if "sympy_format" in row_data and row_data["sympy_format"] is not None:
                expr_str = str(row_data["sympy_format"])
            elif "equation" in row_data and row_data["equation"] is not None:
                expr_str = str(row_data["equation"])
            else:
                continue
            
            if verbose and i < 3:
                print(f"  Candidate {i+1}/{K}: {expr_str[:60]}...")
            
            # Compute CV metrics
            cv_loss, instability, cv_r2 = crossval_pysr(
                expr_str, X, y, feature_names, n_splits=5, random_state=random_state
            )
            
            # Test evaluation
            test_loss, test_r2 = (np.nan, np.nan)
            if X_test is not None and y_test is not None:
                yhat_test = evaluate_pysr_expr(expr_str, X_test, feature_names)
                yhat_test = clip_and_sanitize(yhat_test)
                test_loss = mse(y_test, yhat_test)
                test_r2 = r2(y_test, yhat_test)
            
            # Create row with unified format
            rows.append({
                "rank": i,
                "expr": expr_str,
                "cv_loss": float(cv_loss),
                "cv_r2": float(cv_r2),
                "instability": float(instability),
                "test_loss": float(test_loss) if np.isfinite(test_loss) else np.nan,
                "test_r2": float(test_r2) if np.isfinite(test_r2) else np.nan,
                "complexity": calc_weighted_complexity(expr_str),
                "runtime_sec": float(time.time() - t0),
            })
    
    except Exception as e:
        print(f"[PySR] Error processing equations: {e}")
        return pd.DataFrame()
    
    df = pd.DataFrame(rows)
    if len(df) > 0:
        # Sort by CV loss as primary metric
        df = df.sort_values(["cv_loss", "complexity"], ascending=[True, True]).reset_index(drop=True)
    
    if verbose:
        print(f"[PySR] Completed in {time.time() - t0:.2f}s")
    
    return df

# ==========================================================
# Standardized classes for HOF compatibility
# ==========================================================
class StandardizedFitness:
    """Fitness class compatible with DEAP-style individuals"""
    def __init__(self, values):
        self.values = values

class StandardizedIndividual:
    """Individual class compatible with DEAP-style HOF"""
    def __init__(self, expr, mse, complexity):
        self.expr = expr
        self.fitness = StandardizedFitness((mse, complexity))
        
    def __str__(self):
        return str(self.expr)
    
    @property
    def height(self):
        """Estimate tree height from expression"""
        return self._estimate_height()
    
    def _estimate_height(self):
        """Simple heuristic: count parentheses depth"""
        max_depth = 0
        current = 0
        for char in str(self.expr):
            if char == '(':
                current += 1
                max_depth = max(max_depth, current)
            elif char == ')':
                current -= 1
        return max_depth

def extract_best_from_pysr(df_results: pd.DataFrame, feature_names: List[str]) -> List:
    """
    Extract best expressions from PySR results to create HOF-like structure

    Returns list of StandardizedIndividual objects
    """
    hof = []

    # Take top expressions by CV loss
    for _, row in df_results.iterrows():
        ind = create_standardized_individual(
            row['expr'],
            row['cv_loss'],
            row['complexity']
        )
        hof.append(ind)

    return hof


# ==========================================================
# Unified BaselineSR interface
# ==========================================================
class PySRSR(BaselineSR):
    """PySR wrapped in the unified BaselineSR interface."""

    @property
    def name(self) -> str:
        return "PySR"

    def fit(self, X_train, y_train, X_test=None, y_test=None,
            time_budget=60.0, random_state=42, **kwargs):
        feature_names = kwargs.pop(
            "feature_names",
            [f"x{i}" for i in range(X_train.shape[1])],
        )
        niterations = kwargs.pop("niterations", 40)
        population_size = kwargs.pop("population_size", 33)
        maxsize = kwargs.pop("maxsize", 20)
        parsimony = kwargs.pop("parsimony", 0.0032)

        t0 = time.time()

        df = run_pysr_unified(
            X_train, y_train,
            feature_names=feature_names,
            niterations=niterations,
            population_size=population_size,
            maxsize=maxsize,
            parsimony=parsimony,
            random_state=random_state,
            X_test=X_test, y_test=y_test,
            verbose=False,
            time_budget=time_budget,
            **kwargs,
        )

        elapsed = time.time() - t0

        if df is None or len(df) == 0:
            nan_arr = np.full(len(y_train), np.nan)
            return {
                "expression": "FAILED",
                "y_pred_train": nan_arr,
                "y_pred_test": np.full(len(y_test), np.nan) if y_test is not None else None,
                "complexity": 0,
                "complexity_chars": 0,
                "runtime": elapsed,
                "model": None,
            }

        best = df.iloc[0]
        expr_str = str(best["expr"])

        y_pred_train = evaluate_pysr_expr(expr_str, X_train, feature_names)
        y_pred_train = clip_and_sanitize(y_pred_train)

        result = {
            "expression": expr_str,
            "y_pred_train": y_pred_train,
            "complexity": int(best.get("complexity", 0)),
            "complexity_chars": self.compute_complexity_chars(expr_str),
            "runtime": elapsed,
            "model": df,
        }

        if X_test is not None:
            y_pred_test = evaluate_pysr_expr(expr_str, X_test, feature_names)
            result["y_pred_test"] = clip_and_sanitize(y_pred_test)

        return result