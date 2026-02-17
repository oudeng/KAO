# -*- coding: utf-8 -*-
"""
gplearn_wrapper.py - gplearn baseline with unified BaselineSR interface
pip install gplearn
"""

# sklearn >= 1.6 compatibility patch for gplearn 0.4.2
# gplearn calls self._validate_data() which was removed in sklearn 1.6+
import sklearn.base
if not hasattr(sklearn.base.BaseEstimator, '_validate_data'):
    try:
        from sklearn.utils.validation import validate_data as _sklearn_validate_data
        # Re-attach as instance method with same signature
        sklearn.base.BaseEstimator._validate_data = (
            lambda self, *args, **kwargs: _sklearn_validate_data(self, *args, **kwargs)
        )
    except ImportError:
        pass

from baselines import BaselineSR
import numpy as np
import time


class GPLearnSR(BaselineSR):
    """Classic GP symbolic regression via gplearn (scikit-learn API)."""

    @property
    def name(self) -> str:
        return "gplearn"

    def fit(self, X_train, y_train, X_test=None, y_test=None,
            time_budget=60.0, random_state=42, **kwargs):
        from gplearn.genetic import SymbolicRegressor
        import warnings
        warnings.filterwarnings("ignore")

        pop_size = kwargs.pop("population_size", 1000)

        start = time.perf_counter()

        base_params = dict(
            population_size=pop_size,
            function_set=["add", "sub", "mul", "div",
                          "sqrt", "log", "abs", "neg"],
            parsimony_coefficient=0.001,
            max_samples=0.9,
            p_crossover=0.7,
            p_subtree_mutation=0.1,
            p_hoist_mutation=0.05,
            p_point_mutation=0.1,
            random_state=random_state,
            verbose=0,
            n_jobs=1,
        )

        # Phase 1: run 1 generation to measure wall-clock cost
        est = SymbolicRegressor(generations=1, warm_start=False, **base_params)
        est.fit(X_train, y_train)
        gen1_time = time.perf_counter() - start

        # Phase 2: estimate how many generations fit in the remaining budget
        remaining = time_budget - gen1_time
        if remaining > gen1_time and gen1_time > 0:
            total_gens = min(1 + int(remaining / gen1_time), 200)
            est2 = SymbolicRegressor(
                generations=total_gens, warm_start=False, **base_params,
            )
            est2.fit(X_train, y_train)
            est = est2  # use the fully-trained model

        elapsed = time.perf_counter() - start

        # Safely retrieve the best program
        if hasattr(est, "_program") and est._program is not None:
            best = est._program
            expr_str = str(best)
            complexity = best.length_
            y_pred_train = est.predict(X_train)
            y_pred_test = (
                est.predict(X_test) if X_test is not None else None
            )
        else:
            # Fallback when _program is missing
            expr_str = "FAILED"
            complexity = 0
            y_pred_train = np.full(len(y_train), np.mean(y_train))
            y_pred_test = (
                np.full(len(X_test), np.mean(y_train))
                if X_test is not None else None
            )

        result = {
            "expression": expr_str,
            "y_pred_train": y_pred_train,
            "complexity": complexity,
            "complexity_chars": self.compute_complexity_chars(expr_str),
            "runtime": elapsed,
            "model": est,
        }
        if X_test is not None:
            result["y_pred_test"] = y_pred_test

        return result
