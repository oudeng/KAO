# -*- coding: utf-8 -*-
"""
operon_wrapper.py - Operon C++ baseline with unified BaselineSR interface
pip install pyoperon
"""
from baselines import BaselineSR
import numpy as np
import time

try:
    from pyoperon.sklearn import SymbolicRegressor as _OperonSR
    OPERON_AVAILABLE = True
except ImportError:
    OPERON_AVAILABLE = False

_OPERON_INSTALL_MSG = (
    "pyoperon is required for the Operon baseline. "
    "Install with: pip install pyoperon"
)


class OperonSR(BaselineSR):
    """Operon: high-performance C++ GP with native NSGA-II."""

    @property
    def name(self) -> str:
        return "Operon"

    def fit(self, X_train, y_train, X_test=None, y_test=None,
            time_budget=60.0, random_state=42, feature_names=None, **kwargs):
        if not OPERON_AVAILABLE:
            raise ImportError(_OPERON_INSTALL_MSG)

        import inspect

        pop_size = kwargs.pop("population_size", 1000)
        start = time.perf_counter()

        # --- Dynamic parameter mapping (pyoperon API varies by version) ---
        sig = inspect.signature(_OperonSR.__init__)
        valid_params = list(sig.parameters.keys())

        params: dict = {"random_state": random_state}

        # Time limit (pyoperon 0.4 uses time_limit, 0.5 uses max_time)
        for tp in ("max_time", "time_limit", "time_budget", "max_seconds"):
            if tp in valid_params:
                params[tp] = int(time_budget)
                break

        # Thread count
        for tp in ("n_threads", "num_threads", "n_jobs"):
            if tp in valid_params:
                params[tp] = 1
                break

        # Population size
        if "population_size" in valid_params:
            params["population_size"] = pop_size

        # Standardise X for numerical stability (Y kept raw)
        from sklearn.preprocessing import StandardScaler
        self._scaler_X = StandardScaler()
        X_train_sc = self._scaler_X.fit_transform(X_train)

        # Debug: report params and data ranges
        print(f"    Operon params: {params}")
        print(f"    X_train raw  range: {X_train.min():.3f} ~ {X_train.max():.3f}")
        print(f"    X_train_sc   range: {X_train_sc.min():.3f} ~ {X_train_sc.max():.3f}")
        print(f"    y_train      range: {y_train.min():.3f} ~ {y_train.max():.3f}")

        try:
            est = _OperonSR(**params)
            est.fit(X_train_sc, y_train)
        except Exception as e:
            import traceback
            print(f"    Operon internal error: {e}")
            traceback.print_exc()
            result = {
                "expression": f"FAILED: {e}",
                "y_pred_train": np.full(len(y_train), np.mean(y_train)),
                "complexity": 0,
                "complexity_chars": 0,
                "runtime": time.perf_counter() - start,
                "model": None,
            }
            if X_test is not None:
                result["y_pred_test"] = np.full(len(X_test), np.mean(y_train))
            return result

        elapsed = time.perf_counter() - start

        # --- Extract expression string (API varies) ---
        expr_str = "UNKNOWN"
        if hasattr(est, "get_model_string") and hasattr(est, "model_"):
            # pyoperon 0.5+: get_model_string(model, precision, names)
            # Use caller-provided feature names if available
            fn_for_model = None
            if feature_names is not None and len(feature_names) > 0:
                fn_for_model = list(feature_names)
            elif hasattr(est, "variables_") and est.variables_:
                vdict = est.variables_
                if isinstance(vdict, dict):
                    fn_for_model = [vdict[k] for k in vdict]
                else:
                    fn_for_model = list(vdict)
            try:
                expr_str = est.get_model_string(
                    est.model_, precision=6, names=fn_for_model,
                )
            except TypeError:
                # Older pyoperon: get_model_string() takes no args
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

        # --- Extract complexity (API varies) ---
        complexity = 0
        if hasattr(est, "stats_") and isinstance(est.stats_, dict):
            complexity = est.stats_.get("model_length", 0)
        elif hasattr(est, "model_") and hasattr(est.model_, "__len__"):
            complexity = len(est.model_)
        elif hasattr(est, "complexity_"):
            complexity = est.complexity_
        else:
            # Last resort: use expression character length
            complexity = len(expr_str) if expr_str else 0

        # Clip predictions to a safe range based on training target
        y_min, y_max = float(np.min(y_train)), float(np.max(y_train))
        y_range = y_max - y_min if y_max > y_min else 1.0
        clip_lo = y_min - 3 * y_range
        clip_hi = y_max + 3 * y_range

        raw_pred_train = est.predict(X_train_sc)
        raw_pred_train = np.asarray(raw_pred_train, dtype=float)
        raw_pred_train = np.where(
            np.isfinite(raw_pred_train), raw_pred_train, np.mean(y_train),
        )
        raw_pred_train = np.clip(raw_pred_train, clip_lo, clip_hi)

        # Debug: report prediction range and expression
        print(f"    y_pred_train range: {raw_pred_train.min():.3f} ~ {raw_pred_train.max():.3f}")
        print(f"    Operon model string: {expr_str[:200]}")
        print(f"    Operon complexity: {complexity}")

        result = {
            "expression": expr_str,
            "y_pred_train": raw_pred_train,
            "complexity": int(complexity),
            "complexity_chars": self.compute_complexity_chars(expr_str),
            "runtime": elapsed,
            "model": est,
            "hparams_effective": params,
        }
        if X_test is not None:
            X_test_sc = self._scaler_X.transform(X_test)
            raw_pred_test = est.predict(X_test_sc)
            raw_pred_test = np.asarray(raw_pred_test, dtype=float)
            raw_pred_test = np.where(
                np.isfinite(raw_pred_test), raw_pred_test, np.mean(y_train),
            )
            raw_pred_test = np.clip(raw_pred_test, clip_lo, clip_hi)
            result["y_pred_test"] = raw_pred_test

        return result
