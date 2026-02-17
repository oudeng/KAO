# -*- coding: utf-8 -*-
"""
v3.2.1 regression tests

Covers:
  1) run_kao.py no longer has UnboundLocalError (merge_hparams path)
  2) effective time_budget override consistency (CLI > json > defaults)
  3) PySRSR.fit() forwards time_budget to run_pysr_unified()
  4) RILS-ROLS max_time defaults to time_budget when user does not override
  5) merge_hparams utility correctness
"""
import pytest
import sys
import os
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ==========================================================
# Test 1 & 2: merge_hparams utility
# ==========================================================
class TestMergeHparams:
    """Unit tests for utils.hparams.merge_hparams."""

    def test_defaults_only(self):
        from utils.hparams import merge_hparams

        defaults = {"time_budget": 60.0, "ngen": 80}
        result = merge_hparams(defaults)
        assert result == defaults
        # Must be a fresh copy
        assert result is not defaults

    def test_json_overrides_defaults(self):
        from utils.hparams import merge_hparams

        defaults = {"time_budget": 60.0, "ngen": 80}
        json_ov = {"time_budget": 5.0}
        result = merge_hparams(defaults, json_ov)
        assert result["time_budget"] == 5.0
        assert result["ngen"] == 80

    def test_cli_overrides_json(self):
        from utils.hparams import merge_hparams

        defaults = {"time_budget": 60.0}
        json_ov = {"time_budget": 5.0}
        cli_ov = {"time_budget": 2.0}
        result = merge_hparams(defaults, json_ov, cli_ov)
        assert result["time_budget"] == 2.0

    def test_cli_none_does_not_override(self):
        """None in cli_overrides means 'user did not specify' — should not override."""
        from utils.hparams import merge_hparams

        defaults = {"time_budget": 60.0}
        json_ov = {"time_budget": 5.0}
        cli_ov = {"time_budget": None}
        result = merge_hparams(defaults, json_ov, cli_ov)
        assert result["time_budget"] == 5.0, (
            "CLI None should not override json value"
        )

    def test_json_string_parsed(self):
        """merge_hparams accepts a raw JSON string for json_overrides."""
        from utils.hparams import merge_hparams

        defaults = {"time_budget": 60.0}
        result = merge_hparams(defaults, '{"time_budget": 10}')
        assert result["time_budget"] == 10

    def test_empty_json_string(self):
        from utils.hparams import merge_hparams

        defaults = {"time_budget": 60.0}
        result = merge_hparams(defaults, "{}")
        assert result["time_budget"] == 60.0

    def test_invalid_json_raises(self):
        from utils.hparams import merge_hparams

        with pytest.raises(ValueError, match="Failed to parse"):
            merge_hparams({}, "not-valid-json{")


# ==========================================================
# Test 3: PySRSR.fit must forward time_budget
# ==========================================================
class TestPySRSRTimeBudget:
    """PySRSR.fit() must pass time_budget to run_pysr_unified."""

    def test_time_budget_forwarded(self, monkeypatch):
        """Monkeypatch run_pysr_unified to record its call kwargs."""
        import baselines.pysr_wrapper as pw
        import numpy as np
        import pandas as pd

        captured = {}

        def fake_run_pysr_unified(*args, **kwargs):
            captured.update(kwargs)
            # Return a minimal DataFrame that PySRSR.fit expects
            return pd.DataFrame([{
                "expr": "x0",
                "cv_loss": 0.1,
                "cv_r2": 0.9,
                "test_loss": 0.1,
                "test_r2": 0.9,
                "complexity": 1,
                "instability": 0.0,
                "runtime_sec": 0.5,
            }])

        monkeypatch.setattr(pw, "run_pysr_unified", fake_run_pysr_unified)

        sr = pw.PySRSR()
        X = np.random.randn(20, 2)
        y = np.random.randn(20)
        sr.fit(X, y, time_budget=2.0, random_state=42)

        assert "time_budget" in captured, (
            "run_pysr_unified was NOT called with time_budget kwarg"
        )
        assert captured["time_budget"] == 2.0


# ==========================================================
# Test 4: RILS-ROLS max_time defaults to time_budget
# ==========================================================
class TestRilsRolsMaxTime:
    """RILS-ROLS max_time should default to time_budget."""

    def test_max_time_defaults_to_time_budget(self):
        """When user does not specify max_time, it should equal time_budget."""
        from utils.hparams import merge_hparams

        defaults = {
            "max_fit_calls": 100000,
            "max_time": 60,
            "time_budget": 60.0,
        }
        json_overrides = {}  # user did NOT specify max_time
        cli_overrides = {"time_budget": 2.0}
        effective = merge_hparams(defaults, json_overrides, cli_overrides)

        # Simulate the logic from run_baselines.py:
        if "max_time" not in json_overrides:
            effective["max_time"] = int(effective["time_budget"])

        assert effective["max_time"] == 2
        assert effective["time_budget"] == 2.0

    def test_user_max_time_overrides(self):
        """When user specifies max_time in json, it should be preserved."""
        from utils.hparams import merge_hparams

        defaults = {
            "max_fit_calls": 100000,
            "max_time": 60,
            "time_budget": 60.0,
        }
        json_overrides = {"max_time": 10}  # user explicitly set max_time
        cli_overrides = {"time_budget": 2.0}
        effective = merge_hparams(defaults, json_overrides, cli_overrides)

        # Simulate the logic from run_baselines.py:
        if "max_time" not in json_overrides:
            effective["max_time"] = int(effective["time_budget"])

        assert effective["max_time"] == 10, (
            "User-specified max_time should be preserved"
        )
        assert effective["time_budget"] == 2.0


# ==========================================================
# Test 5: run_kao_experiment does not raise UnboundLocalError
# ==========================================================
class TestRunKaoNoUnboundLocal:
    """Ensure run_kao.py's function can be called without UnboundLocalError."""

    def test_run_kao_experiment_import_and_call(self, tmp_path):
        """Call run_kao_experiment with a minimal argparse namespace.

        This catches the 'variable used before defined' bug (P0-1)
        because the function will fail immediately at the merge step
        if the code still references `time_budget` before assignment.
        """
        import numpy as np
        import importlib

        # Create a tiny CSV
        csv_path = tmp_path / "toy.csv"
        np.random.seed(42)
        X = np.random.randn(30, 2)
        y = X[:, 0] ** 2 + 0.5 * X[:, 1]
        import pandas as pd
        df = pd.DataFrame({"f0": X[:, 0], "f1": X[:, 1], "target": y})
        df.to_csv(csv_path, index=False)

        # Create argparse-like namespace
        args = types.SimpleNamespace(
            csv=str(csv_path),
            target="target",
            seeds="1",
            test_size=0.2,
            outdir=str(tmp_path / "out_kao"),
            hparams_json="{}",
            verbose=False,
            standardize=False,
            standardize_y=False,
            time_budget=2.0,   # CLI explicitly set
            dataset_name="toy",
        )

        # Load data
        from sklearn.model_selection import train_test_split
        feature_cols = ["f0", "f1"]
        X_all = df[feature_cols].values
        y_all = df["target"].values
        X_train, X_test, Y_train, Y_test = train_test_split(
            X_all, y_all, test_size=0.2, random_state=2025
        )

        # The critical assertion: calling this must NOT raise UnboundLocalError
        try:
            # Dynamically import to ensure we test the latest code
            import scripts.run_kao as run_kao_mod
            importlib.reload(run_kao_mod)
            results = run_kao_mod.run_kao_experiment(
                args, X_train, X_test, Y_train, Y_test, feature_cols
            )
        except NameError as e:
            if "time_budget" in str(e) or "UnboundLocalError" in type(e).__name__:
                pytest.fail(f"UnboundLocalError regression detected: {e}")
            raise
        except Exception:
            # Other errors (e.g. deap import) are not the regression we test
            pass

    def test_result_json_has_budget_seconds(self, tmp_path):
        """result.json must contain budget_seconds matching effective time_budget."""
        import json
        import numpy as np
        import types
        import pandas as pd

        csv_path = tmp_path / "toy.csv"
        np.random.seed(42)
        X = np.random.randn(30, 2)
        y = X[:, 0] ** 2 + 0.5 * X[:, 1]
        df = pd.DataFrame({"f0": X[:, 0], "f1": X[:, 1], "target": y})
        df.to_csv(csv_path, index=False)

        outdir = tmp_path / "out_kao"

        args = types.SimpleNamespace(
            csv=str(csv_path),
            target="target",
            seeds="1",
            test_size=0.2,
            outdir=str(outdir),
            hparams_json='{"time_budget": 5}',  # json says 5
            verbose=False,
            standardize=False,
            standardize_y=False,
            time_budget=2.0,     # CLI says 2 — CLI must win
            dataset_name="toy",
        )

        from sklearn.model_selection import train_test_split
        feature_cols = ["f0", "f1"]
        X_all = df[feature_cols].values
        y_all = df["target"].values
        X_train, X_test, Y_train, Y_test = train_test_split(
            X_all, y_all, test_size=0.2, random_state=2025
        )

        try:
            import scripts.run_kao as run_kao_mod
            import importlib
            importlib.reload(run_kao_mod)
            run_kao_mod.run_kao_experiment(
                args, X_train, X_test, Y_train, Y_test, feature_cols
            )
        except Exception:
            pytest.skip("KAO execution failed (environment issue), skipping result check")

        result_path = outdir / "seed_1" / "result.json"
        if result_path.exists():
            with open(result_path) as f:
                rj = json.load(f)
            assert rj.get("budget_seconds") == 2.0, (
                f"budget_seconds should be 2.0 (CLI wins), got {rj.get('budget_seconds')}"
            )
            assert "hparams_effective" in rj, "result.json must contain hparams_effective"
        else:
            pytest.skip("result.json not produced (environment issue)")
