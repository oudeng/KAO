# -*- coding: utf-8 -*-
"""Baseline method tests (small data, 10s budget)."""
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def toy_data():
    rng = np.random.RandomState(42)
    X = rng.randn(200, 3)
    y = X[:, 0] ** 2 + 0.5 * X[:, 1]
    return X[:160], y[:160], X[160:], y[160:]


@pytest.mark.parametrize("name", ["PySR", "RILS-ROLS", "gplearn"])
def test_baseline_runs(toy_data, name):
    """Each core baseline should return a dict with expression + complexity."""
    from baselines.registry import get_baseline_by_name

    X_tr, y_tr, X_te, y_te = toy_data
    b = get_baseline_by_name(name)
    result = b.fit(X_tr, y_tr, X_te, y_te, time_budget=10, random_state=42)
    assert "expression" in result, f"{name}: missing 'expression' key"
    assert "complexity" in result, f"{name}: missing 'complexity' key"


@pytest.mark.parametrize("name", ["PySR", "RILS-ROLS", "gplearn"])
def test_baseline_predictions(toy_data, name):
    """Predictions should have correct shape and be finite."""
    from baselines.registry import get_baseline_by_name

    X_tr, y_tr, X_te, y_te = toy_data
    b = get_baseline_by_name(name)
    result = b.fit(X_tr, y_tr, X_te, y_te, time_budget=10, random_state=42)
    y_pred = result.get("y_pred_train")
    assert y_pred is not None, f"{name}: missing y_pred_train"
    assert y_pred.shape == y_tr.shape, f"{name}: shape mismatch"


def test_baseline_registry_list():
    """get_all_baselines should return a non-empty list."""
    from baselines.registry import get_all_baselines

    baselines = get_all_baselines(include_optional=False)
    assert len(baselines) >= 3, f"Expected >= 3 baselines, got {len(baselines)}"
    names = [b.name for b in baselines]
    assert "PySR" in names
    assert "RILS-ROLS" in names
    assert "gplearn" in names


def test_baseline_unknown_raises():
    """Requesting a non-existent baseline should raise ValueError."""
    from baselines.registry import get_baseline_by_name

    with pytest.raises(ValueError, match="Unknown baseline"):
        get_baseline_by_name("NoSuchMethod")


def test_operon_optional(toy_data):
    """Operon should work if pyoperon is installed, skip otherwise."""
    try:
        from baselines.operon_wrapper import OperonSR, OPERON_AVAILABLE

        if not OPERON_AVAILABLE:
            pytest.skip("pyoperon not installed")
    except ImportError:
        pytest.skip("pyoperon not installed")

    X_tr, y_tr, X_te, y_te = toy_data
    b = OperonSR()
    result = b.fit(X_tr, y_tr, X_te, y_te, time_budget=10, random_state=42)
    assert result["complexity"] > 0
