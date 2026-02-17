# -*- coding: utf-8 -*-
"""KAO core functionality tests (small data, 10s budget)."""
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def toy_data():
    rng = np.random.RandomState(42)
    X = rng.randn(200, 3)
    y = X[:, 0] ** 2 + 0.5 * X[:, 1] + 0.1 * X[:, 2]
    return X[:160], y[:160], X[160:], y[160:]


def test_kao_basic(toy_data):
    """run_single returns a KAOResult with an expression."""
    from kao.KAO_v3_1 import run_single

    X_tr, y_tr, X_te, y_te = toy_data
    result = run_single(X_tr, y_tr, X_te, y_te, time_budget=10, seed=42)
    assert hasattr(result, "expression"), "KAOResult must have 'expression'"
    assert isinstance(result.expression, str)
    assert len(result.expression) > 0


def test_kao_r2_is_finite(toy_data):
    """R^2 values should be finite numbers."""
    from kao.KAO_v3_1 import run_single

    X_tr, y_tr, X_te, y_te = toy_data
    result = run_single(X_tr, y_tr, X_te, y_te, time_budget=10, seed=42)
    assert np.isfinite(result.r2_test), f"r2_test is not finite: {result.r2_test}"
    assert np.isfinite(result.r2_cv), f"r2_cv is not finite: {result.r2_cv}"


def test_kao_no_kao_toggle(toy_data):
    """Both use_kao_leaf=True and False should complete without error."""
    from kao.KAO_v3_1 import run_single

    X_tr, y_tr, X_te, y_te = toy_data
    r1 = run_single(X_tr, y_tr, X_te, y_te, use_kao_leaf=True, time_budget=10, seed=1)
    r2 = run_single(X_tr, y_tr, X_te, y_te, use_kao_leaf=False, time_budget=10, seed=1)
    assert hasattr(r1, "expression")
    assert hasattr(r2, "expression")


def test_kao_complexity_cap(toy_data):
    """max_complexity should constrain the returned complexity_nodes."""
    from kao.KAO_v3_1 import run_single

    X_tr, y_tr, X_te, y_te = toy_data
    result = run_single(
        X_tr, y_tr, X_te, y_te, max_complexity=5, time_budget=10, seed=42,
    )
    assert result.complexity_nodes <= 5, (
        f"complexity_nodes={result.complexity_nodes} exceeds cap 5"
    )


def test_kao_time_checkpoints(toy_data):
    """pareto_snapshots should contain entries for requested checkpoints."""
    from kao.KAO_v3_1 import run_single

    X_tr, y_tr, X_te, y_te = toy_data
    result = run_single(
        X_tr, y_tr, X_te, y_te,
        time_checkpoints=[3, 6, 10],
        time_budget=10,
        seed=42,
    )
    assert isinstance(result.pareto_snapshots, dict)
    assert len(result.pareto_snapshots) > 0, "No pareto snapshots captured"


def test_kao_predictions_shape(toy_data):
    """Prediction arrays should match input sizes."""
    from kao.KAO_v3_1 import run_single

    X_tr, y_tr, X_te, y_te = toy_data
    result = run_single(X_tr, y_tr, X_te, y_te, time_budget=10, seed=42)
    assert result.y_pred_train.shape == y_tr.shape
    assert result.y_pred_test.shape == y_te.shape
