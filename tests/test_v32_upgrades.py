# -*- coding: utf-8 -*-
"""v3.2 upgrade-specific tests."""
import json
import pytest
import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def toy_data():
    rng = np.random.RandomState(42)
    X = rng.randn(200, 3)
    y = X[:, 0] ** 2 + 0.5 * X[:, 1] + 0.1 * X[:, 2]
    return X[:160], y[:160], X[160:], y[160:]


# -------------------------------------------------------------------
# 1. time_budget early stop
# -------------------------------------------------------------------
def test_time_budget_stops_early(toy_data):
    """KAO_Core with time_budget=2 should NOT run all 999 generations."""
    from kao.KAO_v3_1 import KAO_Core

    X_tr, y_tr, _, _ = toy_data
    core = KAO_Core(
        ['x0', 'x1', 'x2'], X_tr, y_tr,
        mu=100, ngen=999, time_budget=2, n_jobs=1,
    )
    core.run_evolution()
    assert core._generations_completed < 999, (
        f"Expected early stop but completed {core._generations_completed} generations"
    )
    assert core._evolution_runtime < 30, (
        f"Runtime {core._evolution_runtime:.1f}s is way over budget"
    )


# -------------------------------------------------------------------
# 2. result.json schema
# -------------------------------------------------------------------
def test_result_json_schema(tmp_path):
    """write_result_json produces a file with all required fields."""
    from utils.result_io import write_result_json

    payload = {
        "dataset": "test_ds",
        "method": "KAO",
        "seed": 1,
        "time_budget": 10,
        "budget_label": "10s",
        "r2_test": 0.8,
        "r2_cv": 0.75,
        "test_mse": 0.12,
        "cv_loss": 0.15,
        "instability": 0.05,
        "complexity_nodes": 10,
        "complexity_chars": 50,
        "runtime": 9.5,
        "expression": "x0 + x1",
        "extra": {"status": "ok"},
    }
    path = tmp_path / "result.json"
    write_result_json(path, payload)

    loaded = json.loads(path.read_text())
    required_keys = [
        "dataset", "method", "seed", "time_budget", "budget_label",
        "r2_test", "r2_cv", "test_mse", "cv_loss", "instability",
        "complexity_nodes", "complexity_chars", "runtime", "expression",
    ]
    for key in required_keys:
        assert key in loaded, f"Missing key: {key}"

    # numpy types should be converted
    assert isinstance(loaded["complexity_nodes"], int)
    assert isinstance(loaded["r2_test"], float)


def test_result_json_numpy_safety(tmp_path):
    """numpy types and NaN should serialize safely."""
    from utils.result_io import write_result_json

    payload = {
        "r2_test": np.float64(0.123),
        "complexity_nodes": np.int64(7),
        "bad_value": float("nan"),
        "array": np.array([1, 2, 3]),
    }
    path = tmp_path / "test.json"
    write_result_json(path, payload)
    loaded = json.loads(path.read_text())
    assert loaded["r2_test"] == pytest.approx(0.123)
    assert loaded["complexity_nodes"] == 7
    assert loaded["bad_value"] is None  # NaN -> None
    assert loaded["array"] == [1, 2, 3]


# -------------------------------------------------------------------
# 3. pairwise seed alignment (inner join)
# -------------------------------------------------------------------
def test_pairwise_seed_alignment():
    """Pairwise comparison should inner-join on seed, not truncate."""
    from analysis.statistics import StatisticalAnalyzer

    df = pd.DataFrame({
        'dataset': ['d'] * 6,
        'method': ['KAO'] * 3 + ['PySR'] * 3,
        'seed': [1, 2, 3, 2, 3, 4],
        'r2_test': [0.9, 0.8, 0.7, 0.1, 0.2, 0.3],
    })
    res = StatisticalAnalyzer.pairwise_comparison(df, 'KAO', 'PySR', 'd', 'r2_test')
    # Seeds in common: 2, 3
    assert res['n_pairs'] == 2, f"Expected 2 pairs, got {res['n_pairs']}"


# -------------------------------------------------------------------
# 4. cv_eval utility
# -------------------------------------------------------------------
def test_kfold_eval_fixed_model():
    """kfold_eval_fixed_model should return 3 finite floats."""
    from utils.cv_eval import kfold_eval_fixed_model

    rng = np.random.RandomState(0)
    X = rng.randn(100, 3)
    y = X[:, 0] + X[:, 1]
    predict_fn = lambda X_: X_[:, 0] + X_[:, 1]  # perfect model

    cv_loss, cv_r2, instability = kfold_eval_fixed_model(predict_fn, X, y)
    assert np.isfinite(cv_loss)
    assert np.isfinite(cv_r2)
    assert np.isfinite(instability)
    assert cv_loss < 1e-6, f"Perfect model should have ~0 loss, got {cv_loss}"
    assert cv_r2 > 0.99, f"Perfect model should have R2 ~1, got {cv_r2}"
