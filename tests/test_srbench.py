# -*- coding: utf-8 -*-
"""SRBench data generation tests."""
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_all_benchmarks_generate():
    """Every benchmark should produce correct shapes with no NaN/Inf."""
    from data.srbench_synthetic.generate import BENCHMARKS, generate_dataset

    for key, bench in BENCHMARKS.items():
        X_tr, y_tr, X_te, y_te = generate_dataset(bench)
        assert X_tr.shape == (bench.n_train, bench.n_vars), (
            f"{key}: X_train shape {X_tr.shape} != ({bench.n_train}, {bench.n_vars})"
        )
        assert X_te.shape == (bench.n_test, bench.n_vars), (
            f"{key}: X_test shape {X_te.shape} != ({bench.n_test}, {bench.n_vars})"
        )
        assert len(y_tr) == bench.n_train
        assert len(y_te) == bench.n_test
        assert not np.any(np.isnan(y_tr)), f"{key}: y_train has NaN"
        assert not np.any(np.isinf(y_tr)), f"{key}: y_train has Inf"
        assert not np.any(np.isnan(y_te)), f"{key}: y_test has NaN"
        assert not np.any(np.isinf(y_te)), f"{key}: y_test has Inf"


def test_noise_affects_training_only():
    """Noise should change y_train but not y_test."""
    from data.srbench_synthetic.generate import BENCHMARKS, generate_dataset

    bench = BENCHMARKS["nguyen_1"]
    _, y0, _, yt0 = generate_dataset(bench, seed=42, noise_std=0.0)
    _, yn, _, ytn = generate_dataset(bench, seed=42, noise_std=1.0)

    # Test sets should be identical (same seed, no noise on test)
    np.testing.assert_array_equal(yt0, ytn)
    # Training sets should differ due to noise
    assert not np.allclose(y0, yn), "noise_std=1.0 should change y_train"


def test_seed_reproducibility():
    """Same seed should produce identical datasets."""
    from data.srbench_synthetic.generate import BENCHMARKS, generate_dataset

    bench = BENCHMARKS["vladislavleva_4"]
    X1, y1, Xt1, yt1 = generate_dataset(bench, seed=99)
    X2, y2, Xt2, yt2 = generate_dataset(bench, seed=99)
    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)


def test_symbolic_recovery_trivial():
    """check_symbolic_recovery should match identical expressions."""
    from data.srbench_synthetic.generate import check_symbolic_recovery

    assert check_symbolic_recovery("x0**2 + x0", "x0**2 + x0") is True


def test_benchmark_count():
    """There should be exactly 5 benchmarks."""
    from data.srbench_synthetic.generate import BENCHMARKS

    assert len(BENCHMARKS) == 5
