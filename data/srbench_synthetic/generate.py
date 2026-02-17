#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SRBench Standard Synthetic Benchmarks
======================================
Five benchmark functions spanning distinct structural families:

  Nguyen-1          polynomial          x0^3 + x0^2 + x0
  Nguyen-7          transcendental      log(x0+1) + log(x0^2+1)
  Keijzer-6         discrete sum        sum(1/i, i=1..x)
  Vladislavleva-4   rational multivar   10/(5+sum((xi-3)^2, i=0..4))
  Korns-12          trig high-dim       2 - 2.1*cos(9.8*x0)*sin(1.3*x4)

Each benchmark is defined as a SRBenchmark dataclass with:
  - ground truth expression (for symbolic recovery check)
  - domain, number of variables, default sample sizes

Usage:
  python data/srbench_synthetic/generate.py
  # Prints shape and y-range for each benchmark
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Tuple, Optional


@dataclass
class SRBenchmark:
    name: str
    fn: Callable                # f(X) -> y, X shape = (n, n_vars)
    ground_truth: str           # true expression (for symbolic recovery check)
    n_vars: int
    domain: Tuple[float, float] # (low, high) uniform sampling
    n_train: int = 100
    n_test: int = 100


BENCHMARKS = {
    "nguyen_1": SRBenchmark(
        name="Nguyen-1",
        fn=lambda X: X[:, 0] ** 3 + X[:, 0] ** 2 + X[:, 0],
        ground_truth="x0**3 + x0**2 + x0",
        n_vars=1,
        domain=(-1, 1),
    ),
    "nguyen_7": SRBenchmark(
        name="Nguyen-7",
        fn=lambda X: np.log(X[:, 0] + 1) + np.log(X[:, 0] ** 2 + 1),
        ground_truth="log(x0+1) + log(x0**2+1)",
        n_vars=1,
        domain=(0, 2),
    ),
    "keijzer_6": SRBenchmark(
        name="Keijzer-6",
        fn=lambda X: np.array(
            [sum(1.0 / j for j in range(1, int(x) + 1)) for x in X[:, 0]]
        ),
        ground_truth="sum(1/i, i=1..x)",
        n_vars=1,
        domain=(1, 50),
        n_train=50,
        n_test=50,
    ),
    "vladislavleva_4": SRBenchmark(
        name="Vladislavleva-4",
        fn=lambda X: 10.0 / (5.0 + np.sum((X[:, :5] - 3) ** 2, axis=1)),
        ground_truth="10/(5+sum((xi-3)^2, i=0..4))",
        n_vars=5,
        domain=(0.05, 6.05),
        n_train=1024,
        n_test=5000,
    ),
    "korns_12": SRBenchmark(
        name="Korns-12",
        fn=lambda X: 2.0 - 2.1 * np.cos(9.8 * X[:, 0]) * np.sin(1.3 * X[:, 4]),
        ground_truth="2 - 2.1*cos(9.8*x0)*sin(1.3*x4)",
        n_vars=5,
        domain=(-50, 50),
        n_train=10000,
        n_test=10000,
    ),
}


def generate_dataset(
    bench: SRBenchmark,
    seed: int = 42,
    noise_std: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate train/test arrays for a single benchmark.

    Parameters
    ----------
    bench : SRBenchmark
    seed : int
    noise_std : float  Gaussian noise added to training targets only.

    Returns
    -------
    X_train, y_train, X_test, y_test
    """
    rng = np.random.RandomState(seed)
    X_train = rng.uniform(
        bench.domain[0], bench.domain[1], size=(bench.n_train, bench.n_vars)
    )
    X_test = rng.uniform(
        bench.domain[0], bench.domain[1], size=(bench.n_test, bench.n_vars)
    )
    y_train = bench.fn(X_train) + noise_std * rng.randn(bench.n_train)
    y_test = bench.fn(X_test)  # test set is noise-free
    return X_train, y_train, X_test, y_test


def check_symbolic_recovery(
    predicted_expr: str,
    ground_truth: str,
    tolerance: float = 1e-10,
) -> bool:
    """Use sympy to check whether *predicted_expr* is equivalent to *ground_truth*.

    Returns False (rather than raising) when sympy is unavailable or the
    expressions cannot be compared symbolically.
    """
    try:
        import sympy
        diff = sympy.simplify(
            sympy.sympify(predicted_expr) - sympy.sympify(ground_truth)
        )
        if diff.is_number:
            return abs(float(diff)) < tolerance
        return diff == 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# CLI quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    for key, bench in BENCHMARKS.items():
        X_tr, y_tr, X_te, y_te = generate_dataset(bench)
        print(
            f"{bench.name}: train={X_tr.shape}, test={X_te.shape}, "
            f"y range=[{y_tr.min():.3f}, {y_tr.max():.3f}]"
        )
