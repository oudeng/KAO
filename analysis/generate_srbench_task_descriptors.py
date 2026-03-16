#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SRBench Task Descriptor Generator & 10-Benchmark Selection
============================================================

Phase 6A deliverable:
1. Define ALL candidate SRBench benchmarks with ground-truth formulas
2. Generate task descriptors (family_tag, opset_required, n_vars, etc.)
3. Run stratified coverage selection to pick 10 benchmarks
4. Output: outputs/csv/srbench_task_descriptors.csv + selection report

References for ground-truth formulas:
  - La Cava et al. (2021) "Contemporary Symbolic Regression Methods..."
  - Nguyen et al. (2011) "Semantics based crossover in GP: the case for real-valued function regression"
  - Keijzer (2003) "Improving symbolic regression with interval arithmetic and linear scaling"
  - Vladislavleva et al. (2009) "Order of nonlinearity as a complexity measure..."
  - Pagie & Hogeweg (1997) "Evolutionary consequences of coevolving targets"
  - Korns (2011) "Accuracy in symbolic regression"
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import re
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple
import os
import sys

# ──────────────────────────────────────────────────────────────
# 1. FULL CANDIDATE BENCHMARK DEFINITIONS
# ──────────────────────────────────────────────────────────────

@dataclass
class BenchmarkInfo:
    """Minimal info for task profiling (no callable needed here)."""
    name: str
    formula: str
    n_vars: int
    domain: Tuple[float, float]
    n_train: int = 100
    n_test: int = 100


# All candidates — superset of existing 5 + potential additions
ALL_CANDIDATES: Dict[str, BenchmarkInfo] = {
    # ── Existing 5 (fixed) ──
    "nguyen_1": BenchmarkInfo(
        name="Nguyen-1",
        formula="x0**3 + x0**2 + x0",
        n_vars=1,
        domain=(-1, 1),
    ),
    "nguyen_7": BenchmarkInfo(
        name="Nguyen-7",
        formula="log(x0+1) + log(x0**2+1)",
        n_vars=1,
        domain=(0, 2),
    ),
    "keijzer_6": BenchmarkInfo(
        name="Keijzer-6",
        formula="sum(1/i, i=1..x)",
        n_vars=1,
        domain=(1, 50),
        n_train=50,
        n_test=50,
    ),
    "vladislavleva_4": BenchmarkInfo(
        name="Vladislavleva-4",
        formula="10/(5+sum((xi-3)**2, i=0..4))",
        n_vars=5,
        domain=(0.05, 6.05),
        n_train=1024,
        n_test=5000,
    ),
    "korns_12": BenchmarkInfo(
        name="Korns-12",
        formula="2 - 2.1*cos(9.8*x0)*sin(1.3*x4)",
        n_vars=5,
        domain=(-50, 50),
        n_train=10000,
        n_test=10000,
    ),

    # ── Candidate additions ──
    "nguyen_2": BenchmarkInfo(
        name="Nguyen-2",
        formula="x0**4 + x0**3 + x0**2 + x0",
        n_vars=1,
        domain=(-1, 1),
    ),
    "nguyen_3": BenchmarkInfo(
        name="Nguyen-3",
        formula="x0**5 + x0**4 + x0**3 + x0**2 + x0",
        n_vars=1,
        domain=(-1, 1),
    ),
    "nguyen_5": BenchmarkInfo(
        name="Nguyen-5",
        formula="sin(x0**2)*cos(x0) - 1",
        n_vars=1,
        domain=(-1, 1),
    ),
    "nguyen_6": BenchmarkInfo(
        name="Nguyen-6",
        formula="sin(x0) + sin(x0 + x0**2)",
        n_vars=1,
        domain=(-1, 1),
    ),
    "nguyen_9": BenchmarkInfo(
        name="Nguyen-9",
        formula="sin(x0) + sin(x1**2)",
        n_vars=2,
        domain=(0, 1),
    ),
    "nguyen_10": BenchmarkInfo(
        name="Nguyen-10",
        formula="2*sin(x0)*cos(x1)",
        n_vars=2,
        domain=(0, 1),
    ),
    "pagie_1": BenchmarkInfo(
        name="Pagie-1",
        formula="1/(1+x0**(-4)) + 1/(1+x1**(-4))",
        n_vars=2,
        domain=(-5, 5),
        n_train=676,   # 26x26 grid is standard, but uniform random also common
        n_test=10000,
    ),
    "vladislavleva_1": BenchmarkInfo(
        name="Vladislavleva-1",
        formula="exp(-(x0-1)**2) / (1.2 + (x1-2.5)**2)",
        n_vars=2,
        domain=(0.3, 4.0),
        n_train=100,
        n_test=221,
    ),
    "keijzer_1": BenchmarkInfo(
        name="Keijzer-1",
        formula="0.3*x0*sin(2*3.14159265*x0)",
        n_vars=1,
        domain=(-1, 1),
        n_train=20,    # Keijzer-1 uses E[-1,1,0.1] → 21 points
        n_test=100,
    ),
    "keijzer_4": BenchmarkInfo(
        name="Keijzer-4",
        formula="x0**3*exp(-x0)*cos(x0)*sin(x0)*(sin(x0)**2*cos(x0)-1)",
        n_vars=1,
        domain=(0, 10),
        n_train=200,
        n_test=200,
    ),
    "korns_1": BenchmarkInfo(
        name="Korns-1",
        formula="1.57 + 24.3*x3",
        n_vars=5,
        domain=(-50, 50),
        n_train=10000,
        n_test=10000,
    ),
}


# ──────────────────────────────────────────────────────────────
# 2. TASK PROFILING
# ──────────────────────────────────────────────────────────────

def analyze_formula(formula: str) -> Dict:
    """Parse a formula string to extract operator set and family tags."""
    opset = set()
    family_tags = set()

    # Detect trig
    if re.search(r'\b(sin|cos|tan)\b', formula):
        opset.update(['sin', 'cos'] if 'sin' in formula and 'cos' in formula
                     else ['sin'] if 'sin' in formula
                     else ['cos'] if 'cos' in formula
                     else ['tan'])
        family_tags.add('trig')

    # Detect exp/log
    if re.search(r'\b(exp)\b', formula):
        opset.add('exp')
        family_tags.add('exp_log')
    if re.search(r'\b(log|ln)\b', formula):
        opset.add('log')
        family_tags.add('exp_log')

    # Detect sqrt/abs
    if re.search(r'\b(sqrt|abs)\b', formula):
        opset.add('sqrt' if 'sqrt' in formula else 'abs')
        family_tags.add('sqrt_abs')

    # Detect rational (division or negative power)
    if '/' in formula or '**-' in formula or '**(-' in formula:
        opset.add('div')
        family_tags.add('rational')

    # Detect polynomial (only +, -, *, ** with positive integer exponents)
    if re.search(r'\*\*\d', formula) or (
        not family_tags and re.search(r'[+\-\*]', formula)
    ):
        opset.update(['+', '-', '*', '**'])
        family_tags.add('poly')

    # If still no tags (e.g., simple linear), mark as poly
    if not family_tags:
        family_tags.add('poly')
        opset.update(['+', '*'])

    # Detect sum notation (Keijzer-6 style)
    if 'sum(' in formula:
        family_tags.add('rational')  # harmonic sum involves 1/i
        opset.add('div')

    return {
        'opset_required': '+'.join(sorted(opset)),
        'family_tags': '+'.join(sorted(family_tags)),
    }


def compute_output_stats(bench: BenchmarkInfo) -> Dict:
    """Compute y statistics on noise-free data."""
    rng = np.random.RandomState(2025)
    X = rng.uniform(bench.domain[0], bench.domain[1],
                    size=(max(bench.n_train, 1000), bench.n_vars))

    # Need to compute y — reimport functions for candidates
    y = _eval_formula(bench, X)
    if y is None:
        return {'y_mean': np.nan, 'y_std': np.nan, 'y_range': np.nan}

    valid = np.isfinite(y)
    if valid.sum() < 10:
        return {'y_mean': np.nan, 'y_std': np.nan, 'y_range': np.nan}

    y_valid = y[valid]
    return {
        'y_mean': float(np.mean(y_valid)),
        'y_std': float(np.std(y_valid)),
        'y_range': float(np.ptp(y_valid)),
    }


def _eval_formula(bench: BenchmarkInfo, X: np.ndarray) -> np.ndarray:
    """Evaluate benchmark formula numerically."""
    FN_MAP = {
        "nguyen_1": lambda X: X[:, 0]**3 + X[:, 0]**2 + X[:, 0],
        "nguyen_2": lambda X: X[:, 0]**4 + X[:, 0]**3 + X[:, 0]**2 + X[:, 0],
        "nguyen_3": lambda X: X[:, 0]**5 + X[:, 0]**4 + X[:, 0]**3 + X[:, 0]**2 + X[:, 0],
        "nguyen_5": lambda X: np.sin(X[:, 0]**2)*np.cos(X[:, 0]) - 1,
        "nguyen_6": lambda X: np.sin(X[:, 0]) + np.sin(X[:, 0] + X[:, 0]**2),
        "nguyen_7": lambda X: np.log(X[:, 0]+1) + np.log(X[:, 0]**2+1),
        "nguyen_9": lambda X: np.sin(X[:, 0]) + np.sin(X[:, 1]**2),
        "nguyen_10": lambda X: 2*np.sin(X[:, 0])*np.cos(X[:, 1]),
        "keijzer_1": lambda X: 0.3*X[:, 0]*np.sin(2*np.pi*X[:, 0]),
        "keijzer_4": lambda X: X[:, 0]**3*np.exp(-X[:, 0])*np.cos(X[:, 0])*np.sin(X[:, 0])*(np.sin(X[:, 0])**2*np.cos(X[:, 0])-1),
        "keijzer_6": lambda X: np.array([sum(1.0/j for j in range(1, max(1, int(x))+1)) for x in X[:, 0]]),
        "pagie_1": lambda X: 1.0/(1.0+X[:, 0]**(-4)) + 1.0/(1.0+X[:, 1]**(-4)),
        "vladislavleva_1": lambda X: np.exp(-(X[:, 0]-1)**2) / (1.2 + (X[:, 1]-2.5)**2),
        "vladislavleva_4": lambda X: 10.0/(5.0+np.sum((X[:, :5]-3)**2, axis=1)),
        "korns_1": lambda X: 1.57 + 24.3*X[:, 3],
        "korns_12": lambda X: 2.0 - 2.1*np.cos(9.8*X[:, 0])*np.sin(1.3*X[:, 4]),
    }

    key = bench.name.lower().replace("-", "_")
    if key in FN_MAP:
        try:
            return FN_MAP[key](X)
        except Exception:
            return None
    return None


def kao_can_handle(family_tags: str) -> Tuple[bool, str]:
    """Check if KAO's typed operator set can natively handle this benchmark.

    KAO has: +, -, *, /, quadratic warp (degree-2 polynomial compositions).
    It does NOT have: sin, cos, exp, log, sqrt, abs, etc.
    """
    tags = set(family_tags.split('+'))
    kao_native = {'poly', 'rational'}  # KAO can handle polynomial & rational
    missing = tags - kao_native
    if not missing:
        return True, 'none'
    return False, '+'.join(sorted(missing))


# ──────────────────────────────────────────────────────────────
# 3. STRATIFIED COVERAGE SELECTION
# ──────────────────────────────────────────────────────────────

FIXED_KEYS = ["nguyen_1", "nguyen_7", "keijzer_6", "korns_12", "vladislavleva_4"]

def select_10_benchmarks(descriptors: pd.DataFrame) -> List[str]:
    """
    Select 10 benchmarks using stratified coverage + diversity-aware greedy fill.

    Hard coverage requirements (across all 10):
      - poly >= 2
      - rational >= 2
      - trig or exp_log >= 2
      - multivar (n_vars >= 2) >= 2
      - high_dim (n_vars >= 5) >= 1

    Enhanced diversity requirements (beyond hard constraints):
      - At least 1 true 2D benchmark (n_vars == 2)  [fixed set only has 5D]
      - No more than 2 additional poly-only (avoid Nguyen-{2,3,...} overload)
      - At least 1 benchmark from a non-Nguyen family in the new 5
      - Scale diversity: include benchmarks with very different y_range

    Tie-breaking: prefer benchmarks that bring new structural diversity,
    then classic SR families (Nguyen > Keijzer > Pagie > Vladislavleva > Korns).
    """
    selected = list(FIXED_KEYS)
    candidates = [k for k in descriptors.index if k not in FIXED_KEYS]

    # --- Helper functions ---
    def count_coverage(keys):
        poly = sum(1 for k in keys if 'poly' in descriptors.loc[k, 'family_tags'])
        rational = sum(1 for k in keys if 'rational' in descriptors.loc[k, 'family_tags'])
        trig_explog = sum(1 for k in keys
                         if 'trig' in descriptors.loc[k, 'family_tags']
                         or 'exp_log' in descriptors.loc[k, 'family_tags'])
        multivar = sum(1 for k in keys if descriptors.loc[k, 'n_vars'] >= 2)
        two_d = sum(1 for k in keys if descriptors.loc[k, 'n_vars'] == 2)
        high_dim = sum(1 for k in keys if descriptors.loc[k, 'n_vars'] >= 5)
        return {
            'poly': poly, 'rational': rational,
            'trig_or_explog': trig_explog,
            'multivar': multivar, 'two_d': two_d,
            'high_dim': high_dim,
        }

    # Extended requirements: original + diversity targets
    requirements = {
        'poly': 2, 'rational': 2, 'trig_or_explog': 2,
        'multivar': 2, 'two_d': 1, 'high_dim': 1,
    }

    family_priority = {
        'nguyen': 1, 'keijzer': 2, 'pagie': 3,
        'vladislavleva': 4, 'korns': 5,
    }

    def get_priority(key):
        for fam, pri in family_priority.items():
            if fam in key:
                return pri
        return 99

    def count_new_poly_only(keys):
        """Count new (non-fixed) benchmarks that are poly-only."""
        return sum(1 for k in keys if k not in FIXED_KEYS
                   and descriptors.loc[k, 'family_tags'] == 'poly')

    def has_non_nguyen_new(keys):
        """Check if any new benchmark is not from Nguyen family."""
        return any('nguyen' not in k for k in keys if k not in FIXED_KEYS)

    # --- Greedy selection with diversity scoring ---
    for round_num in range(5):
        if not candidates:
            break

        current_cov = count_coverage(selected)
        gaps = {dim: max(0, req - current_cov[dim])
                for dim, req in requirements.items()}
        total_gap = sum(gaps.values())

        best_key = None
        best_score = (-999,)  # multi-criteria tuple

        for c in candidates:
            row = descriptors.loc[c]

            # 1. Hard constraint gap reduction
            gain = 0
            if gaps['poly'] > 0 and 'poly' in row['family_tags']:
                gain += 1
            if gaps['rational'] > 0 and 'rational' in row['family_tags']:
                gain += 1
            if gaps['trig_or_explog'] > 0 and (
                'trig' in row['family_tags'] or 'exp_log' in row['family_tags']
            ):
                gain += 1
            if gaps['multivar'] > 0 and row['n_vars'] >= 2:
                gain += 1
            if gaps['two_d'] > 0 and row['n_vars'] == 2:
                gain += 1
            if gaps['high_dim'] > 0 and row['n_vars'] >= 5:
                gain += 1

            # 2. Diversity bonus (counted when gap=0, to encourage diversity)
            diversity = 0

            # Bonus for 2D (not just 5D) — we want true 2D benchmarks
            if row['n_vars'] == 2:
                diversity += 2

            # Bonus for multi-tag families (more structurally interesting)
            n_tags = len(row['family_tags'].split('+'))
            diversity += n_tags - 1  # 0 for single-tag, 1 for 2-tag, etc.

            # Bonus for non-Nguyen if we don't have one yet
            if 'nguyen' not in c and not has_non_nguyen_new(selected):
                diversity += 3

            # Penalty for poly-only if we already have too many
            if row['family_tags'] == 'poly' and count_new_poly_only(selected) >= 1:
                diversity -= 2

            # Bonus for extreme scale (large y_range → scale challenge)
            if row['y_range'] > 100:
                diversity += 1

            # 3. Family priority (lower number = better)
            pri = get_priority(c)

            # Score tuple: (gap_gain, diversity, -priority)
            score = (gain, diversity, -pri)

            if score > best_score:
                best_score = score
                best_key = c

        if best_key is not None:
            selected.append(best_key)
            candidates.remove(best_key)

    return selected


# ──────────────────────────────────────────────────────────────
# 4. MAIN
# ──────────────────────────────────────────────────────────────

def main():
    # KAO/analysis/generate_srbench_task_descriptors.py → KAO/analysis → KAO → KAO_v3
    ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)
    )))  # -> KAO_v3

    # Build descriptors for all candidates
    rows = []
    for key, bench in ALL_CANDIDATES.items():
        analysis = analyze_formula(bench.formula)
        stats = compute_output_stats(bench)
        kao_ok, kao_missing = kao_can_handle(analysis['family_tags'])

        rows.append({
            'key': key,
            'benchmark': bench.name,
            'formula': bench.formula,
            'n_vars': bench.n_vars,
            'domain': f"[{bench.domain[0]}, {bench.domain[1]}]",
            'n_train': bench.n_train,
            'n_test': bench.n_test,
            'family_tags': analysis['family_tags'],
            'opset_required': analysis['opset_required'],
            'kao_has_ops': kao_ok,
            'kao_missing_ops': kao_missing,
            'y_mean': stats['y_mean'],
            'y_std': stats['y_std'],
            'y_range': stats['y_range'],
        })

    df = pd.DataFrame(rows).set_index('key')

    # Save full descriptors
    out_dir = os.path.join(ROOT, "outputs", "csv")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "srbench_task_descriptors.csv")
    df.to_csv(csv_path)
    print(f"[✓] Saved task descriptors → {csv_path}")
    print(f"    Total candidates: {len(df)}")
    print()

    # Print all candidates
    print("=" * 100)
    print("ALL CANDIDATE BENCHMARKS")
    print("=" * 100)
    for key, row in df.iterrows():
        fixed = "(FIXED)" if key in FIXED_KEYS else "(candidate)"
        print(f"  {row['benchmark']:20s} | n_vars={row['n_vars']} | "
              f"family={row['family_tags']:20s} | kao_native={row['kao_has_ops']} "
              f"| y_std={row['y_std']:.3f} | {fixed}")
    print()

    # Run selection
    selected = select_10_benchmarks(df)

    # Print selection result
    print("=" * 100)
    print("FINAL 10 BENCHMARK SELECTION")
    print("=" * 100)

    # Verify coverage
    def count_final(keys):
        poly = sum(1 for k in keys if 'poly' in df.loc[k, 'family_tags'])
        rational = sum(1 for k in keys if 'rational' in df.loc[k, 'family_tags'])
        trig = sum(1 for k in keys if 'trig' in df.loc[k, 'family_tags'])
        exp_log = sum(1 for k in keys if 'exp_log' in df.loc[k, 'family_tags'])
        trig_explog = sum(1 for k in keys
                         if 'trig' in df.loc[k, 'family_tags']
                         or 'exp_log' in df.loc[k, 'family_tags'])
        multivar = sum(1 for k in keys if df.loc[k, 'n_vars'] >= 2)
        high_dim = sum(1 for k in keys if df.loc[k, 'n_vars'] >= 5)
        return {
            'poly': poly,
            'rational': rational,
            'trig': trig,
            'exp_log': exp_log,
            'trig_or_explog': trig_explog,
            'multivar': multivar,
            'high_dim': high_dim,
        }

    for i, key in enumerate(selected, 1):
        row = df.loc[key]
        status = "FIXED" if key in FIXED_KEYS else "NEW"
        print(f"  {i:2d}. {row['benchmark']:20s} | n_vars={row['n_vars']} | "
              f"family={row['family_tags']:20s} | "
              f"kao_native={str(row['kao_has_ops']):5s} | "
              f"missing={row['kao_missing_ops']:15s} | "
              f"y_std={row['y_std']:8.3f} | [{status}]")

    cov = count_final(selected)
    print()
    print("COVERAGE VERIFICATION:")
    print(f"  poly >= 2:           {'✅' if cov['poly'] >= 2 else '❌'} ({cov['poly']})")
    print(f"  rational >= 2:       {'✅' if cov['rational'] >= 2 else '❌'} ({cov['rational']})")
    print(f"  trig_or_explog >= 2: {'✅' if cov['trig_or_explog'] >= 2 else '❌'} ({cov['trig_or_explog']})")
    print(f"    - trig: {cov['trig']}, exp_log: {cov['exp_log']}")
    print(f"  multivar >= 2:       {'✅' if cov['multivar'] >= 2 else '❌'} ({cov['multivar']})")
    print(f"  high_dim >= 1:       {'✅' if cov['high_dim'] >= 1 else '❌'} ({cov['high_dim']})")

    # Print selected keys for use in generate.py
    print()
    print("SELECTED KEYS (for generate.py):")
    print(f"  {selected}")

    # Print KAO boundary analysis
    print()
    print("KAO OPERATOR BOUNDARY ANALYSIS:")
    for key in selected:
        row = df.loc[key]
        if not row['kao_has_ops']:
            print(f"  ⚠ {row['benchmark']:20s} — KAO missing: {row['kao_missing_ops']}")
        else:
            print(f"  ✓ {row['benchmark']:20s} — KAO can natively handle")

    return selected, df


if __name__ == "__main__":
    selected, df = main()
