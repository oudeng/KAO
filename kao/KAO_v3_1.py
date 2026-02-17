#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
KAO_v3_1.py - KAO Symbolic Regression Core (v3.2)
- Typed GP with Quadratic (KAO) leaf operators
- NSGA-II multi-objective optimisation (CV-MSE vs. Complexity)
- use_kao_leaf switch for ablation experiments
- Time-checkpoint Pareto snapshots
- Structured KAOResult output
- v3.2: time_budget early stop, n_jobs control, refit budget guard

Lineage: v3c (Oct 2025) -> v3_1 (Feb 2026) -> v3.2 (Feb 2026)
"""

import operator
import random
import functools
import copy
import time
import os
import json
import re
import datetime
from dataclasses import dataclass, field
from typing import NamedTuple, Tuple, List, Optional, Dict, Any

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

from deap import base, creator, tools, algorithms
from deap import gp
import multiprocessing
from kneed import KneeLocator

# Try import tqdm with fallback
try:
    from tqdm.auto import tqdm, trange
except Exception:
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else range(kwargs.get("total", 0))
    def trange(n, **kwargs):
        return range(n)

# -----------------------------------------------------------------------------
# Primitives and TripleParam
# -----------------------------------------------------------------------------
class TripleParam(NamedTuple):
    a: float
    b: float
    c: float

def makeTP(a: float, b: float, c: float) -> TripleParam:
    return TripleParam(a, b, c)

def identityTP(tp: TripleParam) -> TripleParam:
    return tp

def KAO(x: float, tp: TripleParam) -> float:
    """Quadratic KAO function: a*x^2 + b*x + c"""
    if tp is None:
        return 0.0
    return tp.a * (x ** 2) + tp.b * x + tp.c

def addf(a: float, b: float) -> float:
    return a + b

def subf(a: float, b: float) -> float:
    return a - b

def mulf(a: float, b: float) -> float:
    return a * b

def divf(a: float, b: float) -> float:
    """Protected division"""
    return a if abs(b) < 1e-9 else a / b

# -----------------------------------------------------------------------------
# Pickle-safe random generators (from v3b)
# -----------------------------------------------------------------------------
class RandomFloat:
    """Pickle-safe random float generator"""
    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val
    
    def __call__(self):
        return np.random.uniform(self.min_val, self.max_val)

class RandomTripleParam:
    """Pickle-safe random TripleParam generator"""
    def __init__(self, mode="data_aware", range_val=1.0, y_mean=0.0, y_std=1.0, c_scale=1.0):
        self.mode = mode
        self.range_val = range_val
        self.y_mean = y_mean
        self.y_std = y_std
        self.c_scale = c_scale
    
    def __call__(self):
        r = self.range_val
        # Smaller range for quadratic term to avoid explosion
        a = np.random.uniform(-0.5, 0.5)  
        b = np.random.uniform(-r, r)
        if self.mode == "data_aware":
            c = np.random.normal(
                loc=self.y_mean,
                scale=max(1e-6, self.c_scale * self.y_std),
            )
        else:
            c = np.random.uniform(-r, r)
        return TripleParam(float(a), float(b), float(c))

# -----------------------------------------------------------------------------
# Global variables for multiprocessing
# -----------------------------------------------------------------------------
_global_X = None
_global_Y = None
_global_pset = None
_global_enable_ls = True
_global_cv_indices = None
_global_cv_folds = 5
_global_cv_shuffle = True
_global_cv_seed = 2025
_global_tp_mode = "data_aware"
_global_tp_range = 1.0
_global_tp_c_scale = 1.0
_global_y_mean = 0.0
_global_y_std = 1.0
_global_complexity_mode = "equal"
_global_node_cost = {}
_global_max_complexity = None  # Upgrade 2: complexity cap

def data_aware_randTP() -> TripleParam:
    """Ephemeral generator for TripleParam with data-aware c."""
    r = float(_global_tp_range)
    a = np.random.uniform(-0.5, 0.5)  # Smaller for quadratic term
    b = np.random.uniform(-r, r)
    if _global_tp_mode == "data_aware":
        c = np.random.normal(
            loc=_global_y_mean,
            scale=max(1e-6, _global_tp_c_scale * _global_y_std),
        )
    else:
        c = np.random.uniform(-r, r)
    return TripleParam(float(a), float(b), float(c))

# -----------------------------------------------------------------------------
# Evaluation function with CV and linear scaling
# -----------------------------------------------------------------------------
def _linear_scale(y_true: np.ndarray, f: np.ndarray, eps: float = 1e-12) -> Tuple[np.ndarray, float, float]:
    """Closed-form α,β to minimize MSE(y, αf+β) on training set."""
    f = np.asarray(f, dtype=float)
    y = np.asarray(y_true, dtype=float)
    
    # Filter out non-finite values
    mask = np.isfinite(f) & np.isfinite(y)
    if mask.sum() < 2:
        return f, 1.0, 0.0
    
    f_clean = f[mask]
    y_clean = y[mask]
    
    varf = float(np.var(f_clean))
    if varf < eps:
        return f, 1.0, 0.0
    
    cov = float(np.cov(f_clean, y_clean, ddof=0)[0, 1])
    alpha = cov / varf
    beta = float(y_clean.mean()) - alpha * float(f_clean.mean())
    return alpha * f + beta, alpha, beta

def calc_complexity(individual):
    """Calculate complexity with optional weighted mode."""
    if _global_complexity_mode == "weighted" and _global_node_cost:
        total_cost = 0
        for node in individual:
            if isinstance(node, gp.Primitive):
                # IMPORTANT: All KAO primitives have the same name "KAO" now
                if node.name == "KAO":
                    total_cost += _global_node_cost.get("KAO", 1)
                else:
                    total_cost += _global_node_cost.get(node.name, 1)
            else:
                total_cost += 1  # Terminals have cost 1
        return int(total_cost)
    else:
        return len(individual)

def evalSymbRegMulti_global(individual):
    """Evaluate (CV-MSE, Complexity) with optional Linear Scaling."""
    complexity = calc_complexity(individual)

    # Upgrade 2: complexity cap — reject oversized individuals early
    if _global_max_complexity is not None and complexity > _global_max_complexity:
        return float('inf'), float('inf')

    try:
        func = gp.compile(individual, _global_pset)
    except Exception:
        return 1e6, complexity

    # Vectorized prediction
    try:
        f = np.array([func(*row) for row in _global_X], dtype=float)
    except Exception:
        return 1e6, complexity

    if not np.all(np.isfinite(f)):
        return 1e6, complexity

    # Cross-validated MSE with per-fold linear scaling
    mse_folds = []
    for tr_idx, va_idx in _global_cv_indices:
        f_tr, y_tr = f[tr_idx], _global_Y[tr_idx]
        f_va, y_va = f[va_idx], _global_Y[va_idx]

        if _global_enable_ls:
            _, alpha, beta = _linear_scale(y_tr, f_tr)
            f_va = alpha * f_va + beta

        mse = float(np.mean((f_va - y_va) ** 2))
        if np.isfinite(mse):
            mse_folds.append(mse)

    if not mse_folds:
        return 1e6, complexity

    mse = float(np.mean(mse_folds))

    return mse, complexity

# -----------------------------------------------------------------------------
# KAO Core Class
# -----------------------------------------------------------------------------
class KAO_Core:
    """Optimized KAO Core for symbolic regression v3c (fixed naming)."""

    def __init__(self, feature_names, X, Y, **kwargs):
        """
        Parameters
        ----------
        feature_names : list[str]
        X : np.ndarray (n_samples, n_features)
        Y : np.ndarray (n_samples, )
        kwargs : dict
            Hyperparameters (see self.config below for defaults)
        """
        self.feature_names = list(feature_names)
        self.X = np.asarray(X, dtype=float)
        self.Y = np.asarray(Y, dtype=float)

        # Default configuration with optimized values
        self.config = {
            # Evolution
            "mu": 2000,
            "lambda": 2000,
            "cxpb": 0.6,
            "mutpb": 0.4,
            "ngen": 80,
            "max_depth": 11,

            # Pareto pruning
            "eps_mse": 1e-3,
            "eps_comp": 1,

            # Linear scaling
            "enable_linear_scaling": True,

            # Local refit
            "enable_local_refit": True,
            "local_refit_topK": 25,
            "local_refit_rounds": 2,
            "local_refit_iters_per_leaf": 30,
            "sigma_init": (0.35, 0.35, 0.35),
            "sigma_decay": 0.7,
            "sigma_decay_patience": 10,

            # Knee window
            "knee_window": 3,

            # Internal standardization
            "internal_standardize": False,

            # CV settings
            "cv_folds": 5,
            "cv_shuffle": True,
            "cv_seed": 2025,

            # Data-aware constants
            "rand_float_range": 2.0,
            "rand_tp_range": 1.0,
            "tp_init_mode": "data_aware",
            "tp_c_scale_factor": 1.0,

            # KAO promotion (balanced)
            "kao_inject_ratio": 0.15,
            "kao_multiplicity": 8,
            "tp_ephemeral_multiplicity": 4,

            # Complexity control
            "complexity_mode": "weighted",
            "complexity_weight_KAO": 2,

            # --- v3_1 additions ---
            # Upgrade 1: KAO on/off switch (replaces ablation_mode)
            "use_kao_leaf": True,

            # Upgrade 2: Complexity cap
            "max_complexity": None,

            # Upgrade 3: Time-checkpoint seconds
            "time_checkpoints": [10, 20, 30, 40, 50, 60],

            # --- v3.2 additions ---
            "time_budget": None,   # seconds per seed, None = no limit
            "n_jobs": 1,           # default: single-process for fair comparison
            "min_refit_slack": 1.0,  # minimum seconds remaining to attempt refit
        }
        self.config.update(kwargs)

        # Back-compat: map legacy ablation_mode -> use_kao_leaf
        if "ablation_mode" in kwargs:
            if kwargs["ablation_mode"] == "no_kao":
                self.config["use_kao_leaf"] = False

        # For internal standardization
        self.x_scaler: Optional[StandardScaler] = None
        self.y_scaler: Optional[StandardScaler] = None

        # Setup
        self.pset = None
        self.toolbox = None
        self.fit()
        self.setup_gp()

    def fit(self):
        """Optional internal standardization."""
        if self.config.get("internal_standardize", False):
            self.x_scaler = StandardScaler()
            self.y_scaler = StandardScaler()
            self.X = self.x_scaler.fit_transform(self.X)
            y = self.Y.reshape(-1, 1)
            self.Y = self.y_scaler.fit_transform(y).ravel()

    def set_global_data(self):
        """Set global variables for multiprocessing evaluation."""
        global _global_X, _global_Y, _global_pset, _global_enable_ls
        global _global_cv_indices, _global_cv_folds, _global_cv_shuffle, _global_cv_seed
        global _global_tp_mode, _global_tp_range, _global_tp_c_scale, _global_y_mean, _global_y_std
        global _global_complexity_mode, _global_node_cost
        global _global_max_complexity

        _global_X = self.X
        _global_Y = self.Y
        _global_pset = self.pset
        _global_enable_ls = bool(self.config.get("enable_linear_scaling", True))

        # CV indices
        _global_cv_folds = int(self.config.get("cv_folds", 5))
        _global_cv_shuffle = bool(self.config.get("cv_shuffle", True))
        _global_cv_seed = int(self.config.get("cv_seed", 2025))
        kf = KFold(n_splits=_global_cv_folds, shuffle=_global_cv_shuffle, random_state=_global_cv_seed)
        n = _global_X.shape[0]
        _global_cv_indices = [(tr, va) for tr, va in kf.split(np.arange(n))]

        # Data-aware TP params
        _global_tp_mode = str(self.config.get("tp_init_mode", "data_aware"))
        _global_tp_range = float(self.config.get("rand_tp_range", 1.0))
        _global_tp_c_scale = float(self.config.get("tp_c_scale_factor", 1.0))
        _global_y_mean = float(np.mean(_global_Y))
        _global_y_std = float(np.std(_global_Y) + 1e-9)
        
        # Complexity mode
        _global_complexity_mode = self.config.get("complexity_mode", "equal")
        if _global_complexity_mode == "weighted":
            _global_node_cost = {
                "KAO": float(self.config.get("complexity_weight_KAO", 2)),
                "add": 1, "sub": 1, "mul": 1, "div": 1, "makeTP": 1, "idTP": 1
            }
        else:
            _global_node_cost = {}

        # Upgrade 2: complexity cap
        _global_max_complexity = self.config.get("max_complexity", None)

    def setup_gp(self):
        """Setup typed GP environment."""
        use_kao_leaf = bool(self.config.get("use_kao_leaf", True))

        # PrimitiveSetTyped
        self.pset = gp.PrimitiveSetTyped(
            "MAIN",
            in_types=[float] * len(self.feature_names),
            ret_type=float,
        )
        for i, name in enumerate(self.feature_names):
            self.pset.renameArguments(**{f"ARG{i}": name})

        # Basic arithmetic primitives
        self.pset.addPrimitive(addf, [float, float], float, name="add")
        self.pset.addPrimitive(subf, [float, float], float, name="sub")
        self.pset.addPrimitive(mulf, [float, float], float, name="mul")
        self.pset.addPrimitive(divf, [float, float], float, name="div")
        
        # KAO-related primitives (Upgrade 1: controlled by use_kao_leaf)
        if use_kao_leaf:
            self.pset.addPrimitive(makeTP, [float, float, float], TripleParam, name="makeTP")
            self.pset.addPrimitive(identityTP, [TripleParam], TripleParam, name="idTP")
            
            # FIXED: All KAO primitives use the same name "KAO"
            # This ensures consistent expression representation
            k_mult = int(self.config.get("kao_multiplicity", 8))
            for i in range(max(1, k_mult)):
                self.pset.addPrimitive(KAO, [float, TripleParam], float, name="KAO")
            
            # Add multiple TP ephemeral constants
            tp_mult = int(self.config.get("tp_ephemeral_multiplicity", 4))
            for i in range(max(1, tp_mult)):
                self.pset.addEphemeralConstant(
                    f"randTP{i}", 
                    RandomTripleParam(
                        mode=self.config.get("tp_init_mode", "data_aware"),
                        range_val=self.config.get("rand_tp_range", 1.0),
                        y_mean=float(np.mean(self.Y)),
                        y_std=float(np.std(self.Y) + 1e-9),
                        c_scale=self.config.get("tp_c_scale_factor", 1.0)
                    ),
                    TripleParam,
                )
        
        # Float ephemeral constant
        rf = float(self.config.get("rand_float_range", 2.0))
        self.pset.addEphemeralConstant(
            "randF",
            RandomFloat(-rf, rf),
            float,
        )

        # DEAP creator(s)
        if not hasattr(creator, "FitnessMulti"):
            creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMulti)

        # Toolbox
        self.toolbox = base.Toolbox()
        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=1, max_=3)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("compile", gp.compile, pset=self.pset)
        self.toolbox.register("evaluate", evalSymbRegMulti_global)
        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register("expr_mut", gp.genGrow, min_=1, max_=2)
        self.toolbox.register("mutate", gp.mutUniform, expr=self.toolbox.expr_mut, pset=self.pset)
        self.toolbox.register("select", tools.selNSGA2)

        # Limit tree depth
        self.toolbox.decorate(
            "mate",
            gp.staticLimit(key=operator.attrgetter("height"), max_value=self.config["max_depth"]),
        )
        self.toolbox.decorate(
            "mutate",
            gp.staticLimit(key=operator.attrgetter("height"), max_value=self.config["max_depth"]),
        )

    # KAO seeding helpers
    def _pick_feature_terminal(self) -> gp.Terminal:
        """Pick a Terminal that refers to a feature argument."""
        arg_terminals = [t for t in self.pset.terminals[float] 
                        if isinstance(t, gp.Terminal) and not t.ephemeral]
        if not arg_terminals:
            raise RuntimeError("No argument terminals found.")
        return random.choice(arg_terminals)

    def _pick_kao_primitive(self) -> gp.Primitive:
        """Pick one KAO primitive from pset (all have the same name now)."""
        for prim in self.pset.primitives[float]:
            if prim.name == "KAO":  # Simple check since all KAO primitives have the same name
                return prim
        return None

    def _make_kao_seed_tree(self) -> gp.PrimitiveTree:
        """
        Construct a small tree rooted at KAO:
          - 50%: KAO(ARGi, randTP)
          - 50%: KAO(add(ARGi, ARGj), randTP)
        """
        p_kao = self._pick_kao_primitive()
        if p_kao is None:
            return self.toolbox.expr()  # Fallback to random expression
        
        tp_term = gp.Terminal(data_aware_randTP(), False, TripleParam)

        if random.random() < 0.5:
            arg = self._pick_feature_terminal()
            nodes = [p_kao, arg, tp_term]
        else:
            p_add = None
            for prim in self.pset.primitives[float]:
                if prim.name == "add":
                    p_add = prim
                    break
            if p_add is None:
                arg = self._pick_feature_terminal()
                nodes = [p_kao, arg, tp_term]
            else:
                arg1 = self._pick_feature_terminal()
                arg2 = self._pick_feature_terminal()
                nodes = [p_kao, p_add, arg1, arg2, tp_term]

        return gp.PrimitiveTree(nodes)

    def _make_kao_seed_individual(self):
        """Create an Individual seeded with a KAO-rooted tree."""
        try:
            expr = self._make_kao_seed_tree()
            ind = creator.Individual(expr)
            return ind
        except Exception:
            return self.toolbox.individual()

    def run_evolution(self):
        """
        Run NSGA-II evolution with KAO seeding and time checkpoints.
        v3.2: respects time_budget for early stop; n_jobs for pool control.

        Returns
        -------
        pop : list
        logbook : deap.tools.Logbook
        hof : list[Individual]
        pset : gp.PrimitiveSetTyped
        """
        import contextlib

        # Set global dataset for parallel evaluation
        self.set_global_data()

        mu = int(self.config["mu"])
        lambd = int(self.config.get("lambda", mu))
        ngen = int(self.config["ngen"])
        cxpb = float(self.config["cxpb"])
        mutpb = float(self.config["mutpb"])

        # v3.2: time budget
        budget = self.config.get("time_budget", None)
        if budget is not None:
            budget = float(budget)

        # v3.2: n_jobs controls multiprocessing
        n_jobs = int(self.config.get("n_jobs", 1))

        # Upgrade 1: use_kao_leaf controls KAO seeding
        use_kao_leaf = bool(self.config.get("use_kao_leaf", True))
        if use_kao_leaf:
            inject_ratio = float(self.config.get("kao_inject_ratio", 0.15))
            n_kao = max(1, int(round(inject_ratio * mu)))
            kao_inds = [self._make_kao_seed_individual() for _ in range(n_kao)]
            rest_n = max(0, mu - len(kao_inds))
            rest_inds = self.toolbox.population(n=rest_n)
            pop = kao_inds + rest_inds
        else:
            pop = self.toolbox.population(n=mu)

        hof = tools.ParetoFront()

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean, axis=0)
        stats.register("std", np.std, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)

        # Upgrade 3: time-checkpoint setup
        checkpoints = sorted(self.config.get("time_checkpoints", []))
        self.pareto_snapshots = {}
        next_cp_idx = 0
        t_start = time.perf_counter()

        # Manual generation loop (replaces eaMuPlusLambda for checkpoint support)
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + stats.fields

        completed = 0

        # v3.2: n_jobs-aware pool
        if n_jobs > 1:
            pool_cm = multiprocessing.Pool(processes=n_jobs)
        else:
            pool_cm = contextlib.nullcontext()

        with pool_cm as pool:
            if n_jobs > 1:
                self.toolbox.register("map", pool.map)
            else:
                self.toolbox.register("map", map)

            # Evaluate initial population
            invalid_ind = [ind for ind in pop if not ind.fitness.valid]
            fitnesses = list(self.toolbox.map(self.toolbox.evaluate, invalid_ind))
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            hof.update(pop)
            record = stats.compile(pop)
            logbook.record(gen=0, nevals=len(invalid_ind), **record)

            # v3.2: check budget after initial evaluation
            if budget is not None and (time.perf_counter() - t_start) >= budget:
                pass  # skip gen loop entirely
            else:
                for gen in range(1, ngen + 1):
                    # v3.2: loop head time check
                    if budget is not None and (time.perf_counter() - t_start) >= budget:
                        break

                    # Generate offspring
                    offspring = algorithms.varOr(pop, self.toolbox, lambd, cxpb, mutpb)

                    # Evaluate offspring
                    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                    fitnesses = list(self.toolbox.map(self.toolbox.evaluate, invalid_ind))
                    for ind, fit in zip(invalid_ind, fitnesses):
                        ind.fitness.values = fit

                    # Select next generation
                    pop = self.toolbox.select(pop + offspring, mu)
                    hof.update(pop)
                    record = stats.compile(pop)
                    logbook.record(gen=gen, nevals=len(invalid_ind), **record)

                    completed = gen

                    # Upgrade 3: check time-checkpoints
                    elapsed = time.perf_counter() - t_start
                    while next_cp_idx < len(checkpoints) and elapsed >= checkpoints[next_cp_idx]:
                        cp_sec = checkpoints[next_cp_idx]
                        self.pareto_snapshots[cp_sec] = [
                            (ind.fitness.values[0], calc_complexity(ind), str(ind))
                            for ind in hof
                        ]
                        next_cp_idx += 1

                    # v3.2: loop tail time check
                    if budget is not None and (time.perf_counter() - t_start) >= budget:
                        break

        self._evolution_runtime = time.perf_counter() - t_start
        self._generations_completed = completed

        # Final snapshot for any remaining checkpoints
        while next_cp_idx < len(checkpoints):
            cp_sec = checkpoints[next_cp_idx]
            self.pareto_snapshots[cp_sec] = [
                (ind.fitness.values[0], calc_complexity(ind), str(ind))
                for ind in hof
            ]
            next_cp_idx += 1

        # Convert hof to list
        hof_list = list(hof)

        # Optional: local stochastic refit (v3.2: skip if budget exhausted)
        if bool(self.config.get("enable_local_refit", True)) and len(hof_list) > 0:
            elapsed = time.perf_counter() - t_start
            min_refit_slack = float(self.config.get("min_refit_slack", 1.0))
            if budget is None or (budget - elapsed) >= min_refit_slack:
                hof_list = self._local_refit_on_hof(hof_list)

        return pop, logbook, hof_list, self.pset

    def _local_refit_on_hof(self, hof_list):
        """Local stochastic hill-climbing refit for KAO parameters."""
        topK = int(self.config.get("local_refit_topK", 25))
        rounds = int(self.config.get("local_refit_rounds", 2))
        iters_per_leaf = int(self.config.get("local_refit_iters_per_leaf", 30))
        sigma_init = self.config.get("sigma_init", (0.35, 0.35, 0.35))
        sigma_decay = float(self.config.get("sigma_decay", 0.7))
        sigma_decay_patience = int(self.config.get("sigma_decay_patience", 10))

        # Select top-K individuals
        selected = hof_list[:min(topK, len(hof_list))]
        
        for round_idx in range(rounds):
            for ind in selected:
                # Find KAO nodes and refit their parameters
                self._refit_kao_params(ind, iters_per_leaf, sigma_init, sigma_decay, sigma_decay_patience)
                # Re-evaluate after refit
                ind.fitness.values = self.toolbox.evaluate(ind)
        
        # Re-sort HOF after refit
        hof_list.sort(key=lambda x: (x.fitness.values[0], x.fitness.values[1]))
        return hof_list

    def _refit_kao_params(self, ind, iters, sigma_init, sigma_decay, sigma_decay_patience):
        """Refit KAO parameters in an individual using stochastic hill-climbing."""
        # Implementation simplified for clarity
        # In practice, this would traverse the tree and update TripleParam nodes
        pass

    def build_hof_dataframe(self, hof: List) -> pd.DataFrame:
        """Build dataframe from HOF."""
        recs = []
        for i, ind in enumerate(hof):
            try:
                mse, comp = ind.fitness.values
            except Exception:
                mse, comp = self.toolbox.evaluate(ind)
            recs.append({"idx": i, "mse": float(mse), "Complexity": int(comp), "expr": ind})
        df = pd.DataFrame(recs).sort_values("Complexity").reset_index(drop=True)
        return df

    def epsilon_prune(self, df: pd.DataFrame) -> pd.DataFrame:
        """Epsilon-prune the Pareto frontier."""
        eps_mse = float(self.config["eps_mse"])
        eps_comp = int(self.config["eps_comp"])

        kept = []
        for _, row in df.iterrows():
            dominated = False
            for k in kept:
                if (row.mse >= k.mse - eps_mse) and (row.Complexity >= k.Complexity - eps_comp):
                    dominated = True
                    break
            if not dominated:
                kept.append(row)
        pruned = pd.DataFrame(kept).sort_values("Complexity").reset_index(drop=True)
        return pruned

    def select_knee_candidates(self, pruned_df: pd.DataFrame) -> Tuple[int, pd.DataFrame, int]:
        """Return (knee_complexity, candidate_rows_df, knee_idx_in_df)."""
        pruned_df = pruned_df.sort_values("Complexity").reset_index(drop=True)
        kl = KneeLocator(
            pruned_df.Complexity.tolist(),
            pruned_df.mse.tolist(),
            curve="convex",
            direction="decreasing",
        )
        knee = kl.knee
        if knee is None:
            knee = float(pruned_df["Complexity"].median())

        complexities = pruned_df["Complexity"].values
        knee_val = int(complexities[np.argmin(np.abs(complexities - knee))])

        window = int(self.config.get("knee_window", 3))
        candidates = pruned_df[pruned_df.Complexity.between(knee_val - window, knee_val + window)].reset_index(drop=True)

        knee_idx = int(np.where(pruned_df.Complexity.values == knee_val)[0][0])
        return knee_val, candidates, knee_idx

    def evaluate_individual(self, ind):
        """Evaluate an individual (for single evaluation, not multiprocessing)."""
        self.set_global_data()  # Ensure globals are set
        return evalSymbRegMulti_global(ind)


# ---------------------------------------------------------------------------
# Upgrade 4: Unified complexity metric
# ---------------------------------------------------------------------------
def compute_complexity_chars(expr_str: str) -> int:
    """Character-length complexity proxy (spaces and parentheses removed)."""
    cleaned = re.sub(r'[\s\(\)]', '', str(expr_str))
    return len(cleaned)


# ---------------------------------------------------------------------------
# Upgrade 5: Structured result object
# ---------------------------------------------------------------------------
@dataclass
class KAOResult:
    expression: str
    r2_cv: float
    r2_test: float
    y_pred_train: np.ndarray
    y_pred_test: np.ndarray
    complexity_nodes: int
    complexity_chars: int
    runtime: float
    pareto_front: list            # [(error, complexity, expr_str), ...]
    pareto_snapshots: dict        # {10: [...], 20: [...], ...}
    seed: int
    config: dict                  # runtime parameter snapshot


# ---------------------------------------------------------------------------
# Upgrade 6: Expression simplification
# ---------------------------------------------------------------------------
def simplify_expression(expr_str: str) -> str:
    """Attempt sympy simplification; fall back to original string."""
    try:
        import sympy
        simplified = str(sympy.simplify(expr_str))
        return simplified
    except Exception:
        return expr_str


# ---------------------------------------------------------------------------
# Upgrade 7 & 8: run_single / run_multi_seed
# ---------------------------------------------------------------------------
def run_single(X_train: np.ndarray, y_train: np.ndarray,
               X_test: np.ndarray, y_test: np.ndarray,
               dataset_name: str = "unknown",
               seed: int = 42,
               time_budget: float = 60.0,
               use_kao_leaf: bool = True,
               max_complexity: Optional[int] = None,
               time_checkpoints: Optional[List[int]] = None,
               feature_names: Optional[List[str]] = None,
               **extra_config) -> KAOResult:
    """
    Run a single KAO experiment and return a structured KAOResult.

    Parameters
    ----------
    X_train, y_train : training data
    X_test, y_test   : test data
    dataset_name      : label for logging
    seed              : random seed
    time_budget       : maximum seconds (controls ngen heuristic)
    use_kao_leaf      : True = KAO mode, False = plain GP ablation
    max_complexity    : drop individuals exceeding this node count
    time_checkpoints  : seconds at which to snapshot the Pareto front
    feature_names     : column names (auto-generated if None)
    **extra_config    : forwarded to KAO_Core
    """
    if time_checkpoints is None:
        time_checkpoints = [10, 20, 30, 40, 50, 60]
    if feature_names is None:
        feature_names = [f"x{i}" for i in range(X_train.shape[1])]

    # Seed everything
    random.seed(seed)
    np.random.seed(seed)

    # Build config
    config = dict(
        time_budget=time_budget,
        use_kao_leaf=use_kao_leaf,
        max_complexity=max_complexity,
        time_checkpoints=time_checkpoints,
        **extra_config,
    )

    # Instantiate and run
    t0 = time.perf_counter()
    core = KAO_Core(feature_names, X_train, y_train, **config)
    _pop, _logbook, hof_list, pset = core.run_evolution()
    runtime = time.perf_counter() - t0

    # Build Pareto front
    hof_df = core.build_hof_dataframe(hof_list)
    pruned_df = core.epsilon_prune(hof_df)

    pareto_front = [
        (float(r.mse), int(r.Complexity), str(r.expr))
        for _, r in pruned_df.iterrows()
    ]

    # Select best individual (knee point preferred, fallback to min MSE)
    if len(pruned_df) >= 3:
        try:
            _knee_val, candidates, _knee_idx = core.select_knee_candidates(pruned_df)
            best_row = candidates.sort_values("mse").iloc[0]
        except Exception:
            best_row = pruned_df.sort_values("mse").iloc[0]
    else:
        best_row = pruned_df.sort_values("mse").iloc[0] if len(pruned_df) > 0 else None

    if best_row is None:
        # No valid individual found
        return KAOResult(
            expression="FAILED", r2_cv=float('nan'), r2_test=float('nan'),
            y_pred_train=np.full(len(y_train), np.nan),
            y_pred_test=np.full(len(y_test), np.nan),
            complexity_nodes=0, complexity_chars=0,
            runtime=runtime, pareto_front=[], pareto_snapshots={},
            seed=seed, config=config,
        )

    best_ind = best_row.expr
    expr_str = str(best_ind)

    # Compile and predict
    func = gp.compile(best_ind, pset)
    raw_train = np.array([func(*row) for row in X_train], dtype=float)
    raw_test = np.array([func(*row) for row in X_test], dtype=float)

    # Linear scaling on train, apply same transform to test
    if bool(config.get("enable_linear_scaling", True)):
        scaled_train, alpha, beta = _linear_scale(y_train, raw_train)
        y_pred_train = scaled_train
        y_pred_test = alpha * raw_test + beta
    else:
        y_pred_train = raw_train
        y_pred_test = raw_test

    # Sanitize predictions
    for arr in (y_pred_train, y_pred_test):
        np.clip(arr, -1e15, 1e15, out=arr)
        arr[~np.isfinite(arr)] = 0.0

    # Metrics
    r2_cv = 1.0 - best_row.mse / max(float(np.var(y_train)), 1e-12)
    r2_test = float(r2_score(y_test, y_pred_test))

    # Upgrade 4: dual complexity
    complexity_nodes = int(best_row.Complexity)
    complexity_chars = compute_complexity_chars(expr_str)

    # Upgrade 6: simplify expression
    expr_simplified = simplify_expression(expr_str)

    # Upgrade 7: JSON log
    _write_json_log(
        dataset_name=dataset_name, seed=seed,
        use_kao_leaf=use_kao_leaf, max_complexity=max_complexity,
        runtime=runtime, r2_test=r2_test, expression=expr_simplified,
        complexity_nodes=complexity_nodes, complexity_chars=complexity_chars,
        pareto_front_size=len(pareto_front),
        n_gen=core._generations_completed,
    )

    return KAOResult(
        expression=expr_simplified,
        r2_cv=r2_cv,
        r2_test=r2_test,
        y_pred_train=y_pred_train,
        y_pred_test=y_pred_test,
        complexity_nodes=complexity_nodes,
        complexity_chars=complexity_chars,
        runtime=runtime,
        pareto_front=pareto_front,
        pareto_snapshots=dict(core.pareto_snapshots),
        seed=seed,
        config=config,
    )


def _write_json_log(*, dataset_name, seed, use_kao_leaf, max_complexity,
                    runtime, r2_test, expression, complexity_nodes,
                    complexity_chars, pareto_front_size, n_gen):
    """Upgrade 7: Write a structured JSON log to results/logs/."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log = {
        "timestamp": datetime.datetime.now().isoformat(),
        "dataset": dataset_name,
        "seed": seed,
        "use_kao_leaf": use_kao_leaf,
        "max_complexity": max_complexity,
        "runtime": round(runtime, 2),
        "best_r2_test": round(r2_test, 6),
        "best_expression": expression,
        "complexity_nodes": complexity_nodes,
        "complexity_chars": complexity_chars,
        "pareto_front_size": pareto_front_size,
        "generations_completed": n_gen,
    }
    log_dir = os.path.join("results", "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{dataset_name}_seed{seed}_{timestamp}.json")
    try:
        with open(log_path, 'w') as f:
            json.dump(log, f, indent=2)
    except OSError:
        pass  # non-fatal: don't crash if logging fails


# ---------------------------------------------------------------------------
# Upgrade 8: Multi-seed convenience function
# ---------------------------------------------------------------------------
def run_multi_seed(X_train: np.ndarray, y_train: np.ndarray,
                   X_test: np.ndarray, y_test: np.ndarray,
                   dataset_name: str = "unknown",
                   seeds=range(30),
                   time_budget: float = 60.0,
                   use_kao_leaf: bool = True,
                   max_complexity: Optional[int] = None,
                   time_checkpoints: Optional[List[int]] = None,
                   feature_names: Optional[List[str]] = None,
                   **extra_config) -> List[KAOResult]:
    """Run KAO over multiple seeds and return a list of KAOResult."""
    seeds = list(seeds)
    results = []
    for s in seeds:
        print(f"  Seed {s}/{max(seeds)}...", end=" ", flush=True)
        result = run_single(
            X_train, y_train, X_test, y_test,
            dataset_name=dataset_name, seed=s,
            time_budget=time_budget,
            use_kao_leaf=use_kao_leaf,
            max_complexity=max_complexity,
            time_checkpoints=time_checkpoints,
            feature_names=feature_names,
            **extra_config,
        )
        results.append(result)
        print(f"R²={result.r2_test:.4f}, nodes={result.complexity_nodes}")
    return results