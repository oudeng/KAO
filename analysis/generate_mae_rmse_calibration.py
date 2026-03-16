#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 2 — MAE/RMSE + Calibration (Method A': expression re-evaluation)

From existing result.json files, parse each method's expression string,
re-create predictions on the held-out test set, compute MAE / RMSE,
and draw binned reliability (calibration) plots.

R^2 sanity check: the re-computed R^2 must match result.json within 0.01,
otherwise the seed is flagged as a parser failure.

Outputs
-------
  outputs/csv/mae_rmse_all.csv
  outputs/tables/Table_mae_rmse_healthcare.tex   (main text — compact)
  outputs/tables/Table_mae_rmse_supp.tex         (SM — full)
  outputs/figures/Fig_calib_mimic.pdf
  outputs/figures/Fig_calib_eicu.pdf
  outputs/figures/Fig_calib_nhanes.pdf
"""

from __future__ import annotations

import json, glob, os, re, sys, warnings, traceback
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent      # KAO_v3/

HEALTHCARE = {
    "mimic_iv": {
        "csv": "KAO/data/mimic_iv/ICU_composite_risk_score.csv",
        "target": "composite_risk_score",
        "display": "MIMIC-IV",
    },
    "eicu": {
        "csv": "KAO/data/eicu/eICU_composite_risk_score.csv",
        "target": "composite_risk_score",
        "display": "eICU",
    },
    "nhanes": {
        "csv": "KAO/data/nhanes/NHANES_metabolic_score.csv",
        "target": "metabolic_score",
        "display": "NHANES",
    },
}

METHODS       = ["KAO", "PySR", "Operon", "RILS-ROLS", "gplearn"]
METHOD_DIRS   = {"KAO": "kao", "PySR": "pysr", "Operon": "operon",
                 "RILS-ROLS": "rils_rols", "gplearn": "gplearn"}
SEEDS         = list(range(1, 31))
SPLIT_SEED    = 2025
TEST_SIZE     = 0.2
R2_TOL        = 0.01        # sanity-check tolerance

METHOD_COLORS = {
    "KAO":       "#0072B2",
    "PySR":      "#D55E00",
    "RILS-ROLS": "#009E73",
    "gplearn":   "#CC79A7",
    "Operon":    "#E69F00",
}
METHOD_MARKERS = {
    "KAO": "o", "PySR": "s", "RILS-ROLS": "^",
    "gplearn": "D", "Operon": "v",
}

# ─────────────────────────────────────────────────────────────────────
# Data loading (mirrors run_kao.py / run_baselines.py exactly)
# ─────────────────────────────────────────────────────────────────────
_data_cache: Dict[str, Any] = {}

def load_test_data(dataset_key: str):
    """Return (X_test, y_test, feature_names) using the same split as the
    original experiments."""
    if dataset_key in _data_cache:
        return _data_cache[dataset_key]

    info = HEALTHCARE[dataset_key]
    df = pd.read_csv(ROOT / info["csv"]).dropna()
    target = info["target"]
    feature_names = [c for c in df.columns if c != target]
    X = df[feature_names].values
    y = df[target].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SPLIT_SEED,
    )
    # Fit a StandardScaler on X_train for methods that trained on
    # standardised features (Operon, RILS-ROLS).
    scaler = StandardScaler()
    scaler.fit(X_train)

    _data_cache[dataset_key] = (X_test, y_test, feature_names, X_train, y_train, scaler)
    return X_test, y_test, feature_names, X_train, y_train, scaler


# ─────────────────────────────────────────────────────────────────────
# Prefix-notation recursive parser  (KAO & gplearn)
# ─────────────────────────────────────────────────────────────────────
class _Tokenizer:
    """Simple tokenizer for prefix-notation expressions."""
    # Matches: function names, numbers (incl. negative / sci notation), commas, parens
    _TOKEN_RE = re.compile(
        r"([A-Za-z_]\w*)"           # identifier
        r"|(-?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)"  # number
        r"|([(),])"                  # punctuation
    )

    def __init__(self, text: str):
        self.tokens = [m.group() for m in self._TOKEN_RE.finditer(text)]
        self.pos = 0

    def peek(self) -> Optional[str]:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def consume(self) -> str:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok


def _parse_prefix(tok: _Tokenizer, feat_idx: Dict[str, int],
                  X: np.ndarray, method: str = "") -> np.ndarray:
    """Recursively evaluate a prefix-notation expression on *X* (n_samples, n_feat)."""
    cur = tok.peek()
    if cur is None:
        raise ValueError("Unexpected end of expression")

    # ── Number literal ──
    try:
        val = float(cur)
        tok.consume()
        return np.full(X.shape[0], val, dtype=float)
    except (ValueError, TypeError):
        pass

    # ── Variable name ──
    if cur in feat_idx:
        tok.consume()
        return X[:, feat_idx[cur]].astype(float)

    # ── Function call:  funcname '(' args ')'  ──
    fname = tok.consume()           # function name
    assert tok.consume() == "(", f"Expected '(' after {fname}"
    args: List[np.ndarray] = []
    while tok.peek() != ")":
        if tok.peek() == ",":
            tok.consume()           # skip comma
        args.append(_parse_prefix(tok, feat_idx, X, method))
    tok.consume()                   # consume ')'

    return _apply_prefix_func(fname, args, method)


def _apply_prefix_func(fname: str, args: List[np.ndarray],
                       method: str = "") -> np.ndarray:
    """Dispatch a prefix function call.

    Protected-function semantics differ between KAO and gplearn:
      - KAO:     div(a, b) → a when |b| < 1e-9
      - gplearn: div(a, b) → 1.0 when |b| ≤ 0.001
    The *method* flag selects the convention.
    """
    _gp = (method == "gplearn")

    # --- basic arithmetic (arity 2) ---
    if fname == "add":
        return args[0] + args[1]
    if fname == "sub":
        return args[0] - args[1]
    if fname == "mul":
        return args[0] * args[1]
    if fname == "div":
        with np.errstate(divide="ignore", invalid="ignore"):
            if _gp:
                # gplearn: div(a,b) → 1.0 where |b| ≤ 0.001
                return np.where(np.abs(args[1]) > 0.001,
                                np.divide(args[0], args[1]), 1.0)
            else:
                # KAO: div(a,b) → a where |b| < 1e-9
                denom = args[1].copy()
                mask = np.abs(denom) < 1e-9
                denom[mask] = 1.0
                result = args[0] / denom
                result[mask] = args[0][mask]
                return result

    # --- unary ---
    if fname == "neg":
        return -args[0]
    if fname == "abs":
        return np.abs(args[0])
    if fname == "sqrt":
        return np.sqrt(np.abs(args[0]))
    if fname == "log":
        with np.errstate(divide="ignore", invalid="ignore"):
            if _gp:
                # gplearn: log(x) → 0.0 where |x| ≤ 0.001
                return np.where(np.abs(args[0]) > 0.001,
                                np.log(np.abs(args[0])), 0.0)
            else:
                return np.log(np.abs(args[0]) + 1e-15)
    if fname == "exp":
        return np.exp(np.clip(args[0], -500, 500))
    if fname == "sin":
        return np.sin(args[0])
    if fname == "cos":
        return np.cos(args[0])

    # --- KAO-specific ---
    if fname == "makeTP":
        return (args[0], args[1], args[2])          # type: ignore[return-value]
    if fname == "idTP":
        return args[0]
    if fname == "KAO":
        x = args[0]
        tp = args[1]
        if isinstance(tp, tuple) and len(tp) == 3:
            a_arr, b_arr, c_arr = tp
            return a_arr * (x ** 2) + b_arr * x + c_arr
        raise ValueError(f"KAO second arg is not a TripleParam tuple: {type(tp)}")

    raise ValueError(f"Unknown prefix function: {fname}")


def eval_prefix_expression(expr_str: str, X: np.ndarray,
                           feature_names: List[str],
                           method: str = "") -> np.ndarray:
    """Parse and evaluate a prefix-notation expression (KAO or gplearn)."""
    feat_idx = {name: i for i, name in enumerate(feature_names)}
    tok = _Tokenizer(expr_str)
    result = _parse_prefix(tok, feat_idx, X, method)
    if isinstance(result, tuple):
        raise ValueError("Top-level expression returned TripleParam — expected scalar array")
    return np.asarray(result, dtype=float).flatten()


# ─────────────────────────────────────────────────────────────────────
# Infix parser via sympy   (PySR, Operon, RILS-ROLS)
# ─────────────────────────────────────────────────────────────────────
def _protected_numpy_module():
    """Return a numpy-based module dict with protected math functions
    for use with sympy.lambdify."""
    def _plog(x):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log(np.where(np.abs(x) > 1e-15, np.abs(x), 1e-15))

    def _pexp(x):
        return np.exp(np.clip(np.asarray(x, dtype=float), -500, 500))

    def _psqrt(x):
        return np.sqrt(np.abs(x))

    return {
        "log": _plog,
        "exp": _pexp,
        "sqrt": _psqrt,
        "sin": np.sin,
        "cos": np.cos,
        "Abs": np.abs,
    }


def eval_infix_expression(expr_str: str, X: np.ndarray,
                          feature_names: List[str]) -> np.ndarray:
    """Parse an infix mathematical expression via sympy and evaluate."""
    import sympy

    local_dict = {name: sympy.Symbol(name) for name in feature_names}
    local_dict["sqrt"] = sympy.sqrt
    local_dict["log"]  = sympy.log
    local_dict["exp"]  = sympy.exp
    local_dict["sin"]  = sympy.sin
    local_dict["cos"]  = sympy.cos
    local_dict["abs"]  = sympy.Abs

    parsed = sympy.sympify(expr_str, locals=local_dict)
    ordered_syms = [sympy.Symbol(name) for name in feature_names]

    # Use protected numpy functions to avoid overflow / log-of-zero
    modules = [_protected_numpy_module(), "numpy"]
    func = sympy.lambdify(ordered_syms, parsed, modules=modules)
    y_pred = func(*[X[:, i] for i in range(len(feature_names))])
    return np.asarray(y_pred, dtype=float).flatten()


# ─────────────────────────────────────────────────────────────────────
# Unified evaluator
# ─────────────────────────────────────────────────────────────────────
_PREFIX_FUNCS = {"add", "sub", "mul", "div", "sqrt", "log", "abs", "neg",
                 "exp", "sin", "cos", "KAO", "makeTP", "idTP"}

def _is_prefix_notation(expr: str) -> bool:
    """Heuristic: prefix notation if the *outermost* construct is a known
    prefix function call like ``add(…)`` or ``KAO(…)``.

    Infix expressions that happen to start with ``(`` (e.g. PySR's
    ``(a + b)*c``) must NOT be classified as prefix.
    """
    m = re.match(r'^([a-zA-Z_]\w*)\s*\(', expr.strip())
    if m is None:
        return False
    return m.group(1) in _PREFIX_FUNCS


def eval_expression(expr_str: str, method: str, X: np.ndarray,
                    feature_names: List[str],
                    alpha: float = 1.0, beta: float = 0.0,
                    scaler: Optional[Any] = None) -> np.ndarray:
    """
    Evaluate an expression string and return y_pred.
    Applies linear scaling for KAO (alpha * raw + beta).

    For methods that trained on standardised features (Operon, RILS-ROLS),
    *scaler* (a fitted StandardScaler) is applied to X before evaluation.
    """
    # Standardise inputs when the method was trained on scaled features
    if method in ("Operon", "RILS-ROLS") and scaler is not None:
        X = scaler.transform(X)

    # KAO and gplearn always use prefix notation.
    # PySR, Operon, RILS-ROLS always use infix notation (even when the
    # expression starts with a math function like log(...) or sqrt(...)).
    if method in ("KAO", "gplearn"):
        raw = eval_prefix_expression(expr_str, X, feature_names, method)
    else:
        raw = eval_infix_expression(expr_str, X, feature_names)

    # Apply linear scaling
    y_pred = alpha * raw + beta

    # Sanitize
    y_pred = np.clip(y_pred, -1e15, 1e15)
    y_pred[~np.isfinite(y_pred)] = 0.0

    return y_pred


# ─────────────────────────────────────────────────────────────────────
# Result-file discovery
# ─────────────────────────────────────────────────────────────────────
def find_result_json(dataset_key: str, method: str, seed: int) -> Optional[Path]:
    method_dir = METHOD_DIRS[method]
    p = ROOT / "results" / dataset_key / "60s" / method_dir / f"seed_{seed}" / "result.json"
    return p if p.exists() else None


# ─────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────
def main():
    warnings.filterwarnings("ignore")

    out_csv   = ROOT / "outputs" / "csv"
    out_tab   = ROOT / "outputs" / "tables"
    out_fig   = ROOT / "outputs" / "figures"
    for d in (out_csv, out_tab, out_fig):
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Phase 2: MAE / RMSE / Calibration  (Method A' — expression re-eval)")
    print("=" * 70)

    # ── Collect metrics ──────────────────────────────────────────────
    all_rows: List[dict]           = []
    all_preds: Dict[tuple, tuple]  = {}      # (ds, method, seed) → (y_true, y_pred)
    eval_ok:   Dict[tuple, int]    = {}      # (ds, method) → count of successes
    eval_fail: List[dict]          = []

    for ds_key, ds_info in HEALTHCARE.items():
        X_test, y_test, feat_names, X_train, y_train, scaler = load_test_data(ds_key)
        print(f"\n{'─'*50}")
        print(f"Dataset: {ds_info['display']}  (n_test={len(y_test)}, n_feat={len(feat_names)})")

        for method in METHODS:
            n_ok = 0
            n_fail = 0
            method_mae = []
            method_rmse = []

            for seed in SEEDS:
                rf = find_result_json(ds_key, method, seed)
                if rf is None:
                    continue
                with open(rf) as fh:
                    d = json.load(fh)
                expr = d.get("expression", "")
                r2_stored = d.get("r2_test")

                # Read alpha / beta (KAO only; baselines → 1.0 / 0.0)
                extra = d.get("extra", {})
                alpha = float(extra.get("alpha", 1.0))
                beta  = float(extra.get("beta",  0.0))

                # ── Try eval ─────────────────────────────────────
                try:
                    y_pred = eval_expression(expr, method, X_test,
                                             feat_names, alpha, beta,
                                             scaler=scaler)
                    r2_recomp = float(r2_score(y_test, y_pred))
                except Exception as exc:
                    eval_fail.append({
                        "dataset": ds_key, "method": method,
                        "seed": seed, "error": str(exc)[:120],
                    })
                    n_fail += 1
                    continue

                # ── R² sanity check ──────────────────────────────
                if r2_stored is not None and np.isfinite(r2_stored):
                    delta = abs(r2_recomp - r2_stored)
                    if delta > R2_TOL:
                        eval_fail.append({
                            "dataset": ds_key, "method": method,
                            "seed": seed,
                            "error": (f"R2 mismatch: recomp={r2_recomp:.6f} "
                                      f"vs stored={r2_stored:.6f} "
                                      f"(delta={delta:.6f})"),
                        })
                        n_fail += 1
                        continue

                # ── Metrics ──────────────────────────────────────
                mae  = float(mean_absolute_error(y_test, y_pred))
                rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

                nodes = d.get("complexity_nodes")

                all_rows.append({
                    "dataset": ds_info["display"],
                    "dataset_key": ds_key,
                    "method": method,
                    "seed": seed,
                    "mae": mae,
                    "rmse": rmse,
                    "r2_recomp": r2_recomp,
                    "r2_stored": r2_stored,
                    "nodes": nodes,
                })
                all_preds[(ds_key, method, seed)] = (y_test, y_pred)
                n_ok += 1
                method_mae.append(mae)
                method_rmse.append(rmse)

            eval_ok[(ds_key, method)] = n_ok
            status = "OK" if n_ok == len(SEEDS) else f"WARN({n_ok}/{len(SEEDS)})"
            mean_mae  = np.mean(method_mae) if method_mae else float("nan")
            mean_rmse = np.mean(method_rmse) if method_rmse else float("nan")
            print(f"  {method:12s}  eval={status:12s}  "
                  f"MAE={mean_mae:.4f}  RMSE={mean_rmse:.4f}")

    # ── Report eval failures ─────────────────────────────────────────
    if eval_fail:
        print(f"\n{'='*70}")
        print(f"EVAL FAILURES: {len(eval_fail)}")
        print(f"{'='*70}")
        for ef in eval_fail:
            print(f"  {ef['dataset']}/{ef['method']}/seed_{ef['seed']}: {ef['error']}")

    # ── Save raw CSV ─────────────────────────────────────────────────
    df_all = pd.DataFrame(all_rows)
    df_all.to_csv(out_csv / "mae_rmse_all.csv", index=False)
    print(f"\nSaved {out_csv / 'mae_rmse_all.csv'}  ({len(df_all)} rows)")

    # ── Aggregated stats ─────────────────────────────────────────────
    agg = (df_all.groupby(["dataset", "method"])
           .agg(mae_mean=("mae", "mean"), mae_std=("mae", "std"),
                rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
                nodes_mean=("nodes", "mean"), nodes_std=("nodes", "std"),
                n_seeds=("seed", "count"))
           .reset_index())

    # ── LaTeX: SM full table ─────────────────────────────────────────
    _write_supp_table(agg, out_tab / "Table_mae_rmse_supp.tex")

    # ── LaTeX: main-text compact table ───────────────────────────────
    _write_main_table(agg, out_tab / "Table_mae_rmse_healthcare.tex")

    # ── Calibration figures ──────────────────────────────────────────
    for ds_key, ds_info in HEALTHCARE.items():
        _plot_calibration(ds_key, ds_info["display"], all_preds, df_all, out_fig)

    print("\nDone.")


# ─────────────────────────────────────────────────────────────────────
# LaTeX table generators
# ─────────────────────────────────────────────────────────────────────
def _fmt(mean: float, std: float) -> str:
    return f"${mean:.3f} \\pm {std:.3f}$"


def _fmt_nodes(mean: float, std: float) -> str:
    return f"${mean:.1f} \\pm {std:.1f}$"


def _write_supp_table(agg: pd.DataFrame, path: Path):
    """SM table: all datasets × all methods."""
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\caption{Test-set MAE, RMSE, and expression complexity (Nodes) on healthcare datasets "
        r"(30 seeds, mean $\pm$ std). "
        r"Bold values indicate the best (lowest) result in each column per dataset. "
        r"KAO expressions are selected at the NSGA-II knee point (mean $\le$\,8 nodes); "
        r"baselines use default configurations without explicit complexity constraints. "
        r"The accuracy gap reflects the cost of auditability and motivates the "
        r"complexity-matched comparison (Supplementary Table~\ref{tab:complexity_matched_summary}).}",
        r"\label{tab:mae_rmse_supp}",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Dataset & Method & MAE & RMSE & Nodes \\",
        r"\midrule",
    ]

    ds_order = ["MIMIC-IV", "eICU", "NHANES"]
    for i, ds in enumerate(ds_order):
        sub = agg[agg["dataset"] == ds].copy()
        if sub.empty:
            continue
        best_mae  = sub["mae_mean"].min()
        best_rmse = sub["rmse_mean"].min()
        for _, row in sub.iterrows():
            mae_s  = _fmt(row.mae_mean, row.mae_std)
            rmse_s = _fmt(row.rmse_mean, row.rmse_std)
            nodes_s = _fmt_nodes(row.nodes_mean, row.nodes_std)
            if abs(row.mae_mean - best_mae) < 1e-6:
                mae_s = r"\textbf{" + mae_s + "}"
            if abs(row.rmse_mean - best_rmse) < 1e-6:
                rmse_s = r"\textbf{" + rmse_s + "}"
            lines.append(f"{ds} & {row.method} & {mae_s} & {rmse_s} & {nodes_s} \\\\")
        if i < len(ds_order) - 1:
            lines.append(r"\midrule")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {path}")


def _write_main_table(agg: pd.DataFrame, path: Path):
    """Main-text table: 6 rows = 3 datasets × (KAO + best baseline)."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Mean absolute error (MAE), root mean squared error (RMSE), and "
        r"expression complexity (Nodes) on healthcare test sets "
        r"(30 seeds, mean $\pm$ std). "
        r"Best MAE per dataset is in bold. "
        r"KAO expressions are selected at the NSGA-II knee point "
        r"(mean $\le$\,8 nodes); baselines use default configurations "
        r"without explicit complexity constraints. "
        r"The accuracy gap reflects the cost of auditability and motivates "
        r"the complexity-matched comparison "
        r"(Supplementary Table~\ref{tab:complexity_matched_summary}).}",
        r"\label{tab:mae_rmse_healthcare}",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Dataset & Method & MAE & RMSE & Nodes \\",
        r"\midrule",
    ]

    ds_order = ["MIMIC-IV", "eICU", "NHANES"]
    for i, ds in enumerate(ds_order):
        sub = agg[agg["dataset"] == ds]
        kao = sub[sub["method"] == "KAO"]
        baselines = sub[sub["method"] != "KAO"]

        if kao.empty or baselines.empty:
            continue

        kao_row = kao.iloc[0]
        best_bl = baselines.loc[baselines["mae_mean"].idxmin()]

        # Determine best MAE across KAO and best baseline
        best_mae_val = min(kao_row.mae_mean, best_bl.mae_mean)

        # KAO row
        kao_mae  = _fmt(kao_row.mae_mean, kao_row.mae_std)
        kao_rmse = _fmt(kao_row.rmse_mean, kao_row.rmse_std)
        kao_nodes = _fmt_nodes(kao_row.nodes_mean, kao_row.nodes_std)
        if abs(kao_row.mae_mean - best_mae_val) < 1e-6:
            kao_mae = r"\textbf{" + kao_mae + "}"
        lines.append(f"{ds} & KAO & {kao_mae} & {kao_rmse} & {kao_nodes} \\\\")

        # Best baseline row
        bl_mae  = _fmt(best_bl.mae_mean, best_bl.mae_std)
        bl_rmse = _fmt(best_bl.rmse_mean, best_bl.rmse_std)
        bl_nodes = _fmt_nodes(best_bl.nodes_mean, best_bl.nodes_std)
        bl_name = best_bl.method
        if abs(best_bl.mae_mean - best_mae_val) < 1e-6:
            bl_mae = r"\textbf{" + bl_mae + "}"
        lines.append(f" & {bl_name} & {bl_mae} & {bl_rmse} & {bl_nodes} \\\\")

        if i < len(ds_order) - 1:
            lines.append(r"\midrule")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {path}")


# ─────────────────────────────────────────────────────────────────────
# Calibration plots
# ─────────────────────────────────────────────────────────────────────
def _select_median_seed(ds_key: str, method: str,
                        df_all: pd.DataFrame) -> Optional[int]:
    """Pick the seed whose R^2 is closest to the method's mean R^2."""
    sub = df_all[(df_all["dataset_key"] == ds_key) &
                 (df_all["method"] == method)]
    if sub.empty:
        return None
    mean_r2 = sub["r2_recomp"].mean()
    idx = (sub["r2_recomp"] - mean_r2).abs().idxmin()
    return int(sub.loc[idx, "seed"])


def _plot_calibration(ds_key: str, display_name: str,
                      all_preds: dict, df_all: pd.DataFrame,
                      out_dir: Path):
    fig, ax = plt.subplots(figsize=(5, 5))

    for method in METHODS:
        seed = _select_median_seed(ds_key, method, df_all)
        if seed is None or (ds_key, method, seed) not in all_preds:
            continue
        y_true, y_pred = all_preds[(ds_key, method, seed)]

        # Decile binning
        try:
            bins = pd.qcut(y_pred, q=10, duplicates="drop")
        except ValueError:
            bins = pd.cut(y_pred, bins=10)

        tmp = pd.DataFrame({"true": y_true, "pred": y_pred, "bin": bins})
        grp = tmp.groupby("bin", observed=True)
        m_pred  = grp["pred"].mean()
        m_true  = grp["true"].mean()
        counts  = grp.size()
        sizes   = (counts / counts.max()) * 150 + 20

        ax.scatter(m_pred, m_true, s=sizes,
                   c=METHOD_COLORS[method],
                   marker=METHOD_MARKERS[method],
                   label=method, alpha=0.8,
                   edgecolors="white", linewidth=0.5)

    # y = x reference
    lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
    hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4, linewidth=1,
            label="Perfect calibration")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("Mean predicted score")
    ax.set_ylabel("Mean observed score")
    ax.set_title(display_name)
    ax.legend(fontsize=8, framealpha=0.8)
    plt.tight_layout()

    # File naming:  mimic_iv → mimic,  eicu → eicu,  nhanes → nhanes
    slug = ds_key.replace("_iv", "")
    outpath = out_dir / f"Fig_calib_{slug}.pdf"
    fig.savefig(outpath, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {outpath}")


# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
