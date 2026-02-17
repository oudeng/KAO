# -*- coding: utf-8 -*-
"""
cv_eval.py - Fold-based evaluation for already-trained models (KAO v3.2)

Instead of retraining per fold (which would cost 5x time_budget), we
evaluate the *same* trained model on each validation split and report
mean MSE, mean R2, and instability (CV-of-MSE).
"""
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

EPS = 1e-12


def sanitize_pred(yhat: np.ndarray) -> np.ndarray:
    """Clip and sanitize predictions."""
    yhat = np.asarray(yhat, dtype=float)
    yhat = np.clip(yhat, -1e15, 1e15)
    yhat[~np.isfinite(yhat)] = 0.0
    return yhat


def kfold_eval_fixed_model(predict_fn, X, y, n_splits=5, random_state=0):
    """Evaluate a fixed (already-trained) model on K validation folds.

    Parameters
    ----------
    predict_fn : callable
        ``predict_fn(X_subset) -> y_pred`` for a numpy array subset.
    X : np.ndarray (n, d)
    y : np.ndarray (n,)
    n_splits : int
    random_state : int

    Returns
    -------
    cv_loss : float   — mean MSE across folds
    cv_r2   : float   — mean R2 across folds
    instability : float — std(MSE) / mean(MSE)
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    losses, r2s = [], []
    for _, va in kf.split(X):
        yhat = sanitize_pred(predict_fn(X[va]))
        losses.append(mean_squared_error(y[va], yhat))
        try:
            r2s.append(r2_score(y[va], yhat))
        except Exception:
            r2s.append(np.nan)

    losses = np.asarray(losses, dtype=float)
    r2s = np.asarray(r2s, dtype=float)

    cv_loss = float(np.nanmean(losses))
    cv_r2 = float(np.nanmean(r2s))
    instability = float(np.nanstd(losses) / max(np.nanmean(losses), EPS))
    return cv_loss, cv_r2, instability
