"""Ensembling and stacking utilities for competition experiments."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.base import clone


def normalize_weights(weights: Sequence[float]) -> np.ndarray:
    """Normalize blend weights to sum to one."""
    w = np.asarray(weights, dtype=float)
    total = float(w.sum())
    if total <= 0:
        raise ValueError("weights must sum to a positive value")
    return w / total


def blend_predictions(pred_matrix: np.ndarray, weights: Sequence[float]) -> np.ndarray:
    """Weighted blend y_hat = sum_j w_j * y_hat_j."""
    preds = np.asarray(pred_matrix, dtype=float)
    if preds.ndim != 2:
        raise ValueError("pred_matrix must be 2D: [n_samples, n_models]")
    w = normalize_weights(weights)
    if preds.shape[1] != len(w):
        raise ValueError("weights length must match number of model columns")
    return preds @ w


def ensemble_variance_from_covariance(cov: np.ndarray, weights: Sequence[float]) -> float:
    """
    Var(sum_j w_j f_j) = w^T Sigma w
    where Sigma_{ij} = Cov(f_i, f_j).
    """
    sigma = np.asarray(cov, dtype=float)
    w = normalize_weights(weights)
    if sigma.shape[0] != sigma.shape[1] or sigma.shape[0] != len(w):
        raise ValueError("covariance shape must match weights length")
    return float(w.T @ sigma @ w)


def bagging_variance_formula(base_variance: float, pairwise_corr: float, n_models: int) -> float:
    """
    For equal-variance models with correlation rho:
      Var(avg) = sigma^2 * (rho + (1-rho)/M)
    """
    sigma2 = float(base_variance)
    rho = float(pairwise_corr)
    m = int(n_models)
    if m <= 0:
        raise ValueError("n_models must be >= 1")
    return sigma2 * (rho + (1.0 - rho) / m)


def oof_stacking(
    base_estimators: Sequence[tuple[str, object]],
    meta_estimator,
    X: np.ndarray,
    y: np.ndarray,
    splitter,
    task: str = "regression",
) -> dict[str, object]:
    """Train stacking pipeline with strict out-of-fold meta-features."""
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)
    n = len(y_arr)
    n_models = len(base_estimators)
    oof_meta = np.zeros((n, n_models), dtype=float)
    trained_base = [None] * n_models

    for tr_idx, va_idx in splitter.split(X_arr, y_arr):
        for model_idx, (_, estimator) in enumerate(base_estimators):
            est = clone(estimator)
            est.fit(X_arr[tr_idx], y_arr[tr_idx])
            if task == "classification" and hasattr(est, "predict_proba"):
                fold_pred = est.predict_proba(X_arr[va_idx])[:, 1]
            else:
                fold_pred = est.predict(X_arr[va_idx])
            oof_meta[va_idx, model_idx] = fold_pred

    meta = clone(meta_estimator)
    meta.fit(oof_meta, y_arr)

    for model_idx, (_, estimator) in enumerate(base_estimators):
        fit_model = clone(estimator)
        fit_model.fit(X_arr, y_arr)
        trained_base[model_idx] = fit_model

    return {
        "meta_model": meta,
        "base_models": trained_base,
        "oof_meta": oof_meta,
    }


def stacking_predict(
    stack_bundle: dict[str, object],
    X: np.ndarray,
    task: str = "regression",
) -> np.ndarray:
    """Predict from trained stacking bundle."""
    X_arr = np.asarray(X)
    base_models = stack_bundle["base_models"]
    meta_model = stack_bundle["meta_model"]
    meta_features = []
    for model in base_models:
        if task == "classification" and hasattr(model, "predict_proba"):
            meta_features.append(model.predict_proba(X_arr)[:, 1])
        else:
            meta_features.append(model.predict(X_arr))
    meta_X = np.column_stack(meta_features)
    if task == "classification" and hasattr(meta_model, "predict_proba"):
        return meta_model.predict_proba(meta_X)[:, 1]
    return meta_model.predict(meta_X)
