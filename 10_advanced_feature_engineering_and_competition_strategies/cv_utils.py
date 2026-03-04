"""Cross-validation helpers for leakage-safe competition workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedKFold,
    TimeSeriesSplit,
)


def empirical_risk(
    y_true: np.ndarray, y_pred: np.ndarray, loss: str = "mse"
) -> float:
    """Compute empirical risk R_hat = (1/n) * sum_i L(y_i, yhat_i)."""
    y_t = np.asarray(y_true).reshape(-1)
    y_p = np.asarray(y_pred).reshape(-1)
    if y_t.shape[0] != y_p.shape[0]:
        raise ValueError("y_true and y_pred must have equal length")

    if loss == "mse":
        return float(np.mean((y_t - y_p) ** 2))
    if loss == "mae":
        return float(np.mean(np.abs(y_t - y_p)))
    if loss == "logloss":
        eps = 1e-12
        p = np.clip(y_p, eps, 1 - eps)
        return float(-np.mean(y_t * np.log(p) + (1 - y_t) * np.log(1 - p)))
    raise ValueError(f"Unsupported loss: {loss}")


def make_splitter(
    split_kind: str,
    n_splits: int = 5,
    seed: int = 42,
) -> object:
    """Factory for common CV splitter strategies."""
    if split_kind == "kfold":
        return KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    if split_kind == "stratified":
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    if split_kind == "group":
        return GroupKFold(n_splits=n_splits)
    if split_kind == "timeseries":
        return TimeSeriesSplit(n_splits=n_splits)
    raise ValueError(f"Unknown split_kind: {split_kind}")


@dataclass
class CVResult:
    fold_scores: list[float]
    mean: float
    std: float
    oof_pred: np.ndarray


def _score_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task: str,
    metric: str,
) -> float:
    if task == "regression":
        if metric == "rmse":
            return float(np.sqrt(mean_squared_error(y_true, y_pred)))
        return float(mean_squared_error(y_true, y_pred))

    if metric == "auc":
        return float(roc_auc_score(y_true, y_pred))
    labels = (y_pred >= 0.5).astype(int)
    return float(accuracy_score(y_true, labels))


def oof_cv_predictions(
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    splitter,
    task: str = "regression",
    metric: str = "rmse",
    groups: np.ndarray | None = None,
) -> CVResult:
    """Compute out-of-fold predictions and fold-wise scores."""
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)
    oof = np.zeros(len(y_arr), dtype=float)
    fold_scores: list[float] = []

    if groups is None:
        split_iter = splitter.split(X_arr, y_arr)
    else:
        split_iter = splitter.split(X_arr, y_arr, groups=groups)

    for tr_idx, va_idx in split_iter:
        est = clone(estimator)
        est.fit(X_arr[tr_idx], y_arr[tr_idx])
        if task == "classification" and hasattr(est, "predict_proba"):
            pred = est.predict_proba(X_arr[va_idx])[:, 1]
        else:
            pred = est.predict(X_arr[va_idx])
        oof[va_idx] = pred
        fold_scores.append(_score_predictions(y_arr[va_idx], pred, task=task, metric=metric))

    return CVResult(
        fold_scores=fold_scores,
        mean=float(np.mean(fold_scores)),
        std=float(np.std(fold_scores)),
        oof_pred=oof,
    )


def cv_bias_variance_decomposition(fold_scores: Iterable[float]) -> dict[str, float]:
    """
    Approximate estimator bias/variance decomposition around fold mean:
      bias proxy = |E[score] - median(score)|
      variance = Var(score)
    """
    s = np.asarray(list(fold_scores), dtype=float)
    return {
        "mean": float(np.mean(s)),
        "median": float(np.median(s)),
        "bias_proxy": float(abs(np.mean(s) - np.median(s))),
        "variance": float(np.var(s)),
        "std": float(np.std(s)),
    }


def leakage_inflation(clean_score: float, leaky_score: float) -> float:
    """Positive value means the leaky pipeline looked artificially better."""
    return float(clean_score - leaky_score)


def simulate_public_private_variance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    public_fraction: float = 0.5,
    n_trials: int = 200,
    seed: int = 42,
) -> dict[str, float]:
    """Bootstrap leaderboard shake between public and private subsets."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    n_public = int(n * public_fraction)
    gaps = []
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)

    for _ in range(n_trials):
        perm = rng.permutation(n)
        pub_idx = perm[:n_public]
        prv_idx = perm[n_public:]
        public_score = metric(y_t[pub_idx], y_p[pub_idx])
        private_score = metric(y_t[prv_idx], y_p[prv_idx])
        gaps.append(private_score - public_score)

    gaps_arr = np.asarray(gaps, dtype=float)
    return {
        "gap_mean": float(gaps_arr.mean()),
        "gap_std": float(gaps_arr.std()),
        "gap_abs_p90": float(np.quantile(np.abs(gaps_arr), 0.90)),
    }
