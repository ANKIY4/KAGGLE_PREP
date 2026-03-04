"""Reusable feature-engineering helpers for the advanced Kaggle module."""

from __future__ import annotations

import random
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures


def set_global_seed(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def standardize_train_valid(
    train: np.ndarray, valid: np.ndarray, eps: float = 1e-12
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Standardize train/valid arrays using train statistics only."""
    train = np.asarray(train, dtype=float)
    valid = np.asarray(valid, dtype=float)
    mu = train.mean(axis=0)
    sigma = train.std(axis=0)
    sigma = np.where(sigma < eps, 1.0, sigma)
    return (train - mu) / sigma, (valid - mu) / sigma, mu, sigma


def monotone_log1p(x: np.ndarray | pd.Series, clip_min: float = 0.0) -> np.ndarray:
    """Apply a monotone log(1+x) transform with clipping for safety."""
    arr = np.asarray(x, dtype=float)
    clipped = np.maximum(arr, clip_min)
    return np.log1p(clipped)


def one_hot_basis(values: Sequence[str]) -> tuple[np.ndarray, OneHotEncoder]:
    """Return one-hot basis expansion and fitted encoder."""
    series = pd.Series(values).astype(str).to_frame(name="category")
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # sklearn < 1.2
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    encoded = encoder.fit_transform(series)
    return np.asarray(encoded, dtype=float), encoder


def polynomial_basis(
    X: np.ndarray, degree: int = 2, include_bias: bool = False
) -> tuple[np.ndarray, PolynomialFeatures]:
    """Generate a polynomial basis expansion."""
    poly = PolynomialFeatures(degree=degree, include_bias=include_bias)
    out = poly.fit_transform(np.asarray(X, dtype=float))
    return np.asarray(out, dtype=float), poly


def add_interaction_columns(df: pd.DataFrame, pairs: Iterable[tuple[str, str]]) -> pd.DataFrame:
    """Add pairwise interaction columns x_i * x_j for selected feature pairs."""
    out = df.copy()
    for left, right in pairs:
        out[f"{left}__x__{right}"] = out[left].astype(float) * out[right].astype(float)
    return out


def target_encode_oof(
    categories: pd.Series,
    target: np.ndarray | pd.Series,
    n_splits: int = 5,
    smooth: float = 20.0,
    seed: int = 42,
) -> np.ndarray:
    """
    Leakage-safe out-of-fold target encoding with additive smoothing.
    Encoded value for category c in fold k:
        (n_c * mean_c + smooth * global_mean) / (n_c + smooth)
    """
    cats = categories.astype(str).reset_index(drop=True)
    y = np.asarray(target, dtype=float).reshape(-1)
    if len(cats) != len(y):
        raise ValueError("categories and target must have the same length")

    global_mean = float(y.mean())
    encoded = np.zeros(len(y), dtype=float)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for tr_idx, va_idx in kf.split(cats):
        tr_cat = cats.iloc[tr_idx]
        tr_y = y[tr_idx]
        stats = (
            pd.DataFrame({"cat": tr_cat, "y": tr_y})
            .groupby("cat")["y"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "cat_mean", "count": "cat_count"})
        )
        smooth_mean = (
            stats["cat_mean"] * stats["cat_count"] + smooth * global_mean
        ) / (stats["cat_count"] + smooth)
        fold_values = cats.iloc[va_idx].map(smooth_mean).fillna(global_mean).to_numpy(dtype=float)
        encoded[va_idx] = fold_values
    return encoded


def conditional_expectation_feature(
    df: pd.DataFrame, group_col: str, value_col: str, smooth: float = 10.0
) -> pd.Series:
    """Create a smoothed conditional expectation feature E[value | group]."""
    work = df[[group_col, value_col]].copy()
    global_mean = float(work[value_col].mean())
    grouped = work.groupby(group_col)[value_col].agg(["mean", "count"])
    smooth_mean = (grouped["mean"] * grouped["count"] + smooth * global_mean) / (
        grouped["count"] + smooth
    )
    return work[group_col].map(smooth_mean).fillna(global_mean)
