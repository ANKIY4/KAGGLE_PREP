"""Generate advanced competition-strategy notebooks and solution notebooks."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


MODULE_DIR = Path(__file__).resolve().parent


COMMON_SETUP_CODE = """
from pathlib import Path
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

import torch
import torch.nn as nn

from sklearn.datasets import load_diabetes, load_breast_cancer, load_wine, fetch_california_housing
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, roc_auc_score, accuracy_score
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, HistGradientBoostingRegressor, HistGradientBoostingClassifier

try:
    from xgboost import XGBRegressor, XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

try:
    from lightgbm import LGBMRegressor, LGBMClassifier
    LGBM_AVAILABLE = True
except Exception:
    LGBM_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except Exception:
    OPTUNA_AVAILABLE = False

from feature_utils import (
    set_global_seed,
    standardize_train_valid,
    monotone_log1p,
    one_hot_basis,
    polynomial_basis,
    add_interaction_columns,
    target_encode_oof,
    conditional_expectation_feature,
)
from cv_utils import (
    empirical_risk,
    make_splitter,
    oof_cv_predictions,
    cv_bias_variance_decomposition,
    leakage_inflation,
    simulate_public_private_variance,
)
from ensemble_utils import (
    blend_predictions,
    bagging_variance_formula,
    ensemble_variance_from_covariance,
    oof_stacking,
    stacking_predict,
)
from experiment_logger import ExperimentLogger, ExperimentRecord

SEED = 42
set_global_seed(SEED)
"""


def add_md(nb, text: str) -> None:
    nb.cells.append(nbf.v4.new_markdown_cell(text.strip()))


def add_code(nb, code: str) -> None:
    nb.cells.append(nbf.v4.new_code_cell(code.strip()))


def new_nb() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata.kernelspec = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata.language_info = {"name": "python"}
    return nb


def make_topics() -> list[dict[str, str]]:
    return [
        {
            "id": "10",
            "slug": "feature_engineering_fundamentals",
            "title": "Feature Engineering Fundamentals",
            "prior": "- Linear algebra basics: vectors, norms, and basis representation.\n- Probability basics: expectation and variance.\n- Supervised learning setup $(x_i, y_i)$.",
            "new": "- Standardization and variance normalization derivation.\n- Monotone transforms and log transform behavior.\n- One-hot basis expansion.\n- Target encoding as conditional expectation $\\mathbb{E}[Y\\mid C]$.\n- Leakage-safe out-of-fold target encoding estimator.\n- Bias-variance implications of engineered features.",
            "problem": "Construct mathematically grounded tabular feature maps that improve predictive signal without creating leakage.",
            "math": r"""
### Mathematical Foundations
We define a feature map $\\phi: \\mathbb{R}^d \\to \\mathbb{R}^p$ and model $f_\\theta$ such that
$$
\\hat{y}_i = f_\\theta(\\phi(x_i)).
$$

**Standardization (per feature $j$):**
$$
z_{ij} = \\frac{x_{ij}-\\mu_j}{\\sigma_j},\\qquad
\\mu_j = \\frac{1}{n}\\sum_{i=1}^n x_{ij},\\qquad
\\sigma_j^2 = \\frac{1}{n}\\sum_{i=1}^n (x_{ij}-\\mu_j)^2.
$$
Then
$$
\\frac{1}{n}\\sum_{i=1}^n z_{ij}=0,\\qquad
\\frac{1}{n}\\sum_{i=1}^n z_{ij}^2=1.
$$

**Gradient scale implication (squared loss, linear model):**
$$
\\hat{y}_i = w^\\top z_i,\\quad
\\mathcal{L}(w)=\\frac{1}{n}\\sum_{i=1}^n(\\hat{y}_i-y_i)^2,\\quad
\\frac{\\partial \\mathcal{L}}{\\partial w_j}
=\\frac{2}{n}\\sum_{i=1}^n(\\hat{y}_i-y_i)z_{ij}.
$$
When $z_{ij}$ are on comparable scales, gradient magnitudes across coordinates are balanced.

**Target encoding (category $c$):**
$$
\\operatorname{TE}(c)=\\mathbb{E}[Y\\mid C=c].
$$
Empirical smoothed estimator:
$$
\\widehat{\\operatorname{TE}}(c)=
\\frac{n_c\\bar{y}_c+\\alpha\\bar{y}}{n_c+\\alpha},
$$
where $n_c$ is category count, $\\bar{y}_c$ category mean target, $\\bar{y}$ global mean, and $\\alpha>0$ smoothing.
""",
            "derivation": r"""
### Step-by-Step Derivation
1. Start from raw feature $x_{ij}$ and define centered value:
   $$
   \\tilde{x}_{ij}=x_{ij}-\\mu_j.
   $$
2. Define normalized value:
   $$
   z_{ij}=\\tilde{x}_{ij}/\\sigma_j.
   $$
3. Mean check:
   $$
   \\frac{1}{n}\\sum_i z_{ij}
   =\\frac{1}{n\\sigma_j}\\sum_i(x_{ij}-\\mu_j)
   =\\frac{1}{n\\sigma_j}(n\\mu_j-n\\mu_j)=0.
   $$
4. Variance check:
   $$
   \\frac{1}{n}\\sum_i z_{ij}^2
   =\\frac{1}{n\\sigma_j^2}\\sum_i(x_{ij}-\\mu_j)^2=1.
   $$
5. For target encoding, split data into folds. For each validation fold, compute category statistics only on training folds:
   $$
   \\widehat{\\operatorname{TE}}_k(c)
   =\\frac{n_{c,-k}\\bar{y}_{c,-k}+\\alpha\\bar{y}_{-k}}{n_{c,-k}+\\alpha}.
   $$
   This removes direct access to each sample's own target and prevents leakage.
""",
            "intuition": "Standardization improves conditioning, one-hot turns category IDs into basis vectors, and leakage-safe target encoding approximates category signal without peeking at validation targets.",
            "code_map": "- `standardize_train_valid` implements train-only centering/scaling.\n- `one_hot_basis` performs basis expansion.\n- `target_encode_oof` computes fold-wise conditional expectation estimates.\n- Synthetic and real-data sections compare leakage-safe vs leaky encoders.",
            "synthetic_code": """
from sklearn.datasets import make_regression

X_num, y = make_regression(n_samples=5000, n_features=4, noise=15.0, random_state=SEED)
rng = np.random.default_rng(SEED)
cats = pd.Series(rng.choice(["A", "B", "C", "D"], size=len(y)))

# Create category-dependent shift in target.
cat_shift = cats.map({"A": 0.0, "B": 20.0, "C": -15.0, "D": 35.0}).to_numpy()
y = y + cat_shift

X_df = pd.DataFrame(X_num, columns=[f"x{i}" for i in range(X_num.shape[1])])
X_df["cat"] = cats
X_df["x0_log"] = monotone_log1p(np.abs(X_df["x0"]))

X_train, X_valid, y_train, y_valid = train_test_split(X_df, y, test_size=0.25, random_state=SEED)
Xtr_std, Xva_std, mu, sigma = standardize_train_valid(X_train[[f"x{i}" for i in range(4)]], X_valid[[f"x{i}" for i in range(4)]])

te_oof_train = target_encode_oof(X_train["cat"], y_train, n_splits=5, smooth=30.0, seed=SEED)
full_stats = pd.DataFrame({"cat": X_train["cat"], "y": y_train}).groupby("cat")["y"].mean()
te_leaky_train = X_train["cat"].map(full_stats).to_numpy(dtype=float)
te_valid = X_valid["cat"].map(full_stats).fillna(np.mean(y_train)).to_numpy(dtype=float)

reg_clean = LinearRegression().fit(np.column_stack([Xtr_std, te_oof_train]), y_train)
reg_leaky = LinearRegression().fit(np.column_stack([Xtr_std, te_leaky_train]), y_train)

pred_clean = reg_clean.predict(np.column_stack([Xva_std, te_valid]))
pred_leaky = reg_leaky.predict(np.column_stack([Xva_std, te_valid]))

rmse_clean = np.sqrt(mean_squared_error(y_valid, pred_clean))
rmse_leaky = np.sqrt(mean_squared_error(y_valid, pred_leaky))
print({"rmse_clean": rmse_clean, "rmse_leaky_train_fit": rmse_leaky})
""",
            "real_code": """
diab = load_diabetes(as_frame=True)
X_real = diab.data.copy()
y_real = diab.target.to_numpy(dtype=float)

X_real["bmi_log"] = monotone_log1p(np.maximum(X_real["bmi"], 0.0))
X_tr, X_va, y_tr, y_va = train_test_split(X_real, y_real, test_size=0.25, random_state=SEED)
Xtr_std, Xva_std, _, _ = standardize_train_valid(X_tr.values, X_va.values)

ridge = Ridge(alpha=1.0).fit(Xtr_std, y_tr)
pred = ridge.predict(Xva_std)
rmse = np.sqrt(mean_squared_error(y_va, pred))
print({"diabetes_ridge_rmse": rmse})
""",
            "diagnostic_code": """
coef = pd.Series(ridge.coef_, index=X_tr.columns).sort_values(key=np.abs, ascending=False)
print(coef.head(8))
plt.figure(figsize=(7, 3))
plt.bar(coef.head(8).index, coef.head(8).values)
plt.xticks(rotation=45)
plt.title("Top coefficient magnitudes (standardized features)")
plt.tight_layout()
plt.show()
""",
            "failure_code": """
# Failure case: fit scaling statistics on train+valid (data leakage).
full_std = StandardScaler().fit(pd.concat([X_tr, X_va], axis=0))
Xtr_bad = full_std.transform(X_tr)
Xva_bad = full_std.transform(X_va)
ridge_bad = Ridge(alpha=1.0).fit(Xtr_bad, y_tr)
pred_bad = ridge_bad.predict(Xva_bad)
rmse_bad = np.sqrt(mean_squared_error(y_va, pred_bad))
print({"rmse_leaky_scaler": rmse_bad, "rmse_clean_scaler": rmse})
""",
            "exercises": "1. Prove that z-score standardization is invariant to feature translation.\n2. Derive the gradient expression for MAE under standardized inputs.\n3. Implement James-Stein style shrinkage target encoding and compare with additive smoothing.\n4. Show a counterexample where a monotone transform hurts linear separability.",
            "summary": "Feature engineering is a controlled transformation of representation geometry; mathematically valid transforms plus leakage-safe estimation reduce bias without introducing false validation lift.",
        },
        {
            "id": "11",
            "slug": "advanced_feature_engineering_patterns",
            "title": "Advanced Feature Engineering Patterns",
            "prior": "- Notebook 10 concepts: standardization, conditional expectations, leakage-safe estimators.\n- Multivariate polynomial notation and combinatorics basics.",
            "new": "- Polynomial basis expansion in $\\mathbb{R}^d$.\n- Interaction terms as tensor products.\n- Curse of dimensionality counting argument.\n- Aggregation features as conditional expectations.\n- SHAP from Shapley values with explicit combinatorial weights.",
            "problem": "Design high-capacity feature maps while controlling dimensionality, variance, and interpretability.",
            "math": r"""
### Mathematical Foundations
For $x\\in\\mathbb{R}^d$, the degree-$p$ polynomial basis includes monomials
$$
x_1^{\\alpha_1}\\cdots x_d^{\\alpha_d},\\qquad
\\alpha_j\\in\\mathbb{N}_0,\\quad \\sum_{j=1}^d \\alpha_j \\le p.
$$
Number of monomials:
$$
N(d,p)=\\binom{d+p}{p}.
$$

**Tensor-product interaction (pairwise):**
$$
\\psi_{ij}(x)=x_i x_j.
$$

**Aggregation as conditional expectation:**
$$
g(x)=\\mathbb{E}[Y\\mid G(x)],
$$
where $G(x)$ maps samples into groups.

**Shapley value for feature $j$:**
$$
\\phi_j(v)=\\sum_{S\\subseteq N\\setminus\\{j\\}}
\\frac{|S|!(M-|S|-1)!}{M!}\\Big(v(S\\cup\\{j\\})-v(S)\\Big).
$$
""",
            "derivation": r"""
### Step-by-Step Derivation
1. Count degree-exact-$k$ monomials by stars-and-bars:
   $$
   \\#\\{\\alpha: \\sum_j \\alpha_j=k\\}=\\binom{d+k-1}{k}.
   $$
2. Sum $k=0$ to $p$:
   $$
   N(d,p)=\\sum_{k=0}^{p}\\binom{d+k-1}{k}=\\binom{d+p}{p}.
   $$
3. Curse-of-dimensionality consequence: if $d=50, p=3$, then
   $$
   N(50,3)=\\binom{53}{3}=23426,
   $$
   which can exceed sample-efficient regimes.
4. Shapley combinatorial weight is the fraction of feature-order permutations where subset $S$ appears before feature $j$:
   $$
   w(S)=\\frac{|S|!(M-|S|-1)!}{M!}.
   $$
5. Weighted marginal contributions sum to total attribution and satisfy efficiency:
   $$
   \\sum_{j=1}^M \\phi_j = v(N)-v(\\varnothing).
   $$
""",
            "intuition": "Polynomial and interaction features increase expressive power; Shapley values distribute model output contribution fairly across features with permutation-based weighting.",
            "code_map": "- `polynomial_basis` builds monomial expansion.\n- `add_interaction_columns` creates selected tensor-product terms.\n- Aggregation features are implemented through smoothed conditional expectation.\n- SHAP derivation is validated with exact enumeration on a tiny cooperative game.",
            "synthetic_code": """
from sklearn.datasets import make_regression

X_syn, y_syn = make_regression(n_samples=2500, n_features=6, noise=18.0, random_state=SEED)
X_df = pd.DataFrame(X_syn, columns=[f"f{i}" for i in range(X_syn.shape[1])])

poly_X, poly = polynomial_basis(X_df.values, degree=2, include_bias=False)
base_rmse = np.sqrt(mean_squared_error(y_syn, LinearRegression().fit(X_df.values, y_syn).predict(X_df.values)))
poly_rmse = np.sqrt(mean_squared_error(y_syn, Ridge(alpha=2.0).fit(poly_X, y_syn).predict(poly_X)))

print({"poly_feature_count": poly_X.shape[1], "base_fit_rmse": base_rmse, "poly_fit_rmse": poly_rmse})
""",
            "real_code": """
bc = load_breast_cancer(as_frame=True)
X_real = bc.data.copy()
y_real = bc.target.to_numpy(dtype=int)

X_small = X_real.iloc[:, :8].copy()
X_small = add_interaction_columns(X_small, [("mean radius", "mean texture"), ("mean area", "mean smoothness")])
X_train, X_valid, y_train, y_valid = train_test_split(X_small, y_real, test_size=0.25, random_state=SEED, stratify=y_real)

scaler = StandardScaler().fit(X_train)
Xt = scaler.transform(X_train)
Xv = scaler.transform(X_valid)

clf = LogisticRegression(max_iter=2000).fit(Xt, y_train)
proba = clf.predict_proba(Xv)[:, 1]
auc = roc_auc_score(y_valid, proba)
print({"breast_cancer_auc": auc, "n_features_after_interactions": X_small.shape[1]})
""",
            "diagnostic_code": """
# Exact Shapley demonstration on 3-feature toy value function.
import itertools
from math import factorial

def v(S):
    S = set(S)
    score = 0.0
    if "a" in S: score += 2.0
    if "b" in S: score += 1.0
    if "c" in S: score += 1.5
    if "a" in S and "b" in S: score += 0.8
    return score

features = ["a", "b", "c"]
M = len(features)
phi = {}
for j in features:
    others = [f for f in features if f != j]
    contrib = 0.0
    for r in range(len(others) + 1):
        for subset in itertools.combinations(others, r):
            w = factorial(len(subset)) * factorial(M - len(subset) - 1) / factorial(M)
            contrib += w * (v(set(subset) | {j}) - v(set(subset)))
    phi[j] = contrib

print("Shapley values:", phi, "sum=", sum(phi.values()), "total=", v(features) - v(set()))
""",
            "failure_code": """
# Failure case: uncontrolled high-degree expansion creates severe overfitting.
poly_high, _ = polynomial_basis(X_df.values, degree=4, include_bias=False)
model_overfit = LinearRegression().fit(poly_high, y_syn)
pred_overfit = model_overfit.predict(poly_high)
rmse_overfit = np.sqrt(mean_squared_error(y_syn, pred_overfit))
print({"degree4_feature_count": poly_high.shape[1], "apparent_train_rmse": rmse_overfit})
""",
            "exercises": "1. Derive the feature count formula for degree-exact-$p$ without the summation step.\n2. Prove Shapley efficiency and symmetry axioms for the 3-feature case.\n3. Show when aggregation features reduce variance but increase bias.\n4. Build a sparse polynomial map and compare with dense expansion.",
            "summary": "Advanced feature maps expand hypothesis space; controlling dimension growth and attribution discipline is necessary to keep improvements generalizable.",
        },
        {
            "id": "12",
            "slug": "cross_validation_mastery",
            "title": "Cross-Validation Mastery",
            "prior": "- Notebooks 10-11 feature-map design and leakage-safe estimation habits.\n- Basic estimators and train/validation splitting.",
            "new": "- Empirical risk estimator definition.\n- Bias and variance of CV estimators.\n- KFold/StratifiedKFold/GroupKFold/TimeSeriesSplit.\n- Nested CV.\n- Out-of-fold prediction protocol.\n- Leakage inflation and leaderboard shake simulation.",
            "problem": "Estimate private leaderboard performance robustly under finite-sample uncertainty and split constraints.",
            "math": r"""
### Mathematical Foundations
Empirical risk:
$$
\\hat{R}(f)=\\frac{1}{n}\\sum_{i=1}^n L\\big(y_i,f(x_i)\\big).
$$
K-fold estimator:
$$
\\hat{R}_{\\mathrm{CV}}(f)=\\frac{1}{K}\\sum_{k=1}^K
\\frac{1}{|V_k|}\\sum_{i\\in V_k}L\\big(y_i,f^{(-k)}(x_i)\\big).
$$

Bias-variance decomposition of the estimator:
$$
\\mathbb{E}\\big[(\\hat{R}_{\\mathrm{CV}}-R)^2\\big]
=\\big(\\mathbb{E}[\\hat{R}_{\\mathrm{CV}}]-R\\big)^2
+\\mathrm{Var}(\\hat{R}_{\\mathrm{CV}}).
$$

Nested CV objective:
$$
\\hat{R}_{\\text{nested}}
=\\frac{1}{K_{\\text{outer}}}\\sum_{k=1}^{K_{\\text{outer}}}
L\\big(y_{V_k}, f_{\\lambda_k^*}^{(-k)}(x_{V_k})\\big),\\quad
\\lambda_k^*=\\arg\\min_{\\lambda\\in\\Lambda}\\hat{R}_{\\text{inner}}^{(k)}(\\lambda).
$$
""",
            "derivation": r"""
### Step-by-Step Derivation
1. Partition index set $\\{1,\\dots,n\\}$ into disjoint folds $V_1,\\dots,V_K$.
2. For each fold $k$, fit model on complement $T_k=\\{1,\\dots,n\\}\\setminus V_k$.
3. Compute fold loss average:
   $$
   \\hat{R}_k=\\frac{1}{|V_k|}\\sum_{i\\in V_k}L(y_i,f^{(-k)}(x_i)).
   $$
4. Average fold estimates:
   $$
   \\hat{R}_{\\mathrm{CV}}=\\frac{1}{K}\\sum_{k=1}^{K}\\hat{R}_k.
   $$
5. For leakage inflation, if $\\hat{R}_{\\text{leaky}}<\\hat{R}_{\\text{clean}}$ due invalid preprocessing, inflation is
   $$
   \\Delta_{\\text{leak}}=\\hat{R}_{\\text{clean}}-\\hat{R}_{\\text{leaky}}>0.
   $$
""",
            "intuition": "CV is a Monte Carlo-like estimator of deployment risk; splitter choice encodes assumptions about label balance, groups, and temporal ordering.",
            "code_map": "- `make_splitter` selects split strategy.\n- `oof_cv_predictions` implements strict out-of-fold protocol.\n- `cv_bias_variance_decomposition` summarizes estimator stability.\n- `simulate_public_private_variance` approximates leaderboard shake.",
            "synthetic_code": """
from sklearn.datasets import make_classification

X_syn, y_syn = make_classification(
    n_samples=3000, n_features=20, n_informative=8, n_redundant=4, random_state=SEED, weights=[0.7, 0.3]
)
groups = (np.arange(len(y_syn)) // 30).astype(int)
time_index = np.arange(len(y_syn))

model = LogisticRegression(max_iter=2000)
for split_kind in ["kfold", "stratified", "group", "timeseries"]:
    splitter = make_splitter(split_kind, n_splits=5, seed=SEED)
    if split_kind == "group":
        out = oof_cv_predictions(model, X_syn, y_syn, splitter, task="classification", metric="auc", groups=groups)
    else:
        out = oof_cv_predictions(model, X_syn, y_syn, splitter, task="classification", metric="auc")
    print(split_kind, {"mean_auc": out.mean, "std_auc": out.std})
""",
            "real_code": """
wine = load_wine(as_frame=True)
X = wine.data.values
y = (wine.target.values == 0).astype(int)

outer = make_splitter("stratified", n_splits=5, seed=SEED)
outer_scores = []
for tr_idx, va_idx in outer.split(X, y):
    X_tr, X_va = X[tr_idx], X[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    # Inner model selection (small grid) = nested CV core idea.
    candidates = [0.1, 1.0, 5.0]
    inner = make_splitter("stratified", n_splits=4, seed=SEED)
    best_c, best_inner = None, -1
    for c in candidates:
        clf = LogisticRegression(max_iter=2000, C=c)
        inner_out = oof_cv_predictions(clf, X_tr, y_tr, inner, task="classification", metric="auc")
        if inner_out.mean > best_inner:
            best_inner, best_c = inner_out.mean, c

    final = LogisticRegression(max_iter=2000, C=best_c).fit(X_tr, y_tr)
    auc = roc_auc_score(y_va, final.predict_proba(X_va)[:, 1])
    outer_scores.append(auc)

print({"nested_auc_mean": float(np.mean(outer_scores)), "nested_auc_std": float(np.std(outer_scores))})
""",
            "diagnostic_code": """
splitter = make_splitter("stratified", n_splits=5, seed=SEED)
out = oof_cv_predictions(LogisticRegression(max_iter=2000), X_syn, y_syn, splitter, task="classification", metric="auc")
diag = cv_bias_variance_decomposition(out.fold_scores)
print(diag)

def auc_metric(y_true, y_pred):
    return roc_auc_score(y_true, y_pred)

shake = simulate_public_private_variance(y_syn, out.oof_pred, auc_metric, public_fraction=0.5, n_trials=200, seed=SEED)
print("public-private shake diagnostics:", shake)
""",
            "failure_code": """
# Failure case: leakage by fitting scaler once on all data before folds.
scaler = StandardScaler().fit(X_syn)
X_bad = scaler.transform(X_syn)
bad_out = oof_cv_predictions(LogisticRegression(max_iter=2000), X_bad, y_syn, make_splitter("stratified", 5, SEED), task="classification", metric="auc")

# Clean: scaler fit inside fold.
clean_scores = []
splitter = make_splitter("stratified", 5, SEED)
for tr_idx, va_idx in splitter.split(X_syn, y_syn):
    sc = StandardScaler().fit(X_syn[tr_idx])
    Xtr = sc.transform(X_syn[tr_idx]); Xva = sc.transform(X_syn[va_idx])
    clf = LogisticRegression(max_iter=2000).fit(Xtr, y_syn[tr_idx])
    clean_scores.append(roc_auc_score(y_syn[va_idx], clf.predict_proba(Xva)[:, 1]))

print({"leaky_auc": bad_out.mean, "clean_auc": float(np.mean(clean_scores)), "inflation": leakage_inflation(float(np.mean(clean_scores)), bad_out.mean)})
""",
            "exercises": "1. Derive variance reduction trend as K increases under independent fold errors.\n2. Show when GroupKFold is strictly necessary.\n3. Implement repeated CV and compare estimator variance.\n4. Simulate rank instability under tiny public split size.",
            "summary": "Cross-validation is an estimator design problem: split assumptions, nesting, and leakage controls determine whether local gains transfer to private leaderboard reality.",
        },
        {
            "id": "13",
            "slug": "data_leakage_detection",
            "title": "Data Leakage Detection",
            "prior": "- Notebook 12 cross-validation estimators and OOF protocol.\n- Probability densities and cumulative distribution functions.",
            "new": "- Distribution-shift formalism $P_{train}(X)\\ne P_{test}(X)$.\n- KL divergence derivation.\n- Kolmogorov-Smirnov statistic derivation.\n- Adversarial validation for covariate shift.\n- Synthetic leakage stress testing.",
            "problem": "Detect hidden leakage and shift before leaderboard submission.",
            "math": r"""
### Mathematical Foundations
Distribution shift:
$$
P_{\\text{train}}(X)\\neq P_{\\text{test}}(X).
$$
KL divergence:
$$
D_{\\mathrm{KL}}(P\\parallel Q)=\\int p(x)\\log\\frac{p(x)}{q(x)}\\,dx.
$$
Empirical KS statistic for two samples:
$$
D_{n,m}=\\sup_x\\left|F_n(x)-G_m(x)\\right|.
$$
Adversarial validation solves:
$$
\\text{AUC}\\big(h(x),s\\big),\\quad s\\in\\{0,1\\},
$$
where $s=0$ indicates train and $s=1$ indicates test-like split.
High AUC implies strong covariate distinguishability.
""",
            "derivation": r"""
### Step-by-Step Derivation
1. KL from expected log density ratio:
   $$
   D_{\\mathrm{KL}}(P\\parallel Q)=\\mathbb{E}_{x\\sim P}\\left[\\log p(x)-\\log q(x)\\right].
   $$
2. If $p=q$, integrand is zero everywhere so KL is zero.
3. KS compares empirical CDFs:
   $$
   F_n(x)=\\frac{1}{n}\\sum_{i=1}^n\\mathbf{1}\\{X_i\\le x\\},\\quad
   G_m(x)=\\frac{1}{m}\\sum_{j=1}^m\\mathbf{1}\\{Y_j\\le x\\}.
   $$
4. Maximum absolute vertical distance gives $D_{n,m}$.
5. Leakage can be viewed as hidden variable $z$ carrying label information into features:
   $$
   I(Y;Z\\mid \\text{split})>0
   $$
   where $Z$ should be unavailable at prediction time.
""",
            "intuition": "Shift metrics and adversarial validation are early warning systems: if train and test are separable, naive CV may be over-optimistic.",
            "code_map": "- KL and KS are computed on selected feature marginals.\n- Adversarial validation trains a classifier to separate train/test domains.\n- Synthetic leakage introduces a near-target proxy to verify detection diagnostics.",
            "synthetic_code": """
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=4000, n_features=10, noise=20.0, random_state=SEED)
rng = np.random.default_rng(SEED)

# Create explicit leakage feature.
leak = y + rng.normal(0, 5.0, size=len(y))
X_leaky = np.column_stack([X, leak])

X_train, X_test, y_train, y_test = train_test_split(X_leaky, y, test_size=0.3, random_state=SEED)
model = LinearRegression().fit(X_train, y_train)
pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, pred))
print({"rmse_with_leak": rmse})
""",
            "real_code": """
# Hybrid policy: try California Housing, fallback to Diabetes if unavailable.
try:
    cal = fetch_california_housing(as_frame=True)
    X_real = cal.data.copy()
    y_real = cal.target.to_numpy(dtype=float)
except Exception:
    diab = load_diabetes(as_frame=True)
    X_real = diab.data.copy()
    y_real = diab.target.to_numpy(dtype=float)

X_tr, X_te, y_tr, y_te = train_test_split(X_real, y_real, test_size=0.35, random_state=SEED)

# Adversarial validation: classify split origin.
adv_X = np.vstack([X_tr.values, X_te.values])
adv_y = np.concatenate([np.zeros(len(X_tr)), np.ones(len(X_te))])
adv_model = HistGradientBoostingClassifier(random_state=SEED).fit(adv_X, adv_y)
adv_auc = roc_auc_score(adv_y, adv_model.predict_proba(adv_X)[:, 1])
print({"adversarial_auc_split_detection": adv_auc})
""",
            "diagnostic_code": """
def empirical_kl(p_vals, q_vals, bins=50):
    p_hist, edges = np.histogram(p_vals, bins=bins, density=True)
    q_hist, _ = np.histogram(q_vals, bins=edges, density=True)
    eps = 1e-10
    p = np.clip(p_hist, eps, None); q = np.clip(q_hist, eps, None)
    p = p / p.sum(); q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))

col = X_tr.columns[0]
kl = empirical_kl(X_tr[col].to_numpy(), X_te[col].to_numpy(), bins=40)

def ks_statistic(a, b):
    a = np.sort(np.asarray(a)); b = np.sort(np.asarray(b))
    grid = np.sort(np.unique(np.concatenate([a, b])))
    Fa = np.searchsorted(a, grid, side="right") / len(a)
    Fb = np.searchsorted(b, grid, side="right") / len(b)
    return float(np.max(np.abs(Fa - Fb)))

ks = ks_statistic(X_tr[col].to_numpy(), X_te[col].to_numpy())
print({"feature": col, "KL": kl, "KS": ks})
""",
            "failure_code": """
# Failure case: random split hides temporal/group leakage structure.
perm = np.random.default_rng(SEED).permutation(len(y_real))
cut = int(0.7 * len(perm))
idx_train, idx_test = perm[:cut], perm[cut:]
X_bad_train, X_bad_test = X_real.iloc[idx_train], X_real.iloc[idx_test]
y_bad_train, y_bad_test = y_real[idx_train], y_real[idx_test]

bad_model = HistGradientBoostingRegressor(random_state=SEED).fit(X_bad_train, y_bad_train)
bad_rmse = np.sqrt(mean_squared_error(y_bad_test, bad_model.predict(X_bad_test)))
print({"random_split_rmse_may_hide_shift": bad_rmse})
""",
            "exercises": "1. Derive non-negativity of KL using Jensen's inequality.\n2. Implement multidimensional shift detection via MMD.\n3. Build a leakage checklist for timestamped features.\n4. Compare adversarial AUC before/after feature sanitization.",
            "summary": "Leakage detection combines theory (KL/KS/information flow) and diagnostics (adversarial validation) to reject deceptively strong but non-deployable pipelines.",
        },
        {
            "id": "14",
            "slug": "ensembling_and_stacking",
            "title": "Ensembling and Stacking",
            "prior": "- Notebook 12 OOF/CV discipline.\n- Notebook 13 leakage-safe diagnostics.",
            "new": "- Bagging variance reduction with correlation terms.\n- Ensemble covariance formula.\n- Stacking as second-level ERM.\n- OOF stacking implementation.\n- Blending vs stacking empirical comparison.",
            "problem": "Combine diverse models to reduce variance and improve leaderboard stability.",
            "math": r"""
### Mathematical Foundations
For ensemble
$$
\\hat{y}_{ens}=\\sum_{m=1}^M w_m\\hat{y}^{(m)},\\quad \\sum_m w_m=1,
$$
variance is
$$
\\mathrm{Var}(\\hat{y}_{ens})=\\sum_{m=1}^M\\sum_{\\ell=1}^M w_m w_\\ell\\,\\mathrm{Cov}(\\hat{y}^{(m)},\\hat{y}^{(\\ell)}).
$$

Equal-variance equal-correlation case ($\\sigma^2,\\rho$):
$$
\\mathrm{Var}(\\bar{y})=\\sigma^2\\left(\\rho+\\frac{1-\\rho}{M}\\right).
$$

Stacking objective:
$$
\\theta^*=\\arg\\min_\\theta\\frac{1}{n}\\sum_{i=1}^n
L\\left(y_i,g_\\theta\\left(\\hat{y}^{(1)}_{i,\\mathrm{OOF}},\\dots,\\hat{y}^{(M)}_{i,\\mathrm{OOF}}\\right)\\right).
$$
""",
            "derivation": r"""
### Step-by-Step Derivation
1. Write ensemble prediction as weighted sum.
2. Apply variance bilinearity:
   $$
   \\mathrm{Var}\\left(\\sum_m w_m Z_m\\right)=\\sum_m\\sum_\\ell w_m w_\\ell\\mathrm{Cov}(Z_m,Z_\\ell).
   $$
3. Substitute $Z_m=\\hat{y}^{(m)}$.
4. In equal-correlation case:
   - diagonal terms: $M\\cdot (1/M^2)\\sigma^2 = \\sigma^2/M$
   - off-diagonal terms: $M(M-1)\\cdot (1/M^2)\\rho\\sigma^2 = \\rho\\sigma^2(1-1/M)$
5. Sum both terms to get
   $$
   \\sigma^2\\left(\\rho+\\frac{1-\\rho}{M}\\right).
   $$
""",
            "intuition": "Ensembles only help when members are not perfectly correlated; OOF stacking learns data-driven combination weights while preserving validation integrity.",
            "code_map": "- `bagging_variance_formula` computes theoretical reduction.\n- `ensemble_variance_from_covariance` evaluates empirical covariance effects.\n- `oof_stacking` enforces leakage-safe meta-feature creation.\n- `blend_predictions` implements static weighted averaging.",
            "synthetic_code": """
rng = np.random.default_rng(SEED)
n = 5000
signal = rng.normal(size=n)
eps = rng.normal(size=(n, 3))

# Correlated predictors around shared signal.
pred1 = signal + 0.9 * eps[:, 0]
pred2 = signal + 0.9 * (0.7 * eps[:, 0] + 0.3 * eps[:, 1])
pred3 = signal + 0.9 * (0.5 * eps[:, 0] + 0.5 * eps[:, 2])
pred_matrix = np.column_stack([pred1, pred2, pred3])
cov = np.cov(pred_matrix, rowvar=False)

w = np.array([1/3, 1/3, 1/3])
var_emp = ensemble_variance_from_covariance(cov, w)
print({"empirical_weighted_variance": var_emp, "cov_matrix": cov})
print({"theory_example": bagging_variance_formula(base_variance=1.0, pairwise_corr=0.6, n_models=3)})
""",
            "real_code": """
bc = load_breast_cancer(as_frame=True)
X = bc.data.values
y = bc.target.values
splitter = make_splitter("stratified", n_splits=5, seed=SEED)

base_models = [
    ("lr", LogisticRegression(max_iter=2000)),
    ("rf", RandomForestClassifier(n_estimators=250, random_state=SEED)),
    ("hgb", HistGradientBoostingClassifier(random_state=SEED)),
]
meta = LogisticRegression(max_iter=2000)

stack_bundle = oof_stacking(base_models, meta, X, y, splitter, task="classification")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=SEED, stratify=y)

# Refit base models on train for direct blend baseline.
trained = []
for name, est in base_models:
    fit_est = est.fit(X_train, y_train)
    trained.append((name, fit_est))
preds = np.column_stack([m.predict_proba(X_test)[:, 1] for _, m in trained])
blend_pred = blend_predictions(preds, [0.25, 0.4, 0.35])

stack_pred = stacking_predict(stack_bundle, X_test, task="classification")
print({
    "blend_auc": roc_auc_score(y_test, blend_pred),
    "stack_auc": roc_auc_score(y_test, stack_pred),
})
""",
            "diagnostic_code": """
pred_corr = np.corrcoef(preds, rowvar=False)
print("base prediction correlation matrix:\\n", pred_corr)
plt.figure(figsize=(4, 3))
plt.imshow(pred_corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar()
plt.title("Prediction correlation")
plt.tight_layout()
plt.show()
""",
            "failure_code": """
# Failure case: training meta-model on in-sample base predictions (leaky stacking).
leaky_meta_X = np.column_stack([m.predict_proba(X_train)[:, 1] for _, m in trained])
leaky_meta = LogisticRegression(max_iter=2000).fit(leaky_meta_X, y_train)
test_meta_X = np.column_stack([m.predict_proba(X_test)[:, 1] for _, m in trained])
leaky_pred = leaky_meta.predict_proba(test_meta_X)[:, 1]
print({"leaky_stack_auc": roc_auc_score(y_test, leaky_pred), "proper_stack_auc": roc_auc_score(y_test, stack_pred)})
""",
            "exercises": "1. Derive optimal 2-model blend weight under quadratic loss and known covariance matrix.\n2. Prove blending is a restricted linear stacking case.\n3. Add XGBoost and LightGBM base learners and quantify diversity gain.\n4. Evaluate stack robustness under fold perturbation.",
            "summary": "Stacking is controlled meta-learning over OOF predictions; variance reduction depends on diversity, not model count alone.",
        },
        {
            "id": "15",
            "slug": "pseudo_labeling_and_semi_supervised",
            "title": "Pseudo-Labeling and Semi-Supervised Learning",
            "prior": "- Notebooks 12 and 14 for reliable OOF validation and ensembling strategy.\n- Classification losses and confidence interpretation.",
            "new": "- Semi-supervised objective decomposition.\n- Confidence-threshold pseudo-label rule.\n- Confirmation-bias risk derivation.\n- Iterative pseudo-label loop with diagnostics.",
            "problem": "Leverage unlabeled data without reinforcing model errors.",
            "math": r"""
### Mathematical Foundations
Semi-supervised objective:
$$
\\mathcal{L}(\\theta)=\\mathcal{L}_{sup}(\\theta)+\\lambda\\,\\mathcal{L}_{unsup}(\\theta).
$$
For pseudo-labeling with threshold $\\tau$:
$$
\\tilde{y}_u = \\mathbf{1}[p_\\theta(y=1\\mid x_u)\\ge \\tau],\\qquad
\\mathcal{U}_\\tau=\\{u:\\max_c p_\\theta(c\\mid x_u)\\ge\\tau\\}.
$$

Unsupervised term:
$$
\\mathcal{L}_{unsup}(\\theta)=\\frac{1}{|\\mathcal{U}_\\tau|}\\sum_{u\\in\\mathcal{U}_\\tau}
\\ell\\big(f_\\theta(x_u),\\tilde{y}_u\\big).
$$
""",
            "derivation": r"""
### Step-by-Step Derivation
1. Start from supervised empirical risk on labeled set $\\mathcal{D}_L$:
   $$
   \\mathcal{L}_{sup}=\\frac{1}{|\\mathcal{D}_L|}\\sum_{(x,y)\\in\\mathcal{D}_L}\\ell(f_\\theta(x),y).
   $$
2. Generate pseudo-labels on unlabeled pool $\\mathcal{D}_U$ using current model.
3. Keep only confident points $\\mathcal{U}_\\tau$.
4. Add pseudo-label loss with multiplier $\\lambda$.
5. Confirmation bias risk: if pseudo-label error rate is $\\epsilon$, expected noisy contribution scales as
   $$
   \\lambda\\,\\epsilon\\,\\mathbb{E}[\\Delta\\ell],
   $$
   so large $\\lambda$ and low $\\tau$ can amplify wrong gradients.
""",
            "intuition": "Pseudo-labeling is self-training; confidence filtering trades data quantity for label quality.",
            "code_map": "- We maintain separate labeled/unlabeled splits.\n- At each iteration, high-confidence unlabeled samples are added with pseudo-labels.\n- Metrics track gain versus confirmation-bias drift.",
            "synthetic_code": """
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=5000, n_features=20, n_informative=10, class_sep=1.0, flip_y=0.03, random_state=SEED
)
X_lab, X_pool, y_lab, y_pool_true = train_test_split(X, y, test_size=0.7, random_state=SEED, stratify=y)
X_unl, X_test, y_unl_true, y_test = train_test_split(X_pool, y_pool_true, test_size=0.4, random_state=SEED, stratify=y_pool_true)

clf = LogisticRegression(max_iter=2000).fit(X_lab, y_lab)
base_auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])

tau = 0.9
for step in range(3):
    proba = clf.predict_proba(X_unl)[:, 1]
    conf = np.maximum(proba, 1 - proba)
    mask = conf >= tau
    pseudo = (proba[mask] >= 0.5).astype(int)
    if mask.sum() == 0:
        break
    X_lab = np.vstack([X_lab, X_unl[mask]])
    y_lab = np.concatenate([y_lab, pseudo])
    X_unl = X_unl[~mask]
    clf = LogisticRegression(max_iter=2000).fit(X_lab, y_lab)

pl_auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
print({"baseline_auc": base_auc, "pseudo_label_auc": pl_auc, "remaining_unlabeled": len(X_unl)})
""",
            "real_code": """
wine = load_wine(as_frame=True)
X = wine.data.values
y = (wine.target.values == 2).astype(int)
X_lab, X_unl, y_lab, y_unl_true = train_test_split(X, y, test_size=0.6, random_state=SEED, stratify=y)
X_unl_pool, X_test, y_unl_pool_true, y_test = train_test_split(X_unl, y_unl_true, test_size=0.4, random_state=SEED, stratify=y_unl_true)

base = HistGradientBoostingClassifier(random_state=SEED).fit(X_lab, y_lab)
base_auc = roc_auc_score(y_test, base.predict_proba(X_test)[:, 1])

proba = base.predict_proba(X_unl_pool)[:, 1]
mask = np.maximum(proba, 1 - proba) >= 0.92
pseudo = (proba[mask] >= 0.5).astype(int)
X_aug = np.vstack([X_lab, X_unl_pool[mask]])
y_aug = np.concatenate([y_lab, pseudo])
aug = HistGradientBoostingClassifier(random_state=SEED).fit(X_aug, y_aug)
aug_auc = roc_auc_score(y_test, aug.predict_proba(X_test)[:, 1])
print({"wine_base_auc": base_auc, "wine_pseudo_auc": aug_auc, "pseudo_points": int(mask.sum())})
""",
            "diagnostic_code": """
conf = np.maximum(proba, 1 - proba)
plt.figure(figsize=(6, 3))
plt.hist(conf, bins=30)
plt.title("Unlabeled confidence distribution")
plt.xlabel("max class probability")
plt.tight_layout()
plt.show()

if mask.sum() > 0:
    pseudo_error = float(np.mean(pseudo != y_unl_pool_true[mask]))
    print({"pseudo_label_error_rate": pseudo_error})
""",
            "failure_code": """
# Failure case: low threshold increases noisy pseudo-labels.
low_tau = 0.55
mask_low = np.maximum(proba, 1 - proba) >= low_tau
pseudo_low = (proba[mask_low] >= 0.5).astype(int)
X_low = np.vstack([X_lab, X_unl_pool[mask_low]])
y_low = np.concatenate([y_lab, pseudo_low])
model_low = HistGradientBoostingClassifier(random_state=SEED).fit(X_low, y_low)
auc_low = roc_auc_score(y_test, model_low.predict_proba(X_test)[:, 1])
print({"low_threshold_auc": auc_low, "high_threshold_auc": aug_auc, "low_tau_points": int(mask_low.sum())})
""",
            "exercises": "1. Derive gradient contribution of pseudo-label noise under logistic loss.\n2. Compare fixed threshold vs quantile-based threshold schedules.\n3. Add teacher-student ensembling for pseudo labels.\n4. Analyze when pseudo-labeling hurts under class imbalance.",
            "summary": "Pseudo-labeling is useful only with confidence discipline and strict monitoring of pseudo-label error propagation.",
        },
        {
            "id": "16",
            "slug": "kaggle_competition_simulator",
            "title": "Kaggle Competition Simulator",
            "prior": "- Notebooks 10-15: feature engineering, CV rigor, leakage detection, ensembling, pseudo-labeling.\n- End-to-end reproducible ML pipelines.",
            "new": "- Formal competition objective under hidden private distribution.\n- Public/private split variance simulation.\n- Full pipeline integration with logging and gap analysis.",
            "problem": "Simulate competition decision-making under leaderboard uncertainty with fully reproducible experiments.",
            "math": r"""
### Mathematical Foundations
Competition objective:
$$
\\theta^*=\\arg\\min_{\\theta\\in\\Theta}\\mathbb{E}[\\mathcal{M}_{private}(\\theta)].
$$
Observed public metric:
$$
\\widehat{\\mathcal{M}}_{public}(\\theta)=\\mathcal{M}(\\theta;\\mathcal{D}_{public}),
$$
hidden private metric:
$$
\\widehat{\\mathcal{M}}_{private}(\\theta)=\\mathcal{M}(\\theta;\\mathcal{D}_{private}).
$$
Leaderboard shake:
$$
\\Delta_{shake}(\\theta)=\\widehat{\\mathcal{M}}_{private}(\\theta)-\\widehat{\\mathcal{M}}_{public}(\\theta).
$$

Stacked submission prediction:
$$
\\hat{y}=g_\\phi\\big(\\hat{y}^{(torch)},\\hat{y}^{(xgb)},\\hat{y}^{(lgbm)},\\hat{y}^{(rf)}\\big).
$$
""",
            "derivation": r"""
### Step-by-Step Derivation
1. Define hidden test set split into public and private subsets:
   $$
   \\mathcal{D}_{test}=\\mathcal{D}_{public}\\cup\\mathcal{D}_{private},\\quad
   \\mathcal{D}_{public}\\cap\\mathcal{D}_{private}=\\varnothing.
   $$
2. For each candidate $\\theta$, compute CV proxy score $\\widehat{\\mathcal{M}}_{CV}(\\theta)$.
3. Public board provides noisy sample of true private behavior.
4. Approximate risk-adjusted selection:
   $$
   J(\\theta)=\\widehat{\\mathcal{M}}_{CV}(\\theta)+\\beta\\,\\widehat{\\mathrm{Std}}_{fold}(\\theta)+\\gamma|\\Delta_{shake}(\\theta)|.
   $$
5. Choose candidate minimizing $J$ to avoid unstable leaderboard spikes.
""",
            "intuition": "Winning strategy is robust estimation under hidden evaluation; reproducible logging plus fold-stability constraints beats overfitting public board noise.",
            "code_map": "- Features from `feature_utils`.\n- Split discipline from `cv_utils`.\n- Blending and stacking from `ensemble_utils`.\n- Experiment registry from `experiment_logger`.\n- Optional XGBoost/LightGBM used when installed.",
            "synthetic_code": """
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=8000, n_features=30, n_informative=14, noise=22.0, random_state=SEED)
X = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
X["x0_x1"] = X["x0"] * X["x1"]
X["x2_log"] = monotone_log1p(np.abs(X["x2"]))

X_train_full, X_lb, y_train_full, y_lb = train_test_split(X, y, test_size=0.30, random_state=SEED)
X_public, X_private, y_public, y_private = train_test_split(X_lb, y_lb, test_size=0.50, random_state=SEED)

print("Shapes:", X_train_full.shape, X_public.shape, X_private.shape)
""",
            "real_code": """
# Real dataset benchmark (hybrid policy).
try:
    data = fetch_california_housing(as_frame=True)
    X_real = data.data.copy()
    y_real = data.target.to_numpy(dtype=float)
except Exception:
    diab = load_diabetes(as_frame=True)
    X_real = diab.data.copy()
    y_real = diab.target.to_numpy(dtype=float)

X_real["ratio_rooms"] = X_real.iloc[:, 0] / (np.abs(X_real.iloc[:, 1]) + 1e-6)
X_train, X_hold, y_train, y_hold = train_test_split(X_real.values, y_real, test_size=0.25, random_state=SEED)

# PyTorch baseline regressor.
sc_torch = StandardScaler().fit(X_train)
X_train_t = torch.tensor(sc_torch.transform(X_train), dtype=torch.float32)
y_train_t = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
X_hold_t = torch.tensor(sc_torch.transform(X_hold), dtype=torch.float32)

class TorchMLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
    def forward(self, x):
        return self.net(x)

torch_model = TorchMLP(X_train.shape[1])
opt = torch.optim.AdamW(torch_model.parameters(), lr=1e-3, weight_decay=1e-5)
loss_fn = nn.MSELoss()
for _ in range(120):
    opt.zero_grad()
    loss = loss_fn(torch_model(X_train_t), y_train_t)
    loss.backward()
    opt.step()

with torch.no_grad():
    pred_torch = torch_model(X_hold_t).squeeze(1).numpy()

base_rf = RandomForestRegressor(n_estimators=300, random_state=SEED)
base_hgb = HistGradientBoostingRegressor(random_state=SEED)
base_rf.fit(X_train, y_train)
base_hgb.fit(X_train, y_train)
pred_rf = base_rf.predict(X_hold)
pred_hgb = base_hgb.predict(X_hold)

models = {"torch_mlp": pred_torch, "rf": pred_rf, "hgb": pred_hgb}

if XGB_AVAILABLE:
    xgb = XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, random_state=SEED
    )
    xgb.fit(X_train, y_train)
    models["xgb"] = xgb.predict(X_hold)

if LGBM_AVAILABLE:
    lgbm = LGBMRegressor(
        n_estimators=300, learning_rate=0.05, num_leaves=31, subsample=0.9, colsample_bytree=0.9, random_state=SEED
    )
    lgbm.fit(X_train, y_train)
    models["lgbm"] = lgbm.predict(X_hold)

pred_matrix = np.column_stack(list(models.values()))
weights = np.ones(pred_matrix.shape[1]) / pred_matrix.shape[1]
blend = blend_predictions(pred_matrix, weights)
rmse_blend = np.sqrt(mean_squared_error(y_hold, blend))

single_scores = {name: np.sqrt(mean_squared_error(y_hold, p)) for name, p in models.items()}
print({"single_rmse": single_scores, "blend_rmse": rmse_blend, "n_models": pred_matrix.shape[1]})
""",
            "diagnostic_code": """
logger = ExperimentLogger(MODULE_DIR / "competition_experiments.jsonl")
for name, score in single_scores.items():
    logger.log(
        ExperimentRecord(
            exp_id=f"{name}_base",
            seed=SEED,
            feature_set="core+ratio",
            model_name=name,
            cv_score=score,
            public_score=score * 1.01,
            private_score=score * 1.015,
            params={"model": name},
        )
    )

logger.log(
    ExperimentRecord(
        exp_id="blend_equal",
        seed=SEED,
        feature_set="core+ratio",
        model_name="blend",
        cv_score=rmse_blend,
        public_score=rmse_blend * 1.008,
        private_score=rmse_blend * 1.010,
        params={"weights": weights.tolist()},
    )
)

summary_df = logger.summary()
display(summary_df.tail(5))

metric = lambda yt, yp: np.sqrt(mean_squared_error(yt, yp))
shake = simulate_public_private_variance(y_hold, blend, metric, public_fraction=0.5, n_trials=250, seed=SEED)
print("shake_summary:", shake)

if OPTUNA_AVAILABLE:
    X_tune_train, X_tune_valid, y_tune_train, y_tune_valid = train_test_split(
        X_train, y_train, test_size=0.25, random_state=SEED
    )

    def objective(trial):
        depth = trial.suggest_int("max_depth", 2, 8)
        lr = trial.suggest_float("learning_rate", 0.02, 0.2, log=True)
        n_est = trial.suggest_int("n_estimators", 120, 420)
        model = HistGradientBoostingRegressor(max_depth=depth, learning_rate=lr, max_iter=n_est, random_state=SEED)
        model.fit(X_tune_train, y_tune_train)
        pred = model.predict(X_tune_valid)
        return np.sqrt(mean_squared_error(y_tune_valid, pred))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10, show_progress_bar=False)
    print("optuna_best:", {"value": study.best_value, "params": study.best_params})
""",
            "failure_code": """
# Failure case: choosing model only by a favorable public proxy can increase private risk.
public_proxy = {name: score * np.random.default_rng(SEED).uniform(0.98, 1.02) for name, score in single_scores.items()}
chosen = min(public_proxy, key=public_proxy.get)
private_proxy = {name: score * np.random.default_rng(SEED + 1).uniform(0.98, 1.02) for name, score in single_scores.items()}
print({"public_proxy": public_proxy, "chosen_by_public": chosen, "private_proxy": private_proxy})
""",
            "exercises": "1. Derive expected rank instability under random public/private partitioning.\n2. Add Optuna-based hyperparameter search with nested-CV objective.\n3. Implement OOF stacking with RF/HGB/XGB/LGBM and compare against blending.\n4. Write a rollback strategy based on shake quantiles.",
            "summary": "Competition success is risk-aware optimization: feature quality, CV reliability, ensemble diversity, and experiment logging must be optimized as one system.",
        },
    ]


def build_notebook(spec: dict[str, str], solution: bool) -> nbf.NotebookNode:
    nb = new_nb()
    suffix = "Solutions" if solution else "Lesson"
    add_md(
        nb,
        f"""
# {spec['id']} — {spec['title']} ({suffix})

## Problem Definition
{spec['problem']}

## Required Prior Knowledge
{spec['prior']}

## New Concepts Introduced
{spec['new']}
""",
    )
    add_md(nb, spec["math"])
    add_md(
        nb,
        """
## Symbol-by-Symbol Explanation
| Symbol | Meaning |
|---|---|
| $x_i$ | feature vector for sample $i$ |
| $y_i$ | target for sample $i$ |
| $f_\\theta$ | model parameterized by $\\theta$ |
| $L(\\cdot,\\cdot)$ | per-sample loss function |
| $n$ | number of samples |
| $d$ | raw feature dimension |
| $p$ | transformed feature dimension / polynomial degree context |
| $K$ | number of folds / partitions |
| $V_k$ | validation index set for fold $k$ |
| $\\lambda,\\alpha,\\tau$ | regularization/smoothing/confidence hyperparameters |
""",
    )
    add_md(nb, spec["derivation"])
    add_md(nb, f"## Intuition\n{spec['intuition']}")
    add_md(nb, f"## Mapping from Math to Implementation\n{spec['code_map']}")
    add_code(nb, COMMON_SETUP_CODE + "\nMODULE_DIR = Path('.').resolve()")
    add_md(nb, "## Synthetic Experiment")
    add_code(nb, spec["synthetic_code"])
    add_md(nb, "## Real Dataset Experiment")
    add_code(nb, spec["real_code"])
    add_md(nb, "## Diagnostic Analysis")
    add_code(nb, spec["diagnostic_code"])
    add_md(nb, "## Failure Case Demonstration")
    add_code(nb, spec["failure_code"])
    add_md(nb, f"## Exercise Ladder (basic → advanced → research-level)\n{spec['exercises']}")
    if solution:
        add_md(
            nb,
            """
## Solution Notes
- Verify deterministic behavior by re-running all cells with the same seed and matching key metrics.
- Confirm that no fold-level preprocessing leaks validation targets/features into training statistics.
- Compare synthetic-vs-real conclusions and report where assumptions diverge.
""",
        )
    add_md(nb, f"## Summary of Mathematical Insights\n{spec['summary']}")
    return nb


def main() -> None:
    topics = make_topics()
    for spec in topics:
        lesson_nb = build_notebook(spec, solution=False)
        solution_nb = build_notebook(spec, solution=True)
        lesson_name = f"{spec['id']}_{spec['slug']}.ipynb"
        solution_name = f"{spec['id']}_{spec['slug']}_solutions.ipynb"
        with (MODULE_DIR / lesson_name).open("w", encoding="utf-8") as f:
            nbf.write(lesson_nb, f)
        with (MODULE_DIR / solution_name).open("w", encoding="utf-8") as f:
            nbf.write(solution_nb, f)
    print(f"Generated {len(topics) * 2} notebooks in {MODULE_DIR}")


if __name__ == "__main__":
    main()
