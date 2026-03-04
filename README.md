# Prepare_kaggle

`Prepare_kaggle` is a notebook-first machine learning curriculum for Kaggle-style workflows, covering deep learning fundamentals through advanced competition strategy.

## Repository modules

### 1) `deep learning/` (core curriculum)
- Foundation-to-intermediate track (`00` to `09`) with lesson + solutions notebooks.
- Emphasis on PyTorch training dynamics, regularization, hyperparameter tuning, and tabular competition practice.
- Includes notebook generator tooling in `deep learning/generators/`.

See: `deep learning/README.md`

### 2) `10_advanced_feature_engineering_and_competition_strategies/` (advanced track)
- Advanced lessons (`10` to `16`) with lesson + solutions notebooks.
- Focus on mathematically rigorous feature engineering, CV design, leakage detection, ensembling/stacking, pseudo-labeling, and competition simulation.
- Includes reusable utility modules:
  - `feature_utils.py`
  - `cv_utils.py`
  - `ensemble_utils.py`
  - `experiment_logger.py`
- Includes notebook generator:
  - `build_advanced_module_notebooks.py`

See: `10_advanced_feature_engineering_and_competition_strategies/README.md`

## Full notebook progression

Recommended learning order:
1. `deep learning/00` to `deep learning/06`
2. `deep learning/08`
3. `deep learning/09`
4. `deep learning/07` (boosting comparison)
5. `10_advanced_feature_engineering_and_competition_strategies/10` to `.../16`

This order keeps conceptual prerequisites aligned and moves from baseline training mechanics to full competition decision systems.

## Setup

From repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

Install dependencies for the core module:

```bash
pip install -r "deep learning/requirements.txt"
```

Install dependencies for the advanced module:

```bash
pip install -r "10_advanced_feature_engineering_and_competition_strategies/requirements.txt"
```

Launch Jupyter:

```bash
jupyter notebook
```

## Reproducibility principles

- Seed-controlled experiments are used throughout notebooks and utilities.
- Leakage-safe patterns (especially OOF encodings and OOF stacking) are emphasized.
- Dataset handling follows a hybrid policy:
  - sklearn built-ins by default
  - auto-download where needed
  - deterministic fallbacks where possible

## Regenerating notebooks

Core module:

```bash
cd "deep learning/generators"
python3 generate_advanced_notebooks.py
```

Advanced module:

```bash
cd "10_advanced_feature_engineering_and_competition_strategies"
python3 build_advanced_module_notebooks.py
```

## Notes

- Lesson notebooks are designed for self-attempt first.
- Solution notebooks provide full worked implementations and interpretation guidance.
- For competition practice, keep a strict experiment log (seed, feature set, CV design, model config, and public/private behavior).

