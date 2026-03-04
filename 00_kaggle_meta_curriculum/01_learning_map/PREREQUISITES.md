# Prerequisites

## Purpose
Define concrete prerequisite requirements for this repository’s Kaggle track, with direct pointers to where each requirement is learned in existing folders.

## Non-Negotiable Gate
**Anti-leakage and CV discipline are gating prerequisites.** Do not start:
- `10_advanced_feature_engineering_and_competition_strategies/14_ensembling_and_stacking.ipynb`
- `10_advanced_feature_engineering_and_competition_strategies/15_pseudo_labeling_and_semi_supervised.ipynb`
- `10_advanced_feature_engineering_and_competition_strategies/16_kaggle_competition_simulator.ipynb`
until the Validation Discipline row below is met.

## Prerequisite Matrix

| Domain | Concrete prerequisite | Minimum bar to proceed | Learn in existing repo content | Evidence check |
|---|---|---|---|---|
| Math | Gradients/optimization behavior, regularization effects, bias-variance, metric interpretation, ensemble variance intuition | Can explain why a change should help before running it (not only after seeing score movement) | `deep learning/00_baseline.ipynb` -> `06_optuna_hyperparameter_tuning.ipynb`, `deep learning/10_formula_deep_dive_compendium.ipynb`, `10_advanced_feature_engineering_and_competition_strategies/17_formula_deep_dive_compendium.ipynb` | You can derive/interpret core trade-offs (underfit vs overfit, variance reduction via ensembles) from notebook outputs. |
| ML fundamentals | End-to-end supervised workflow, feature preprocessing, model family comparison, reproducibility habits | Can train, validate, compare, and document at least one strong baseline pipeline | `deep learning/08_feature_engineering.ipynb`, `deep learning/09_kaggle_competition_playbook.ipynb`, `deep learning/07_xgboost_lightgbm.ipynb`, root `README.md` | You can run a baseline and explain why it is your reference model. |
| Coding & experiment operations | Comfortable with notebook execution, utility modules, and deterministic experiment tracking | Can modify and rerun notebook pipelines safely with fixed seeds/logged configs | `deep learning/` notebooks + `*_solutions.ipynb`, `10_advanced_feature_engineering_and_competition_strategies/feature_utils.py`, `experiment_logger.py` | You can reproduce a prior run and show what changed between two experiments. |
| **Validation discipline (gating)** | Split strategy selection, OOF logic, leakage detection, stability checks, public/private gap awareness | **Must pass before advanced strategy notebooks (14-16)** | `00_kaggle_meta_curriculum/02_kaggle_thinking/`, `00_kaggle_meta_curriculum/03_cross_validation_mastery/`, `10_advanced_feature_engineering_and_competition_strategies/12_cross_validation_mastery.ipynb`, `13_data_leakage_detection.ipynb`, `cv_utils.py` | You can defend CV design, produce leakage-safe OOF predictions, and identify suspicious score inflation before trusting results. |

## Stage Gates
1. Before serious competition work (`deep learning/09_kaggle_competition_playbook.ipynb` onward), be comfortable with baseline CV reasoning.
2. Before advanced strategy (`10_advanced.../14-16`), Validation Discipline must be fully satisfied.
3. If leakage or unstable folds are detected at any stage, roll back and rework CV design before adding model complexity.
