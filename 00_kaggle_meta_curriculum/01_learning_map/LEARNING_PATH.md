# Learning Path

## Purpose
Define a stage-by-stage route through the existing repository so progress is systematic, leakage-safe, and measurable. This map references current folders/notebooks instead of duplicating lessons.

## Progression Rules
1. Complete stages in order.
2. **Anti-leakage and cross-validation (CV) discipline are mandatory gates** before advanced ensembling/pseudo-labeling/simulation.
3. For each notebook: attempt lesson first, then use `*_solutions.ipynb` for gap-closing.

## Stage-by-Stage Route

| Stage | Route through existing repo content | Milestone objective | Evidence of mastery |
|---|---|---|---|
| 0. Orientation + validation mindset | `README.md`, `00_kaggle_meta_curriculum/02_kaggle_thinking/`, `00_kaggle_meta_curriculum/03_cross_validation_mastery/` | Internalize why CV quality and leakage control matter more than public leaderboard spikes. | You can explain failure modes (split mismatch, leakage, shake) and state a fold strategy before training any model. |
| 1. Core training mechanics | `deep learning/00_baseline.ipynb` -> `06_optuna_hyperparameter_tuning.ipynb` (+ solutions) | Build stable training intuition: optimization, regularization, and tuning. | You reproduce consistent train/validation behavior across seeds and justify major hyperparameter choices. |
| 2. Baseline competition workflow | `deep learning/08_feature_engineering.ipynb`, `09_kaggle_competition_playbook.ipynb`, then `07_xgboost_lightgbm.ipynb` | Produce a reproducible tabular baseline and compare model families pragmatically. | You can run a full baseline loop (feature prep -> CV -> experiment notes) and defend model selection with CV evidence. |
| 3. Advanced feature engineering | `10_advanced_feature_engineering_and_competition_strategies/10_feature_engineering_fundamentals.ipynb`, `11_advanced_feature_engineering_patterns.ipynb`, `feature_utils.py`, `experiment_logger.py` | Engineer stronger features without contaminating validation. | You show incremental CV lift from new features and log each experiment with reproducible settings. |
| 4. **CV + leakage gate (must pass)** | `10_advanced_feature_engineering_and_competition_strategies/12_cross_validation_mastery.ipynb`, `13_data_leakage_detection.ipynb`, `cv_utils.py`, plus `00_kaggle_meta_curriculum/03_cross_validation_mastery/` | Select correct split strategy and detect/prevent leakage patterns. | You can justify split choice, produce OOF predictions correctly, and demonstrate how leakage inflates validation metrics. |
| 5. Ensembling and semi-supervised strategy | `10_advanced_feature_engineering_and_competition_strategies/14_ensembling_and_stacking.ipynb`, `15_pseudo_labeling_and_semi_supervised.ipynb`, `ensemble_utils.py` | Improve robustness via OOF stacking/blending and controlled pseudo-labeling. | Ensemble gains are validated by fold-level stability (not single-fold luck), and pseudo-labeling is introduced with risk checks. |
| 6. Competition simulation + readiness review | `10_advanced_feature_engineering_and_competition_strategies/16_kaggle_competition_simulator.ipynb`, `00_kaggle_meta_curriculum/04_competition_simulation/`, `00_kaggle_meta_curriculum/05_evaluation_rubric/` | Execute an end-to-end competition cycle with post-mortem discipline. | You complete a full simulation, explain public/private gap handling, and pass your own readiness checklist. |

## Completion Signal
You are competition-ready when you can repeatedly deliver leakage-safe OOF pipelines, stable cross-validated improvements, and defensible experiment decisions across multiple simulated rounds.
