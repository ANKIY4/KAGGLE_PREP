# Competition Simulation Overview

## Purpose
Define a repeatable, leakage-safe, leaderboard-aware operating system for end-to-end Kaggle competition execution in this repository, from first baseline to final submission and post-competition learning loop.

## Implementation anchors in this repository
- **Playbook foundation:** `deep learning/09_kaggle_competition_playbook.ipynb`
- **Advanced execution track:** `10_advanced_feature_engineering_and_competition_strategies/10_*.ipynb` through `16_*.ipynb`
- **Feature controls:** `10_advanced_feature_engineering_and_competition_strategies/feature_utils.py`
- **Validation and shake analysis:** `10_advanced_feature_engineering_and_competition_strategies/cv_utils.py`
- **Blending/stacking decisions:** `10_advanced_feature_engineering_and_competition_strategies/ensemble_utils.py`
- **Experiment traceability:** `10_advanced_feature_engineering_and_competition_strategies/experiment_logger.py`

## End-to-end competition lifecycle

| Phase | Goal | Primary outputs | Repo references |
|---|---|---|---|
| 1. Problem framing | Lock metric, constraints, and risk model | Metric definition, split hypothesis, resource budget | `deep learning/09_kaggle_competition_playbook.ipynb`, `12_cross_validation_mastery.ipynb` |
| 2. Baseline establishment | Create stable baseline and sanity checks | Baseline CV score, reproducible seed, first submission | `feature_utils.set_global_seed`, `cv_utils.oof_cv_predictions` |
| 3. Validation hardening | Reduce false optimism from CV and leakage | Splitter choice, leakage tests, shake simulation | `cv_utils.make_splitter`, `13_data_leakage_detection.ipynb`, `cv_utils.leakage_inflation`, `cv_utils.simulate_public_private_variance` |
| 4. Feature iteration | Add high-signal features without leakage | Feature registry, ablations, OOF-safe encodings | `10_feature_engineering_fundamentals.ipynb`, `11_advanced_feature_engineering_patterns.ipynb`, `feature_utils.target_encode_oof` |
| 5. Model/ensemble optimization | Improve robustness and rank consistency | Single-model leaderboard, blend/stack candidates | `14_ensembling_and_stacking.ipynb`, `ensemble_utils.blend_predictions`, `ensemble_utils.oof_stacking` |
| 6. Submission operations | Convert experiments to controlled submissions | Submission queue, rationale, expected variance band | `16_kaggle_competition_simulator.ipynb`, `experiment_logger.ExperimentLogger` |
| 7. Freeze + review | Protect final standing, prevent late regressions | Final candidate set, fallback package, handoff notes | `experiment_logger.ExperimentLogger.summary`, `post_mortem.md` |

## Decision checkpoints (go/no-go gates)
1. **C0 - Baseline validity:** baseline pipeline runs end-to-end with fixed seed and no schema drift.
2. **C1 - CV trustworthiness:** splitter reflects competition structure (time/group/stratified), and leakage checks are passed.
3. **C2 - Feature acceptance:** new feature set must show repeatable OOF gain across folds, not just single-fold lift.
4. **C3 - Submission eligibility:** candidate must beat current champion by predefined CV margin and stay inside expected shake range.
5. **C4 - Ensemble readiness:** components must show partial error diversity before blending/stacking.
6. **C5 - Final freeze gate:** last 24h submissions prioritize stability; risky exploratory submissions require explicit rollback candidate.
7. **C6 - Post-result closure:** publish post-mortem with mistakes, tradeoffs, and next-iteration backlog.

## Experiment governance

### 1) Logging standard
- Log every run as an `ExperimentRecord` with: seed, feature set, model name, CV score, notes, params.
- Persist with `ExperimentLogger.log(...)`; inspect drift and stability using `ExperimentLogger.summary()`.
- Require a clear experiment ID naming scheme (example: `exp_2026_03_04_lgbm_te_v2`).

### 2) Reproducibility controls
- Fix seeds via `feature_utils.set_global_seed`.
- Store fold strategy and splitter config (`cv_utils.make_splitter`) in params.
- Keep train/validation transformation boundaries explicit (`feature_utils.standardize_train_valid`, OOF-only encodings).

### 3) Promotion rules
- Promote experiment only if:
  - OOF improvement exceeds minimum threshold,
  - variance across folds is acceptable,
  - leakage checks remain clean.
- Document promotion/demotion reason in experiment notes.

### 4) Review cadence
- Daily review: top 3 candidates by CV with risk commentary.
- Milestone review (mid-competition and freeze): audit public/private gap assumptions using shake simulation tools.

## Submission policy
- **Quota discipline:** reserve submissions for hypotheses with evidence; avoid random probing.
- **Eligibility rule:** submit only after passing checkpoints C1-C3.
- **Portfolio policy:** maintain 3 lanes:
  - **Safe:** strongest stable CV candidate.
  - **Balanced:** modestly higher risk with controlled variance.
  - **Aggressive:** exploratory idea capped by rollback readiness.
- **Metadata requirement:** each submission links to experiment ID, config hash, expected behavior, and rollback parent.
- **Freeze window policy:** last phase favors conservative submissions unless aggressive candidate has both CV support and rollback coverage.

## Rollback strategy

### Rollback triggers
- Public score drop beyond expected shake interval.
- Sudden private mismatch signals from validation diagnostics.
- Detection of leakage or pipeline inconsistency after submission.

### Rollback mechanism
1. Keep a **champion artifact** (features, model params, seed, splitter, inference script path) for current best stable run.
2. Keep a **shadow backup** from previous stable milestone.
3. If trigger fires:
   - Re-run champion with identical seed/split config.
   - Verify score parity against historical logs (`ExperimentLogger.summary` + config hash).
   - Submit rollback artifact before testing new speculative variant.

### Minimum rollback package
- Last stable experiment ID + config hash
- Reproducible training command/notebook path
- Expected score interval from validation/shake analysis
- Known risks and invalidated assumptions
