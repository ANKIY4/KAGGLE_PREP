## Purpose
Choose the correct validation split strategy for Kaggle datasets so CV reflects leaderboard reality, while enforcing leakage checks and metric alignment before submission.

## CV Split Choice Decision Tree
Follow this in order:

1. **Is there time/order dependence (forecasting, sequential events, “future” rows)?**
   - **Yes -> Time-based split**
     - Use `TimeSeriesSplit`, rolling/expanding windows, or purged/embargoed variants.
     - Validation rows must always be later than training rows.
   - **No -> go to 2**

2. **Can one entity appear in multiple rows (user, patient, product, session, device, location)?**
   - **Yes -> Group-aware split**
     - Use `GroupKFold` (or `StratifiedGroupKFold` for classification imbalance).
     - Same group must never appear in both train and validation.
   - **No -> go to 3**

3. **Is this classification with class imbalance or rare labels?**
   - **Yes -> Stratified split**
     - Use `StratifiedKFold` to preserve label proportions.
   - **No -> go to 4**

4. **Default IID case**
   - Use `KFold` (with shuffle + fixed seed when order is arbitrary).

## Action Table (Quick Mapping)
- IID regression/classification (balanced): `KFold`
- Imbalanced classification: `StratifiedKFold`
- Repeated entities/groups: `GroupKFold`
- Groups + imbalance: `StratifiedGroupKFold`
- Time-dependent data: time split / purged time split

## Leakage Checks (Mandatory at Every Branch)
Hard-fail your experiment if any check fails:

1. **Preprocessing leakage:** scaler/imputer/PCA fitted before splitting.
2. **Encoding leakage:** target/frequency encoding not done out-of-fold.
3. **Group leakage:** same entity across train/valid in a fold.
4. **Temporal leakage:** future information used in historical samples.
5. **Join leakage:** aggregated or external tables include post-cutoff data.

If one leakage check fails, CV score is invalid regardless of magnitude.

## Metric Alignment Rules
Your CV objective must mirror the competition metric exactly:
- Same metric function (e.g., AUC, logloss, RMSE, MAP@K, custom weighted metrics).
- Same post-processing (clipping, rounding, ranking, thresholding) used at inference.
- Same sample weights/group logic when metric depends on them.
- For ranking/retrieval tasks, split and evaluate at query/group level, not row level.

Model selection rule: prefer improvements that are consistent on the exact target metric, not proxy metrics.

## Pre-Submit Validation Protocol
Execute this checklist before submitting:

1. Confirm split type matches data-generating process (IID/stratified/group/time).
2. Re-run CV with fixed seeds and log per-fold scores.
3. Compare candidate vs baseline on mean, std, and worst fold.
4. Re-run leakage checks (preprocess, encoding, join, temporal, group boundaries).
5. Verify metric implementation parity with competition definition.
6. Ensure OOF predictions are generated only from unseen folds.
7. Retrain final model with the same validated pipeline.
8. Submit only if gain is stable and leakage-clean.

This protocol is your final guardrail against public LB overfitting and private shake.
