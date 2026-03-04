## Purpose
Build a cross-validation (CV) system that predicts leaderboard behavior reliably enough to make robust Kaggle model-selection decisions, not just optimistic local scores.

## CV as Kaggle System Design
Treat CV as an estimation system:
- **Input:** train data + split strategy + metric + pipeline.
- **Output:** score estimate for unseen data (public/private test).
- **Failure mode:** overfitting to folds, leakage, or metric mismatch.

Your goal is to minimize decision error (picking the wrong model), not only maximize one CV mean.

## Estimation Error (Why CV Is Never the Truth)
Let:
- `S_cv` = CV estimate,
- `S_priv` = private leaderboard score.

Then the practical concern is `S_cv - S_priv` (estimation error).  
Even with correct splits, error exists because folds are finite samples.

Common sources:
1. **Sampling noise:** folds are small snapshots of full distribution.
2. **Distribution mismatch:** train folds do not match hidden test distribution.
3. **Procedure mismatch:** training on k-1 folds differs from training on full data at submission.
4. **Leakage artifacts:** hidden information inflates CV.

Practical implication: use CV intervals/variance, not single-point means.

## Fold Variance and Stability
Always inspect per-fold scores:
- `mean(score)` for central estimate,
- `std(score)` for stability,
- min/max fold behavior for tail risk.

Rules of thumb:
- **Low std + consistent gain** over baseline -> usually deployable.
- **High std + tiny mean gain** -> likely noise.
- **One fold carries all gain** -> likely split artifact or leakage.

For close models, prefer the one with:
1. Lower fold variance,
2. Better worst-fold score,
3. Simpler feature/pipeline assumptions.

## Public vs Private Shake (Leaderboard Reality)
Public LB is a sample of test data; private LB is a different sample.  
A model can climb public and drop private (**shake**) if it overfits:
- Public subset quirks,
- CV split artifacts,
- Validation leakage patterns.

Reduce shake by:
1. Designing splits that mirror test generation process.
2. Avoiding leaderboard-driven over-tuning.
3. Requiring repeatable gains across seeds/folds.
4. Ensembling diverse but independently validated models.

## Robust Model Selection Protocol
Use a gating process instead of “best mean wins”:

1. **Baseline lock:** keep a fixed reference pipeline/split.
2. **Single change test:** modify one component at a time.
3. **Significance gate:** accept only gains larger than noise level.
4. **Stability gate:** reject models with worse variance/worst-fold risk.
5. **Leakage gate:** rerun with stricter fold hygiene before accepting.
6. **Final retrain parity check:** ensure train/infer pipeline is identical.

## Explicit Leakage Warnings
Never trust CV if any of these occur:
- Fitting preprocessors/encoders/scalers on full data before splitting.
- Target encoding computed without out-of-fold logic.
- Feature engineering that uses future timestamps for past rows.
- Group/entity records split across train and validation folds.
- Pseudo-labels or external data merged with test-derived keys.

If leakage is suspected, invalidate historical CV comparisons and restart from a clean baseline.

## Pre-Submit Validation Protocol
Before every Kaggle submission:

1. Freeze split code, seeds, and metric implementation.
2. Confirm no transformer/feature is fitted outside fold boundaries.
3. Print per-fold scores and compare mean/std vs baseline.
4. Check OOF predictions for impossible confidence/perfect segments.
5. Verify local CV metric exactly matches competition metric definition.
6. Retrain on full train using the exact validated pipeline.
7. Submit only if gain survives variance and leakage checks.

A disciplined protocol beats aggressive leaderboard chasing over time.
