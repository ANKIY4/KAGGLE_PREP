# WHY CROSS-VALIDATION FAILS IN COMPETITIONS

## Purpose
Provide a reusable diagnostic framework for CV reliability: identify failure modes early, map them to concrete fixes, and make validation scores predictive of private-test performance.

## First principle
Validation is a simulator of deployment. CV fails when simulation assumptions differ from the real test-generation process.

## Failure taxonomy

### 1) Split mismatch
**What it is:** train/validation partition does not match how test data is generated (time, groups, geography, users, sessions).  
**Symptom:** strong CV, weak leaderboard or private shake-up.  
**Fix:** choose split by data-generating process (TimeSeriesSplit, GroupKFold, PurgedGroupTimeSeries-like patterns).

### 2) Leakage channels
**What it is:** validation fold indirectly sees future/target information through preprocessing, joins, encodings, or feature construction.  
**Common channels:** global target encoding, fit-on-full-data scalers/imputers, post-event features, duplicate entities across folds.  
**Fix:** enforce fold-local pipelines and out-of-fold feature generation; audit each feature’s timestamp and information availability.

### 3) Distribution drift (covariate/label/prior shift)
**What it is:** \(P(X)\), \(P(Y)\), or \(P(Y|X)\) differs between validation and test regimes.  
**Symptom:** unstable fold performance; model wins on some folds but fails on holdout era/domain.  
**Fix:** drift-aware splits, time-based holdouts, reweighting/slice monitoring, and robust features less sensitive to transient correlations.

### 4) Selection bias from repeated tuning
**What it is:** trying many ideas against the same CV effectively overfits the validator.  
**Symptom:** incremental CV gains that never translate.  
**Fix:** maintain a locked shadow holdout, track number of trials, require effect-size + variance thresholds before accepting changes.

### 5) Metric mismatch
**What it is:** optimizing loss/early stopping/objective that differs from competition metric behavior.  
**Symptom:** lower training loss but flat/worse competition score.  
**Fix:** align objective, thresholding, and post-processing with evaluation metric; validate on metric itself, not proxies.

## Corrective protocol (repeatable)
1. **Reconstruct test scenario**: infer what is known at prediction time and by whom.
2. **Design split from scenario**: time/group/purge constraints first, convenience second.
3. **Build leakage-proof pipeline**: every transform fitted within fold only.
4. **Run variance-aware CV**: report mean, std, and per-fold distribution (not just one number).
5. **Stress test by slices**: rare classes, late periods, cold-start entities, extreme targets.
6. **Use a locked checkpoint set**: untouched holdout for periodic reality checks.
7. **Promote only robust changes**: require consistent uplift across folds/slices, not single-fold spikes.

## Quick reliability checklist
- Does each validation row mirror test-time information availability?
- Can any feature be computed only after the prediction timestamp?
- Are entities leaking across folds?
- Is fold-to-fold variance small relative to claimed improvement?
- Is the optimized target exactly the scored metric?

If any answer is uncertain, trust the uncertainty: fix validation before trusting improvements.
