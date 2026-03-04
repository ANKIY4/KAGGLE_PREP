# COMMON KAGGLE MISTAKES (AND HOW TO PREVENT THEM)

## Purpose
Document recurring anti-patterns in competition workflows and provide prevention playbooks that transfer across datasets, metrics, and domains.

## Guiding idea
Most failures are process failures, not algorithm failures. Good process compounds; hacks do not.

## Anti-pattern catalog with prevention playbooks

### 1) Mistake: model-first, problem-second
**Pattern:** immediately trying advanced models before understanding target behavior and data generation.  
**Consequence:** noisy iteration with little causal learning.  
**Prevention playbook:**
- Build a strong baseline fast.
- Profile target, missingness, cardinality, leakage risks.
- Prioritize feature hypotheses from error slices before model complexity.

### 2) Mistake: wrong validation split
**Pattern:** random KFold on time/grouped data.  
**Consequence:** optimistic CV and leaderboard collapse.  
**Prevention playbook:**
- Match split to deployment (time, group, geography, session).
- Add purge gaps when temporal leakage is plausible.
- Keep one untouched holdout for final confidence checks.

### 3) Mistake: silent leakage in feature pipelines
**Pattern:** fitting imputers/encoders/scalers on full data or using future information in joins.  
**Consequence:** inflated CV that cannot reproduce.  
**Prevention playbook:**
- Use fold-local pipelines end-to-end.
- Generate target statistics out-of-fold only.
- Tag each feature with “available_at_prediction_time = yes/no”.

### 4) Mistake: overfitting to public leaderboard noise
**Pattern:** selecting experiments by tiny public LB bumps.  
**Consequence:** private LB drop and unstable strategy.  
**Prevention playbook:**
- Treat public LB as low-sample feedback, not truth.
- Accept changes only with CV effect size beyond fold variance.
- Prefer improvements consistent across folds and slices.

### 5) Mistake: poor experiment hygiene
**Pattern:** ad-hoc notebooks, inconsistent seeds, missing metadata.  
**Consequence:** cannot reproduce wins or diagnose losses.  
**Prevention playbook:**
- Log split version, seed, features, params, and metric for every run.
- Freeze data snapshots for major milestones.
- Require reproducible rerun before promoting an experiment.

### 6) Mistake: no structured error analysis
**Pattern:** optimizing aggregate score only.  
**Consequence:** blind spots in rare classes, tails, and cold starts.  
**Prevention playbook:**
- Analyze residuals by segment (time/group/value range).
- Build targeted features for worst slices.
- Track slice metrics alongside global metric.

### 7) Mistake: confusing significance with noise
**Pattern:** shipping +0.0002 CV gains without uncertainty analysis.  
**Consequence:** random walk disguised as progress.  
**Prevention playbook:**
- Report mean ± std across folds.
- Use repeated CV or multiple seeds for fragile setups.
- Set a minimum practical improvement threshold.

### 8) Mistake: ensembling without diversity
**Pattern:** averaging many highly correlated models.  
**Consequence:** extra complexity with negligible gain.  
**Prevention playbook:**
- Measure out-of-fold prediction correlation.
- Ensemble models with complementary error profiles.
- Keep simplest ensemble that preserves most of the gain.

### 9) Mistake: metric misalignment
**Pattern:** training for one objective while competition scores another behavior.  
**Consequence:** local optimization, global underperformance.  
**Prevention playbook:**
- Validate directly on competition metric.
- Tune thresholds/calibration if metric is decision-based.
- Monitor metric-sensitive regions (top-k, tails, rank zones).

### 10) Mistake: premature optimization of compute
**Pattern:** spending time on speed tricks before proving idea quality.  
**Consequence:** efficient execution of weak hypotheses.  
**Prevention playbook:**
- First prove signal on subset with honest CV.
- Then optimize runtime for promising directions.
- Preserve a fast baseline for sanity checks.

## What to internalize
- Reliable CV is the foundation of strategy.
- Feature and representation quality dominate most gains.
- Process discipline beats leaderboard chasing over the long run.

These principles transfer to any competition because they focus on generalization mechanics, not short-lived tricks.
