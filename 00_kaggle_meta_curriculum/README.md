# Kaggle Meta Curriculum (Control Layer)

## Purpose
This folder is the **control system** for the repository: it tells you **what to run, in what order, why it matters, and how to decide what to do next** during a Kaggle workflow.  
It is intentionally orchestration-first and does not duplicate the lesson content in other folders.

---

## What this folder is (and is not)

### This folder **is**
- A navigation and decision framework.
- A CV/leakage rigor layer.
- A competition execution simulator.
- A readiness evaluation system.

### This folder **is not**
- A replacement for `deep learning/`.
- A replacement for `10_advanced_feature_engineering_and_competition_strategies/`.
- A place for leaderboard hacks without validation discipline.

---

## How it connects to the existing repository

| Existing folder | Primary strength | How this control layer uses it |
|---|---|---|
| `deep learning/` | Baselines, optimization fundamentals, model behavior, practical training loops | Used to build reliable first systems and model intuition before advanced competition strategy |
| `10_advanced_feature_engineering_and_competition_strategies/` | Feature engineering, CV mastery, leakage detection, ensembling, pseudo-labeling, competition simulator | Used for high-leverage tabular upgrades after validation discipline is proven |

Operational rule: **use this folder to route decisions; use the other two folders to execute technical depth**.

---

## Folder map and exact role of each file

### `01_learning_map/`
- `LEARNING_PATH.md`  
  Purpose: stage-by-stage progression with milestones and mastery evidence.
- `PREREQUISITES.md`  
  Purpose: prerequisite matrix and hard gating rules (especially CV/leakage discipline).

### `02_kaggle_thinking/`
- `WHY_FEATURES_BEAT_MODELS.md`  
  Purpose: reasoning framework for representation-first improvements.
- `WHY_CV_FAILS.md`  
  Purpose: CV failure taxonomy and corrective protocol.
- `COMMON_KAGGLE_MISTAKES.md`  
  Purpose: anti-patterns and prevention playbooks.

### `03_cross_validation_mastery/`
- `cv_theory.md`  
  Purpose: CV as system design under estimation uncertainty.
- `cv_decision_tree.md`  
  Purpose: practical split-selection decision tree (IID/stratified/group/time).
- `leakage_examples.ipynb`  
  Purpose: runnable leakage demonstrations (flawed vs corrected pipelines).

### `04_competition_simulation/`
- `competition_overview.md`  
  Purpose: end-to-end lifecycle, checkpoints, governance, rollback policy.
- `baseline.ipynb`  
  Purpose: reproducible baseline with correct CV protocol.
- `feature_iteration.ipynb`  
  Purpose: controlled feature ablations and acceptance criteria.
- `ensembling.ipynb`  
  Purpose: OOF-aware blend/stack workflow and risk framing.
- `post_mortem.md`  
  Purpose: structured retrospective template + worked example.

### `05_evaluation_rubric/`
- `SELF_EVALUATION.md`  
  Purpose: measurable 0–4 competency rubric.
- `READINESS_CHECKLIST.md`  
  Purpose: hard go/no-go checks before competition entry and submission.

---

## Recommended usage modes

## 1) First-time repository user
1. Read `01_learning_map/LEARNING_PATH.md`.
2. Validate entry conditions with `01_learning_map/PREREQUISITES.md`.
3. Build mental model via `02_kaggle_thinking/`.
4. Do leakage/CV hardening in `03_cross_validation_mastery/`.
5. Run simulation sequence in `04_competition_simulation/`.
6. Score yourself in `05_evaluation_rubric/`.

## 2) Active competition user
1. Start each cycle from `04_competition_simulation/competition_overview.md`.
2. Enforce split/metric/leakage checks via `03_cross_validation_mastery/`.
3. Route implementation to:
   - `deep learning/` for model mechanics and baseline strength.
   - `10_advanced_feature_engineering_and_competition_strategies/` for feature/CV/ensemble upgrades.
4. Run post-mortem after milestone submissions.

## 3) Recovering from unstable leaderboard behavior
1. Pause model complexity.
2. Re-run `WHY_CV_FAILS.md` + `cv_decision_tree.md`.
3. Reproduce leakage notebook patterns against your pipeline assumptions.
4. Rebuild from stable baseline and re-promote only variance-robust gains.

---

## Non-negotiable operating rules

1. **CV design comes before model tuning.**
2. **Leakage checks are mandatory for every promoted experiment.**
3. **Feature engineering is prioritized before architecture swapping (for tabular tasks).**
4. **Every experiment must be reproducible and decision-traceable.**
5. **Public leaderboard movement alone is not sufficient evidence.**

If any rule fails, do not promote the experiment.

---

## End-to-end execution loop (competition-ready)

Use this loop continuously:

1. Frame problem and metric.
2. Build baseline with trustworthy split.
3. Run diagnostics and slice errors.
4. Propose feature/model changes from evidence.
5. Validate with fold mean + fold variance + leakage checks.
6. Promote only robust gains.
7. Submit under controlled policy.
8. Run post-mortem and update next-cycle hypotheses.

This loop is designed to build **skill compounding**, not random leaderboard spikes.

---

## Quick start checklist

Before serious competition work, confirm:
- [ ] I can justify my split strategy from data generation.
- [ ] I can explain at least 3 leakage channels in my domain.
- [ ] I have a reproducible baseline pipeline.
- [ ] I can show fold-level score variance, not just a mean.
- [ ] I know my rollback candidate before risky submissions.

If any box is unchecked, start from `01_learning_map/` and `03_cross_validation_mastery/`.

---

## Why this layer matters

Many Kaggle workflows fail due to process gaps, not model quality.  
This folder exists to prevent:
- split mismatch,
- hidden leakage,
- noisy experiment selection,
- public-LB overreaction,
- and non-reproducible “wins.”

If you follow this layer rigorously, you can use the repository as a coherent system and progress toward competition-grade execution without guesswork.
