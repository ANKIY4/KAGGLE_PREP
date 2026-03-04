# READINESS CHECKLIST

## Purpose
Use this checklist for two hard gates: **before entering** a competition and **before submitting** predictions. Every item is go/no-go and tied to an action.

## Gate 1: Before Entering a Competition

| Check | GO when... (measurable) | NO-GO action |
|---|---|---|
| Objective clarity | You can state target, metric, and submission format in 3 sentences or less. | Write a one-paragraph problem brief before doing any modeling. |
| Time budget | You can commit a minimum weekly schedule (e.g., >=6 focused hours/week) until deadline. | Skip/observe competition or reduce scope to a learning sprint. |
| Baseline pipeline | A baseline run produces a valid submission file end-to-end. | Build baseline first; do not start feature engineering yet. |
| Validation design | CV split matches competition structure (time/group/stratified) and is documented. | Redesign split before trusting any score. |
| Leakage risk scan | Feature availability timing and fold isolation are checked for baseline features. | Freeze new features and run leakage checklist. |
| Experiment tracking | You have a log template with: run id, features, params, CV mean/std, notes, decision. | Set up tracker before running more experiments. |
| Resource feasibility | Training + inference cycle fits your hardware/time constraints. | Simplify model class or sampling strategy. |

**Gate-1 decision:**
- **GO** only if all checks pass.
- **NO-GO** if any check fails.

## Gate 2: Before Submitting (each submission)

| Check | GO when... (measurable) | NO-GO action |
|---|---|---|
| Submission objective | This submission tests a specific hypothesis (not random trial). | Define hypothesis + expected direction of change first. |
| CV evidence | Candidate beats current reference by a meaningful margin relative to CV noise (or is a deliberate risk test). | Run more validation/ablation before submitting. |
| Leakage re-check | New features/joins pass availability and fold-isolation checks. | Remove suspect features and rerun CV. |
| Reproducibility | Submission can be regenerated from code/config without manual notebook state. | Rebuild run into a reproducible script/config path. |
| Change isolation | You can list exactly what changed since last submission. | Re-run with isolated changes for attribution. |
| Artifact logging | You saved model version, OOF/pred files, CV summary, and submission id. | Log artifacts before uploading next file. |
| Risk management | You still retain at least one known-safe submission near deadline. | Re-upload safe baseline before high-risk tests. |

**Gate-2 decision:**
- **GO** only if every check is true.
- **NO-GO** if any item is false; fix first, then submit.
