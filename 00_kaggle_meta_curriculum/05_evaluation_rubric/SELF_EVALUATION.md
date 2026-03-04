# SELF EVALUATION

## Purpose
Use this rubric to score your Kaggle execution quality with evidence, not intuition. Score each competency from **0-4** using your latest experiments (ideally last 2-3 competitions).

## Scoring Scale (applies to every competency)
- **0** = Not done.
- **1** = Attempted once, inconsistent, not documented.
- **2** = Basic implementation, partially repeatable.
- **3** = Strong practice, consistently applied and tracked.
- **4** = Competition-grade practice with measurable impact and clear decision logic.

## Competency Rubric

### 1) Problem Framing (0-4)
- **0:** Target, metric, and constraints are not explicitly written.
- **1:** Target and metric are written, but no baseline or hypothesis list.
- **2:** Baseline model + split strategy documented.
- **3:** At least **3 ranked hypotheses** are documented with expected impact before experiments.
- **4:** Each tested hypothesis has **expected lift vs actual lift** logged; low-ROI paths are stopped after clear evidence.

### 2) Cross-Validation Rigor (0-4)
- **0:** Single random split only.
- **1:** CV used, but split type does not match competition structure (time/group/stratification) or seed control is missing.
- **2:** Correct splitter used with fixed seeds and fold scores saved.
- **3:** Mean and std across folds are tracked; leaderboard-vs-CV gap is reviewed after submissions.
- **4:** Stability checks across multiple seeds/repeats are tracked; promotion requires stable gains (not one-fold spikes).

### 3) Leakage Handling (0-4)
- **0:** No explicit leakage checks.
- **1:** Leakage discussed informally only.
- **2:** Feature availability timing and fold isolation are manually checked before training.
- **3:** Leakage checks are part of a repeatable pre-train checklist; suspect features are ablated and compared.
- **4:** Leakage defense is systematic: every high-gain feature has provenance + availability validation, and leakage tests are logged.

### 4) Feature Iteration Quality (0-4)
- **0:** Features added ad hoc with no experiment trail.
- **1:** Experiments run, but changes are bundled and attribution is unclear.
- **2:** Feature families are tested with CV deltas recorded.
- **3:** Controlled iteration (single change or clearly scoped batch) with accept/reject decisions based on CV evidence.
- **4:** Feature work is ROI-driven: retained/rejected features include quantitative rationale and rollback path.

### 5) Ensembling Decisions (0-4)
- **0:** No ensemble strategy.
- **1:** Simple averaging without diversity analysis.
- **2:** Multiple blending options compared on out-of-fold (OOF) predictions.
- **3:** Ensemble components chosen using OOF performance + correlation/diversity evidence.
- **4:** Final ensemble choice is justified by repeatable OOF gain and risk analysis (variance/overfit checks).

### 6) Reproducibility Discipline (0-4)
- **0:** Best result cannot be reproduced reliably.
- **1:** Partial notebook history exists, but exact rerun path is unclear.
- **2:** Seeds, dependencies, and key parameters are captured.
- **3:** End-to-end rerun (data -> train -> inference -> submission) is documented and repeatable.
- **4:** Reproduction is deterministic enough for competition ops: artifact versions, config, and run metadata are all traceable.

### 7) Post-Mortem Quality (0-4)
- **0:** No retrospective.
- **1:** Generic notes with no metrics.
- **2:** Win/loss summary with top contributing decisions.
- **3:** Root-cause analysis links leaderboard movement to concrete experiment decisions.
- **4:** Next-cycle plan is measurable: prioritized hypotheses, expected impact, stop criteria, and timeline.

## Readiness Thresholds
- **0-13:** Foundation stage — focus on process basics.
- **14-20:** Developing — enter for practice, not optimization pressure.
- **21-24:** Ready — competitive workflow is reliable.
- **25-28:** Advanced readiness — strong execution discipline.

## Action Rule
Treat any competency scored **<=2** as a mandatory improvement item before your next serious competition push.
