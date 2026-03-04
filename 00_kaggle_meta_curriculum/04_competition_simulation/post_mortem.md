# Competition Post-Mortem Template

## Purpose
Provide a structured framework to analyze competition outcomes, isolate mistakes vs variance, evaluate tradeoffs made under constraints, and convert lessons into concrete next-iteration actions tied to this repository’s modules.

## Implementation anchors in this repository
- `deep learning/09_kaggle_competition_playbook.ipynb` (execution narrative)
- `10_advanced_feature_engineering_and_competition_strategies/12_cross_validation_mastery.ipynb`
- `10_advanced_feature_engineering_and_competition_strategies/13_data_leakage_detection.ipynb`
- `10_advanced_feature_engineering_and_competition_strategies/14_ensembling_and_stacking.ipynb`
- `10_advanced_feature_engineering_and_competition_strategies/16_kaggle_competition_simulator.ipynb`
- `10_advanced_feature_engineering_and_competition_strategies/{feature_utils.py,cv_utils.py,ensemble_utils.py,experiment_logger.py}`

## Structured template

### 1) Competition snapshot
- Competition name:
- Metric:
- Final rank / team count:
- Final public score:
- Final private score:
- Best CV score (pre-submission):
- Duration:
- Submission count:

### 2) Objective vs outcome
- Initial hypothesis:
- What worked as expected:
- What diverged from expectation:
- Largest contributor to final rank (single factor):

### 3) Decision log (high impact only)
| Decision | Expected upside | Real outcome | Keep/Change |
|---|---|---|---|
| Split strategy choice |  |  |  |
| Top feature block |  |  |  |
| Ensemble or single-model choice |  |  |  |
| Final freeze strategy |  |  |  |

### 4) Mistake analysis
| Mistake | Category (validation/feature/model/process) | Symptom observed | Root cause | Detection signal | Prevention for next run |
|---|---|---|---|---|---|
|  |  |  |  |  |  |

### 5) Tradeoff analysis
| Tradeoff | Option chosen | Option rejected | Why chosen then | Cost paid | Would choose again? |
|---|---|---|---|---|---|
|  |  |  |  |  |  |

### 6) Quantitative diagnostics
- CV fold mean/std and trend:
- Public-private gap:
- Shake sensitivity estimate:
- Leaderboard delta vs champion experiment:
- Leakage inflation check:

Use:
- `cv_utils.cv_bias_variance_decomposition`
- `cv_utils.simulate_public_private_variance`
- `cv_utils.leakage_inflation`
- `experiment_logger.ExperimentLogger.summary`

### 7) Next-iteration action plan
| Action ID | Action | Owner | Module/file reference | Success metric | Priority | Deadline |
|---|---|---|---|---|---|---|
| A1 |  |  |  |  |  |  |
| A2 |  |  |  |  |  |  |
| A3 |  |  |  |  |  |  |

### 8) Reproducibility attachments
- Final experiment IDs and config hashes:
- Feature list and generation path:
- Splitter config:
- Training/inference notebook or script paths:
- Notes on discarded ideas:

## Worked example (filled)

### 1) Competition snapshot
- Competition name: Tabular Binary Classification Challenge
- Metric: ROC-AUC
- Final rank / team count: 128 / 2,430
- Final public score: 0.8902
- Final private score: 0.8829
- Best CV score: 0.8916
- Duration: 6 weeks
- Submission count: 57

### 2) Objective vs outcome
- Initial hypothesis: OOF target encoding + stacking would improve both CV and private leaderboard.
- What worked: Feature interactions and calibrated blending improved robustness.
- What diverged: Late pseudo-label variant improved public score but degraded private score.
- Largest contributor: Switching to competition-aligned CV split (`cv_utils.make_splitter`) reduced optimism bias.

### 3) Decision log (high impact)
| Decision | Expected upside | Real outcome | Keep/Change |
|---|---|---|---|
| Group-aware CV instead of plain KFold | Better generalization estimate | Public-private gap narrowed by ~0.004 | Keep |
| Added OOF target encoding (`feature_utils.target_encode_oof`) | Strong feature lift | +0.006 CV gain with stable folds | Keep |
| Added aggressive pseudo-labeling late | Extra data signal | Public +0.0015, private -0.0048 | Change |
| Final hour exploratory submission | Potential upside | Replaced safer candidate with noisier model | Change |

### 4) Mistake analysis
| Mistake | Category (validation/feature/model/process) | Symptom observed | Root cause | Detection signal | Prevention for next run |
|---|---|---|---|---|---|
| Treated public bump as robust gain | process | Public rose while CV unchanged | Over-weighted leaderboard feedback | `simulate_public_private_variance` showed bump inside shake band | Require CV delta threshold before final submissions |
| Late pseudo-label adoption without stress test | model | Private drop on final board | Distribution mismatch in pseudo labels | Fold variance increased after pseudo-label injection | Add holdout stress check in `15_pseudo_labeling_and_semi_supervised.ipynb` workflow |
| Weak rollback discipline in freeze window | process | Lost safer final position | No enforced champion/challenger lock | Logger showed safer candidate available but not submitted | Enforce rollback checklist from `competition_overview.md` |

### 5) Tradeoff analysis
| Tradeoff | Option chosen | Option rejected | Why chosen then | Cost paid | Would choose again? |
|---|---|---|---|---|---|
| Exploration vs stability in final 24h | Exploration | Stability | Chasing top-100 upside | Rank volatility and private drop | No |
| Model complexity vs interpretability | Two-layer stack (`ensemble_utils.oof_stacking`) | Weighted blend only | Better CV in offline tests | Harder failure attribution | Maybe, with stricter ablations |
| Speed vs rigor | Rapid iteration | Full leak audit each run | Time pressure | Missed warning signs | No |

### 6) Quantitative diagnostics
- CV fold mean/std: 0.8916 / 0.0062 (higher dispersion than prior stable runs)
- Public-private gap: -0.0073 (`private - public`)
- Shake sensitivity estimate: ±0.0029 equivalent band
- Leaderboard delta vs champion candidate: -0.0038 private
- Leakage inflation check: negligible (0.0006), so primary issue was process risk, not direct leakage

### 7) Next-iteration action plan
| Action ID | Action | Owner | Module/file reference | Success metric | Priority | Deadline |
|---|---|---|---|---|---|---|
| A1 | Enforce submission gate: CV improvement + shake-aware threshold | Competition lead | `experiment_logger.py`, `16_kaggle_competition_simulator.ipynb` | <0.002 median public-private gap in top 5 submissions | P0 | Week 1 |
| A2 | Add pseudo-label stress protocol before promotion | Modeling lead | `15_pseudo_labeling_and_semi_supervised.ipynb`, `cv_utils.py` | No fold-variance inflation >10% after pseudo-labeling | P1 | Week 2 |
| A3 | Pre-freeze rollback lock (champion + backup) | Ops lead | `competition_overview.md`, `experiment_logger.py` | Rollback candidate always available during freeze | P0 | Before next competition freeze |

### 8) Reproducibility attachments
- Final experiment IDs/config hashes: `exp_2026_02_14_stack_v3 (a13bf4e99d1c)`, `exp_2026_02_16_pseudo_v1 (c9f8e2a31b77)`
- Feature generation: `10_feature_engineering_fundamentals.ipynb`, `11_advanced_feature_engineering_patterns.ipynb`
- Splitter config: group-aware 5-fold via `cv_utils.make_splitter`
- Training/inference paths: `14_ensembling_and_stacking.ipynb`, `16_kaggle_competition_simulator.ipynb`
- Discarded ideas: high-variance pseudo-label variant, leakage-risk cross-fold target aggregation shortcut
