# Kaggle Meta Curriculum (Orchestration Layer)

**Purpose:** Use this folder as the control layer that orchestrates all existing learning assets in this repository into a competition-ready execution system.

## Role of this folder
This is not a content track and does **not** duplicate notebook material.  
It is the systems layer that defines sequencing, integration points, and decision gates across the repo so you can train, iterate, and submit with intent.

## Explicit map to existing tracks
| Existing folder | Primary scope | How this meta layer uses it |
|---|---|---|
| `deep learning/` | Neural-model workflows, representation learning, architecture-driven gains | Routed in when problem framing benefits from model capacity and learned features; integrated with validation and submission loops defined here |
| `10_advanced_feature_engineering_and_competition_strategies/` | Feature engineering, CV strategy, ensembling, competition tactics | Routed in for high-leverage tabular improvements, robust validation, and leaderboard-efficient iteration |

This layer coordinates both tracks as complementary subsystems rather than isolated study paths.

## Navigation guidance
1. **Define competition objective:** metric, constraints, timeline, submission budget.
2. **Pick initial route:** start in the most relevant source track above.
3. **Run iteration cycles:** build → validate → diagnose → improve.
4. **Switch or blend tracks intentionally:** move between deep models and feature/strategy work based on error analysis and CV/LB behavior.
5. **Converge to submission system:** lock reproducible pipelines, ensemble only when justified, and optimize for final ranking impact.

## Learning outcomes
After using this control layer, you should be able to:
- Design a competition workflow as an end-to-end system, not a notebook collection.
- Decide when to prioritize model architecture vs. feature/strategy leverage.
- Manage validation rigor, leaderboard risk, and iteration speed under competition pressure.
- Produce reproducible, decision-traceable experiments that improve placement odds.

## How to use this control layer
- Treat this folder as your **operating model** for each competition sprint.
- Pull implementation detail from the mapped source folders, not from duplicated notes here.
- Use this layer to enforce sequencing, checkpoint discipline, and cross-track integration.
- Revisit the route map after each major result change (CV jump, LB drift, or failure mode discovery).
