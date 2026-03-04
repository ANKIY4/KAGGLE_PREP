# Prepare_kaggle

A university-style, notebook-first curriculum for advanced model training and competition practice, centered on **PyTorch MLP workflows** with deep mathematical explanations and applied Kaggle-style experimentation.

This folder contains:
- **20 Jupyter notebooks** (10 lessons + 10 solution notebooks)
- A generator script to recreate notebooks consistently
- A progression from fundamentals to competition playbooks

---

## What this project is for

This project is designed to help you move from “I can train a model” to “I can reason, debug, and compete.”

Core goals:
- Learn *why* each training technique works (not just how)
- Understand optimization and generalization behavior through equations + experiments
- Build strong tabular and vision habits for Kaggle-style workflows
- Practice with structured exercises from easy to challenge level

---

## Curriculum overview

Each topic has:
- `XX_topic.ipynb` (lesson)
- `XX_topic_solutions.ipynb` (fully worked solutions)

### Notebook list (10 topics, 20 files total)

| ID | Lesson Notebook | Solution Notebook | Focus |
|---|---|---|---|
| 00 | `00_baseline.ipynb` | `00_baseline_solutions.ipynb` | Baseline training dynamics |
| 01 | `01_weight_initialization.ipynb` | `01_weight_initialization_solutions.ipynb` | Xavier/He and gradient flow |
| 02 | `02_batch_normalization.ipynb` | `02_batch_normalization_solutions.ipynb` | Stabilizing optimization with BN |
| 03 | `03_dropout.ipynb` | `03_dropout_solutions.ipynb` | Stochastic regularization |
| 04 | `04_regularization.ipynb` | `04_regularization_solutions.ipynb` | L1/L2, bias-variance control |
| 05 | `05_early_stopping.ipynb` | `05_early_stopping_solutions.ipynb` | Patience, checkpointing, ablations |
| 06 | `06_optuna_hyperparameter_tuning.ipynb` | `06_optuna_hyperparameter_tuning_solutions.ipynb` | Hyperparameter search with Optuna |
| 07 | `07_xgboost_lightgbm.ipynb` | `07_xgboost_lightgbm_solutions.ipynb` | Boosting comparison for tabular competitions |
| 08 | `08_feature_engineering.ipynb` | `08_feature_engineering_solutions.ipynb` | Feature pipelines with PyTorch MLP modeling |
| 09 | `09_kaggle_competition_playbook.ipynb` | `09_kaggle_competition_playbook_solutions.ipynb` | End-to-end Kaggle preparation workflow |

---

## Pedagogical format used in notebooks

Most notebooks follow this structure:
1. **Math (LaTeX)** with symbol-by-symbol explanation
2. **Equation lineage** (what transforms into what, and why)
3. **Code walkthroughs** before/after key code cells
4. **Synthetic experiments** (controlled behavior checks)
5. **Real dataset experiments** (MNIST / California Housing / competition-style tabular data)
6. **Visual diagnostics** (loss curves, metric trends, ablations)
7. **Best-practice notes** and anti-patterns
8. **Exercise ladder** (easy -> medium -> hard -> challenge)

Solution notebooks include:
- Complete runnable implementations
- Expected outcomes and interpretation notes
- Alternative approaches and trade-off discussions

---

## PyTorch-specific implementation policy

This curriculum is PyTorch-centric.

- PyTorch (`torch`, `torch.nn`, `DataLoader`, training loops) is the primary modeling framework.
- `08` and `09` are aligned to **PyTorch MLP training/inference flows**.
- `sklearn` may be used for utilities (splits, scaling, preprocessing helpers, metrics).
- `07` is intentionally retained as a dedicated **XGBoost vs LightGBM** comparison notebook.

---

## Folder structure

```text
Prepare_kaggle/
├── 00_baseline.ipynb
├── 00_baseline_solutions.ipynb
├── ...
├── 09_kaggle_competition_playbook.ipynb
├── 09_kaggle_competition_playbook_solutions.ipynb
├── requirements.txt
├── data/
└── generators/
    ├── nb_helper.py
    └── generate_advanced_notebooks.py
```

---

## Setup

From `~/Desktop/Prepare_kaggle`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Launch Jupyter:

```bash
jupyter notebook
```

---

## How to use the curriculum

Recommended order:
1. `00` -> `06` (core deep-learning training techniques)
2. `08` (feature engineering workflows)
3. `09` (competition playbook)
4. `07` (boosting comparison for broader tabular perspective)

Study flow per topic:
- Attempt lesson notebook exercises first
- Use solutions only after writing your own reasoning
- Compare your metric curves to the solution’s expected behavior
- Record what changed optimization speed, stability, and generalization

---

## Regenerating notebooks

If you modify templates or content generation logic:

```bash
cd generators
python3 generate_advanced_notebooks.py
```

This regenerates all lesson + solution notebooks using a consistent format.

---

## Tips for Kaggle-focused practice

- Keep a strict experiment log: seed, features, model config, CV strategy, public/private behavior
- Optimize for validation discipline, not leaderboard luck
- Track generalization gaps and metric volatility across folds
- Treat notebook 09 as your competition rehearsal environment

---

## Troubleshooting

- **Dataset download delays**: rerun affected cells; some datasets download on first use.
- **CPU-only training is slow**: reduce epochs or sample sizes while debugging.
- **Inconsistent metrics**: set seeds and verify train/val split reproducibility.
- **Package import errors**: ensure virtual environment is active and `pip install -r requirements.txt` completed.

---

## Notes

- These notebooks are intentionally dense and explanation-heavy.
- The goal is deep understanding suitable for interview-level reasoning and competition performance.

