# Prepare_kaggle

`Prepare_kaggle` is a full Kaggle learning system with three layers:
1) core deep-learning practice,  
2) advanced feature-engineering/competition strategy,  
3) a meta-curriculum control layer that tells you how to use both correctly.

---

## Repository modules

### 1) `00_kaggle_meta_curriculum/` (control layer)
- Orchestration-first guidance for learning sequence, competition operating rules, and self-evaluation.
- Includes:
  - learning path and prerequisites
  - Kaggle strategy docs (feature-first reasoning, CV failure modes, common mistakes)
  - CV mastery assets + leakage examples notebook
  - end-to-end competition simulation notebooks
  - readiness rubric/checklists

See: `00_kaggle_meta_curriculum/README.md`

### 2) `deep learning/` (core track)
- Foundation-to-intermediate notebooks (`00` to `09`) with lesson + solution pairs.
- Focus: optimization, regularization, hyperparameter tuning, model behavior, and tabular competition foundations.
- Includes:
  - generator tooling in `deep learning/generators/`
  - formula deep-dive notebook: `deep learning/10_formula_deep_dive_compendium.ipynb`

See: `deep learning/README.md`

### 3) `10_advanced_feature_engineering_and_competition_strategies/` (advanced track)
- Advanced notebooks (`10` to `16`) with lesson + solution pairs.
- Focus: feature engineering rigor, CV mastery, leakage detection, ensembling/stacking, pseudo-labeling, competition simulation.
- Includes reusable utilities:
  - `feature_utils.py`
  - `cv_utils.py`
  - `ensemble_utils.py`
  - `experiment_logger.py`
- Includes:
  - generator + validator: `build_advanced_module_notebooks.py`
  - formula deep-dive notebook: `17_formula_deep_dive_compendium.ipynb`

See: `10_advanced_feature_engineering_and_competition_strategies/README.md`

---

## Recommended learning flow

1. Start in `00_kaggle_meta_curriculum/01_learning_map/` to understand progression and prerequisites.
2. Build fundamentals in `deep learning/00` -> `deep learning/06`.
3. Do competition-oriented core notebooks: `deep learning/08`, `deep learning/09`, then `deep learning/07`.
4. Move to advanced track: `10_.../10` -> `10_.../16`.
5. Use `00_kaggle_meta_curriculum/04_competition_simulation/` and `05_evaluation_rubric/` for repeated practice and readiness checks.

If you want formula-level depth, use:
- `deep learning/10_formula_deep_dive_compendium.ipynb`
- `10_advanced_feature_engineering_and_competition_strategies/17_formula_deep_dive_compendium.ipynb`

---

## Setup

From repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

Install core track dependencies:

```bash
pip install -r "deep learning/requirements.txt"
```

Install advanced track dependencies:

```bash
pip install -r "10_advanced_feature_engineering_and_competition_strategies/requirements.txt"
```

Launch Jupyter:

```bash
jupyter notebook
```

---

## Reproducibility and validation principles

- Use fixed seeds for all experiments.
- Treat CV design as a first-class system decision.
- Reject any gain that fails leakage checks.
- Prefer stable fold-level improvements over public leaderboard spikes.
- Keep experiment tracking disciplined (features, splitter, seed, params, metric, notes).

---

## Regenerating notebooks

Core track:

```bash
cd "deep learning/generators"
python3 generate_advanced_notebooks.py
```

Advanced track:

```bash
cd "10_advanced_feature_engineering_and_competition_strategies"
python3 build_advanced_module_notebooks.py
python3 build_advanced_module_notebooks.py --validate-only
```

---

## Usage note

Use `00_kaggle_meta_curriculum/` as your operational control panel, and use the other two folders as execution engines.  
That separation is intentional and is the fastest path to reliable Kaggle skill growth.
