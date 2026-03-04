# Advanced Feature Engineering and Competition Strategies

This module is a competition-oriented extension of `Prepare_kaggle`, focused on rigorous feature engineering, validation design, leakage detection, ensembling, semi-supervised learning, and leaderboard simulation.

## What is included

### Notebooks (lesson + solutions)
- `10_feature_engineering_fundamentals.ipynb`
- `10_feature_engineering_fundamentals_solutions.ipynb`
- `11_advanced_feature_engineering_patterns.ipynb`
- `11_advanced_feature_engineering_patterns_solutions.ipynb`
- `12_cross_validation_mastery.ipynb`
- `12_cross_validation_mastery_solutions.ipynb`
- `13_data_leakage_detection.ipynb`
- `13_data_leakage_detection_solutions.ipynb`
- `14_ensembling_and_stacking.ipynb`
- `14_ensembling_and_stacking_solutions.ipynb`
- `15_pseudo_labeling_and_semi_supervised.ipynb`
- `15_pseudo_labeling_and_semi_supervised_solutions.ipynb`
- `16_kaggle_competition_simulator.ipynb`
- `16_kaggle_competition_simulator_solutions.ipynb`

### Reusable utilities
- `feature_utils.py` - feature transforms, polynomial expansion, interactions, OOF target encoding
- `cv_utils.py` - empirical risk helpers, splitters, OOF CV predictions, leakage inflation, shake simulation
- `ensemble_utils.py` - blending, variance/covariance formulas, OOF stacking
- `experiment_logger.py` - reproducible experiment logging and summary views

### Generator
- `build_advanced_module_notebooks.py` - regenerates all 14 notebooks with consistent structure

## Pedagogical structure in each notebook

Each notebook follows:
1. Problem Definition
2. Required Prior Knowledge
3. New Concepts Introduced
4. Formal Definition
5. Variables and Assumptions
6. Symbol-by-Symbol Explanation
7. Zero-Skip Derivation
8. Explicit Logical Transitions
9. Intuition
10. Mapping from Math to Implementation
11. Synthetic Experiment
12. Real Dataset Experiment
13. Diagnostic Analysis
14. Failure Analysis
15. Exercise Ladder
16. Summary of Mathematical Insights

## Strict concept ordering

This module is intentionally dependency-safe:
- Concepts are introduced in order from 10 to 16.
- No notebook should rely on concepts first introduced in later notebooks.
- Each notebook explicitly lists prerequisites and newly introduced concepts.

Recommended order:
1. 10 Fundamentals
2. 11 Advanced Patterns
3. 12 Cross-Validation Mastery
4. 13 Leakage Detection
5. 14 Ensembling and Stacking
6. 15 Pseudo-Labeling and Semi-Supervised
7. 16 Kaggle Competition Simulator

## Reproducibility and datasets

- Seed control is built into the notebooks/utilities.
- Dataset policy is hybrid:
  - Prefer sklearn built-ins by default
  - Use auto-download sources when needed
  - Keep deterministic fallbacks where possible

## Setup

From this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Launch Jupyter and open notebooks:

```bash
jupyter notebook
```

## Regenerate notebooks

```bash
python3 build_advanced_module_notebooks.py
```

Run validations only (structure, concept ordering, MathJax-oriented checks):

```bash
python3 build_advanced_module_notebooks.py --validate-only
```
