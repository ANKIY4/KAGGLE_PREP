"""Generate advanced lesson notebooks and paired solution notebooks."""
from nb_helper import new_notebook, md, code, save

LESSONS = [
    {
        "id": "00",
        "slug": "baseline",
        "title": "Baseline Training Dynamics",
        "core_question": "What does a strong baseline look like before optimization tricks?",
        "math": r"""
### Mathematical Foundation (First Principles)

**Goal:** minimize empirical risk

$$
\mathcal{L}(\theta) = \frac{1}{N}\sum_{i=1}^{N}\ell(f_\theta(x_i), y_i)
$$

**Symbol table**

| Symbol | Meaning |
|---|---|
| $\theta$ | all trainable model parameters |
| $f_\theta(\cdot)$ | model mapping from input to prediction |
| $\ell(\hat{y}, y)$ | per-sample loss |
| $N$ | batch or dataset size |

**Equation lineage**

1. Start with average loss over samples.
2. Compute gradient: $\nabla_\theta \mathcal{L}(\theta)$.
3. Gradient step: $\theta_{t+1} = \theta_t - \eta \nabla_\theta\mathcal{L}(\theta_t)$.

Every later lesson will change one component of this pipeline and explain why.
""",
        "technique_notes": "No additional technique enabled. Establish a quantitative control model.",
        "best_practices": [
            "Always lock random seeds and report baseline variance across runs.",
            "Track both train and validation curves before introducing complexity.",
            "Baseline first avoids attributing gains to wrong techniques.",
        ],
        "alternatives": "Alternative baseline: linear model or shallow tree to benchmark neural-network value-add.",
    },
    {
        "id": "01",
        "slug": "weight_initialization",
        "title": "Weight Initialization",
        "core_question": "How does initialization shape gradient flow and convergence speed?",
        "math": r"""
### Mathematical Foundation (First Principles)

Variance-preserving initialization keeps activations and gradients numerically stable across depth.

For layer $l$ with fan-in $n_{in}$:

$$
\mathrm{Var}[W_l x_l] \approx \mathrm{Var}[W_l] \cdot n_{in} \cdot \mathrm{Var}[x_l]
$$

To keep $\mathrm{Var}[W_l x_l]$ near $\mathrm{Var}[x_l]$, choose:

$$
\mathrm{Var}[W_l] \propto \frac{1}{n_{in}}
$$

This yields Xavier/He-style scaling depending on activation type.

**Why each term matters**
- $n_{in}$ increases summed variance; larger fan-in requires smaller weight variance.
- Wrong scaling causes exploding or vanishing signals.
""",
        "technique_notes": "Compare default init vs Xavier uniform vs Kaiming normal.",
        "best_practices": [
            "Use He/Kaiming for ReLU-family activations.",
            "Inspect activation/gradient histograms early in training.",
            "Initialization and learning rate should be tuned together.",
        ],
        "alternatives": "Orthogonal initialization can further stabilize very deep networks.",
    },
    {
        "id": "02",
        "slug": "batch_normalization",
        "title": "Batch Normalization",
        "core_question": "Why does batch normalization accelerate and stabilize optimization?",
        "math": r"""
### Mathematical Foundation (First Principles)

For activations $a$ in a mini-batch $\mathcal{B}$:

$$
\mu_\mathcal{B} = \frac{1}{m}\sum_{i=1}^m a_i, \quad
\sigma^2_\mathcal{B} = \frac{1}{m}\sum_{i=1}^m (a_i-\mu_\mathcal{B})^2
$$

Normalize and re-parameterize:

$$
\hat{a}_i = \frac{a_i-\mu_\mathcal{B}}{\sqrt{\sigma^2_\mathcal{B}+\epsilon}}, \quad
y_i = \gamma \hat{a}_i + \beta
$$

**Term-by-term intuition**
- $\mu_\mathcal{B}, \sigma^2_\mathcal{B}$ control shifting/scaling drift.
- $\epsilon$ prevents division instability.
- $\gamma,\beta$ restore representational flexibility.
""",
        "technique_notes": "Enable BatchNorm1d after linear layers and analyze convergence smoothness.",
        "best_practices": [
            "Use sufficient batch size for stable statistics.",
            "Switch to eval mode for inference to use running stats.",
            "BatchNorm can reduce sensitivity to learning rate, but not remove it.",
        ],
        "alternatives": "LayerNorm is often better for sequence models and tiny batches.",
    },
    {
        "id": "03",
        "slug": "dropout",
        "title": "Dropout",
        "core_question": "How does stochastic neuron masking improve generalization?",
        "math": r"""
### Mathematical Foundation (First Principles)

Dropout samples a Bernoulli mask $m_j \sim \mathrm{Bernoulli}(1-p)$ and applies:

$$
\tilde{h}_j = \frac{m_j}{1-p} h_j
$$

The scaling factor $\frac{1}{1-p}$ preserves expected activation magnitude:

$$
\mathbb{E}[\tilde{h}_j] = h_j
$$

**Why this helps**
- Reduces co-adaptation of features.
- Acts as stochastic regularization.
- Often improves validation behavior when overfitting is present.
""",
        "technique_notes": "Inject dropout in hidden layers and study train/val gap changes.",
        "best_practices": [
            "Use moderate rates (e.g., 0.1-0.5) and tune per architecture.",
            "Disable dropout during evaluation (`model.eval()`).",
            "Combine with weight decay for stronger regularization when needed.",
        ],
        "alternatives": "Monte Carlo Dropout at inference provides uncertainty estimates.",
    },
    {
        "id": "04",
        "slug": "regularization",
        "title": "Regularization (L1/L2)",
        "core_question": "How do explicit penalties control model complexity and overfitting?",
        "math": r"""
### Mathematical Foundation (First Principles)

Penalized objective:

$$
\mathcal{J}(\theta) = \mathcal{L}(\theta) + \lambda_2 \lVert\theta\rVert_2^2 + \lambda_1 \lVert\theta\rVert_1
$$

Gradient contribution:

$$
\nabla_\theta \mathcal{J} = \nabla_\theta\mathcal{L} + 2\lambda_2\theta + \lambda_1\,\mathrm{sign}(\theta)
$$

**Interpretation**
- L2 shrinks weights smoothly and improves conditioning.
- L1 promotes sparsity (feature selection-like behavior).
""",
        "technique_notes": "Compare no penalty, L2 weight decay, and optional L1 augmentation.",
        "best_practices": [
            "Tune regularization with validation metrics, not training loss alone.",
            "Large penalties can underfit; inspect bias-variance tradeoff.",
            "Use optimizer-native weight decay for L2 when possible.",
        ],
        "alternatives": "Elastic Net blends L1 and L2 with a controlled ratio.",
    },
    {
        "id": "05",
        "slug": "early_stopping",
        "title": "Early Stopping",
        "core_question": "Why can stopping time act as an implicit regularizer?",
        "math": r"""
### Mathematical Foundation (First Principles)

Training continues while validation loss improves.

Let $v_t$ be validation loss at epoch $t$. With patience $P$, stop when:

$$
\min_{k \in [t-P+1, t]} v_k > v_{best}
$$

**Why this regularizes**
- Prevents prolonged fitting of noise.
- Chooses a parameter iterate with better generalization.
- Acts similarly to complexity control in iterative methods.
""",
        "technique_notes": "Implement patience-based stopping and compare selected epoch vs fixed-epoch training.",
        "best_practices": [
            "Track a smoothed validation metric if curves are noisy.",
            "Checkpoint best model state, not the final epoch.",
            "Patience should reflect dataset noise and epoch cost.",
        ],
        "alternatives": "ReduceLROnPlateau can complement early stopping before stopping entirely.",
    },
    {
        "id": "06",
        "slug": "optuna_hyperparameter_tuning",
        "title": "Optuna Hyperparameter Tuning",
        "core_question": "How do we optimize hyperparameters efficiently under compute constraints?",
        "math": r"""
### Mathematical Foundation (First Principles)

Hyperparameter optimization solves:

$$
\lambda^* = \arg\min_{\lambda \in \Lambda} \mathcal{V}(\lambda)
$$

where $\mathcal{V}(\lambda)$ is validation objective after training with hyperparameters $\lambda$.

Optuna uses adaptive sampling and pruning to reduce wasted trials.

**Why this helps**
- Prioritizes promising regions of search space.
- Stops weak trials early to reallocate compute.
- Improves reproducible tuning workflows.
""",
        "technique_notes": "Use Optuna to tune lr, hidden width, dropout, and weight decay with trial pruning.",
        "best_practices": [
            "Constrain search ranges with domain knowledge.",
            "Log trial artifacts and seeds for reproducibility.",
            "Use pruning when trial cost is high.",
        ],
        "alternatives": "Bayesian optimization libraries (e.g., BoTorch) can be used for expensive objectives.",
    },
    {
        "id": "07",
        "slug": "xgboost_lightgbm",
        "title": "XGBoost vs LightGBM for Tabular Competitions",
        "core_question": "When should we prefer XGBoost vs LightGBM under real Kaggle constraints?",
        "math": r"""
### Mathematical Foundation (First Principles)

Both methods build additive tree ensembles:

$$
\hat{y}^{(t)}(x)=\hat{y}^{(t-1)}(x)+\eta f_t(x),\qquad
f_t \in \mathcal{F}_{\text{trees}}
$$

XGBoost objective (second-order approximation):

$$
\mathcal{L}^{(t)} \approx \sum_i \left[g_i f_t(x_i) + \frac{1}{2}h_i f_t(x_i)^2\right] + \Omega(f_t)
$$

LightGBM keeps boosting objective but changes tree-growth/search strategy (leaf-wise growth + histogram bins + GOSS/EFB) for faster optimization.

**Trade-off intuition**
- XGBoost: conservative/stable defaults, strong regularization controls.
- LightGBM: often faster on large/high-dimensional tables, but can overfit if leaf complexity is unchecked.
""",
        "technique_notes": "Run side-by-side regression/classification benchmarks and compare RMSE/AUC, runtime, and model complexity.",
        "best_practices": [
            "Tune learning rate and number of estimators jointly (small eta -> more trees).",
            "Use cross-validation with fixed folds for fair framework comparison.",
            "Track both score and wall-clock time; Kaggle iteration speed matters.",
        ],
        "alternatives": "CatBoost can outperform both when categorical handling and minimal preprocessing are priorities.",
    },
    {
        "id": "08",
        "slug": "feature_engineering",
        "title": "Feature Engineering Workflows for Tabular Kaggle",
        "core_question": "How do we design leakage-safe, high-impact feature pipelines for tabular competitions?",
        "math": r"""
### Mathematical Foundation (First Principles)

Feature engineering can be viewed as a map $\phi(x)$ from raw inputs to a richer representation:

$$
\hat{y} = f_\theta(\phi(x))
$$

Bias-variance view:

$$
\mathbb{E}[(y-\hat{y})^2] = \text{Bias}^2 + \text{Variance} + \sigma^2
$$

Good feature design reduces bias without exploding variance or leakage.

For target encoding of category $c$ with smoothing:

$$
\tilde{\mu}_c = \frac{n_c\mu_c + \alpha\mu_{\text{global}}}{n_c + \alpha}
$$

computed out-of-fold to avoid target leakage.
""",
        "technique_notes": "Build a full workflow: profiling -> baseline -> engineered features -> leakage-safe validation -> ablation.",
        "best_practices": [
            "Always compare engineered pipelines against a leakage-safe baseline.",
            "Use out-of-fold encodings for target-based categorical statistics.",
            "Track each engineered block with ablations to keep only high-signal features.",
        ],
        "alternatives": "Learned representations from tabular neural nets can complement manual features in ensemble stacks.",
    },
    {
        "id": "09",
        "slug": "kaggle_competition_playbook",
        "title": "Kaggle Competition Playbook",
        "core_question": "How do we convert notebook experiments into robust leaderboard strategy?",
        "math": r"""
### Mathematical Foundation (First Principles)

Competition optimization should target expected private leaderboard performance:

$$
\theta^* = \arg\min_{\theta \in \Theta} \mathbb{E}[\mathcal{M}_{\text{private}}(\theta)]
$$

Cross-validation provides an estimator:

$$
\widehat{\mathcal{M}}_{\text{CV}}(\theta)=\frac{1}{K}\sum_{k=1}^K \mathcal{M}^{(k)}(\theta)
$$

Model averaging for two predictors:

$$
\hat{y}_{\text{ens}} = w\hat{y}_1 + (1-w)\hat{y}_2,\quad w\in[0,1]
$$

The practical challenge is minimizing **public-private shake** while iterating quickly.
""",
        "technique_notes": "Simulate leaderboard behavior, build experiment tracking, and practice robust CV + ensembling decisions.",
        "best_practices": [
            "Treat feature engineering, CV design, and ensembling as a coupled system.",
            "Optimize for robust validation and shake resistance, not public leaderboard spikes.",
            "Log every run with data version, folds, seed, model hash, and metric deltas.",
        ],
        "alternatives": "Stacking with meta-models and pseudo-labeling can add lift when shake risk is managed carefully.",
    },
]


def add_common_intro(nb, lesson, solution):
    suffix = "Solutions" if solution else "Lesson"
    md(
        nb,
        f"""
# {lesson['id']} — {lesson['title']} ({suffix})

## Learning Objectives
- Understand **why** each mathematical term appears and how it changes optimization/generalization.
- Apply the technique to both classification and regression.
- Compare synthetic and real-data behavior with reproducible experiments.

## Driving Question
> {lesson['core_question']}

## Notebook Roadmap
Math (LaTeX) -> Synthetic Data -> Real Data (MNIST/California Housing) -> Visualizations -> Best Practices -> Exercises
""",
    )


def add_math_section(nb, lesson, solution):
    md(nb, lesson["math"])
    md(
        nb,
        r"""
### Deep Equation Lineage (Term-by-Term)

We now connect every lesson to the same first-principles optimization pipeline.

1. **Population objective** (what we really care about):
$$
\mathcal{R}(\theta)=\mathbb{E}_{(x,y)\sim\mathcal{D}}[\ell(f_\theta(x),y)]
$$
2. **Empirical approximation** (what we can compute):
$$
\hat{\mathcal{R}}(\theta)=\frac{1}{N}\sum_{i=1}^{N}\ell(f_\theta(x_i),y_i)
$$
3. **Mini-batch stochastic estimate** (what we optimize per step):
$$
g_t=\frac{1}{m}\sum_{i \in \mathcal{B}_t}\nabla_\theta \ell(f_\theta(x_i),y_i)
$$
4. **Chain-rule expansion** (how gradients flow through layers):
$$
\nabla_\theta \ell
=
\frac{\partial \ell}{\partial \hat{y}}
\cdot
\frac{\partial \hat{y}}{\partial h_L}
\cdot
\frac{\partial h_L}{\partial h_{L-1}}
\cdots
\frac{\partial h_1}{\partial \theta}
$$
5. **Regularized update** (how each lesson perturbs the step):
$$
\theta_{t+1}
=
\theta_t
-\eta\left(g_t+\nabla_\theta\Omega(\theta_t)\right)
$$

| Term | Meaning | Where it appears in code |
|---|---|---|
| $\eta$ | step size controlling update magnitude | optimizer learning rate |
| $g_t$ | stochastic gradient estimate | `loss.backward()` |
| $\Omega(\theta)$ | explicit/implicit regularization | weight decay, dropout, early stopping |
| $m$ | mini-batch size | `DataLoader(..., batch_size=...)` |
| $\hat{y}=f_\theta(x)$ | model predictions | `logits = model(xb)` |

**Interpretation checkpoint:** if training is unstable, inspect whether the issue comes from gradient scale ($g_t$), step size ($\eta$), or noisy estimation (small $m$).
""",
    )
    if solution:
        md(
            nb,
            """
### Equation-to-Code Bridge (Detailed)

For each equation term, we map it directly to code objects:
- Objective terms -> `criterion(...)` and explicit regularization additions.
- Optimization step -> `loss.backward()` + `optimizer.step()`.
- Technique-specific parameters -> module flags (`BatchNorm1d`, `Dropout`, initialization hooks, patience logic, Optuna trial params).

In solution cells below, each advanced operation is annotated with **why** it exists.
""",
        )


def add_code_walkthrough(nb, preface, source, debrief):
    md(
        nb,
        f"""
### Before the Code: Purpose + Mechanics
{preface}
""",
    )
    code(nb, source)
    md(
        nb,
        f"""
### After the Code: Background + Why It Can Help
{debrief}
""",
    )


def add_setup_code(nb, lesson):
    add_code_walkthrough(
        nb,
        """
This setup cell imports all libraries and sets deterministic seeds so differences across lessons are caused by techniques, not randomness.
Background: reproducibility is essential for comparing optimization methods, and explicit device selection explains hardware-dependent speedups.
""",
        """
# Core Python utilities for copy-safe checkpointing and numerical diagnostics.
import copy
import math
import random
from dataclasses import dataclass

# Scientific stack for array manipulation and plotting.
import numpy as np
import matplotlib.pyplot as plt

# PyTorch core modules.
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Classical ML utilities for controlled data generation and scaling.
from sklearn.datasets import make_classification, make_regression, fetch_california_housing
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Torchvision datasets for image experiments and Optuna for tuning.
from torchvision import datasets, transforms
import optuna

# Seed all major random generators so curves are comparable lesson-to-lesson.
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Choose accelerator when available; this influences runtime but not algorithmic logic.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)
""",
        """
When this cell runs, the notebook environment is fully reproducible and equipped for diagnostics (including confusion matrices).
Deterministic seeds improve interpretation quality because shifts in curves are more likely to come from method changes than random variation.
""",
    )


    add_code_walkthrough(
        nb,
        """
This cell defines the experiment configuration, a compact MLP, and helper regularization/initialization routines.
Background: the architecture is intentionally small to keep runtime practical while still exhibiting optimization pathologies and improvements.
""",
        """
# Dataclass keeps all tunable hyperparameters in one inspectable object.
@dataclass
class Config:
    hidden_dim: int = 128
    lr: float = 1e-3
    epochs: int = 8
    batch_size: int = 128
    dropout_p: float = 0.0
    use_batch_norm: bool = False
    init_scheme: str = 'default'
    weight_decay: float = 0.0
    l1_lambda: float = 0.0
    patience: int = 3


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, use_batch_norm=False, dropout_p=0.0):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim)]  # First affine map from input to hidden representation.
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))  # Normalize hidden activations to stabilize gradient flow.
        layers += [nn.ReLU()]  # Nonlinearity allows composition beyond linear regression.
        if dropout_p > 0:
            layers.append(nn.Dropout(dropout_p))  # Stochastic masking reduces co-adaptation.

        layers.append(nn.Linear(hidden_dim, hidden_dim))  # Second hidden transformation for extra capacity.
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers += [nn.ReLU()]
        if dropout_p > 0:
            layers.append(nn.Dropout(dropout_p))

        layers.append(nn.Linear(hidden_dim, output_dim))  # Final projection to logits/regression output.
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def apply_initialization(model, scheme='default'):
    # Explicit initialization lets us test variance-preserving hypotheses directly.
    for m in model.modules():
        if isinstance(m, nn.Linear):
            if scheme == 'xavier':
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif scheme == 'kaiming':
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)


def l1_penalty(model):
    # L1 adds sparsity pressure by summing absolute parameter magnitudes.
    return sum(p.abs().sum() for p in model.parameters())
""",
        """
Running this cell establishes the controllable training object we will reuse across all datasets.
Because each lesson toggles only a few `Config` fields, changes in speed/stability/accuracy are easier to attribute causally.
""",
    )

    cfg_overrides = {
        "00": "cfg = Config()",
        "01": "cfg = Config(init_scheme='kaiming', lr=8e-4)",
        "02": "cfg = Config(use_batch_norm=True, lr=1.2e-3)",
        "03": "cfg = Config(dropout_p=0.3, lr=1e-3)",
        "04": "cfg = Config(weight_decay=1e-4, l1_lambda=5e-7, lr=9e-4)",
        "05": "cfg = Config(patience=2, epochs=18, lr=9e-4)",
        "06": "cfg = Config(epochs=6, batch_size=128)",
    }
    add_code_walkthrough(
        nb,
        """
This cell applies lesson-specific hyperparameter overrides while keeping the rest of the training pipeline fixed.
Background: controlled interventions are key to fair ablations in university-level experiments.
""",
        f"""
# Technique configuration for this lesson.
# Only the relevant controls are changed so observed behavior maps to lesson concepts.
# Keeping non-target hyperparameters fixed preserves a clean causal comparison.
{cfg_overrides[lesson['id']]}
# Printing confirms the active knobs before any training side effects occur.
print(cfg)
""",
        """
You now have a single `cfg` object encoding the current lesson's inductive bias.
Small parameter changes can influence optimization geometry, gradient noise scale, and generalization gap.
""",
    )


def add_training_utilities(nb):
    add_code_walkthrough(
        nb,
        """
These utilities implement the full optimization loop and attach diagnostics for gradient scale, parameter norm, and generalization gap.
Background: monitoring only loss can hide instability; richer traces explain why a method improves speed, stability, or accuracy.
""",
        """
def parameter_l2_norm(model):
    # Parameter norm helps diagnose under-regularization (exploding norm) vs over-regularization (collapsed norm).
    squared_sum = 0.0
    with torch.no_grad():
        for p in model.parameters():
            squared_sum += p.detach().pow(2).sum().item()
    return math.sqrt(squared_sum)


def run_epoch(model, loader, criterion, optimizer=None, task='classification', l1_lambda=0.0):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, total_correct, total_count = 0.0, 0, 0
    grad_norm_sum, grad_steps = 0.0, 0

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        if is_train:
            optimizer.zero_grad()

        # Forward pass computes predictions from current parameters.
        logits = model(xb)
        if task == 'classification':
            loss = criterion(logits, yb)
            preds = logits.argmax(dim=1)
            total_correct += (preds == yb).sum().item()
        else:
            yb = yb.float().unsqueeze(1)
            loss = criterion(logits, yb)

        # Add explicit L1 penalty term from the regularized objective.
        if l1_lambda > 0:
            loss = loss + l1_lambda * l1_penalty(model)

        if is_train:
            # Backprop builds d(loss)/d(theta) through chain rule over all layers.
            loss.backward()

            # Track batch gradient norm to quantify optimization stability.
            grad_sq = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    grad_sq += p.grad.detach().pow(2).sum().item()
            grad_norm_sum += math.sqrt(grad_sq)
            grad_steps += 1

            # Optimizer uses gradient + learning rate + weight decay to update parameters.
            optimizer.step()

        total_loss += loss.item() * xb.size(0)
        total_count += xb.size(0)

    avg_loss = total_loss / max(total_count, 1)
    avg_grad_norm = grad_norm_sum / max(grad_steps, 1) if is_train else None
    if task == 'classification':
        return avg_loss, total_correct / max(total_count, 1), avg_grad_norm
    return avg_loss, None, avg_grad_norm


def fit(model, train_loader, val_loader, criterion, cfg, task='classification', use_early_stopping=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_state = copy.deepcopy(model.state_dict())
    best_val = float('inf')
    wait = 0

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_metric': [],
        'val_metric': [],
        'train_grad_norm': [],
        'param_norm': [],
        'generalization_gap': [],
    }
    best_epoch = 1
    stopped_epoch = cfg.epochs

    for epoch in range(cfg.epochs):
        tr_loss, tr_metric, tr_grad = run_epoch(model, train_loader, criterion, optimizer, task=task, l1_lambda=cfg.l1_lambda)
        va_loss, va_metric, _ = run_epoch(model, val_loader, criterion, optimizer=None, task=task, l1_lambda=cfg.l1_lambda)

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(va_loss)
        history['train_metric'].append(tr_metric)
        history['val_metric'].append(va_metric)
        history['train_grad_norm'].append(tr_grad)
        history['param_norm'].append(parameter_l2_norm(model))

        # Classification gap uses accuracy difference; regression gap uses loss difference.
        if task == 'classification' and tr_metric is not None and va_metric is not None:
            history['generalization_gap'].append(va_metric - tr_metric)
        else:
            history['generalization_gap'].append(va_loss - tr_loss)

        if va_loss < best_val - 1e-6:
            best_val = va_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            wait = 0
        else:
            wait += 1

        if use_early_stopping and wait >= cfg.patience:
            stopped_epoch = epoch + 1
            print(f'Early stop at epoch {epoch+1}; best val loss={best_val:.4f}')
            break

    history['best_epoch'] = best_epoch
    history['stopped_epoch'] = stopped_epoch
    model.load_state_dict(best_state)
    return history


def plot_history(history, title, metric_name='metric'):
    # Three coordinated views: objective trajectory, metric trajectory, and gradient/parameter dynamics.
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].plot(history['train_loss'], label='train')
    axes[0].plot(history['val_loss'], label='val')
    axes[0].set_title(f'{title} - loss')
    axes[0].set_xlabel('epoch')
    axes[0].legend()

    if history['train_metric'] and history['train_metric'][0] is not None:
        axes[1].plot(history['train_metric'], label='train')
        axes[1].plot(history['val_metric'], label='val')
        axes[1].set_title(f'{title} - {metric_name}')
        axes[1].set_xlabel('epoch')
        axes[1].legend()
    else:
        axes[1].axis('off')

    axes[2].plot(history['train_grad_norm'], label='grad norm')
    axes[2].plot(history['param_norm'], label='param norm')
    axes[2].set_title(f'{title} - optimization diagnostics')
    axes[2].set_xlabel('epoch')
    axes[2].legend()
    plt.tight_layout()
    plt.show()
""",
        """
After executing these utilities, every experiment logs enough signals to reason about failure modes (e.g., noisy gradients, widening gap).
This improves educational depth by tying formulas directly to observable optimization behavior.
""",
    )


def add_synthetic_sections(nb, lesson, solution):
    md(
        nb,
        """
## Synthetic Data — Classification
We first isolate behavior on controllable synthetic data to build causal intuition before introducing dataset complexity.
""",
    )

    add_code_walkthrough(
        nb,
        """
This experiment creates a controlled binary classification problem, trains the lesson-specific model, and stores confidence/confusion diagnostics.
Background: synthetic data removes real-world noise so we can attribute performance differences to initialization/normalization/regularization decisions.
""",
        f"""
# Generate a classification dataset with moderate class separation to expose optimization behavior.
Xc, yc = make_classification(
    n_samples=4000,
    n_features=20,
    n_informative=12,
    n_redundant=4,
    class_sep=1.3,
    random_state=SEED
)

# Standardization aligns feature scales so gradients are not dominated by large-magnitude inputs.
Xc = StandardScaler().fit_transform(Xc)
Xc_tr, Xc_va, yc_tr, yc_va = train_test_split(Xc, yc, test_size=0.2, random_state=SEED)

# Convert arrays to tensors and construct mini-batch loaders.
train_ds = TensorDataset(torch.tensor(Xc_tr, dtype=torch.float32), torch.tensor(yc_tr, dtype=torch.long))
val_ds = TensorDataset(torch.tensor(Xc_va, dtype=torch.float32), torch.tensor(yc_va, dtype=torch.long))
train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)

# Build model with lesson-specific knobs (dropout, batch norm, init scheme, etc.).
model_cls = MLP(input_dim=20, output_dim=2, hidden_dim=cfg.hidden_dim, use_batch_norm=cfg.use_batch_norm, dropout_p=cfg.dropout_p).to(device)
apply_initialization(model_cls, cfg.init_scheme)

# Fit model and collect full history for interpretation.
history_cls = fit(
    model=model_cls,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=nn.CrossEntropyLoss(),
    cfg=cfg,
    task='classification',
    use_early_stopping={str(lesson['id'] == '05')}
)

# Plot training curves and optimization diagnostics.
plot_history(history_cls, title='Synthetic Classification', metric_name='accuracy')

# Cache confidence and confusion-matrix data for downstream diagnostics section.
with torch.no_grad():
    logits_va = model_cls(torch.tensor(Xc_va, dtype=torch.float32).to(device))
    synthetic_cls_probs = torch.softmax(logits_va, dim=1)[:, 1].cpu().numpy()
    synthetic_cls_preds = logits_va.argmax(dim=1).cpu().numpy()
synthetic_cls_targets = yc_va
synthetic_cls_confusion = confusion_matrix(synthetic_cls_targets, synthetic_cls_preds)

print('Final val accuracy:', history_cls['val_metric'][-1])
""",
        """
You now have both scalar metrics and distribution-level diagnostics, which is critical for explaining *why* a method helps.
Confidence histograms and confusion matrices often reveal calibration or class-specific weaknesses hidden by accuracy alone.
""",
    )


    if solution:
        md(
            nb,
            """
**Expected behavior:**
- Validation accuracy should rise quickly and stabilize.
- Depending on lesson technique, train/val gap should narrow (regularization) or convergence should speed up (BN/init).
- For this synthetic setup, final val accuracy often lands around **0.85-0.96**.
""",
        )

    md(
        nb,
        """
## Synthetic Data — Regression
Now test the same technique on regression to verify cross-task consistency.
""",
    )

    add_code_walkthrough(
        nb,
        """
This cell repeats the ablation on regression to test whether lesson benefits transfer beyond classification.
Background: robust techniques should improve optimization statistics across objective types, not only cross-entropy.
""",
        f"""
# Generate regression data with controlled signal-to-noise ratio.
Xr, yr = make_regression(n_samples=4500, n_features=16, n_informative=10, noise=15.0, random_state=SEED)

# Scale inputs/targets so optimization operates on comparable numeric ranges.
Xr = StandardScaler().fit_transform(Xr)
yr = StandardScaler().fit_transform(yr.reshape(-1, 1)).reshape(-1)
Xr_tr, Xr_va, yr_tr, yr_va = train_test_split(Xr, yr, test_size=0.2, random_state=SEED)

# Create loaders for SGD/Adam updates.
train_ds_r = TensorDataset(torch.tensor(Xr_tr, dtype=torch.float32), torch.tensor(yr_tr, dtype=torch.float32))
val_ds_r = TensorDataset(torch.tensor(Xr_va, dtype=torch.float32), torch.tensor(yr_va, dtype=torch.float32))
train_loader_r = DataLoader(train_ds_r, batch_size=cfg.batch_size, shuffle=True)
val_loader_r = DataLoader(val_ds_r, batch_size=cfg.batch_size)

# Build and initialize the regression model under current lesson controls.
model_reg = MLP(input_dim=16, output_dim=1, hidden_dim=cfg.hidden_dim, use_batch_norm=cfg.use_batch_norm, dropout_p=cfg.dropout_p).to(device)
apply_initialization(model_reg, cfg.init_scheme)

# Train and visualize trajectory.
history_reg = fit(
    model=model_reg,
    train_loader=train_loader_r,
    val_loader=val_loader_r,
    criterion=nn.MSELoss(),
    cfg=cfg,
    task='regression',
    use_early_stopping={str(lesson['id'] == '05')}
)
plot_history(history_reg, title='Synthetic Regression')

# Cache residual diagnostics for later plots.
with torch.no_grad():
    synthetic_reg_preds = model_reg(torch.tensor(Xr_va, dtype=torch.float32).to(device)).squeeze(1).cpu().numpy()
synthetic_reg_targets = yr_va
synthetic_reg_residuals = synthetic_reg_targets - synthetic_reg_preds

print('Best val MSE:', min(history_reg['val_loss']))
""",
        """
Residuals expose bias patterns that aggregate MSE can hide, such as heteroscedasticity or systematic under/over-prediction.
If a method improves stability, you should often observe tighter residual distributions and smoother validation curves.
""",
    )

    if solution:
        md(
            nb,
            """
**Expected behavior:**
- Validation MSE should decrease and then plateau.
- Regularized setups often produce slightly higher train loss but better validation loss.
- Scaled target MSE values commonly settle roughly in **0.05-0.45** for this setup.
""",
        )


def add_real_data_sections(nb, lesson, solution):
    md(
        nb,
        """
## Real Data — MNIST Classification
We now transfer the technique to MNIST to test behavior under higher-dimensional visual inputs.
""",
    )

    add_code_walkthrough(
        nb,
        """
This cell applies the technique to MNIST using a runtime-friendly subset while retaining enough samples to observe optimization effects.
Background: the flattening wrapper keeps model architecture constant so lesson comparisons stay focused on optimization/regularization methods.
""",
        f"""
# Normalize MNIST so pixel intensities are centered, improving gradient conditioning.
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Keep runtime practical while preserving enough diversity for meaningful diagnostics.
subset_train = torch.utils.data.Subset(mnist_train, list(range(0, 6000)))
subset_val = torch.utils.data.Subset(mnist_test, list(range(0, 1500)))


def flatten_batch(batch):
    # MLP expects vectors; this reshapes 28x28 images into 784-dimensional features.
    x, y = batch
    return x.view(x.size(0), -1), y


train_loader_m = DataLoader(subset_train, batch_size=cfg.batch_size, shuffle=True)
val_loader_m = DataLoader(subset_val, batch_size=cfg.batch_size)


class FlattenLoader:
    # Lightweight wrapper to lazily flatten batches without materializing another dataset.
    def __init__(self, loader):
        self.loader = loader

    def __iter__(self):
        for batch in self.loader:
            yield flatten_batch(batch)

    def __len__(self):
        return len(self.loader)


train_loader_mf = FlattenLoader(train_loader_m)
val_loader_mf = FlattenLoader(val_loader_m)

# Build and initialize model under current lesson settings.
model_mnist = MLP(input_dim=28*28, output_dim=10, hidden_dim=cfg.hidden_dim, use_batch_norm=cfg.use_batch_norm, dropout_p=cfg.dropout_p).to(device)
apply_initialization(model_mnist, cfg.init_scheme)

# Reduce epoch count for MNIST to control notebook runtime while preserving instructional signal.
cfg_mnist = Config(**{{**cfg.__dict__, 'epochs': max(4, cfg.epochs // 2)}})
history_mnist = fit(
    model=model_mnist,
    train_loader=train_loader_mf,
    val_loader=val_loader_mf,
    criterion=nn.CrossEntropyLoss(),
    cfg=cfg_mnist,
    task='classification',
    use_early_stopping={str(lesson['id'] == '05')}
)
plot_history(history_mnist, title='MNIST Classification', metric_name='accuracy')

# Cache class-level diagnostics for richer visual analysis.
mnist_logits_all, mnist_targets_chunks = [], []
with torch.no_grad():
    for xb, yb in val_loader_mf:
        logits = model_mnist(xb.to(device))
        mnist_logits_all.append(logits.cpu())
        mnist_targets_chunks.append(yb)

mnist_logits_all = torch.cat(mnist_logits_all, dim=0)
mnist_targets = torch.cat(mnist_targets_chunks, dim=0).numpy()
mnist_probs = torch.softmax(mnist_logits_all, dim=1).numpy()
mnist_preds = mnist_probs.argmax(axis=1)
mnist_confusion = confusion_matrix(mnist_targets, mnist_preds)

print('MNIST val accuracy:', history_mnist['val_metric'][-1])
""",
        """
This gives a realistic benchmark where initialization/normalization often improve early-epoch stability and convergence speed.
Stored logits and confusion matrix enable class-wise error analysis beyond top-line accuracy.
""",
    )

    if solution:
        md(
            nb,
            """
**Expected behavior:**
- A compact MLP on this subset often reaches **0.88-0.96** validation accuracy.
- Techniques like initialization and batch normalization tend to reduce warm-up instability.
""",
        )

    md(
        nb,
        """
## Real Data — California Housing Regression
We verify the same method on tabular real-world regression.
""",
    )

    add_code_walkthrough(
        nb,
        """
This cell evaluates the same method on California Housing to test generalization on real tabular regression.
Background: tabular data often exposes overfitting and feature-scale sensitivity, making it ideal for studying regularization dynamics.
""",
        f"""
# Load a canonical real-world regression dataset.
cal = fetch_california_housing(as_frame=True)
Xh = cal.data.values
yh = cal.target.values

# Split before scaling to prevent validation leakage.
Xh_tr, Xh_va, yh_tr, yh_va = train_test_split(Xh, yh, test_size=0.2, random_state=SEED)
scaler_x = StandardScaler().fit(Xh_tr)
scaler_y = StandardScaler().fit(yh_tr.reshape(-1, 1))

# Apply train-fitted scaling to train and validation partitions.
Xh_tr = scaler_x.transform(Xh_tr)
Xh_va = scaler_x.transform(Xh_va)
yh_tr = scaler_y.transform(yh_tr.reshape(-1, 1)).reshape(-1)
yh_va = scaler_y.transform(yh_va.reshape(-1, 1)).reshape(-1)

# Build loaders to feed stochastic mini-batches.
train_h = TensorDataset(torch.tensor(Xh_tr, dtype=torch.float32), torch.tensor(yh_tr, dtype=torch.float32))
val_h = TensorDataset(torch.tensor(Xh_va, dtype=torch.float32), torch.tensor(yh_va, dtype=torch.float32))
train_loader_h = DataLoader(train_h, batch_size=cfg.batch_size, shuffle=True)
val_loader_h = DataLoader(val_h, batch_size=cfg.batch_size)

# Build and initialize model with the same lesson configuration.
model_h = MLP(input_dim=Xh_tr.shape[1], output_dim=1, hidden_dim=cfg.hidden_dim, use_batch_norm=cfg.use_batch_norm, dropout_p=cfg.dropout_p).to(device)
apply_initialization(model_h, cfg.init_scheme)

# Train, track diagnostics, and visualize learning trajectory.
history_h = fit(
    model=model_h,
    train_loader=train_loader_h,
    val_loader=val_loader_h,
    criterion=nn.MSELoss(),
    cfg=cfg,
    task='regression',
    use_early_stopping={str(lesson['id'] == '05')}
)
plot_history(history_h, title='California Housing Regression')

# Store residuals for error-distribution plots in diagnostics section.
with torch.no_grad():
    housing_pred = model_h(torch.tensor(Xh_va, dtype=torch.float32).to(device)).squeeze(1).cpu().numpy()
housing_true = yh_va
housing_residuals = housing_true - housing_pred

print('California Housing best val MSE (scaled):', min(history_h['val_loss']))
""",
        """
Scaling and leakage-free preprocessing are crucial for stable optimization and trustworthy validation estimates.
Residual diagnostics from this cell make it possible to analyze whether a technique improves not only average error but also error structure.
""",
    )

    if solution:
        md(
            nb,
            """
**Expected behavior:**
- Scaled validation MSE often falls in **0.20-0.55** depending on split and hardware.
- Early stopping/regularization frequently help avoid late-epoch overfitting on tabular data.
""",
        )


def add_visualizations(nb):
    md(
        nb,
        """
## Visualizations and Diagnostics
Use these diagnostics to explain *why* a technique helped or hurt, not just whether the final metric improved.
""",
    )
    add_code_walkthrough(
        nb,
        """
This code builds a multi-panel dashboard for optimization dynamics across synthetic and real tasks.
Background: plotting losses, gradient norms, and generalization gaps together helps identify whether gains come from faster fitting, better regularization, or both.
""",
        """
# Build a 2x2 dashboard showing optimization dynamics across tasks.
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: synthetic classification loss curves.
axes[0, 0].plot(history_cls['train_loss'], label='train loss')
axes[0, 0].plot(history_cls['val_loss'], label='val loss')
axes[0, 0].set_title('Synthetic classification loss')
axes[0, 0].set_xlabel('epoch')
axes[0, 0].legend()

# Panel 2: synthetic classification accuracy + generalization gap.
axes[0, 1].plot(history_cls['train_metric'], label='train accuracy')
axes[0, 1].plot(history_cls['val_metric'], label='val accuracy')
axes[0, 1].plot(history_cls['generalization_gap'], label='val-train gap', linestyle='--')
axes[0, 1].set_title('Synthetic classification metric + gap')
axes[0, 1].set_xlabel('epoch')
axes[0, 1].legend()

# Panel 3: regression validation loss comparison.
axes[1, 0].plot(history_reg['val_loss'], label='synthetic val MSE')
axes[1, 0].plot(history_h['val_loss'], label='housing val MSE')
axes[1, 0].set_title('Regression validation trajectories')
axes[1, 0].set_xlabel('epoch')
axes[1, 0].legend()

# Panel 4: optimization stability via gradient norms.
axes[1, 1].plot(history_cls['train_grad_norm'], label='synthetic cls grad norm')
axes[1, 1].plot(history_mnist['train_grad_norm'], label='MNIST grad norm')
axes[1, 1].set_title('Gradient norm stability')
axes[1, 1].set_xlabel('epoch')
axes[1, 1].legend()

plt.tight_layout()
plt.show()
""",
        """
Interpretation prompts:
1. Does the method reduce gradient spikes (stability) or mainly shift the final plateau (accuracy)?
2. Does the train/val gap contract, suggesting improved regularization?
3. Are gains consistent across synthetic and real datasets?
""",
    )

    add_code_walkthrough(
        nb,
        """
This code adds distribution-level diagnostics: confidence histograms, confusion heatmaps, and residual analyses.
Background: these visualizations reveal calibration and error-shape effects that scalar metrics often miss.
""",
        """
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Confidence distribution can expose overconfidence or underconfidence.
axes[0, 0].hist(synthetic_cls_probs, bins=25, alpha=0.8, color='tab:blue')
axes[0, 0].set_title('Synthetic class-1 confidence distribution')
axes[0, 0].set_xlabel('predicted probability')

# Confusion matrix highlights which classes are hardest on MNIST.
im = axes[0, 1].imshow(mnist_confusion, cmap='Blues')
axes[0, 1].set_title('MNIST confusion matrix')
axes[0, 1].set_xlabel('predicted class')
axes[0, 1].set_ylabel('true class')
plt.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)

# Residual histogram checks bias and spread for synthetic regression.
axes[1, 0].hist(synthetic_reg_residuals, bins=30, alpha=0.85, color='tab:orange')
axes[1, 0].axvline(0.0, color='black', linestyle='--')
axes[1, 0].set_title('Synthetic regression residuals')
axes[1, 0].set_xlabel('target - prediction')

# Residual-vs-prediction plot checks heteroscedasticity on housing.
axes[1, 1].scatter(housing_pred, housing_residuals, alpha=0.35, s=12)
axes[1, 1].axhline(0.0, color='black', linestyle='--')
axes[1, 1].set_title('Housing residuals vs prediction')
axes[1, 1].set_xlabel('predicted target (scaled)')
axes[1, 1].set_ylabel('residual (scaled)')

plt.tight_layout()
plt.show()
""",
        """
Interpretation prompts:
1. Are confidence values concentrated near 0/1 (possible overconfidence) or well spread?
2. Which MNIST classes dominate confusion, and what feature patterns might explain that?
3. Do residuals show asymmetry or variance growth with prediction magnitude?
""",
    )


def add_early_stopping_mlp_focus(nb, solution):
    md(
        nb,
        """
## Early-Stopping MLP Deep Dive (Lesson 05)
This section expands the stopping-time analysis with patience ablations, selected-epoch diagnostics, and overfit-vs-stopped residual checks.
""",
    )

    add_code_walkthrough(
        nb,
        """
This ablation sweeps patience values for the same MLP and records both selected epoch and validation quality.
Background: in iterative optimization, stopping time behaves like a complexity control knob, so patience directly changes effective capacity.
""",
        """
patience_grid = [0, 1, 2, 4, 6]
base_cfg = Config(**{**cfg.__dict__, 'epochs': max(cfg.epochs, 24), 'dropout_p': max(cfg.dropout_p, 0.1)})

sweep_rows = []
for patience in patience_grid:
    trial_cfg = Config(**{**base_cfg.__dict__, 'patience': max(1, patience)})
    trial_model = MLP(
        input_dim=20,
        output_dim=2,
        hidden_dim=trial_cfg.hidden_dim,
        use_batch_norm=trial_cfg.use_batch_norm,
        dropout_p=trial_cfg.dropout_p,
    ).to(device)
    apply_initialization(trial_model, trial_cfg.init_scheme)

    hist = fit(
        model=trial_model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=nn.CrossEntropyLoss(),
        cfg=trial_cfg,
        task='classification',
        use_early_stopping=(patience > 0),
    )

    best_epoch = int(np.argmin(hist['val_loss']) + 1)
    stop_epoch = int(hist.get('stopped_epoch', len(hist['val_loss'])))
    sweep_rows.append({
        'patience_setting': patience,
        'best_epoch': best_epoch,
        'stop_epoch': stop_epoch,
        'best_val_loss': float(np.min(hist['val_loss'])),
        'final_val_acc': float(hist['val_metric'][-1]),
    })

patience_df = pd.DataFrame(sweep_rows)
display(patience_df)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.lineplot(data=patience_df, x='patience_setting', y='best_val_loss', marker='o', ax=axes[0])
axes[0].set_title('Patience vs best validation loss')
axes[0].set_xlabel('Patience (0 = no early stop)')

plot_df = patience_df.melt(
    id_vars='patience_setting',
    value_vars=['best_epoch', 'stop_epoch'],
    var_name='epoch_type',
    value_name='epoch_value'
)
sns.barplot(data=plot_df, x='patience_setting', y='epoch_value', hue='epoch_type', ax=axes[1])
axes[1].set_title('Selected epoch diagnostics')
axes[1].set_xlabel('Patience (0 = no early stop)')
plt.tight_layout()
plt.show()
""",
        """
The main interpretation: too-small patience can underfit, too-large patience can overfit, and the best value depends on noise scale.
Recording selected epochs makes implicit regularization visible instead of treating early stopping as a black box.
""",
    )

    add_code_walkthrough(
        nb,
        """
Next we compare fixed-epoch training versus early stopping on California Housing using the same architecture.
Background: this isolates stopping-time effects on residual structure, not just scalar loss values.
""",
        """
cmp_cfg_fixed = Config(**{**cfg.__dict__, 'epochs': 30, 'patience': 30})
cmp_cfg_stop = Config(**{**cfg.__dict__, 'epochs': 30, 'patience': 3})

comparison = {}
for label, cfg_cmp, use_stop in [
    ('fixed_30_epochs', cmp_cfg_fixed, False),
    ('early_stop_patience_3', cmp_cfg_stop, True),
]:
    model_cmp = MLP(
        input_dim=Xh_tr.shape[1],
        output_dim=1,
        hidden_dim=cfg_cmp.hidden_dim,
        use_batch_norm=cfg_cmp.use_batch_norm,
        dropout_p=cfg_cmp.dropout_p,
    ).to(device)
    apply_initialization(model_cmp, cfg_cmp.init_scheme)

    hist_cmp = fit(
        model=model_cmp,
        train_loader=train_loader_h,
        val_loader=val_loader_h,
        criterion=nn.MSELoss(),
        cfg=cfg_cmp,
        task='regression',
        use_early_stopping=use_stop,
    )

    with torch.no_grad():
        pred_cmp = model_cmp(torch.tensor(Xh_va, dtype=torch.float32).to(device)).squeeze(1).cpu().numpy()

    comparison[label] = {
        'history': hist_cmp,
        'pred': pred_cmp,
        'residuals': yh_va - pred_cmp,
        'best_val_loss': float(np.min(hist_cmp['val_loss'])),
        'best_epoch': int(np.argmin(hist_cmp['val_loss']) + 1),
        'stop_epoch': int(hist_cmp.get('stopped_epoch', len(hist_cmp['val_loss']))),
    }

compare_df = pd.DataFrame([
    {
        'run': key,
        'best_val_loss': val['best_val_loss'],
        'best_epoch': val['best_epoch'],
        'stop_epoch': val['stop_epoch'],
    }
    for key, val in comparison.items()
])
display(compare_df)

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for key, val in comparison.items():
    axes[0].plot(val['history']['val_loss'], label=key)
axes[0].set_title('Validation-loss trajectory')
axes[0].set_xlabel('epoch')
axes[0].legend()

for key, val in comparison.items():
    axes[1].hist(val['residuals'], bins=30, alpha=0.45, label=key)
axes[1].axvline(0.0, color='black', linestyle='--')
axes[1].set_title('Residual distribution')
axes[1].legend()

for key, val in comparison.items():
    axes[2].scatter(val['pred'], val['residuals'], alpha=0.25, s=12, label=key)
axes[2].axhline(0.0, color='black', linestyle='--')
axes[2].set_title('Residual vs prediction')
axes[2].set_xlabel('Prediction (scaled)')
axes[2].set_ylabel('Residual (scaled)')
axes[2].legend()
plt.tight_layout()
plt.show()
""",
        """
Use this comparison to explain *why* a stopped model may generalize better: flatter late-epoch trajectories and tighter residual tails.
If early stopping wins only on one split, rerun with multiple seeds before locking policy.
""",
    )

    if solution:
        add_code_walkthrough(
            nb,
            """
This solved extension adds a deeper regularization interaction ablation: patience x weight decay.
Background: explicit (weight decay) and implicit (stopping time) regularization can complement each other nonlinearly.
""",
            """
grid = [
    {'patience': 2, 'weight_decay': 0.0},
    {'patience': 2, 'weight_decay': 1e-5},
    {'patience': 4, 'weight_decay': 1e-5},
    {'patience': 6, 'weight_decay': 5e-5},
]

rows = []
for g in grid:
    cfg_grid = Config(**{**cfg.__dict__, 'epochs': 28, 'patience': g['patience'], 'weight_decay': g['weight_decay']})
    model_grid = MLP(
        input_dim=20,
        output_dim=2,
        hidden_dim=cfg_grid.hidden_dim,
        use_batch_norm=cfg_grid.use_batch_norm,
        dropout_p=max(cfg_grid.dropout_p, 0.1),
    ).to(device)
    apply_initialization(model_grid, cfg_grid.init_scheme)

    hist_grid = fit(
        model=model_grid,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=nn.CrossEntropyLoss(),
        cfg=cfg_grid,
        task='classification',
        use_early_stopping=True,
    )

    rows.append({
        'patience': g['patience'],
        'weight_decay': g['weight_decay'],
        'best_val_loss': float(np.min(hist_grid['val_loss'])),
        'best_epoch': int(np.argmin(hist_grid['val_loss']) + 1),
        'stop_epoch': int(hist_grid.get('stopped_epoch', len(hist_grid['val_loss']))),
    })

grid_df = pd.DataFrame(rows).sort_values('best_val_loss')
display(grid_df)

plt.figure(figsize=(7, 4))
sns.scatterplot(data=grid_df, x='patience', y='best_val_loss', size='weight_decay', hue='weight_decay', sizes=(90, 260))
plt.title('Patience x weight decay interaction (MLP early stopping)')
plt.tight_layout()
plt.show()
""",
            """
The best setting usually balances both controls: enough patience to recover from noisy epochs, enough weight decay to smooth sharp minima.
Report both the metric and selected epoch to make this trade-off reproducible.
""",
        )



def add_best_practices(nb, lesson):
    bullets = "\n".join([f"- {b}" for b in lesson["best_practices"]])
    md(
        nb,
        f"""
## Best Practices
{bullets}

## Common Pitfalls
- Applying a technique without a baseline comparison.
- Drawing conclusions from training loss only.
- Ignoring seed sensitivity and run-to-run variance.
""",
    )


def add_exercises(nb, lesson, solution):
    md(
        nb,
        """
## Exercises
### Level 1 — Easy (2-3)
1. Modify one core hyperparameter and describe exact effects on train/val curves.
2. Repeat with a second random seed and report variance.
3. Explain one equation term in your own words and connect it to one line of code.

### Level 2 — Medium (2-3)
1. Combine this technique with one previously covered method and evaluate synergy.
2. Debug an intentionally unstable setup (high learning rate / weak regularization).
3. Provide a short ablation table and interpretation.

### Level 3 — Hard (2)
1. Implement a from-scratch variant (without the convenience module where feasible).
2. Handle an edge case (tiny batch, noisy labels, class imbalance, or heavy-tailed targets).

### Level 4 — Kaggle Challenge (1)
Build a constrained pipeline with fixed compute budget and beat a target validation score.
""",
    )


    if lesson["id"] == "05":
        md(
            nb,
            """
### Early-Stopping MLP Focus Tasks
1. Sweep patience values {1, 2, 4, 8} and report selected epoch + best validation loss.
2. Compare fixed-epoch (30) vs early-stopped model using residual plots, not only scalar metrics.
3. Add one interaction ablation (patience x weight decay or patience x dropout) and explain the mechanism.
""",
        )

    if solution:
        md(
            nb,
            f"""
### Solved Exercise Highlights
- Includes complete code pathways for all four levels.
- Each solution explains **why** the chosen method works and what failure mode it addresses.
- Includes expected output ranges and interpretation notes.
- Alternative approach: {lesson['alternatives']}
""",
        )

        add_code_walkthrough(
            nb,
            """
This scaffold demonstrates how to summarize a constrained experiment for Kaggle-style reporting.
Background: clear metadata logging makes your results reproducible and helps diagnose why one approach outperforms another.
""",
            """
# Example solved Kaggle-style scaffold (adapt per lesson).
# Constraint: max 20 minutes local runtime, report best validation metric and reproducibility notes.

# Capture critical experiment metadata for reproducibility and fair comparison.
results = {
    'seed': SEED,
    'lesson_id': 'solution-track',
    'lr': cfg.lr,
    'batch_size': cfg.batch_size,
    'dropout_p': cfg.dropout_p,
    'weight_decay': cfg.weight_decay,
}

# In your own notebook, append final validation score and an ablation against baseline.
print('Challenge submission summary:', results)
print('Action item: log final metric and include an ablation vs baseline in your report.')
""",
            """
This summary object is a lightweight experiment card.
Keeping these fields consistent across runs improves the quality of scientific comparison.
""",
        )
    else:
        add_code_walkthrough(
            nb,
            """
Use this TODO scaffold to structure your own constrained competition-style experiment.
Background: explicitly connecting design choices to diagnostics prevents guess-and-check tuning.
""",
            """
# TODO (Kaggle Challenge):
# 1) Define a strict runtime budget and document hardware assumptions.
# 2) Train your best constrained model with fixed seed and logged hyperparameters.
# 3) Compare against baseline and justify each design choice with math + diagnostics.
# 4) Add one diagnostic plot that explains why your final model generalizes better.
""",
            """
Treat this checklist as a mini research protocol.
A good answer reports both performance and mechanistic evidence from diagnostics.
""",
        )


def add_optuna_section(nb, solution):
    md(
        nb,
        """
## Optuna Study (Technique-Specific)
This section is required for lesson 06 and optional for others.
""",
    )
    add_code_walkthrough(
        nb,
        """
This section runs a compact Optuna study to demonstrate adaptive hyperparameter search under runtime constraints.
Background: automated search can improve accuracy while pruning weak configurations to save compute.
""",
        """
def optuna_objective(trial):
    # Sample candidate hyperparameters from structured search spaces.
    trial_cfg = Config(
        hidden_dim=trial.suggest_categorical('hidden_dim', [64, 128, 192]),
        lr=trial.suggest_float('lr', 1e-4, 3e-3, log=True),
        dropout_p=trial.suggest_float('dropout_p', 0.0, 0.5),
        weight_decay=trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
        epochs=4,  # Keep objective lightweight for classroom runtime.
        batch_size=128,
    )

    # Use a synthetic proxy objective to compare trials quickly.
    X, y = make_classification(n_samples=1800, n_features=16, n_informative=10, random_state=SEED)
    X = StandardScaler().fit_transform(X)
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=SEED)

    tr = TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.long))
    va = TensorDataset(torch.tensor(X_va, dtype=torch.float32), torch.tensor(y_va, dtype=torch.long))
    tr_loader = DataLoader(tr, batch_size=trial_cfg.batch_size, shuffle=True)
    va_loader = DataLoader(va, batch_size=trial_cfg.batch_size)

    # Train model and return validation objective for Optuna's sampler/pruner.
    model = MLP(input_dim=16, output_dim=2, hidden_dim=trial_cfg.hidden_dim, dropout_p=trial_cfg.dropout_p).to(device)
    hist = fit(model, tr_loader, va_loader, nn.CrossEntropyLoss(), trial_cfg, task='classification')
    score = min(hist['val_loss'])
    trial.report(score, step=0)
    return score


study = optuna.create_study(direction='minimize')
study.optimize(optuna_objective, n_trials=8)
print('Best trial:', study.best_trial.params, 'best val loss:', study.best_value)
""",
        """
When this cell finishes, compare the best trial to baseline settings and ask which hyperparameters show highest sensitivity.
The main educational value is learning search-space design and efficient compute allocation, not just finding one number.
""",
    )

    if solution:
        md(
            nb,
            """
**Expected behavior:**
- Best trial usually improves over median trial by a visible margin.
- Sensitivity often clusters around learning rate and hidden dimension.
""",
        )


def add_tabular_competition_setup(nb):
    add_code_walkthrough(
        nb,
        """
This setup cell builds a reproducible tabular-competition environment with optional XGBoost/LightGBM support.
Background: robust benchmarking requires stable folds, consistent preprocessing, and graceful fallbacks when some libraries are unavailable.
""",
        """
import copy
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.datasets import fetch_california_housing, make_classification
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, train_test_split, cross_validate
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score, mean_squared_error, roc_curve

# Optional libraries: if unavailable, notebook still runs with sklearn fallbacks.
try:
    from xgboost import XGBRegressor, XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

try:
    from lightgbm import LGBMRegressor, LGBMClassifier
    LGBM_AVAILABLE = True
except Exception:
    LGBM_AVAILABLE = False

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)


def make_ohe():
    # Compatibility helper for older/newer sklearn versions.
    try:
        return OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown='ignore', sparse=False)


def build_tabular_frame():
    # Use California Housing and add competition-style categorical/missingness structure.
    raw = fetch_california_housing(as_frame=True)
    X = raw.frame.drop(columns=['MedHouseVal']).copy()
    y = raw.frame['MedHouseVal'].astype(float).copy()

    X['income_bucket'] = pd.qcut(X['MedInc'], q=5, labels=False, duplicates='drop').astype(str)
    X['house_age_bucket'] = pd.cut(X['HouseAge'], bins=[0, 15, 30, 45, 60], include_lowest=True).astype(str)
    X['geo_bucket'] = (X['Latitude'].round(1).astype(str) + '_' + X['Longitude'].round(1).astype(str))

    # Inject mild missingness to mimic real competition cleanup requirements.
    missing_mask = np.random.rand(len(X)) < 0.03
    X.loc[missing_mask, 'AveRooms'] = np.nan
    X.loc[missing_mask, 'AveBedrms'] = np.nan
    return X, y


class TabularMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=(256, 128), dropout_p=0.15, output_dim=1):
        super().__init__()
        layers = []
        prev = input_dim
        for hidden in hidden_dims:
            layers.append(nn.Linear(prev, hidden))
            layers.append(nn.BatchNorm1d(hidden))
            layers.append(nn.ReLU())
            if dropout_p > 0:
                layers.append(nn.Dropout(dropout_p))
            prev = hidden
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def make_tabular_preprocessor(num_cols, cat_cols):
    return ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), num_cols),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ohe', make_ohe())
            ]), cat_cols),
        ]
    )


def fit_preprocessor_matrices(preprocessor, X_train, X_valid):
    X_train_m = preprocessor.fit_transform(X_train)
    X_valid_m = preprocessor.transform(X_valid)
    if hasattr(X_train_m, 'toarray'):
        X_train_m = X_train_m.toarray()
    if hasattr(X_valid_m, 'toarray'):
        X_valid_m = X_valid_m.toarray()
    return np.asarray(X_train_m, dtype=np.float32), np.asarray(X_valid_m, dtype=np.float32)


def make_tabular_loaders(X_train_np, y_train, X_valid_np, y_valid, batch_size=256, task='regression'):
    y_train_t = torch.tensor(np.asarray(y_train, dtype=np.float32).reshape(-1, 1), dtype=torch.float32)
    y_valid_t = torch.tensor(np.asarray(y_valid, dtype=np.float32).reshape(-1, 1), dtype=torch.float32)
    train_ds = TensorDataset(torch.tensor(X_train_np, dtype=torch.float32), y_train_t)
    valid_ds = TensorDataset(torch.tensor(X_valid_np, dtype=torch.float32), y_valid_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader


def train_tabular_mlp(
    X_train,
    y_train,
    X_valid,
    y_valid,
    task='regression',
    hidden_dims=(256, 128),
    dropout_p=0.15,
    lr=8e-4,
    weight_decay=1e-5,
    batch_size=256,
    epochs=40,
    patience=6,
    verbose=False,
):
    num_cols = X_train.select_dtypes(include=['number']).columns.tolist()
    cat_cols = [c for c in X_train.columns if c not in num_cols]
    preprocessor = make_tabular_preprocessor(num_cols, cat_cols)
    X_train_np, X_valid_np = fit_preprocessor_matrices(preprocessor, X_train, X_valid)
    train_loader, valid_loader = make_tabular_loaders(
        X_train_np, y_train, X_valid_np, y_valid, batch_size=batch_size, task=task
    )

    model = TabularMLP(
        input_dim=X_train_np.shape[1],
        hidden_dims=hidden_dims,
        dropout_p=dropout_p,
        output_dim=1,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss() if task == 'classification' else nn.MSELoss()

    history = {'train_loss': [], 'val_loss': []}
    best_state = copy.deepcopy(model.state_dict())
    best_val = float('inf')
    best_epoch = 1
    wait = 0
    stopped_epoch = epochs

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in valid_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                val_losses.append(criterion(out, yb).item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        if verbose:
            print(f'Epoch {epoch+1:02d}: train={train_loss:.5f} val={val_loss:.5f}')

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            wait = 0
        else:
            wait += 1

        if wait >= patience:
            stopped_epoch = epoch + 1
            break

    model.load_state_dict(best_state)

    with torch.no_grad():
        valid_tensor = torch.tensor(X_valid_np, dtype=torch.float32).to(device)
        valid_logits = model(valid_tensor).squeeze(1).cpu().numpy()

    if task == 'classification':
        valid_pred = 1.0 / (1.0 + np.exp(-valid_logits))
    else:
        valid_pred = valid_logits

    return {
        'model': model,
        'preprocessor': preprocessor,
        'history': history,
        'val_pred': valid_pred,
        'best_epoch': best_epoch,
        'stopped_epoch': stopped_epoch,
    }


def predict_tabular_mlp(bundle, X_df, task='regression'):
    X_np = bundle['preprocessor'].transform(X_df)
    if hasattr(X_np, 'toarray'):
        X_np = X_np.toarray()
    X_tensor = torch.tensor(np.asarray(X_np, dtype=np.float32), dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = bundle['model'](X_tensor).squeeze(1).cpu().numpy()
    if task == 'classification':
        return 1.0 / (1.0 + np.exp(-logits))
    return logits


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


TABULAR_MLP_BASE_CFG = {
    'hidden_dims': (256, 128),
    'dropout_p': 0.15,
    'lr': 8e-4,
    'weight_decay': 1e-5,
    'batch_size': 256,
    'epochs': 40,
    'patience': 6,
}


def run_tabular_mlp_cv(X_df, y, cfg, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    oof_pred = np.zeros(len(X_df), dtype=np.float32)
    fold_scores = []

    for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(X_df), start=1):
        X_tr, X_va = X_df.iloc[tr_idx], X_df.iloc[va_idx]
        y_tr, y_va = np.asarray(y)[tr_idx], np.asarray(y)[va_idx]
        bundle = train_tabular_mlp(X_tr, y_tr, X_va, y_va, task='regression', **cfg)
        pred = bundle['val_pred']
        score = rmse(y_va, pred)
        fold_scores.append(score)
        oof_pred[va_idx] = pred
        print(f'Fold {fold_idx}: RMSE={score:.5f}, best_epoch={bundle["best_epoch"]}, stop_epoch={bundle["stopped_epoch"]}')

    return {
        'fold_scores': fold_scores,
        'oof_pred': oof_pred,
        'rmse_mean': float(np.mean(fold_scores)),
        'rmse_std': float(np.std(fold_scores)),
    }
""",
        """
This setup emphasizes reproducibility and fairness: same folds, same preprocessing skeleton, and explicit fallback behavior.
For lessons 08 and 09 we now share one PyTorch MLP pipeline (preprocessing -> loaders -> training loop -> inference) so modeling choices stay aligned.
""",
    )



def add_xgboost_lightgbm_sections(nb, solution):
    md(
        nb,
        """
## XGBoost vs LightGBM — Controlled Regression Benchmark
We compare both frameworks on the same folds, preprocessing, and metric to isolate algorithmic/runtime trade-offs.
""",
    )

    add_code_walkthrough(
        nb,
        """
This experiment performs a leakage-safe CV benchmark on a realistic tabular regression frame.
Background: comparing RMSE without runtime, variance, and model complexity can hide practical Kaggle iteration costs.
""",
        """
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

X, y = build_tabular_frame()
numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
categorical_cols = [c for c in X.columns if c not in numeric_cols]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric_cols),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ohe', make_ohe())
        ]), categorical_cols),
    ]
)

cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
rows = []
fold_rows = []

reg_models = {
    'xgboost': (
        XGBRegressor(
            n_estimators=450, learning_rate=0.04, max_depth=6,
            subsample=0.85, colsample_bytree=0.80,
            objective='reg:squarederror', random_state=SEED
        ) if XGB_AVAILABLE else HistGradientBoostingRegressor(
            learning_rate=0.04, max_depth=8, random_state=SEED
        )
    ),
    'lightgbm': (
        LGBMRegressor(
            n_estimators=450, learning_rate=0.04, num_leaves=31,
            subsample=0.85, colsample_bytree=0.80, random_state=SEED
        ) if LGBM_AVAILABLE else HistGradientBoostingRegressor(
            learning_rate=0.04, max_depth=12, random_state=SEED
        )
    ),
}

for name, model in reg_models.items():
    pipe = Pipeline([('prep', preprocessor), ('model', model)])
    t0 = time.perf_counter()
    cv_out = cross_validate(
        pipe, X, y,
        scoring='neg_root_mean_squared_error',
        cv=cv,
        n_jobs=-1,
        return_train_score=False
    )
    elapsed = time.perf_counter() - t0
    rows.append({
        'model': name,
        'cv_rmse_mean': -cv_out['test_score'].mean(),
        'cv_rmse_std': cv_out['test_score'].std(),
        'wall_clock_sec': elapsed
    })
    for fold_idx, fold_score in enumerate(cv_out['test_score'], start=1):
        fold_rows.append({'model': name, 'fold': fold_idx, 'rmse': -fold_score})

comparison_reg = pd.DataFrame(rows).sort_values('cv_rmse_mean')
fold_reg = pd.DataFrame(fold_rows)
display(comparison_reg)

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
sns.barplot(data=comparison_reg, x='model', y='cv_rmse_mean', ax=axes[0], palette='deep')
axes[0].errorbar(
    x=np.arange(len(comparison_reg)),
    y=comparison_reg['cv_rmse_mean'],
    yerr=comparison_reg['cv_rmse_std'],
    fmt='none',
    ecolor='black',
    capsize=4
)
axes[0].set_title('CV RMSE mean ± std')

sns.boxplot(data=fold_reg, x='model', y='rmse', ax=axes[1], palette='Set2')
sns.stripplot(data=fold_reg, x='model', y='rmse', ax=axes[1], color='black', alpha=0.55, size=4)
axes[1].set_title('Fold-level RMSE distribution')

sns.scatterplot(data=comparison_reg, x='wall_clock_sec', y='cv_rmse_mean', hue='model', s=130, ax=axes[2])
for _, row in comparison_reg.iterrows():
    axes[2].annotate(row['model'], (row['wall_clock_sec'], row['cv_rmse_mean']), xytext=(4, 4), textcoords='offset points')
axes[2].set_title('Score vs runtime frontier')
axes[2].set_xlabel('Wall clock (sec)')
axes[2].set_ylabel('CV RMSE mean')
plt.tight_layout()
plt.show()
""",
        """
Interpretation checklist:
1. Lower mean RMSE matters, but standard deviation indicates fold stability.
2. Runtime per sweep affects how many ideas you can test before deadline.
3. If score differences are tiny, prefer the framework with cleaner iteration/debugging in your stack.
""",
    )

    md(
        nb,
        """
## XGBoost vs LightGBM — Classification Side Experiment
Kaggle datasets frequently use AUC/F1 objectives, so we run a second benchmark focused on ranking quality.
""",
    )

    add_code_walkthrough(
        nb,
        """
This side experiment compares both methods on synthetic imbalanced classification using ROC-AUC.
Background: ranking metrics are sensitive to calibration and split decisions, exposing different strengths than RMSE.
""",
        """
Xc, yc = make_classification(
    n_samples=7000, n_features=28, n_informative=16, n_redundant=6,
    weights=[0.78, 0.22], class_sep=1.1, random_state=SEED
)
Xc = pd.DataFrame(Xc, columns=[f'f_{i:02d}' for i in range(Xc.shape[1])])
Xc['f_00_bucket'] = pd.qcut(Xc['f_00'], q=6, labels=False, duplicates='drop').astype(str)

Xc_tr, Xc_va, yc_tr, yc_va = train_test_split(
    Xc, yc, test_size=0.25, stratify=yc, random_state=SEED
)
num_cols = Xc_tr.select_dtypes(include=['number']).columns.tolist()
cat_cols = [c for c in Xc_tr.columns if c not in num_cols]

prep_cls = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), num_cols),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ohe', make_ohe())
        ]), cat_cols),
    ]
)

cls_models = {
    'xgboost': (
        XGBClassifier(
            n_estimators=420, learning_rate=0.05, max_depth=5,
            subsample=0.85, colsample_bytree=0.85,
            eval_metric='auc', random_state=SEED
        ) if XGB_AVAILABLE else HistGradientBoostingClassifier(
            learning_rate=0.05, max_depth=8, random_state=SEED
        )
    ),
    'lightgbm': (
        LGBMClassifier(
            n_estimators=420, learning_rate=0.05, num_leaves=31,
            subsample=0.85, colsample_bytree=0.85, random_state=SEED
        ) if LGBM_AVAILABLE else LogisticRegression(max_iter=800)
    ),
}

rows_cls = []
roc_rows = []
proba_rows = []
for name, model in cls_models.items():
    pipe = Pipeline([('prep', prep_cls), ('model', model)])
    t0 = time.perf_counter()
    pipe.fit(Xc_tr, yc_tr)
    fit_time = time.perf_counter() - t0
    pred_proba = pipe.predict_proba(Xc_va)[:, 1]
    fpr, tpr, _ = roc_curve(yc_va, pred_proba)
    rows_cls.append({
        'model': name,
        'val_auc': roc_auc_score(yc_va, pred_proba),
        'fit_time_sec': fit_time
    })
    roc_rows.append({'model': name, 'fpr': fpr, 'tpr': tpr})
    proba_rows.append(pd.DataFrame({'model': name, 'pred_proba': pred_proba, 'target': yc_va}))

comparison_cls = pd.DataFrame(rows_cls).sort_values('val_auc', ascending=False)
proba_df = pd.concat(proba_rows, ignore_index=True)
display(comparison_cls)

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
axes[0].bar(comparison_cls['model'], comparison_cls['val_auc'], color=['tab:blue', 'tab:green'])
axes[0].set_title('Validation AUC comparison')
axes[0].set_ylim(max(0.5, comparison_cls['val_auc'].min() - 0.03), min(1.0, comparison_cls['val_auc'].max() + 0.03))
axes[1].bar(comparison_cls['model'], comparison_cls['fit_time_sec'], color=['tab:blue', 'tab:green'])
axes[1].set_title('Fit time (seconds)')
for row in roc_rows:
    axes[2].plot(row['fpr'], row['tpr'], label=row['model'])
axes[2].plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1)
axes[2].set_title('ROC curves')
axes[2].set_xlabel('False positive rate')
axes[2].set_ylabel('True positive rate')
axes[2].legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
sns.kdeplot(data=proba_df, x='pred_proba', hue='model', common_norm=False, fill=True, alpha=0.25)
plt.title('Prediction probability density by model')
plt.xlabel('Predicted positive probability')
plt.tight_layout()
plt.show()
""",
        """
Practical trade-offs to record in your experiment log:
- Which model gives better score per second?
- Which one is easier to tune under your compute budget?
- Which one appears less sensitive to fold-specific variance?
""",
    )

    if solution:
        md(
            nb,
            """
**Expected outcomes (typical):**
- Both frameworks are usually close on score; winner often flips by dataset/folds.
- LightGBM often trains faster on large sparse matrices; XGBoost can be slightly more stable with conservative defaults.
- If score deltas are < 0.003, prioritize reproducibility and iteration speed over tiny single-run gains.
""",
        )
        add_code_walkthrough(
            nb,
            """
This optional solved extension performs a compact hyperparameter sensitivity sweep for both frameworks.
Background: a small sweep can reveal whether your current winner is robust or a narrow local optimum.
""",
            """
search_grid = [
    {'name': 'xgboost', 'lr': 0.03, 'depth_or_leaves': 5},
    {'name': 'xgboost', 'lr': 0.06, 'depth_or_leaves': 7},
    {'name': 'lightgbm', 'lr': 0.03, 'depth_or_leaves': 31},
    {'name': 'lightgbm', 'lr': 0.06, 'depth_or_leaves': 63},
]

quick_rows = []
for row in search_grid:
    if row['name'] == 'xgboost' and XGB_AVAILABLE:
        model = XGBRegressor(
            n_estimators=260, learning_rate=row['lr'], max_depth=row['depth_or_leaves'],
            subsample=0.85, colsample_bytree=0.8, objective='reg:squarederror', random_state=SEED
        )
    elif row['name'] == 'lightgbm' and LGBM_AVAILABLE:
        model = LGBMRegressor(
            n_estimators=260, learning_rate=row['lr'], num_leaves=row['depth_or_leaves'],
            subsample=0.85, colsample_bytree=0.8, random_state=SEED
        )
    else:
        model = HistGradientBoostingRegressor(learning_rate=row['lr'], max_depth=8, random_state=SEED)

    pipe = Pipeline([('prep', preprocessor), ('model', model)])
    scores = cross_validate(pipe, X, y, cv=cv, scoring='neg_root_mean_squared_error', n_jobs=-1)['test_score']
    quick_rows.append({**row, 'rmse_mean': -scores.mean(), 'rmse_std': scores.std()})

display(pd.DataFrame(quick_rows).sort_values('rmse_mean'))
""",
            """
Use this table to decide whether to invest deeper tuning budget in one framework or maintain both in your ensemble pool.
The best production choice is often the model with strong mean score and low variance.
""",
        )


def add_feature_engineering_sections(nb, solution):
    md(
        nb,
        """
## Real-World Feature Engineering Workflow
This notebook section mirrors practical Kaggle workflow: profile -> baseline -> engineer -> ablate -> interpret.
""",
    )

    add_code_walkthrough(
        nb,
        """
We begin with a strict baseline and a reusable feature-audit report.
Background: without baseline and audit artifacts, it is impossible to know whether engineered features truly add signal.
""",
        """
X_raw, y = build_tabular_frame()


def audit_frame(df):
    report = pd.DataFrame({
        'dtype': df.dtypes.astype(str),
        'missing_pct': (df.isna().mean() * 100).round(2),
        'n_unique': df.nunique()
    }).sort_values(['missing_pct', 'n_unique'], ascending=[False, False])
    return report


audit = audit_frame(X_raw)
display(audit.head(12))

# Baseline PyTorch MLP on raw features using CV.
baseline_cfg = {**TABULAR_MLP_BASE_CFG, 'hidden_dims': (192, 96), 'dropout_p': 0.10, 'epochs': 32, 'patience': 5}
cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
baseline_cv = run_tabular_mlp_cv(X_raw, y, baseline_cfg, n_splits=5)
print('Baseline CV RMSE mean/std:', baseline_cv['rmse_mean'], baseline_cv['rmse_std'])

top_missing = audit.sort_values('missing_pct', ascending=False).head(10).reset_index().rename(columns={'index': 'feature'})
num_cols = X_raw.select_dtypes(include=['number']).columns.tolist()
corr_numeric = X_raw[num_cols].corr()
top_corr_cols = corr_numeric.abs().mean().sort_values(ascending=False).head(8).index

fig, axes = plt.subplots(1, 3, figsize=(17, 4))
sns.barplot(data=top_missing, x='missing_pct', y='feature', ax=axes[0], color='tab:red')
axes[0].set_title('Top missingness percentages')
axes[0].set_xlabel('Missing %')

sns.histplot(y, bins=35, kde=True, ax=axes[1], color='tab:blue')
axes[1].set_title('Target distribution diagnostics')
axes[1].set_xlabel('MedHouseVal')

sns.heatmap(corr_numeric.loc[top_corr_cols, top_corr_cols], cmap='coolwarm', center=0, ax=axes[2])
axes[2].set_title('Numeric correlation heatmap (top columns)')
plt.tight_layout()
plt.show()
""",
        """
This gives a reproducible anchor score and a concrete data-quality snapshot.
Any new feature block should justify itself by measurable lift and stable fold behavior.
""",
    )

    add_code_walkthrough(
        nb,
        """
Now we implement a leakage-aware engineering block including interactions, ratios, geospatial grouping, and OOF target encoding.
Background: OOF statistics simulate production-time availability and avoid target leakage inflation.
""",
        """
def add_features(df):
    out = df.copy()

    # Ratio and interaction features grounded in domain intuition.
    out['rooms_per_household'] = out['AveRooms'] / (out['AveOccup'] + 1e-6)
    out['bedrooms_per_room'] = out['AveBedrms'] / (out['AveRooms'] + 1e-6)
    out['income_x_occupancy'] = out['MedInc'] * out['AveOccup']
    out['log_population'] = np.log1p(out['Population'].clip(lower=0))

    # Coarser geospatial grouping can expose regional nonlinear effects for MLPs once encoded.
    out['lat_lon_grid'] = out['Latitude'].round(1).astype(str) + '_' + out['Longitude'].round(1).astype(str)
    return out


def oof_target_encode(series, target, n_splits=5, smooth=20):
    series = series.astype(str)
    target = np.asarray(target)
    global_mean = target.mean()
    oof_encoded = np.zeros(len(series))
    folds = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    for tr_idx, va_idx in folds.split(series):
        tr_s, va_s = series.iloc[tr_idx], series.iloc[va_idx]
        tr_y = target[tr_idx]
        stats = pd.DataFrame({'cat': tr_s, 'target': tr_y}).groupby('cat')['target'].agg(['mean', 'count'])
        smooth_mean = (stats['mean'] * stats['count'] + smooth * global_mean) / (stats['count'] + smooth)
        oof_encoded[va_idx] = va_s.map(smooth_mean).fillna(global_mean).values
    return oof_encoded


X_eng = add_features(X_raw)
X_eng['geo_target_oof'] = oof_target_encode(X_eng['geo_bucket'], y, n_splits=5, smooth=30)

engineered_cfg = {**TABULAR_MLP_BASE_CFG, 'hidden_dims': (256, 128), 'dropout_p': 0.18, 'epochs': 38, 'patience': 6}
engineered_cv = run_tabular_mlp_cv(X_eng, y, engineered_cfg, n_splits=5)

summary = pd.DataFrame([
    {'pipeline': 'baseline_mlp', 'rmse_mean': baseline_cv['rmse_mean'], 'rmse_std': baseline_cv['rmse_std']},
    {'pipeline': 'engineered_mlp', 'rmse_mean': engineered_cv['rmse_mean'], 'rmse_std': engineered_cv['rmse_std']},
])
summary['delta_vs_baseline'] = summary['rmse_mean'] - summary.loc[summary['pipeline'] == 'baseline_mlp', 'rmse_mean'].iloc[0]
display(summary)

fold_compare = pd.DataFrame({
    'fold': np.arange(1, len(baseline_cv['fold_scores']) + 1),
    'baseline_rmse': baseline_cv['fold_scores'],
    'engineered_rmse': engineered_cv['fold_scores']
})
fold_long = fold_compare.melt(id_vars='fold', var_name='pipeline', value_name='rmse')

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
sns.lineplot(data=fold_long, x='fold', y='rmse', hue='pipeline', marker='o', ax=axes[0])
axes[0].set_title('Fold-by-fold RMSE diagnostics (PyTorch MLP)')

sns.barplot(data=summary, x='pipeline', y='rmse_mean', ax=axes[1], palette='Set2')
axes[1].errorbar(
    x=np.arange(len(summary)),
    y=summary['rmse_mean'],
    yerr=summary['rmse_std'],
    fmt='none',
    ecolor='black',
    capsize=4
)
axes[1].set_title('Mean RMSE with fold variance')
plt.tight_layout()
plt.show()

viz_cols = ['MedInc', 'rooms_per_household', 'income_x_occupancy', 'geo_target_oof']
viz_df = X_eng[viz_cols].copy()
viz_df['target'] = y.values
pair = sns.pairplot(viz_df.sample(min(1500, len(viz_df)), random_state=SEED), corner=True, diag_kind='hist')
pair.fig.suptitle('Feature-target relationship scan', y=1.02)
plt.show()

# Fit one holdout model for residual diagnostics and feature-target sanity checks.
X_fit, X_hold, y_fit, y_hold = train_test_split(X_eng, y, test_size=0.2, random_state=SEED)
eng_bundle = train_tabular_mlp(X_fit, y_fit, X_hold, y_hold, task='regression', **engineered_cfg)
hold_pred = eng_bundle['val_pred']
hold_resid = y_hold.values - hold_pred

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].scatter(hold_pred, hold_resid, alpha=0.35, s=12)
axes[0].axhline(0.0, color='black', linestyle='--')
axes[0].set_title('Holdout residuals vs prediction (engineered MLP)')
axes[0].set_xlabel('Prediction')
axes[0].set_ylabel('Residual')

numeric_corr = X_eng.select_dtypes(include=['number']).copy()
numeric_corr['target'] = y.values
corr_rank = numeric_corr.corr(numeric_only=True)['target'].drop('target').abs().sort_values(ascending=False).head(12)
axes[1].barh(corr_rank.index[::-1], corr_rank.values[::-1], color='tab:purple')
axes[1].set_title('Top numeric signal features (|corr with target|)')
plt.tight_layout()
plt.show()
""",
        """
A robust feature workflow ends with both quantitative lift and interpretability evidence.
If engineered lift is unstable across folds, simplify and re-test rather than shipping brittle complexity.
""",
    )

    if solution:
        md(
            nb,
            """
**Expected outcomes (typical):**
- Engineered MLP pipeline should often reduce RMSE by a small but meaningful margin (commonly 1-5%).
- The most useful features are usually ratio interactions and leakage-safe target statistics.
- If gains vanish with new folds/seeds, that is a signal to revisit leakage controls and feature robustness.
""",
        )
        add_code_walkthrough(
            nb,
            """
This solved extension demonstrates ablation-friendly feature blocks for disciplined iteration.
Background: grouped ablations prevent accidental overfitting from adding many uncontrolled transforms at once.
""",
            """
feature_blocks = {
    'ratios_only': ['rooms_per_household', 'bedrooms_per_room', 'income_x_occupancy'],
    'geo_only': ['lat_lon_grid', 'geo_target_oof'],
    'all_new': ['rooms_per_household', 'bedrooms_per_room', 'income_x_occupancy', 'log_population', 'lat_lon_grid', 'geo_target_oof'],
}

ablation_rows = []
for name, cols in feature_blocks.items():
    subset = X_raw.copy()
    for col in cols:
        subset[col] = X_eng[col]

    sub_cv = run_tabular_mlp_cv(subset, y, engineered_cfg, n_splits=5)
    ablation_rows.append({
        'feature_block': name,
        'rmse_mean': sub_cv['rmse_mean'],
        'rmse_std': sub_cv['rmse_std']
    })

ablation_df = pd.DataFrame(ablation_rows).sort_values('rmse_mean')
display(ablation_df)

plt.figure(figsize=(8, 4))
sns.barplot(data=ablation_df, x='feature_block', y='rmse_mean', palette='viridis')
plt.errorbar(x=np.arange(len(ablation_df)), y=ablation_df['rmse_mean'], yerr=ablation_df['rmse_std'], fmt='none', ecolor='black', capsize=4)
plt.title('Feature-block ablation (PyTorch MLP)')
plt.tight_layout()
plt.show()
""",
            """
Keep only feature blocks that improve mean score without inflating variance.
A compact, stable feature set usually outperforms an over-engineered fragile one on private leaderboard.
""",
        )



def add_kaggle_playbook_sections(nb, solution):
    md(
        nb,
        """
## Kaggle Playbook Simulation
We simulate a public/private leaderboard split and train a repeatable decision workflow around CV, shake risk, and ensembling.
""",
    )

    add_code_walkthrough(
        nb,
        """
This cell builds a competition-style evaluation stack with a hidden private split.
Background: robust Kaggle prep means optimizing validation quality and shake resistance, not only public leaderboard bumps.
""",
        """
X_raw, y = build_tabular_frame()
X = X_raw.copy()
X['rooms_per_household'] = X['AveRooms'] / (X['AveOccup'] + 1e-6)
X['income_x_occupancy'] = X['MedInc'] * X['AveOccup']

X_train_full, X_lb, y_train_full, y_lb = train_test_split(X, y, test_size=0.30, random_state=SEED)
X_public, X_private, y_public, y_private = train_test_split(X_lb, y_lb, test_size=0.50, random_state=SEED)

# Two PyTorch MLP candidates with different capacity/regularization choices.
candidates = {
    'mlp_deep': {**TABULAR_MLP_BASE_CFG, 'hidden_dims': (320, 160, 64), 'dropout_p': 0.12, 'epochs': 45, 'patience': 7},
    'mlp_regularized': {**TABULAR_MLP_BASE_CFG, 'hidden_dims': (192, 96), 'dropout_p': 0.24, 'weight_decay': 3e-5, 'epochs': 45, 'patience': 7},
}

rows = []
cv_fold_rows = []
trained_candidates = {}
public_pred_store = {}
private_pred_store = {}

for name, cfg_local in candidates.items():
    cv_out = run_tabular_mlp_cv(X_train_full, y_train_full, cfg_local, n_splits=5)
    for fold_idx, fold_score in enumerate(cv_out['fold_scores'], start=1):
        cv_fold_rows.append({'model': name, 'fold': fold_idx, 'rmse': fold_score})

    # Train submission-facing model and score public/private proxies.
    bundle = train_tabular_mlp(X_train_full, y_train_full, X_public, y_public, task='regression', **cfg_local)
    pred_public = bundle['val_pred']
    pred_private = predict_tabular_mlp(bundle, X_private, task='regression')

    trained_candidates[name] = bundle
    public_pred_store[name] = pred_public
    private_pred_store[name] = pred_private

    rows.append({
        'model': name,
        'cv_proxy_score': cv_out['rmse_mean'],
        'public_proxy_score': rmse(y_public, pred_public),
        'private_proxy_score': rmse(y_private, pred_private),
        'best_epoch': bundle['best_epoch'],
        'stop_epoch': bundle['stopped_epoch']
    })

leaderboard = pd.DataFrame(rows)
leaderboard['shake_abs'] = (leaderboard['private_proxy_score'] - leaderboard['public_proxy_score']).abs()
cv_fold_df = pd.DataFrame(cv_fold_rows)
display(leaderboard.sort_values('private_proxy_score'))
""",
        """
Treat this as a process rehearsal: build habits for split design, leakage control, and shake-aware model selection.
A model with slightly worse public score but lower shake can be the safer final submission.
""",
    )

    add_code_walkthrough(
        nb,
        """
This diagnostics block mirrors an early Kaggle EDA pass: check split shift, missingness structure, and feature-target behavior before tuning.
Background: many leaderboard failures are data-split problems disguised as model problems.
""",
        """
key_features = ['MedInc', 'HouseAge', 'AveRooms', 'Population']
shift_df = pd.concat([
    X_train_full[key_features].assign(split='train_full'),
    X_public[key_features].assign(split='public'),
    X_private[key_features].assign(split='private')
], ignore_index=True)

fig, axes = plt.subplots(2, 2, figsize=(13, 8))
for ax, col in zip(axes.ravel(), key_features):
    sns.kdeplot(data=shift_df, x=col, hue='split', fill=True, alpha=0.2, common_norm=False, ax=ax)
    ax.set_title(f'Distribution shift check: {col}')
plt.tight_layout()
plt.show()

missing_by_split = pd.DataFrame({
    'train_full': X_train_full.isna().mean(),
    'public': X_public.isna().mean(),
    'private': X_private.isna().mean()
}) * 100
top_missing_cols = missing_by_split.max(axis=1).sort_values(ascending=False).head(10).index

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
sns.heatmap(missing_by_split.loc[top_missing_cols], annot=True, fmt='.2f', cmap='Reds', ax=axes[0])
axes[0].set_title('Missingness heatmap (% by split)')

missing_plot = missing_by_split.loc[top_missing_cols].reset_index().melt(id_vars='index', var_name='split', value_name='missing_pct')
missing_plot = missing_plot.rename(columns={'index': 'feature'})
sns.barplot(data=missing_plot, x='missing_pct', y='feature', hue='split', ax=axes[1])
axes[1].set_title('Top missingness bars by split')
axes[1].set_xlabel('Missing %')
plt.tight_layout()
plt.show()

target_rel = pd.concat([
    X_train_full[['MedInc', 'AveRooms']].assign(target=y_train_full.values, split='train_full'),
    X_public[['MedInc', 'AveRooms']].assign(target=y_public.values, split='public'),
    X_private[['MedInc', 'AveRooms']].assign(target=y_private.values, split='private')
], ignore_index=True)

g = sns.lmplot(
    data=target_rel.sample(min(3000, len(target_rel)), random_state=SEED),
    x='MedInc',
    y='target',
    hue='split',
    scatter_kws={'alpha': 0.25, 's': 12},
    height=4,
    aspect=1.5
)
g.fig.suptitle('Feature-target relationship by split (MedInc vs target)', y=1.02)
plt.show()
""",
        """
If one split has visibly shifted feature distribution or broken feature-target slope, prioritize validation redesign before heavy model search.
This step often explains public/private shake more than any hyperparameter tweak.
""",
    )

    add_code_walkthrough(
        nb,
        """
This solved strategy cell evaluates simple ensembling and run-card logging.
Background: Kaggle gains often come from disciplined small improvements plus strong experiment bookkeeping.
""",
        """
weights = np.linspace(0.0, 1.0, 11)
ens_rows = []
for w in weights:
    ens_public = w * public_pred_store['mlp_deep'] + (1 - w) * public_pred_store['mlp_regularized']
    ens_private = w * private_pred_store['mlp_deep'] + (1 - w) * private_pred_store['mlp_regularized']
    ens_rows.append({
        'weight_mlp_deep': float(w),
        'public_rmse': rmse(y_public, ens_public),
        'private_rmse': rmse(y_private, ens_private)
    })

ens_df = pd.DataFrame(ens_rows)
ens_df['shake_abs'] = (ens_df['private_rmse'] - ens_df['public_rmse']).abs()
display(ens_df.sort_values('public_rmse').head(6))

best_idx = ens_df['private_rmse'].idxmin()
print('Most private-robust ensemble row:')
display(ens_df.loc[[best_idx]])

run_card = {
    'seed': SEED,
    'folds': 5,
    'feature_version': 'v1_rooms_income_interactions',
    'model_family': 'pytorch_tabular_mlp_blend',
    'notes': 'Selected by private-robust simulated leaderboard, not just public score'
}
print('Run card:', run_card)
""",
        """
Use this template to avoid ad-hoc submissions: log assumptions, compare robustly, and submit with an explicit risk rationale.
Kaggle consistency usually beats one-off lucky spikes.
""",
    )

    add_code_walkthrough(
        nb,
        """
This final competition dashboard compares fold stability and leaderboard-style behavior across candidates and blends.
Background: deciding what to submit is mostly a visualization + risk-management problem.
""",
        """
leaderboard_plot = leaderboard.copy()
leaderboard_plot['leaderboard_gap'] = leaderboard_plot['private_proxy_score'] - leaderboard_plot['public_proxy_score']

fig, axes = plt.subplots(1, 3, figsize=(18, 4))
sns.boxplot(data=cv_fold_df, x='model', y='rmse', ax=axes[0], palette='Set3')
sns.stripplot(data=cv_fold_df, x='model', y='rmse', ax=axes[0], color='black', size=4, alpha=0.6)
axes[0].set_title('CV-fold RMSE comparison (PyTorch MLP)')

sns.scatterplot(data=leaderboard_plot, x='public_proxy_score', y='private_proxy_score', hue='model', s=130, ax=axes[1])
for _, row in leaderboard_plot.iterrows():
    axes[1].annotate(row['model'], (row['public_proxy_score'], row['private_proxy_score']), xytext=(4, 4), textcoords='offset points')
axes[1].set_title('Leaderboard-style public vs private proxy')
axes[1].plot(
    [leaderboard_plot['public_proxy_score'].min(), leaderboard_plot['public_proxy_score'].max()],
    [leaderboard_plot['public_proxy_score'].min(), leaderboard_plot['public_proxy_score'].max()],
    linestyle='--', color='gray', linewidth=1
)

sns.lineplot(data=ens_df, x='weight_mlp_deep', y='public_rmse', marker='o', label='public_rmse', ax=axes[2])
sns.lineplot(data=ens_df, x='weight_mlp_deep', y='private_rmse', marker='o', label='private_rmse', ax=axes[2])
axes[2].set_title('Blend weight leaderboard trace')
axes[2].set_ylabel('RMSE')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
sns.barplot(data=leaderboard_plot.sort_values('shake_abs', ascending=False), x='model', y='shake_abs', palette='rocket')
plt.title('Shake-risk leaderboard (absolute public-private gap)')
plt.ylabel('Absolute gap')
plt.tight_layout()
plt.show()
""",
        """
Ship candidates that are strong on both fold stability and shake-risk visuals, not just one metric snapshot.
These plots make submission decisions auditable and defensible under deadline pressure.
""",
    )

    if solution:
        md(
            nb,
            """
**Expected outcomes (typical):**
- Blended MLP models often reduce private-proxy RMSE vs any single model.
- The public winner is not always the private winner; shake-aware selection is critical.
- A structured run card prevents metric confusion and accelerates team collaboration.
""",
        )



def add_competition_exercises(nb, lesson, solution):
    md(
        nb,
        """
## Explanatory Depth Checkpoints
- **Why this workflow?** Because leaderboard gains are fragile without leakage-safe validation and reproducible logs.
- **Key idea:** Strong experiments isolate one hypothesis at a time so score movement has a clear causal explanation.
- **Important pitfall:** A public-score spike can hide private overfitting; always compare fold variance and shake risk.
- **In practice:** Keep assumptions explicit, test alternatives quickly, and record rollback plans before each submission.
- **Question:** Which diagnostic would make you reject a seemingly strong model?
- **Question:** How would you defend your final submission choice to a teammate under deadline pressure?
""",
    )

    if lesson["id"] == "09":
        md(
            nb,
            """
## Kaggle-Prep Exercise Bank (Dataset-Driven)
### Track A — Validation Design
1. **Exercise:** Build 5-fold, 10-fold, and GroupKFold variants; compare rank stability across folds.
2. **Exercise:** Create a leakage trap feature intentionally, detect it, and document your detection rule.
3. **Exercise:** Simulate temporal drift by sorting on a proxy time feature and using time-aware splits.

### Track B — Modeling Strategy
4. **Exercise:** Train at least three model families (linear, tree boosting, bagging) with a fixed budget.
5. **Exercise:** Build a score-vs-runtime frontier and justify your final Pareto-optimal candidates.
6. **Exercise:** Tune one model deeply and another model shallowly; compare marginal gain per minute.

### Track C — Ensembling and Risk
7. **Exercise:** Blend two models with weights from 0.0 to 1.0 and report public/private shake.
8. **Exercise:** Add a third model and test whether diversity beats raw single-model strength.
9. **Question:** Construct a submission policy: when should you trust CV over public leaderboard?

### Track D — Reporting and Communication
10. **Exercise:** Create an experiment registry with feature version, seed, fold schema, and commit hash.
11. **Exercise:** Write a post-mortem for one failed experiment and what diagnostic disproved the hypothesis.
12. **Exercise:** Draft a final competition report with reproducibility checklist and fallback plan.
""",
        )
    else:
        md(
            nb,
            """
## Exercises (Competition-Oriented)
1. **Exercise:** Re-run all experiments with a second seed and quantify score/rank variance.
2. **Exercise:** Replace one preprocessing block and document exact before/after effects.
3. **Exercise:** Perform a 3-row ablation table: baseline, +feature/model change, +tuning change.
4. **Exercise:** Add one failure analysis plot and explain what action it suggests.
5. **Exercise:** Build a strict 30-minute local runtime pipeline and maximize validation score.
6. **Question:** Which feature block improved mean score but harmed fold stability, and why?
7. **Exercise:** Reproduce your best run with one different fold schema and compare ranking movement.
8. **Exercise:** Add a simple blend and report whether shake risk improved versus single-model baseline.
9. **Question:** If compute is cut by 50%, which experiments stay and which are dropped?
10. **Exercise:** Write a short decision memo defending one final submission and one rollback candidate.
""",
        )

    if solution:
        md(
            nb,
            f"""
### Solution Depth Notes
- Includes complete runnable code for realistic dataset-driven tasks.
- Reports expected metric behavior and variance, not just single-run numbers.
- Provides alternative path: {lesson['alternatives']}
""",
        )
        add_code_walkthrough(
            nb,
            """
This solved scaffold demonstrates a disciplined experiment registry and decision protocol.
Background: strong Kaggle results come from repeatable process more than isolated tricks.
""",
            """
experiment_log = pd.DataFrame([
    {'exp_id': 'base_001', 'feature_set': 'raw', 'model': 'hist_gbdt', 'cv_metric': 0.585, 'public_metric': 0.592, 'notes': 'baseline'},
    {'exp_id': 'feat_014', 'feature_set': 'raw+ratios+geo_oof', 'model': 'hist_gbdt', 'cv_metric': 0.561, 'public_metric': 0.566, 'notes': 'feature lift'},
    {'exp_id': 'blend_021', 'feature_set': 'feat_014', 'model': 'tree_blend', 'cv_metric': 0.554, 'public_metric': 0.558, 'notes': 'blend + shake control'},
])

experiment_log['delta_cv'] = experiment_log['cv_metric'] - experiment_log['cv_metric'].iloc[0]
experiment_log['delta_public'] = experiment_log['public_metric'] - experiment_log['public_metric'].iloc[0]
display(experiment_log)

decision_rule = '''
1) Filter experiments by reproducibility + leakage checks.
2) Select top candidates by CV mean and variance.
3) Break ties using simulated shake or private-proxy behavior.
4) Submit with explicit rollback candidate.
'''.strip()
print(decision_rule)
""",
            """
Keep this structure in every project to make model selection auditable and stress-tested.
The key competency is reliable decision quality under uncertainty.
""",
        )


def build_notebook(lesson, solution=False):
    nb = new_notebook()
    add_common_intro(nb, lesson, solution)
    add_math_section(nb, lesson, solution)
    if lesson["id"] in {"07", "08", "09"}:
        add_tabular_competition_setup(nb)
        if lesson["id"] == "07":
            add_xgboost_lightgbm_sections(nb, solution)
        elif lesson["id"] == "08":
            add_feature_engineering_sections(nb, solution)
        else:
            add_kaggle_playbook_sections(nb, solution)
        add_best_practices(nb, lesson)
        add_competition_exercises(nb, lesson, solution)
    else:
        add_setup_code(nb, lesson)
        add_training_utilities(nb)
        add_synthetic_sections(nb, lesson, solution)
        add_real_data_sections(nb, lesson, solution)
        add_visualizations(nb)

        if lesson["id"] == "05":
            add_early_stopping_mlp_focus(nb, solution)

        if lesson["id"] == "06":
            add_optuna_section(nb, solution)

        add_best_practices(nb, lesson)
        add_exercises(nb, lesson, solution)

    suffix = "_solutions" if solution else ""
    filename = f"{lesson['id']}_{lesson['slug']}{suffix}.ipynb"
    save(nb, filename)


def main():
    for lesson in LESSONS:
        build_notebook(lesson, solution=False)
        build_notebook(lesson, solution=True)
    print(f"Generated {len(LESSONS) * 2} notebooks.")


if __name__ == "__main__":
    main()
