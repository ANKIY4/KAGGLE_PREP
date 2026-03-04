# WHY FEATURES BEAT MODELS (MOST OF THE TIME)

## Purpose
Build a transferable mental model for tabular competitions: most sustainable gains come from better representation of signal (features and data framing), while model swapping usually gives smaller, less reliable improvements.

## Core claim
On structured data, feature quality changes the problem you are asking the model to solve; model swapping usually changes only how aggressively you fit the same problem.

## 1) Bias-variance view: where the big gains usually are
For squared error intuition:

\[
\mathbb{E}[(y-\hat f(x))^2] = \text{noise} + \text{bias}^2 + \text{variance}
\]

- **Model swaps** (e.g., XGBoost \(\leftrightarrow\) LightGBM \(\leftrightarrow\) CatBoost) mostly trade off bias/variance within a similar hypothesis class for tabular data.
- **Feature work** changes \(x \rightarrow \phi(x)\), often reducing effective bias by exposing structure the model could not express cleanly from raw columns.
- If \(\phi(x)\) is poor, stronger models may just fit noise harder (variance up) with little true gain.

Practical consequence: moving to a better representation often shifts the entire Pareto frontier; changing boosters usually moves you along it.

## 2) Signal extraction beats optimizer cleverness
Raw columns often have weak local signal-to-noise ratio. Feature engineering increases usable signal by:

- **Aggregation** (user/item/history summaries) to reveal stable tendencies.
- **Interactions** to encode mechanisms (price × season, user × category).
- **Temporal framing** (lags, rolling windows, recency decay) to expose dynamics.
- **Leakage-safe encodings** (out-of-fold target stats) to summarize high-cardinality categories.

These steps raise \(I(\phi(X); Y)\) (information about the target), so even simpler models can learn robustly. Better optimization cannot recover signal that was never represented.

## 3) Representation quality vs model swapping
A useful empirical pattern in competitions:

- With weak features, many models cluster at similar CV scores.
- With strong features, many models improve together; simple baselines become competitive.
- Final model choice then becomes a **second-order** decision (often basis points), while feature framing was the **first-order** jump.

So the key question is usually not “which model is best?” but “what structure of the data is currently invisible to the model?”

## 4) When model changes do matter
Model choice can still be material when:

- your metric is highly rank-sensitive and calibration behavior differs;
- data size pushes one algorithm into a better compute/regularization regime;
- modality shifts (images/text/sequences) where architecture defines representation power.

But even here, gains are largest when representation and split design match the real prediction scenario.

## 5) Transferable workflow
1. Lock a trustworthy CV scheme that matches test-time constraints.
2. Do error analysis by slice (time, group, rarity, tail targets).
3. Propose mechanism-based features from failures.
4. Validate features with ablations and variance-aware CV deltas.
5. Only then spend time on model/hyperparameter swaps.

This workflow generalizes across competitions because it is about information and generalization, not leaderboard-specific tricks.
