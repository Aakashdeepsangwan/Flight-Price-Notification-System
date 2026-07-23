# ML Structure — Flight Price Notification System

A learning roadmap for the ML pipeline. Every algorithm has prerequisites. Every concept has a reason.  
Read this top to bottom — each section unlocks the next.

---

## Layer 0 — Foundations (learn these before any algorithm)

These are not algorithms. They are the **rules of the game** that every ML algorithm follows.  
Skipping them means you'll misread your model's results.

```
┌─────────────────────────────────────────────────────────────────┐
│                        FOUNDATIONS                              │
│                                                                 │
│  Train / Test Split                                             │
│      Why  : models memorize training data — you need unseen     │
│             data to know if they actually learned               │
│      Rule : 80% train · 20% test (time-ordered for prices)     │
│                                                                 │
│  Overfitting vs Underfitting                    ← learn first   │
│      Overfit  : model memorized training data, fails on new     │
│      Underfit : model too simple, misses the pattern            │
│      Balance  : this tension drives every decision below        │
│                                                                 │
│  Bias–Variance Tradeoff                                         │
│      High Bias    → underfitting (too simple)                   │
│      High Variance → overfitting (too complex)                  │
│      Goal : minimize both simultaneously                        │
│                                                                 │
│  Evaluation Metrics                                             │
│      Regression    : RMSE · MAE · R²                            │
│      Classification: Precision · Recall · F1 · AUC-ROC         │
│      Time-series   : MAPE (mean absolute % error)               │
│                                                                 │
│  Feature Scaling                                                │
│      Standardization (z-score) : required for distance-based    │
│      Normalization (min-max)   : required for neural networks   │
│      Tree models               : scaling not required           │
│                                                                 │
│  Regularization                                                 │
│      L1 (Lasso) : shrinks weak features to zero                 │
│      L2 (Ridge) : shrinks all features proportionally           │
│      Why        : prevents overfitting in linear models         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Layer 1 — Cross-Validation (bridge between foundations and algorithms)

Cross-validation is **how you trust your evaluation**. It must be understood before training any model.

```
Standard K-Fold CV                    Time-Series CV (Walk-Forward)
─────────────────────                 ──────────────────────────────
 Fold 1: [Train][Test]                Week 1–4  : Train → Test on week 5
 Fold 2: [Train][Test]                Week 1–5  : Train → Test on week 6
 Fold 3: [Train][Test]                Week 1–6  : Train → Test on week 7
 Fold 4: [Train][Test]       vs             ...
 Fold 5: [Train][Test]                Never use future data to predict past

Use K-Fold for:                       Use Walk-Forward for:
  general ML models                     flight price models ← this project
  classification                        any time-ordered data
```

> **Why time-series CV matters here:** flight prices on Monday affect Tuesday's prices.  
> Shuffling data randomly leaks future information into training — your model looks accurate but isn't.

**Prerequisite chain:**
```
Train/Test Split → Overfitting concept → Cross-Validation → Evaluate any model below
```

---

## Layer 2 — Unsupervised Learning

No labels. No target variable. The model finds structure on its own.  
**These run first in the pipeline — day 1, before any labeled data exists.**

### Prerequisite: Distance Metrics

K-Means and DBSCAN are both distance-based. Learn this first.

```
Euclidean Distance  : straight-line distance between two price points
Manhattan Distance  : sum of absolute differences (more robust to outliers)

Feature Scaling is REQUIRED here — unscaled features distort distances.
```

---

### 2A — K-Means Clustering

```
Prerequisite : Distance Metrics + Feature Scaling

Concept  : place K centroids, assign each route to nearest centroid,
           recompute centroids, repeat until stable

For this project:
  Input  : per-route stats (mean price, std, min, trend slope)
  Output : cluster label — "stable" | "volatile" | "seasonal"

Watch for:
  Choosing K  → use Elbow Method (plot inertia vs K, pick the "elbow")
  Overfit risk → none (unsupervised), but wrong K gives useless clusters
```

**Elbow method:**
```
Inertia
  │  \
  │   \
  │    \──────────────────  ← pick the bend (K=3 here)
  └────────────────────── K
       1  2  3  4  5  6
```

---

### 2B — Isolation Forest

```
Prerequisite : understanding of decision trees (even a basic idea)

Concept  : randomly partitions data — anomalies get isolated in fewer
           splits because they are "different" from the majority

For this project:
  Input  : current price snapshot vs route history
  Output : anomaly score  (−1 = anomalous price drop, +1 = normal)

Watch for:
  contamination param → set to expected % of anomalous prices (~5–10%)
  Overfit risk        → low (isolation is inherently generalizable)
```

---

### 2C — DBSCAN

```
Prerequisite : Distance Metrics + Isolation Forest concept

Concept  : groups nearby points into clusters, labels isolated points
           as outliers — no need to specify K upfront

For this project:
  Input  : price sequences per route
  Output : outlier flag (−1 = structural outlier, 0+ = cluster member)

vs K-Means:
  K-Means → forces every point into a cluster (no outlier concept)
  DBSCAN  → explicitly marks outliers, handles irregular cluster shapes

Watch for:
  eps (neighborhood radius) → sensitive parameter, tune with k-distance plot
  Overfit risk              → moderate if eps is too small
```

---

### 2D — STL Decomposition *(not a clustering algorithm — a signal processor)*

```
Prerequisite : basic time-series understanding (trend, seasonality)

Concept  : splits a time series into three additive components:
           Price(t) = Trend(t) + Seasonal(t) + Residual(t)

For this project:
  Trend    → long-run direction of prices for a route
  Seasonal → weekly / annual demand cycles
  Residual → unexplained noise → fed into Isolation Forest

Why before regression?
  Training a regressor on raw prices mixes trend with noise.
  Separating them gives Phase 2 models cleaner, more predictable input.
```

---

### Unsupervised Summary

```
Feature Scaling
      │
      ├──▶ Distance Metrics
      │          │
      │          ├──▶ K-Means ──────────────────────▶ cluster labels (→ Phase 3 feature)
      │          └──▶ DBSCAN ──────────────────────▶ outlier flags  (→ anomaly alert)
      │
      ├──▶ Isolation Forest ──────────────────────▶ anomaly score  (→ anomaly alert)
      │
      └──▶ STL Decomposition ─────────────────────▶ trend + seasonal components (→ Phase 3 features)
```

---

## Layer 3 — Supervised Learning: Regression

**Predicts a continuous value — the future price.**  
Requires labeled data: historical (feature vector → actual price) pairs.

### Prerequisite: Cost Functions + Gradient Descent

```
Cost Function (MSE): measures how wrong the model is
  MSE = (1/n) Σ (predicted − actual)²

Gradient Descent: iteratively reduces the cost by adjusting weights
  θ = θ − α · ∇J(θ)     (α = learning rate)

These underpin every regression model below.
```

---

### 3A — Linear / Ridge Regression

```
Prerequisite : Cost Functions + Gradient Descent + Regularization (L2)

Concept  : fits a straight line (or hyperplane) through the data
           Ridge adds L2 penalty to prevent overfitting

For this project:
  Input  : engineered features
  Output : predicted price in N days
  Role   : baseline — if XGBoost doesn't beat this, something is wrong

Overfit risk : LOW (high bias model — more likely to underfit)
Underfit risk: HIGH on non-linear price patterns → use as benchmark only
```

---

### 3B — Prophet *(time-series specific)*

```
Prerequisite : STL Decomposition understanding + Linear Regression

Concept  : decomposes time series like STL but is trainable — learns
           trend, weekly seasonality, and holiday effects as parameters

For this project:
  Input  : (timestamp, price) pairs per route
  Output : price forecast for next 7–14 days with confidence intervals

vs Linear Regression:
  Linear Reg → ignores the time dimension
  Prophet    → explicitly models it

Overfit risk : LOW (built-in regularization on trend flexibility)
Underfit risk: MODERATE on highly volatile routes
```

---

### 3C — LSTM *(deep learning — optional Phase 3)*

```
Prerequisite : Prophet + basic neural network understanding (weights, activations)

Concept  : a recurrent neural network that learns from sequences —
           each price influences the prediction of the next

For this project:
  Input  : sequence of last N price snapshots
  Output : predicted price at step N+1

vs Prophet:
  Prophet → interpretable, fast, good with seasonality
  LSTM    → better on complex non-linear sequences, needs more data

Overfit risk : HIGH — requires dropout regularization + enough data (10k+ rows)
When to use  : only after 3–6 months of price history
```

---

### 3D — XGBoost Regressor *(primary regression model)*

```
Prerequisite : Decision Trees → Random Forest → Gradient Boosting concept

Concept  : builds trees sequentially — each tree corrects the errors
           of the previous one (gradient boosting)

Learning chain to XGBoost:
  Decision Tree → single tree that splits on features
       ↓
  Random Forest → many trees in parallel (bagging) — reduces variance
       ↓
  Gradient Boosting → trees in sequence, each fixing prior errors
       ↓
  XGBoost → optimized gradient boosting with regularization + speed

For this project:
  Input  : full feature vector + cluster labels + STL components
  Output : predicted price in N days
  Eval   : RMSE across time-fold cross-validation

Overfit risk : MODERATE — tune max_depth, learning_rate, n_estimators
Key params   : learning_rate (0.01–0.1) · max_depth (3–6) · subsample (0.8)
```

---

### Regression Learning Chain

```
Cost Function + Gradient Descent
         │
         ├──▶ Linear / Ridge Regression     (learn the baseline)
         │           │
         │           ▼
         │     Does it overfit?
         │     Yes → add Ridge (L2)
         │     No  → baseline is strong enough to compare against
         │
         ├──▶ Prophet / LSTM                (learn the time dimension)
         │           │
         │           ▼
         │     captures weekly/annual patterns Linear Reg misses
         │
         └──▶ XGBoost                       (primary model)
                     │
                     ▼
               combine all signals → best price prediction
               → output feeds Phase 4 classification
```

---

## Layer 4 — Supervised Learning: Classification

**Answers: "Buy now (1) or Wait (0)?"**  
Input includes predicted prices from Layer 3.

---

### 4A — Logistic Regression

```
Prerequisite : Linear Regression + understanding of probability / sigmoid

Concept  : applies sigmoid to linear output to produce a probability
           P(buy) = σ(w·x + b)   → value between 0 and 1

For this project:
  Input  : feature vector + predicted price from XGBoost
  Output : P(buy now) — threshold at 0.5 by default

Overfit risk : LOW (simple decision boundary)
When to use  : always train this first as a classification baseline

vs Linear Regression:
  Linear Reg  → outputs any number (−∞ to +∞)
  Logistic Reg → squashes output to [0, 1] — interpretable as probability
```

---

### 4B — Random Forest Classifier *(primary classification model)*

```
Prerequisite : Decision Trees + Random Forest Regressor concept (same idea, different output)

Concept  : ensemble of decision trees voting on Buy vs Wait

For this project:
  Input  : full feature vector + XGBoost predicted price + anomaly score
  Output : class (Buy=1 / Wait=0) + probability score
  Eval   : Precision · Recall · F1

Overfit risk : LOW–MODERATE (bagging reduces variance naturally)
Tune         : n_estimators (100–500) · max_depth · min_samples_leaf

Why not XGBoost for classification?
  RF Classifier is more stable with imbalanced labels (few "buy" signals
  vs many "wait" signals) and produces well-calibrated probabilities.
  XGBoost can be added later as an ensemble member.
```

---

### Classification Learning Chain

```
Logistic Regression     (baseline — learn the boundary)
         │
         ▼
  Does precision / recall meet threshold?
  Yes → ship it
  No  → move to ensemble

         │
         ▼
Random Forest Classifier  (primary)
         │
         ▼
  Output: P(buy) → if P > 0.7 → trigger alert
```

---

## Full ML Learning Prerequisite Map

```
FOUNDATIONS
│
├── Train/Test Split
├── Overfitting / Underfitting  ◄── learn before anything else
├── Bias–Variance Tradeoff
├── Evaluation Metrics
├── Feature Scaling
└── Regularization (L1 / L2)
         │
         ▼
Cross-Validation (Walk-Forward for time-series)
         │
         ├────────────────────────────────────────────────────────────┐
         ▼                                                            ▼
UNSUPERVISED (no labels)                             SUPERVISED (needs history)
         │                                                            │
Distance Metrics                              Cost Function + Gradient Descent
    │                                                  │
    ├──▶ K-Means                          ┌──▶ Linear / Ridge Regression
    └──▶ DBSCAN                           │         │
                                          │         ▼
Isolation Forest                          │    Decision Trees
    │                                     │         │
    ▼                                     │         ▼
Anomaly Alerts (day 1)                    │    Random Forest Regressor
                                          │         │
STL Decomposition                         │         ▼
    │                                     │    Gradient Boosting
    ▼                                     │         │
Trend / Seasonal features ────────────────┤         ▼
                                          └──▶ XGBoost Regressor
K-Means cluster labels ───────────────────────────  │
                                                     ▼
                                             Prophet / LSTM (time-series)
                                                     │
                                                     ▼
                                          ┌── Predicted Price ──────────────────┐
                                          │                                     │
                                          ▼                                     ▼
                                  Logistic Regression             Random Forest Classifier
                                  (classification baseline)       (Buy / Wait — primary)
                                          │                                     │
                                          └──────────────┬──────────────────────┘
                                                         ▼
                                                   Decision Engine
                                                         │
                                                         ▼
                                                   Notification
```

---

## When Each Layer Fires

| Layer | Algorithms | Fires when | Overfitting risk |
|---|---|---|---|
| 0 | Foundations — no training | Always | — |
| 1 | Walk-Forward Cross-Validation | Before every model | — |
| 2A | K-Means, DBSCAN, Isolation Forest | Week 1 (no labels needed) | Low |
| 2B | STL Decomposition | Week 1 (signal processing) | None |
| 3A | Linear / Ridge Regression | After ~4 weeks of data | Low |
| 3B | Prophet / LSTM | After ~4 weeks (time-series) | Low / High |
| 3C | XGBoost Regressor | After ~4 weeks (primary) | Moderate |
| 4A | Logistic Regression | After Phase 3 predictions | Low |
| 4B | Random Forest Classifier | After Phase 3 predictions | Low–Moderate |
