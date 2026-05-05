# Midway Progress Report
## Predicting Wine Quality from Physicochemical Properties: A Comparative Study of Supervised Learning Methods

**Student:** Mengjia Shang
**Student ID:** 7338151449
**Course:** EE559 / CSCI 559 — Machine Learning, Spring 2026
**Instructor:** Prof. Anand A. Joshi
**Report Date:** 04/13/2026

---

## 1. Project Status Summary

All proposed midway deliverables are complete: the dataset has been downloaded and validated, exploratory data analysis (EDA) has been performed, feature engineering has been finalized, and two of three planned models — Logistic Regression (L2) and Support Vector Machine (RBF kernel) — have been fully trained and evaluated on the validation set. The test set remains held out. The third model (Random Forest) is planned for the final report phase.

---

## 2. Data Exploration & Preprocessing

### 2.1 Dataset Overview

The UCI Wine Quality dataset was downloaded from the UCI Machine Learning Repository. Red wine (1,599 samples) and white wine (4,898 samples) CSV files were merged into a single dataset. A binary `wine_type` indicator (0 = red, 1 = white) was added as an engineered feature before merging.

| Property | Value |
|---|---|
| Total samples | 6,497 |
| Original features | 11 physicochemical + 1 quality score |
| Missing values | **0** (no imputation required) |
| Feature dtypes | All numeric (float64 / int64) |

### 2.2 Target Variable Construction

The continuous quality score (integer, range 3–9) was binarized into a high-quality label:

- **High quality (label = 1):** quality ≥ 6 → **4,113 samples (63.3%)**
- **Low quality (label = 0):** quality < 6 → **2,384 samples (36.7%)**

The ~27-point class imbalance is moderate but non-trivial. It motivates the use of ROC-AUC and macro-averaged F1 as primary evaluation metrics rather than raw accuracy.

### 2.3 Descriptive Statistics (Selected Features)

| Feature | Mean | Std | Min | Max |
|---|---|---|---|---|
| fixed acidity | 7.22 | 1.30 | 3.80 | 15.90 |
| volatile acidity | 0.34 | 0.16 | 0.08 | 1.58 |
| residual sugar | 5.44 | 4.76 | 0.60 | 65.80 |
| chlorides | 0.056 | 0.035 | 0.009 | 0.611 |
| alcohol | 10.49 | 1.19 | 8.00 | 14.90 |
| quality | 5.82 | 0.87 | 3.00 | 9.00 |

Notable observations from EDA:
- **Alcohol** is the feature most positively correlated with quality (r ≈ +0.44), confirming domain knowledge that higher-alcohol wines tend to score higher.
- **Volatile acidity** is the most negatively correlated (r ≈ −0.27); high volatile acidity produces vinegar-like off-flavors.
- **Residual sugar** and **chlorides** exhibit extreme right-skewed distributions with outliers that were addressed in preprocessing.
- **Density** is highly collinear with alcohol and residual sugar (|r| > 0.8 in each case), suggesting potential redundancy.

### 2.4 Preprocessing Steps

The following steps were applied in order:

1. **Outlier winsorization:** `chlorides` and `residual sugar` were clipped at the 1st and 99th percentiles to reduce the influence of extreme values without discarding samples.
2. **Feature engineering:**
   - `so2_ratio = free SO₂ / (total SO₂ + ε)` — captures the effective preservation fraction.
   - `acidity_ratio = volatile acidity / (fixed acidity + ε)` — encodes the relative contribution of spoilage-related acidity.
   - `wine_type` (0/1) — encodes systematic compositional differences between red and white wines.
3. **Standardization:** All 14 features were standardized to zero mean and unit variance using `StandardScaler` fit **exclusively on the training fold** to prevent data leakage.

Final feature set: **14 features** (11 original physicochemical + wine_type + so2_ratio + acidity_ratio).

### 2.5 Data Splits

A stratified 70 / 15 / 15 split was applied to preserve class ratios across all partitions:

| Split | Samples | % High Quality |
|---|---|---|
| Train | 4,547 | 63.3% |
| Validation | 975 | 63.3% |
| Test | 975 | 63.3% |

The test set is **held out** and will be evaluated once per model in the final report.

---

## 3. Model Training & Baseline Results

### 3.1 Hyperparameter Tuning Protocol

All hyperparameter selection used **5-fold stratified cross-validation on the training set only**, with ROC-AUC as the scoring criterion. The validation set was not used during model selection — it serves as a held-out estimate of generalization.

### 3.2 Logistic Regression (L2 Regularization)

The regularization strength *C* was searched over `{0.001, 0.01, 0.1, 1, 10, 100}`.

| Hyperparameter | Best Value | CV ROC-AUC |
|---|---|---|
| C | 0.1 | 0.7998 |

**Validation set performance:**

| Metric | Value |
|---|---|
| Accuracy | 0.7590 |
| Macro F1-score | 0.7209 |
| ROC-AUC | **0.8247** |

Per-class breakdown: the model achieves precision/recall of 0.74/0.53 on the low-quality class and 0.77/0.89 on the high-quality class. The lower recall on the minority class (low quality) is expected with mild class imbalance and a linear decision boundary.

### 3.3 Support Vector Machine (RBF Kernel)

A grid search was run over `C ∈ {0.1, 1, 10, 100}` and `gamma ∈ {scale, 0.01, 0.1}`.

| Hyperparameter | Best Value | CV ROC-AUC |
|---|---|---|
| C | 1 | 0.8222 |
| gamma | 0.1 | — |

**Validation set performance:**

| Metric | Value |
|---|---|
| Accuracy | 0.7949 |
| Macro F1-score | 0.7685 |
| ROC-AUC | **0.8552** |

The RBF-SVM improves over Logistic Regression by **+3.6 pp in Accuracy**, **+4.8 pp in Macro F1**, and **+3.1 pp in ROC-AUC**, confirming that the quality decision boundary is non-linear. Notably, low-quality recall improves from 0.53 (LR) to 0.62 (SVM), reducing the class-imbalance gap.

### 3.4 Comparison Table (Validation Set)

| Model | Best Hyperparameters | CV ROC-AUC | Val Accuracy | Val Macro F1 | Val ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression (L2) | C = 0.1 | 0.7998 | 0.7590 | 0.7209 | 0.8247 |
| SVM (RBF kernel) | C = 1, γ = 0.1 | 0.8222 | 0.7949 | 0.7685 | 0.8552 |

Both models outperform the majority-class baseline (accuracy = 0.633, AUC = 0.500) by a substantial margin, validating the feature engineering and split design.

---

## 4. Challenges & Mitigations

| Challenge | Status | Planned Mitigation |
|---|---|---|
| Mild class imbalance (63/37) | Addressed | Stratified splits + macro-F1 / AUC metrics as primaries |
| Density multicollinearity | Observed in EDA | Will investigate feature ablation in final report |
| Low recall on minority class (low quality) | Partially addressed by SVM | Random Forest with class_weight="balanced" may further improve |
| SVM training cost at scale | Manageable (4.5 K samples) | No issue; grid search completes in < 2 minutes |

---

## 5. Remaining Work (toward Final Report)

- **Random Forest:** Train with `n_estimators`, `max_depth`, and `class_weight` hyperparameter search; extract feature importances for interpretability analysis.
- **Test set evaluation:** Report all three models on the held-out test set.
- **Ablation study:** Compare pipeline with and without engineered features (so2_ratio, acidity_ratio, wine_type) to quantify their contribution.
- **Bias-variance analysis:** Plot validation error curves and discuss underfitting / overfitting behavior across models.
- **Final report & slides:** Write up complete IEEE/ACM-style paper (max 6 pages) and record 15-minute Zoom presentation.

---

## 6. Figures Generated

All figures are saved in `figures/`:

| Filename | Description |
|---|---|
| `quality_distribution.png` | Bar chart of raw quality score distribution (3–9) |
| `correlation_heatmap.png` | Full 15×15 Pearson correlation matrix |
| `feature_distributions.png` | Alcohol and volatile acidity histograms by class |
| `lr_tuning.png` | LR cross-validation ROC-AUC vs. regularization C |
| `roc_curves_val.png` | ROC curves for LR and SVM on validation set |
| `confusion_matrices.png` | Side-by-side confusion matrices (LR vs. SVM) |
| `lr_coefficients.png` | Signed L2 coefficients for feature interpretability |
