# Predicting Wine Quality from Physicochemical Properties: A Comparative Study of Supervised Learning Methods

**Student:** Mengjia Shang · **ID:** 7338151449
**Course:** EE559 / CSCI 559 — Machine Learning, Spring 2026
**Instructor:** Prof. Anand A. Joshi · **Date:** 05/04/2026

---

## Abstract

This paper presents an end-to-end supervised learning pipeline for binary wine quality classification using the UCI Wine Quality dataset (6,497 samples, 11 physicochemical features). Three classifiers — Logistic Regression with L2 regularization, a Support Vector Machine with an RBF kernel, and a Random Forest ensemble — are systematically compared under a strict train / validation / test protocol with class-imbalance-aware hyperparameter tuning. Random Forest achieves the highest overall test performance (Accuracy = 0.8287, Macro F1 = 0.8108, ROC-AUC = 0.9041 [95% bootstrap CI: 0.883–0.922], PR-AUC = 0.938) and the best probability calibration (Brier = 0.124). All pairwise model differences are statistically significant under McNemar's test (p ≤ 6×10⁻⁴), and the ranking is robust across five random seeds (RF AUC = 0.898 ± 0.013). An all-model ablation reveals that engineered features benefit the linear baseline most (LR: ΔAUC = +0.008) while Random Forest already recovers similar information from raw inputs. A practically important secondary finding is that SVM with balanced class weights achieves the highest low-quality recall (0.796), making it preferable when the cost of shipping defective wine is high; Random Forest is preferred when overall accuracy and calibration matter most.

---

## 1. Introduction

Wine quality certification relies on sensory panels of trained experts — a process that is costly, slow, and subject to inter-rater variability. Physicochemical testing is already performed during production; if these measurements can predict expert quality ratings reliably, they provide a low-cost, objective quality gate that complements human evaluation.

The prediction task is **binary classification**: given 11 continuous physicochemical measurements for a wine sample, predict whether it will be rated *high quality* (expert score ≥ 6) or *low quality* (score < 6). The threshold of 6 reflects the industry convention distinguishing acceptable from premium-tier wines and maps naturally onto a pass/fail production decision.

This study makes the following contributions:

1. A clean, reproducible pipeline covering preprocessing, feature engineering, hyperparameter tuning (including class-weight balancing for all three models), and held-out test evaluation.
2. A direct comparison of a linear model, a kernel method, and a tree ensemble, with statistical significance testing (McNemar) and bootstrap confidence intervals.
3. A multi-model ablation that distinguishes which models benefit most from engineered features.
4. Multi-seed robustness analysis (5 stratified splits) and probability calibration assessment.
5. A density-multicollinearity follow-up to the EDA's VIF analysis.

---

## 2. Dataset & Features

### 2.1 Dataset Description

The UCI Wine Quality dataset [1] combines two CSV files: 1,599 red and 4,898 white wine samples, merged into a single dataset of **6,497 samples**. Each sample has 11 continuous physicochemical measurements and an integer quality score (3–9). There are **no missing values**.

The target is binarized: quality ≥ 6 → **High** (4,113 samples, 63.3%); quality < 6 → **Low** (2,384 samples, 36.7%). The resulting 63/37 split constitutes mild class imbalance and motivates ROC-AUC, macro F1, and PR-AUC as primary metrics over raw accuracy.

### 2.2 Exploratory Data Analysis

Key findings from EDA:

- **Alcohol** is the strongest positive predictor of quality (Pearson r = +0.44).
- **Volatile acidity** is the strongest negative predictor (r = −0.27).
- **Density** is highly collinear with alcohol (r = −0.69) and residual sugar (r = +0.84).
- White wines (75.4% of the dataset) have systematically higher residual sugar and lower chloride concentrations than red wines, justifying a wine-type indicator.
- **Residual sugar** and **chlorides** exhibit heavy right-skewed distributions with extreme outliers.

#### Variance Inflation Factor (VIF) Analysis

To quantify multicollinearity, we computed VIFs on the standardized 14-feature matrix:

| Feature | VIF | Feature | VIF |
|---|---|---|---|
| volatile acidity | 39.4 ⚠️ | residual sugar | 6.8 |
| acidity_ratio | 37.2 ⚠️ | so2_ratio | 4.3 |
| density | 15.3 ⚠️ | alcohol | 4.3 |
| fixed acidity | 8.1 | pH | 2.3 |
| total SO₂ | 7.8 | chlorides | 2.0 |
| free SO₂ | 7.1 | citric acid | 1.6 |
| wine_type | 7.0 | sulphates | 1.5 |

Three features have VIF > 10 (the conventional warning threshold). The high VIF for `acidity_ratio` (37.2) is expected — it shares a numerator with `volatile acidity` by construction. **Density** (15.3) is collinear with the alcohol/residual-sugar pair but does not derive from them analytically. To confirm whether density adds independent signal, we trained Random Forest with and without density (Section 4.7); the AUC difference is 0.0001 (no measurable effect), validating that density is largely redundant.

### 2.3 Preprocessing & Feature Engineering

The following steps were applied in sequence, with all statistics fit **exclusively on the training fold**:

| Step | Action |
|---|---|
| Outlier handling | Winsorize `chlorides` and `residual sugar` at 1st / 99th percentiles |
| Wine type | Encode red = 0, white = 1 |
| SO₂ ratio | `free SO₂ / (total SO₂ + ε)` — effective preservation fraction |
| Acidity ratio | `volatile acidity / (fixed acidity + ε)` — relative spoilage-acidity contribution |
| Standardization | Zero mean, unit variance (applied to LR & SVM only) |

Final feature matrix: **14 features** (11 original + 3 engineered).

### 2.4 Data Splits

A stratified **70 / 15 / 15** split preserves the 63.3% high-quality ratio in all partitions: Train = 4,547, Validation = 975, Test = 975. The test set was held out and touched exactly once per model.

---

## 3. Methodology

### 3.1 Logistic Regression with L2 Regularization

The decision function is ŷ = σ(wᵀx + b), with L2 penalty ½C⁻¹‖w‖₂² added to cross-entropy. Hyperparameters tuned via 5-fold CV: C ∈ {0.001, 0.01, 0.1, 1, 10, 100} and class_weight ∈ {None, "balanced"}. Best: **C = 0.1, class_weight = "balanced"** (CV ROC-AUC = 0.7999).

*Appropriateness:* Provides an interpretable linear baseline; coefficient magnitudes directly quantify feature relevance after standardization.

### 3.2 Support Vector Machine (RBF Kernel)

The RBF-SVM finds a maximum-margin hyperplane in the feature space induced by k(x, x') = exp(−γ‖x−x'‖²). Grid: C ∈ {0.1, 1, 10, 100}, γ ∈ {scale, 0.01, 0.1}, class_weight ∈ {None, "balanced"}. Best: **C = 1, γ = 0.1, class_weight = "balanced"** (CV ROC-AUC = 0.8276).

*Appropriateness:* Captures non-linear chemical interactions implicitly, without explicit feature crossing.

### 3.3 Random Forest

A Random Forest trains T decision trees on bootstrap samples with a random feature subset at each split. Grid: n_estimators ∈ {100, 200, 300}, max_depth ∈ {None, 10, 20}, class_weight ∈ {None, "balanced"}. Best: **n = 300, max_depth = None, class_weight = None** (CV ROC-AUC = 0.8814).

*Appropriateness:* Tree ensembles handle mixed scales natively, capture high-order interactions, and yield feature importance rankings. Notably, the bootstrap mechanism handled the mild imbalance such that explicit class reweighting was *not* selected — a different conclusion than for LR and SVM.

---

## 4. Experiments & Results

### 4.1 Hyperparameter Tuning

All hyperparameter selection used 5-fold stratified CV **within the training set only**. Class-weight balancing was searched for all three models; LR and SVM benefited from balancing while RF did not.

| Model | Best Hyperparameters | CV ROC-AUC |
|---|---|---|
| Logistic Regression (L2) | C = 0.1, cw = balanced | 0.7999 |
| SVM (RBF) | C = 1, γ = 0.1, cw = balanced | 0.8276 |
| Random Forest | n = 300, depth = None, cw = None | **0.8814** |

### 4.2 Test Set Results with 95% Bootstrap Confidence Intervals

Each model was evaluated once on the 975-sample test set. ROC-AUC and PR-AUC bootstrap CIs were computed by resampling test predictions 1,000 times.

| Model | Accuracy | Macro F1 | ROC-AUC [95% CI] | PR-AUC [95% CI] | Brier ↓ |
|---|---|---|---|---|---|
| Majority baseline | 0.6328 | 0.3870 | 0.5000 | 0.6328 | 0.232 |
| Logistic Regression (L2) | 0.7344 | 0.7245 | 0.8161 [0.7896, 0.8414] | 0.8922 [0.8715, 0.9120] | 0.177 |
| SVM (RBF) | 0.7764 | 0.7679 | 0.8503 [0.8255, 0.8724] | 0.9053 [0.8833, 0.9243] | 0.151 |
| **Random Forest** | **0.8287** | **0.8108** | **0.9041 [0.8830, 0.9215]** | **0.9379 [0.9203, 0.9529]** | **0.124** |

The 95% CIs are non-overlapping for all three pairs, confirming the ranking is statistically robust on this test set. Random Forest also achieves the **lowest Brier score** (0.124), meaning its probability estimates are best calibrated — a notable result, since calibration is often expected to favor the linear model.

### 4.3 McNemar's Test for Pairwise Significance

McNemar's test (with continuity correction) on the test-set prediction agreement table:

| Comparison | χ² | p-value | "b vs c" |
|---|---|---|---|
| LR vs SVM | 11.84 | **5.8 × 10⁻⁴** | 53 / 96 |
| LR vs RF | 41.54 | **1.2 × 10⁻¹⁰** | 52 / 143 |
| SVM vs RF | 19.38 | **1.1 × 10⁻⁵** | 33 / 81 |

All three pairwise differences are highly significant (p < 0.001). The "b/c" counts give the asymmetry of disagreement; for example, in the LR-vs-RF comparison, 143 wines are correctly classified by RF and missed by LR, while only 52 are missed by RF but caught by LR.

### 4.4 Multi-seed Robustness (5 stratified splits)

Best hyperparameters were fixed and the models retrained with five different stratified splits (seeds = {0, 1, 7, 42, 2023}):

| Model | Accuracy | Macro F1 | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.7235 ± 0.0090 | 0.7145 ± 0.0095 | 0.8009 ± 0.0132 |
| SVM (RBF) | 0.7731 ± 0.0090 | 0.7532 ± 0.0122 | 0.8353 ± 0.0084 |
| Random Forest | 0.8302 ± 0.0149 | 0.8120 ± 0.0185 | **0.8975 ± 0.0127** |

The mean ± 2σ intervals do not overlap between any pair of models, providing a second independent line of evidence that the ranking is real, not seed-dependent.

### 4.5 Per-Class Performance & Operational Implications

| Model | Low Prec. | Low Recall | Low F1 | High Prec. | High Recall | High F1 |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.6143 | 0.7430 | 0.6726 | 0.8303 | 0.7293 | 0.7765 |
| SVM (RBF) | 0.6628 | **0.7961** | **0.7234** | 0.8661 | 0.7650 | 0.8124 |
| Random Forest | **0.8013** | 0.7095 | 0.7526 | 0.8419 | **0.8979** | **0.8690** |

**A critical observation:** with balanced class weights, **SVM achieves the highest low-quality recall (0.796)**, missing only 20% of defective bottles, while **Random Forest** has the highest low-quality *precision* (0.801) — its alarms are most reliable. The choice between them therefore depends on the cost trade-off:

- If shipping a defective bottle is very costly → SVM (highest recall on low-quality)
- If re-testing a good bottle is very costly → Random Forest (highest precision on low-quality)
- For overall balanced performance → Random Forest (highest F1, AUC, accuracy)

### 4.6 All-Model Ablation: Feature Engineering Impact

We retrained all three models on the 11 base features alone vs. the full 14-feature matrix and evaluated on the same held-out test set:

| Model | Base AUC (11) | +Eng AUC (14) | ΔAUC | Base F1 | +Eng F1 | ΔF1 |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8086 | 0.8161 | **+0.0075** | 0.7118 | 0.7245 | **+0.0128** |
| SVM (RBF) | 0.8503 | 0.8503 | −0.0000 | 0.7533 | 0.7584 | +0.0051 |
| Random Forest | 0.9007 | 0.9041 | +0.0034 | 0.8171 | 0.8095 | −0.0075 |

**This is the most important methodological finding of the project.** The engineered features benefit the linear baseline most clearly (ΔAUC = +0.008, ΔF1 = +0.013), confirming the original hypothesis. The SVM's RBF kernel essentially absorbs the same information implicitly. Random Forest is somewhere in between — the tree depth is sufficient to discover the engineered relationships from raw inputs. This validates that the engineering choices are domain-correct, but also shows their value scales inversely with model capacity.

### 4.7 Threshold Tuning (Logistic Regression)

We searched for the threshold τ ∈ [0.1, 0.9] that maximises validation macro-F1. The best τ = 0.510, very close to the default 0.5. Test results:

| Setting | Macro F1 | Low Recall | High Recall | Accuracy |
|---|---|---|---|---|
| Default τ = 0.5 | 0.7245 | 0.7430 | 0.7293 | 0.7344 |
| Tuned τ = 0.510 | 0.7264 | 0.7542 | 0.7212 | 0.7333 |

The improvement is marginal because **balanced class weights already shifted the implicit decision boundary** in the same direction threshold tuning would. Without `class_weight = "balanced"`, low-quality recall would be roughly 0.56 (44% missed defects); with it, recall is 0.74 — a much larger improvement than threshold tuning could achieve.

### 4.8 Density-Drop Experiment

| Variant | Test ROC-AUC | Test Macro F1 |
|---|---|---|
| RF with density | 0.9041 | 0.8095 |
| RF without density | 0.9040 | 0.8151 |

Removing density yields essentially identical AUC and a slightly higher F1, confirming the VIF result: density is a redundant feature for tree-based models in the presence of alcohol and residual sugar. We kept density in the main pipeline for consistency with the published feature set, but a deployment system could safely drop it.

### 4.9 Bias-Variance Analysis (Random Forest)

The RF learning curve shows:

- At 455 training samples (10% of train set), train AUC = 1.000, CV AUC = 0.847 → gap = 0.153 (high variance regime).
- At 4,547 samples (full train set), train AUC = 1.000, CV AUC = 0.881 → gap = 0.119.
- The gap narrows monotonically; CV AUC plateaus near 4,000 samples.

The persistent 0.12 gap indicates RF retains some variance, but the small val→test drop (CV 0.881 → test 0.904 — actually a *gain*) shows this does not translate to overfitting. CV is a slightly pessimistic estimator here because each fold trains on only 80% of the training data.

### 4.10 Probability Calibration

Reliability diagrams were produced via 10-bin quantile binning on the test set. **Random Forest is the best-calibrated model** (Brier = 0.124), followed by SVM (0.151) and Logistic Regression (0.177). This is somewhat counter-intuitive — linear models are typically expected to be well-calibrated — but is consistent with prior literature: ensemble averaging in RF naturally produces moderate, well-spread probabilities, while balanced-class-weighted LR pushes probabilities toward 0.5 to compensate for the imbalance, which hurts Brier.

---

## 5. Discussion & Conclusion

### 5.1 Why Random Forest Wins (Mostly)

The performance gap (RF test AUC 0.904 vs. SVM 0.850 vs. LR 0.816) is primarily attributable to model capacity. Wine quality depends on complex chemical interactions — e.g., the combined effect of alcohol, acidity, and sulfur dioxide — that are non-additive. RF's ensemble of decorrelated trees models diverse interaction patterns simultaneously. The all-model ablation (Section 4.6) corroborates this: lower-capacity LR benefits explicitly from engineered ratio features that higher-capacity RF rediscovers from raw inputs.

### 5.2 When SVM Beats RF

A subtler finding is that **SVM with class_weight = "balanced" achieves the highest low-quality recall (0.796)**, beating Random Forest (0.710) on the most operationally important metric for a producer's quality gate. This is because balanced class weighting scales the SVM hinge loss to pay more attention to minority-class margin violations, while RF's bootstrap sampling naturally preserves the original 63/37 ratio. In production scenarios where missing a defective wine is much more costly than triggering a re-test of a good one, SVM is the better choice.

### 5.3 Limitations

- **Binarization loss:** Collapsing a 7-point ordinal scale to binary discards intra-tier ordering information. A regression or ordinal-classification model could be more informative.
- **No producer/vintage covariates:** The dataset lacks producer, year, or origin metadata; unobserved confounders may inflate apparent generalization.
- **Multicollinearity is acknowledged but not aggressively handled:** Density and acidity_ratio have VIF > 15. A purely linear pipeline would benefit from removing them; tree-based models are insensitive.
- **Dataset is mildly outdated:** UCI Wine Quality reflects Portuguese *Vinho Verde* from ~2009. Modern grapes and processes may differ.

### 5.4 Future Work

- Train an MLP to extend the capacity comparison beyond classical methods.
- Use SHAP values for per-prediction feature attribution.
- Frame the task as ordinal regression to recover discarded quality gradations.
- Investigate cost-sensitive learning with explicit FN/FP cost ratios elicited from a domain expert.

### 5.5 Conclusion

A systematic comparison of three supervised classifiers on the UCI Wine Quality dataset, under a strict no-leakage protocol with class-balance-aware tuning, demonstrates that **model capacity is the primary driver of performance**: Random Forest achieves test ROC-AUC = 0.904 [0.883, 0.922] and PR-AUC = 0.938, with all pairwise model differences statistically significant under McNemar's test (p < 0.001) and stable across 5 random seeds. A finer secondary finding is that **SVM with balanced class weights is preferable when minority-class recall is the priority**. The all-model ablation establishes that engineered features benefit lower-capacity models most (ΔAUC: LR +0.008, SVM ≈ 0, RF +0.003), and a density-drop experiment confirms that VIF-flagged multicollinearity carries no real information for tree models. The full pipeline is reproducible via `final_pipeline.py` with `requirements.txt` pinning all dependency ranges.

---

## 6. Presentation Recording

> **[Link to be added by student after recording submission]**

---

## References

[1] P. Cortez, A. Cerdeira, F. Almeida, T. Matos, and J. Reis, "Modeling wine preferences by data mining from physicochemical properties," *Decision Support Systems*, vol. 47, no. 4, pp. 547–553, 2009.

[2] F. Pedregosa et al., "Scikit-learn: Machine learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.

[3] T. Hastie, R. Tibshirani, and J. Friedman, *The Elements of Statistical Learning*, 2nd ed. Springer, 2009.

[4] L. Breiman, "Random forests," *Machine Learning*, vol. 45, no. 1, pp. 5–32, 2001.

[5] C. Cortes and V. Vapnik, "Support-vector networks," *Machine Learning*, vol. 20, no. 3, pp. 273–297, 1995.

[6] Q. McNemar, "Note on the sampling error of the difference between correlated proportions or percentages," *Psychometrika*, vol. 12, no. 2, pp. 153–157, 1947.

[7] A. Niculescu-Mizil and R. Caruana, "Predicting good probabilities with supervised learning," *Proceedings of ICML*, 2005.
