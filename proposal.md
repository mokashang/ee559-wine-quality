# Predicting Wine Quality from Physicochemical Properties: A Comparative Study of Supervised Learning Methods

**Student:** Mengjia Shang
**Student ID:** 7338151449
**Course:** EE559 / CSCI 559 — Machine Learning, Spring 2026
**Instructor:** Prof. Anand A. Joshi

---

## 1. Problem Motivation & Objective

Wine quality certification is traditionally performed by human expert panels — a process that is expensive, time-consuming, and prone to subjective variation. An automated, data-driven quality classifier could assist producers and certifiers by providing fast, reproducible quality estimates based purely on measurable chemical composition.

This is a **binary classification** problem. The prediction target is a binary wine quality label derived from the original 0–10 expert score: wines scoring **6 or above** are labeled *high quality* (1), and wines scoring **below 6** are labeled *low quality* (0). This framing is practically motivated (a pass/fail quality gate) and creates a class distribution that is moderately imbalanced, making metric selection non-trivial.

---

## 2. Dataset Description

| Property | Details |
|---|---|
| **Name** | Wine Quality Dataset |
| **Source** | UCI Machine Learning Repository |
| **URL** | https://archive.ics.uci.edu/dataset/186/wine+quality |
| **Samples** | 6,497 total (1,599 red + 4,898 white) |
| **Features** | 11 physicochemical inputs (fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free/total sulfur dioxide, density, pH, sulphates, alcohol) |
| **Target distribution** | ~57% high quality, ~43% low quality (after binarization) |

**Known challenges:**
- Slight class imbalance requiring explicit handling (e.g., stratified splits, F1/AUC evaluation)
- Outliers in several features (residual sugar, sulfur dioxide, chlorides) that must be addressed
- Wine type (red vs. white) is not provided as a feature and must be engineered from the combined file
- Multicollinearity between acidity-related features warrants correlation analysis

---

## 3. Proposed Methods

Three supervised learning algorithms will be implemented, tuned, and compared:

**1. Logistic Regression (with L2 regularization)**
Serves as the interpretable linear baseline. Logistic regression is appropriate here because it produces calibrated probability estimates, handles the moderate feature count (≤15 after engineering) efficiently, and provides coefficient-level interpretability for understanding which chemical properties most influence quality. The regularization strength *C* will be tuned via cross-validation.

**2. Support Vector Machine (RBF kernel)**
Chemical feature interactions (e.g., the interplay of alcohol content and acidity) are unlikely to be linearly separable. An SVM with an RBF kernel can capture these non-linear boundaries without requiring explicit interaction terms. Hyperparameters *C* and *γ* will be searched via grid search with cross-validation.

**3. Random Forest**
A tree ensemble is well-suited to this dataset because it naturally handles mixed feature scales (no normalization required), captures high-order feature interactions, and provides feature importance rankings that can validate domain knowledge (e.g., alcohol is widely reported as the strongest quality predictor). The number of trees and maximum depth will be tuned.

---

## 4. Evaluation Plan

**Metrics:**
- **ROC-AUC** — primary metric, robust to the mild class imbalance and measures discriminative power across all thresholds
- **F1-score (macro)** — penalizes poor recall on the minority class equally with the majority class
- **Accuracy** — reported for completeness and comparability

**Data splits:**
The 6,497 samples will be partitioned using a **70 / 15 / 15** stratified split:
- Training set (~4,548 samples): model fitting and feature selection
- Validation set (~974 samples): hyperparameter tuning and model selection
- Test set (~975 samples): held out until final evaluation; reported once per model

Stratification ensures consistent class ratios across all splits. Cross-validation (5-fold) will be used within the training set during hyperparameter search.

---

## 5. Feature Engineering Plan

- Encode wine type as a binary indicator feature (red = 0, white = 1)
- Construct ratio features: free-to-total sulfur dioxide ratio, volatile-to-fixed acidity ratio
- Winsorize outliers at the 1st/99th percentile for chlorides and residual sugar
- Standardize all continuous features (zero mean, unit variance) for Logistic Regression and SVM; left unscaled for Random Forest

---

## 6. Timeline

| Milestone | Date | Planned Deliverables |
|---|---|---|
| **Proposal** | 03/30/2026 | This document; dataset downloaded and validated |
| **Midway Report** | 04/13/2026 | EDA complete; preprocessing and feature engineering finalized; Logistic Regression and SVM trained and validated; preliminary results table |
| **Presentation + Final Submission** | 05/04/2026 | Random Forest trained; all three models compared on test set; ablation study on feature engineering impact; final report and slides |
