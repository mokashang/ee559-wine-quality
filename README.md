# Wine Quality Binary Classification

**Course:** EE559 / CSCI 559 — Machine Learning, Spring 2026
**Author:** Mengjia Shang (USC ID 7338151449)
**Instructor:** Prof. Anand A. Joshi

End-to-end supervised learning pipeline that predicts wine quality (binary: high ≥ 6 / low < 6) from 11 physicochemical measurements using the [UCI Wine Quality Dataset](https://archive.ics.uci.edu/dataset/186/wine+quality). Three classifiers — Logistic Regression, SVM (RBF), and Random Forest — are compared under a strict no-leakage protocol with class-balance-aware tuning, bootstrap CIs, McNemar's test, and a multi-seed robustness study.

**Headline result:** Random Forest — Test ROC-AUC **0.904 [95% CI 0.883, 0.922]**, PR-AUC **0.938**, Brier **0.124**.

---

## Repository contents

```
.
├── final_pipeline.py     # Reproducible pipeline (single entry point)
├── requirements.txt      # Pinned Python dependency ranges
├── final_report.pdf      # IEEE conference format final report (5 pages)
├── README.md             # This file
└── .gitignore
```

The pipeline downloads the dataset, performs preprocessing and feature engineering, runs hyperparameter tuning for all three models, and writes figures and metrics to disk.

---

## Reproducing the results

### Requirements

```bash
pip install -r requirements.txt
```

Pinned ranges:
```
numpy>=1.24,<3.0
pandas>=2.0,<3.0
scikit-learn>=1.3,<1.7
scipy>=1.10,<2.0
matplotlib>=3.7,<4.0
seaborn>=0.12,<0.14
```

No GPU required.

### Run

```bash
python3 final_pipeline.py
```

This single command:
1. Downloads `winequality-red.csv` and `winequality-white.csv` from UCI (cached in `data/`)
2. Runs preprocessing, feature engineering, and VIF analysis
3. Creates stratified 70/15/15 splits (seed 42)
4. Tunes all three models via 5-fold stratified CV — including `class_weight ∈ {None, "balanced"}` for all three
5. Computes test metrics with **95% bootstrap CIs**, **PR-AUC**, and **Brier scores**
6. Runs **McNemar's test** for pairwise model comparisons
7. Runs **multi-seed robustness study** (5 stratified splits)
8. Runs **all-model ablation**: base (11) vs engineered (14) features for each model
9. Runs **threshold tuning** for Logistic Regression
10. Runs **density-drop experiment** to confirm the VIF finding
11. Saves figures to `figures/` and dumps all numbers to `results.json`

Expected runtime: 6–12 minutes on a modern laptop.

---

## Final test set results (seed = 42)

| Model | Acc | Macro F1 | ROC-AUC [95% CI] | PR-AUC | Brier ↓ |
|---|---|---|---|---|---|
| Majority Baseline | 0.633 | 0.387 | 0.500 | 0.633 | 0.232 |
| Logistic Regression (L2, balanced) | 0.734 | 0.725 | 0.816 [0.79, 0.84] | 0.892 | 0.177 |
| SVM (RBF, balanced) | 0.776 | 0.768 | 0.850 [0.83, 0.87] | 0.905 | 0.151 |
| **Random Forest (n=300)** | **0.829** | **0.811** | **0.904 [0.88, 0.92]** | **0.938** | **0.124** |

McNemar's pairwise tests give p < 0.001 for all three model pairs. Five-seed mean ± std AUCs: LR 0.801 ± 0.013, SVM 0.835 ± 0.008, RF 0.898 ± 0.013 (non-overlapping ±2σ ranges).

---

## Citation

P. Cortez, A. Cerdeira, F. Almeida, T. Matos, and J. Reis, "Modeling wine preferences by data mining from physicochemical properties," *Decision Support Systems*, vol. 47, no. 4, pp. 547–553, 2009.
