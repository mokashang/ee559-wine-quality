"""
EE559 / CSCI 559 — Spring 2026
Wine Quality Binary Classification Pipeline
Author: Mengjia Shang (7338151449)

Midway deliverables:
  - Download & merge red/white wine datasets
  - EDA (shape, class balance, missing values, correlation)
  - Feature engineering
  - Train/val/test split (70/15/15 stratified)
  - Logistic Regression (L2) with CV hyperparameter tuning
  - SVM (RBF) with CV hyperparameter tuning
  - Report val metrics; hold test set for final report
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # no display needed
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             confusion_matrix, classification_report)
import os, urllib.request

# ── 0. Paths ──────────────────────────────────────────────────────────────────
BASE = "/Users/moka/Documents/EE559/Project"
DATA = os.path.join(BASE, "data")
FIG  = os.path.join(BASE, "figures")
os.makedirs(DATA, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

# ── 1. Download data ──────────────────────────────────────────────────────────
RED_URL   = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
WHITE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"

def fetch(url, dest):
    if not os.path.exists(dest):
        print(f"  Downloading {os.path.basename(dest)} ...")
        urllib.request.urlretrieve(url, dest)
    else:
        print(f"  Found cached {os.path.basename(dest)}")

fetch(RED_URL,   os.path.join(DATA, "winequality-red.csv"))
fetch(WHITE_URL, os.path.join(DATA, "winequality-white.csv"))

red   = pd.read_csv(os.path.join(DATA, "winequality-red.csv"),  sep=";")
white = pd.read_csv(os.path.join(DATA, "winequality-white.csv"), sep=";")
red["wine_type"]   = 0   # red
white["wine_type"] = 1   # white
df = pd.concat([red, white], ignore_index=True)

print("\n=== 1. RAW DATASET ===")
print(f"Shape : {df.shape}")
print(f"Missing values: {df.isnull().sum().sum()}")
print(df.dtypes)
print(df.describe().T.to_string())

# ── 2. Target engineering ──────────────────────────────────────────────────────
df["label"] = (df["quality"] >= 6).astype(int)
print("\n=== 2. CLASS DISTRIBUTION ===")
vc = df["label"].value_counts()
print(f"High quality (>=6): {vc[1]} ({vc[1]/len(df)*100:.1f}%)")
print(f"Low  quality (< 6): {vc[0]} ({vc[0]/len(df)*100:.1f}%)")

# ── 3. Feature engineering ────────────────────────────────────────────────────
# Ratio features
df["so2_ratio"]      = df["free sulfur dioxide"] / (df["total sulfur dioxide"] + 1e-9)
df["acidity_ratio"]  = df["volatile acidity"] / (df["fixed acidity"] + 1e-9)

# Winsorize chlorides and residual sugar at 1st/99th pct
for col in ["chlorides", "residual sugar"]:
    lo, hi = df[col].quantile(0.01), df[col].quantile(0.99)
    df[col] = df[col].clip(lo, hi)

FEATURES = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol", "wine_type", "so2_ratio", "acidity_ratio"
]
X = df[FEATURES].values
y = df["label"].values

print(f"\n=== 3. FEATURE MATRIX ===")
print(f"X shape: {X.shape}, y shape: {y.shape}")

# ── 4. EDA plots ──────────────────────────────────────────────────────────────
# 4a. Quality score distribution
fig, ax = plt.subplots(figsize=(7, 4))
df["quality"].value_counts().sort_index().plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
ax.set_title("Distribution of Wine Quality Scores")
ax.set_xlabel("Quality Score"); ax.set_ylabel("Count")
ax.axvline(x=2.5, color="crimson", linestyle="--", linewidth=1.5, label="Threshold (6)")
plt.tight_layout()
plt.savefig(os.path.join(FIG, "quality_distribution.png"), dpi=150)
plt.close()

# 4b. Correlation heatmap
fig, ax = plt.subplots(figsize=(10, 8))
corr = df[FEATURES + ["label"]].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax,
            annot_kws={"size": 7}, linewidths=0.3)
ax.set_title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig(os.path.join(FIG, "correlation_heatmap.png"), dpi=150)
plt.close()

# 4c. Feature distributions by class (alcohol + volatile acidity)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, feat in zip(axes, ["alcohol", "volatile acidity"]):
    df.groupby("label")[feat].plot(kind="hist", alpha=0.6, bins=30, ax=ax, legend=True)
    ax.set_title(f"{feat} by Quality Class")
    ax.set_xlabel(feat)
    ax.legend(["Low", "High"])
plt.tight_layout()
plt.savefig(os.path.join(FIG, "feature_distributions.png"), dpi=150)
plt.close()

print("  EDA figures saved to figures/")

# ── 5. Train / Val / Test split ───────────────────────────────────────────────
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.15/0.85, stratify=y_trainval, random_state=42)

print(f"\n=== 5. SPLITS ===")
print(f"Train: {X_train.shape[0]}  Val: {X_val.shape[0]}  Test: {X_test.shape[0]}")
for split, ys in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
    print(f"  {split} class balance: {ys.mean()*100:.1f}% high-quality")

# Standardize (fit only on train)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)

# ── 6. Logistic Regression ────────────────────────────────────────────────────
print("\n=== 6. LOGISTIC REGRESSION (L2) ===")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
lr_params = {"C": [0.001, 0.01, 0.1, 1, 10, 100]}
lr_gs = GridSearchCV(
    LogisticRegression(penalty="l2", solver="lbfgs", max_iter=1000, random_state=42),
    lr_params, cv=cv, scoring="roc_auc", n_jobs=-1, verbose=0)
lr_gs.fit(X_train_s, y_train)

best_lr = lr_gs.best_estimator_
print(f"  Best C = {lr_gs.best_params_['C']}  (CV ROC-AUC = {lr_gs.best_score_:.4f})")

# Val metrics
y_val_pred_lr   = best_lr.predict(X_val_s)
y_val_prob_lr   = best_lr.predict_proba(X_val_s)[:, 1]
lr_val_acc  = accuracy_score(y_val, y_val_pred_lr)
lr_val_f1   = f1_score(y_val, y_val_pred_lr, average="macro")
lr_val_auc  = roc_auc_score(y_val, y_val_prob_lr)
print(f"  Val  Accuracy : {lr_val_acc:.4f}")
print(f"  Val  Macro-F1 : {lr_val_f1:.4f}")
print(f"  Val  ROC-AUC  : {lr_val_auc:.4f}")
print(classification_report(y_val, y_val_pred_lr, target_names=["Low", "High"]))

# Learning curve (train accuracy vs val accuracy across C values)
cv_means = lr_gs.cv_results_["mean_test_score"]
C_vals   = lr_params["C"]
fig, ax = plt.subplots(figsize=(6, 4))
ax.semilogx(C_vals, cv_means, marker="o", color="steelblue", label="CV ROC-AUC")
ax.axvline(lr_gs.best_params_["C"], color="crimson", linestyle="--", label=f"Best C={lr_gs.best_params_['C']}")
ax.set_xlabel("Regularization C"); ax.set_ylabel("CV ROC-AUC")
ax.set_title("Logistic Regression: Hyperparameter Tuning")
ax.legend(); plt.tight_layout()
plt.savefig(os.path.join(FIG, "lr_tuning.png"), dpi=150); plt.close()

# ── 7. SVM (RBF) ──────────────────────────────────────────────────────────────
print("\n=== 7. SVM (RBF KERNEL) ===")
svm_params = {"C": [0.1, 1, 10, 100], "gamma": ["scale", 0.01, 0.1]}
svm_gs = GridSearchCV(
    SVC(kernel="rbf", probability=True, random_state=42),
    svm_params, cv=cv, scoring="roc_auc", n_jobs=-1, verbose=0)
svm_gs.fit(X_train_s, y_train)

best_svm = svm_gs.best_estimator_
print(f"  Best C={svm_gs.best_params_['C']}, gamma={svm_gs.best_params_['gamma']}"
      f"  (CV ROC-AUC = {svm_gs.best_score_:.4f})")

y_val_pred_svm  = best_svm.predict(X_val_s)
y_val_prob_svm  = best_svm.predict_proba(X_val_s)[:, 1]
svm_val_acc = accuracy_score(y_val, y_val_pred_svm)
svm_val_f1  = f1_score(y_val, y_val_pred_svm, average="macro")
svm_val_auc = roc_auc_score(y_val, y_val_prob_svm)
print(f"  Val  Accuracy : {svm_val_acc:.4f}")
print(f"  Val  Macro-F1 : {svm_val_f1:.4f}")
print(f"  Val  ROC-AUC  : {svm_val_auc:.4f}")
print(classification_report(y_val, y_val_pred_svm, target_names=["Low", "High"]))

# ── 8. ROC curves (val) ───────────────────────────────────────────────────────
from sklearn.metrics import RocCurveDisplay
fig, ax = plt.subplots(figsize=(6, 5))
RocCurveDisplay.from_predictions(y_val, y_val_prob_lr,  name=f"LR (AUC={lr_val_auc:.3f})",  ax=ax)
RocCurveDisplay.from_predictions(y_val, y_val_prob_svm, name=f"SVM (AUC={svm_val_auc:.3f})", ax=ax)
ax.plot([0,1],[0,1],"k--", linewidth=0.8)
ax.set_title("ROC Curves — Validation Set"); plt.tight_layout()
plt.savefig(os.path.join(FIG, "roc_curves_val.png"), dpi=150); plt.close()

# ── 9. Confusion matrices ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, pred, title in zip(axes,
                            [y_val_pred_lr, y_val_pred_svm],
                            ["Logistic Regression", "SVM (RBF)"]):
    cm = confusion_matrix(y_val, pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Low","High"], yticklabels=["Low","High"])
    ax.set_title(f"{title}\nValidation Confusion Matrix")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(FIG, "confusion_matrices.png"), dpi=150); plt.close()

# ── 10. Feature importance (LR coefficients) ──────────────────────────────────
coef = pd.Series(best_lr.coef_[0], index=FEATURES).sort_values()
fig, ax = plt.subplots(figsize=(7, 5))
coef.plot(kind="barh", ax=ax, color=["crimson" if v < 0 else "steelblue" for v in coef])
ax.set_title("Logistic Regression Coefficients (L2)"); ax.set_xlabel("Coefficient value")
plt.tight_layout()
plt.savefig(os.path.join(FIG, "lr_coefficients.png"), dpi=150); plt.close()

# ── 11. Summary table ─────────────────────────────────────────────────────────
print("\n=== SUMMARY (Validation Set) ===")
summary = pd.DataFrame({
    "Model":    ["Logistic Regression (L2)", "SVM (RBF)"],
    "Best HP":  [f"C={lr_gs.best_params_['C']}",
                 f"C={svm_gs.best_params_['C']}, γ={svm_gs.best_params_['gamma']}"],
    "CV AUC":   [f"{lr_gs.best_score_:.4f}", f"{svm_gs.best_score_:.4f}"],
    "Val Acc":  [f"{lr_val_acc:.4f}", f"{svm_val_acc:.4f}"],
    "Val F1":   [f"{lr_val_f1:.4f}", f"{svm_val_f1:.4f}"],
    "Val AUC":  [f"{lr_val_auc:.4f}", f"{svm_val_auc:.4f}"],
})
print(summary.to_string(index=False))

print("\nAll figures saved to:", FIG)
print("Pipeline complete.")
