"""
EE559 / CSCI 559 — Spring 2026
Wine Quality Binary Classification — FINAL Pipeline (v2)
Author: Mengjia Shang (7338151449)

This v2 pipeline incorporates rigor improvements over v1:
  (1)  Hard-coded paths replaced with file-relative paths
  (2)  Narrowed warnings filter (only sklearn parallel UserWarning)
  (3)  PR-AUC reported alongside ROC-AUC (better for imbalanced data)
  (4)  class_weight ∈ {None, "balanced"} now tuned for LR & SVM (not just RF)
  (5)  All-model ablation: LR, SVM, and RF tested base vs engineered features
  (6)  Multi-seed robustness: 5 random seeds → mean ± std test metrics
  (7)  Bootstrap 95% confidence intervals on all test metrics
  (8)  McNemar's test for pairwise model comparison
  (9)  Threshold tuning for low-quality recall on validation set
  (10) Calibration: Brier score + reliability curves
  (11) Density multicollinearity experiment (drop density, compare RF)
  (12) LR coefficient figure regenerated in this pipeline
  (13) Results dumped to results.json for downstream consumption
"""

import warnings, os, json, urllib.request
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.parallel")
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     GridSearchCV, learning_curve)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             average_precision_score, recall_score,
                             confusion_matrix, classification_report,
                             RocCurveDisplay, brier_score_loss)
from sklearn.calibration import calibration_curve

# ── 0. Paths ──────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
FIG  = os.path.join(BASE, "figures")
os.makedirs(DATA, exist_ok=True)
os.makedirs(FIG,  exist_ok=True)

# ── 1. Download & load data ───────────────────────────────────────────────────
RED_URL   = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
WHITE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"

def fetch(url, dest):
    if not os.path.exists(dest):
        print(f"  Downloading {os.path.basename(dest)}")
        urllib.request.urlretrieve(url, dest)

fetch(RED_URL,   os.path.join(DATA, "winequality-red.csv"))
fetch(WHITE_URL, os.path.join(DATA, "winequality-white.csv"))

red   = pd.read_csv(os.path.join(DATA, "winequality-red.csv"),  sep=";")
white = pd.read_csv(os.path.join(DATA, "winequality-white.csv"), sep=";")
red["wine_type"]   = 0
white["wine_type"] = 1
df = pd.concat([red, white], ignore_index=True)
print(f"Dataset loaded: {len(df)} samples")

# ── 2. Target & feature engineering ───────────────────────────────────────────
df["label"] = (df["quality"] >= 6).astype(int)

for col in ["chlorides", "residual sugar"]:
    lo, hi = df[col].quantile(0.01), df[col].quantile(0.99)
    df[col] = df[col].clip(lo, hi)

df["so2_ratio"]     = df["free sulfur dioxide"] / (df["total sulfur dioxide"] + 1e-9)
df["acidity_ratio"] = df["volatile acidity"]    / (df["fixed acidity"] + 1e-9)

BASE_FEATURES = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol"
]
ENG_FEATURES = BASE_FEATURES + ["wine_type", "so2_ratio", "acidity_ratio"]

X_base = df[BASE_FEATURES].values
X_eng  = df[ENG_FEATURES].values
y      = df["label"].values

# ── 3. Density multicollinearity diagnostic (VIF) ─────────────────────────────
def compute_vif(X, names):
    """Compute Variance Inflation Factor for each feature."""
    from numpy.linalg import inv
    Xc = (X - X.mean(0)) / X.std(0)
    R = np.corrcoef(Xc, rowvar=False)
    try:
        Rinv = inv(R)
        vifs = np.diag(Rinv)
    except np.linalg.LinAlgError:
        vifs = np.full(X.shape[1], np.nan)
    return dict(zip(names, vifs))

vif_dict = compute_vif(X_eng, ENG_FEATURES)
print("\n=== VIF (Variance Inflation Factor) ===")
for f, v in sorted(vif_dict.items(), key=lambda kv: -kv[1]):
    print(f"  {f:<24} VIF = {v:6.2f}{' ⚠️' if v>10 else ''}")

# ── 4. Splits helper ──────────────────────────────────────────────────────────
def make_splits(X, y, seed=42):
    X_tv, X_te, y_tv, y_te = train_test_split(X, y, test_size=0.15,
                                               stratify=y, random_state=seed)
    X_tr, X_va, y_tr, y_va = train_test_split(X_tv, y_tv,
                                               test_size=0.15/0.85,
                                               stratify=y_tv, random_state=seed)
    sc = StandardScaler().fit(X_tr)
    return (sc.transform(X_tr), sc.transform(X_va), sc.transform(X_te),
            y_tr, y_va, y_te, sc)

Xtr, Xva, Xte, ytr, yva, yte, scaler = make_splits(X_eng, y)
print(f"\nSplits  train={len(ytr)}  val={len(yva)}  test={len(yte)}")

# ── 5. CV object ──────────────────────────────────────────────────────────────
cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ── 6. Logistic Regression — now with class_weight tuning ─────────────────────
print("\n=== Tuning LR (with class_weight) ===")
lr_gs = GridSearchCV(
    LogisticRegression(penalty="l2", solver="lbfgs", max_iter=2000, random_state=42),
    {"C": [0.001, 0.01, 0.1, 1, 10, 100],
     "class_weight": [None, "balanced"]},
    cv=cv5, scoring="roc_auc", n_jobs=-1)
lr_gs.fit(Xtr, ytr)
best_lr = lr_gs.best_estimator_
print(f"  best C={lr_gs.best_params_['C']}  cw={lr_gs.best_params_['class_weight']}  "
      f"CV-AUC={lr_gs.best_score_:.4f}")

# ── 7. SVM — now with class_weight tuning ─────────────────────────────────────
print("\n=== Tuning SVM (with class_weight) ===")
svm_gs = GridSearchCV(
    SVC(kernel="rbf", probability=True, random_state=42),
    {"C": [0.1, 1, 10, 100],
     "gamma": ["scale", 0.01, 0.1],
     "class_weight": [None, "balanced"]},
    cv=cv5, scoring="roc_auc", n_jobs=-1)
svm_gs.fit(Xtr, ytr)
best_svm = svm_gs.best_estimator_
print(f"  best C={svm_gs.best_params_['C']}  γ={svm_gs.best_params_['gamma']}  "
      f"cw={svm_gs.best_params_['class_weight']}  CV-AUC={svm_gs.best_score_:.4f}")

# ── 8. Random Forest ──────────────────────────────────────────────────────────
print("\n=== Tuning Random Forest ===")
rf_gs = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    {"n_estimators": [100, 200, 300],
     "max_depth":    [None, 10, 20],
     "class_weight": [None, "balanced"]},
    cv=cv5, scoring="roc_auc", n_jobs=-1)
rf_gs.fit(Xtr, ytr)
best_rf = rf_gs.best_estimator_
print(f"  best n={rf_gs.best_params_['n_estimators']}  d={rf_gs.best_params_['max_depth']}  "
      f"cw={rf_gs.best_params_['class_weight']}  CV-AUC={rf_gs.best_score_:.4f}")

# ── 9. Metrics helpers ────────────────────────────────────────────────────────
def metrics(model, X, y, threshold=0.5):
    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= threshold).astype(int)
    return {
        "acc":     float(accuracy_score(y, pred)),
        "f1":      float(f1_score(y, pred, average="macro")),
        "auc":     float(roc_auc_score(y, prob)),
        "pr_auc":  float(average_precision_score(y, prob)),
        "low_recall":  float(recall_score(y, pred, pos_label=0)),
        "high_recall": float(recall_score(y, pred, pos_label=1)),
        "brier":   float(brier_score_loss(y, prob)),
        "pred":    pred,
        "prob":    prob,
    }

def bootstrap_ci(y_true, y_prob, metric_fn, n_boot=1000, seed=0):
    """Return (point_estimate, lo_95, hi_95) for a metric via bootstrap."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    point = metric_fn(y_true, y_prob)
    samples = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            samples.append(metric_fn(y_true[idx], y_prob[idx]))
        except ValueError:
            continue
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return point, lo, hi

# ── 10. Test set evaluation with bootstrap CIs ────────────────────────────────
val_lr  = metrics(best_lr,  Xva, yva)
val_svm = metrics(best_svm, Xva, yva)
val_rf  = metrics(best_rf,  Xva, yva)

te_lr   = metrics(best_lr,  Xte, yte)
te_svm  = metrics(best_svm, Xte, yte)
te_rf   = metrics(best_rf,  Xte, yte)

print("\n=== TEST SET — point + 95% bootstrap CI ===")
results_ci = {}
for name, m in [("LR", te_lr), ("SVM", te_svm), ("RF", te_rf)]:
    auc_pt, auc_lo, auc_hi = bootstrap_ci(yte, m["prob"], roc_auc_score)
    pr_pt,  pr_lo,  pr_hi  = bootstrap_ci(yte, m["prob"], average_precision_score)
    results_ci[name] = {"auc":(auc_pt,auc_lo,auc_hi), "pr_auc":(pr_pt,pr_lo,pr_hi)}
    print(f"  {name:<4} AUC = {auc_pt:.4f}  [{auc_lo:.4f}, {auc_hi:.4f}]   "
          f"PR-AUC = {pr_pt:.4f}  [{pr_lo:.4f}, {pr_hi:.4f}]")

# ── 11. McNemar's test for pairwise model comparison ──────────────────────────
def mcnemar(y, p1, p2):
    """McNemar's test on two binary prediction vectors. Returns (chi2, p-value)."""
    from scipy.stats import chi2_contingency
    correct1 = (p1 == y); correct2 = (p2 == y)
    b = int(np.sum(correct1 & ~correct2))   # model1 right, model2 wrong
    c = int(np.sum(~correct1 & correct2))   # model1 wrong, model2 right
    if (b + c) == 0:
        return 0.0, 1.0
    # Continuity-corrected McNemar
    chi2 = ((abs(b - c) - 1) ** 2) / (b + c)
    from scipy.stats import chi2 as chi2_dist
    p = 1 - chi2_dist.cdf(chi2, df=1)
    return float(chi2), float(p), b, c

print("\n=== McNemar's pairwise test ===")
mcn = {}
for n1, m1 in [("LR", te_lr), ("SVM", te_svm)]:
    for n2, m2 in [("SVM", te_svm), ("RF", te_rf)]:
        if n1 == n2: continue
        chi2, pval, b, c = mcnemar(yte, m1["pred"], m2["pred"])
        mcn[f"{n1}_vs_{n2}"] = {"chi2":chi2, "p":pval, "b":b, "c":c}
        print(f"  {n1:<3} vs {n2:<3}: χ²={chi2:6.3f}  p={pval:.4g}  "
              f"(b={b}, c={c})")

# ── 12. Multi-seed robustness study ───────────────────────────────────────────
print("\n=== Multi-seed robustness (5 seeds) ===")
SEEDS = [0, 1, 7, 42, 2023]
seed_results = {"LR": [], "SVM": [], "RF": []}

for seed in SEEDS:
    Xtr_s, Xva_s, Xte_s, ytr_s, yva_s, yte_s, _ = make_splits(X_eng, y, seed=seed)
    # Re-train with best HPs (from seed=42 tuning)
    lr_s  = LogisticRegression(penalty="l2", solver="lbfgs", max_iter=2000,
            **{k:v for k,v in lr_gs.best_params_.items()},
            random_state=42).fit(Xtr_s, ytr_s)
    svm_s = SVC(kernel="rbf", probability=True,
            **{k:v for k,v in svm_gs.best_params_.items()},
            random_state=42).fit(Xtr_s, ytr_s)
    rf_s  = RandomForestClassifier(n_jobs=-1,
            **{k:v for k,v in rf_gs.best_params_.items()},
            random_state=42).fit(Xtr_s, ytr_s)
    for name, mdl in [("LR", lr_s), ("SVM", svm_s), ("RF", rf_s)]:
        seed_results[name].append(metrics(mdl, Xte_s, yte_s))
    print(f"  seed={seed:>4}: "
          f"LR-AUC={seed_results['LR'][-1]['auc']:.4f}  "
          f"SVM-AUC={seed_results['SVM'][-1]['auc']:.4f}  "
          f"RF-AUC={seed_results['RF'][-1]['auc']:.4f}")

multi_seed_summary = {}
for name in ["LR", "SVM", "RF"]:
    aucs = [r["auc"] for r in seed_results[name]]
    f1s  = [r["f1"]  for r in seed_results[name]]
    accs = [r["acc"] for r in seed_results[name]]
    multi_seed_summary[name] = {
        "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
        "f1_mean":  float(np.mean(f1s)),  "f1_std":  float(np.std(f1s)),
        "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
    }
    print(f"  {name:<4} AUC = {np.mean(aucs):.4f} ± {np.std(aucs):.4f}  "
          f"F1 = {np.mean(f1s):.4f} ± {np.std(f1s):.4f}  "
          f"ACC = {np.mean(accs):.4f} ± {np.std(accs):.4f}")

# ── 13. All-model ablation: base (11) vs engineered (14) ──────────────────────
print("\n=== All-model ablation: base (11) vs engineered (14) ===")
Xtr_b, Xva_b, Xte_b, _, _, _, _ = make_splits(X_base, y)

ablation = {}
def quick_train_eval(model_name, X_tr_loc, X_te_loc, ytr_loc, yte_loc, params):
    if model_name == "LR":
        m = LogisticRegression(penalty="l2", solver="lbfgs", max_iter=2000,
                               random_state=42, **params).fit(X_tr_loc, ytr_loc)
    elif model_name == "SVM":
        m = SVC(kernel="rbf", probability=True, random_state=42,
                **params).fit(X_tr_loc, ytr_loc)
    else:
        m = RandomForestClassifier(random_state=42, n_jobs=-1,
                **params).fit(X_tr_loc, ytr_loc)
    return metrics(m, X_te_loc, yte_loc)

for name, gs in [("LR", lr_gs), ("SVM", svm_gs), ("RF", rf_gs)]:
    base_res = quick_train_eval(name, Xtr_b, Xte_b, ytr, yte, gs.best_params_)
    eng_res  = quick_train_eval(name, Xtr,   Xte,   ytr, yte, gs.best_params_)
    ablation[name] = {
        "base_auc": base_res["auc"], "eng_auc": eng_res["auc"],
        "base_f1":  base_res["f1"],  "eng_f1":  eng_res["f1"],
        "delta_auc": eng_res["auc"] - base_res["auc"],
        "delta_f1":  eng_res["f1"]  - base_res["f1"],
    }
    print(f"  {name:<4} base AUC={base_res['auc']:.4f}  eng AUC={eng_res['auc']:.4f}  "
          f"ΔAUC={ablation[name]['delta_auc']:+.4f}  ΔF1={ablation[name]['delta_f1']:+.4f}")

# ── 14. Threshold tuning for LR (low-quality recall) ──────────────────────────
print("\n=== Threshold tuning (LR, on validation set) ===")
val_prob_lr = best_lr.predict_proba(Xva)[:, 1]
ths = np.linspace(0.1, 0.9, 81)
best_th, best_macro_f1 = 0.5, -1
for th in ths:
    pr = (val_prob_lr >= th).astype(int)
    score = f1_score(yva, pr, average="macro")
    if score > best_macro_f1:
        best_macro_f1, best_th = score, th

# Default threshold
te_lr_default = metrics(best_lr, Xte, yte, threshold=0.5)
te_lr_tuned   = metrics(best_lr, Xte, yte, threshold=best_th)
print(f"  Best threshold (max macro-F1 on val): {best_th:.3f}  "
      f"val macro-F1 = {best_macro_f1:.4f}")
print(f"  Test @ τ=0.5     : F1={te_lr_default['f1']:.4f}  "
      f"low_recall={te_lr_default['low_recall']:.4f}  AUC={te_lr_default['auc']:.4f}")
print(f"  Test @ τ={best_th:.3f}: F1={te_lr_tuned['f1']:.4f}  "
      f"low_recall={te_lr_tuned['low_recall']:.4f}  AUC={te_lr_tuned['auc']:.4f}")

threshold_tuning = {
    "best_th": float(best_th),
    "default": {"f1": te_lr_default["f1"], "low_recall": te_lr_default["low_recall"],
                "high_recall": te_lr_default["high_recall"], "acc": te_lr_default["acc"]},
    "tuned":   {"f1": te_lr_tuned["f1"],   "low_recall": te_lr_tuned["low_recall"],
                "high_recall": te_lr_tuned["high_recall"], "acc": te_lr_tuned["acc"]},
}

# ── 15. Density-drop experiment ────────────────────────────────────────────────
print("\n=== Density-drop experiment (RF) ===")
NO_DENSITY = [f for f in ENG_FEATURES if f != "density"]
X_nodens = df[NO_DENSITY].values
Xtr_nd, Xva_nd, Xte_nd, _, _, _, _ = make_splits(X_nodens, y)
rf_nd = RandomForestClassifier(random_state=42, n_jobs=-1, **rf_gs.best_params_).fit(Xtr_nd, ytr)
m_nd = metrics(rf_nd, Xte_nd, yte)
print(f"  RF with density   : AUC={te_rf['auc']:.4f}  F1={te_rf['f1']:.4f}")
print(f"  RF without density: AUC={m_nd['auc']:.4f}  F1={m_nd['f1']:.4f}")
density_exp = {"with":   {"auc": te_rf["auc"], "f1": te_rf["f1"]},
               "without":{"auc": m_nd["auc"],  "f1": m_nd["f1"]}}

# ── 16. Calibration analysis ──────────────────────────────────────────────────
print("\n=== Calibration (Brier scores) ===")
print(f"  LR  Brier = {te_lr['brier']:.4f}")
print(f"  SVM Brier = {te_svm['brier']:.4f}")
print(f"  RF  Brier = {te_rf['brier']:.4f}")

# ── 17. PRINT FINAL TABLES ────────────────────────────────────────────────────
print("\n" + "="*78)
print("FINAL TEST SET — POINT, 95% CI, AND PR-AUC")
print("="*78)
hdr = f"{'Model':<14}{'Acc':>9}{'Macro F1':>11}{'ROC-AUC [95% CI]':>27}{'PR-AUC':>11}"
print(hdr); print("-"*78)
for name, m in [("LR", te_lr), ("SVM", te_svm), ("RF", te_rf)]:
    a, lo, hi = results_ci[name]["auc"]
    pa = results_ci[name]["pr_auc"][0]
    print(f"{name:<14}{m['acc']:>9.4f}{m['f1']:>11.4f}"
          f"  {a:.4f} [{lo:.4f}, {hi:.4f}]  {pa:>10.4f}")

print("\n" + "="*78)
print("MULTI-SEED MEAN ± STD (5 SEEDS)")
print("="*78)
print(f"{'Model':<14}{'Acc':>16}{'Macro F1':>17}{'ROC-AUC':>17}")
for name in ["LR", "SVM", "RF"]:
    s = multi_seed_summary[name]
    print(f"{name:<14}{s['acc_mean']:.4f} ± {s['acc_std']:.4f}   "
          f"{s['f1_mean']:.4f} ± {s['f1_std']:.4f}   "
          f"{s['auc_mean']:.4f} ± {s['auc_std']:.4f}")

print("\n" + "="*78)
print("ALL-MODEL ABLATION — TEST SET")
print("="*78)
print(f"{'Model':<8}{'Base AUC':>12}{'+Eng AUC':>12}{'ΔAUC':>10}{'Base F1':>11}{'+Eng F1':>11}{'ΔF1':>9}")
for name in ["LR", "SVM", "RF"]:
    a = ablation[name]
    print(f"{name:<8}{a['base_auc']:>12.4f}{a['eng_auc']:>12.4f}{a['delta_auc']:>+10.4f}"
          f"{a['base_f1']:>11.4f}{a['eng_f1']:>11.4f}{a['delta_f1']:>+9.4f}")

# ── 18. FIGURES ───────────────────────────────────────────────────────────────
# Fig 1: ROC curves with bootstrap CIs in legend
fig, ax = plt.subplots(figsize=(6.2, 5))
for m_res, lbl, clr in [(te_lr, "LR", "steelblue"),
                         (te_svm, "SVM", "darkorange"),
                         (te_rf, "RF", "forestgreen")]:
    a, lo, hi = bootstrap_ci(yte, m_res["prob"], roc_auc_score)
    RocCurveDisplay.from_predictions(yte, m_res["prob"],
        name=f"{lbl}  AUC={a:.3f} [{lo:.3f}, {hi:.3f}]", ax=ax, color=clr)
ax.plot([0,1],[0,1], "k--", linewidth=0.8)
ax.set_title("ROC Curves with 95% Bootstrap CIs — Test Set")
ax.legend(loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "roc_test_all.png"), dpi=150); plt.close()

# Fig 2: PR curves
from sklearn.metrics import PrecisionRecallDisplay
fig, ax = plt.subplots(figsize=(6.2, 5))
for m_res, lbl, clr in [(te_lr, "LR", "steelblue"),
                         (te_svm, "SVM", "darkorange"),
                         (te_rf, "RF", "forestgreen")]:
    PrecisionRecallDisplay.from_predictions(yte, m_res["prob"],
        name=f"{lbl}  AP={m_res['pr_auc']:.3f}", ax=ax, color=clr)
ax.set_title("Precision-Recall Curves — Test Set")
ax.legend(loc="lower left", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "pr_curves_test.png"), dpi=150); plt.close()

# Fig 3: Calibration (reliability) curves
fig, ax = plt.subplots(figsize=(6.2, 5))
for m_res, lbl, clr in [(te_lr, "LR", "steelblue"),
                         (te_svm, "SVM", "darkorange"),
                         (te_rf, "RF", "forestgreen")]:
    pt, pp = calibration_curve(yte, m_res["prob"], n_bins=10, strategy="quantile")
    ax.plot(pp, pt, "o-", color=clr,
            label=f"{lbl}  Brier={m_res['brier']:.3f}")
ax.plot([0,1], [0,1], "k--", linewidth=0.8, label="Perfect")
ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("Observed frequency")
ax.set_title("Reliability Diagram — Test Set")
ax.legend(); plt.tight_layout()
plt.savefig(os.path.join(FIG, "calibration_test.png"), dpi=150); plt.close()

# Fig 4: All-model ablation bar chart
fig, ax = plt.subplots(figsize=(7.5, 4.5))
xs = np.arange(3); width = 0.35
base_aucs = [ablation[n]["base_auc"] for n in ["LR","SVM","RF"]]
eng_aucs  = [ablation[n]["eng_auc"]  for n in ["LR","SVM","RF"]]
ax.bar(xs - width/2, base_aucs, width, label="Base (11)", color="lightsteelblue")
ax.bar(xs + width/2, eng_aucs,  width, label="+Engineered (14)", color="steelblue")
for i, (b, e) in enumerate(zip(base_aucs, eng_aucs)):
    ax.text(i - width/2, b + 0.005, f"{b:.3f}", ha="center", fontsize=9)
    ax.text(i + width/2, e + 0.005, f"{e:.3f}", ha="center", fontsize=9)
    delta = e - b
    ax.text(i, max(b,e) + 0.025, f"Δ={delta:+.4f}",
            ha="center", fontsize=9, fontweight="bold",
            color="green" if delta > 0 else "red")
ax.set_xticks(xs); ax.set_xticklabels(["Logistic Reg.", "SVM (RBF)", "Random Forest"])
ax.set_ylabel("Test ROC-AUC"); ax.set_ylim(0.78, 0.95)
ax.set_title("All-Model Ablation: Feature Engineering Impact")
ax.legend(); plt.tight_layout()
plt.savefig(os.path.join(FIG, "ablation_all_models.png"), dpi=150); plt.close()

# Fig 5: Multi-seed robustness boxplot
fig, ax = plt.subplots(figsize=(7, 4.5))
data = [[r["auc"] for r in seed_results[n]] for n in ["LR","SVM","RF"]]
bp = ax.boxplot(data, labels=["LR", "SVM", "RF"], patch_artist=True, widths=0.55)
for patch, color in zip(bp['boxes'], ["lightsteelblue", "navajowhite", "lightgreen"]):
    patch.set_facecolor(color)
ax.set_ylabel("Test ROC-AUC"); ax.set_title("Multi-seed Robustness (5 seeds)")
ax.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "multiseed_boxplot.png"), dpi=150); plt.close()

# Fig 6: Threshold tuning curve for LR
ths_plot = np.linspace(0.1, 0.9, 81)
f1s = [f1_score(yva, (val_prob_lr >= t).astype(int), average="macro") for t in ths_plot]
lows = [recall_score(yva, (val_prob_lr >= t).astype(int), pos_label=0) for t in ths_plot]
highs= [recall_score(yva, (val_prob_lr >= t).astype(int), pos_label=1) for t in ths_plot]
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(ths_plot, f1s,   color="darkorange", label="Macro F1", linewidth=2)
ax.plot(ths_plot, lows,  color="crimson",     label="Low recall", linewidth=2)
ax.plot(ths_plot, highs, color="forestgreen", label="High recall", linewidth=2)
ax.axvline(best_th, color="black", linestyle="--", label=f"Best τ = {best_th:.3f}")
ax.axvline(0.5,     color="gray",  linestyle=":",  label="Default τ = 0.5")
ax.set_xlabel("Decision threshold τ"); ax.set_ylabel("Score")
ax.set_title("Threshold Tuning for Logistic Regression (Validation Set)")
ax.legend(); plt.tight_layout()
plt.savefig(os.path.join(FIG, "threshold_tuning.png"), dpi=150); plt.close()

# Fig 7: Confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, res, title in zip(axes, [te_lr, te_svm, te_rf],
    ["Logistic Regression", "SVM (RBF)", "Random Forest"]):
    cm = confusion_matrix(yte, res["pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Low","High"], yticklabels=["Low","High"])
    ax.set_title(f"{title}\n(Test Set)"); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(FIG, "confusion_test_all.png"), dpi=150); plt.close()

# Fig 8: Random Forest feature importance
importances = pd.Series(best_rf.feature_importances_, index=ENG_FEATURES).sort_values()
fig, ax = plt.subplots(figsize=(7, 6))
colors = ["darkorange" if f in ["wine_type","so2_ratio","acidity_ratio"] else "steelblue"
          for f in importances.index]
importances.plot(kind="barh", ax=ax, color=colors)
ax.set_title("Random Forest Feature Importances\n(orange = engineered features)")
ax.set_xlabel("Mean Decrease in Impurity")
plt.tight_layout()
plt.savefig(os.path.join(FIG, "rf_feature_importance.png"), dpi=150); plt.close()

# Fig 9: LR Coefficients (regenerated in v2)
coef = pd.Series(best_lr.coef_[0], index=ENG_FEATURES).sort_values()
fig, ax = plt.subplots(figsize=(7, 6))
coef.plot(kind="barh", ax=ax,
          color=["crimson" if v < 0 else "steelblue" for v in coef])
ax.set_title("Logistic Regression Coefficients (L2)\n(after standardization)")
ax.set_xlabel("Coefficient value")
plt.tight_layout()
plt.savefig(os.path.join(FIG, "lr_coefficients.png"), dpi=150); plt.close()

# Fig 10: Learning curve for RF
train_sizes, train_sc, val_sc = learning_curve(
    RandomForestClassifier(**rf_gs.best_params_, random_state=42, n_jobs=-1),
    Xtr, ytr, cv=cv5, scoring="roc_auc",
    train_sizes=np.linspace(0.1, 1.0, 8), n_jobs=-1)
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(train_sizes, train_sc.mean(axis=1), "o-", color="steelblue", label="Train AUC")
ax.plot(train_sizes, val_sc.mean(axis=1),   "s-", color="darkorange", label="CV Val AUC")
ax.fill_between(train_sizes,
    train_sc.mean(1)-train_sc.std(1), train_sc.mean(1)+train_sc.std(1),
    alpha=0.15, color="steelblue")
ax.fill_between(train_sizes,
    val_sc.mean(1)-val_sc.std(1), val_sc.mean(1)+val_sc.std(1),
    alpha=0.15, color="darkorange")
ax.set_xlabel("Training samples"); ax.set_ylabel("ROC-AUC")
ax.set_title("Learning Curve — Random Forest"); ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG, "learning_curve_rf.png"), dpi=150); plt.close()

# Capture exact learning-curve numbers for the report
lc_data = {
    "train_sizes": train_sizes.tolist(),
    "train_mean": train_sc.mean(axis=1).tolist(),
    "val_mean":   val_sc.mean(axis=1).tolist(),
    "min_gap":    float(train_sc.mean(1).min() - val_sc.mean(1).min()),
    "max_gap":    float(train_sc.mean(1).max() - val_sc.mean(1).max()),
    "final_gap":  float(train_sc.mean(1)[-1] - val_sc.mean(1)[-1]),
}

# ── 19. Dump everything to JSON ───────────────────────────────────────────────
out = {
    "best_params": {
        "LR":  lr_gs.best_params_,
        "SVM": svm_gs.best_params_,
        "RF":  rf_gs.best_params_,
    },
    "cv_auc": {
        "LR":  lr_gs.best_score_,
        "SVM": svm_gs.best_score_,
        "RF":  rf_gs.best_score_,
    },
    "validation": {n: {k:v for k,v in d.items() if k not in ("pred","prob")}
                   for n,d in [("LR",val_lr),("SVM",val_svm),("RF",val_rf)]},
    "test": {n: {k:v for k,v in d.items() if k not in ("pred","prob")}
             for n,d in [("LR",te_lr),("SVM",te_svm),("RF",te_rf)]},
    "test_ci": {n: {"auc": list(results_ci[n]["auc"]),
                    "pr_auc": list(results_ci[n]["pr_auc"])}
                for n in ["LR","SVM","RF"]},
    "mcnemar": mcn,
    "multiseed": multi_seed_summary,
    "ablation": ablation,
    "threshold_tuning": threshold_tuning,
    "density_experiment": density_exp,
    "vif": vif_dict,
    "learning_curve": lc_data,
}
with open(os.path.join(BASE, "results.json"), "w") as f:
    json.dump(out, f, indent=2, default=float)
print("\nResults saved to results.json")
print("Pipeline complete.")
