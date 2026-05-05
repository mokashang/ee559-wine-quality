# Presentation Transcript (v2)
## Predicting Wine Quality from Physicochemical Properties
**EE559/CSCI 559 — Spring 2026 | Mengjia Shang | 7338151449**

*Approximate total runtime: 14–15 minutes for 15 slides.*
*This transcript reflects the v2 pipeline with class-balance-aware tuning, bootstrap CIs, McNemar's tests, multi-seed robustness, and an all-model ablation.*

---

## SLIDE 1 — Title [~0:00–0:30]

Hello everyone. My name is Mengjia Shang and this is my EE559 machine learning project for Spring 2026. The title is *Predicting Wine Quality from Physicochemical Properties: A Comparative Study of Supervised Learning Methods*.

The core idea is to use measurable chemical properties — alcohol content, acidity, sulfur dioxide levels, and so on — to automatically predict whether a wine will be rated high or low quality by expert tasters. I'll walk you through the dataset, my feature engineering decisions, three supervised learning models, and a rigorous statistical comparison that includes bootstrap confidence intervals, McNemar's significance test, and a multi-seed robustness study.

---

## SLIDE 2 — Motivation [~0:30–1:30]

Wine quality certification is traditionally done by trained human tasting panels. The process is expensive, slow, and not very reproducible. Different panels can give the same wine very different scores.

The good news is that physicochemical testing is already standard practice during wine production. Every batch is tested for alcohol, pH, acidity, sulfur dioxide levels, and other properties before it's bottled. These numbers are already sitting in databases.

The question this project answers is: *can we use those measurements to predict what a human expert panel would score?* If yes, we get a fast, objective, zero-extra-cost quality gate. The dataset has 6,497 samples with 11 features and zero missing values — clean data that lets us focus on model differences rather than data cleaning.

---

## SLIDE 3 — Problem Formulation [~1:30–2:30]

The task is **binary classification**. The input is a vector of 11 continuous physicochemical measurements. The target is a binary label derived from the original expert quality score: wines with a score of 6 or above are *high quality*, below 6 are *low quality*.

I chose 6 as the threshold because it's the industry pass/fail convention.

The class distribution is 63.3% high quality and 36.7% low quality. This mild imbalance motivates two design choices: ROC-AUC and macro F1 as primary metrics rather than raw accuracy, and — importantly — *tuning class_weight equals "balanced" as a hyperparameter for both Logistic Regression and SVM*. This is an upgrade over my midway report, which only tuned class weights for the Random Forest.

---

## SLIDE 4 — Dataset, EDA & VIF [~2:30–3:45]

In addition to standard EDA, I computed Variance Inflation Factors to formally quantify multicollinearity. Three features cross the conventional warning threshold of 10: volatile acidity at 39, the engineered acidity ratio at 37 — which is expected because it shares a numerator with volatile acidity — and density at 15.

Density was the most actionable concern, because it isn't derived from the others by construction. So I ran a follow-up experiment: training Random Forest with and without density. The AUC difference is essentially zero — 0.9041 versus 0.9040. This confirms that density adds no independent information for tree-based models, validating the VIF analysis.

The strongest positive predictor remains alcohol with Pearson r equals plus 0.44, and the strongest negative is volatile acidity at minus 0.27.

---

## SLIDE 5 — Feature Engineering [~3:45–5:00]

Five preprocessing steps, all with statistics computed on the training fold only.

First, I winsorized chlorides and residual sugar at the 1st and 99th percentiles. Second, a wine-type indicator captures the systematic chemistry differences between red and white. Third, an SO2 ratio — free over total — captures the active preservative fraction. Fourth, an acidity ratio — volatile over fixed — isolates spoilage-related acidity. Fifth, standardization to zero mean and unit variance, applied only to LR and SVM.

The final feature matrix has 14 features.

---

## SLIDE 6 — Experimental Protocol [~5:00–6:15]

The protocol has four pillars.

First: a stratified 70-15-15 split — training, validation, and a held-out test set. The test set is touched exactly once per model.

Second: hyperparameter tuning happens entirely inside 5-fold stratified cross-validation on the training set. The validation set is never used for tuning.

Third — and this is the major upgrade from my midway pipeline — class_weight equals balanced is now searched as a hyperparameter for *all three models*, not just the Random Forest. This turns out to matter a lot, especially for the linear model.

Fourth: instead of just point estimates, I report 95% bootstrap confidence intervals, McNemar's significance tests for pairwise comparisons, and a 5-seed robustness study where I retrain each model under five different stratified splits.

---

## SLIDE 7 — Logistic Regression [~6:15–7:15]

The first model is Logistic Regression with L2 regularization. The decision function is sigmoid of the linear combination, with an L2 penalty term half-C-inverse times the squared norm of the weights.

Grid search over C and class_weight gave best C equals 0.1 with **class_weight equals balanced**. Cross-validation AUC is 0.7999.

The balanced class weight is the key tuning choice. Without it, low-quality recall would be only 0.56 — meaning 44% of defective bottles would be missed. With balanced weighting, recall jumps to 0.74, more than compensating for a small accuracy decrease.

---

## SLIDE 8 — SVM [~7:15–8:15]

The RBF SVM uses the kernel exp of minus gamma times the squared distance. Joint grid search over C, gamma, and class_weight gave best C equals 1, gamma equals 0.1, again with **class_weight equals balanced**. CV AUC is 0.8276 — a clear improvement over the unbalanced-tuning version.

A key result from this slide: with balanced weights, the SVM achieves the **highest minority-class recall of any model** — 0.796. It catches the most defective bottles. We'll see why that matters for the practical interpretation in a few slides.

---

## SLIDE 9 — Random Forest [~8:15–9:15]

The Random Forest is an ensemble of 300 decision trees trained on bootstrap samples with random feature subsets at each split. Grid search over n_estimators, max_depth, and class_weight gave best n equals 300, depth equals None, and — interestingly — **class_weight equals None**.

Random Forest is the only model where balanced weighting did *not* win the grid search. This is because the bootstrap mechanism already provides a kind of implicit balancing — each tree sees a re-sampled training set with its own minority-class proportion. Adding explicit reweighting on top hurt performance.

CV AUC is 0.8814, a 5.4 point jump over SVM.

---

## SLIDE 10 — Test Results with Bootstrap CIs [~9:15–10:30]

These are the final test set results, each with a 95% bootstrap confidence interval, PR-AUC, and the Brier score for calibration.

Random Forest achieves accuracy 0.829, macro F1 0.811, ROC-AUC 0.904 with confidence interval 0.88 to 0.92, and PR-AUC 0.94. Most importantly: the AUC confidence intervals do not overlap between any two models. So the ranking is statistically robust on this test set.

The Brier score column is also interesting. Random Forest has the lowest Brier — 0.124 — meaning its probabilities are best calibrated. This is somewhat counter-intuitive — linear models are usually expected to be best calibrated — but it's consistent with the literature: ensemble averaging in RF produces moderate, well-spread probabilities, while balanced-class-weighted LR pushes probabilities toward 0.5 to compensate for the imbalance, which hurts Brier.

---

## SLIDE 11 — McNemar + Multi-seed [~10:30–11:30]

Two independent statistical analyses corroborate the ranking.

On the left: McNemar's test on the test-set agreement table. All three pairwise p-values are below 0.001 — LR vs SVM at 5.8 times 10 to the negative 4, LR vs RF at 1.2 times 10 to the negative 10, and SVM vs RF at 1.1 times 10 to the negative 5. Every pairwise difference is highly statistically significant.

On the right: multi-seed robustness. I retrained each model under five different stratified splits — seeds 0, 1, 7, 42, and 2023. The mean AUCs are LR 0.801 plus or minus 0.013, SVM 0.835 plus or minus 0.008, and RF 0.898 plus or minus 0.013. Note that even the *plus or minus 2 sigma* intervals do not overlap between any two models. The ranking is not seed-dependent.

---

## SLIDE 12 — Per-Class & Operational Trade-off [~11:30–12:30]

This slide is one of the most interesting findings.

Look at the low-quality recall column. **SVM with balanced weights gives 0.796** — the highest. But Random Forest gives 0.710 with the highest *precision* of 0.801.

What does this mean operationally?

- If you're a wine producer and **shipping a defective bottle is very costly** — reputation damage, returns, possibly health issues — you want SVM. It catches 80% of defects, missing only 20%.
- If **re-testing a good bottle is the dominant cost** — wasted lab time, delayed bottling — you want Random Forest. 80% of its low-quality alarms are real.
- For overall balanced performance, F1-wise, RF wins.

So the right model depends on the cost ratio. This is a much richer story than just picking the model with the highest AUC.

---

## SLIDE 13 — All-Model Ablation [~12:30–13:30]

This is the single most important methodological insight.

I ran the ablation across **all three models**, not just RF. The results match theoretical expectation perfectly.

Logistic Regression gets the largest benefit from engineered features — delta-AUC of plus 0.0075 and delta-F1 of plus 0.013. SVM gets essentially zero — its RBF kernel implicitly captures the same information. Random Forest gets a small positive boost of plus 0.003 — its deep trees rediscover most of the ratio relationships from raw inputs.

The hierarchy is exactly what we'd predict: lower-capacity models benefit more from explicit feature engineering. This validates that the engineering choices are domain-correct, but also clarifies their value scales inversely with model capacity.

---

## SLIDE 14 — Calibration & Feature Importance [~13:30–14:15]

Two secondary results.

On the left, Brier scores. Random Forest is best calibrated at 0.124. SVM at 0.151. Logistic Regression worst at 0.177. The LR result is somewhat surprising but is explained by the balanced class weighting pushing probabilities toward 0.5.

On the right, RF feature importances. Alcohol dominates with importance 0.21, followed by volatile acidity at 0.13. This precisely matches the EDA correlations — alcohol at plus 0.44 and volatile acidity at minus 0.27 — so the model is learning something domain-meaningful, not memorizing noise. Density appears with non-zero importance even though we showed it's redundant — that's because RF has many trees that randomly selected density at split time, but the ensemble doesn't *need* it.

---

## SLIDE 15 — Discussion & Conclusion [~14:15–15:00]

To summarize the key findings.

Random Forest wins on overall ROC-AUC (0.904), F1, accuracy, and probability calibration. **But SVM with balanced class weights wins on minority-class recall**, which is the most operationally important metric for production quality gates.

All three pairwise differences are statistically significant under McNemar's test, with p-values below 0.001. The ranking is robust across five random seeds — non-overlapping plus-or-minus-2-sigma intervals.

The all-model ablation confirms that engineered features benefit linear models the most, validating the theoretical expectation that explicit feature engineering matters less as model capacity increases.

Density was confirmed redundant by both VIF and a controlled drop experiment — could be removed in deployment.

The main limitations are the binarization of an ordinal scale and the lack of producer or vintage metadata. Future work would extend to MLPs, SHAP attribution, ordinal regression, and cost-sensitive learning with explicitly elicited false-negative versus false-positive cost ratios.

Thank you for listening. I'm happy to take any questions.

---

*[End of presentation — ~15:00]*

---

## Q&A Preparation Notes

**Q: Why did class_weight = None win for Random Forest but "balanced" for LR and SVM?**
A: RF's bootstrap sampling provides implicit balancing — each tree sees a re-sampled training set. Adding explicit reweighting on top hurts because trees become too biased toward minority class memorization. For LR and SVM, the loss functions don't have this implicit mechanism, so explicit weighting helps.

**Q: Bootstrap CIs vs multi-seed — aren't they measuring the same thing?**
A: They measure different sources of variance. Bootstrap CIs vary the test set (sampling noise on the same model). Multi-seed varies the train/val/test partitions (split sensitivity). Both being narrow is stronger evidence than either alone.

**Q: Why is your low-quality recall so different between LR (0.74) and what you'd expect from a 0.82 AUC?**
A: AUC measures discriminative ranking, not threshold-specific outcomes. Class_weight = "balanced" effectively shifts the decision threshold to roughly 0.4, so recall on the minority class jumps. Without balanced weighting, the same model's recall would be 0.56 at threshold 0.5.

**Q: Why is RF *better* calibrated than LR? That seems backwards.**
A: It's well-documented in Niculescu-Mizil & Caruana 2005. Tree ensembles produce moderate, well-spread probabilities. Balanced-class-weighted LR pushes outputs toward 0.5 to equalize misclassification costs, which inflates Brier. If we used unweighted LR, calibration would improve but minority recall would crash.

**Q: How would you deploy this?**
A: Two recommendations: (1) drop density to simplify the feature pipeline (no AUC loss). (2) Pick model based on cost asymmetry — SVM if shipping defects is dominant cost, RF otherwise. Monitor for distribution shift; periodic retraining recommended as new grape varieties or production processes are introduced.
