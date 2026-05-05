// EE559 Wine Quality — Presentation Builder (v2)
// Mengjia Shang | 7338151449 | Spring 2026
// Reflects v2 pipeline: balanced class weights, bootstrap CIs, McNemar's test,
// multi-seed robustness, all-model ablation, calibration, density experiment.

const pptxgen = require("pptxgenjs");
const pres = new pptxgen();
pres.layout  = "LAYOUT_16x9";
pres.author  = "Mengjia Shang";
pres.title   = "Predicting Wine Quality from Physicochemical Properties";

// Palette
const C = {
  navyDark:"0D1B2A", navy:"1A2744", navyMid:"1E3A5F",
  wine:"7B1E2C", wineLight:"A83246", gold:"C9A84C",
  white:"FFFFFF", offWhite:"F0F4FA", muted:"9BB0CC",
  lightLine:"2C3E5A", tableHdr:"1E3A5F", tableAlt:"152338",
  green:"2A7A4B", orange:"C06020",
};

// Helpers
function makeShadow() {
  return { type:"outer", blur:8, offset:3, angle:135, color:"000000", opacity:0.25 };
}
function darkBg(slide, accentColor) {
  slide.background = { color: C.navyDark };
  slide.addShape(pres.shapes.RECTANGLE, {
    x:0, y:0, w:10, h:0.05, fill:{ color:accentColor||C.wine }, line:{ color:accentColor||C.wine }
  });
}
function slideTitle(slide, text, y) {
  const yy = y !== undefined ? y : 0.2;
  slide.addText(text, {
    x:0.5, y:yy, w:9, h:0.6, fontSize:24, fontFace:"Calibri",
    bold:true, color:C.white, margin:0
  });
  slide.addShape(pres.shapes.RECTANGLE, {
    x:0.5, y:yy+0.62, w:1.2, h:0.035,
    fill:{ color:C.gold }, line:{ color:C.gold }
  });
}
function sectionBadge(slide, text, x, y, w) {
  slide.addShape(pres.shapes.RECTANGLE, {
    x, y, w:w||3, h:0.32, fill:{ color:C.wine }, line:{ color:C.wine },
    shadow: makeShadow()
  });
  slide.addText(text, {
    x, y, w:w||3, h:0.32, fontSize:10, fontFace:"Calibri", bold:true,
    color:C.white, align:"center", valign:"middle", margin:0
  });
}
function card(slide, x, y, w, h, color) {
  slide.addShape(pres.shapes.RECTANGLE, {
    x, y, w, h, fill:{ color:color||C.navyMid },
    line:{ color:C.lightLine, pt:1 }, shadow: makeShadow()
  });
}
function statBox(slide, value, label, x, y, w) {
  card(slide, x, y, w||2.2, 1.1);
  slide.addText(value, {
    x, y:y+0.06, w:w||2.2, h:0.55, fontSize:32, fontFace:"Calibri",
    bold:true, color:C.gold, align:"center", valign:"middle", margin:0
  });
  slide.addText(label, {
    x, y:y+0.62, w:w||2.2, h:0.38, fontSize:11, fontFace:"Calibri",
    color:C.muted, align:"center", valign:"middle", margin:0
  });
}

// ── SLIDE 1 — TITLE ──────────────────────────────────────────────────────────
{
  const s = pres.addSlide();
  s.background = { color: C.navyDark };
  s.addShape(pres.shapes.RECTANGLE, { x:0, y:4.7, w:10, h:0.925,
    fill:{ color:C.wine }, line:{ color:C.wine } });
  s.addShape(pres.shapes.RECTANGLE, { x:0, y:0, w:0.12, h:5.625,
    fill:{ color:C.wine }, line:{ color:C.wine } });
  s.addText("Predicting Wine Quality", {
    x:0.35, y:0.9, w:9.3, h:0.85, fontSize:38, fontFace:"Calibri",
    bold:true, color:C.white, align:"left", margin:0
  });
  s.addText("from Physicochemical Properties", {
    x:0.35, y:1.72, w:9.3, h:0.75, fontSize:32, fontFace:"Calibri",
    color:C.gold, align:"left", margin:0
  });
  s.addShape(pres.shapes.RECTANGLE, { x:0.35, y:2.55, w:4.2, h:0.04,
    fill:{ color:C.muted }, line:{ color:C.muted } });
  s.addText("A Comparative Study of Supervised Learning Methods", {
    x:0.35, y:2.65, w:9.3, h:0.45, fontSize:16, fontFace:"Calibri",
    italic:true, color:C.muted, align:"left", margin:0
  });
  s.addText("Mengjia Shang  ·  ID: 7338151449", {
    x:0.35, y:4.75, w:6, h:0.32, fontSize:13, fontFace:"Calibri",
    bold:true, color:C.white, align:"left", margin:0
  });
  s.addText("EE559/CSCI 559 — Spring 2026  ·  Prof. Anand A. Joshi  ·  USC", {
    x:0.35, y:5.07, w:9.3, h:0.3, fontSize:11, fontFace:"Calibri",
    color:"FFCCD5", align:"left", margin:0
  });
  s.addNotes("Welcome everyone. I'm Mengjia Shang and today I'll present my EE559 project: predicting wine quality from physicochemical measurements. I'll walk through the dataset, my feature engineering decisions, three supervised learning models, and a rigorous comparison including bootstrap confidence intervals, McNemar's significance tests, and a multi-seed robustness study.");
}

// ── SLIDE 2 — MOTIVATION ──────────────────────────────────────────────────────
{
  const s = pres.addSlide();
  darkBg(s);
  slideTitle(s, "Why Automate Wine Quality Assessment?");
  card(s, 0.4, 1.0, 5.6, 3.9);
  sectionBadge(s, "THE PROBLEM", 0.4, 1.0, 2.2);
  s.addText([
    { text:"Traditional certification relies on expert sensory panels", options:{ bullet:true, breakLine:true } },
    { text:"Process is expensive, slow, and highly subjective", options:{ bullet:true, breakLine:true } },
    { text:"Scores vary between panels — reproducibility is limited", options:{ bullet:true, breakLine:true } },
    { text:"Physicochemical testing already happens during production", options:{ bullet:true, breakLine:true } },
    { text:"Goal: predict expert ratings from measurable chemistry", options:{ bullet:true, breakLine:true } },
    { text:"Practical impact: low-cost, objective quality gate for producers", options:{ bullet:true } },
  ], { x:0.6, y:1.42, w:5.2, h:3.2, fontSize:13, fontFace:"Calibri",
       color:C.offWhite, paraSpaceAfter:6 });
  statBox(s, "6,497", "Wine Samples", 6.2, 1.05, 3.3);
  statBox(s, "11", "Physicochemical\nFeatures", 6.2, 2.28, 3.3);
  statBox(s, "0", "Missing Values", 6.2, 3.51, 3.3);
  s.addNotes("Wine quality is traditionally assessed by certified tasters — but this is expensive and inconsistent. Physicochemical properties like alcohol content, acidity, and sulfur dioxide are already measured in the lab. If we can train a model on these measurements, we get a reproducible, instant quality estimate at no extra cost.");
}

// ── SLIDE 3 — PROBLEM FORMULATION ────────────────────────────────────────────
{
  const s = pres.addSlide();
  s.background = { color: C.navyDark };
  s.addShape(pres.shapes.RECTANGLE, { x:0, y:0, w:10, h:0.05,
    fill:{ color:C.wine }, line:{ color:C.wine } });
  slideTitle(s, "Problem Formulation");
  card(s, 0.35, 1.05, 4.5, 3.9);
  sectionBadge(s, "TASK DEFINITION", 0.35, 1.05, 2.5);
  s.addText([
    { text:"Type:", options:{ bold:true, color:C.gold } },
    { text:"  Binary Classification", options:{ breakLine:true, color:C.offWhite } },
    { text:"Input:", options:{ bold:true, color:C.gold } },
    { text:"  11 physicochemical\nmeasurements", options:{ breakLine:true, color:C.offWhite } },
    { text:"Target:", options:{ bold:true, color:C.gold } },
    { text:"  High Quality (≥6)\nvs Low Quality (<6)", options:{ breakLine:true, color:C.offWhite } },
    { text:"Threshold:", options:{ bold:true, color:C.gold } },
    { text:"  Score of 6 = industry\npass/fail convention", options:{ color:C.offWhite } },
  ], { x:0.55, y:1.48, w:4.1, h:3.3, fontSize:13, fontFace:"Calibri", paraSpaceAfter:8 });
  card(s, 5.15, 1.05, 4.5, 3.9);
  sectionBadge(s, "CLASS DISTRIBUTION", 5.15, 1.05, 2.8);
  const barData = [
    { label:"High Quality\n(score ≥ 6)", val:63.3, color:C.gold },
    { label:"Low Quality\n(score < 6)", val:36.7, color:C.wine }
  ];
  barData.forEach((d, i) => {
    const bw = 1.3, maxH = 1.6, bh = maxH * d.val / 100;
    const bx = 5.75 + i * 1.9, by = 3.85 - bh;
    s.addShape(pres.shapes.RECTANGLE, { x:bx, y:by, w:bw, h:bh,
      fill:{ color:d.color }, line:{ color:d.color } });
    s.addText(d.val + "%", { x:bx, y:by-0.35, w:bw, h:0.32,
      fontSize:18, fontFace:"Calibri", bold:true, color:d.color,
      align:"center", margin:0 });
    s.addText(d.label, { x:bx, y:3.9, w:bw, h:0.6,
      fontSize:10, fontFace:"Calibri", color:C.muted, align:"center", margin:0 });
  });
  s.addText("Mild imbalance — class_weight='balanced'\ntuned for LR & SVM", {
    x:5.2, y:4.62, w:4.3, h:0.4, fontSize:10, fontFace:"Calibri",
    italic:true, color:C.muted, align:"center", margin:0
  });
  s.addNotes("This is binary classification. Given 11 physicochemical inputs, we predict whether a wine will score 6 or above. The class distribution is mildly imbalanced at 63 to 37 — which is why I tuned class_weight equals balanced for the linear and kernel models alongside the regularization parameter.");
}

// ── SLIDE 4 — DATASET, EDA & VIF ─────────────────────────────────────────────
{
  const s = pres.addSlide();
  darkBg(s);
  slideTitle(s, "Dataset, EDA & VIF Multicollinearity Check");
  const findings = [
    { val:"r = +0.44", lbl:"Alcohol\n(top positive predictor)", col:C.gold },
    { val:"r = −0.27", lbl:"Volatile Acidity\n(top negative)", col:C.wineLight },
    { val:"VIF=15.3", lbl:"Density\n(multicollinear ⚠️)", col:C.muted },
    { val:"1,599",     lbl:"Red wine samples", col:C.wineLight },
    { val:"4,898",     lbl:"White wine samples", col:C.gold },
    { val:"0",         lbl:"Missing values", col:"2A7A4B" },
  ];
  findings.forEach((f, i) => {
    const col = i % 3, row = Math.floor(i / 3);
    const x = 0.35 + col * 3.1, y = 1.05 + row * 1.3;
    card(s, x, y, 2.9, 1.15);
    s.addText(f.val, { x, y:y+0.1, w:2.9, h:0.5, fontSize:24, fontFace:"Calibri",
      bold:true, color:f.col, align:"center", valign:"middle", margin:0 });
    s.addText(f.lbl, { x, y:y+0.62, w:2.9, h:0.48, fontSize:10, fontFace:"Calibri",
      color:C.muted, align:"center", margin:0 });
  });
  card(s, 0.35, 3.65, 9.3, 1.65);
  sectionBadge(s, "VIF FINDINGS & FOLLOW-UP", 0.35, 3.65, 2.8);
  s.addText([
    { text:"3 features have VIF > 10: volatile_acidity (39.4), acidity_ratio (37.2), density (15.3)\n",
      options:{ bullet:true, breakLine:true } },
    { text:"Density-drop experiment: RF AUC = 0.9041 (with) vs 0.9040 (without) — no meaningful change\n",
      options:{ bullet:true, breakLine:true } },
    { text:"Confirms density adds essentially zero independent information to tree-based models",
      options:{ bullet:true } },
  ], { x:0.55, y:4.08, w:9.0, h:1.18, fontSize:11.5, fontFace:"Calibri",
       color:C.offWhite, paraSpaceAfter:4 });
  s.addNotes("Beyond standard EDA, I computed Variance Inflation Factors. Three features cross the conventional warning threshold of 10 — volatile acidity at 39, the engineered acidity_ratio at 37, and density at 15. To act on this insight, I ran a follow-up experiment: training Random Forest with and without density. The AUC difference is essentially zero, confirming density adds no independent information for tree models.");
}

// ── SLIDE 5 — FEATURE ENGINEERING ────────────────────────────────────────────
{
  const s = pres.addSlide();
  darkBg(s, C.gold);
  slideTitle(s, "Preprocessing & Feature Engineering");
  const tbl = [
    [
      { text:"Step", options:{ bold:true, color:C.white, fill:{ color:C.wine }, align:"center" } },
      { text:"Action", options:{ bold:true, color:C.white, fill:{ color:C.wine }, align:"center" } },
      { text:"Rationale", options:{ bold:true, color:C.white, fill:{ color:C.wine }, align:"center" } },
    ],
    ["Outlier handling", "Winsorize chlorides & residual sugar\nat 1st / 99th percentile", "Reduce extreme-value influence"],
    ["Wine type indicator", "Encode red = 0, white = 1", "Red & white have different chemistry"],
    ["SO₂ ratio", "free SO₂ / (total SO₂ + ε)", "Effective preservation fraction"],
    ["Acidity ratio", "volatile acidity / (fixed acidity + ε)", "Relative spoilage-acidity contribution"],
    ["Standardization", "Zero mean, unit variance", "Required for LR & SVM; skipped for RF"],
  ];
  s.addTable(tbl, {
    x:0.35, y:1.02, w:9.3, h:3.45,
    colW:[1.8, 3.0, 4.5],
    border:{ pt:0.5, color:C.lightLine }, fill:{ color:C.navyMid },
    color:C.offWhite, fontSize:11, fontFace:"Calibri", valign:"middle",
    rowH:[0.38, 0.55, 0.55, 0.55, 0.55, 0.55],
  });
  card(s, 0.35, 4.6, 9.3, 0.72);
  s.addText([
    { text:"Final feature matrix: ", options:{ color:C.muted } },
    { text:"14 features ", options:{ bold:true, color:C.gold } },
    { text:"(11 original  +  3 engineered)     ", options:{ color:C.muted } },
    { text:"Scaler fit on training set only ", options:{ bold:true, color:C.wineLight } },
    { text:"— no data leakage", options:{ color:C.muted } },
  ], { x:0.5, y:4.66, w:9.0, h:0.6, fontSize:12, fontFace:"Calibri", margin:0 });
  s.addNotes("Five preprocessing steps. Outlier winsorization, wine-type indicator, SO2 ratio, acidity ratio, and standardization. All statistics computed on the training fold only — no leakage.");
}

// ── SLIDE 6 — EXPERIMENTAL PROTOCOL ──────────────────────────────────────────
{
  const s = pres.addSlide();
  darkBg(s);
  slideTitle(s, "Experimental Protocol — Rigor & Reproducibility");
  const splits = [
    { label:"TRAIN", n:"4,547", desc:"Model fitting,\nGridSearchCV", col:C.navyMid, accent:C.gold },
    { label:"VALIDATION", n:"975", desc:"Threshold tuning,\nmodel selection", col:C.navyMid, accent:C.wine },
    { label:"TEST (HOLDOUT)", n:"975", desc:"Reported once\nper model", col:"152338", accent:"2A7A4B" },
  ];
  splits.forEach((sp, i) => {
    const x = 0.35 + i * 3.1;
    card(s, x, 1.05, 2.95, 2.5, sp.col);
    s.addShape(pres.shapes.RECTANGLE, { x, y:1.05, w:2.95, h:0.38,
      fill:{ color:sp.accent }, line:{ color:sp.accent } });
    s.addText(sp.label, { x, y:1.05, w:2.95, h:0.38, fontSize:11,
      fontFace:"Calibri", bold:true, color:C.white,
      align:"center", valign:"middle", margin:0 });
    s.addText(sp.n, { x, y:1.5, w:2.95, h:0.7, fontSize:36, fontFace:"Calibri",
      bold:true, color:sp.accent, align:"center", valign:"middle", margin:0 });
    s.addText(sp.desc, { x, y:2.55, w:2.95, h:0.85, fontSize:11, fontFace:"Calibri",
      italic:true, color:C.muted, align:"center", valign:"middle", margin:4 });
  });
  card(s, 0.35, 3.7, 9.3, 1.65);
  sectionBadge(s, "TUNING & EVALUATION PROTOCOL", 0.35, 3.7, 3.0);
  s.addText([
    { text:"Hyperparameter search: ", options:{ bold:true, color:C.gold } },
    { text:"5-fold stratified CV inside training set, scoring=ROC-AUC\n", options:{ color:C.offWhite, breakLine:true } },
    { text:"class_weight ∈ {None, 'balanced'} ", options:{ bold:true, color:C.gold } },
    { text:"tuned for all 3 models (not just RF)\n", options:{ color:C.offWhite, breakLine:true } },
    { text:"Test metrics include: ", options:{ bold:true, color:C.gold } },
    { text:"ROC-AUC + PR-AUC, Brier score, 95% bootstrap CI, McNemar's test\n", options:{ color:C.offWhite, breakLine:true } },
    { text:"Multi-seed robustness: ", options:{ bold:true, color:C.gold } },
    { text:"5 seeds × full retraining → mean ± std", options:{ color:C.offWhite } },
  ], { x:0.55, y:4.13, w:9.0, h:1.2, fontSize:11, fontFace:"Calibri", paraSpaceAfter:3 });
  s.addNotes("The protocol has four pillars. First, stratified 70-15-15 split. Second, hyperparameter tuning happens entirely inside cross-validation on the training set. Third — and this is upgraded from my midway report — class_weight balanced is tuned for all three models, not just Random Forest. Fourth, I report not just point estimates but bootstrap confidence intervals, McNemar significance tests, and multi-seed robustness.");
}

// ── SLIDE 7 — MODEL 1: LR ────────────────────────────────────────────────────
{
  const s = pres.addSlide();
  darkBg(s, C.gold);
  slideTitle(s, "Model 1 — Logistic Regression with L2");
  card(s, 0.35, 1.05, 5.5, 3.45);
  sectionBadge(s, "HOW IT WORKS", 0.35, 1.05, 2.0);
  s.addText([
    { text:"Linear classifier — fits a weighted sum of features\n", options:{ color:C.offWhite } },
    { text:"ŷ = σ(wᵀx + b)\n\n", options:{ bold:true, color:C.gold, breakLine:true } },
    { text:"L2 penalty: ", options:{ bold:true, color:C.gold } },
    { text:"½C⁻¹ ‖w‖₂² added to cross-entropy\n", options:{ color:C.offWhite, breakLine:true } },
    { text:"Grid search: C × class_weight\n", options:{ color:C.muted, breakLine:true } },
    { text:"Best: C = 0.1, class_weight = balanced\n", options:{ bold:true, color:C.gold, breakLine:true } },
    { text:"Balanced weights compensate for 63/37 imbalance", options:{ italic:true, color:C.muted } },
  ], { x:0.55, y:1.48, w:5.1, h:2.8, fontSize:12.5, fontFace:"Calibri", paraSpaceAfter:5 });
  card(s, 6.1, 1.05, 3.6, 3.45);
  sectionBadge(s, "RESULTS", 6.1, 1.05, 1.6);
  const lrStats = [
    ["Best C", "0.1"], ["class_weight", "balanced"],
    ["CV AUC", "0.7999"], ["Test AUC", "0.8161"],
    ["Test PR-AUC", "0.8922"],
  ];
  lrStats.forEach(([k, v], i) => {
    const y = 1.55 + i * 0.54;
    s.addText(k, { x:6.2, y, w:2.0, h:0.42, fontSize:11.5, fontFace:"Calibri",
      color:C.muted, valign:"middle", margin:0 });
    s.addText(v, { x:8.0, y, w:1.6, h:0.42, fontSize:13.5, fontFace:"Calibri",
      bold:true, color:C.gold, align:"right", valign:"middle", margin:0 });
    if (i < lrStats.length-1) {
      s.addShape(pres.shapes.LINE, { x:6.2, y:y+0.43, w:3.3, h:0,
        line:{ color:C.lightLine, pt:0.5 } });
    }
  });
  card(s, 0.35, 4.6, 9.3, 0.72);
  s.addText([
    { text:"Why appropriate: ", options:{ bold:true, color:C.gold } },
    { text:"Interpretable linear baseline. Coefficient magnitudes directly quantify each feature's influence on quality. Establishes the upper bound of linear separability.",
      options:{ color:C.offWhite } }
  ], { x:0.5, y:4.66, w:9.0, h:0.58, fontSize:11.5, fontFace:"Calibri", margin:0 });
  s.addNotes("Logistic Regression is the interpretable linear baseline. The balanced class weight is the key tuning choice — without it, low-quality recall would be only 0.56. With balanced weighting, recall climbs to 0.74, more than compensating for the modest accuracy decrease.");
}

// ── SLIDE 8 — MODEL 2: SVM ────────────────────────────────────────────────────
{
  const s = pres.addSlide();
  darkBg(s, C.wine);
  slideTitle(s, "Model 2 — Support Vector Machine (RBF Kernel)");
  card(s, 0.35, 1.05, 5.5, 3.45);
  sectionBadge(s, "HOW IT WORKS", 0.35, 1.05, 2.0);
  s.addText([
    { text:"RBF kernel: k(x, x') = exp(−γ‖x − x'‖²)\n\n",
      options:{ bold:true, color:C.gold, breakLine:true } },
    { text:"Maps inputs into high-dim space implicitly\n",
      options:{ color:C.offWhite, breakLine:true } },
    { text:"Finds maximum-margin hyperplane in kernel space\n",
      options:{ color:C.muted, breakLine:true } },
    { text:"Grid: C × γ × class_weight\n",
      options:{ color:C.muted, breakLine:true } },
    { text:"Best: C = 1, γ = 0.1, class_weight = balanced\n",
      options:{ bold:true, color:C.gold, breakLine:true } },
    { text:"CV AUC jumps from 0.822 (no cw) to 0.828 with balanced cw",
      options:{ italic:true, color:C.muted } },
  ], { x:0.55, y:1.48, w:5.1, h:2.8, fontSize:12, fontFace:"Calibri", paraSpaceAfter:5 });
  card(s, 6.1, 1.05, 3.6, 3.45);
  sectionBadge(s, "RESULTS", 6.1, 1.05, 1.6);
  const svmStats = [
    ["Best C", "1"], ["Best γ", "0.1"],
    ["class_weight", "balanced"],
    ["CV AUC", "0.8276"], ["Test AUC", "0.8503"],
  ];
  svmStats.forEach(([k, v], i) => {
    const y = 1.55 + i * 0.54;
    s.addText(k, { x:6.2, y, w:2.0, h:0.42, fontSize:11.5, fontFace:"Calibri",
      color:C.muted, valign:"middle", margin:0 });
    s.addText(v, { x:8.0, y, w:1.6, h:0.42, fontSize:13.5, fontFace:"Calibri",
      bold:true, color:C.gold, align:"right", valign:"middle", margin:0 });
    if (i < svmStats.length-1) {
      s.addShape(pres.shapes.LINE, { x:6.2, y:y+0.43, w:3.3, h:0,
        line:{ color:C.lightLine, pt:0.5 } });
    }
  });
  card(s, 0.35, 4.6, 9.3, 0.72);
  s.addText([
    { text:"Why appropriate: ", options:{ bold:true, color:C.gold } },
    { text:"Captures non-linear chemical interactions implicitly. With balanced weighting, achieves the best low-quality recall (0.796) of any model — most defective bottles caught.",
      options:{ color:C.offWhite } }
  ], { x:0.5, y:4.66, w:9.0, h:0.58, fontSize:11.5, fontFace:"Calibri", margin:0 });
  s.addNotes("The RBF SVM also benefits from balanced class weighting. With balanced weights, the SVM achieves the best minority-class recall of any model — 0.796 — meaning it catches the most defective bottles. This becomes important when we discuss the practical interpretation later.");
}

// ── SLIDE 9 — MODEL 3: RF ─────────────────────────────────────────────────────
{
  const s = pres.addSlide();
  darkBg(s, "2A7A4B");
  slideTitle(s, "Model 3 — Random Forest Ensemble");
  card(s, 0.35, 1.05, 5.5, 3.45);
  sectionBadge(s, "HOW IT WORKS", 0.35, 1.05, 2.0);
  s.addText([
    { text:"300 decision trees on bootstrap samples\n", options:{ color:C.offWhite } },
    { text:"\nRandom feature subset at each split → decorrelated trees\n",
      options:{ color:C.offWhite, breakLine:true } },
    { text:"Aggregates by majority vote\n", options:{ color:C.muted, breakLine:true } },
    { text:"Grid: n × depth × class_weight\n", options:{ color:C.muted, breakLine:true } },
    { text:"Best: n=300, depth=None, cw=None\n",
      options:{ bold:true, color:C.gold, breakLine:true } },
    { text:"Bootstrap sampling already handles imbalance — no balanced cw needed",
      options:{ italic:true, color:C.muted } },
  ], { x:0.55, y:1.48, w:5.1, h:2.8, fontSize:12, fontFace:"Calibri", paraSpaceAfter:5 });
  card(s, 6.1, 1.05, 3.6, 3.45);
  sectionBadge(s, "RESULTS", 6.1, 1.05, 1.6);
  const rfStats = [
    ["Best n", "300"], ["Best depth", "None"],
    ["class_weight", "None"],
    ["CV AUC", "0.8814"], ["Test AUC", "0.9041"],
  ];
  rfStats.forEach(([k, v], i) => {
    const y = 1.55 + i * 0.54;
    s.addText(k, { x:6.2, y, w:2.0, h:0.42, fontSize:11.5, fontFace:"Calibri",
      color:C.muted, valign:"middle", margin:0 });
    s.addText(v, { x:8.0, y, w:1.6, h:0.42, fontSize:13.5, fontFace:"Calibri",
      bold:true, color:C.gold, align:"right", valign:"middle", margin:0 });
    if (i < rfStats.length-1) {
      s.addShape(pres.shapes.LINE, { x:6.2, y:y+0.43, w:3.3, h:0,
        line:{ color:C.lightLine, pt:0.5 } });
    }
  });
  card(s, 0.35, 4.6, 9.3, 0.72);
  s.addText([
    { text:"Notable: ", options:{ bold:true, color:C.gold } },
    { text:"RF was the only model where class_weight = None won — bootstrap sampling provides implicit balancing. RF achieves highest AUC, best Brier score (0.124), and highest low-class precision.",
      options:{ color:C.offWhite } }
  ], { x:0.5, y:4.66, w:9.0, h:0.58, fontSize:11.5, fontFace:"Calibri", margin:0 });
  s.addNotes("An interesting finding: RF was the only model where class_weight equals None won the grid search. This is because bootstrap sampling provides implicit balancing — each tree sees a re-sampled training set. RF achieves the highest overall AUC and, surprisingly, the best probability calibration as measured by the Brier score.");
}

// ── SLIDE 10 — TEST RESULTS WITH ROC FIGURE ──────────────────────────────────
{
  const s = pres.addSlide();
  darkBg(s, "2A7A4B");
  slideTitle(s, "Test Set Results — ROC + 95% Bootstrap CIs");
  // Left: ROC figure
  s.addImage({
    path: "/Users/moka/Documents/EE559/Project/figures/roc_test_all.png",
    x: 0.25, y: 1.05, w: 4.7, h: 3.9,
  });
  // Right: stat callouts
  card(s, 5.15, 1.05, 4.55, 3.9);
  sectionBadge(s, "TEST METRICS (RF ★)", 5.15, 1.05, 2.5);
  const rfMetrics = [
    ["Accuracy",     "0.829"],
    ["Macro F1",     "0.811"],
    ["ROC-AUC",      "0.904"],
    ["95% CI",       "[0.883, 0.922]"],
    ["PR-AUC",       "0.938"],
    ["Brier (↓)",    "0.124"],
  ];
  rfMetrics.forEach(([k, v], i) => {
    const y = 1.5 + i * 0.55;
    s.addText(k, { x:5.35, y, w:2.0, h:0.42, fontSize:12, fontFace:"Calibri",
      color:C.muted, valign:"middle", margin:0 });
    s.addText(v, { x:7.4, y, w:2.2, h:0.42, fontSize:14, fontFace:"Calibri",
      bold:true, color:C.gold, align:"right", valign:"middle", margin:0 });
    if (i < rfMetrics.length-1) {
      s.addShape(pres.shapes.LINE, { x:5.35, y:y+0.43, w:4.25, h:0,
        line:{ color:C.lightLine, pt:0.5 } });
    }
  });
  card(s, 0.25, 5.05, 9.45, 0.45);
  s.addText([
    { text:"All three AUC CIs are non-overlapping ", options:{ bold:true, color:C.gold } },
    { text:"→ ranking is statistically robust on this test set", options:{ color:C.offWhite } },
  ], { x:0.4, y:5.07, w:9.2, h:0.42, fontSize:11, fontFace:"Calibri",
       align:"center", valign:"middle", margin:0 });
  s.addNotes("These are the final test set ROC curves with their 95 percent bootstrap confidence intervals. The Random Forest curve dominates throughout the operating range. Random Forest achieves test ROC-AUC of 0.904 with confidence interval 0.883 to 0.922, PR-AUC of 0.938, and the lowest Brier score at 0.124. Critically, none of the AUC confidence intervals overlap between any two models, so the ranking is statistically robust.");
}

// ── SLIDE 11 — MCNEMAR + MULTI-SEED (with boxplot figure) ────────────────────
{
  const s = pres.addSlide();
  darkBg(s);
  slideTitle(s, "Statistical Significance & Multi-seed Robustness");
  // Left: McNemar table
  card(s, 0.35, 1.05, 4.5, 3.95);
  sectionBadge(s, "MCNEMAR'S TEST", 0.35, 1.05, 2.0);
  const mcTbl = [
    [
      { text:"Comparison", options:{ bold:true, color:C.white, fill:{ color:C.wine } } },
      { text:"χ²", options:{ bold:true, color:C.white, fill:{ color:C.wine }, align:"center" } },
      { text:"p-value", options:{ bold:true, color:C.white, fill:{ color:C.wine }, align:"center" } },
    ],
    ["LR vs SVM", "11.84", "5.8 × 10⁻⁴"],
    ["LR vs RF", "41.54", "1.2 × 10⁻¹⁰"],
    ["SVM vs RF", "19.38", "1.1 × 10⁻⁵"],
  ];
  s.addTable(mcTbl, {
    x:0.55, y:1.5, w:4.1, h:2.0, colW:[1.7, 1.0, 1.4],
    border:{ pt:0.5, color:C.lightLine },
    color:C.offWhite, fill:{ color:C.navyMid },
    fontSize:12, fontFace:"Calibri", valign:"middle",
    rowH:[0.4, 0.42, 0.42, 0.42],
  });
  s.addText([
    { text:"All p-values < 0.001\n", options:{ bold:true, color:C.gold, breakLine:true } },
    { text:"Pairwise differences are highly significant",
      options:{ italic:true, color:C.muted } },
  ], { x:0.55, y:3.7, w:4.1, h:1.0, fontSize:11.5, fontFace:"Calibri", margin:0 });

  // Right: boxplot figure
  card(s, 5.15, 1.05, 4.5, 3.95);
  sectionBadge(s, "MULTI-SEED BOXPLOT (5 SEEDS)", 5.15, 1.05, 2.8);
  s.addImage({
    path: "/Users/moka/Documents/EE559/Project/figures/multiseed_boxplot.png",
    x: 5.25, y: 1.5, w: 4.3, h: 3.0,
  });
  s.addText([
    { text:"Boxes are disjoint → ", options:{ bold:true, color:C.gold } },
    { text:"ranking is robust across splits", options:{ italic:true, color:C.muted } },
  ], { x:5.35, y:4.55, w:4.1, h:0.4, fontSize:10.5, fontFace:"Calibri", margin:0 });
  s.addNotes("Two independent statistical analyses confirm the ranking is real. On the left, McNemar's test on test-set predictions gives p-values below 0.001 for all three pairwise comparisons. On the right, the multi-seed boxplot shows test ROC-AUC across 5 different stratified splits — and the boxes are clearly disjoint between every pair of models. The ranking is not seed-dependent.");
}

// ── SLIDE 12 — PER-CLASS + OPERATIONAL TRADE-OFF ─────────────────────────────
{
  const s = pres.addSlide();
  darkBg(s, C.wine);
  slideTitle(s, "Per-Class — Operational Cost Trade-off");
  const pcTbl = [
    [
      { text:"Model", options:{ bold:true, color:C.white, fill:{ color:C.wine }, align:"center" } },
      { text:"Low Prec.", options:{ bold:true, color:C.white, fill:{ color:C.wine }, align:"center" } },
      { text:"Low Recall", options:{ bold:true, color:C.white, fill:{ color:C.wine }, align:"center" } },
      { text:"Low F1", options:{ bold:true, color:C.white, fill:{ color:C.wine }, align:"center" } },
      { text:"High F1", options:{ bold:true, color:C.white, fill:{ color:C.wine }, align:"center" } },
    ],
    [
      { text:"Logistic Reg.", options:{ color:C.offWhite, fill:{ color:C.navyMid } } },
      { text:"0.614", options:{ color:C.offWhite, fill:{ color:C.navyMid } } },
      { text:"0.743", options:{ color:C.offWhite, fill:{ color:C.navyMid } } },
      { text:"0.673", options:{ color:C.offWhite, fill:{ color:C.navyMid } } },
      { text:"0.777", options:{ color:C.offWhite, fill:{ color:C.navyMid } } },
    ],
    [
      { text:"SVM (RBF) ★", options:{ bold:true, color:"E8A040", fill:{ color:C.tableAlt } } },
      { text:"0.663", options:{ color:C.offWhite, fill:{ color:C.tableAlt } } },
      { text:"0.796", options:{ bold:true, color:"E8A040", fill:{ color:C.tableAlt } } },
      { text:"0.723", options:{ color:C.offWhite, fill:{ color:C.tableAlt } } },
      { text:"0.812", options:{ color:C.offWhite, fill:{ color:C.tableAlt } } },
    ],
    [
      { text:"Random Forest ★", options:{ bold:true, color:C.gold, fill:{ color:"1A3020" } } },
      { text:"0.801", options:{ bold:true, color:C.gold, fill:{ color:"1A3020" } } },
      { text:"0.710", options:{ color:C.offWhite, fill:{ color:"1A3020" } } },
      { text:"0.753", options:{ bold:true, color:C.gold, fill:{ color:"1A3020" } } },
      { text:"0.869", options:{ bold:true, color:C.gold, fill:{ color:"1A3020" } } },
    ],
  ];
  s.addTable(pcTbl, {
    x:0.35, y:1.02, w:9.3, h:2.4,
    colW:[2.4, 1.6, 1.7, 1.7, 1.9],
    border:{ pt:0.5, color:C.lightLine },
    fontSize:12, fontFace:"Calibri", valign:"middle",
    rowH:[0.42, 0.5, 0.5, 0.5],
  });
  card(s, 0.35, 3.55, 4.55, 1.75);
  sectionBadge(s, "WHEN TO PICK SVM", 0.35, 3.55, 2.5);
  s.addText([
    { text:"Highest low-quality recall (0.796)\n", options:{ bold:true, color:C.gold, breakLine:true } },
    { text:"Misses only 20% of defective bottles\n", options:{ color:C.offWhite, breakLine:true } },
    { text:"Best when: ", options:{ bold:true, color:C.gold } },
    { text:"FN cost (shipping defective wine) >> FP cost",
      options:{ color:C.offWhite } },
  ], { x:0.55, y:3.95, w:4.2, h:1.3, fontSize:11.5, fontFace:"Calibri", paraSpaceAfter:4 });
  card(s, 5.1, 3.55, 4.55, 1.75);
  sectionBadge(s, "WHEN TO PICK RANDOM FOREST", 5.1, 3.55, 3.0);
  s.addText([
    { text:"Highest low-quality precision (0.801)\n", options:{ bold:true, color:C.gold, breakLine:true } },
    { text:"80% of low-quality alarms are real\n", options:{ color:C.offWhite, breakLine:true } },
    { text:"Best when: ", options:{ bold:true, color:C.gold } },
    { text:"FP cost (re-testing good wine) >> FN cost",
      options:{ color:C.offWhite } },
  ], { x:5.3, y:3.95, w:4.2, h:1.3, fontSize:11.5, fontFace:"Calibri", paraSpaceAfter:4 });
  s.addNotes("This is one of the most interesting findings of the project. SVM with balanced class weights achieves the highest minority-class recall — it catches the most defective bottles. Random Forest achieves the highest minority-class precision — its alarms are most reliable. The right model depends on the cost ratio between shipping a defective bottle and re-testing a good one. This is a much richer story than just picking the model with the highest AUC.");
}

// ── SLIDE 13 — ALL-MODEL ABLATION (with figure) ──────────────────────────────
{
  const s = pres.addSlide();
  darkBg(s, C.gold);
  slideTitle(s, "All-Model Ablation — Who Benefits from Engineering?");
  // Top: ablation bar chart figure (full width)
  s.addImage({
    path: "/Users/moka/Documents/EE559/Project/figures/ablation_all_models.png",
    x: 0.5, y: 1.0, w: 6.5, h: 3.6,
  });
  // Right side: interpretation card
  card(s, 7.2, 1.0, 2.55, 3.6);
  sectionBadge(s, "PATTERN", 7.2, 1.0, 1.5);
  s.addText([
    { text:"LR\n", options:{ bold:true, color:C.gold, breakLine:true } },
    { text:"ΔAUC = +0.0075\n", options:{ color:C.offWhite, breakLine:true } },
    { text:"largest gain\n\n", options:{ italic:true, color:C.muted, breakLine:true } },
    { text:"SVM\n", options:{ bold:true, color:C.gold, breakLine:true } },
    { text:"ΔAUC ≈ 0\n", options:{ color:C.offWhite, breakLine:true } },
    { text:"RBF absorbs info\n\n", options:{ italic:true, color:C.muted, breakLine:true } },
    { text:"RF\n", options:{ bold:true, color:C.gold, breakLine:true } },
    { text:"ΔAUC = +0.0034\n", options:{ color:C.offWhite, breakLine:true } },
    { text:"trees rediscover", options:{ italic:true, color:C.muted } },
  ], { x:7.3, y:1.4, w:2.4, h:3.1, fontSize:11, fontFace:"Calibri",
       paraSpaceAfter:0, margin:0 });
  // Bottom takeaway bar
  card(s, 0.35, 4.75, 9.3, 0.55);
  s.addText([
    { text:"KEY INSIGHT: ", options:{ bold:true, color:C.gold } },
    { text:"benefit of feature engineering scales ", options:{ color:C.offWhite } },
    { text:"inversely with model capacity ", options:{ bold:true, italic:true, color:C.gold } },
    { text:"— exactly the predicted hierarchy", options:{ color:C.offWhite } },
  ], { x:0.5, y:4.78, w:9.1, h:0.5, fontSize:12, fontFace:"Calibri",
       align:"center", valign:"middle", margin:0 });
  s.addNotes("This is the single most important methodological insight of the project. The ablation bar chart shows test ROC-AUC for each model trained on the 11 base features alone versus all 14 features. Logistic Regression gets the largest benefit — delta-AUC of plus 0.008 — because the linear boundary cannot discover the ratio relationships on its own. SVM gets essentially zero because the RBF kernel implicitly captures the same information. Random Forest gets a small boost because deep trees can already rediscover most ratio relationships from raw inputs. The pattern is exactly the predicted hierarchy: feature engineering matters more for lower-capacity models.");
}

// ── SLIDE 14 — CALIBRATION + FEATURE IMPORTANCE (real figures) ──────────────
{
  const s = pres.addSlide();
  darkBg(s);
  slideTitle(s, "Calibration & Feature Importance");
  // Left: calibration figure
  sectionBadge(s, "RELIABILITY DIAGRAM (BRIER ↓)", 0.25, 1.0, 3.2);
  s.addImage({
    path: "/Users/moka/Documents/EE559/Project/figures/calibration_test.png",
    x: 0.3, y: 1.4, w: 4.5, h: 3.6,
  });
  // Right: RF importance figure
  sectionBadge(s, "RF FEATURE IMPORTANCE", 5.1, 1.0, 2.8);
  s.addImage({
    path: "/Users/moka/Documents/EE559/Project/figures/rf_feature_importance.png",
    x: 5.05, y: 1.4, w: 4.6, h: 3.6,
  });
  // Bottom takeaway bar
  card(s, 0.25, 5.05, 9.45, 0.45);
  s.addText([
    { text:"RF has lowest Brier (0.124) — best calibrated  |  ", options:{ bold:true, color:C.gold } },
    { text:"Alcohol dominates importance (matches EDA r = +0.44)", options:{ color:C.offWhite } },
  ], { x:0.4, y:5.07, w:9.2, h:0.42, fontSize:11, fontFace:"Calibri",
       align:"center", valign:"middle", margin:0 });
  s.addNotes("Two important secondary results, both shown as actual figures from the pipeline. On the left, the reliability diagram. The diagonal represents perfect calibration. Random Forest tracks the diagonal most closely — Brier 0.124. Logistic Regression systematically over-predicts in the low-probability bins because balanced class weighting pushes probabilities toward 0.5. On the right, Random Forest feature importances. Alcohol dominates as predicted by the EDA correlation of plus 0.44. The engineered acidity_ratio and so2_ratio features rank in the top-5, confirming their domain-motivated construction was useful.");
}

// ── SLIDE 15 — DISCUSSION & CONCLUSION ───────────────────────────────────────
{
  const s = pres.addSlide();
  s.background = { color: C.navyDark };
  s.addShape(pres.shapes.RECTANGLE, { x:0, y:0, w:10, h:0.06,
    fill:{ color:C.wine }, line:{ color:C.wine } });
  s.addShape(pres.shapes.RECTANGLE, { x:0, y:5.565, w:10, h:0.06,
    fill:{ color:C.wine }, line:{ color:C.wine } });
  s.addText("Discussion & Conclusion", {
    x:0.5, y:0.25, w:9, h:0.65, fontSize:30, fontFace:"Calibri",
    bold:true, color:C.white, margin:0
  });
  s.addShape(pres.shapes.RECTANGLE, { x:0.5, y:0.92, w:1.0, h:0.04,
    fill:{ color:C.gold }, line:{ color:C.gold } });
  const conclusions = [
    "Random Forest wins on overall AUC (0.904), F1, and calibration (Brier 0.124)",
    "SVM with balanced class weights wins on low-quality recall (0.796) — best for production quality gates",
    "All three pairwise differences are statistically significant (McNemar p < 0.001)",
    "Ranking is robust across 5 random seeds (non-overlapping ±2σ intervals)",
    "Engineered features benefit linear models most (LR ΔAUC = +0.008, RF ΔAUC = +0.003)",
    "Density confirmed redundant via VIF + ablation — could be dropped in deployment",
    "Limitations: binarization loses ordinal info; no producer/vintage covariates",
    "Future: MLP, SHAP attribution, ordinal regression, cost-sensitive learning with elicited FN/FP costs",
  ];
  s.addText(
    conclusions.map((t, i) => ({ text: t,
      options: { bullet: true, breakLine: i < conclusions.length - 1 } })),
    { x:0.5, y:1.05, w:9.0, h:3.7, fontSize:12, fontFace:"Calibri",
      color:C.offWhite, paraSpaceAfter:8 }
  );
  s.addShape(pres.shapes.RECTANGLE, { x:0.5, y:4.85, w:9.0, h:0.55,
    fill:{ color:C.wine }, line:{ color:C.wine }, shadow: makeShadow() });
  s.addText("Thank you!  Questions welcome.", {
    x:0.5, y:4.85, w:9.0, h:0.55, fontSize:18, fontFace:"Calibri",
    bold:true, color:C.white, align:"center", valign:"middle", margin:0
  });
  s.addText("Mengjia Shang  ·  7338151449  ·  EE559 Spring 2026  ·  Prof. Anand A. Joshi", {
    x:0.5, y:5.4, w:9.0, h:0.2, fontSize:9, fontFace:"Calibri",
    color:C.muted, align:"center", margin:0
  });
  s.addNotes("To summarize: Random Forest wins on overall metrics including AUC, F1, and calibration. But SVM with balanced class weights actually wins on minority-class recall — the most operationally important metric for a production quality gate. All pairwise differences are statistically significant under McNemar's test, and the ranking is robust across five random seeds. The all-model ablation confirms engineered features help linear models most. Density is confirmed redundant by both VIF and ablation. Thank you, I'm happy to take questions.");
}

// ── Write file ─────────────────────────────────────────────────────────────────
pres.writeFile({ fileName: "/Users/moka/Documents/EE559/Project/presentation.pptx" })
  .then(() => console.log("✅ presentation.pptx written"))
  .catch(e => { console.error("❌", e); process.exit(1); });
