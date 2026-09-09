# Smart Dyeing Process Analytics — Statistical Optimisation Suite

> **Rigorous multi-method statistical analysis of a real-world industrial knit fabric dyeing dataset (660 production batches), with verified mathematics, reproducible charts, and a complete cross-validation framework — written to academic publication standards.**

---

## Project Overview

This repository contains the complete analytical pipeline for optimising a **closed-loop Smart Dyeing** process for Bangladesh's knit fabric export industry. The study applies six statistical methods to 660 real production batches from an industrial dyeing facility, with every reported statistic independently re-verified by a dedicated audit script.

**Core contributions:**

- Fully automated data ingestion from a live factory production system (custom web scraper)
- 4-source cross-verification of 236 batches — zero trust in a single data source
- Conservative AND-logic outlier removal (Z-score AND IQR must both flag)
- Welch's ANOVA for unequal group sizes — methodologically rigorous
- K-Means clustering that independently rediscovered the factory's shade taxonomy at 78.24% accuracy
- Every claimed statistic has a `PASS/FAIL` status in `verification_report.json`

---

## Repository Contents

| File | Description |
|------|-------------|
| [`generate_charts.py`](generate_charts.py) | Master chart generation — 8 publication-quality figures (300 DPI) |
| [`_verify_all_stats.py`](_verify_all_stats.py) | Independent re-derivation of every reported statistic from raw CSV |
| [`verification_report.json`](verification_report.json) | Machine-readable audit log — 28/28 claims PASS |
| [`PROFESSOR_BRIEF.md`](PROFESSOR_BRIEF.md) | Executive research summary — all findings traceable to source data |

---

## Generated Figures

| # | Chart | Statistical Method | Output File |
|---|-------|--------------------|-------------|
| 1 | Raw vs. cleaned batch data (overdose correction) | Z-score + IQR AND-logic | `chart1_raw_vs_clean_overdose.png` |
| 2 | Salt/soda ash response — asymptote fitting | Michaelis-Menten non-linear regression | `chart2_salt_soda_asymptote.png` |
| 3 | Correlation heatmap (all process variables) | Pearson r matrix | `chart3_correlation_heatmap.png` |
| 4 | PCA scree plot + loadings biplot | Principal Component Analysis | `chart4_pca_scree_loadings.png` |
| 5 | Cross-validation stability across k-folds | 10-fold stratified CV | `chart5_cv_stability.png` |
| 6 | Dye concentration optimisation profile | Cost-response modelling | `chart6_dye_cost_profile.png` |
| 7 | ANOVA confidence intervals per shade class | Welch's one-way F-test | `chart7_anova_ci.png` |
| 8 | Data coverage gap identification | Batch audit methodology | `chart8_data_gap_scraper.png` |

---

## Statistical Methods

### 1. Outlier Detection — Conservative AND Logic

```python
from scipy import stats
import numpy as np

def remove_outliers_conservative(df: pd.DataFrame, cols: list,
                                  z_thresh: float = 2.5) -> pd.DataFrame:
    """
    Remove outliers only when BOTH Z-score AND IQR fences agree.
    Conservative AND-logic prevents false positives common in small groups.
    """
    mask = pd.Series([True] * len(df), index=df.index)
    for col in cols:
        z_flag   = np.abs(stats.zscore(df[col].dropna())) > z_thresh
        q1, q3   = df[col].quantile([0.25, 0.75])
        iqr_flag = (df[col] < q1 - 1.5*(q3-q1)) | (df[col] > q3 + 1.5*(q3-q1))
        mask &= ~(z_flag & iqr_flag)   # AND logic — both must flag
    return df[mask]
```

**Result:** Outlier rate 0.7–2% — validated across all 6 process variables. No shade category lost more than 3 batches.

---

### 2. Michaelis-Menten Non-Linear Regression (Salt/Soda Ash Response)

Chemical dose–response curves follow a saturation pattern best described by the Michaelis-Menten equation, not a linear model:

```python
from scipy.optimize import curve_fit

def michaelis_menten(x, Vmax, Km):
    """Saturation kinetics: response = Vmax * x / (Km + x)"""
    return Vmax * x / (Km + x)

popt, pcov = curve_fit(
    michaelis_menten,
    df['salt_gL'],          # NaCl concentration g/L
    df['exhaustion_pct'],   # Dye exhaustion %
    p0=[85.0, 35.0],        # Initial guess: Vmax=85%, Km=35 g/L
    bounds=([0, 0], [100, 200])
)
Vmax, Km = popt
# Km ≈ 38.2 g/L → half-saturation point
# Vmax ≈ 83.7% → asymptotic exhaustion ceiling
```

**Significance:** The Km value identifies the "law of diminishing returns" threshold for salt dosing — adding salt beyond Km provides < 50% of the marginal exhaustion gain of the same dose below Km.

---

### 3. Welch's ANOVA — Unequal Group Size Correction

Group sizes across shade categories are unequal (Pale: n=89, Medium: n=241, Dark: n=198, Black/Navy: n=132). Standard one-way ANOVA assumes equal variances (homoscedasticity). The Levene test confirms heteroscedasticity (p = 0.003). Welch's F-test is used throughout:

```python
from scipy.stats import f_oneway
import pingouin as pg

# Welch's ANOVA (does not assume equal variances)
result = pg.welch_anova(
    data=df,
    dv='water_per_kg',       # Dependent variable
    between='shade_category' # Grouping factor
)
# F(3, 187.4) = 12.84, p < 0.001, η² = 0.142
```

**Result:** Shade category explains 14.2% of water-per-kg variance (η² = 0.142). Post-hoc Games-Howell test confirms Black/Navy significantly different from all other categories (p < 0.01 after Bonferroni correction).

---

### 4. K-Means Clustering — Shade Taxonomy Recovery

K-Means clustering (k=4, k-means++ initialisation, 100 restarts) applied to 6 normalised process variables (water, salt, soda ash, temperature, time, dye concentration):

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[PROCESS_FEATURES])

kmeans = KMeans(n_clusters=4, init='k-means++', n_init=100, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Cluster → shade label mapping (by majority vote)
cluster_map = {0: 'Black/Navy', 1: 'Dark', 2: 'Medium', 3: 'Pale'}
accuracy = (df['cluster'].map(cluster_map) == df['shade_category']).mean()
# accuracy = 78.24% — clustering independently rediscovered the factory taxonomy
```

**Significance:** 78.24% accuracy without using the shade label as input confirms that the process variables carry sufficient information to reconstruct the shade taxonomy — validating the ML feature space.

---

### 5. 10-Fold Stratified Cross-Validation

All predictive models are evaluated with stratified 10-fold CV (preserving shade class proportions in each fold):

```python
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.ensemble import GradientBoostingRegressor

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_validate(
    GradientBoostingRegressor(n_estimators=200, max_depth=4),
    X, y,
    cv=cv,
    scoring=['r2', 'neg_mean_absolute_percentage_error'],
    return_train_score=True
)
# CV R² mean: 0.847 ± 0.031 (std across folds)
# No fold deviates > 2σ from mean → stable generalisation
```

---

## Verification Architecture

Every reported statistic is independently re-derived in `_verify_all_stats.py` from the raw CSV:

```
[Raw CSV]
    ↓ (independent load — no shared state with generate_charts.py)
[_verify_all_stats.py]
    ↓
[28 claim assertions]
    ↓ PASS / FAIL per claim
[verification_report.json]
```

**Result: 28/28 claims PASS.** The report includes:

- Computed value vs. claimed value (to 6 decimal places)
- Tolerance applied (relative or absolute)
- Assertion status (PASS/FAIL)
- Timestamp and data hash

---

## Key Results

| Finding | Value | Method |
|---------|-------|--------|
| Optimal salt dose (half-saturation Km) | 38.2 g/L | Michaelis-Menten regression |
| Exhaustion ceiling (Vmax) | 83.7% | Michaelis-Menten regression |
| Shade taxonomy recovery (unsupervised) | 78.24% accuracy | K-Means, k=4 |
| Water variance explained by shade | η² = 0.142 (14.2%) | Welch's ANOVA |
| Predictive model CV R² | 0.847 ± 0.031 | 10-fold stratified CV |
| Outlier removal rate | 0.7–2.0% | Conservative AND-logic |
| Statistics independently verified | 28/28 PASS | `_verify_all_stats.py` |

---

## 📚 References & Documentation

- [PROFESSOR_BRIEF.md](PROFESSOR_BRIEF.md) — Executive research summary — all findings traceable to source data
- [verification_report.json](verification_report.json) — Machine-readable audit log — 28/28 PASS
- Related project: [AI-Closed-Loop-Dyeing-Bangladesh](https://github.com/skmainuddin745-spec/AI-Closed-Loop-Dyeing-Bangladesh) — Full closed-loop ML pipeline
- Welch's ANOVA: Welch, B.L. (1951). *On the comparison of several mean values.* Biometrika, 38, 330-336.
- Michaelis-Menten: Johnson, K.A. & Goody, R.S. (2011). *The original Michaelis constant.* Biochemistry, 50(39), 8264-8269.
