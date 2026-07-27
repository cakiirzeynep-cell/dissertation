import warnings
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.families.links import Logit, CLogLog
from sklearn.metrics import roc_auc_score, brier_score_loss

warnings.filterwarnings("ignore")

DATA = Path("/Users/zeynepcakir/Desktop/msc dissertation/data files ")
OUT  = Path("/Users/zeynepcakir/Desktop/msc dissertation/analysis/output")

print("Task 4 — First-pass discrete-time hazard model")

# 1. Load + apply exclusions
panel = pd.read_csv(DATA / "artlogic_panel_enriched_v2.csv", low_memory=False)
panel["period_month"] = pd.to_datetime(panel["period_month"], format="%Y-%m")

print(f"\nRaw enriched panel: {panel.shape}")

last_month = panel["period_month"].max()
panel = panel.loc[
    (~panel["kept_changed_sub_flag"]) &
    (panel["period_month"] < last_month)
].copy()
print(f"After exclusions (kept-changed + final month): {panel.shape}")
print(f"  Customers: {panel['customer_id'].nunique():,}")
print(f"  Churn events: {int(panel['churned_next_month'].sum()):,}")

# Drop rows with missing key covariates 
need_cols = ["customer_type", "billing_period_months", "country", "tenure_months_recomp"]
before = len(panel)
panel = panel.dropna(subset=need_cols)
print(f"After dropping rows missing key covariates: {len(panel):,} ({before-len(panel)} dropped)")

# 2. Feature engineering
def bin_tenure(t):
    if t < 3:    return "0-2"
    if t < 6:    return "3-5"
    if t < 12:   return "6-11"
    if t < 24:   return "12-23"
    if t < 36:   return "24-35"
    return "36+"
panel["tenure_bin"] = panel["tenure_months_recomp"].apply(bin_tenure)
print(f"\nTenure bin distribution:")
print(panel["tenure_bin"].value_counts().to_string())

# Customer type: top 5 + Other
top5_types = ["Gallery", "Collector", "Artist", "Art Dealer", "Art advisory"]
panel["customer_type_grouped"] = panel["customer_type"].apply(
    lambda x: x if x in top5_types else "Other"
)
print(f"\nCustomer type (grouped) distribution:")
print(panel["customer_type_grouped"].value_counts().to_string())

# Billing period: 1, 3, 12, Other
def bin_billing(b):
    if b == 1:   return "monthly"
    if b == 3:   return "quarterly"
    if b == 12:  return "annual"
    return "other"
panel["billing_grouped"] = panel["billing_period_months"].apply(bin_billing)
print(f"\nBilling group distribution:")
print(panel["billing_grouped"].value_counts().to_string())

# Country: top regions + Other
top_countries = panel["country"].value_counts().head(5).index.tolist()
print(f"\nTop 5 countries: {top_countries}")
panel["country_grouped"] = panel["country"].apply(
    lambda x: x if x in top_countries else "Other"
)
print(panel["country_grouped"].value_counts().to_string())

# Calendar-month dummies via period_month as categorical
panel["period_str"] = panel["period_month"].dt.strftime("%Y-%m")

# 3. Time-aware split
TRAIN_END = pd.Timestamp("2025-06-30")
train = panel.loc[panel["period_month"] <= TRAIN_END].copy()
test  = panel.loc[panel["period_month"] >  TRAIN_END].copy()

print(f"\nTime-aware split")
print(f"Train (Sep 2023 – Jun 2025): {len(train):>6,} rows, "
      f"{train['churned_next_month'].sum():>4} churn events "
      f"({train['churned_next_month'].mean()*100:.2f}% rate)")
print(f"Test  (Jul 2025 – Apr 2026): {len(test):>6,} rows, "
      f"{test['churned_next_month'].sum():>4} churn events "
      f"({test['churned_next_month'].mean()*100:.2f}% rate)")

# 4. Design matrix construction
# Reference categories (chosen for interpretability):
#   tenure_bin: "12-23" (mature-but-not-old; median customer)
#   customer_type_grouped: "Gallery" (largest segment)
#   billing_grouped: "quarterly" (modal billing)
#   country_grouped: "US" (modal country, presumably)

# Use C() factors with explicit references via patsy-style dummies
def build_design(df, fit_columns=None):
    """One-hot encode categoricals; align columns to fit_columns if provided."""
    X = pd.get_dummies(
        df[["tenure_bin", "customer_type_grouped", "billing_grouped",
            "country_grouped", "period_str"]],
        prefix={"tenure_bin": "ten", "customer_type_grouped": "ct",
                "billing_grouped": "bill", "country_grouped": "ctry",
                "period_str": "mo"},
        drop_first=False
    )
    # Drop reference categories explicitly
    refs = ["ten_12-23", "ct_Gallery", "bill_quarterly", "ctry_US"]
    for ref in refs:
        if ref in X.columns:
            X = X.drop(columns=ref)
    # Drop first month dummy as period reference
    period_cols = sorted([c for c in X.columns if c.startswith("mo_")])
    if period_cols:
        X = X.drop(columns=period_cols[0])
    X["is_cross_platform"] = df["is_cross_platform"].values
    X = sm.add_constant(X, has_constant="add")
    # Coerce all to float (booleans from get_dummies need conversion)
    X = X.astype(float)
    if fit_columns is not None:
        for col in fit_columns:
            if col not in X.columns:
                X[col] = 0.0
        X = X[fit_columns]
    return X

X_train = build_design(train)
y_train = train["churned_next_month"].astype(float).values
X_test  = build_design(test, fit_columns=X_train.columns.tolist())
y_test  = test["churned_next_month"].astype(float).values

print(f"\nDesign matrix: {X_train.shape[1]} columns")
print(f"  Train: {X_train.shape}")
print(f"  Test:  {X_test.shape}")

# 5. Fit both models
print("\nFitting logit model")
logit_model = GLM(y_train, X_train, family=Binomial(link=Logit()))
logit_fit = logit_model.fit()
print(f"Converged: {logit_fit.converged}")
print(f"Train deviance: {logit_fit.deviance:.2f}")
print(f"AIC:            {logit_fit.aic:.2f}")
print(f"Log-likelihood: {logit_fit.llf:.2f}")

print("\nFitting cloglog model")
cloglog_model = GLM(y_train, X_train, family=Binomial(link=CLogLog()))
cloglog_fit = cloglog_model.fit()
print(f"Converged: {cloglog_fit.converged}")
print(f"Train deviance: {cloglog_fit.deviance:.2f}")
print(f"AIC:            {cloglog_fit.aic:.2f}")
print(f"Log-likelihood: {cloglog_fit.llf:.2f}")

# 6. Held-out performance
print("\nHeld-out test performance")
p_logit_test   = logit_fit.predict(X_test)
p_cloglog_test = cloglog_fit.predict(X_test)

def perf(y, p, label):
    auc = roc_auc_score(y, p)
    brier = brier_score_loss(y, p)
    print(f"  {label:>10s}: AUC = {auc:.4f}  Brier = {brier:.5f}")
    return auc, brier

auc_l, brier_l = perf(y_test, p_logit_test, "logit")
auc_c, brier_c = perf(y_test, p_cloglog_test, "cloglog")

# 7. Per-segment held-out performance
print("\nPer-segment AUC + Brier (held-out test)")
test_eval = test.copy()
test_eval["p_logit"]   = p_logit_test
test_eval["p_cloglog"] = p_cloglog_test

def seg_perf(df, group_col, min_events=5):
    rows = []
    for grp, g in df.groupby(group_col):
        n_events = int(g["churned_next_month"].sum())
        n_obs    = len(g)
        if n_events < min_events:
            rows.append({group_col: grp, "n_obs": n_obs, "n_events": n_events,
                         "auc_logit": np.nan, "auc_cloglog": np.nan,
                         "brier_logit": np.nan, "brier_cloglog": np.nan})
            continue
        rows.append({
            group_col: grp,
            "n_obs": n_obs,
            "n_events": n_events,
            "auc_logit":   roc_auc_score(g["churned_next_month"], g["p_logit"]),
            "auc_cloglog": roc_auc_score(g["churned_next_month"], g["p_cloglog"]),
            "brier_logit":   brier_score_loss(g["churned_next_month"], g["p_logit"]),
            "brier_cloglog": brier_score_loss(g["churned_next_month"], g["p_cloglog"]),
        })
    return pd.DataFrame(rows).sort_values("n_events", ascending=False)

seg_ct  = seg_perf(test_eval, "customer_type_grouped")
seg_bill = seg_perf(test_eval, "billing_grouped")
seg_ten  = seg_perf(test_eval, "tenure_bin")
seg_cp   = seg_perf(test_eval, "is_cross_platform")

print("\nBy customer_type:")
print(seg_ct.round(4).to_string(index=False))
print("\nBy billing:")
print(seg_bill.round(4).to_string(index=False))
print("\nBy tenure bin:")
print(seg_ten.round(4).to_string(index=False))
print("\nBy cross-platform:")
print(seg_cp.round(4).to_string(index=False))

# 8. Save outputs
def coef_table(fit, name):
    out = pd.DataFrame({
        "covariate": fit.params.index,
        "coef": fit.params.values,
        "std_err": fit.bse.values,
        "z": fit.tvalues.values,
        "p_value": fit.pvalues.values,
        "exp_coef": np.exp(fit.params.values),
    })
    return out

logit_coefs   = coef_table(logit_fit, "logit")
cloglog_coefs = coef_table(cloglog_fit, "cloglog")
logit_coefs.to_csv(OUT / "04_hazard_logit_coefficients.csv", index=False)
cloglog_coefs.to_csv(OUT / "04_hazard_cloglog_coefficients.csv", index=False)

diag = pd.DataFrame({
    "metric": ["train_deviance", "train_AIC", "train_llf",
               "test_AUC", "test_Brier", "test_n_events", "test_rate"],
    "logit":   [logit_fit.deviance, logit_fit.aic, logit_fit.llf,
                auc_l, brier_l, int(y_test.sum()), y_test.mean()],
    "cloglog": [cloglog_fit.deviance, cloglog_fit.aic, cloglog_fit.llf,
                auc_c, brier_c, int(y_test.sum()), y_test.mean()],
})
diag.to_csv(OUT / "04_hazard_model_diagnostics.csv", index=False)

# Combined segment performance
seg_combined = pd.concat([
    seg_ct.assign(dimension="customer_type").rename(columns={"customer_type_grouped": "segment"}),
    seg_bill.assign(dimension="billing").rename(columns={"billing_grouped": "segment"}),
    seg_ten.assign(dimension="tenure_bin").rename(columns={"tenure_bin": "segment"}),
    seg_cp.assign(dimension="is_cross_platform").rename(columns={"is_cross_platform": "segment"}),
], ignore_index=True)
seg_combined.to_csv(OUT / "04_per_segment_heldout_performance.csv", index=False)

# 9. Verdict
print("\nMODEL COMPARISON VERDICT")
print(f"Train deviance:   logit {logit_fit.deviance:.2f}  vs  cloglog {cloglog_fit.deviance:.2f}")
print(f"  Delta:          {logit_fit.deviance - cloglog_fit.deviance:+.2f} (negative = logit fits better)")
print(f"Train AIC:        logit {logit_fit.aic:.2f}  vs  cloglog {cloglog_fit.aic:.2f}")
print(f"Test AUC:         logit {auc_l:.4f}  vs  cloglog {auc_c:.4f}")
print(f"Test Brier:       logit {brier_l:.5f}  vs  cloglog {brier_c:.5f}")

if logit_fit.deviance < cloglog_fit.deviance:
    primary = "logit"
else:
    primary = "cloglog"
print(f"\nPRIMARY (better-fitting): {primary.upper()}")

print(f"\nAll outputs saved to: {OUT}")
