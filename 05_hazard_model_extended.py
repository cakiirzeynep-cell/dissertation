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

print("Task 5 — Extended hazard model")

# 1. Load and apply exclusions 
panel = pd.read_csv(DATA / "artlogic_panel_enriched_v2.csv", low_memory=False)
panel["period_month"] = pd.to_datetime(panel["period_month"], format="%Y-%m")

last_month = panel["period_month"].max()
panel = panel.loc[
    (~panel["kept_changed_sub_flag"]) &
    (panel["period_month"] < last_month)
].copy()

need_cols = ["customer_type", "billing_period_months", "country",
             "tenure_months_recomp", "mrr_avg_month"]
panel = panel.dropna(subset=need_cols)
print(f"\nClean panel: {len(panel):,} rows · {panel['customer_id'].nunique():,} customers · "
      f"{int(panel['churned_next_month'].sum())} events")

# 2. Feature engineering 
def bin_tenure(t):
    if t < 3:    return "0-2"
    if t < 6:    return "3-5"
    if t < 12:   return "6-11"
    if t < 24:   return "12-23"
    if t < 36:   return "24-35"
    return "36+"
panel["tenure_bin"] = panel["tenure_months_recomp"].apply(bin_tenure)

top5_types = ["Gallery", "Collector", "Artist", "Art Dealer", "Art advisory"]
panel["customer_type_grouped"] = panel["customer_type"].apply(
    lambda x: x if x in top5_types else "Other"
)

def bin_billing(b):
    if b == 1:   return "monthly"
    if b == 3:   return "quarterly"
    if b == 12:  return "annual"
    return "other"
panel["billing_grouped"] = panel["billing_period_months"].apply(bin_billing)

top_countries = panel["country"].value_counts().head(5).index.tolist()
panel["country_grouped"] = panel["country"].apply(
    lambda x: x if x in top_countries else "Other"
)
panel["period_str"] = panel["period_month"].dt.strftime("%Y-%m")

# log(MRR + 1) — handles MRR=0 first-month rows safely
panel["log_mrr"] = np.log1p(panel["mrr_avg_month"])
print(f"\nlog_mrr distribution:")
print(f"  min:  {panel['log_mrr'].min():.2f}")
print(f"  mean: {panel['log_mrr'].mean():.2f}")
print(f"  med:  {panel['log_mrr'].median():.2f}")
print(f"  max:  {panel['log_mrr'].max():.2f}")

# 3. Time-aware split
TRAIN_END = pd.Timestamp("2025-06-30")
train = panel.loc[panel["period_month"] <= TRAIN_END].copy()
test  = panel.loc[panel["period_month"] >  TRAIN_END].copy()

print(f"\nTrain {len(train):,} rows · {int(train['churned_next_month'].sum())} events")
print(f"Test  {len(test):,} rows · {int(test['churned_next_month'].sum())} events")

# 4. Design matrix with extensions
def build_design(df, fit_columns=None, with_interaction=True):
    X_cat = pd.get_dummies(
        df[["tenure_bin", "customer_type_grouped", "billing_grouped",
            "country_grouped", "period_str"]],
        prefix={"tenure_bin": "ten", "customer_type_grouped": "ct",
                "billing_grouped": "bill", "country_grouped": "ctry",
                "period_str": "mo"},
        drop_first=False
    )
    refs = ["ten_12-23", "ct_Gallery", "bill_quarterly", "ctry_US"]
    for ref in refs:
        if ref in X_cat.columns:
            X_cat = X_cat.drop(columns=ref)
    period_cols = sorted([c for c in X_cat.columns if c.startswith("mo_")])
    if period_cols:
        X_cat = X_cat.drop(columns=period_cols[0])

    X_cat["is_cross_platform"] = df["is_cross_platform"].astype(float).values
    X_cat["log_mrr"] = df["log_mrr"].values  

    # is_cross_platform × tenure_bin interactions (5 terms; tenure 12-23 is the reference, so its interaction is the omitted baseline)
    if with_interaction:
        for bin_name in ["0-2", "3-5", "6-11", "24-35", "36+"]:
            col = f"ten_{bin_name}"
            if col in X_cat.columns:
                X_cat[f"is_cp_x_{col}"] = X_cat["is_cross_platform"] * X_cat[col]

    X = sm.add_constant(X_cat, has_constant="add").astype(float)
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
print(f"\nExtended design matrix: {X_train.shape[1]} columns")

# 5. Fit logit and cloglog
print("\nLogit (extended)")
logit_fit = GLM(y_train, X_train, family=Binomial(link=Logit())).fit()
print(f"  Converged: {logit_fit.converged} · Deviance {logit_fit.deviance:.2f} · "
      f"AIC {logit_fit.aic:.2f} · LL {logit_fit.llf:.2f}")

print("\nCloglog (extended)")
cloglog_fit = GLM(y_train, X_train, family=Binomial(link=CLogLog())).fit()
print(f"  Converged: {cloglog_fit.converged} · Deviance {cloglog_fit.deviance:.2f} · "
      f"AIC {cloglog_fit.aic:.2f} · LL {cloglog_fit.llf:.2f}")

# 6. Test performance
p_logit_test   = logit_fit.predict(X_test)
p_cloglog_test = cloglog_fit.predict(X_test)
auc_l   = roc_auc_score(y_test, p_logit_test)
brier_l = brier_score_loss(y_test, p_logit_test)
auc_c   = roc_auc_score(y_test, p_cloglog_test)
brier_c = brier_score_loss(y_test, p_cloglog_test)
print(f"\nLogit   test:  AUC {auc_l:.4f}  Brier {brier_l:.5f}")
print(f"Cloglog test:  AUC {auc_c:.4f}  Brier {brier_c:.5f}")

# Baseline deviance and AIC for comparison
TASK4_LOGIT_DEVIANCE = 9108.88
TASK4_LOGIT_AIC      = 9190.88
TASK4_LOGIT_AUC      = 0.5887

delta_dev = TASK4_LOGIT_DEVIANCE - logit_fit.deviance
delta_aic = TASK4_LOGIT_AIC - logit_fit.aic
delta_auc = auc_l - TASK4_LOGIT_AUC
print(f"\nDelta vs baseline (positive = extended model better):")
print(f"  Delta deviance: {delta_dev:+.2f}")
print(f"  Delta AIC:      {delta_aic:+.2f}")
print(f"  Delta test AUC: {delta_auc:+.4f}")

# 7. Coefficient table for logit (extended)
coef_logit = pd.DataFrame({
    "covariate": logit_fit.params.index,
    "coef": logit_fit.params.values,
    "std_err": logit_fit.bse.values,
    "z": logit_fit.tvalues.values,
    "p_value": logit_fit.pvalues.values,
    "exp_coef": np.exp(logit_fit.params.values),
})
coef_logit.to_csv(OUT / "05_hazard_extended_logit_coefficients.csv", index=False)

# Highlight non-period covariates with p < 0.05
substantive = coef_logit[
    ~coef_logit["covariate"].str.startswith("mo_") &
    (coef_logit["covariate"] != "const")
].copy()
substantive["sig"] = substantive["p_value"].apply(
    lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
)
substantive_sig = substantive[substantive["p_value"] < 0.05].copy()
substantive_sig["abs_z"] = substantive_sig["z"].abs()
substantive_sig = substantive_sig.sort_values("abs_z", ascending=False)
print(f"\nSignificant covariates (p<0.05), sorted by |z|:")
print(substantive_sig[["covariate", "coef", "exp_coef", "p_value", "sig"]]
      .round(4).to_string(index=False))

# 8. Average Marginal Effects for all substantive covariates
print("\nComputing Average Marginal Effects on training set")
ames = []

p_train_base = logit_fit.predict(X_train)

continuous_cols = ["log_mrr"]
non_period_cols = [c for c in X_train.columns
                   if c != "const" and not c.startswith("mo_")]

for col in non_period_cols:
    beta_k = logit_fit.params[col]
    se_k   = logit_fit.bse[col]
    if col in continuous_cols:
        ame_vals = beta_k * p_train_base * (1 - p_train_base)
        ame_mean = ame_vals.mean()
        ame_se = abs(se_k) * (p_train_base * (1 - p_train_base)).mean()
    else:
        # Binary/dummy AME: counterfactual
        X1 = X_train.copy(); X1[col] = 1.0
        X0 = X_train.copy(); X0[col] = 0.0
        p1 = logit_fit.predict(X1)
        p0 = logit_fit.predict(X0)
        diff = p1 - p0
        ame_mean = diff.mean()
        ame_se = diff.std() / np.sqrt(len(diff))

    ames.append({
        "covariate": col,
        "coef": beta_k,
        "exp_coef": np.exp(beta_k),
        "ame": ame_mean,
        "ame_pct_points": ame_mean * 100,
        "p_value": logit_fit.pvalues[col],
    })

ames_df = pd.DataFrame(ames)
ames_df["abs_ame"] = ames_df["ame"].abs()
ames_df = ames_df.sort_values("abs_ame", ascending=False)
ames_df.to_csv(OUT / "05_hazard_extended_ames.csv", index=False)

print(f"\nTop AMEs (sorted by magnitude, percentage points of monthly churn probability):")
print(ames_df[["covariate", "coef", "exp_coef", "ame_pct_points", "p_value"]]
      .head(15).round(4).to_string(index=False))

# 9. Interaction analysis: is_cross_platform × tenure_bin
print("MECHANISM TEST — cross-platform effect by tenure bin")
# Main effect of is_cross_platform = effect at the reference tenure (12-23)
# Interaction terms add to this for each non-reference tenure bin
print("\n(Reference tenure bin: 12-23 months — its cross-platform effect is given by the is_cross_platform main coefficient alone.)")

base_cp = logit_fit.params.get("is_cross_platform", np.nan)
base_cp_se = logit_fit.bse.get("is_cross_platform", np.nan)

interaction_rows = []
for bin_name in ["0-2", "3-5", "6-11", "12-23", "24-35", "36+"]:
    int_col = f"is_cp_x_ten_{bin_name}"
    if bin_name == "12-23":
        total_coef = base_cp
        total_se = base_cp_se
        int_coef = 0.0
        int_p = np.nan
    else:
        int_coef = logit_fit.params.get(int_col, 0.0)
        int_se   = logit_fit.bse.get(int_col, 0.0)
        int_p    = logit_fit.pvalues.get(int_col, np.nan)
        total_coef = base_cp + int_coef
        # Approximate SE of sum (assuming independence — could be tightened with covariance)
        total_se = np.sqrt(base_cp_se**2 + int_se**2)
    interaction_rows.append({
        "tenure_bin": bin_name,
        "is_cp_coef_total": total_coef,
        "is_cp_HR_total": np.exp(total_coef),
        "interaction_coef": int_coef,
        "interaction_p": int_p,
        "approx_se": total_se,
    })

interaction_df = pd.DataFrame(interaction_rows)
interaction_df.to_csv(OUT / "05_hazard_extended_interaction.csv", index=False)
print(interaction_df.round(4).to_string(index=False))

print("\nMechanism interpretation:")
hrs = interaction_df.set_index("tenure_bin")["is_cp_HR_total"].to_dict()
early_hr = hrs.get("0-2", np.nan)
mid_hr   = hrs.get("12-23", np.nan)
mature_hr = hrs.get("36+", np.nan)
print(f"  HR at 0-2 months (newest):    {early_hr:.2f}")
print(f"  HR at 12-23 months (mid):     {mid_hr:.2f}")
print(f"  HR at 36+ months (mature):    {mature_hr:.2f}")
if early_hr > mid_hr > mature_hr:
    print(f"  Pattern consistent with integration-friction mechanism:")
    print(f"    cross-platform effect strongest in early tenure, attenuates over time.")
elif early_hr > 1.0 and mid_hr > 1.0 and mature_hr > 1.0:
    print(f"  Cross-platform effect elevated across all tenures (no clear attenuation).")
elif all(h < 1.0 for h in [early_hr, mid_hr, mature_hr]):
    print(f"  Cross-platform effect protective across all tenures (unexpected direction).")
else:
    print(f"  Mixed pattern — see table above for full interpretation.")

# 10. Save diagnostics summary
diag = pd.DataFrame({
    "metric": ["train_deviance", "train_AIC", "train_llf",
               "test_AUC_logit", "test_Brier_logit",
               "test_AUC_cloglog", "test_Brier_cloglog",
               "n_covariates", "delta_deviance_vs_task4",
               "delta_AIC_vs_task4", "delta_AUC_vs_task4"],
    "value": [logit_fit.deviance, logit_fit.aic, logit_fit.llf,
              auc_l, brier_l,
              auc_c, brier_c,
              X_train.shape[1], delta_dev, delta_aic, delta_auc],
})
diag.to_csv(OUT / "05_hazard_extended_diagnostics.csv", index=False)

print("\nVERDICT")
log_mrr_coef = logit_fit.params.get("log_mrr", np.nan)
log_mrr_p    = logit_fit.pvalues.get("log_mrr", np.nan)
log_mrr_ame  = ames_df.loc[ames_df["covariate"] == "log_mrr", "ame_pct_points"].values
log_mrr_ame  = log_mrr_ame[0] if len(log_mrr_ame) else np.nan
print(f"log_mrr coefficient: {log_mrr_coef:+.4f} (p={log_mrr_p:.4f}, AME={log_mrr_ame:+.4f} pp)")
print(f"Delta deviance vs baseline: {delta_dev:+.2f}")
print(f"Delta AIC vs baseline:      {delta_aic:+.2f}")
print(f"Delta test AUC vs baseline: {delta_auc:+.4f}")
