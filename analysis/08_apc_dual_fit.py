import warnings
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.families.links import Logit

warnings.filterwarnings("ignore")

DATA = Path(__file__).resolve().parent.parent / "data"
OUT  = Path(__file__).resolve().parent / "output"

# APC dual-fit for SQ3
# 1. Load + prepare data
panel = pd.read_csv(DATA / "artlogic_panel_enriched_v2.csv", low_memory=False)
panel["period_month"] = pd.to_datetime(panel["period_month"], format="%Y-%m")
panel["activation_date"] = pd.to_datetime(panel["activation_date"], errors="coerce")

last_month = panel["period_month"].max()
panel = panel.loc[
    (~panel["kept_changed_sub_flag"]) &
    (panel["period_month"] < last_month)
].copy()
panel = panel.dropna(subset=["customer_type", "tenure_months_recomp",
                              "activation_date", "churned_next_month"])

# Age (tenure bin)
def bin_tenure(t):
    if t < 3: return "0-2"
    if t < 6: return "3-5"
    if t < 12: return "6-11"
    if t < 24: return "12-23"
    if t < 36: return "24-35"
    return "36+"
panel["tenure_bin"] = panel["tenure_months_recomp"].apply(bin_tenure)

# Cohort (activation year)
panel["cohort_year"] = panel["activation_date"].dt.year.astype("Int64")

top5_types = ["Gallery", "Collector", "Artist", "Art Dealer", "Art advisory"]
panel["customer_type_grouped"] = panel["customer_type"].apply(
    lambda x: x if x in top5_types else "Other"
)
panel["period_str"] = panel["period_month"].dt.strftime("%Y-%m")

print(f"\nPanel: {len(panel):,} rows, "
      f"{panel['customer_id'].nunique():,} customers, "
      f"{int(panel['churned_next_month'].sum())} events")
print(f"Activation cohorts: {sorted(panel['cohort_year'].dropna().unique().tolist())}")
print(f"Period range: {panel['period_str'].min()} to {panel['period_str'].max()}")

# 2. Design-matrix builder
REF_TENURE  = "12-23"
REF_COHORT  = 2020
REF_TYPE    = "Gallery"


def build_apc_design(df, fit_columns=None):
    df = df.copy()
    df["cohort_str"] = df["cohort_year"].astype(str)

    X = pd.get_dummies(
        df[["tenure_bin", "period_str", "cohort_str", "customer_type_grouped"]],
        prefix={"tenure_bin": "age", "period_str": "per",
                "cohort_str": "coh", "customer_type_grouped": "ct"},
        drop_first=False,
    )

    # Reference categories
    refs = [
        f"age_{REF_TENURE}",
        f"coh_{REF_COHORT}",
        f"ct_{REF_TYPE}",
    ]
    for ref in refs:
        if ref in X.columns:
            X = X.drop(columns=ref)

    # Drop the earliest period as the reference
    period_cols = sorted([c for c in X.columns if c.startswith("per_")])
    if period_cols:
        X = X.drop(columns=period_cols[0])

    X = sm.add_constant(X, has_constant="add").astype(float)

    if fit_columns is not None:
        for c in fit_columns:
            if c not in X.columns:
                X[c] = 0.0
        X = X[fit_columns]
    return X


def fit_apc(df, label):
    X = build_apc_design(df)
    y = df["churned_next_month"].astype(float).values
    print(f"\n[{label}] design columns: {X.shape[1]}, rows: {X.shape[0]:,}, events: {int(y.sum())}")
    fit = GLM(y, X, family=Binomial(link=Logit())).fit(maxiter=500, tol=1e-8)
    print(f"  Converged: {fit.converged}, Deviance: {fit.deviance:.2f}, AIC: {fit.aic:.2f}")
    if not fit.converged:
        print(f"  [WARN] Non-convergence may indicate sparse cohort×period cells. "
              f"Coefficient estimates retained but should be reported with this caveat.")
    return fit, X.columns.tolist()

# 3. Full APC fit
print("\nFull APC fit (all customers)")
full_fit, full_cols = fit_apc(panel, "FULL")

full_coefs = pd.DataFrame({
    "covariate": full_fit.params.index,
    "coef": full_fit.params.values,
    "std_err": full_fit.bse.values,
    "z": full_fit.tvalues.values,
    "p_value": full_fit.pvalues.values,
    "exp_coef": np.exp(full_fit.params.values),
})
full_coefs.to_csv(OUT / "08_apc_full_coefficients.csv", index=False)

# 4. Restricted APC fit (activations ≥ 5 Jan 2023)
print("\nRestricted APC fit (activations ≥ 2023-01-05)")
restricted = panel.loc[panel["activation_date"] >= pd.Timestamp("2023-01-05")].copy()
print(f"  Customers retained: {restricted['customer_id'].nunique():,}")
print(f"  Cohorts in restricted set: {sorted(restricted['cohort_year'].dropna().unique().tolist())}")
restricted_fit, restricted_cols = fit_apc(restricted, "RESTRICTED")

restricted_coefs = pd.DataFrame({
    "covariate": restricted_fit.params.index,
    "coef": restricted_fit.params.values,
    "std_err": restricted_fit.bse.values,
    "z": restricted_fit.tvalues.values,
    "p_value": restricted_fit.pvalues.values,
    "exp_coef": np.exp(restricted_fit.params.values),
})
restricted_coefs.to_csv(OUT / "08_apc_restricted_coefficients.csv", index=False)

# 5. Sensitivity APC fit (activations ≥ 17 Dec 2022)
print("\nSensitivity APC fit (activations ≥ 2022-12-17)")
sensitivity = panel.loc[panel["activation_date"] >= pd.Timestamp("2022-12-17")].copy()
print(f"  Customers retained: {sensitivity['customer_id'].nunique():,}")
sensitivity_fit, sensitivity_cols = fit_apc(sensitivity, "SENSITIVITY")

sensitivity_coefs = pd.DataFrame({
    "covariate": sensitivity_fit.params.index,
    "coef": sensitivity_fit.params.values,
    "std_err": sensitivity_fit.bse.values,
    "z": sensitivity_fit.tvalues.values,
    "p_value": sensitivity_fit.pvalues.values,
    "exp_coef": np.exp(sensitivity_fit.params.values),
})
sensitivity_coefs.to_csv(OUT / "08_apc_sensitivity_coefficients.csv", index=False)

# 6. Period-coefficient comparison — the SQ3 deliverable
# Extract period coefficients from each fit
def period_coefs(fit, label):
    rows = []
    for name in fit.params.index:
        if name.startswith("per_"):
            rows.append({
                "period": name.replace("per_", ""),
                f"coef_{label}": fit.params[name],
                f"se_{label}": fit.bse[name],
                f"p_{label}": fit.pvalues[name],
            })
    return pd.DataFrame(rows).set_index("period")

per_full = period_coefs(full_fit, "full")
per_restr = period_coefs(restricted_fit, "restricted")
per_sens = period_coefs(sensitivity_fit, "sensitivity")

# Align on common periods (intersection)
common = per_full.index.intersection(per_restr.index).intersection(per_sens.index)
compare = pd.concat([per_full.loc[common], per_restr.loc[common], per_sens.loc[common]], axis=1)
compare = compare.sort_index()

compare["delta_full_minus_restricted"] = compare["coef_full"] - compare["coef_restricted"]
compare["pct_period_effect_attrib_data_completeness"] = np.where(
    compare["coef_full"].abs() > 1e-6,
    (compare["delta_full_minus_restricted"] / compare["coef_full"]) * 100,
    np.nan
)

compare.to_csv(OUT / "08_apc_period_comparison.csv")

print(f"\nPeriod coefficients side-by-side ({len(compare)} common periods):")
print(compare[["coef_full", "coef_restricted", "coef_sensitivity",
               "delta_full_minus_restricted",
               "pct_period_effect_attrib_data_completeness"]].round(4).to_string())

# 7. Summary statistics
print("\nSummary statistics")
n_periods = len(compare)
mean_delta = compare["delta_full_minus_restricted"].mean()
median_delta = compare["delta_full_minus_restricted"].median()
n_collapse = (compare["coef_restricted"].abs() < compare["coef_full"].abs() * 0.5).sum()
n_persist = (compare["coef_restricted"].abs() >= compare["coef_full"].abs() * 0.5).sum()
correlation = compare[["coef_full", "coef_restricted"]].corr().iloc[0, 1]
corr_sens = compare[["coef_full", "coef_sensitivity"]].corr().iloc[0, 1]
print(f"  Common periods compared: {n_periods}")
print(f"  Mean coefficient delta (full - restricted): {mean_delta:+.4f}")
print(f"  Median coefficient delta: {median_delta:+.4f}")
print(f"  Period coefficients that collapse (|restricted| < 50% of |full|): {n_collapse}/{n_periods}")
print(f"  Period coefficients that persist (|restricted| ≥ 50% of |full|): {n_persist}/{n_periods}")
print(f"  Correlation between full and restricted period coefficients: {correlation:+.4f}")
print(f"  Correlation between full and sensitivity period coefficients: {corr_sens:+.4f}") 
