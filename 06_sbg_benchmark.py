import warnings
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln
from sklearn.metrics import roc_auc_score, brier_score_loss

warnings.filterwarnings("ignore")

DATA = Path("/Users/zeynepcakir/Desktop/msc dissertation/data files ")
OUT  = Path("/Users/zeynepcakir/Desktop/msc dissertation/analysis/output")

print("Task 6 — sBG benchmark fit")


# 1. Load panel and apply 
panel = pd.read_csv(DATA / "artlogic_panel_enriched_v2.csv", low_memory=False)
panel["period_month"] = pd.to_datetime(panel["period_month"], format="%Y-%m")

last_month = panel["period_month"].max()
panel = panel.loc[
    (~panel["kept_changed_sub_flag"]) &
    (panel["period_month"] < last_month)
].copy()

panel = panel.dropna(subset=["customer_type", "tenure_months_recomp",
                              "churned_next_month"])

# Customer type grouping 
top5_types = ["Gallery", "Collector", "Artist", "Art Dealer", "Art advisory"]
panel["customer_type_grouped"] = panel["customer_type"].apply(
    lambda x: x if x in top5_types else "Other"
)

panel["tenure_int"] = panel["tenure_months_recomp"].astype(int)

# Train / test split
TRAIN_END = pd.Timestamp("2025-06-30")
train = panel.loc[panel["period_month"] <= TRAIN_END].copy()
test  = panel.loc[panel["period_month"] >  TRAIN_END].copy()
print(f"\nTrain: {len(train):,} rows · {int(train['churned_next_month'].sum())} events")
print(f"Test:  {len(test):,} rows · {int(test['churned_next_month'].sum())} events")

# 2. sBG hazard function + likelihood
def sbg_hazard(t, alpha, beta):
    return alpha / (alpha + beta + t)


def neg_log_likelihood(params, R, F, t_vals):
    alpha, beta = np.exp(params)  # enforce positivity
    h = sbg_hazard(t_vals, alpha, beta)
    # Clip to avoid log(0)
    h = np.clip(h, 1e-9, 1 - 1e-9)
    ll = F * np.log(h) + (R - F) * np.log(1 - h)
    return -np.sum(ll)


def fit_sbg(df):
    # Risk set + failures by integer tenure
    grouped = df.groupby("tenure_int").agg(
        R=("churned_next_month", "size"),
        F=("churned_next_month", "sum"),
    ).reset_index()
    t_vals = grouped["tenure_int"].values.astype(float)
    R = grouped["R"].values.astype(float)
    F = grouped["F"].values.astype(float)

    # Initial guess (log scale): alpha=1, beta=50 (low hazard, lots of survivors)
    x0 = np.log([1.0, 50.0])
    result = minimize(
        neg_log_likelihood, x0, args=(R, F, t_vals),
        method="Nelder-Mead",
        options={"xatol": 1e-7, "fatol": 1e-7, "maxiter": 2000},
    )
    alpha, beta = np.exp(result.x)
    return alpha, beta, -result.fun, len(df), int(F.sum())

# 3. Aggregate sBG fit
print("\nFitting aggregate sBG")
alpha_agg, beta_agg, llf_agg, n_train, n_events_train = fit_sbg(train)
print(f"  α = {alpha_agg:.4f}")
print(f"  β = {beta_agg:.4f}")
print(f"  Implied mean retention E[θ̄] = β/(α+β) = {beta_agg/(alpha_agg+beta_agg):.4f}")
print(f"  Implied initial hazard h(0) = {sbg_hazard(0, alpha_agg, beta_agg):.4f}")
print(f"  Implied h at t=12: {sbg_hazard(12, alpha_agg, beta_agg):.4f}")
print(f"  Implied h at t=36: {sbg_hazard(36, alpha_agg, beta_agg):.4f}")
print(f"  Log-likelihood: {llf_agg:.2f}")

agg_fit_df = pd.DataFrame([{
    "fit_label": "aggregate",
    "alpha": alpha_agg,
    "beta": beta_agg,
    "mean_retention_theta": beta_agg / (alpha_agg + beta_agg),
    "hazard_t0": sbg_hazard(0, alpha_agg, beta_agg),
    "hazard_t12": sbg_hazard(12, alpha_agg, beta_agg),
    "hazard_t36": sbg_hazard(36, alpha_agg, beta_agg),
    "log_likelihood": llf_agg,
    "n_train_rows": n_train,
    "n_train_events": n_events_train,
}])
agg_fit_df.to_csv(OUT / "06_sbg_aggregate_fit.csv", index=False)

# 4. Per-customer_type sBG fits
print("\nFitting per-customer_type sBG")
per_seg_fits = []
for seg in sorted(train["customer_type_grouped"].unique()):
    seg_train = train.loc[train["customer_type_grouped"] == seg]
    if len(seg_train) < 100 or seg_train["churned_next_month"].sum() < 5:
        print(f"  {seg}: skipped (insufficient events)")
        continue
    try:
        a, b, llf, n_train_seg, n_events_seg = fit_sbg(seg_train)
        per_seg_fits.append({
            "segment": seg,
            "alpha": a,
            "beta": b,
            "mean_retention_theta": b / (a + b),
            "hazard_t0": sbg_hazard(0, a, b),
            "hazard_t12": sbg_hazard(12, a, b),
            "hazard_t36": sbg_hazard(36, a, b),
            "log_likelihood": llf,
            "n_train_rows": n_train_seg,
            "n_train_events": n_events_seg,
        })
        print(f"  {seg:15s}: α={a:.3f} β={b:.3f} h(0)={sbg_hazard(0,a,b):.4f} "
              f"events={int(n_events_seg)}")
    except Exception as e:
        print(f"  {seg}: fit failed — {e}")

per_seg_df = pd.DataFrame(per_seg_fits)
per_seg_df.to_csv(OUT / "06_sbg_per_segment_fits.csv", index=False)

# 5. Generate test-set predictions
# For aggregate sBG: every test row gets predicted h_sBG(tenure)
test = test.copy()
test["p_sbg_agg"] = sbg_hazard(test["tenure_int"].values.astype(float),
                                 alpha_agg, beta_agg)

# For per-segment sBG: use each segment's α, β to predict for its own rows
def predict_per_seg(row):
    seg = row["customer_type_grouped"]
    fit = per_seg_df.loc[per_seg_df["segment"] == seg]
    if len(fit) == 0:
        return sbg_hazard(row["tenure_int"], alpha_agg, beta_agg)
    a, b = fit.iloc[0]["alpha"], fit.iloc[0]["beta"]
    return sbg_hazard(row["tenure_int"], a, b)

test["p_sbg_per_seg"] = test.apply(predict_per_seg, axis=1)

# 6. Comparison vs hazard model 
print("\nRe-running Task 5 logit on test for comparison")
import statsmodels.api as sm
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.families.links import Logit

# Quick re-fit using the same design as Task 5
def bin_tenure(t):
    if t < 3: return "0-2"
    if t < 6: return "3-5"
    if t < 12: return "6-11"
    if t < 24: return "12-23"
    if t < 36: return "24-35"
    return "36+"

panel["tenure_bin"] = panel["tenure_months_recomp"].apply(bin_tenure)
top_countries = panel["country"].value_counts().head(5).index.tolist()
panel["country_grouped"] = panel["country"].apply(
    lambda x: x if x in top_countries else "Other"
)
def bin_billing(b):
    if b == 1:   return "monthly"
    if b == 3:   return "quarterly"
    if b == 12:  return "annual"
    return "other"
panel["billing_grouped"] = panel["billing_period_months"].apply(bin_billing)
panel["period_str"] = panel["period_month"].dt.strftime("%Y-%m")
panel["log_mrr"] = np.log1p(panel["mrr_avg_month"].fillna(0))

train_h = panel.loc[panel["period_month"] <= TRAIN_END].dropna(subset=["billing_period_months","country"]).copy()
test_h  = panel.loc[panel["period_month"] >  TRAIN_END].dropna(subset=["billing_period_months","country"]).copy()

def build_design(df, fit_columns=None):
    X_cat = pd.get_dummies(
        df[["tenure_bin", "customer_type_grouped", "billing_grouped",
            "country_grouped", "period_str"]],
        prefix={"tenure_bin": "ten", "customer_type_grouped": "ct",
                "billing_grouped": "bill", "country_grouped": "ctry",
                "period_str": "mo"},
        drop_first=False,
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

X_train_h = build_design(train_h)
y_train_h = train_h["churned_next_month"].astype(float).values
X_test_h  = build_design(test_h, fit_columns=X_train_h.columns.tolist())
y_test_h  = test_h["churned_next_month"].astype(float).values

logit_fit = GLM(y_train_h, X_train_h, family=Binomial(link=Logit())).fit()
p_hazard_test = logit_fit.predict(X_test_h)
test_h["p_hazard"] = p_hazard_test.values

# Merge hazard predictions onto sBG test frame for direct comparison
# Use index-based alignment since both came from `panel`
hazard_map = dict(zip(test_h.index, p_hazard_test))
test["p_hazard"] = test.index.map(hazard_map)
test_eval = test.dropna(subset=["p_hazard"]).copy()
y_eval = test_eval["churned_next_month"].astype(float).values

print(f"\nTest evaluation rows (intersection): {len(test_eval):,}")

# 7. Aggregate held-out comparison
def metrics(y, p, label):
    auc = roc_auc_score(y, p)
    brier = brier_score_loss(y, p)
    return {"model": label, "n": len(y), "events": int(y.sum()),
            "AUC": auc, "Brier": brier}

agg_metrics = [
    metrics(y_eval, test_eval["p_sbg_agg"].values, "sBG aggregate"),
    metrics(y_eval, test_eval["p_sbg_per_seg"].values, "sBG per-segment"),
    metrics(y_eval, test_eval["p_hazard"].values, "Discrete-time hazard (logit)"),
]
agg_df = pd.DataFrame(agg_metrics)
print("\nAggregate held-out comparison")
print(agg_df.round(5).to_string(index=False))

# 8. Per-segment held-out comparison
print("\nPer-segment held-out comparison")
seg_rows = []
for seg in sorted(test_eval["customer_type_grouped"].unique()):
    g = test_eval.loc[test_eval["customer_type_grouped"] == seg]
    y = g["churned_next_month"].astype(float).values
    if len(g) < 100 or y.sum() < 5:
        continue
    seg_rows.append({
        "segment": seg,
        "n": len(g),
        "events": int(y.sum()),
        "AUC_sbg_agg": roc_auc_score(y, g["p_sbg_agg"]),
        "AUC_sbg_per_seg": roc_auc_score(y, g["p_sbg_per_seg"]),
        "AUC_hazard": roc_auc_score(y, g["p_hazard"]),
        "Brier_sbg_agg": brier_score_loss(y, g["p_sbg_agg"]),
        "Brier_sbg_per_seg": brier_score_loss(y, g["p_sbg_per_seg"]),
        "Brier_hazard": brier_score_loss(y, g["p_hazard"]),
    })
seg_comp = pd.DataFrame(seg_rows)
print(seg_comp.round(4).to_string(index=False))
seg_comp.to_csv(OUT / "06_sbg_vs_hazard_heldout.csv", index=False)

