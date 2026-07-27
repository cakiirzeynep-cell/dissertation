import warnings
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.families.links import Logit

warnings.filterwarnings("ignore")

DATA = Path("/Users/zeynepcakir/Desktop/msc dissertation/data files ")
OUT  = Path("/Users/zeynepcakir/Desktop/msc dissertation/analysis/output")

print("Task 11 — SQ2 decomposition (CLV_Markov vs CLV_flat)")

# 1. Reuse Task 9 setup
panel = pd.read_csv(DATA / "artlogic_panel_enriched_v2.csv", low_memory=False)
panel["period_month"] = pd.to_datetime(panel["period_month"], format="%Y-%m")

last_month = panel["period_month"].max()
panel = panel.loc[
    (~panel["kept_changed_sub_flag"]) &
    (panel["period_month"] < last_month)
].copy()

need_cols = ["customer_type", "billing_period_months", "country",
             "tenure_months_recomp", "mrr_end_of_month"]
panel = panel.dropna(subset=need_cols)

def bin_tenure(t):
    if t < 3: return "0-2"
    if t < 6: return "3-5"
    if t < 12: return "6-11"
    if t < 24: return "12-23"
    if t < 36: return "24-35"
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
panel["log_mrr"] = np.log1p(panel["mrr_end_of_month"])

TRAIN_END = pd.Timestamp("2025-06-30")
train = panel.loc[panel["period_month"] <= TRAIN_END].copy()

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
        for c in fit_columns:
            if c not in X.columns:
                X[c] = 0.0
        X = X[fit_columns]
    return X

X_train = build_design(train)
y_train = train["churned_next_month"].astype(float).values
hazard_fit = GLM(y_train, X_train, family=Binomial(link=Logit())).fit(maxiter=300)
hazard_columns = X_train.columns.tolist()

period_cols_in_X = [c for c in hazard_columns if c.startswith("mo_")]
period_cols_dated = sorted(period_cols_in_X, key=lambda c: c.replace("mo_", ""))
last12_period_avg = float(np.mean([hazard_fit.params[c] for c in period_cols_dated[-12:]]))

panel_sorted = panel.sort_values(["customer_id", "period_month"]).reset_index(drop=True)
panel_sorted["mrr_prev"] = panel_sorted.groupby("customer_id")["mrr_end_of_month"].shift(1)
THRESH = 0.05
def classify(row):
    if pd.isna(row["mrr_prev"]) or row["mrr_prev"] == 0:
        return None
    change = (row["mrr_end_of_month"] - row["mrr_prev"]) / row["mrr_prev"]
    if change > THRESH: return "expanded"
    if change < -THRESH: return "contracted"
    return "stable"
panel_sorted["state_current"] = panel_sorted.apply(classify, axis=1)
panel_sorted["state_next"] = panel_sorted.groupby("customer_id")["state_current"].shift(-1)
panel_sorted["mrr_ratio_next"] = panel_sorted.groupby("customer_id")["mrr_end_of_month"].shift(-1) / panel_sorted["mrr_end_of_month"]
panel_sorted["survived_next"] = (1 - panel_sorted["churned_next_month"]).astype(bool)

valid = panel_sorted.dropna(subset=["state_current", "state_next", "mrr_ratio_next"]).copy()
valid = valid.loc[valid["survived_next"] & (valid["mrr_end_of_month"] > 0)]

states = ["contracted", "stable", "expanded"]
segments = sorted(valid["customer_type_grouped"].unique())

P_seg = {}
for seg in segments:
    seg_df = valid.loc[valid["customer_type_grouped"] == seg]
    counts = pd.crosstab(seg_df["state_current"], seg_df["state_next"]).reindex(
        index=states, columns=states, fill_value=0)
    rs = counts.sum(axis=1)
    P_seg[seg] = counts.div(rs, axis=0).fillna(0).values

mrr_mult_rows = []
for seg in segments:
    seg_df = valid.loc[valid["customer_type_grouped"] == seg]
    mults_median = {}
    for s in states:
        sub = seg_df.loc[seg_df["state_current"] == s, "mrr_ratio_next"]
        if len(sub) > 10:
            lo, hi = sub.quantile([0.05, 0.95])
            mults_median[s] = float(sub.clip(lo, hi).median())
        else:
            mults_median[s] = float(sub.median()) if len(sub) > 0 else 1.0
    mrr_mult_rows.append({"segment": seg,
                           **{f"r_{s}_median": mults_median[s] for s in states}})
mrr_mult_df = pd.DataFrame(mrr_mult_rows).set_index("segment")

last_obs = panel_sorted.sort_values("period_month").drop_duplicates("customer_id", keep="last").copy()
last_obs = last_obs.dropna(subset=["state_current"])
print(f"Customers entering projection: {len(last_obs):,}")

# 2. Hazard prediction helpers 
def bin_tenure_int(t):
    if t < 3: return "0-2"
    if t < 6: return "3-5"
    if t < 12: return "6-11"
    if t < 24: return "12-23"
    if t < 36: return "24-35"
    return "36+"

def compute_fixed_lp(df):
    lp = np.zeros(len(df))
    lp += hazard_fit.params.get("const", 0.0)
    for ct in ["Collector", "Artist", "Art Dealer", "Art advisory", "Other"]:
        mask = (df["customer_type_grouped"] == ct).values
        lp = lp + mask.astype(float) * hazard_fit.params.get(f"ct_{ct}", 0.0)
    for bill in ["monthly", "annual", "other"]:
        mask = (df["billing_grouped"] == bill).values
        lp = lp + mask.astype(float) * hazard_fit.params.get(f"bill_{bill}", 0.0)
    for ctry in ["GB", "FR", "CH", "DE", "Other"]:
        mask = (df["country_grouped"] == ctry).values
        lp = lp + mask.astype(float) * hazard_fit.params.get(f"ctry_{ctry}", 0.0)
    lp = lp + df["is_cross_platform"].astype(float).values * hazard_fit.params.get("is_cross_platform", 0.0)
    lp = lp + df["log_mrr"].values * hazard_fit.params.get("log_mrr", 0.0)
    lp = lp + last12_period_avg
    return lp

def tenure_lp_component_full(tenure, df):
    N = len(tenure)
    contrib = np.zeros(N)
    bins = np.array([bin_tenure_int(t) for t in tenure])
    is_cp = df["is_cross_platform"].astype(float).values
    for bin_name in ["0-2", "3-5", "6-11", "24-35", "36+"]:
        mask = (bins == bin_name).astype(float)
        contrib = contrib + mask * hazard_fit.params.get(f"ten_{bin_name}", 0.0)
        contrib = contrib + mask * is_cp * hazard_fit.params.get(f"is_cp_x_ten_{bin_name}", 0.0)
    return contrib

# 3. Two projection variants — Markov vs flat MRR
T = 60
d_annual = 0.10

def project_clv(customers_df, mrr_dynamics="markov"):
    N = len(customers_df)
    customers_df = customers_df.reset_index(drop=True).copy()

    pi = np.zeros((N, 3))
    for i, s in enumerate(customers_df["state_current"]):
        pi[i, states.index(s)] = 1.0

    mrr = customers_df["mrr_end_of_month"].values.astype(float).copy()
    mrr_initial = mrr.copy()
    tenure = customers_df["tenure_months_recomp"].values.astype(float).copy()
    surv = np.ones(N)
    clv = np.zeros(N)

    d_monthly = (1 + d_annual) ** (1.0 / 12.0) - 1

    seg_arr = customers_df["customer_type_grouped"].values
    P_by_cust = np.stack([P_seg[s] for s in seg_arr])
    r_by_cust = np.stack([
        mrr_mult_df.loc[s, ["r_contracted_median", "r_stable_median", "r_expanded_median"]].values
        for s in seg_arr
    ]).astype(float)

    fixed_lp = compute_fixed_lp(customers_df)

    for t in range(1, T + 1):
        if mrr_dynamics == "markov":
            pi = np.einsum("ni,nij->nj", pi, P_by_cust)
            mult = np.einsum("ni,ni->n", pi, r_by_cust)
            mrr = mrr * mult
        else:
            # Flat MRR: hold at initial value
            mrr = mrr_initial.copy()

        tenure = tenure + 1
        lp = fixed_lp + tenure_lp_component_full(tenure, customers_df)
        p_churn = 1.0 / (1.0 + np.exp(-lp))
        surv = surv * (1.0 - p_churn)
        discount = (1.0 + d_monthly) ** (-t)
        clv = clv + mrr * surv * discount

    return clv

print("\nProjecting CLV (Markov MRR)")
clv_markov = project_clv(last_obs, "markov")
print("Projecting CLV (flat MRR)")
clv_flat = project_clv(last_obs, "flat")

# 4. Per-customer decomposition + segment aggregation
result = last_obs[["customer_id", "customer_type_grouped", "billing_grouped",
                    "country_grouped", "is_cross_platform",
                    "tenure_months_recomp", "mrr_end_of_month",
                    "state_current"]].copy().reset_index(drop=True)
result["clv_markov"] = clv_markov
result["clv_flat"] = clv_flat
result["clv_revenue_contribution"] = clv_markov - clv_flat
result["clv_revenue_contribution_pct"] = np.where(
    clv_flat > 1.0,
    100.0 * (clv_markov - clv_flat) / clv_flat,
    np.nan
)
result.to_csv(OUT / "11_sq2_decomposition_per_customer.csv", index=False)

print(f"\nAggregate decomposition (T=60, d=10%, n={len(result):,})")
print(f"  Mean CLV (Markov):                 £{clv_markov.mean():>10,.0f}")
print(f"  Mean CLV (flat MRR):               £{clv_flat.mean():>10,.0f}")
print(f"  Mean revenue-trajectory contrib:   £{(clv_markov - clv_flat).mean():>10,.0f}")
print(f"  Mean contribution as % of flat:    {result['clv_revenue_contribution_pct'].median():>10,.2f}% (median)")
total_markov = clv_markov.sum()
total_flat = clv_flat.sum()
print(f"\n  Total CLV (Markov):                £{total_markov:>15,.0f}")
print(f"  Total CLV (flat):                  £{total_flat:>15,.0f}")
print(f"  Total revenue-trajectory contrib:  £{total_markov - total_flat:>15,.0f}")
print(f"   As % of flat-MRR baseline:       {100*(total_markov - total_flat)/total_flat:>10,.1f}%")

print("\nBy customer_type")
seg_rows = []
for grp_col in ["customer_type_grouped", "billing_grouped", "is_cross_platform"]:
    for grp, sub in result.groupby(grp_col):
        seg_rows.append({
            "dimension": grp_col,
            "segment": grp,
            "n": len(sub),
            "mean_clv_markov": sub["clv_markov"].mean(),
            "mean_clv_flat": sub["clv_flat"].mean(),
            "mean_contribution": (sub["clv_markov"] - sub["clv_flat"]).mean(),
            "median_contribution_pct": sub["clv_revenue_contribution_pct"].median(),
        })
seg_df = pd.DataFrame(seg_rows)
seg_df.to_csv(OUT / "11_sq2_decomposition_by_segment.csv", index=False)
print(seg_df.round(0).to_string(index=False))
