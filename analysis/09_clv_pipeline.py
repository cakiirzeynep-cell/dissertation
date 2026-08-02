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

# CLV pipeline
# 1. Load panel + apply exclusions + feature engineering (same as Task 5)
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

# 2. Re-fit Task 5 hazard model (logit with interactions)
TRAIN_END = pd.Timestamp("2025-06-30")
train = panel.loc[panel["period_month"] <= TRAIN_END].copy()
print(f"\nTraining rows: {len(train):,} · events: {int(train['churned_next_month'].sum())}")

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
print(f"  Converged: {hazard_fit.converged} · AIC: {hazard_fit.aic:.2f}")
hazard_columns = X_train.columns.tolist()

# Calendar-month period coefficients: average of last 12 observed periods
period_cols_in_X = [c for c in hazard_columns if c.startswith("mo_")]
# Get the last 12 by date order
period_cols_dated = sorted(period_cols_in_X, key=lambda c: c.replace("mo_", ""))
last12_period_cols = period_cols_dated[-12:]
last12_period_avg = float(np.mean([hazard_fit.params[c] for c in last12_period_cols]))
print(f"  Average period coefficient over last 12 observed months: {last12_period_avg:+.4f}")

# 3. Markov transition matrices + MRR multipliers per segment
panel_sorted = panel.sort_values(["customer_id", "period_month"]).reset_index(drop=True)
panel_sorted["mrr_prev"] = panel_sorted.groupby("customer_id")["mrr_end_of_month"].shift(1)

THRESH = 0.05
def classify(row):
    if pd.isna(row["mrr_prev"]) or row["mrr_prev"] == 0:
        return None
    change = (row["mrr_end_of_month"] - row["mrr_prev"]) / row["mrr_prev"]
    if change > THRESH:
        return "expanded"
    if change < -THRESH:
        return "contracted"
    return "stable"

panel_sorted["state_current"] = panel_sorted.apply(classify, axis=1)
panel_sorted["state_next"] = panel_sorted.groupby("customer_id")["state_current"].shift(-1)
panel_sorted["mrr_ratio_next"] = panel_sorted.groupby("customer_id")["mrr_end_of_month"].shift(-1) / panel_sorted["mrr_end_of_month"]
panel_sorted["survived_next"] = (1 - panel_sorted["churned_next_month"]).astype(bool)

valid = panel_sorted.dropna(subset=["state_current", "state_next", "mrr_ratio_next"]).copy()
valid = valid.loc[valid["survived_next"] & (valid["mrr_end_of_month"] > 0)]

states = ["contracted", "stable", "expanded"]
segments = sorted(valid["customer_type_grouped"].unique())

# Per-segment transition matrices
P_seg = {}
for seg in segments:
    seg_df = valid.loc[valid["customer_type_grouped"] == seg]
    counts = pd.crosstab(seg_df["state_current"], seg_df["state_next"]).reindex(
        index=states, columns=states, fill_value=0)
    rs = counts.sum(axis=1)
    P_seg[seg] = counts.div(rs, axis=0).fillna(0).values

# Per-segment MRR multiplier per state.
# Use median rather than mean — small-MRR customers with large percentagemoves bias the mean upward dramatically. 
mrr_mult_rows = []
for seg in segments:
    seg_df = valid.loc[valid["customer_type_grouped"] == seg]
    mults_median = {}
    mults_mean = {}
    for s in states:
        sub = seg_df.loc[seg_df["state_current"] == s, "mrr_ratio_next"]
        # Winsorise the ratios at the 5th and 95th percentiles within each state to bound extreme tail influence. 
        if len(sub) > 10:
            lo, hi = sub.quantile([0.05, 0.95])
            sub_w = sub.clip(lo, hi)
            mults_median[s] = float(sub_w.median())
            mults_mean[s] = float(sub_w.mean())
        else:
            mults_median[s] = float(sub.median()) if len(sub) > 0 else 1.0
            mults_mean[s] = float(sub.mean()) if len(sub) > 0 else 1.0
    mrr_mult_rows.append({"segment": seg,
                           **{f"r_{s}_median": mults_median[s] for s in states},
                           **{f"r_{s}_mean": mults_mean[s] for s in states}})
mrr_mult_df = pd.DataFrame(mrr_mult_rows).set_index("segment")
mrr_mult_df.to_csv(OUT / "09_mrr_multipliers_per_segment.csv")
print(f"\nMRR multipliers per segment (median of winsorised ratios — primary):")
print(mrr_mult_df[[f"r_{s}_median" for s in states]].round(4).to_string())
print(f"\nMean (for reference — tail-biased):")
print(mrr_mult_df[[f"r_{s}_mean" for s in states]].round(4).to_string())

# 4. Initial state per customer = their last observed panel row
last_obs = panel_sorted.sort_values("period_month").drop_duplicates("customer_id", keep="last").copy()
# Need state_current at the customer's last observation
last_obs = last_obs.dropna(subset=["state_current"])
# Initial MRR = MRR at last observed month
print(f"\nCustomers entering CLV projection: {len(last_obs):,}")

# 5. CLV projection
def compute_fixed_lp(df, hazard_fit, hazard_columns, last12_period_avg):
    """Linear predictor for the part that does not change with future tenure:
    intercept, customer-type/billing/country dummies, the is_cross_platform
    main effect, log_mrr, and the averaged period effect. Tenure-bin effects
    and the is_cross_platform x tenure interaction change as tenure advances,
    so they are computed separately in tenure_lp_component."""
    # Build a one-row design template per customer that doesn't include tenure dummies or interactions.
    lp = np.zeros(len(df))

    # Intercept
    lp += hazard_fit.params.get("const", 0.0)

    # Customer type dummies 
    for ct in ["Collector", "Artist", "Art Dealer", "Art advisory", "Other"]:
        mask = (df["customer_type_grouped"] == ct).values
        lp = lp + mask.astype(float) * hazard_fit.params.get(f"ct_{ct}", 0.0)

    # Billing dummies
    for bill in ["monthly", "annual", "other"]:
        mask = (df["billing_grouped"] == bill).values
        lp = lp + mask.astype(float) * hazard_fit.params.get(f"bill_{bill}", 0.0)

    # Country dummies
    for ctry in ["GB", "FR", "CH", "DE", "Other"]:
        mask = (df["country_grouped"] == ctry).values
        lp = lp + mask.astype(float) * hazard_fit.params.get(f"ctry_{ctry}", 0.0)

    # is_cross_platform main effect
    lp = lp + df["is_cross_platform"].astype(float).values * hazard_fit.params.get("is_cross_platform", 0.0)

    # log_mrr (continuous)
    lp = lp + df["log_mrr"].values * hazard_fit.params.get("log_mrr", 0.0)

    # Period: use the last-12-month average for all future months
    lp = lp + last12_period_avg

    return lp


def bin_tenure_int(t):
    if t < 3: return "0-2"
    if t < 6: return "3-5"
    if t < 12: return "6-11"
    if t < 24: return "12-23"
    if t < 36: return "24-35"
    return "36+"


def tenure_lp_component(tenure, df, hazard_fit):
    """Tenure-dependent LP contribution including is_cp × tenure interactions."""
    N = len(tenure)
    contrib = np.zeros(N)
    bins = np.array([bin_tenure_int(t) for t in tenure])
    is_cp = df["is_cross_platform"].astype(float).values

    for bin_name in ["0-2", "3-5", "6-11", "24-35", "36+"]:
        mask = (bins == bin_name).astype(float)
        contrib = contrib + mask * hazard_fit.params.get(f"ten_{bin_name}", 0.0)
        # Interaction: is_cp × ten_{bin_name}
        contrib = contrib + mask * is_cp * hazard_fit.params.get(
            f"is_cp_x_ten_{bin_name}", 0.0
        )
    return contrib


def project_clv(customers_df, T, d_annual):
    N = len(customers_df)
    customers_df = customers_df.reset_index(drop=True).copy()

    pi = np.zeros((N, 3))
    for i, s in enumerate(customers_df["state_current"]):
        pi[i, states.index(s)] = 1.0

    mrr = customers_df["mrr_end_of_month"].values.astype(float).copy()
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

    fixed_lp = compute_fixed_lp(customers_df, hazard_fit, hazard_columns,
                                 last12_period_avg)

    for t in range(1, T + 1):
        pi = np.einsum("ni,nij->nj", pi, P_by_cust)
        mult = np.einsum("ni,ni->n", pi, r_by_cust)
        mrr = mrr * mult
        tenure = tenure + 1
        lp = fixed_lp + tenure_lp_component(tenure, customers_df, hazard_fit)
        p_churn = 1.0 / (1.0 + np.exp(-lp))
        surv = surv * (1.0 - p_churn)
        discount = (1.0 + d_monthly) ** (-t)
        clv = clv + mrr * surv * discount

    return clv

# 7. Run all T × d combinations
T_grid = [24, 36, 48, 60]
d_grid = [0.08, 0.10, 0.12]

result_df = last_obs[["customer_id", "customer_type_grouped", "billing_grouped",
                      "country_grouped", "is_cross_platform",
                      "tenure_months_recomp", "mrr_end_of_month",
                      "state_current"]].copy()

for T in T_grid:
    for d in d_grid:
        pct = int(d * 100)
        result_df[f"clv_T{T}_d{pct}"] = project_clv(last_obs, T, d)

result_df.to_csv(OUT / "09_clv_per_customer.csv", index=False)
print(f"\nPer-customer CLV saved: shape {result_df.shape}")

# 8. Sensitivity grid summary
grid_rows = []
for T in T_grid:
    for d in d_grid:
        pct = int(d * 100)
        col = f"clv_T{T}_d{pct}"
        grid_rows.append({
            "T_months": T, "d_annual": d,
            "mean_CLV_GBP": result_df[col].mean(),
            "median_CLV_GBP": result_df[col].median(),
            "p10_CLV_GBP": result_df[col].quantile(0.10),
            "p90_CLV_GBP": result_df[col].quantile(0.90),
        })
grid_df = pd.DataFrame(grid_rows)
grid_df.to_csv(OUT / "09_clv_sensitivity_grid.csv", index=False)
print("\nCLV sensitivity grid (T × d)")
print(grid_df.round(0).to_string(index=False))

# 9. Findings summary
base = result_df["clv_T60_d10"]
print(f"\nBASE CASE (T=60, d=10%)")
print(f"  Customers projected: {len(base):,}")
print(f"  Mean CLV: £{base.mean():,.0f}")
print(f"  Median CLV: £{base.median():,.0f}")
print(f"  Total customer-base value: £{base.sum():,.0f}")
