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

# SQ4 retention sensitivity
# 1. Panel + feature engineering (identical to Tasks 9/11)
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
print(f"  Converged: {hazard_fit.converged} · AIC: {hazard_fit.aic:.2f}")

period_cols_in_X = [c for c in hazard_columns if c.startswith("mo_")]
period_cols_dated = sorted(period_cols_in_X, key=lambda c: c.replace("mo_", ""))
last12_period_avg = float(np.mean([hazard_fit.params[c] for c in period_cols_dated[-12:]]))

# Markov + multipliers 
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
last_obs = last_obs.dropna(subset=["state_current"]).reset_index(drop=True)
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

# 3. CLV projection with per-customer hazard reduction
def project_clv(customers_df, T=60, d_annual=0.10, hazard_reduction=None):
    """Project CLV with optional per-customer hazard reduction.

    hazard_reduction: None, scalar, or length-N array. The per-period
    churn probability is multiplied by (1 - hazard_reduction). A value
    of 0.10 means the customer's monthly churn probability is reduced
    by 10%. A None or 0 leaves hazard unchanged.
    """
    N = len(customers_df)
    customers_df = customers_df.reset_index(drop=True).copy()

    if hazard_reduction is None:
        haz_red = np.zeros(N)
    elif np.isscalar(hazard_reduction):
        haz_red = np.full(N, float(hazard_reduction))
    else:
        haz_red = np.asarray(hazard_reduction, dtype=float)
        assert len(haz_red) == N

    pi = np.zeros((N, 3))
    for i, s in enumerate(customers_df["state_current"]):
        pi[i, states.index(s)] = 1.0

    mrr = customers_df["mrr_end_of_month"].values.astype(float).copy()
    tenure = customers_df["tenure_months_recomp"].values.astype(float).copy()
    surv = np.ones(N)
    clv = np.zeros(N)

    d_monthly = (1 + d_annual) ** (1.0 / 12.0) - 1
    keep_factor = 1.0 - haz_red  # multiplicative reduction on p_churn

    seg_arr = customers_df["customer_type_grouped"].values
    P_by_cust = np.stack([P_seg[s] for s in seg_arr])
    r_by_cust = np.stack([
        mrr_mult_df.loc[s, ["r_contracted_median", "r_stable_median", "r_expanded_median"]].values
        for s in seg_arr
    ]).astype(float)

    fixed_lp = compute_fixed_lp(customers_df)

    for t in range(1, T + 1):
        pi = np.einsum("ni,nij->nj", pi, P_by_cust)
        mult = np.einsum("ni,ni->n", pi, r_by_cust)
        mrr = mrr * mult
        tenure = tenure + 1
        lp = fixed_lp + tenure_lp_component_full(tenure, customers_df)
        p_churn = 1.0 / (1.0 + np.exp(-lp))
        p_churn = p_churn * keep_factor  # apply reduction
        surv = surv * (1.0 - p_churn)
        discount = (1.0 + d_monthly) ** (-t)
        clv = clv + mrr * surv * discount

    return clv

# 4. Baseline at T=60, d=10%, no reduction
clv_baseline = project_clv(last_obs, T=60, d_annual=0.10, hazard_reduction=None)
total_baseline = clv_baseline.sum()
mean_baseline = clv_baseline.mean()
print(f"  Mean baseline CLV: £{mean_baseline:,.0f}")
print(f"  Total customer-base value: £{total_baseline:,.0f}")

# 5. Slice 1: Population-wide retention scenarios
reduction_levels = [0.05, 0.10, 0.20]
pop_rows = [{
    "scenario": "baseline",
    "hazard_reduction_pct": 0,
    "n_customers_targeted": len(last_obs),
    "mean_CLV_GBP": mean_baseline,
    "total_CLV_GBP": total_baseline,
    "delta_total_GBP": 0.0,
    "delta_total_pct": 0.0,
    "delta_per_targeted_customer_GBP": 0.0,
}]
clv_pop_scenarios = {0.0: clv_baseline}
for x in reduction_levels:
    clv_x = project_clv(last_obs, T=60, d_annual=0.10, hazard_reduction=x)
    clv_pop_scenarios[x] = clv_x
    delta_total = clv_x.sum() - total_baseline
    pop_rows.append({
        "scenario": f"population_{int(x*100)}pct",
        "hazard_reduction_pct": int(x * 100),
        "n_customers_targeted": len(last_obs),
        "mean_CLV_GBP": clv_x.mean(),
        "total_CLV_GBP": clv_x.sum(),
        "delta_total_GBP": delta_total,
        "delta_total_pct": 100 * delta_total / total_baseline,
        "delta_per_targeted_customer_GBP": delta_total / len(last_obs),
    })
    print(f"  {int(x*100):>2}% reduction:  total ΔCLV £{delta_total:>14,.0f}  "
          f"({100*delta_total/total_baseline:+5.2f}%)  "
          f"per-customer £{delta_total/len(last_obs):>8,.0f}")
pop_df = pd.DataFrame(pop_rows)
pop_df.to_csv(OUT / "12_sq4_retention_population.csv", index=False)

# 6. Slice 2: Per-segment retention scenarios
seg_rows = []
seg_arr = last_obs["customer_type_grouped"].values
for seg in segments:
    seg_mask = (seg_arr == seg)
    n_seg = int(seg_mask.sum())
    baseline_seg_total = clv_baseline[seg_mask].sum()
    seg_rows.append({
        "segment": seg,
        "hazard_reduction_pct": 0,
        "n_customers_targeted": n_seg,
        "segment_baseline_total_GBP": baseline_seg_total,
        "segment_new_total_GBP": baseline_seg_total,
        "delta_total_GBP": 0.0,
        "delta_per_targeted_customer_GBP": 0.0,
        "delta_pct_of_segment_baseline": 0.0,
        "delta_pct_of_customerbase_total": 0.0,
    })
    for x in reduction_levels:
        haz_vec = np.where(seg_mask, x, 0.0)
        clv_seg_x = project_clv(last_obs, T=60, d_annual=0.10, hazard_reduction=haz_vec)
        new_seg_total = clv_seg_x[seg_mask].sum()
        delta = new_seg_total - baseline_seg_total
        seg_rows.append({
            "segment": seg,
            "hazard_reduction_pct": int(x * 100),
            "n_customers_targeted": n_seg,
            "segment_baseline_total_GBP": baseline_seg_total,
            "segment_new_total_GBP": new_seg_total,
            "delta_total_GBP": delta,
            "delta_per_targeted_customer_GBP": delta / n_seg if n_seg > 0 else 0,
            "delta_pct_of_segment_baseline": 100 * delta / baseline_seg_total if baseline_seg_total > 0 else 0,
            "delta_pct_of_customerbase_total": 100 * delta / total_baseline,
        })
        print(f"  {seg:>14}  n={n_seg:>4}  {int(x*100):>2}% reduction:  "
              f"ΔTotal £{delta:>12,.0f}  per-customer £{delta/n_seg:>8,.0f}")
seg_df_out = pd.DataFrame(seg_rows)
seg_df_out.to_csv(OUT / "12_sq4_retention_per_segment.csv", index=False)

# 7. Slice 3: Top-decile-by-MRR retention scenarios
mrr_initial = last_obs["mrr_end_of_month"].values
top_decile_thresh = np.quantile(mrr_initial, 0.90)
top_decile_mask = (mrr_initial >= top_decile_thresh)
n_top = int(top_decile_mask.sum())
top_baseline_total = clv_baseline[top_decile_mask].sum()
print(f"  Top-decile threshold (MRR): £{top_decile_thresh:,.0f}")
print(f"  Customers targeted: {n_top} ({100*n_top/len(last_obs):.1f}% of base)")
print(f"  Top-decile baseline total CLV: £{top_baseline_total:,.0f} "
      f"({100*top_baseline_total/total_baseline:.1f}% of customer-base total)")

top_rows = [{
    "scenario": "baseline",
    "hazard_reduction_pct": 0,
    "n_customers_targeted": n_top,
    "share_of_base_pct": 100 * n_top / len(last_obs),
    "targeted_baseline_total_GBP": top_baseline_total,
    "targeted_new_total_GBP": top_baseline_total,
    "delta_total_GBP": 0.0,
    "delta_per_targeted_customer_GBP": 0.0,
    "delta_pct_of_customerbase_total": 0.0,
    "leverage_ratio_vs_population": 1.0,
}]
for x in reduction_levels:
    haz_vec = np.where(top_decile_mask, x, 0.0)
    clv_top_x = project_clv(last_obs, T=60, d_annual=0.10, hazard_reduction=haz_vec)
    new_top_total = clv_top_x[top_decile_mask].sum()
    delta = new_top_total - top_baseline_total
    # Leverage ratio: ΔCLV per targeted customer (top decile) ÷ ΔCLV per targeted customer (population)
    pop_delta_per_cust = (clv_pop_scenarios[x].sum() - total_baseline) / len(last_obs)
    leverage = (delta / n_top) / pop_delta_per_cust if pop_delta_per_cust > 0 else np.nan
    top_rows.append({
        "scenario": f"top_decile_{int(x*100)}pct",
        "hazard_reduction_pct": int(x * 100),
        "n_customers_targeted": n_top,
        "share_of_base_pct": 100 * n_top / len(last_obs),
        "targeted_baseline_total_GBP": top_baseline_total,
        "targeted_new_total_GBP": new_top_total,
        "delta_total_GBP": delta,
        "delta_per_targeted_customer_GBP": delta / n_top,
        "delta_pct_of_customerbase_total": 100 * delta / total_baseline,
        "leverage_ratio_vs_population": leverage,
    })
    print(f"  {int(x*100):>2}% reduction:  ΔTotal £{delta:>12,.0f}  "
          f"per-targeted-customer £{delta/n_top:>8,.0f}  "
          f"leverage vs pop = {leverage:.2f}×")
top_df = pd.DataFrame(top_rows)
top_df.to_csv(OUT / "12_sq4_retention_top_decile.csv", index=False)

# 8. Highest-leverage cells: segment × tenure_bin at 10% population reduction
print("\nHighest-leverage cells (segment × tenure_bin) at 10% population reduction")
clv_pop_10 = clv_pop_scenarios[0.10]
delta_per_cust = clv_pop_10 - clv_baseline
cell_df = last_obs[["customer_type_grouped", "tenure_bin", "is_cross_platform"]].copy()
cell_df["delta_clv"] = delta_per_cust
cell_df["clv_baseline"] = clv_baseline

cells = cell_df.groupby(["customer_type_grouped", "tenure_bin"]).agg(
    n_customers=("delta_clv", "size"),
    mean_delta_GBP=("delta_clv", "mean"),
    total_delta_GBP=("delta_clv", "sum"),
    mean_baseline_CLV=("clv_baseline", "mean"),
).reset_index()
cells["delta_as_pct_of_baseline"] = 100 * cells["mean_delta_GBP"] / cells["mean_baseline_CLV"]
cells = cells.sort_values("mean_delta_GBP", ascending=False).reset_index(drop=True)
cells.to_csv(OUT / "12_sq4_highest_leverage_cells.csv", index=False)
print(cells.head(10).round(2).to_string(index=False))

# Also: segment × is_cross_platform for the cross-platform mechanism finding
print("\nHighest-leverage cells (segment × is_cross_platform)")
cells_cp = cell_df.groupby(["customer_type_grouped", "is_cross_platform"]).agg(
    n_customers=("delta_clv", "size"),
    mean_delta_GBP=("delta_clv", "mean"),
    total_delta_GBP=("delta_clv", "sum"),
    mean_baseline_CLV=("clv_baseline", "mean"),
).reset_index()
cells_cp["delta_as_pct_of_baseline"] = 100 * cells_cp["mean_delta_GBP"] / cells_cp["mean_baseline_CLV"]
cells_cp = cells_cp.sort_values("mean_delta_GBP", ascending=False).reset_index(drop=True)
print(cells_cp.head(12).round(2).to_string(index=False))

# Cross-platform × tenure_bin 
cells_cp_tenure = cell_df.groupby(["is_cross_platform", "tenure_bin"]).agg(
    n_customers=("delta_clv", "size"),
    mean_delta_GBP=("delta_clv", "mean"),
    total_delta_GBP=("delta_clv", "sum"),
    mean_baseline_CLV=("clv_baseline", "mean"),
).reset_index()
cells_cp_tenure["delta_as_pct_of_baseline"] = 100 * cells_cp_tenure["mean_delta_GBP"] / cells_cp_tenure["mean_baseline_CLV"]
cells_cp_tenure = cells_cp_tenure.sort_values("mean_delta_GBP", ascending=False).reset_index(drop=True)
cells_cp_tenure.to_csv(OUT / "12_sq4_leverage_cp_x_tenure.csv", index=False)
print("\nHighest-leverage cells (is_cross_platform × tenure_bin)")
print(cells_cp_tenure.round(2).to_string(index=False))

# 9. Horizon sensitivity (T) — baseline and 10% population reduction
T_grid = [24, 36, 48, 60]
horizon_rows = []
for T in T_grid:
    clv_base_T = project_clv(last_obs, T=T, d_annual=0.10, hazard_reduction=None)
    clv_red_T  = project_clv(last_obs, T=T, d_annual=0.10, hazard_reduction=0.10)
    delta_T = clv_red_T.sum() - clv_base_T.sum()
    horizon_rows.append({
        "T_months": T,
        "d_annual": 0.10,
        "baseline_total_GBP": clv_base_T.sum(),
        "baseline_mean_GBP": clv_base_T.mean(),
        "pop_10pct_total_GBP": clv_red_T.sum(),
        "pop_10pct_delta_total_GBP": delta_T,
        "pop_10pct_delta_pct": 100 * delta_T / clv_base_T.sum(),
    })
    print(f"  T={T:>2}m  baseline £{clv_base_T.sum():>12,.0f}  "
          f"+10% pop ΔCLV £{delta_T:>11,.0f}  ({100*delta_T/clv_base_T.sum():+.2f}%)")
horizon_df = pd.DataFrame(horizon_rows)
horizon_df.to_csv(OUT / "12_sq4_sensitivity_horizon.csv", index=False)

# 10. Discount sensitivity (d) — baseline and 10% population reduction
d_grid = [0.08, 0.10, 0.12]
disc_rows = []
for d in d_grid:
    clv_base_d = project_clv(last_obs, T=60, d_annual=d, hazard_reduction=None)
    clv_red_d  = project_clv(last_obs, T=60, d_annual=d, hazard_reduction=0.10)
    delta_d = clv_red_d.sum() - clv_base_d.sum()
    disc_rows.append({
        "T_months": 60,
        "d_annual": d,
        "baseline_total_GBP": clv_base_d.sum(),
        "baseline_mean_GBP": clv_base_d.mean(),
        "pop_10pct_total_GBP": clv_red_d.sum(),
        "pop_10pct_delta_total_GBP": delta_d,
        "pop_10pct_delta_pct": 100 * delta_d / clv_base_d.sum(),
    })
    print(f"  d={int(d*100):>2}%  baseline £{clv_base_d.sum():>12,.0f}  "
          f"+10% pop ΔCLV £{delta_d:>11,.0f}  ({100*delta_d/clv_base_d.sum():+.2f}%)")
disc_df = pd.DataFrame(disc_rows)
disc_df.to_csv(OUT / "12_sq4_sensitivity_discount.csv", index=False)

# 11. Findings summary
pop10_delta = pop_df.loc[pop_df["hazard_reduction_pct"] == 10, "delta_total_GBP"].iloc[0]
pop10_pct = pop_df.loc[pop_df["hazard_reduction_pct"] == 10, "delta_total_pct"].iloc[0]
pop10_per_cust = pop_df.loc[pop_df["hazard_reduction_pct"] == 10, "delta_per_targeted_customer_GBP"].iloc[0]

pop5_delta = pop_df.loc[pop_df["hazard_reduction_pct"] == 5, "delta_total_GBP"].iloc[0]
pop20_delta = pop_df.loc[pop_df["hazard_reduction_pct"] == 20, "delta_total_GBP"].iloc[0]

top10_delta = top_df.loc[top_df["hazard_reduction_pct"] == 10, "delta_total_GBP"].iloc[0]
top10_per_cust = top_df.loc[top_df["hazard_reduction_pct"] == 10, "delta_per_targeted_customer_GBP"].iloc[0]
top10_leverage = top_df.loc[top_df["hazard_reduction_pct"] == 10, "leverage_ratio_vs_population"].iloc[0]
