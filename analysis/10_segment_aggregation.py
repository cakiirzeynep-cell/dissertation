from pathlib import Path
import pandas as pd
import numpy as np

OUT = Path(__file__).resolve().parent / "output"

# Segment aggregation for SQ1
clv = pd.read_csv(OUT / "09_clv_per_customer.csv")
print(f"\nPer-customer CLV loaded: {len(clv):,} customers")
print(f"Base case CLV (T=60, d=10%) summary:")
print(f"  Mean: £{clv['clv_T60_d10'].mean():,.0f}")
print(f"  Median: £{clv['clv_T60_d10'].median():,.0f}")
print(f"  Total: £{clv['clv_T60_d10'].sum():,.0f}")

BASE = "clv_T60_d10"
TOTAL_VALUE = clv[BASE].sum()

def bootstrap_mean_ci(values, n_boot=2000, alpha=0.05, seed=42):
    """Bootstrap 95% CI on the mean."""
    rng = np.random.default_rng(seed)
    n = len(values)
    if n < 5:
        return np.nan, np.nan
    boots = np.empty(n_boot)
    arr = np.asarray(values)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[b] = arr[idx].mean()
    lo = np.percentile(boots, 100 * alpha / 2)
    hi = np.percentile(boots, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


def segment_summary(df, group_col):
    rows = []
    for grp, sub in df.groupby(group_col):
        n = len(sub)
        vals = sub[BASE].values
        ci_lo, ci_hi = bootstrap_mean_ci(vals)
        rows.append({
            group_col: grp,
            "n_customers": n,
            "mean_CLV_GBP": vals.mean(),
            "median_CLV_GBP": np.median(vals),
            "p10_CLV_GBP": np.percentile(vals, 10),
            "p90_CLV_GBP": np.percentile(vals, 90),
            "mean_CI_lo_95": ci_lo,
            "mean_CI_hi_95": ci_hi,
            "total_segment_value_GBP": vals.sum(),
            "share_of_total_value_pct": 100.0 * vals.sum() / TOTAL_VALUE,
        })
    return pd.DataFrame(rows).sort_values("total_segment_value_GBP", ascending=False)

# 1. By customer_type
print("\nCLV by customer_type")
ct_summary = segment_summary(clv, "customer_type_grouped")
ct_summary.to_csv(OUT / "10_segment_clv_customer_type.csv", index=False)
print(ct_summary.round(0).to_string(index=False))

# 2. By billing_period_months (using billing_grouped from Task 9)
print("\nCLV by billing structure")
bill_summary = segment_summary(clv, "billing_grouped")
bill_summary.to_csv(OUT / "10_segment_clv_billing.csv", index=False)
print(bill_summary.round(0).to_string(index=False))

# 3. By is_cross_platform
print("\nCLV by is_cross_platform")
cp_summary = segment_summary(clv, "is_cross_platform")
cp_summary.to_csv(OUT / "10_segment_clv_cross_platform.csv", index=False)
print(cp_summary.round(0).to_string(index=False))

# 4. Cross-tabs: customer_type × is_cross_platform
print("\nCLV cross-tabs: customer_type × is_cross_platform")
cross_rows = []
for ct in sorted(clv["customer_type_grouped"].unique()):
    for cp in [0, 1]:
        sub = clv.loc[(clv["customer_type_grouped"] == ct) &
                       (clv["is_cross_platform"] == cp)]
        if len(sub) == 0:
            continue
        vals = sub[BASE].values
        ci_lo, ci_hi = bootstrap_mean_ci(vals)
        cross_rows.append({
            "customer_type": ct,
            "is_cross_platform": cp,
            "n_customers": len(sub),
            "mean_CLV_GBP": vals.mean(),
            "median_CLV_GBP": np.median(vals),
            "mean_CI_lo_95": ci_lo,
            "mean_CI_hi_95": ci_hi,
            "total_segment_value_GBP": vals.sum(),
        })
cross_df = pd.DataFrame(cross_rows)
cross_df.to_csv(OUT / "10_segment_clv_cross_tabs.csv", index=False)
print(cross_df.round(0).to_string(index=False))
