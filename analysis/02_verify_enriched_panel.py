from pathlib import Path
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
panel = pd.read_csv(DATA / "artlogic_panel_enriched_v2.csv", low_memory=False)

# Verify enriched panel readiness
# Shape and basic integrity
print(f"\nShape: {panel.shape[0]:,} rows × {panel.shape[1]} cols")
print(f"Unique customers: {panel['customer_id'].nunique():,}")
panel["period_month"] = pd.to_datetime(panel["period_month"], format="%Y-%m")
print(f"Date range: {panel['period_month'].min():%Y-%m} to {panel['period_month'].max():%Y-%m}")
print(f"# months: {panel['period_month'].nunique()}")

# Kept-changed exclusion 
kc_customers = panel.loc[panel["kept_changed_sub_flag"], "customer_id"].nunique()
kc_rows = panel["kept_changed_sub_flag"].sum()
print(f"Customers with kept_changed_sub_flag=True: {kc_customers}")
print(f"Customer-month rows flagged: {kc_rows:,}")

# Implied: filter out kept-changed before model fit
panel_kept_changed_excluded = panel.loc[~panel["kept_changed_sub_flag"]].copy()
print(f"Rows after excluding kept-changed: {len(panel_kept_changed_excluded):,}")

# Tenure recomputed sanity
print("\ntenure_months_recomp distribution")
ten = panel["tenure_months_recomp"]
print(f"min:  {ten.min():.1f}")
print(f"mean: {ten.mean():.1f}")
print(f"med:  {ten.median():.1f}")
print(f"p90:  {ten.quantile(0.9):.1f}")
print(f"max:  {ten.max():.1f}")
# Should be positive, sensible months-since-activation
neg = (ten < 0).sum()
print(f"Negative values (should be 0): {neg}")

# Country fill 
n_missing_country = panel["country"].isna().sum()
n_total_rows = len(panel)
print(f"Missing country rows: {n_missing_country} ({n_missing_country/n_total_rows:.1%})")
# How many unique customers have missing country
custs_missing = panel.loc[panel["country"].isna(), "customer_id"].nunique()
print(f"Unique customers with any missing country row: {custs_missing}")

# billing_period_months distribution
print(panel.drop_duplicates("customer_id")["billing_period_months"].value_counts(dropna=False).to_string())

# customer_type 
print("\ncustomer_type distribution (customer-level)")
ct = panel.drop_duplicates("customer_id")["customer_type"].value_counts(dropna=False)
print(ct.to_string())

# parent_customer_id
print("\nparent_customer_id")
has_parent = panel["parent_customer_id"].notna()
print(f"Rows with non-null parent: {has_parent.sum():,} ({has_parent.mean():.1%})")
n_custs_with_parent = panel.loc[has_parent, "customer_id"].nunique()
n_unique_parents = panel.loc[has_parent, "parent_customer_id"].nunique()
print(f"Unique customers with a parent: {n_custs_with_parent}")
print(f"Unique parent IDs: {n_unique_parents}")

# is_cross_platform 
ic = panel.drop_duplicates("customer_id")["is_cross_platform"].value_counts()
print(ic.to_string())

# Churn outcome verification 
# Full panel rate (incl. final month + kept-changed)
print(f"Raw panel rate (incl. last month + kept-changed): {panel['churned_next_month'].mean()*100:.3f}%")

# Excluding final month
last_month = panel["period_month"].max()
panel_no_last = panel.loc[panel["period_month"] < last_month]
print(f"Excluding final month: {panel_no_last['churned_next_month'].mean()*100:.3f}%")

# Excluding kept-changed
panel_no_kc = panel.loc[~panel["kept_changed_sub_flag"]]
print(f"Excluding kept-changed: {panel_no_kc['churned_next_month'].mean()*100:.3f}%")

# Both exclusions
panel_clean = panel.loc[(~panel["kept_changed_sub_flag"]) & (panel["period_month"] < last_month)]
print(f"Both exclusions: {panel_clean['churned_next_month'].mean()*100:.3f}%")
print(f"Rows in clean modelling panel: {len(panel_clean):,}")
print(f"Customers in clean modelling panel: {panel_clean['customer_id'].nunique():,}")
print(f"Churn events in clean modelling panel: {int(panel_clean['churned_next_month'].sum()):,}")

# Readiness Verdict
checks = [
    ("kept_changed_sub_flag column present", "kept_changed_sub_flag" in panel.columns),
    ("tenure_months_recomp column present", "tenure_months_recomp" in panel.columns),
    ("Customer type populated >99%", panel["customer_type"].notna().mean() > 0.99),
    ("Country populated >99%", panel["country"].notna().mean() > 0.99),
    ("Billing period populated >99%", panel["billing_period_months"].notna().mean() > 0.99),
    ("is_cross_platform present", "is_cross_platform" in panel.columns),
    ("Churn rate ~1.87% after clean exclusions",
        abs(panel_clean["churned_next_month"].mean() - 0.0187) < 0.005),
]
for name, ok in checks:
    print(f"  {'+' if ok else '-'}  {name}")
