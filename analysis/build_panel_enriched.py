import pandas as pd
import numpy as np
from pathlib import Path

# paths
BASE   = Path(__file__).resolve().parent.parent / "data"
PANEL  = BASE / "subscription_event_anon_cltv_tidy.csv"
LL     = BASE / "customer_linelevel_anon.csv"
EVENTS = BASE / "subscription_events_anon.csv"
OUT    = BASE / "artlogic_panel_enriched.csv"

KEPT_CHANGED_REASON = "Customer kept but changed subscription"

# 1. Load files 
panel  = pd.read_csv(PANEL)
ll     = pd.read_csv(LL)
events = pd.read_csv(EVENTS)

print(f"Panel:      {panel.shape[0]:,} rows, {panel['customer_id'].nunique():,} customers")
print(f"Line-level: {ll.shape[0]:,} rows, {ll['customer_id'].nunique():,} customers")
print(f"Events:     {events.shape[0]:,} rows, {events['customer_id'].nunique():,} customers")

# 2. Parse dates
ll["subscription_activated_date"] = pd.to_datetime(
    ll["subscription_activated_date"], errors="coerce", utc=True
).dt.tz_localize(None)  
events["occurred_at_ts"] = pd.to_datetime(
    events["occurred_at_timestamp"], errors="coerce", utc=True
).dt.tz_localize(None)  
panel["period_month_dt"] = pd.to_datetime(panel["period_month"])

# 3. Build activation_date per customer 
# Tier 1: earliest activation date from plan items only
tier1 = (ll[ll["item_type"] == "plan"]
         .groupby("customer_id")["subscription_activated_date"]
         .min()
         .rename("activation_date_t1"))

# Tier 2: earliest activation date from all line-level rows (plan + addon)
tier2 = (ll.groupby("customer_id")["subscription_activated_date"]
         .min()
         .rename("activation_date_t2"))

# Tier 3: earliest subscription_created event timestamp
created_ev = events[events["event_type"].isin(
    ["subscription_created", "subscription_created_with_backdating"]
)]
tier3 = (created_ev.groupby("customer_id")["occurred_at_ts"]
         .min()
         .rename("activation_date_t3"))

# Tier 4: earliest event of any type
tier4 = (events.groupby("customer_id")["occurred_at_ts"]
         .min()
         .rename("activation_date_t4"))

# Merge tiers onto the set of panel customer IDs
panel_custs = pd.DataFrame({"customer_id": panel["customer_id"].unique()})
act = (panel_custs
       .merge(tier1, on="customer_id", how="left")
       .merge(tier2, on="customer_id", how="left")
       .merge(tier3, on="customer_id", how="left")
       .merge(tier4, on="customer_id", how="left"))

# Apply cascade: use earliest non-null tier
act["activation_date"] = act["activation_date_t1"]
act["activation_source"] = "plan_linelevel"

mask2 = act["activation_date"].isna() & act["activation_date_t2"].notna()
act.loc[mask2, "activation_date"]   = act.loc[mask2, "activation_date_t2"]
act.loc[mask2, "activation_source"] = "all_linelevel"

mask3 = act["activation_date"].isna() & act["activation_date_t3"].notna()
act.loc[mask3, "activation_date"]   = act.loc[mask3, "activation_date_t3"]
act.loc[mask3, "activation_source"] = "created_event"

mask4 = act["activation_date"].isna() & act["activation_date_t4"].notna()
act.loc[mask4, "activation_date"]   = act.loc[mask4, "activation_date_t4"]
act.loc[mask4, "activation_source"] = "any_event"

# Tier 5 correction: for customers whose line-level activation_date is later than their earliest known event, take the event date instead.  
mask5 = act["activation_date_t4"].notna() & (act["activation_date_t4"] < act["activation_date"])
n_corrected = mask5.sum()
act.loc[mask5, "activation_date"]   = act.loc[mask5, "activation_date_t4"]
act.loc[mask5, "activation_source"] = "event_earlier_than_linelevel"
print(f"\nTier-5 correction applied to: {n_corrected} customers (line-level date was later than earliest event)")

src_counts = act["activation_source"].value_counts(dropna=False)
print("\nActivation date source breakdown:")
for src, n in src_counts.items():
    print(f"    {str(src):<20} {n:,} customers")
still_null = act["activation_date"].isna().sum()
print(f"Still null after all fallbacks: {still_null}")

act = act[["customer_id", "activation_date", "activation_source"]]

# 4. Build static customer attributes from line-level
# customer_type: unique per customer 
cust_type = (ll.groupby("customer_id")["customer_type"]
             .first()
             .reset_index()
             .rename(columns={"customer_type": "customer_type"}))

# parent_customer_id: take first non-null value per customer
parent = (ll[ll["parent_customer_id"].notna()]
          .groupby("customer_id")["parent_customer_id"]
          .first()
          .reset_index())

# billing_period_months: mode of plan-item billing periods per customer
#   (quarterly=3, annual=12, monthly=1)
billing = (ll[ll["item_type"] == "plan"]
           .groupby("customer_id")["billing_period_months"]
           .agg(lambda x: x.mode().iloc[0] if len(x) > 0 else np.nan)
           .reset_index()
           .rename(columns={"billing_period_months": "billing_period_months"}))

# country: most common non-null country per customer
country = (ll[ll["billing_address_country"].notna()]
           .groupby("customer_id")["billing_address_country"]
           .agg(lambda x: x.mode().iloc[0])
           .reset_index()
           .rename(columns={"billing_address_country": "country"}))

# kept_changed_sub_flag: True if any subscription has this cancel reason
kept_ids = set(
    ll[ll["cancel_reason_code"] == KEPT_CHANGED_REASON]["customer_id"].unique()
)

print(f"Customers flagged as 'kept but changed subscription': {len(kept_ids)}")

# 5. Merge all attributes together
enriched = panel.copy()

for df in [act, cust_type, parent, billing, country]:
    enriched = enriched.merge(df, on="customer_id", how="left")

# kept_changed_sub_flag: True if the customer had that cancel reason
enriched["kept_changed_sub_flag"] = enriched["customer_id"].isin(kept_ids)

# 6. Recompute tenure_months 
# tenure = number of complete months from activation_date to period_month
# Using relativedelta would be ideal but we use a fast approximation:
# floor((period_month_dt - activation_date).days / 30.44)
# Minimum is 0 (customer active in their first month)
enriched["tenure_months_recomp"] = (
    (enriched["period_month_dt"] - enriched["activation_date"])
    .dt.days
    .div(30.4375)     # average days per month
    .clip(lower=0)
    .round(1)
)

print(f"  Old tenure_months  — mean: {enriched['tenure_months'].mean():.1f}, "
      f"median: {enriched['tenure_months'].median():.1f}, "
      f"max: {enriched['tenure_months'].max():.1f}")
print(f"  New tenure_months  — mean: {enriched['tenure_months_recomp'].mean():.1f}, "
      f"median: {enriched['tenure_months_recomp'].median():.1f}, "
      f"max: {enriched['tenure_months_recomp'].max():.1f}")
null_tenure = enriched["tenure_months_recomp"].isna().sum()
print(f"  Null recomputed tenure (no activation date): {null_tenure} rows "
      f"({enriched[enriched['tenure_months_recomp'].isna()]['customer_id'].nunique()} customers)")

# 7. Sanity checks 
print("\nSanity checks")
assert len(enriched) == len(panel), \
    f"Row count changed: {len(panel)} → {len(enriched)}"
assert enriched["customer_id"].nunique() == panel["customer_id"].nunique(), \
    "Customer count changed after join"

print(f"Row count preserved:      {len(enriched):,} rows")
print(f"Customer count preserved: {enriched['customer_id'].nunique():,} customers")
print(f"customer_type null:       {enriched['customer_type'].isna().sum():,} rows "
      f"({enriched[enriched['customer_type'].isna()]['customer_id'].nunique()} customers)")
print(f"activation_date null:     {enriched['activation_date'].isna().sum():,} rows")
print(f"kept_changed_sub_flag=T:  "
      f"{enriched[enriched['kept_changed_sub_flag']]['customer_id'].nunique():,} unique customers")

# 8. Summary table 
print("\nCustomer type distribution (unique customers):")
ct = (enriched.drop_duplicates("customer_id")["customer_type"]
      .value_counts(dropna=False))
for ctype, n in ct.items():
    print(f"  {str(ctype):<20} {n:,}")

print("\nBilling period distribution (unique customers, plan subscriptions):")
bp_map = {1: "Monthly", 3: "Quarterly", 12: "Annual", 24: "Biennial", 6: "Semi-annual"}
bp = (enriched.drop_duplicates("customer_id")["billing_period_months"]
      .value_counts(dropna=False))
for period, n in bp.items():
    label = bp_map.get(period, str(period))
    print(f"  {label:<15} ({period} months)  {n:,}")

# 9. Save output 
# Drop the helper period_month_dt column before saving
enriched = enriched.drop(columns=["period_month_dt"])
enriched.to_csv(OUT, index=False)
