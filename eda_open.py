from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_DIR = Path("/Users/zeynepcakir/Desktop/msc dissertation")
DATA_DIR    = PROJECT_DIR / "data"  
FIG_DIR     = PROJECT_DIR / "analysis" / "figures_open"
OUT_DIR     = PROJECT_DIR / "analysis" / "output_open"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# SNAPSHOT_DATE = date on which this analysis was run.
# Used for computing durations of still-active subscriptions.
SNAPSHOT_DATE = pd.Timestamp.today().normalize()

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150,
                     "savefig.bbox": "tight", "figure.figsize": (10, 5)})

def savefig(name):
    plt.savefig(FIG_DIR / f"{name}.png")
    plt.close()

def savetab(df, name):
    df.to_csv(OUT_DIR / f"{name}.csv")

# 1.  First contact — what do we actually have?
# Three files. What is one row in each of them?
# What time period does each cover? How do they relate to each other?

# Load 
ll = pd.read_csv(DATA_DIR / "customer_linelevel_anon.csv",
                 parse_dates=["subscription_activated_date",
                               "current_billing_period_start",
                               "current_billing_period_end",
                               "subscription_cancellation_date"],
                 low_memory=False)

# Coerce timezone-aware date columns to naive once at load 
for _col in ["subscription_activated_date", "subscription_cancellation_date",
             "current_billing_period_start", "current_billing_period_end"]:
    if hasattr(ll[_col].dt, "tz") and ll[_col].dt.tz is not None:
        ll[_col] = ll[_col].dt.tz_localize(None)

# Normalise a known capitalisation typo: 'ArtBase' to 'Artbase'
# 'Artbase' is the dominant spelling; 'ArtBase' is the rare variant 
ll["product_family_group"] = ll["product_family_group"].str.replace(
    r"\bArtBase\b", "Artbase", regex=True)

ev = pd.read_csv(DATA_DIR / "subscription_events_anon.csv",
                 parse_dates=["occurred_at_timestamp",
                               "current_term_start",
                               "current_term_end",
                               "next_billing_at"],
                 low_memory=False)
# Events file may carry timezone info
if hasattr(ev["occurred_at_timestamp"].dt, "tz") and ev["occurred_at_timestamp"].dt.tz is not None:
    ev["occurred_at_timestamp"] = ev["occurred_at_timestamp"].dt.tz_localize(None)

panel = pd.read_csv(DATA_DIR / "artlogic_panel_enriched.csv",
                    low_memory=False)
panel["period_month"] = pd.to_datetime(panel["period_month"], format="%Y-%m")
panel["activation_date"] = pd.to_datetime(panel["activation_date"])
# artlogic_panel_enriched.csv adds activation_date, tenure_months_recomp,
# customer_type, billing_period_months, country, kept_changed_sub_flag, etc.

# What is one row? 
print("\nLine-level file — column names:")
print(ll.columns.tolist())
print(f"\nShape: {ll.shape}  |  memory: {ll.memory_usage(deep=True).sum()/1e6:.1f} MB")

print("\nEvents file — column names:")
print(ev.columns.tolist())
print(f"Shape: {ev.shape}  |  memory: {ev.memory_usage(deep=True).sum()/1e6:.1f} MB")

print("\nPanel file — column names:")
print(panel.columns.tolist())
print(f"Shape: {panel.shape}  |  memory: {panel.memory_usage(deep=True).sum()/1e6:.1f} MB")

# Unique keys in each 
print("\nUnique identifiers")
print(f"Line-level : {ll['customer_id'].nunique():>6,} customers, "
      f"{ll['subscription_id'].nunique():>6,} subscriptions, "
      f"{ll['subscription_id'].count():>7,} rows "
      f"(avg {ll.shape[0]/ll['subscription_id'].nunique():.1f} line items per sub)")
print(f"Events     : {ev['customer_id'].nunique():>6,} customers, "
      f"{ev['subscription_id'].nunique():>6,} subscriptions, "
      f"{ev.shape[0]:>7,} rows")
print(f"Panel      : {panel['customer_id'].nunique():>6,} customers, "
      f"{panel['period_month'].nunique():>6,} months,   "
      f"{panel.shape[0]:>7,} rows (customer-month pairs)")


# Date ranges 
print("\nDate ranges")
for col in ["subscription_activated_date", "subscription_cancellation_date"]:
    s = ll[col]
    print(f"line-level.{col:<35}  "
          f"min={s.min()!s:<24}  max={s.max()!s:<24}  "
          f"missing={s.isna().mean():.1%}")
s = ev["occurred_at_timestamp"]
print(f"events.occurred_at_timestamp                           "
      f"min={str(s.min())[:19]}  max={str(s.max())[:19]}  "
      f"missing={s.isna().mean():.1%}")


# 2.  Who are the customers?
# What kinds of customers does Artlogic have? Where are they?
# How concentrated is the customer base?

# Customer type 
# Deduplicate to one row per customer (using most-recent subscription)
cust = (ll.sort_values("subscription_activated_date")
          .drop_duplicates("customer_id", keep="last"))

print("\nCustomer types")
ct = cust["customer_type"].value_counts(dropna=False)
ct_pct = cust["customer_type"].value_counts(normalize=True, dropna=False)
print(pd.concat([ct, ct_pct.map("{:.1%}".format)], axis=1,
                keys=["count", "share"]).to_string())
savetab(ct.to_frame("count"), "02_customer_types")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ct.head(10).plot(kind="barh", ax=axes[0], color="#4c72b0")
axes[0].invert_yaxis()
axes[0].set(title="Customer types (count)", xlabel="# unique customers")
ct_pct.head(10).mul(100).plot(kind="barh", ax=axes[1], color="#4c72b0")
axes[1].invert_yaxis()
axes[1].set(title="Customer types (share %)", xlabel="% of customers")
plt.tight_layout()
savefig("02a_customer_types")


# Geography 
print("\nGeography (top 15 countries by customer count)")
geo = (cust["billing_address_country"]
       .value_counts(dropna=False).head(15))
print(geo.to_string())
savetab(geo.to_frame("n_customers"), "02_geography")

plt.figure(figsize=(10, 5))
geo.plot(kind="bar", color="#55a868")
plt.title("Unique customers by country (top 15)")
plt.xlabel("country code")
plt.ylabel("# customers")
plt.xticks(rotation=45)
savefig("02b_geography")


# Parent-child hierarchy 
print("\nParent-child hierarchy")
n_with_parent = cust["parent_customer_id"].notna().sum()
n_unique_parents = cust["parent_customer_id"].nunique()
print(f"Customers with a parent_customer_id: {n_with_parent:,} "
      f"({n_with_parent/len(cust):.1%})")
print(f"Distinct parent IDs: {n_unique_parents:,}")

children_per_parent = (cust.dropna(subset=["parent_customer_id"])
                           .groupby("parent_customer_id")["customer_id"]
                           .nunique())
print(f"Children per parent: mean={children_per_parent.mean():.1f}, "
      f"max={children_per_parent.max()}")


# 3.  What are customers buying?
# Products, billing periods, currencies, and the plan/add-on split.

# Product families 
print("\nProduct families")
print(ll["product_family_group"].value_counts().to_string())

print("\nItem types (plan vs add-on)")
print(ll["item_type"].value_counts().to_string())

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ll["product_family_group"].value_counts().plot(
    kind="barh", ax=axes[0], color="#4c72b0")
axes[0].invert_yaxis()
axes[0].set(title="Line items by product family", xlabel="# line items")
ll["item_type"].value_counts().plot(
    kind="bar", ax=axes[1], color="#55a868")
axes[1].set(title="Line items by type (plan vs add-on)", xlabel="item type")
plt.tight_layout()
savefig("03a_products")


# Plan vs add-on revenue split 
print("\nMRR by item type (plan vs add-on, all subscriptions)")
mrr_by_item_type = (ll.groupby("item_type")["MRR_GBP"]
                      .agg(["sum", "mean", "count"])
                      .sort_values("sum", ascending=False))
mrr_by_item_type["share_of_mrr"] = (mrr_by_item_type["sum"] /
                                    mrr_by_item_type["sum"].sum())
print(mrr_by_item_type.round(2).to_string())
savetab(mrr_by_item_type, "03_mrr_by_item_type")


# Add-on adoption per subscription 
print("\nAdd-on adoption per subscription (active subs only)")
addon_counts = (ll.loc[ll["status"] == "active"]
                  .groupby("subscription_id")["item_type"]
                  .apply(lambda x: (x == "addon").sum())
                  .rename("n_addons"))
print(addon_counts.value_counts().sort_index().to_string())
print(f"\n  Share with ≥1 add-on: {(addon_counts >= 1).mean():.0%}")
print(f"  Mean add-ons per active sub: {addon_counts.mean():.2f}")
savetab(addon_counts.value_counts().sort_index().to_frame("n_subs"),
        "03_addon_adoption")

# Add-on rate by customer type
addon_by_type = (ll.loc[ll["status"] == "active"]
                   .assign(has_addon=ll["item_type"] == "addon")
                   .groupby(["subscription_id","customer_type"])["has_addon"]
                   .max()
                   .reset_index()
                   .groupby("customer_type")["has_addon"]
                   .agg(["mean","size"])
                   .sort_values("mean", ascending=False)
                   .rename(columns={"mean":"addon_rate","size":"n_subs"}))
print("\nAdd-on adoption rate by customer type (active subs)")
print(addon_by_type.round(3).to_string())
savetab(addon_by_type, "03_addon_by_type")


# Quantity field 
print("\nQuantity field (are there multi-seat line items?)")
print(ll["quantity"].describe().to_string())
qty_gt1 = ll.loc[ll["quantity"] > 1, ["item_type","product_family_group","MRR_GBP","quantity"]]
print(f"\nLine items with quantity > 1: {len(qty_gt1):,} "
      f"({len(qty_gt1)/len(ll):.1%} of all rows)")
if len(qty_gt1) > 0:
    print(qty_gt1["item_type"].value_counts().to_string())
    print(f"MRR on quantity>1 rows: £{qty_gt1['MRR_GBP'].sum():,.0f}")


# Billing period 
print("\nBilling periods")
bp = ll.drop_duplicates("subscription_id")["billing_period_months"]
print(bp.value_counts().sort_index().to_string())

plt.figure(figsize=(8, 4))
bp.value_counts().sort_index().plot(kind="bar", color="#4c72b0")
plt.title("Unique subscriptions by billing period (months)")
plt.xlabel("billing period (months)")
plt.ylabel("# subscriptions")
plt.xticks(rotation=0)
savefig("03b_billing_period")


# Currency mix 
print("\nCurrency mix")
curr = ll.drop_duplicates("subscription_id")["currency_code"].value_counts()
print(curr.to_string())


# 4.  How much do customers pay?
# Revenue distributions at subscription and customer level.
# Who pays the most? Is revenue concentrated or spread?
# What about zero-MRR rows?

# Sub-level MRR (summed across line items) 
sub_mrr = (ll.groupby("subscription_id")
             .agg(MRR_GBP=("MRR_GBP", "sum"),
                  status=("status", "first"),
                  customer_id=("customer_id", "first"),
                  customer_type=("customer_type", "first"),
                  billing_period_months=("billing_period_months", "first"))
             .reset_index())

active_mrr = sub_mrr.loc[sub_mrr["status"] == "active", "MRR_GBP"]
print("\nMRR per active subscription (GBP, summed across line items)")
print(active_mrr.describe(percentiles=[.1, .25, .5, .75, .9, .95, .99])
                .to_string())

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.histplot(active_mrr[active_mrr > 0], bins=60, ax=axes[0], color="#4c72b0")
axes[0].set(title="MRR per active subscription (linear scale)",
            xlabel="MRR (£)", ylabel="# subscriptions")
sns.histplot(np.log10(active_mrr[active_mrr > 0]), bins=60,
             ax=axes[1], color="#4c72b0")
axes[1].set(title="MRR per active subscription (log10 scale)",
            xlabel="log10(MRR £)", ylabel="# subscriptions")
plt.tight_layout()
savefig("04a_mrr_distribution")


# Zero-MRR active subscriptions 
# MRR_GBP structural missingness by status 
print("\nMRR_GBP missingness by status")
print(ll.groupby("status")["MRR_GBP"].apply(lambda s: s.isna().mean())
        .round(3).to_string())


# Zero-MRR active line items 
print("\nZero-MRR active subscriptions")
zero_mrr = ll.loc[(ll["status"] == "active") & (ll["MRR_GBP"] == 0)]
zero_subs = sub_mrr.loc[(sub_mrr["status"] == "active") & (sub_mrr["MRR_GBP"] == 0)]
print(f"Active line items with MRR_GBP = 0:      {len(zero_mrr):,} "
      f"({len(zero_mrr)/ll['status'].eq('active').sum():.1%} of active rows)")
print(f"Active subscriptions with total MRR = 0: {len(zero_subs):,} "
      f"({len(zero_subs)/sub_mrr['status'].eq('active').sum():.1%} of active subs)")
print("  (Line-item zeros exist but subscriptions are not net-zero because "
      "each sub sums multiple line items.)")
if len(zero_mrr) > 0:
    print("\nProduct families among zero-MRR active rows:")
    print(zero_mrr["product_family_group"].value_counts().head(10).to_string())
    print("\nCustomer types among zero-MRR active rows:")
    print(zero_mrr["customer_type"].value_counts().head(8).to_string())


# Concentration 
sorted_mrr = active_mrr.sort_values(ascending=False).reset_index(drop=True)
cumshare = sorted_mrr.cumsum() / sorted_mrr.sum()

plt.figure(figsize=(9, 4))
plt.plot(np.arange(1, len(cumshare)+1) / len(cumshare) * 100,
         cumshare.values * 100, color="#4c72b0")
plt.axhline(80, color="red", linestyle="--", alpha=0.5, label="80% of MRR")
plt.title("Lorenz-style curve: cumulative MRR share (active subs, ranked)")
plt.xlabel("% of subscriptions (highest MRR first)")
plt.ylabel("% of total MRR")
plt.legend()
savefig("04b_mrr_concentration")

subs_for_80 = int(np.searchsorted(cumshare.values, 0.80)) + 1
print(f"\nConcentration: top {subs_for_80:,} subscriptions "
      f"({subs_for_80/len(active_mrr):.0%} of active subs) "
      f"account for 80% of total active MRR.")


# MRR by customer type (subscription level) 
print("\nMRR by customer type (active subs only)")
mrr_by_type = (sub_mrr.loc[sub_mrr["status"] == "active"]
               .groupby("customer_type")["MRR_GBP"]
               .agg(["count", "median", "mean", "sum"])
               .sort_values("sum", ascending=False))
print(mrr_by_type.round(0).to_string())
savetab(mrr_by_type, "04_mrr_by_customer_type")

plt.figure(figsize=(10, 5))
order = mrr_by_type.index
sns.boxplot(data=sub_mrr.loc[sub_mrr["status"] == "active"],
            x="customer_type", y="MRR_GBP", order=order,
            showfliers=False, color="#4c72b0")
plt.xticks(rotation=30, ha="right")
plt.title("MRR per active subscription by customer type (outliers hidden)")
plt.ylabel("MRR (£)")
savefig("04c_mrr_by_type")

# Customer-level MRR (aggregate across all active subs per customer) 
print("\nMRR per customer (all active subscriptions summed)")
cust_mrr = (sub_mrr.loc[sub_mrr["status"] == "active"]
              .groupby("customer_id")["MRR_GBP"]
              .sum()
              .rename("total_mrr_per_customer"))
print(cust_mrr.describe(percentiles=[.1,.25,.5,.75,.9,.95,.99]).round(0).to_string())

multi_sub_custs = (sub_mrr.loc[sub_mrr["status"] == "active"]
                   .groupby("customer_id")["subscription_id"]
                   .count())
pct_multi = (multi_sub_custs > 1).mean()
print(f"\n  Customers with >1 active subscription: {pct_multi:.1%}")
print(f"  Max active subs for one customer: {multi_sub_custs.max()}")

# Lorenz at customer level
sorted_cust_mrr = cust_mrr.sort_values(ascending=False).reset_index(drop=True)
cust_cumshare = sorted_cust_mrr.cumsum() / sorted_cust_mrr.sum()
custs_for_80 = int(np.searchsorted(cust_cumshare.values, 0.80)) + 1

plt.figure(figsize=(9, 4))
plt.plot(np.arange(1, len(cust_cumshare)+1) / len(cust_cumshare) * 100,
         cust_cumshare.values * 100, color="#4c72b0")
plt.axhline(80, color="red", linestyle="--", alpha=0.5, label="80% of MRR")
plt.title("Lorenz curve: cumulative MRR share at the CUSTOMER level")
plt.xlabel("% of customers (highest MRR first)")
plt.ylabel("% of total MRR")
plt.legend()
savefig("04d_mrr_concentration_customer")
savetab(cust_mrr.describe().to_frame(), "04_mrr_per_customer")


# MRR by billing period 
print("\nAvg MRR by billing period (active subs)")
mrr_bp = (sub_mrr.loc[sub_mrr["status"] == "active"]
          .groupby("billing_period_months")["MRR_GBP"]
          .agg(["count", "median", "mean"]).round(0))
print(mrr_bp.to_string())


# 5.  When did customers join and how long do they stay?
# Activation history, the pre/post-acquisition split, tenure distribution, and how the customer base has grown over time.

# Activation history 
act = (ll.dropna(subset=["subscription_activated_date"])
         .drop_duplicates("subscription_id")
         .assign(act_month=lambda d:
                 d["subscription_activated_date"]
                 .dt.to_period("M").dt.to_timestamp()))

act["act_year"] = act["act_month"].dt.year
monthly_new = act.groupby("act_month").size().rename("new_subs")

plt.figure(figsize=(12, 4))
monthly_new.plot(color="#4c72b0")
plt.title("New subscription activations per month (all time)")
plt.ylabel("# new subscriptions")
plt.xlabel("activation month")
savefig("05a_activation_history")
savetab(monthly_new.to_frame(), "05_monthly_activations")

print("\nActivations by year")
print(act.groupby("act_year").size().to_string())
print("\nTop 10 activation months")
print(monthly_new.sort_values(ascending=False).head(10).to_string())


# Pre / post-acquisition split 
ACQ_DATE = pd.Timestamp("2025-07-29")  

act["cohort"] = np.where(act["act_month"] < ACQ_DATE, "pre-acquisition", "post-acquisition")
ll_nodup = ll.drop_duplicates("subscription_id")
ll_nodup = ll_nodup.merge(act[["subscription_id","cohort"]], on="subscription_id", how="left")

print(f"\nPre / post-acquisition split (cut: {ACQ_DATE:%Y-%m-%d})")
cohort_counts = act["cohort"].value_counts()
print(cohort_counts.to_string())

# Geography mix by cohort — did US share shift after acquisition?
geo_cohort = (ll_nodup.dropna(subset=["cohort"])
               .groupby(["cohort","billing_address_country"])
               .size()
               .reset_index(name="n")
               .sort_values(["cohort","n"], ascending=[True, False]))
# Vectorised 
geo_pivot = (geo_cohort.sort_values(["cohort", "n"], ascending=[True, False])
               .groupby("cohort")[["cohort","billing_address_country","n"]]
               .head(5)
               .reset_index(drop=True))
print("\nTop 5 countries by cohort")
print(geo_pivot.to_string(index=False))

# Customer type mix by cohort
type_cohort = (ll_nodup.dropna(subset=["cohort"])
                .groupby(["cohort","customer_type"])
                .size()
                .unstack(fill_value=0))
type_cohort_pct = type_cohort.div(type_cohort.sum(axis=1), axis=0)
print("\nCustomer type mix by cohort (share)")
print(type_cohort_pct.round(3).to_string())

# MRR by cohort
mrr_cohort = (sub_mrr.merge(ll_nodup[["subscription_id","cohort"]],
                             on="subscription_id", how="left")
               .loc[lambda d: d["status"] == "active"]
               .groupby("cohort")["MRR_GBP"]
               .agg(["count","median","mean","sum"]).round(0))
print("\nMRR (active subs) by acquisition cohort")
print(mrr_cohort.to_string())
savetab(mrr_cohort, "05_mrr_by_cohort")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
monthly_new.plot(color="#4c72b0", ax=axes[0])
axes[0].axvline(ACQ_DATE, color="red", linestyle="--", alpha=0.7,
                label=f"ACQ_DATE ({ACQ_DATE:%Y-%m-%d})")
axes[0].set(title="Activations per month — with acquisition cut", ylabel="# new subs")
axes[0].legend()
type_cohort_pct.T.plot(kind="bar", ax=axes[1], color=["#4c72b0","#c44e52"])
axes[1].set(title="Customer type mix: pre vs post acquisition",
            xlabel="customer type", ylabel="share")
axes[1].tick_params(axis="x", rotation=30)
plt.tight_layout()
savefig("05d_pre_post_acquisition")


# Current age of active subscriptions 
sub_level = (ll.sort_values("subscription_activated_date")
               .drop_duplicates("subscription_id", keep="first")
               .drop(columns=["MRR_GBP","MRR_LCUR","quantity",
                               "item_price_name","item_type",
                               "product_family_group"]))

# sub_mrr already has MRR_GBP at subscription level
sub_level = sub_level.merge(sub_mrr[["subscription_id","MRR_GBP"]],
                            on="subscription_id", how="left")
sub_level["observed_months"] = np.where(
    sub_level["subscription_cancellation_date"].notna(),
    ((sub_level["subscription_cancellation_date"]
      - sub_level["subscription_activated_date"]) /
     np.timedelta64(1,"D")) / 30.44,
    ((SNAPSHOT_DATE - sub_level["subscription_activated_date"]) /
     np.timedelta64(1,"D")) / 30.44,
)
sub_level["observed_months"] = sub_level["observed_months"].clip(lower=0)
sub_level["ended"] = (sub_level["status"] == "cancelled").astype(int)

print("\nObserved duration by status")
print(sub_level.groupby("status")["observed_months"]
               .describe(percentiles=[.25, .5, .75, .9])
               .round(1).to_string())

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.histplot(data=sub_level, x="observed_months", hue="status",
             bins=40, multiple="stack", ax=axes[0])
axes[0].set(title="Observed duration by status",
            xlabel="months (active → today  or  active → cancelled)")
sns.ecdfplot(data=sub_level, x="observed_months", hue="status",
             ax=axes[1])
axes[1].set(title="Cumulative distribution of observed duration",
            xlabel="months")
plt.tight_layout()
savefig("05b_duration_distribution")


# When in a subscription's life do cancellations happen?
cancelled = sub_level.loc[sub_level["status"] == "cancelled"].copy()

plt.figure(figsize=(10, 4))
sns.histplot(cancelled["observed_months"].clip(upper=60), bins=40,
             color="#c44e52")
plt.title("Duration at cancellation (months, capped at 60)")
plt.xlabel("months from activation to cancellation")
plt.ylabel("# cancelled subscriptions")
savefig("05c_duration_at_cancellation")


# 6.  What happens to subscriptions over time?

print("\nOverall subscription status")
status_counts = sub_level["status"].value_counts()
status_pct = sub_level["status"].value_counts(normalize=True)
print(pd.concat([status_counts, status_pct.map("{:.1%}".format)],
                axis=1, keys=["count","share"]).to_string())

plt.figure(figsize=(7, 4))
status_counts.plot(kind="bar", color=["#55a868","#c44e52","#dd8452","#8172b2"])
plt.title("Subscription status (all time)")
plt.ylabel("# subscriptions")
plt.xticks(rotation=0)
savefig("06a_status_distribution")


print("\n'non_renewing' — what is it?")
nr = sub_level.loc[sub_level["status"] == "non_renewing"]
print(f"  {len(nr):,} subscriptions are non_renewing "
      f"(scheduled to cancel at end of current term)")
print(f"  Customer types among non_renewing:")
print(nr["customer_type"].value_counts().head(5).to_string())


# Pause / resume cycle 
print("\nPause / resume cycle")
paused_subs = sub_level.loc[sub_level["status"] == "paused"]
print(f"  Subscriptions currently paused: {len(paused_subs):,}")

if len(ev) > 0:
    pause_events   = ev.loc[ev["event_type"].str.contains("pause", case=False, na=False)]
    resume_events  = ev.loc[ev["event_type"].str.contains("resume|reactivat", case=False, na=False)]
    print(f"  Pause-related events in event stream:  {len(pause_events):,}")
    print(f"  Resume-related events in event stream: {len(resume_events):,}")
    if len(pause_events) > 0:
        print("\n  Pause event types seen:")
        print(pause_events["event_type"].value_counts().to_string())


# Cancellation rate over calendar time 
can = (ll.dropna(subset=["subscription_cancellation_date"])
         .drop_duplicates("subscription_id")
         .assign(can_month=lambda d:
                 d["subscription_cancellation_date"]
                 .dt.to_period("M").dt.to_timestamp()))
monthly_cancel = can.groupby("can_month").size().rename("cancellations")

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
monthly_new.plot(ax=axes[0], color="#55a868", label="new activations")
monthly_cancel.plot(ax=axes[0], color="#c44e52", label="cancellations")
axes[0].set(title="Monthly activations vs cancellations", ylabel="# subscriptions")
axes[0].legend()

by_year = pd.DataFrame({
    "activations": act.groupby("act_year").size(),
    "cancellations": can.assign(year=can["can_month"].dt.year).groupby("year").size(),
})
by_year["ratio"] = by_year["cancellations"] / by_year["activations"]
print("\nAnnual activations vs cancellations")
print(by_year.to_string())
by_year[["activations","cancellations"]].plot(kind="bar", ax=axes[1],
                                              color=["#55a868","#c44e52"])
axes[1].set(title="Annual activations vs cancellations", ylabel="# subscriptions")
axes[1].tick_params(axis="x", rotation=0)
plt.tight_layout()
savefig("06b_activations_vs_cancellations")


# Seasonality: do cancellations cluster in any month of the year? 
print("\nCancellations by month of year (seasonality check)")
can["cancel_month_of_year"] = can["can_month"].dt.month
monthly_seasonality = can.groupby("cancel_month_of_year").size().rename("n_cancellations")
print(monthly_seasonality.to_string())

plt.figure(figsize=(9, 4))
monthly_seasonality.plot(kind="bar", color="#c44e52")
plt.title("Cancellations by calendar month (all years combined)")
plt.xlabel("month of year (1=Jan, 12=Dec)")
plt.ylabel("# cancellations")
plt.xticks(rotation=0)
savefig("06c_cancellation_seasonality")
savetab(monthly_seasonality.to_frame(), "06_cancellation_seasonality")


# 7.  Why do subscriptions end?
# Cancel reasons

# Raw reason distribution 
reasons = (ll.loc[ll["status"] == "cancelled"]
             .drop_duplicates("subscription_id")["cancel_reason_code"]
             .value_counts(dropna=False))
print("\nCancel reason codes (cancelled subs, top 25)")
print(reasons.head(25).to_string())
savetab(reasons.to_frame("count"), "07_cancel_reasons")

fig, ax = plt.subplots(figsize=(9, 7))
reasons.head(15).sort_values().plot(kind="barh", ax=ax, color="#c44e52")
ax.set(title="Top 15 cancel reasons (cancelled subscriptions)",
       xlabel="# subscriptions")
plt.tight_layout()
savefig("07a_cancel_reasons")

print(f"\nCancelled subs with no reason recorded: "
      f"~{ll.loc[ll['status']=='cancelled','cancel_reason_code'].isna().mean():.0%}")


# Cancel reasons by customer type 
print("\nCancel reasons: present vs missing by customer type")
canc_ll = (ll.loc[ll["status"] == "cancelled"]
             .drop_duplicates("subscription_id")
             [["customer_type","cancel_reason_code","billing_period_months"]])
canc_ll["has_reason"] = canc_ll["cancel_reason_code"].notna()
print(canc_ll.groupby("customer_type")["has_reason"]
             .agg(["mean","size"])
             .sort_values("size", ascending=False)
             .round(2).to_string())


# Cancel reason by billing period 
print("\nReason coverage by billing period")
bp_reason = (canc_ll.groupby("billing_period_months")["has_reason"]
             .agg(["mean","size"]).sort_values("billing_period_months"))
print(bp_reason.to_string())


# 8.  What does the event stream tell us?

print("\nEvent types (all events)")
et = ev["event_type"].value_counts()
print(et.to_string())
savetab(et.to_frame("count"), "08_event_types")

fig, ax = plt.subplots(figsize=(10, 6))
et.sort_values().plot(kind="barh", ax=ax, color="#8172b2")
ax.set(title="Event types in the events file", xlabel="# events")
plt.tight_layout()
savefig("08a_event_types")


# Events per customer 
events_per_cust = ev.groupby("customer_id").size()
print("\nEvents per customer (in the events file)")
print(events_per_cust.describe(percentiles=[.5,.9,.99]).to_string())

plt.figure(figsize=(9, 4))
sns.histplot(events_per_cust.clip(upper=100), bins=40, color="#8172b2")
plt.title("Events per customer (capped at 100)")
plt.xlabel("# events")
savefig("08b_events_per_customer")


# Customer journeys: first and last event 
journey = (ev.dropna(subset=["occurred_at_timestamp"])
             .drop_duplicates(["event_id","subscription_id"])
             .sort_values(["subscription_id","occurred_at_timestamp"])
             .groupby("subscription_id")
             .agg(first_event=("event_type","first"),
                  last_event=("event_type","last"),
                  n_events=("event_type","size"),
                  span_days=("occurred_at_timestamp",
                              lambda s: (s.max()-s.min()).days)))

print("\nFirst event type (what starts a subscription journey?)")
print(journey["first_event"].value_counts(normalize=True).to_string())
print("\nLast event type (what ends a subscription journey?)")
print(journey["last_event"].value_counts(normalize=True).to_string())


# Look at one complex customer journey in detail 
print("\nSample detailed customer journey")
most_events = ev["customer_id"].value_counts().index[0]
journey_sample = (ev.loc[ev["customer_id"] == most_events]
                    .dropna(subset=["occurred_at_timestamp"])
                    .drop_duplicates(["event_id","subscription_id"])
                    .sort_values("occurred_at_timestamp")
                    [["occurred_at_timestamp","event_type","subscription_id",
                      "mrr","status"]])
print(f"\nCustomer with most events (ID suffix: ...{most_events[-8:]}):")
print(journey_sample.to_string(index=False))


# 9.  The monthly panel — customer trajectories over time

print(f"\nPanel covers: {panel['period_month'].min():%b %Y} "
      f"to {panel['period_month'].max():%b %Y} "
      f"({panel['period_month'].nunique()} months)")
print(f"Unique customers in panel: {panel['customer_id'].nunique():,}")

# How many customers are active each month?
monthly_active = (panel.groupby("period_month")
                  .agg(n_customers=("customer_id","nunique"),
                       n_active=("mrr_end_of_month", lambda s: (s>0).sum()),
                       total_mrr=("mrr_end_of_month","sum"))
                  .reset_index())
monthly_active["arpu"] = monthly_active["total_mrr"] / monthly_active["n_active"]

fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
axes[0].plot(monthly_active["period_month"], monthly_active["n_active"], "-o", ms=3)
axes[0].set(ylabel="# customers with MRR > 0", title="Panel: active customers per month")
axes[1].plot(monthly_active["period_month"], monthly_active["total_mrr"]/1000,
             "-o", ms=3, color="#55a868")
axes[1].set(ylabel="Total MRR (£000s)", title="Panel: total MRR per month")
axes[2].plot(monthly_active["period_month"], monthly_active["arpu"],
             "-o", ms=3, color="#c44e52")
axes[2].set(ylabel="ARPU (£)", title="Panel: average MRR per active customer")
plt.tight_layout()
savefig("09a_panel_monthly_overview")
savetab(monthly_active.set_index("period_month"), "09_monthly_overview")

print("\nMonthly panel overview")
print(monthly_active.to_string(index=False))


# MRR per customer: how stable is it month-to-month? 
panel_sorted = panel.sort_values(["customer_id","period_month"])
panel_sorted["mrr_prev"] = panel_sorted.groupby("customer_id")["mrr_end_of_month"].shift(1)
panel_sorted["mrr_delta"] = panel_sorted["mrr_end_of_month"] - panel_sorted["mrr_prev"]

stable = (panel_sorted["mrr_delta"].abs() < 0.01).mean()
expanding = (panel_sorted["mrr_delta"] > 0.01).mean()
contracting = (panel_sorted["mrr_delta"] < -0.01).mean()

print(f"\nMonth-over-month MRR stability")
print(f"  MRR unchanged (±£0.01):  {stable:.1%}")
print(f"  MRR increased:           {expanding:.1%}")
print(f"  MRR decreased (not zero):{contracting:.1%}")


# churned_next_month: the panel's churn signal 
print("\nchurned_next_month distribution")
print(panel["churned_next_month"].value_counts().to_string())

LAST_MONTH = panel["period_month"].max()
# panel_obs excludes the final month (churned_next_month is undefined there)
# and excludes "customer kept but changed subscription" — their cancellation
# is a plan downgrade/change, not true churn, so including them would inflate
# the churn rate and bias the hazard model.
n_kept = panel["kept_changed_sub_flag"].sum()
kept_custs = panel["kept_changed_sub_flag"].map(bool)
panel_obs = panel.loc[(panel["period_month"] < LAST_MONTH) & (~kept_custs)]
print(f"  Excluded {panel['customer_id'][kept_custs].nunique()} 'kept but changed' customers "
      f"({n_kept:,} rows) from churn analysis")
monthly_churn = panel_obs["churned_next_month"].mean()
print(f"\nShare of customer-months ending in churn (last month excluded): "
      f"{monthly_churn:.4f} ({monthly_churn*100:.2f}%)")

monthly_churn_ts = (panel_obs.groupby("period_month")["churned_next_month"]
                   .mean().rename("churn_rate"))

plt.figure(figsize=(11, 4))
monthly_churn_ts.plot(marker="o", ms=3, color="#c44e52")
plt.axhline(monthly_churn, linestyle="--", color="black",
            alpha=0.5, label=f"overall mean {monthly_churn:.2%}")
plt.title("Monthly churn rate over time (panel; last month excluded)")
plt.ylabel("share of customers leaving next month")
plt.legend()
savefig("09b_monthly_churn_ts")


# has_churn_event vs churned_next_month 
print("\nhas_churn_event vs churned_next_month")
xt = pd.crosstab(panel["has_churn_event"], panel["churned_next_month"],
                 margins=True, normalize="index").round(3)
print(xt.to_string())


# Recurrent churn events: customers with multiple churn-event months 
print("\nRecurrent churn signals (customers with >1 churn-event month)")
churn_event_months = (panel.loc[panel["has_churn_event"] == 1]
                       .groupby("customer_id")["period_month"]
                       .nunique()
                       .rename("n_churn_event_months"))
print(churn_event_months.value_counts().sort_index().to_string())
pct_recurrent = (churn_event_months > 1).mean()
print(f"\nCustomers with >1 churn-event month: "
      f"{(churn_event_months > 1).sum():,} of "
      f"{len(churn_event_months):,} ({pct_recurrent:.1%})")
savetab(churn_event_months.value_counts().sort_index().to_frame("n_customers"),
        "09_recurrent_churn_events")


# tenure_months in the panel 
print("\nWhat does tenure_months in the panel actually measure?")
first_panel_row = panel.groupby("customer_id")["period_month"].min()
panel_with_first = panel.merge(first_panel_row.rename("first_in_panel"),
                                on="customer_id")
panel_with_first["panel_obs_months"] = (
    (panel_with_first["period_month"].dt.year -
     panel_with_first["first_in_panel"].dt.year) * 12 +
    (panel_with_first["period_month"].dt.month -
     panel_with_first["first_in_panel"].dt.month)
)
diff = (panel_with_first["tenure_months"] -
        panel_with_first["panel_obs_months"]).abs()
print(f"  tenure_months == months since first panel row: "
      f"{(diff == 0).mean():.0%} of rows (confirmed)")
print(f"  Panel starts: {panel['period_month'].min():%b %Y}")


# 10.  Patterns that cut across everything
# Enrich panel with attributes 
# customer_type, country, billing_period_months are already in the enriched panel
panel_rich = panel_obs.copy()

# Churn rate by every segment 
# min_n=100 consistent with two-way pivot below
def churn_by(col, min_n=100):
    return (panel_rich.groupby(col)
                      .agg(n_obs=("churned_next_month","size"),
                           n_custs=("customer_id","nunique"),
                           churn_rate=("churned_next_month","mean"))
                      .query("n_obs >= @min_n")
                      .sort_values("churn_rate", ascending=False)
                      .assign(annualised=lambda d: 1-(1-d["churn_rate"])**12))

print("\nChurn rate by customer_type")
ct_churn = churn_by("customer_type")
print(ct_churn.round(4).to_string())
savetab(ct_churn, "10_churn_by_type")

print("\nChurn rate by billing_period_months")
bp_churn = churn_by("billing_period_months")
print(bp_churn.round(4).to_string())
savetab(bp_churn, "10_churn_by_billing")

print("\nChurn rate by country (top 10 by obs)")
co_churn = churn_by("country")
print(co_churn.head(10).round(4).to_string())
savetab(co_churn, "10_churn_by_country")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
ct_churn["churn_rate"].mul(100).plot(kind="barh", ax=axes[0], color="#c44e52")
axes[0].invert_yaxis()
axes[0].set(title="Monthly churn % by customer type", xlabel="churn rate (%)")
bp_churn["churn_rate"].mul(100).plot(kind="bar", ax=axes[1], color="#c44e52")
axes[1].set(title="Monthly churn % by billing period", xlabel="months")
co_churn.head(10)["churn_rate"].mul(100).plot(kind="barh", ax=axes[2], color="#c44e52")
axes[2].invert_yaxis()
axes[2].set(title="Monthly churn % by country (top 10)", xlabel="churn rate (%)")
plt.tight_layout()
savefig("10a_churn_by_segment")

# The interaction: type × billing period 
two_way = (panel_rich.groupby(["customer_type","billing_period_months"],
                              observed=True)
           .agg(n_obs=("churned_next_month","size"),
                churn_rate=("churned_next_month","mean"))
           .reset_index()
           .query("n_obs >= 100")
           .pivot(index="customer_type",
                  columns="billing_period_months",
                  values="churn_rate"))

print("\nMonthly churn: customer_type × billing_period (N≥100)")
print(two_way.round(4).to_string())
savetab(two_way, "10_churn_type_x_billing")

plt.figure(figsize=(9, 5))
sns.heatmap(two_way.mul(100).round(2), annot=True, fmt=".2f",
            cmap="YlOrRd", cbar_kws={"label": "monthly churn rate (%)"})
plt.title("Monthly churn rate (%): customer type × billing period")
savefig("10b_churn_heatmap")


# MRR at the start vs churn 
panel_with_mrr = panel_obs.copy()
panel_with_mrr["mrr_bin"] = pd.qcut(
    panel_with_mrr["mrr_end_of_month"].clip(lower=1),
    q=5, labels=["Q1 (lowest)","Q2","Q3","Q4","Q5 (highest)"],
    duplicates="drop"
)
mrr_churn = (panel_with_mrr.groupby("mrr_bin", observed=True)
             .agg(n_obs=("churned_next_month","size"),
                  churn_rate=("churned_next_month","mean"))
             .assign(annualised=lambda d: 1-(1-d["churn_rate"])**12))
print("\nChurn rate by MRR quintile")
print(mrr_churn.round(4).to_string())
savetab(mrr_churn, "10_churn_by_mrr_quintile")

plt.figure(figsize=(8, 4))
mrr_churn["churn_rate"].mul(100).plot(kind="bar", color="#c44e52")
plt.title("Monthly churn rate by MRR quintile")
plt.xlabel("MRR quintile")
plt.ylabel("monthly churn rate (%)")
plt.xticks(rotation=20)
savefig("10c_churn_by_mrr")


# 11.  What the data is asking
# A summary of the open questions the EDA has generated.

# Compute the recurrent figures 
_n_churn_custs = (panel.loc[panel["has_churn_event"]==1]
                   .groupby("customer_id")["period_month"].nunique())
_n_recurrent     = (_n_churn_custs > 1).sum()          # 1,182
_pct_of_all      = _n_recurrent / panel["customer_id"].nunique()  # % of all panel customers
_pct_of_churners = (_n_churn_custs > 1).mean()          # % of those who ever had a churn event
