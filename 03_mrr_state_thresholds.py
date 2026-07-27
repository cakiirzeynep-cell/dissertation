from pathlib import Path
import pandas as pd
import numpy as np

DATA = Path("/Users/zeynepcakir/Desktop/msc dissertation/data files ")
OUT  = Path("/Users/zeynepcakir/Desktop/msc dissertation/analysis/output")

panel = pd.read_csv(DATA / "artlogic_panel_enriched_v2.csv", low_memory=False)
panel["period_month"] = pd.to_datetime(panel["period_month"], format="%Y-%m")

print("Task 3 — MRR state thresholds for Markov")

panel = panel.sort_values(["customer_id", "period_month"]).reset_index(drop=True)

panel["mrr_prev"] = panel.groupby("customer_id")["mrr_end_of_month"].shift(1)

# Keep only customer-months where prev MRR is available and > 0
# (state transition only defined when both months are observed)
valid = panel.dropna(subset=["mrr_end_of_month", "mrr_prev"]).copy()
valid = valid.loc[valid["mrr_prev"] > 0]
print(f"\nValid customer-month transitions (prev MRR > 0): {len(valid):,}")

valid["mrr_change_pct"] = (valid["mrr_end_of_month"] - valid["mrr_prev"]) / valid["mrr_prev"]

# Define state per threshold
def classify(pct_change, threshold):
    if pct_change > threshold:
        return "expanded"
    elif pct_change < -threshold:
        return "contracted"
    else:
        return "stable"

thresholds = {"±3%": 0.03, "±5%": 0.05, "±10%": 0.10}
state_dists = {}
for label, t in thresholds.items():
    valid[f"state_{label}"] = valid["mrr_change_pct"].apply(lambda x: classify(x, t))
    state_dists[label] = valid[f"state_{label}"].value_counts(normalize=True)

print("\nState distribution by threshold")
dist_df = pd.DataFrame(state_dists).fillna(0)
dist_df = dist_df.reindex(["stable", "expanded", "contracted"])
print((dist_df * 100).round(1).astype(str) + "%")

# Save state distributions
dist_path = OUT / "03_mrr_state_distribution.csv"
dist_df.to_csv(dist_path)
print(f"\nSaved: {dist_path}")

# Verify EDA's headline at default ±5%
print(f"\nEDA reconciliation at default ±5%")
exp_stable = state_dists["±5%"].get("stable", 0)
exp_expand = state_dists["±5%"].get("expanded", 0)
exp_contract = state_dists["±5%"].get("contracted", 0)
print(f"  Stable:     {exp_stable*100:.1f}% (EDA expected ~70%)")
print(f"  Expanded:   {exp_expand*100:.1f}% (EDA expected ~15.5%)")
print(f"  Contracted: {exp_contract*100:.1f}% (EDA expected ~5%)")
print(f"  (NB: EDA also counted first-month / churn / recovery cases not modelled here)")

# Decision: commit to ±5% as default
print(f"\nDecision")
print(f"Default threshold for Markov state assignment: ±5% (per EDA)")
print(f"Sensitivity analysis: refit transition matrices at ±3% and ±10%")

# Save per-customer-month default state assignment for downstream use
default_state = valid[["customer_id", "period_month", "mrr_prev",
                       "mrr_end_of_month", "mrr_change_pct",
                       "state_±5%"]].rename(columns={"state_±5%": "mrr_state"})
state_path = OUT / "03_mrr_state_per_customer_month.csv"
default_state.to_csv(state_path, index=False)
print(f"\nDefault-threshold state assignment saved: {state_path}")
print(f"  Shape: {default_state.shape}")
