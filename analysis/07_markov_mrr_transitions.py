from pathlib import Path
import pandas as pd
import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data"
OUT  = Path(__file__).resolve().parent / "output"

# Markov MRR transition matrices
# 1. Load enriched panel + state assignments
panel = pd.read_csv(DATA / "artlogic_panel_enriched_v2.csv", low_memory=False)
panel["period_month"] = pd.to_datetime(panel["period_month"], format="%Y-%m")

# Apply standard exclusions (kept-changed + final month)
last_month = panel["period_month"].max()
panel = panel.loc[
    (~panel["kept_changed_sub_flag"]) &
    (panel["period_month"] < last_month)
].copy()

panel = panel.dropna(subset=["customer_type", "mrr_end_of_month"])

top5_types = ["Gallery", "Collector", "Artist", "Art Dealer", "Art advisory"]
panel["customer_type_grouped"] = panel["customer_type"].apply(
    lambda x: x if x in top5_types else "Other"
)

# Sort and compute prev MRR + state at default ±5%
panel = panel.sort_values(["customer_id", "period_month"]).reset_index(drop=True)
panel["mrr_prev"] = panel.groupby("customer_id")["mrr_end_of_month"].shift(1)

THRESH = 0.05
def classify(row):
    if pd.isna(row["mrr_prev"]) or row["mrr_prev"] == 0:
        return None
    change = (row["mrr_end_of_month"] - row["mrr_prev"]) / row["mrr_prev"]
    if change > THRESH:
        return "expanded"
    elif change < -THRESH:
        return "contracted"
    else:
        return "stable"
panel["state_current"] = panel.apply(classify, axis=1)
panel["state_next"] = panel.groupby("customer_id")["state_current"].shift(-1)

# Conditional on survival: drop the row where the customer churns
# (state transition only defined when the customer survives to next period)
panel["survived_next"] = (1 - panel["churned_next_month"]).astype(bool)

# Keep only valid transitions: have current state, have next state, customer survives
valid = panel.dropna(subset=["state_current", "state_next"]).copy()
valid = valid.loc[valid["survived_next"]]
print(f"\nValid customer-month transitions for Markov fitting: {len(valid):,}")

# 2. Aggregate transition matrix
states = ["contracted", "stable", "expanded"]

def transition_matrix(df):
    counts = pd.crosstab(df["state_current"], df["state_next"])
    # Reindex to ensure consistent ordering
    counts = counts.reindex(index=states, columns=states, fill_value=0)
    # Row-normalise to get probabilities
    row_sums = counts.sum(axis=1)
    probs = counts.div(row_sums, axis=0).fillna(0)
    return counts, probs

print("\nAggregate transition matrix (counts)")
agg_counts, agg_probs = transition_matrix(valid)
print(agg_counts.to_string())
print("\nAggregate transition matrix (probabilities)")
print(agg_probs.round(4).to_string())

# Save aggregate matrix
agg_out = agg_probs.copy()
agg_out.index.name = "from_state"
agg_out.to_csv(OUT / "07_markov_aggregate_matrix.csv")

# 3. Per-customer_type transition matrices
print("\nPer-customer_type transition matrices")
per_seg_rows = []
for seg in sorted(valid["customer_type_grouped"].unique()):
    seg_df = valid.loc[valid["customer_type_grouped"] == seg]
    if len(seg_df) < 100:
        print(f"  {seg}: skipped (insufficient transitions)")
        continue
    counts, probs = transition_matrix(seg_df)
    print(f"\n  [{seg}] n = {len(seg_df):,} transitions")
    print(probs.round(4).to_string())

    # Flatten into long format for the per-segment CSV
    for from_s in states:
        for to_s in states:
            per_seg_rows.append({
                "customer_type": seg,
                "from_state": from_s,
                "to_state": to_s,
                "count": int(counts.loc[from_s, to_s]),
                "probability": float(probs.loc[from_s, to_s]),
                "row_total": int(counts.loc[from_s].sum()),
            })

per_seg_df = pd.DataFrame(per_seg_rows)
per_seg_df.to_csv(OUT / "07_markov_per_segment_matrices.csv", index=False)

# 4. Steady-state distribution per segment
print("\nSteady-state distribution per customer_type")
print("(long-run probability of being in each state given indefinite survival)")

def steady_state(probs_matrix):
    P = probs_matrix.values
    eigvals, eigvecs = np.linalg.eig(P.T)
    idx = np.argmin(np.abs(eigvals - 1.0))
    ss = np.real(eigvecs[:, idx])
    ss = ss / ss.sum()
    return ss

ss_rows = []
print(f"\n  {'segment':15s}  {'contracted':>11s}  {'stable':>11s}  {'expanded':>11s}")
for seg in sorted(valid["customer_type_grouped"].unique()):
    seg_df = valid.loc[valid["customer_type_grouped"] == seg]
    if len(seg_df) < 100:
        continue
    counts, probs = transition_matrix(seg_df)
    ss = steady_state(probs)
    print(f"  {seg:15s}  {ss[0]:>11.4f}  {ss[1]:>11.4f}  {ss[2]:>11.4f}")
    ss_rows.append({
        "customer_type": seg,
        "steady_contracted": ss[0],
        "steady_stable": ss[1],
        "steady_expanded": ss[2],
        "n_transitions": len(seg_df),
    })

agg_ss = steady_state(agg_probs)
print(f"  {'AGGREGATE':15s}  {agg_ss[0]:>11.4f}  {agg_ss[1]:>11.4f}  {agg_ss[2]:>11.4f}")
ss_rows.append({
    "customer_type": "AGGREGATE",
    "steady_contracted": agg_ss[0],
    "steady_stable": agg_ss[1],
    "steady_expanded": agg_ss[2],
    "n_transitions": len(valid),
})
ss_df = pd.DataFrame(ss_rows)
ss_df.to_csv(OUT / "07_markov_steady_state.csv", index=False)
