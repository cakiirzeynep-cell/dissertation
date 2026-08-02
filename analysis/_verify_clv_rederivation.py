import numpy as np
import pandas as pd
from pathlib import Path

OUT  = Path(__file__).resolve().parent / "output"
DATA = Path(__file__).resolve().parent.parent / "data"
states = ["contracted", "stable", "expanded"]

# saved hazard coefficients 
coef = pd.read_csv(OUT / "05_hazard_extended_logit_coefficients.csv").set_index("covariate")["coef"].to_dict()
def g(name): return float(coef.get(name, 0.0))

# future-period term = mean of last 12 observed period dummies (recomputed)
period = sorted([(k, v) for k, v in coef.items() if k.startswith("mo_")])
last12_period_avg = float(np.mean([v for _, v in period[-12:]]))
print(f"last-12 period avg (recomputed): {last12_period_avg:+.4f}   (findings summary: +0.5519)")

# Markov matrices -> 3x3 per segment 
mm = pd.read_csv(OUT / "07_markov_per_segment_matrices.csv")
P_seg = {}
for seg, gdf in mm.groupby("customer_type"):
    P = gdf.pivot(index="from_state", columns="to_state", values="probability")
    P = P.reindex(index=states, columns=states).fillna(0.0).values
    P_seg[seg] = P

# MRR multipliers 
mult = pd.read_csv(OUT / "09_mrr_multipliers_per_segment.csv").set_index("segment")
r_seg = {s: mult.loc[s, ["r_contracted_median", "r_stable_median", "r_expanded_median"]].values.astype(float)
         for s in mult.index}

# per-customer covariates 
cust = pd.read_csv(OUT / "09_clv_per_customer.csv")
N = len(cust)
print(f"customers: {N}")

# fixed (non-tenure) part of linear predictor 
fixed_lp = np.full(N, g("const") + last12_period_avg)
for ct in ["Collector", "Artist", "Art Dealer", "Art advisory", "Other"]:
    fixed_lp += (cust["customer_type_grouped"] == ct).values * g(f"ct_{ct}")
for b in ["monthly", "annual", "other"]:
    fixed_lp += (cust["billing_grouped"] == b).values * g(f"bill_{b}")
for c in ["GB", "FR", "CH", "DE", "Other"]:
    fixed_lp += (cust["country_grouped"] == c).values * g(f"ctry_{c}")
fixed_lp += cust["is_cross_platform"].astype(float).values * g("is_cross_platform")
fixed_lp += np.log1p(cust["mrr_end_of_month"].values) * g("log_mrr")

iscp = cust["is_cross_platform"].astype(float).values
Pc = np.stack([P_seg[s] for s in cust["customer_type_grouped"]])
rc = np.stack([r_seg[s] for s in cust["customer_type_grouped"]])
edges = [3, 6, 12, 24, 36]
labels = np.array(["0-2", "3-5", "6-11", "12-23", "24-35", "36+"])

def recompute(T, d):
    pi = np.zeros((N, 3))
    for i, s in enumerate(cust["state_current"]):
        pi[i, states.index(s)] = 1.0
    mrr = cust["mrr_end_of_month"].values.astype(float).copy()
    tenure = cust["tenure_months_recomp"].values.astype(float).copy()
    surv = np.ones(N); clv = np.zeros(N)
    dm = (1 + d) ** (1 / 12) - 1
    for t in range(1, T + 1):
        pi = np.einsum("ni,nij->nj", pi, Pc)
        mrr = mrr * np.einsum("ni,ni->n", pi, rc)
        tenure = tenure + 1
        b = labels[np.digitize(tenure, edges)]
        tlp = np.zeros(N)
        for bn in ["0-2", "3-5", "6-11", "24-35", "36+"]:
            mask = (b == bn).astype(float)
            tlp += mask * g(f"ten_{bn}")
            tlp += mask * iscp * g(f"is_cp_x_ten_{bn}")
        p = 1.0 / (1.0 + np.exp(-(fixed_lp + tlp)))
        surv = surv * (1.0 - p)
        clv = clv + mrr * surv * (1 + dm) ** (-t)
    return clv

print("\nelement-wise re-derivation vs saved columns")
checks = [("clv_T60_d10", 60, 0.10), ("clv_T24_d10", 24, 0.10),
          ("clv_T60_d8", 60, 0.08), ("clv_T60_d12", 60, 0.12)]
for col, T, d in checks:
    mine = recompute(T, d)
    file = cust[col].values
    adiff = np.abs(mine - file)
    rel = adiff / np.maximum(np.abs(file), 1e-9)
    print(f"{col}: max|diff|=£{adiff.max():.4f}  mean|diff|=£{adiff.mean():.5f}  "
          f"max rel={rel.max():.2e}  | my mean £{mine.mean():,.2f} vs file £{file.mean():,.2f}  "
          f"my total £{mine.sum():,.0f} vs file £{file.sum():,.0f}")

print("\ninvariants (from saved file)")
mono_T = ((cust["clv_T60_d10"] >= cust["clv_T48_d10"] - 1e-6) &
          (cust["clv_T48_d10"] >= cust["clv_T36_d10"] - 1e-6) &
          (cust["clv_T36_d10"] >= cust["clv_T24_d10"] - 1e-6)).mean()
mono_d = ((cust["clv_T60_d8"] >= cust["clv_T60_d10"] - 1e-6) &
          (cust["clv_T60_d10"] >= cust["clv_T60_d12"] - 1e-6)).mean()
zero_mrr_zero_clv = (cust.loc[cust["mrr_end_of_month"] == 0, "clv_T60_d10"].abs() < 1e-6).mean() \
    if (cust["mrr_end_of_month"] == 0).any() else float("nan")
print(f"CLV non-decreasing in horizon T: {mono_T*100:.1f}% of customers")
print(f"CLV non-increasing in discount d: {mono_d*100:.1f}% of customers")
print(f"min CLV (should be >= 0): £{cust['clv_T60_d10'].min():.2f}")
print(f"zero-MRR customers with CLV 0: {zero_mrr_zero_clv*100 if zero_mrr_zero_clv==zero_mrr_zero_clv else float('nan'):.1f}%")

print("\nprojection-population check (the '5,988 active at panel end' claim)")
panel = pd.read_csv(DATA / "artlogic_panel_enriched_v2.csv", low_memory=False)
panel["period_month"] = pd.to_datetime(panel["period_month"], format="%Y-%m")
lastm = panel["period_month"].max()
pe = panel[(~panel["kept_changed_sub_flag"]) & (panel["period_month"] < lastm)]
model_max = pe["period_month"].max()
last_rows = pe.sort_values("period_month").drop_duplicates("customer_id", keep="last")
lr = last_rows[last_rows["customer_id"].isin(set(cust["customer_id"]))]
print(f"modelling-panel last month: {model_max.date()}")
print(f"of {len(lr)} projected customers, last obs == panel-max month: {(lr['period_month']==model_max).sum()}")
print(f"of {len(lr)} projected customers, churned_next_month==1 at last obs: {int(lr['churned_next_month'].sum())}")
print("last-observed-month distribution (tail):")
print(lr["period_month"].dt.strftime("%Y-%m").value_counts().sort_index().tail(8).to_string())
