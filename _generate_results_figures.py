import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).resolve().parent / "output"
FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(exist_ok=True)

# palette 
C_INPUT    = "#2E5266"   # deep teal-blue
C_PIPELINE = "#3F6634"   # forest green
C_OUTPUT   = "#8C3F2F"   # brick
C_TEXT     = "#1A1A1A"
C_MUTED    = "#5B6770"
C_RULE     = "#C8CFD3"
C_INPUT_TINT    = "#EEF2F5"
C_PIPELINE_TINT = "#EFF4EF"
C_OUTPUT_TINT   = "#F9F0EE"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.edgecolor": C_MUTED,
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.color": C_RULE,
    "grid.linewidth": 0.6,
    "xtick.color": C_TEXT,
    "ytick.color": C_TEXT,
    "axes.labelcolor": C_TEXT,
    "text.color": C_TEXT,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

def despine(ax, left=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if not left:
        ax.spines["left"].set_visible(False)

def gbp(x, _=None):
    return f"£{x:,.0f}"

# Per-segment mean CLV with 95% bootstrap CIs (SQ1)
d = pd.read_csv(OUT / "10_segment_clv_customer_type.csv").sort_values("mean_CLV_GBP")
fig, ax = plt.subplots(figsize=(7.2, 4.2))
y = np.arange(len(d))
err = np.vstack([d["mean_CLV_GBP"] - d["mean_CI_lo_95"], d["mean_CI_hi_95"] - d["mean_CLV_GBP"]])
ax.barh(y, d["mean_CLV_GBP"], color=C_INPUT, height=0.62, zorder=3)
ax.errorbar(d["mean_CLV_GBP"], y, xerr=err, fmt="none", ecolor=C_TEXT, elinewidth=1.1, capsize=4, zorder=4)
for yi, (m, hi, n) in enumerate(zip(d["mean_CLV_GBP"], d["mean_CI_hi_95"], d["n_customers"])):
    ax.text(hi + 450, yi, f"{gbp(m)}  (n={n:,})", va="center", ha="left", fontsize=8.5, color=C_TEXT)
ax.set_yticks(y); ax.set_yticklabels(d["customer_type_grouped"])
ax.set_xlabel("Mean projected CLV (£), base case T = 60 months, d = 10%")
ax.set_title("Figure 4.1  Mean projected CLV by customer-type segment, with 95% bootstrap CIs")
ax.set_xlim(0, d["mean_CI_hi_95"].max() + 6200)
ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(gbp))
ax.grid(axis="y", visible=False)
despine(ax)
fig.savefig(FIG / "fig4_1_segment_clv_v1.png"); plt.close(fig)

# Lorenz curve of CLV concentration (SQ1)
clv = pd.read_csv(OUT / "09_clv_per_customer.csv")["clv_T60_d10"].clip(lower=0).sort_values().values
n = len(clv)
cum = np.cumsum(clv) / clv.sum()
x = np.arange(1, n + 1) / n
x = np.insert(x, 0, 0); cum = np.insert(cum, 0, 0)
top_decile_share = clv[int(np.ceil(0.9 * n)):].sum() / clv.sum()
fig, ax = plt.subplots(figsize=(6.4, 5.0))
ax.plot([0, 1], [0, 1], ls="--", lw=1.0, color=C_MUTED, label="Line of equality")
ax.plot(x, cum, lw=2.2, color=C_OUTPUT, label="CLV Lorenz curve")
ax.fill_between(x, cum, x, color=C_OUTPUT_TINT, zorder=1)
ax.axvline(0.9, ls=":", lw=1.0, color=C_TEXT)
ax.scatter([0.9], [1 - top_decile_share], color=C_TEXT, s=28, zorder=5)
ax.annotate(f"Top 10% of customers\nhold {top_decile_share*100:.1f}% of total value",
            xy=(0.9, 1 - top_decile_share), xytext=(0.40, 0.30),
            fontsize=9.5, color=C_TEXT,
            arrowprops=dict(arrowstyle="->", color=C_TEXT, lw=1.0))
ax.set_xlabel("Cumulative share of customers (ranked by CLV)")
ax.set_ylabel("Cumulative share of customer-base value")
ax.set_title("Figure 4.2  Concentration of projected customer-base value")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.legend(loc="upper left", frameon=False, fontsize=9)
despine(ax)
fig.savefig(FIG / "fig4_2_clv_lorenz_v1.png"); plt.close(fig)

# Revenue-trajectory contribution by segment, Markov - flat (SQ2)
s = pd.read_csv(OUT / "11_sq2_decomposition_by_segment.csv")
s = s[s["dimension"] == "customer_type_grouped"].copy().sort_values("mean_contribution")
colours = [C_MUTED if abs(v) < 1 else C_PIPELINE for v in s["mean_contribution"]]
fig, ax = plt.subplots(figsize=(7.2, 4.2))
yy = np.arange(len(s))
ax.barh(yy, s["mean_contribution"], color=colours, height=0.62, zorder=3)
for yi, v in zip(yy, s["mean_contribution"]):
    if abs(v) < 1:
        ax.text(60, yi, "0  (median multipliers = 1.0)", va="center", ha="left", fontsize=8.3, color=C_MUTED)
    else:
        ax.text(v + 80, yi, gbp(v), va="center", ha="left", fontsize=8.5, color=C_TEXT)
ax.set_yticks(yy); ax.set_yticklabels(s["segment"])
ax.set_xlabel("Mean revenue-trajectory contribution per customer (£): Markov minus flat-MRR")
ax.set_title("Figure 4.3  Contribution of revenue-trajectory modelling to CLV, by segment")
ax.set_xlim(0, s["mean_contribution"].max() * 1.25)
ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(gbp))
ax.grid(axis="y", visible=False)
despine(ax)
fig.savefig(FIG / "fig4_3_sq2_contribution_v1.png"); plt.close(fig)

# APC period coefficients: full vs restricted (SQ3)
a = pd.read_csv(OUT / "08_apc_period_comparison.csv")
r = np.corrcoef(a["coef_full"], a["coef_restricted"])[0, 1]
fig, ax = plt.subplots(figsize=(6.0, 5.6))
lo = min(a["coef_full"].min(), a["coef_restricted"].min()) - 0.2
hi = max(a["coef_full"].max(), a["coef_restricted"].max()) + 0.2
ax.plot([lo, hi], [lo, hi], ls="--", lw=1.0, color=C_MUTED, label="Exact agreement (y = x)")
ax.scatter(a["coef_full"], a["coef_restricted"], color=C_INPUT, s=42, alpha=0.85, edgecolor="white", linewidth=0.5, zorder=4)
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_xlabel("Full-fit period coefficient (log-odds of monthly churn)")
ax.set_ylabel("Restricted-fit period coefficient")
ax.set_title("Figure 4.4  Period coefficients, full panel vs post-migration restricted fit")
ax.text(0.05, 0.92, f"Pearson r = {r:.2f}\n(30 common periods)", transform=ax.transAxes,
        fontsize=10, va="top", bbox=dict(boxstyle="round,pad=0.4", fc=C_INPUT_TINT, ec=C_MUTED, lw=0.8))
ax.legend(loc="lower right", frameon=False, fontsize=9)
despine(ax)
fig.savefig(FIG / "fig4_4_apc_period_scatter_v1.png"); plt.close(fig)
print(f"APC scatter computed r = {r:.4f}")

# Cross-platform hazard ratio by tenure (forest plot) (SQ4 mechanism)
h = pd.read_csv(OUT / "05_hazard_extended_interaction.csv")
order = ["0-2", "3-5", "6-11", "12-23", "24-35", "36+"]
h = h.set_index("tenure_bin").loc[order].reset_index()
hr = h["is_cp_HR_total"].values
lo_ci = np.exp(h["is_cp_coef_total"] - 1.96 * h["approx_se"]).values
hi_ci = np.exp(h["is_cp_coef_total"] + 1.96 * h["approx_se"]).values
pvals = h["interaction_p"].values
yy = np.arange(len(order))[::-1]   # 0-2 at top
fig, ax = plt.subplots(figsize=(7.0, 4.4))
ax.axvline(1.0, ls="--", lw=1.0, color=C_MUTED)
for i, yi in enumerate(yy):
    early = order[i] in ("0-2", "3-5")
    col = C_OUTPUT if early else C_INPUT
    ax.plot([lo_ci[i], hi_ci[i]], [yi, yi], color=col, lw=1.6, zorder=3)
    ax.scatter([hr[i]], [yi], color=col, s=55, zorder=4)
    lab = f"HR {hr[i]:.2f}"
    if not np.isnan(pvals[i]):
        if pvals[i] < 0.10:
            lab += f"  (p = {pvals[i]:.3f}, borderline)"
    else:
        lab += "  (reference tenure)"
    ax.text(hi_ci[i] * 1.04, yi, lab, va="center", ha="left", fontsize=8.3, color=C_TEXT)
ax.set_xscale("log")
ax.set_xticks([0.5, 1, 2, 4, 8])
ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
ax.set_yticks(yy); ax.set_yticklabels([f"{o} months" for o in order])
ax.set_xlabel("Cross-platform hazard ratio vs single-platform at same tenure (log scale)")
ax.set_title("Figure 4.6  Cross-platform churn hazard is concentrated in early tenure")
ax.set_xlim(0.4, 16)
ax.grid(axis="y", visible=False)
despine(ax)
fig.savefig(FIG / "fig4_6_crossplatform_tenure_forest_v1.png"); plt.close(fig)

# Retention value protected, three slices (SQ4)
pop = pd.read_csv(OUT / "12_sq4_retention_population.csv")
top = pd.read_csv(OUT / "12_sq4_retention_top_decile.csv")
levels = [5, 10, 20]
pop_d = [pop.loc[pop.hazard_reduction_pct == L, "delta_total_GBP"].iloc[0] for L in levels]
top_d = [top.loc[top.hazard_reduction_pct == L, "delta_total_GBP"].iloc[0] for L in levels]
pop_pc = pop.loc[pop.hazard_reduction_pct == 10, "delta_per_targeted_customer_GBP"].iloc[0]
top_pc = top.loc[top.hazard_reduction_pct == 10, "delta_per_targeted_customer_GBP"].iloc[0]
lev = top.loc[top.hazard_reduction_pct == 10, "leverage_ratio_vs_population"].iloc[0]

fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.8, 4.4), gridspec_kw={"width_ratios": [1.5, 1]}, layout="constrained")
w = 0.38; xx = np.arange(len(levels))
money_lbl = lambda v: (f"£{v/1e6:.2f}M" if v >= 1e6 else f"£{v/1e3:.0f}k")
axL.bar(xx - w/2, pop_d, w, color=C_INPUT, label="Population-wide (5,988)", zorder=3)
axL.bar(xx + w/2, top_d, w, color=C_OUTPUT, label="Top decile by MRR (600)", zorder=3)
for i, (a_, b_) in enumerate(zip(pop_d, top_d)):
    axL.text(i - w/2, a_ + 4e4, money_lbl(a_), ha="center", fontsize=8, color=C_TEXT)
    axL.text(i + w/2, b_ + 4e4, money_lbl(b_), ha="center", fontsize=8, color=C_TEXT)
axL.set_xticks(xx); axL.set_xticklabels([f"{L}%" for L in levels])
axL.set_xlabel("Monthly churn-hazard reduction")
axL.set_ylabel("Total projected value protected (£)")
axL.set_title("Total value protected, by targeting slice")
axL.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"£{v/1e6:.0f}M"))
axL.legend(frameon=False, fontsize=8.5, loc="upper left")
axL.grid(axis="x", visible=False); despine(axL)

axR.bar([0, 1], [pop_pc, top_pc], color=[C_INPUT, C_OUTPUT], width=0.6, zorder=3)
for i, v in enumerate([pop_pc, top_pc]):
    axR.text(i, v + 25, gbp(v), ha="center", fontsize=9, color=C_TEXT)
axR.set_xticks([0, 1]); axR.set_xticklabels(["Population\nwide", "Top decile\nby MRR"])
axR.set_ylabel("Value protected per targeted customer (£), 10% reduction")
axR.set_title(f"Per-customer leverage: {lev:.2f}×")
axR.set_ylim(0, top_pc * 1.28)
axR.grid(axis="x", visible=False); despine(axR)
fig.suptitle("Figure 4.5  Retention leverage: population-wide vs top-decile-by-MRR targeting", fontsize=12, fontweight="bold")
fig.savefig(FIG / "fig4_5_retention_slices_v1.png"); plt.close(fig)

