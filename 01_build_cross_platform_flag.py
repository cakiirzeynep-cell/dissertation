from pathlib import Path
import pandas as pd

DATA = Path("/Users/zeynepcakir/Desktop/msc dissertation/data files ")
OUT  = Path("/Users/zeynepcakir/Desktop/msc dissertation/analysis/output")
OUT.mkdir(parents=True, exist_ok=True)

print("Task 1 — Build cross-platform flag")

mig = pd.read_csv(DATA / "migration_matches_anon.csv", low_memory=False)
print(f"\nMigration list: {len(mig):,} rows")
print(f"Unique AL IDs:  {mig['al_customer_id'].nunique():,}")
print(f"Unique AC IDs:  {mig['ac_customer_id'].nunique():,}")

# Dedup AL IDs — collect status patterns per customer where they're in multiple rows
al_dedup = (mig.groupby("al_customer_id")
              .agg(n_ac_links=("ac_customer_id", "nunique"),
                   migration_status_concat=("migration_status",
                                             lambda x: " | ".join(sorted(set(x)))))
              .reset_index())
al_dedup["is_cross_platform"] = 1
print(f"\nDeduplicated to {len(al_dedup):,} unique Artlogic customers")
print(f"Of which appear with multiple ArtCloud links: {(al_dedup['n_ac_links'] > 1).sum()}")

panel = pd.read_csv(DATA / "artlogic_panel_enriched.csv", low_memory=False)
print(f"\nEnriched panel: {panel.shape[0]:,} rows × {panel.shape[1]} cols")
print(f"Unique customers: {panel['customer_id'].nunique():,}")

panel_with_flag = panel.merge(
    al_dedup[["al_customer_id", "is_cross_platform", "n_ac_links",
              "migration_status_concat"]],
    left_on="customer_id", right_on="al_customer_id", how="left"
)
panel_with_flag["is_cross_platform"] = panel_with_flag["is_cross_platform"].fillna(0).astype(int)
panel_with_flag = panel_with_flag.drop(columns=["al_customer_id"])

# Verification
cust_summary = panel_with_flag.drop_duplicates("customer_id")
n_flagged = cust_summary["is_cross_platform"].sum()
n_total   = len(cust_summary)
pct       = n_flagged / n_total
print(f"\nVerification")
print(f"Customers flagged in panel: {n_flagged} / {n_total} ({pct:.1%})")

# Cross-platform AL customers not in panel
in_mig_al = set(al_dedup["al_customer_id"])
in_panel  = set(panel["customer_id"])
missing   = in_mig_al - in_panel
print(f"Cross-platform AL customers NOT in panel: {len(missing)}")
if missing:
    print(f"(probably customers without panel observations — line-level but no panel rows)")

print(f"\nCross-platform by customer_type")
cp_by_type = (cust_summary[cust_summary["is_cross_platform"] == 1]
              .groupby("customer_type").size()
              .sort_values(ascending=False))
print(cp_by_type.to_string())

out_path = DATA / "artlogic_panel_enriched_v2.csv"
panel_with_flag.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
print(f"  Shape: {panel_with_flag.shape}")

cp_ref = al_dedup.rename(columns={"al_customer_id": "customer_id"})
cp_ref_path = OUT / "01_cross_platform_customer_reference.csv"
cp_ref.to_csv(cp_ref_path, index=False)
print(f"  Reference table: {cp_ref_path}")
