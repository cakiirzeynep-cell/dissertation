# Estimating Customer Lifetime Value in a B2B SaaS Business

Analysis code for the MSc dissertation "Estimating Customer Lifetime Value in a B2B SaaS Business Using Contract and Revenue Data", carried out with the industry partner Artlogic.

The pipeline estimates and decomposes per-customer CLV by combining a discrete-time logistic hazard model (survival), a Markov MRR transition model (revenue dynamics), and an age–period–cohort decomposition (trend), then translates the results into segment-level retention-investment scenarios.

## Data availability

The underlying Artlogic customer data cannot be shared under a confidentiality agreement, so this repository contains code only. The scripts expect the anonymised extracts in a `data/` folder at the repository root that is not included. The data sources, ETL steps, and full variable dictionary are documented in the dissertation's Technical Appendix (Sections A.1–A.4).

## Environment

Python 3.10+ is recommended. Install the dependencies into a fresh virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run order

The numbered scripts run in sequence; each reads the panel or intermediate outputs produced upstream. Run them from the `analysis/` folder: `cd analysis`, then `python build_panel_enriched.py`, then `01`–`12` in order.

| Order | Script | What it does |
|:---:|---|---|
| — | `build_panel_enriched.py` | Builds the enriched monthly customer panel from the raw extracts (activation-date recompute, customer attributes, churn flags). |
| 01 | `01_build_cross_platform_flag.py` | Adds the `is_cross_platform` indicator by matching Artlogic and ArtCloud customers via the migration file; writes the enriched panel used downstream. |
| 02 | `02_verify_enriched_panel.py` | Integrity checks on the enriched panel (row/customer counts, churn rates under each exclusion rule). |
| 03 | `03_mrr_state_thresholds.py` | Classifies month-on-month MRR change into contracted / stable / expanded states at the 5% threshold. |
| 04 | `04_hazard_model_first_pass.py` | Fits the first-pass discrete-time logistic hazard model and the complementary log-log comparator; reports held-out AUC and Brier score. |
| 05 | `05_hazard_model_extended.py` | Extended hazard model with cross-platform × tenure interactions and average marginal effects. |
| 06 | `06_sbg_benchmark.py` | Fits the shifted Beta-Geometric benchmark and compares it against the hazard model on the held-out test window. |
| 07 | `07_markov_mrr_transitions.py` | Estimates per-segment Markov MRR transition matrices and steady-state distributions. |
| 08 | `08_apc_dual_fit.py` | Age–period–cohort dual-fit decomposing the cancellation trend (SQ3): full panel vs post-migration restriction. |
| 09 | `09_clv_pipeline.py` | Combines survival, Markov revenue, and discounting into per-customer CLV, with the T and discount-rate sensitivity grid. |
| 10 | `10_segment_aggregation.py` | Aggregates per-customer CLV to customer-type and billing-structure segments, with bootstrap confidence intervals (SQ1). |
| 11 | `11_sq2_decomposition.py` | Quantifies the revenue-trajectory contribution to CLV over a flat-MRR baseline (SQ2). |
| 12 | `12_sq4_retention_sensitivity.py` | Retention-sensitivity scenarios and leverage under 5% / 10% / 20% hazard reductions (SQ4). |

## Supporting scripts

| Script | What it does |
|---|---|
| `eda_open.py` | Exploratory data analysis behind the methodology's EDA section. |
| `_generate_results_figures.py` | Regenerates the Chapter 4 figures from the saved analysis outputs. |
| `_verify_clv_rederivation.py` | Independently re-derives per-customer CLV from the saved model components as a cross-check. |

## Outputs

The scripts write intermediate CSVs and findings summaries to `analysis/output/`, and figures to `analysis/figures/`. `eda_open.py` writes its exploratory outputs to `analysis/output_open/` and `analysis/figures_open/`. All are regenerated on each run and are not committed to the repository.
The Chapter 4 figures are included under `analysis/figures/`. Result tables are reproduced in the dissertation.
