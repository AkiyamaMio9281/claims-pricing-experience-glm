# Auto Insurance Pricing Experience Study (Two-Part GLM)

An end-to-end auto insurance pricing prototype that builds a clean policy-level modeling table from **frequency** and **severity** sources, fits a **two-part GLM** framework, and produces portfolio/segment experience tables plus an Excel dashboard (**Summary / Segments / Drilldown**).

---

## Project snapshot

- **Frequency policies:** 678,013  
- **Claim-bearing policies (severity available):** 24,944  
- **Models:**
  - Frequency: **Poisson GLM** with **log(Exposure)** offset
  - Severity: **Gamma GLM** with **log link** (target = AvgSeverity)
- **Test-set lift vs baseline:**
  - Poisson deviance improvement ≈ **1,525**
  - Gamma deviance improvement ≈ **20.7**
- **Key QA finding:** `ClaimNb > 0` but `PaidLoss == 0` = **9,116** policies (missing severity coverage)
- **Dashboard drilldown ranking:** policies ranked by **Predicted Total Loss**  
  `pred_loss_total = pred_pure_premium × Exposure`  
  (more actionable than ranking by annualized rate alone under long-tailed frequency predictions)

---

## What’s inside

### 1) Data engineering + QA
- Aggregates claim transactions to policy level
- Merges frequency and severity sources
- Performs consistency checks and exports anomaly tables

### 2) Modeling (two-part)
- **Frequency (Poisson):** predicts expected claim count per policy-year and converts to rate via exposure
- **Severity (Gamma):** predicts expected average severity on claim-bearing policies
- Combines: `PredPurePremium = PredFrequency × PredAvgSeverity`

### 3) Experience study + dashboard tables
- Generates segment tables (Top Regions, Region×Area, Fuel, Brand, etc.)
- Produces a policy-level drilldown table designed for dashboard slicing and ranking

---

## Data (not included in repo)

This repo assumes you have two raw sources (CSV) under `data/raw/`:

### Frequency (policy level)
Typical columns:
- `IDpol` (policy id)
- `Exposure` (earned exposure)
- `ClaimNb` (claim count)
- plus rating variables used as features (e.g., `Region`, `Area`, `VehBrand`, `VehGas`, `VehAge`, `DrivAge`, `VehPower`, `BonusMalus`, `Density`, ...)

### Severity (claim transactions)
Typical columns:
- `IDpol`
- `PaidLoss` (paid amount per transaction / claim record)
- may contain multiple rows per policy

> Note: Severity coverage can be incomplete. The pipeline documents missing severity records and handles them explicitly in downstream reporting.

---

## Pipeline overview (scripts)

All scripts live in `src/` and write artifacts to `outputs/` and `logs/`.

- `02_build_model_table.py`  
  Builds a policy-level modeling table; merges frequency & severity; exports `data/processed/model_table.csv`.

- `03_eda_qa.py`  
  QA + EDA summaries; exports anomaly tables and figures.

- `05_train_frequency_glm.py`  
  Trains Poisson GLM (log offset); exports coefficients and predictions (including `freq_predictions_all.csv`).

- `06_train_severity_glm.py`  
  Trains Gamma GLM (log link) on claim-bearing policies; exports coefficients and predictions (including `sev_predictions_all.csv`).

- `07_build_dashboard_tables.py`  
  Builds policy-level dashboard table:
  - `pred_frequency`, `pred_avg_severity`, `pred_pure_premium`
  - `pred_claims_total = pred_frequency × Exposure`
  - `pred_loss_total   = pred_pure_premium × Exposure`
  Exports `outputs/tables/dashboard_policy_level.csv`.

- `08_experience_study.py`  
  Builds segment-level experience tables and a portfolio summary.

Optional debugging:
- `10_debug_pred_pp_tail.py`  
  Quantifies tail behavior and validates `pred_pure_premium ≈ pred_frequency × pred_avg_severity`.

---

## Key outputs

### Tables (CSV)
- `outputs/tables/dashboard_policy_level.csv`  
  Policy-level drilldown table for Excel (slicers + Top-N ranking by `pred_loss_total`)

- `outputs/tables/experience_by_region.csv`  
- `outputs/tables/experience_by_region_area.csv`  
- `outputs/tables/experience_by_area.csv`  
- `outputs/tables/experience_by_vehgas.csv`  
- `outputs/tables/experience_by_vehbrand_top15_plus_other.csv`  
- `outputs/tables/portfolio_summary.csv`  

Model artifacts:
- `outputs/tables/freq_glm_coefficients.csv`
- `outputs/tables/sev_glm_coefficients.csv`
- `outputs/tables/freq_predictions_all.csv`
- `outputs/tables/sev_predictions_all.csv`

### Logs
- `logs/02_build_model_table_report.txt`
- `logs/03_eda_qa_report.txt`
- `logs/05_train_frequency_glm_report.txt`
- `logs/06_train_severity_glm_report.txt`
- `logs/07_build_dashboard_tables_report.txt`
- `logs/08_experience_study_report.txt`

---

## Excel dashboard (recommended layout)

- **Summary:** portfolio KPIs + Top 10 Regions (Pred vs Actual Pure Premium)
- **Segments:** Top 10 Region–Area (Pred vs Actual Pure Premium; with exposure thresholding)
- **Drilldown:** Top-N policies ranked by **pred_loss_total** with slicers (Region / Area / Fuel / Brand)

> Rationale: annualized pure premium can show long tails driven by Poisson log-link frequency predictions; ranking by predicted **total loss** highlights material portfolio contributors.

---

## How to run

### 1) Create environment
```bash
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Place data
Put raw CSVs under:
```
data/raw/
  freq.csv
  sev.csv
```

### 3) Run pipeline
If you have a batch runner:
```bash
run_pipeline.bat
```

Or run step-by-step:
```bash
py src/02_build_model_table.py
py src/03_eda_qa.py
py src/05_train_frequency_glm.py
py src/06_train_severity_glm.py
py src/07_build_dashboard_tables.py
py src/08_experience_study.py
```

### 4) Build/refresh dashboard
- Open the Excel dashboard file
- Refresh data connections (`Refresh All`)
- Verify drilldown ranks by `pred_loss_total`

---

## Repository layout

```
claims-pricing-experience-glm/
  assets/                  # dashboard screenshots (recommended)
  data/
    raw/                   # (not committed) input CSVs
    processed/             # pipeline outputs (optional, usually not committed)
  outputs/
    tables/                # generated tables (usually not committed)
    figures/               # generated figures
  logs/                    # run reports (usually not committed)
  src/                     # pipeline scripts
  README.md
  requirements.txt
  run_pipeline.bat
```

---

## Limitations & notes

- **Incomplete severity coverage:** a subset of policies have `ClaimNb > 0` but no severity records; reported explicitly.
- **Long tail in predicted rates:** frequency under a log-link model can yield heavy tails for rare feature combinations; drilldown uses `pred_loss_total` for practical ranking.
- **Simplified modeling:** this is a pricing prototype focused on interpretability and reproducible reporting.

---

## Tech stack
Python, pandas, NumPy, statsmodels, scikit-learn (metrics/splitting), matplotlib, Excel (Power Query / Data Model / PivotTables).

---

## License
No license specified.
