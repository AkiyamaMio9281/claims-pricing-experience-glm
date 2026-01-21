import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IN_PATH  = os.path.join(PROJECT_ROOT, "data", "processed", "model_table.csv")
OUT_TBL  = os.path.join(PROJECT_ROOT, "outputs", "tables")
OUT_FIG  = os.path.join(PROJECT_ROOT, "outputs", "figures")
LOG_DIR  = os.path.join(PROJECT_ROOT, "logs")

REPORT_TXT = os.path.join(LOG_DIR, "03_eda_qa_report.txt")
ANOM_CSV   = os.path.join(OUT_TBL, "anomalies_claimnb_gt0_paidloss_eq0.csv")
SUMMARY_CSV= os.path.join(OUT_TBL, "eda_summary.csv")
SEG_CSV    = os.path.join(OUT_TBL, "experience_by_region.csv")

def ensure_dirs():
    os.makedirs(OUT_TBL, exist_ok=True)
    os.makedirs(OUT_FIG, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

def save_hist(series, title, filename, logy=False):
    plt.figure()
    x = series.dropna().values
    plt.hist(x, bins=60)
    plt.title(title)
    if logy:
        plt.yscale("log")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_FIG, filename), dpi=160)
    plt.close()

def main():
    ensure_dirs()
    df = pd.read_csv(IN_PATH)

    # --- Basic counts ---
    n = len(df)
    n_claim = int((df["ClaimNb"] > 0).sum())
    n_paid  = int((df["PaidLoss"] > 0).sum())

    # --- Key anomaly: ClaimNb>0 but PaidLoss==0 ---
    anom = df[(df["ClaimNb"] > 0) & (df["PaidLoss"] == 0)].copy()
    anom.to_csv(ANOM_CSV, index=False)

    # helpful breakdown: are these missing sev records (ClaimRecords==0)?
    if "ClaimRecords" in df.columns:
        anom_missing_sev = int((anom["ClaimRecords"] == 0).sum())
        anom_has_sevrec  = int((anom["ClaimRecords"] > 0).sum())
    else:
        anom_missing_sev = None
        anom_has_sevrec  = None

    # --- Summary stats for key numeric columns ---
    num_cols = ["Exposure", "ClaimNb", "PaidLoss", "PurePremium", "AvgSeverity", "Density",
                "VehPower", "VehAge", "DrivAge", "BonusMalus"]
    rows = []
    for c in num_cols:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            rows.append({
                "col": c,
                "missing": int(s.isna().sum()),
                "min": float(np.nanmin(s.values)),
                "p50": float(np.nanpercentile(s.values, 50)),
                "p90": float(np.nanpercentile(s.values, 90)),
                "p95": float(np.nanpercentile(s.values, 95)),
                "p99": float(np.nanpercentile(s.values, 99)),
                "max": float(np.nanmax(s.values)),
            })
    pd.DataFrame(rows).to_csv(SUMMARY_CSV, index=False)

    # --- Experience study by Region (quick sanity + later dashboard) ---
    if "Region" in df.columns:
        tmp = df.groupby("Region", as_index=False).agg(
            Exposure=("Exposure", "sum"),
            Claims=("ClaimNb", "sum"),
            PaidLoss=("PaidLoss", "sum"),
            Policies=("IDpol", "count"),
        )
        tmp["Frequency"] = tmp["Claims"] / tmp["Exposure"]
        tmp["Severity"] = np.where(tmp["Claims"] > 0, tmp["PaidLoss"] / tmp["Claims"], np.nan)
        tmp["PurePremium"] = tmp["PaidLoss"] / tmp["Exposure"]
        tmp.sort_values("PurePremium", ascending=False).to_csv(SEG_CSV, index=False)

    # --- Basic plots (distributions) ---
    save_hist(df["Exposure"], "Exposure distribution", "hist_exposure.png", logy=True)
    save_hist(df["ClaimNb"], "ClaimNb distribution", "hist_claimnb.png", logy=True)
    save_hist(df["PaidLoss"], "PaidLoss distribution", "hist_paidloss.png", logy=True)
    save_hist(df["PurePremium"], "PurePremium distribution", "hist_purepremium.png", logy=True)

    # severity only meaningful where ClaimNb>0 and PaidLoss>0
    sev_df = df[(df["ClaimNb"] > 0) & (df["PaidLoss"] > 0)].copy()
    if len(sev_df) > 0:
        save_hist(sev_df["AvgSeverity"], "AvgSeverity distribution (ClaimNb>0 & PaidLoss>0)",
                  "hist_avgseverity.png", logy=True)

    # --- Write text report ---
    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("03_eda_qa.py report\n")
        f.write("===================\n\n")
        f.write(f"Total rows: {n:,}\n")
        f.write(f"Rows with ClaimNb>0: {n_claim:,}\n")
        f.write(f"Rows with PaidLoss>0: {n_paid:,}\n\n")
        f.write("Key anomaly (ClaimNb>0 but PaidLoss==0):\n")
        f.write(f"  Count: {len(anom):,}\n")
        if anom_missing_sev is not None:
            f.write(f"  Of these, ClaimRecords==0 (no sev records found): {anom_missing_sev:,}\n")
            f.write(f"           ClaimRecords>0  (sev exists but sums to 0): {anom_has_sevrec:,}\n")
        f.write("\nSaved files:\n")
        f.write(f"  {ANOM_CSV}\n")
        f.write(f"  {SUMMARY_CSV}\n")
        f.write(f"  {SEG_CSV}\n")
        f.write(f"  {REPORT_TXT}\n")
        f.write("\nFigures saved in outputs/figures/.\n")

    print("Done.")
    print("Wrote:", REPORT_TXT)

if __name__ == "__main__":
    main()
