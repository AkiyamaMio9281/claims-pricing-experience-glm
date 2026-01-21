import os
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IN_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "model_table.csv")
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

FREQ_OUT = os.path.join(OUT_DIR, "model_freq.csv")
SEV_OUT  = os.path.join(OUT_DIR, "model_sev.csv")
CAPS_OUT = os.path.join(OUT_DIR, "winsor_caps.txt")
REPORT   = os.path.join(LOG_DIR, "04_make_train_sets_report.txt")


# ---- Optional: winsorize severity and/or pure premium to reduce extreme tail ----
# You can keep these OFF initially. We'll turn them ON only if your p99/max are crazy.
WINSORIZE = True
WINSOR_PCT = 0.995   # cap at 99.5 percentile


def ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)


def winsorize_series(s: pd.Series, pct: float) -> tuple[pd.Series, float]:
    cap = float(s.quantile(pct))
    return s.clip(upper=cap), cap


def main():
    ensure_dirs()
    df = pd.read_csv(IN_PATH)

    # ----- Frequency dataset (Poisson GLM) -----
    # Keep columns needed for modeling
    freq_cols = [
        "IDpol", "ClaimNb", "Exposure",
        "VehPower", "VehAge", "DrivAge", "BonusMalus",
        "VehBrand", "VehGas", "Area", "Density", "Region"
    ]
    freq_cols = [c for c in freq_cols if c in df.columns]
    model_freq = df[freq_cols].copy()

    # sanity (should already hold)
    model_freq = model_freq.dropna(subset=["ClaimNb", "Exposure"])
    model_freq = model_freq[model_freq["Exposure"] > 0]

    # ----- Severity dataset (Gamma GLM) -----
    # Only where we have positive claim count AND observed positive loss
    sev_cols = freq_cols + ["PaidLoss"]
    sev_cols = [c for c in sev_cols if c in df.columns]
    model_sev = df[sev_cols].copy()
    model_sev = model_sev[(model_sev["ClaimNb"] > 0) & (model_sev["PaidLoss"] > 0)].copy()
    model_sev["AvgSeverity"] = model_sev["PaidLoss"] / model_sev["ClaimNb"]

    sev_cap = None
    pp_cap = None

    if WINSORIZE:
        # cap AvgSeverity (long-tail)
        model_sev["AvgSeverity"], sev_cap = winsorize_series(model_sev["AvgSeverity"], WINSOR_PCT)

        # cap PaidLoss consistently if you want (optional)
        model_sev["PaidLoss"] = model_sev["AvgSeverity"] * model_sev["ClaimNb"]

        with open(CAPS_OUT, "w", encoding="utf-8") as f:
            f.write(f"AvgSeverity capped at {WINSOR_PCT*100:.2f}th percentile: {sev_cap}\n")
    # Save outputs
    model_freq.to_csv(FREQ_OUT, index=False)
    model_sev.to_csv(SEV_OUT, index=False)

    # Report
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("04_make_train_sets.py report\n")
        f.write("============================\n\n")
        f.write(f"model_freq rows: {len(model_freq):,}\n")
        f.write(f"model_sev rows:  {len(model_sev):,}\n\n")
        f.write("Severity selection rule:\n")
        f.write("  ClaimNb > 0 and PaidLoss > 0 (sev data incomplete for some ClaimNb>0 policies)\n\n")
        f.write(f"WINSORIZE: {WINSORIZE}\n")
        if WINSORIZE:
            f.write(f"WINSORT: {WINSOR_PCT}\n")
            f.write(f"AvgSeverity cap: {sev_cap}\n")
        f.write("\nSaved files:\n")
        f.write(f"  {FREQ_OUT}\n")
        f.write(f"  {SEV_OUT}\n")
        f.write(f"  {REPORT}\n")

    print("Done.")
    print("Wrote:", FREQ_OUT)
    print("Wrote:", SEV_OUT)
    print("Wrote:", REPORT)


if __name__ == "__main__":
    main()
