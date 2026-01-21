import os
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FREQ_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "freMTPL2freq.csv")
SEV_PATH  = os.path.join(PROJECT_ROOT, "data", "raw", "freMTPL2sev.csv")

OUT_DIR   = os.path.join(PROJECT_ROOT, "data", "processed")
OUT_PATH  = os.path.join(OUT_DIR, "model_table.csv")

LOG_DIR   = os.path.join(PROJECT_ROOT, "logs")
LOG_PATH  = os.path.join(LOG_DIR, "02_build_model_table_report.txt")


def ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)


def to_numeric_safe(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")  # handles scientific notation like 1.2E-3
    return df


def main():
    ensure_dirs()

    # --- Load ---
    freq = pd.read_csv(FREQ_PATH)
    sev  = pd.read_csv(SEV_PATH)

    # --- Basic type coercion (prevents 'E' scientific notation text issues) ---
    freq = to_numeric_safe(freq, ["ClaimNb", "Exposure", "VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"])
    sev  = to_numeric_safe(sev,  ["ClaimAmount"])

    # --- Minimal cleaning / QA ---
    # Keep only rows with essential keys
    freq = freq.dropna(subset=["IDpol", "ClaimNb", "Exposure"])
    sev  = sev.dropna(subset=["IDpol", "ClaimAmount"])

    # Exposure must be > 0 for offset/log usage later
    freq = freq[freq["Exposure"] > 0]

    # ClaimNb should be nonnegative integer
    freq = freq[(freq["ClaimNb"] >= 0) & (freq["ClaimNb"] == np.floor(freq["ClaimNb"]))]

    # Claim amounts must be >= 0
    sev = sev[sev["ClaimAmount"] >= 0]

    # Trim categorical fields (if present)
    for c in ["VehBrand", "VehGas", "Area", "Region"]:
        if c in freq.columns:
            freq[c] = freq[c].astype(str).str.strip()

    # --- Aggregate sev to policy level PaidLoss ---
    sev_pol = (
        sev.groupby("IDpol", as_index=False)
           .agg(PaidLoss=("ClaimAmount", "sum"),
                ClaimRecords=("ClaimAmount", "size"))
    )

    # --- Merge ---
    df = freq.merge(sev_pol, on="IDpol", how="left")
    df["PaidLoss"] = df["PaidLoss"].fillna(0.0)
    df["ClaimRecords"] = df["ClaimRecords"].fillna(0).astype(int)

    # --- Derived fields (still safe at this stage) ---
    df["PurePremium"] = df["PaidLoss"] / df["Exposure"]
    df["HasClaim"] = (df["ClaimNb"] > 0).astype(int)
    df["AvgSeverity"] = np.where(df["ClaimNb"] > 0, df["PaidLoss"] / df["ClaimNb"], np.nan)

    # --- Consistency checks ---
    bad1 = df[(df["ClaimNb"] == 0) & (df["PaidLoss"] > 0)]
    bad2 = df[(df["ClaimNb"] > 0) & (df["PaidLoss"] == 0)]

    # --- Save output ---
    df.to_csv(OUT_PATH, index=False)

    # --- Write report ---
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write("02_build_model_table.py report\n")
        f.write("============================\n\n")
        f.write(f"freq rows after QA: {len(freq):,}\n")
        f.write(f"sev rows after QA:  {len(sev):,}\n")
        f.write(f"merged rows:        {len(df):,}\n\n")
        f.write("Consistency checks:\n")
        f.write(f"  ClaimNb==0 but PaidLoss>0 : {len(bad1):,}\n")
        f.write(f"  ClaimNb>0  but PaidLoss==0 : {len(bad2):,}\n\n")
        f.write("Output files:\n")
        f.write(f"  {OUT_PATH}\n")
        f.write(f"  {LOG_PATH}\n")

    print("Done.")
    print("Wrote:", OUT_PATH)
    print("Wrote:", LOG_PATH)


if __name__ == "__main__":
    main()
