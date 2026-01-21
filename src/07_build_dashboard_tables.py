import os
import numpy as np
import pandas as pd


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Prefer model_table.csv if present (has PaidLoss), else fall back to model_freq.csv
MODEL_TABLE = os.path.join(PROJECT_ROOT, "data", "processed", "model_table.csv")
MODEL_FREQ  = os.path.join(PROJECT_ROOT, "data", "processed", "model_freq.csv")

FREQ_PRED_ALL = os.path.join(PROJECT_ROOT, "outputs", "tables", "freq_predictions_all.csv")
SEV_PRED_ALL  = os.path.join(PROJECT_ROOT, "outputs", "tables", "sev_predictions_all.csv")

OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "tables")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

DASH_POLICY = os.path.join(OUT_DIR, "dashboard_policy_level.csv")
REPORT = os.path.join(LOG_DIR, "07_build_dashboard_tables_report.txt")


def ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)


def require_file(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required file:\n  {path}")


def to_num(df: pd.DataFrame, cols: list[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def load_base_table() -> pd.DataFrame:
    if os.path.exists(MODEL_TABLE):
        df = pd.read_csv(MODEL_TABLE)
        source = MODEL_TABLE
    else:
        df = pd.read_csv(MODEL_FREQ)
        source = MODEL_FREQ
        # PaidLoss may not exist in model_freq; if missing, set 0 so dashboard still works
        if "PaidLoss" not in df.columns:
            df["PaidLoss"] = 0.0

    # minimal required
    need = ["IDpol", "Exposure", "ClaimNb", "PaidLoss"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"Base table missing column '{c}' from {source}")

    # numeric
    to_num(df, ["Exposure", "ClaimNb", "PaidLoss"])
    df["Exposure"] = df["Exposure"].clip(lower=0)
    df["ClaimNb"] = df["ClaimNb"].fillna(0)
    df["PaidLoss"] = df["PaidLoss"].fillna(0)

    # trim categoricals if exist
    for c in ["Region", "Area", "VehBrand", "VehGas"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    return df


def main():
    ensure_dirs()

    # Required prediction files
    require_file(FREQ_PRED_ALL)
    require_file(SEV_PRED_ALL)
    require_file(MODEL_FREQ)  # used as fallback if model_table absent

    base = load_base_table()

    freq = pd.read_csv(FREQ_PRED_ALL)
    sev = pd.read_csv(SEV_PRED_ALL)

    # freq required
    for c in ["IDpol", "pred_frequency"]:
        if c not in freq.columns:
            raise ValueError(f"freq_predictions_all.csv missing '{c}'")
    to_num(freq, ["pred_frequency", "Exposure", "ClaimNb", "pred_claim_count"])

    # sev required
    for c in ["IDpol", "pred_avg_severity"]:
        if c not in sev.columns:
            raise ValueError(f"sev_predictions_all.csv missing '{c}'")
    to_num(sev, ["pred_avg_severity"])

    # merge predictions onto base
    df = base.merge(freq[["IDpol", "pred_frequency"]], on="IDpol", how="left")

    # fill missing pred_frequency (should be rare)
    if df["pred_frequency"].isna().any():
        df["pred_frequency"] = df["pred_frequency"].fillna(df["pred_frequency"].median())

    # merge severity predictions (sev coverage incomplete)
    df = df.merge(sev[["IDpol", "pred_avg_severity"]], on="IDpol", how="left")
    sev_fill = float(sev["pred_avg_severity"].median())
    df["pred_avg_severity"] = df["pred_avg_severity"].fillna(sev_fill)

    # actual metrics
    df["actual_frequency"] = df["ClaimNb"] / df["Exposure"].replace(0, np.nan)
    df["actual_pure_premium"] = df["PaidLoss"] / df["Exposure"].replace(0, np.nan)
    df["actual_avg_severity"] = np.where(df["ClaimNb"] > 0, df["PaidLoss"] / df["ClaimNb"], np.nan)

    # predicted metrics
    df["pred_pure_premium"] = df["pred_frequency"] * df["pred_avg_severity"]

    # totals per policy (KEY FIX for drilldown)
    df["pred_claims_total"] = df["pred_frequency"] * df["Exposure"]
    df["pred_loss_total"] = df["pred_pure_premium"] * df["Exposure"]
    df["actual_claims_total"] = df["ClaimNb"]
    df["actual_loss_total"] = df["PaidLoss"]

    # basic sanity
    df["pred_frequency"] = df["pred_frequency"].clip(lower=0)
    df["pred_avg_severity"] = df["pred_avg_severity"].clip(lower=0)
    df["pred_pure_premium"] = df["pred_pure_premium"].clip(lower=0)

    # output columns (keep core + key categorical fields if available)
    cols = [
        "IDpol",
        "Region", "Area", "VehGas", "VehBrand",
        "Exposure", "ClaimNb", "PaidLoss",
        "actual_frequency", "actual_avg_severity", "actual_pure_premium",
        "pred_frequency", "pred_avg_severity", "pred_pure_premium",
        "pred_claims_total", "pred_loss_total",
        "actual_claims_total", "actual_loss_total",
    ]
    cols = [c for c in cols if c in df.columns]
    out = df[cols].copy()

    out.to_csv(DASH_POLICY, index=False)

    # report
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("07_build_dashboard_tables.py report\n")
        f.write("===================================\n\n")
        f.write("Inputs:\n")
        f.write(f"  Base: {MODEL_TABLE if os.path.exists(MODEL_TABLE) else MODEL_FREQ}\n")
        f.write(f"  Freq predictions: {FREQ_PRED_ALL}\n")
        f.write(f"  Sev predictions:  {SEV_PRED_ALL}\n\n")
        f.write(f"Rows: {len(out):,}\n")
        f.write(f"Unique policies: {out['IDpol'].nunique():,}\n\n")

        miss_sev = int(df["pred_avg_severity"].isna().sum())
        f.write("Notes:\n")
        f.write(f"  - Sev coverage is incomplete; missing pred_avg_severity filled with median={sev_fill:.6f}\n")
        f.write("  - Added totals for drilldown:\n")
        f.write("      pred_claims_total = pred_frequency * Exposure\n")
        f.write("      pred_loss_total   = pred_pure_premium * Exposure\n\n")

        f.write("Outputs:\n")
        f.write(f"  Dashboard policy-level: {DASH_POLICY}\n")
        f.write(f"  Report: {REPORT}\n")

    print("Done.")
    print("Wrote:", DASH_POLICY)
    print("Wrote:", REPORT)


if __name__ == "__main__":
    main()
