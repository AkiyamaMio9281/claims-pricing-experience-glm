import os
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IN_PATH = os.path.join(PROJECT_ROOT, "outputs", "tables", "dashboard_policy_level.csv")

OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "tables")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

REPORT = os.path.join(LOG_DIR, "08_experience_study_report.txt")


def ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)


def numeric(df: pd.DataFrame, cols: list[str]):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def build_segment_table(df: pd.DataFrame, group_cols: list[str], out_name: str) -> tuple[str, int]:
    """
    Creates an experience study table with:
      - Totals: Exposure, Claims, PaidLoss
      - Portfolio metrics (totals-based): Frequency, Severity, PurePremium
      - Predicted portfolio metrics based on predicted totals
      - Mean metrics (simple average across policies) as extra reference

    Saves to outputs/tables/<out_name>.csv
    Returns: (path, rowcount)
    """
    seg = df.copy()

    # predicted totals per policy
    seg["pred_claims_total"] = seg["pred_frequency"] * seg["Exposure"]
    seg["pred_loss_total"] = seg["pred_pure_premium"] * seg["Exposure"]

    gb = seg.groupby(group_cols, as_index=False)

    out = gb.agg(
        Policies=("IDpol", "count"),
        Exposure=("Exposure", "sum"),
        Claims=("ClaimNb", "sum"),
        PaidLoss=("PaidLoss", "sum"),
        PredClaims=("pred_claims_total", "sum"),
        PredLoss=("pred_loss_total", "sum"),
        # simple means (unweighted)
        Mean_ActualFreq=("actual_frequency", "mean"),
        Mean_ActualPP=("actual_pure_premium", "mean"),
        Mean_PredFreq=("pred_frequency", "mean"),
        Mean_PredSev=("pred_avg_severity", "mean"),
        Mean_PredPP=("pred_pure_premium", "mean"),
    )

    # portfolio metrics (totals / totals)
    out["Portfolio_ActualFreq"] = out["Claims"] / out["Exposure"]
    out["Portfolio_ActualPP"] = out["PaidLoss"] / out["Exposure"]
    out["Portfolio_PredFreq"] = out["PredClaims"] / out["Exposure"]
    out["Portfolio_PredPP"] = out["PredLoss"] / out["Exposure"]

    out["Portfolio_ActualSev"] = np.where(out["Claims"] > 0, out["PaidLoss"] / out["Claims"], np.nan)
    out["Portfolio_PredSev"] = np.where(out["PredClaims"] > 0, out["PredLoss"] / out["PredClaims"], np.nan)

    # Add credibility-ish flags (very simple): exposure threshold
    # (This is NOT actuarial credibility theory, just a stability label.)
    out["StabilityFlag"] = np.where(out["Exposure"] < 100.0, "LowExposure", "OK")

    # Sort by predicted portfolio pure premium desc (useful for dashboard)
    out = out.sort_values("Portfolio_PredPP", ascending=False)

    out_path = os.path.join(OUT_DIR, f"{out_name}.csv")
    out.to_csv(out_path, index=False)
    return out_path, len(out)


def main():
    ensure_dirs()

    if not os.path.exists(IN_PATH):
        raise FileNotFoundError(
            f"Missing input:\n  {IN_PATH}\n\n"
            f"Fix: Run scripts 05, 06, 07 first to generate dashboard_policy_level.csv."
        )

    df = pd.read_csv(IN_PATH)

    # Minimal required columns
    required = ["IDpol", "Exposure", "ClaimNb", "PaidLoss", "pred_frequency", "pred_avg_severity", "pred_pure_premium",
                "actual_frequency", "actual_pure_premium"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"dashboard_policy_level.csv missing columns: {missing}")

    # numeric coercion
    df = numeric(df, ["Exposure", "ClaimNb", "PaidLoss",
                      "pred_frequency", "pred_avg_severity", "pred_pure_premium",
                      "actual_frequency", "actual_pure_premium"])

    # basic cleanup
    df = df.dropna(subset=["IDpol", "Exposure", "ClaimNb", "PaidLoss", "pred_frequency", "pred_avg_severity", "pred_pure_premium"])
    df = df[df["Exposure"] > 0]

    # Trim categoricals if present
    for c in ["Region", "Area", "VehBrand", "VehGas"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    outputs = []

    # Segment tables we generate (only if cols exist)
    if "Region" in df.columns:
        outputs.append(build_segment_table(df, ["Region"], "experience_by_region"))

    if "Area" in df.columns:
        outputs.append(build_segment_table(df, ["Area"], "experience_by_area"))

    if "VehGas" in df.columns:
        outputs.append(build_segment_table(df, ["VehGas"], "experience_by_vehgas"))

    if "VehBrand" in df.columns:
        # Optionally compress rare brands first (to avoid huge table). We'll do it lightly here:
        # keep top 15 brands, others -> Other
        vc = df["VehBrand"].value_counts()
        top = set(vc.head(15).index)
        tmp = df.copy()
        tmp.loc[~tmp["VehBrand"].isin(top), "VehBrand"] = "Other"
        outputs.append(build_segment_table(tmp, ["VehBrand"], "experience_by_vehbrand_top15_plus_other"))

    if ("Region" in df.columns) and ("Area" in df.columns):
        outputs.append(build_segment_table(df, ["Region", "Area"], "experience_by_region_area"))

    # Portfolio summary (single-row CSV)
    portfolio = pd.DataFrame([{
        "Policies": int(df["IDpol"].nunique()),
        "Exposure": float(df["Exposure"].sum()),
        "Claims": float(df["ClaimNb"].sum()),
        "PaidLoss": float(df["PaidLoss"].sum()),
        "Portfolio_ActualFreq": float(df["ClaimNb"].sum() / df["Exposure"].sum()),
        "Portfolio_ActualSev": float(df["PaidLoss"].sum() / df["ClaimNb"].sum()) if df["ClaimNb"].sum() > 0 else np.nan,
        "Portfolio_ActualPP": float(df["PaidLoss"].sum() / df["Exposure"].sum()),
        "Portfolio_PredFreq": float((df["pred_frequency"] * df["Exposure"]).sum() / df["Exposure"].sum()),
        "Portfolio_PredSev": float((df["pred_pure_premium"] * df["Exposure"]).sum() / (df["pred_frequency"] * df["Exposure"]).sum())
                             if (df["pred_frequency"] * df["Exposure"]).sum() > 0 else np.nan,
        "Portfolio_PredPP": float((df["pred_pure_premium"] * df["Exposure"]).sum() / df["Exposure"].sum()),
    }])
    portfolio_path = os.path.join(OUT_DIR, "portfolio_summary.csv")
    portfolio.to_csv(portfolio_path, index=False)

    # Report
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("08_experience_study.py report\n")
        f.write("============================\n\n")
        f.write(f"Input: {IN_PATH}\n")
        f.write(f"Rows used: {len(df):,}\n")
        f.write(f"Unique policies: {df['IDpol'].nunique():,}\n\n")
        f.write("Generated tables:\n")
        for path, rows in outputs:
            f.write(f"  {path}  (rows={rows:,})\n")
        f.write(f"  {portfolio_path}  (rows=1)\n\n")
        f.write("Notes:\n")
        f.write("  - Portfolio_* metrics are totals-based (Claims/Exposure, PaidLoss/Claims, PaidLoss/Exposure).\n")
        f.write("  - Pred* portfolio metrics are based on predicted totals summed within each segment.\n")
        f.write("  - StabilityFlag is a simple low-exposure label, not formal credibility.\n")

    print("Done.")
    print("Wrote:", REPORT)
    print("Wrote:", portfolio_path)
    for path, _ in outputs:
        print("Wrote:", path)


if __name__ == "__main__":
    main()
