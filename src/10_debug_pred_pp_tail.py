import os
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IN_PATH = os.path.join(PROJECT_ROOT, "outputs", "tables", "dashboard_policy_level.csv")

OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "tables")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

OUT_TOP = os.path.join(OUT_DIR, "pred_pp_top200_policies.csv")
REPORT = os.path.join(LOG_DIR, "10_debug_pred_pp_tail_report.txt")

TOL = 1e-3  # more realistic tolerance for float computations


def ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)


def to_num(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def quantiles(s, qs=(0.0, 0.5, 0.9, 0.95, 0.99, 0.999, 1.0)):
    s = s.dropna()
    if len(s) == 0:
        return {q: np.nan for q in qs}
    return {q: float(s.quantile(q)) for q in qs}


def main():
    ensure_dirs()
    df = pd.read_csv(IN_PATH)

    num_cols = [
        "Exposure", "ClaimNb", "PaidLoss",
        "pred_frequency", "pred_avg_severity", "pred_pure_premium",
        "actual_frequency", "actual_avg_severity", "actual_pure_premium",
        "pred_loss_total", "pred_claims_total"
    ]
    df = to_num(df, num_cols)

    df["pred_pp_check"] = df["pred_frequency"] * df["pred_avg_severity"]
    df["pred_pp_absdiff"] = (df["pred_pure_premium"] - df["pred_pp_check"]).abs()

    q_freq = quantiles(df["pred_frequency"])
    q_sev  = quantiles(df["pred_avg_severity"])
    q_pp   = quantiles(df["pred_pure_premium"])

    max_absdiff = float(df["pred_pp_absdiff"].max())
    bad_formula = df[df["pred_pp_absdiff"] > TOL]

    keep = [
        "IDpol", "Region", "Area", "VehGas", "VehBrand",
        "Exposure", "ClaimNb", "PaidLoss",
        "pred_frequency", "pred_avg_severity", "pred_pure_premium",
        "pred_claims_total", "pred_loss_total",
        "pred_pp_check", "pred_pp_absdiff"
    ]
    keep = [c for c in keep if c in df.columns]

    top = df.sort_values("pred_pure_premium", ascending=False).head(200)[keep].copy()
    top.to_csv(OUT_TOP, index=False)

    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("10_debug_pred_pp_tail.py report\n")
        f.write("================================\n\n")
        f.write(f"Input: {IN_PATH}\n")
        f.write(f"Rows: {len(df):,}\n\n")

        f.write("Quantiles (pred_frequency):\n")
        for k, v in q_freq.items():
            f.write(f"  q{int(k*1000)/10:>5}: {v:.6f}\n")
        f.write("\nQuantiles (pred_avg_severity):\n")
        for k, v in q_sev.items():
            f.write(f"  q{int(k*1000)/10:>5}: {v:.6f}\n")
        f.write("\nQuantiles (pred_pure_premium):\n")
        for k, v in q_pp.items():
            f.write(f"  q{int(k*1000)/10:>5}: {v:.6f}\n")

        f.write("\nFormula check:\n")
        f.write(f"  Tolerance: {TOL}\n")
        f.write(f"  Max abs diff: {max_absdiff:.12f}\n")
        f.write(f"  Rows with abs diff > TOL: {len(bad_formula):,}\n\n")

        f.write("Outputs:\n")
        f.write(f"  Top 200 policies: {OUT_TOP}\n")
        f.write(f"  Report: {REPORT}\n")

    print("Done.")
    print("Wrote:", OUT_TOP)
    print("Wrote:", REPORT)


if __name__ == "__main__":
    main()
