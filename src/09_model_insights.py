import os
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FREQ_COEF = os.path.join(PROJECT_ROOT, "outputs", "tables", "freq_glm_coefficients.csv")
SEV_COEF  = os.path.join(PROJECT_ROOT, "outputs", "tables", "sev_glm_coefficients.csv")

OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "tables")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

FREQ_TOP_OUT = os.path.join(OUT_DIR, "top_drivers_frequency.csv")
SEV_TOP_OUT  = os.path.join(OUT_DIR, "top_drivers_severity.csv")
BULLETS_OUT  = os.path.join(OUT_DIR, "resume_bullets_project1.txt")
REPORT       = os.path.join(LOG_DIR, "09_model_insights_report.txt")


def ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)


def load_coef(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}")
    df = pd.read_csv(path)
    return df


def top_by_deviation(df: pd.DataFrame, multiplier_col: str, n: int = 15) -> pd.DataFrame:
    """
    Rank features by |log(multiplier)|, so 0 means no effect (multiplier=1).
    """
    d = df.copy()
    d[multiplier_col] = pd.to_numeric(d[multiplier_col], errors="coerce")
    d = d.dropna(subset=[multiplier_col])
    d["abs_log_effect"] = np.abs(np.log(d[multiplier_col]))
    d = d.sort_values("abs_log_effect", ascending=False)
    return d.head(n).copy()


def main():
    ensure_dirs()

    freq = load_coef(FREQ_COEF)
    sev  = load_coef(SEV_COEF)

    # Identify multiplier columns
    if "rate_ratio_exp_coef" not in freq.columns:
        raise ValueError("freq_glm_coefficients.csv missing rate_ratio_exp_coef")
    if "multiplier_exp_coef" not in sev.columns:
        raise ValueError("sev_glm_coefficients.csv missing multiplier_exp_coef")

    # Take top drivers excluding intercept/const
    freq2 = freq[freq["feature"].astype(str).str.lower() != "const"].copy()
    sev2  = sev[sev["feature"].astype(str).str.lower() != "const"].copy()

    top_freq = top_by_deviation(freq2, "rate_ratio_exp_coef", n=20)
    top_sev  = top_by_deviation(sev2, "multiplier_exp_coef", n=20)

    # Save tables
    top_freq.to_csv(FREQ_TOP_OUT, index=False)
    top_sev.to_csv(SEV_TOP_OUT, index=False)

    # Build resume bullets (generic but strong; you can edit names later)
    # We will cite: dataset size, modeling approach, winsorization, outputs.
    # Use your known counts from prior steps:
    n_policies = 678_013
    n_sev = 24_944

    # Pull headline improvements from reports if you want later; for now use placeholders from what you pasted
    # (You can manually edit the bullet file after generation.)
    freq_dev_improve = 1525.33
    sev_dev_improve = 20.68
    sev_cap_max = 34564.65  # from winsor check max after cap

    bullets = []
    bullets.append(
        f"Built an end-to-end auto insurance pricing prototype using {n_policies:,} policies (frequency) and {n_sev:,} claim-bearing policies (severity); engineered a policy-level modeling table by aggregating claim transactions and validating exposure/claim consistency."
    )
    bullets.append(
        f"Trained a two-part GLM framework (Poisson frequency with log(Exposure) offset + Gamma severity with log link) and generated policy- and segment-level experience studies (Region×Area, Fuel, Brand) for dashboard reporting."
    )
    bullets.append(
        f"Implemented data quality controls and tail stabilization (winsorized severity to reduce extreme-loss influence; capped AvgSeverity with post-cap max ≈ {sev_cap_max:,.2f}); documented model limitations due to incomplete severity coverage."
    )
    bullets.append(
        f"Quantified predictive lift versus portfolio baselines (Poisson deviance improvement ≈ {freq_dev_improve:,.2f}; Gamma deviance improvement ≈ {sev_dev_improve:,.2f}) and exported ATS-friendly artifacts (coefficients, rate ratios, segment KPIs) for reproducible analysis."
    )

    with open(BULLETS_OUT, "w", encoding="utf-8") as f:
        for b in bullets:
            f.write("- " + b + "\n")

    # Report
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("09_model_insights.py report\n")
        f.write("===========================\n\n")
        f.write(f"Inputs:\n  {FREQ_COEF}\n  {SEV_COEF}\n\n")
        f.write("Outputs:\n")
        f.write(f"  {FREQ_TOP_OUT}\n")
        f.write(f"  {SEV_TOP_OUT}\n")
        f.write(f"  {BULLETS_OUT}\n")
        f.write(f"  {REPORT}\n\n")
        f.write("Notes:\n")
        f.write("  - Top drivers ranked by |log(multiplier)| (distance from 1.0).\n")
        f.write("  - Resume bullets are a draft; refine wording once dashboard is finalized.\n")

    print("Done.")
    print("Wrote:", FREQ_TOP_OUT)
    print("Wrote:", SEV_TOP_OUT)
    print("Wrote:", BULLETS_OUT)
    print("Wrote:", REPORT)


if __name__ == "__main__":
    main()
