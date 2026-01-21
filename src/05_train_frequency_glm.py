import os
import numpy as np
import pandas as pd
import statsmodels.api as sm


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IN_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "model_freq.csv")

OUT_TBL = os.path.join(PROJECT_ROOT, "outputs", "tables")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

COEF_OUT = os.path.join(OUT_TBL, "freq_glm_coefficients.csv")
PRED_ALL_OUT = os.path.join(OUT_TBL, "freq_predictions_all.csv")
PRED_TEST_OUT = os.path.join(OUT_TBL, "freq_predictions_test.csv")
REPORT = os.path.join(LOG_DIR, "05_train_frequency_glm_report.txt")


# We'll use LogDensity instead of raw Density when possible
RAW_NUM_COLS = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"]
CAT_COLS = ["VehBrand", "VehGas", "Area", "Region"]


CLIP_Q_LOW = 0.005
CLIP_Q_HIGH = 0.995
RARE_MIN_COUNT = 500  # merge rare categorical levels to "Other" to avoid extreme coefficients


def ensure_dirs():
    os.makedirs(OUT_TBL, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)


def to_num(df: pd.DataFrame, cols: list[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def merge_rare_levels(df: pd.DataFrame, col: str, min_count: int) -> None:
    if col not in df.columns:
        return
    vc = df[col].astype(str).str.strip().value_counts(dropna=False)
    keep = set(vc[vc >= min_count].index.astype(str))
    s = df[col].astype(str).str.strip()
    df[col] = np.where(s.isin(keep), s, "Other")


def clean_freq(df: pd.DataFrame) -> pd.DataFrame:
    # required columns
    need = ["ClaimNb", "Exposure"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"Missing required column in model_freq.csv: {c}")

    # numeric coercion
    to_num(df, ["ClaimNb", "Exposure"] + RAW_NUM_COLS)

    # trim categoricals
    for c in CAT_COLS:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    # basic QA
    df = df.dropna(subset=["ClaimNb", "Exposure"])
    df = df[df["Exposure"] > 0]
    df = df[(df["ClaimNb"] >= 0) & (df["ClaimNb"] == np.floor(df["ClaimNb"]))]

    # merge rare categorical levels (stabilize tail)
    for c in CAT_COLS:
        if c in df.columns:
            merge_rare_levels(df, c, RARE_MIN_COUNT)

    return df


def split_train_test(df: pd.DataFrame, test_size=0.2, seed=42):
    rng = np.random.default_rng(seed)
    idx = df.index.to_numpy()
    rng.shuffle(idx)
    cut = int(len(idx) * (1 - test_size))
    train_idx = idx[:cut]
    test_idx = idx[cut:]
    return df.loc[train_idx].copy(), df.loc[test_idx].copy()


def compute_clip_bounds(train: pd.DataFrame, cols: list[str], q_low: float, q_high: float) -> dict:
    bounds = {}
    for c in cols:
        if c in train.columns:
            s = pd.to_numeric(train[c], errors="coerce").dropna()
            if len(s) == 0:
                continue
            lo = float(s.quantile(q_low))
            hi = float(s.quantile(q_high))
            if lo > hi:
                lo, hi = hi, lo
            bounds[c] = (lo, hi)
    return bounds


def apply_clip(df: pd.DataFrame, bounds: dict) -> None:
    for c, (lo, hi) in bounds.items():
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").clip(lo, hi)


def add_log_density(df: pd.DataFrame) -> None:
    if "Density" in df.columns:
        d = pd.to_numeric(df["Density"], errors="coerce")
        df["LogDensity"] = np.log1p(d.clip(lower=0))
    # If Density not present, LogDensity won't be used


def make_design(df: pd.DataFrame, ref_columns: list[str] | None = None):
    used_idx = df.index

    y = pd.to_numeric(df["ClaimNb"], errors="coerce").astype(float)
    exposure = pd.to_numeric(df["Exposure"], errors="coerce").astype(float)
    offset = np.log(exposure)

    # numeric features: use LogDensity if available, otherwise raw Density
    num_cols = ["VehPower", "VehAge", "DrivAge", "BonusMalus"]
    if "LogDensity" in df.columns:
        num_cols.append("LogDensity")
    elif "Density" in df.columns:
        num_cols.append("Density")

    X_num = df[[c for c in num_cols if c in df.columns]].copy()
    for c in X_num.columns:
        X_num[c] = pd.to_numeric(X_num[c], errors="coerce")

    # categoricals
    cat_present = [c for c in CAT_COLS if c in df.columns]
    if len(cat_present) > 0:
        X_cat = pd.get_dummies(df[cat_present].astype(str), drop_first=True)
    else:
        X_cat = pd.DataFrame(index=df.index)

    X = pd.concat([X_num, X_cat], axis=1)

    # drop rows with NA in y/offset/X
    mask = (~y.isna()) & (~offset.isna()) & (~X.isna().any(axis=1))
    y = y[mask]
    offset = offset[mask]
    X = X.loc[mask].copy()
    used_idx = used_idx[mask.values]

    # add intercept
    X = sm.add_constant(X, has_constant="add")

    # force numeric float
    X = X.apply(pd.to_numeric, errors="coerce")
    mask2 = ~X.isna().any(axis=1)
    X = X.loc[mask2].copy()
    y = y.loc[mask2].copy()
    offset = offset.loc[mask2].copy()
    used_idx = used_idx[mask2.values]

    X = X.astype(float)

    if ref_columns is not None:
        X = X.reindex(columns=ref_columns, fill_value=0.0)

    return y, X, offset, used_idx


def poisson_deviance(y_true: np.ndarray, mu: np.ndarray) -> float:
    mu = np.clip(mu, 1e-12, None)
    y = y_true
    term = np.where(y > 0, y * np.log(y / mu), 0.0)
    return float(2.0 * np.sum(term - (y - mu)))


def main():
    ensure_dirs()

    df = pd.read_csv(IN_PATH)
    df = clean_freq(df)

    train, test = split_train_test(df, test_size=0.2, seed=42)

    # clip numeric cols using TRAIN quantiles, then apply to all
    clip_cols = [c for c in RAW_NUM_COLS if c in df.columns]  # includes Density
    bounds = compute_clip_bounds(train, clip_cols, CLIP_Q_LOW, CLIP_Q_HIGH)
    apply_clip(train, bounds)
    apply_clip(test, bounds)
    apply_clip(df, bounds)

    # add LogDensity after clipping
    add_log_density(train)
    add_log_density(test)
    add_log_density(df)

    # design matrices
    y_tr, X_tr, off_tr, idx_tr = make_design(train)
    y_te, X_te, off_te, idx_te = make_design(test, ref_columns=list(X_tr.columns))

    # fit
    model = sm.GLM(y_tr, X_tr, family=sm.families.Poisson(), offset=off_tr)
    res = model.fit(maxiter=200)

    # baseline (train overall rate)
    base_freq = float(train["ClaimNb"].sum() / train["Exposure"].sum())
    base_mu_te = base_freq * test.loc[idx_te, "Exposure"].values

    # predict test
    mu_te = res.predict(X_te, offset=off_te)
    dev_model = poisson_deviance(y_te.values, mu_te)
    dev_base = poisson_deviance(y_te.values, base_mu_te)

    # claim-rate MAE
    exp_te = test.loc[idx_te, "Exposure"].values
    mae_rate = float(np.mean(np.abs((test.loc[idx_te, "ClaimNb"].values / exp_te) - (mu_te / exp_te))))

    # coefficients
    params = res.params
    coef_tbl = pd.DataFrame({
        "feature": params.index,
        "coef": params.values,
        "std_err": res.bse.values,
        "rate_ratio_exp_coef": np.exp(params.values),
    }).sort_values("rate_ratio_exp_coef", ascending=False)
    coef_tbl.to_csv(COEF_OUT, index=False)

    # predictions ALL
    y_all, X_all, off_all, idx_all = make_design(df, ref_columns=list(X_tr.columns))
    mu_all = res.predict(X_all, offset=off_all)

    out_all_cols = ["IDpol", "Exposure", "ClaimNb"] if "IDpol" in df.columns else ["Exposure", "ClaimNb"]
    out_all = df.loc[idx_all, out_all_cols].copy()
    out_all["pred_claim_count"] = mu_all
    out_all["pred_frequency"] = mu_all / out_all["Exposure"].values
    out_all.to_csv(PRED_ALL_OUT, index=False)

    # predictions TEST
    out_test_cols = ["IDpol", "Exposure", "ClaimNb"] if "IDpol" in df.columns else ["Exposure", "ClaimNb"]
    out_test = df.loc[idx_te, out_test_cols].copy()
    out_test["pred_claim_count"] = mu_te
    out_test["pred_frequency"] = mu_te / out_test["Exposure"].values
    out_test.to_csv(PRED_TEST_OUT, index=False)

    # report
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("05_train_frequency_glm.py report\n")
        f.write("================================\n\n")
        f.write(f"Input: {IN_PATH}\n")
        f.write(f"Rows after cleaning: {len(df):,}\n")
        f.write(f"Train rows used: {len(y_tr):,}\n")
        f.write(f"Test rows used:  {len(y_te):,}\n\n")
        f.write("Model: Poisson GLM with offset log(Exposure)\n\n")
        f.write(f"Baseline frequency (train): {base_freq:.6f}\n\n")
        f.write("Numeric stabilization:\n")
        f.write(f"  Clip quantiles: [{CLIP_Q_LOW:.3f}, {CLIP_Q_HIGH:.3f}] on TRAIN, applied to all\n")
        for k, (lo, hi) in bounds.items():
            f.write(f"  {k}: [{lo:.6f}, {hi:.6f}]\n")
        f.write("  Density used as LogDensity=log1p(Density) when available\n\n")
        f.write("Test metrics:\n")
        f.write(f"  Poisson deviance (model):    {dev_model:,.2f}\n")
        f.write(f"  Poisson deviance (baseline): {dev_base:,.2f}\n")
        f.write(f"  Deviance improvement:        {dev_base - dev_model:,.2f}\n")
        f.write(f"  MAE of claim rate:           {mae_rate:.6f}\n\n")
        f.write("Outputs:\n")
        f.write(f"  Coefficients: {COEF_OUT}\n")
        f.write(f"  Predictions (ALL): {PRED_ALL_OUT}\n")
        f.write(f"  Predictions (TEST): {PRED_TEST_OUT}\n")
        f.write(f"  Report: {REPORT}\n")

    print("Done.")
    print("Wrote:", COEF_OUT)
    print("Wrote:", PRED_ALL_OUT)
    print("Wrote:", PRED_TEST_OUT)
    print("Wrote:", REPORT)


if __name__ == "__main__":
    main()
