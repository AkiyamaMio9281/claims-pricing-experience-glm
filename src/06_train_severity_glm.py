import os
import numpy as np
import pandas as pd
import statsmodels.api as sm


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IN_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "model_sev.csv")

OUT_TBL = os.path.join(PROJECT_ROOT, "outputs", "tables")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

COEF_OUT = os.path.join(OUT_TBL, "sev_glm_coefficients.csv")
PRED_ALL_OUT = os.path.join(OUT_TBL, "sev_predictions_all.csv")
PRED_TEST_OUT = os.path.join(OUT_TBL, "sev_predictions_test.csv")
REPORT = os.path.join(LOG_DIR, "06_train_severity_glm_report.txt")


NUM_COLS = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"]
CAT_COLS = ["VehBrand", "VehGas", "Area", "Region"]


def ensure_dirs():
    os.makedirs(OUT_TBL, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)


def to_num(df: pd.DataFrame, cols: list[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def clean_sev(df: pd.DataFrame) -> pd.DataFrame:
    # required columns
    need = ["AvgSeverity"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"Missing required column in model_sev.csv: {c}")

    to_num(df, ["AvgSeverity"] + NUM_COLS + ["ClaimNb", "PaidLoss"])

    for c in CAT_COLS:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    # only positive severity
    df = df.dropna(subset=["AvgSeverity"])
    df = df[df["AvgSeverity"] > 0]

    return df


def make_design(df: pd.DataFrame, ref_columns: list[str] | None = None):
    used_idx = df.index

    y = pd.to_numeric(df["AvgSeverity"], errors="coerce").astype(float)

    num_present = [c for c in NUM_COLS if c in df.columns]
    X_num = df[num_present].copy()
    for c in num_present:
        X_num[c] = pd.to_numeric(X_num[c], errors="coerce")

    cat_present = [c for c in CAT_COLS if c in df.columns]
    if len(cat_present) > 0:
        X_cat = pd.get_dummies(df[cat_present].astype(str), drop_first=True)
    else:
        X_cat = pd.DataFrame(index=df.index)

    X = pd.concat([X_num, X_cat], axis=1)

    mask = (~y.isna()) & (~X.isna().any(axis=1))
    y = y[mask]
    X = X.loc[mask].copy()
    used_idx = used_idx[mask.values]

    X = sm.add_constant(X, has_constant="add")
    X = X.apply(pd.to_numeric, errors="coerce")
    mask2 = ~X.isna().any(axis=1)
    X = X.loc[mask2].copy()
    y = y.loc[mask2].copy()
    used_idx = used_idx[mask2.values]

    X = X.astype(float)

    if ref_columns is not None:
        X = X.reindex(columns=ref_columns, fill_value=0.0)

    return y, X, used_idx


def split_train_test(df: pd.DataFrame, test_size=0.2, seed=42):
    rng = np.random.default_rng(seed)
    idx = df.index.to_numpy()
    rng.shuffle(idx)
    cut = int(len(idx) * (1 - test_size))
    train_idx = idx[:cut]
    test_idx = idx[cut:]
    return df.loc[train_idx].copy(), df.loc[test_idx].copy()


def gamma_deviance(y_true: np.ndarray, mu: np.ndarray) -> float:
    mu = np.clip(mu, 1e-12, None)
    y = np.clip(y_true, 1e-12, None)
    return float(2.0 * np.sum((y - mu) / mu - np.log(y / mu)))


def main():
    ensure_dirs()

    df = pd.read_csv(IN_PATH)
    df = clean_sev(df)

    train, test = split_train_test(df, test_size=0.2, seed=42)

    y_tr, X_tr, idx_tr = make_design(train)
    y_te, X_te, idx_te = make_design(test, ref_columns=list(X_tr.columns))

    # fit Gamma GLM with log link
    model = sm.GLM(y_tr, X_tr, family=sm.families.Gamma(sm.families.links.log()))
    res = model.fit(maxiter=200)

    # baseline mean severity (train)
    base_sev = float(y_tr.mean())
    base_mu_te = np.full_like(y_te.values, base_sev, dtype=float)

    mu_te = res.predict(X_te)

    dev_model = gamma_deviance(y_te.values, mu_te)
    dev_base = gamma_deviance(y_te.values, base_mu_te)

    mae = float(np.mean(np.abs(y_te.values - mu_te)))
    mape = float(np.mean(np.abs((y_te.values - mu_te) / y_te.values)))

    # coefficients
    params = res.params
    coef_tbl = pd.DataFrame({
        "feature": params.index,
        "coef": params.values,
        "std_err": res.bse.values,
        "multiplier_exp_coef": np.exp(params.values),
    }).sort_values("multiplier_exp_coef", ascending=False)
    coef_tbl.to_csv(COEF_OUT, index=False)

    # --- predictions ALL (within model_sev universe) ---
    y_all, X_all, idx_all = make_design(df, ref_columns=list(X_tr.columns))
    mu_all = res.predict(X_all)

    keep_cols = ["IDpol", "ClaimNb", "PaidLoss", "AvgSeverity"]
    keep_cols = [c for c in keep_cols if c in df.columns]
    out_all = df.loc[idx_all, keep_cols].copy()
    out_all["pred_avg_severity"] = mu_all
    out_all.to_csv(PRED_ALL_OUT, index=False)

    # --- predictions TEST ---
    out_test = df.loc[idx_te, keep_cols].copy()
    out_test["pred_avg_severity"] = mu_te
    out_test.to_csv(PRED_TEST_OUT, index=False)

    # report
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("06_train_severity_glm.py report\n")
        f.write("================================\n\n")
        f.write(f"Input: {IN_PATH}\n")
        f.write(f"Rows after cleaning: {len(df):,}\n")
        f.write(f"Train rows used (after NA filter): {len(y_tr):,}\n")
        f.write(f"Test rows used  (after NA filter): {len(y_te):,}\n\n")
        f.write("Model: Gamma GLM with log link (target = AvgSeverity)\n\n")
        f.write(f"Baseline mean severity (train): {base_sev:.6f}\n\n")
        f.write("Test metrics:\n")
        f.write(f"  Gamma deviance (model):    {dev_model:,.2f}\n")
        f.write(f"  Gamma deviance (baseline): {dev_base:,.2f}\n")
        f.write(f"  Deviance improvement:      {dev_base - dev_model:,.2f}\n")
        f.write(f"  MAE:                       {mae:,.6f}\n")
        f.write(f"  MAPE:                      {mape:.6f}\n\n")
        f.write("Outputs:\n")
        f.write(f"  Coefficients: {COEF_OUT}\n")
        f.write(f"  Predictions (ALL): {PRED_ALL_OUT}\n")
        f.write(f"  Predictions (TEST): {PRED_TEST_OUT}\n")
        f.write(f"  Report: {REPORT}\n\n")
        f.write("Note:\n")
        f.write("  Severity model trained only on rows with observed PaidLoss>0 & ClaimNb>0 (after winsorization),\n")
        f.write("  because sev data does not cover all ClaimNb>0 policies.\n")

    print("Done.")
    print("Wrote:", COEF_OUT)
    print("Wrote:", PRED_ALL_OUT)
    print("Wrote:", PRED_TEST_OUT)
    print("Wrote:", REPORT)


if __name__ == "__main__":
    main()
