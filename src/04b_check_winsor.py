import os
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEV_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "model_sev.csv")
OUT_PATH = os.path.join(PROJECT_ROOT, "logs", "04b_winsor_check.txt")

def q(s, p):
    return float(np.nanpercentile(s.values, p))

df = pd.read_csv(SEV_PATH)
s = df["AvgSeverity"]

with open(OUT_PATH, "w", encoding="utf-8") as f:
    f.write("Winsor check for AvgSeverity (after capping)\n")
    f.write("==========================================\n\n")
    f.write(f"rows: {len(df):,}\n")
    f.write(f"min: {float(np.nanmin(s.values))}\n")
    f.write(f"p50: {q(s, 50)}\n")
    f.write(f"p90: {q(s, 90)}\n")
    f.write(f"p95: {q(s, 95)}\n")
    f.write(f"p99: {q(s, 99)}\n")
    f.write(f"max: {float(np.nanmax(s.values))}\n")

print("Wrote:", OUT_PATH)
