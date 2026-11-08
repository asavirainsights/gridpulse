import os
import sys
import subprocess
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from drift_stats import psi

load_dotenv()
engine = create_engine(os.environ["DATABASE_URL"])
pd.set_option("display.width", 220)

# ---- thresholds (tunable; document them in the README) -----------------------
PSI_WATCH = 0.10                 # small shift
PSI_ALERT = 0.25                 # major shift
ERROR_RATIO_ALERT = 1.30         # monthly error vs validation error
COVERAGE_RANGE = (70.0, 90.0)    # acceptable share of hours inside P10-P90
TRIGGER_FEATURES = ["demand_kwh", "lag_168"]        # can trigger retraining
INFO_FEATURES = ["temperature_c", "radiation_wm2"]  # shown for information only


def evaluate(label, ref, cur, ref_mape):
    row = {"month": label, "hours": len(cur)}
    for col in TRIGGER_FEATURES + INFO_FEATURES:
        row[f"psi_{col}"] = psi(ref[col].to_numpy(), cur[col].to_numpy())
    y = cur["demand_kwh"].to_numpy()
    mape = float((np.abs(cur["q50"].to_numpy() - y) / y).mean() * 100)
    row["mape_pct"] = mape
    row["error_ratio"] = mape / ref_mape
    row["coverage_pct"] = float(((y >= cur["q10"]) & (y <= cur["q90"])).mean() * 100)

    reasons = []
    if max(row[f"psi_{c}"] for c in TRIGGER_FEATURES) > PSI_ALERT:
        reasons.append("demand data drift")
    if row["error_ratio"] > ERROR_RATIO_ALERT:
        reasons.append("forecast error rising")
    if not COVERAGE_RANGE[0] <= row["coverage_pct"] <= COVERAGE_RANGE[1]:
        reasons.append("bands miscalibrated")
    if reasons:
        row["decision"], row["why"] = "RETRAIN", "; ".join(reasons)
    elif max(row[f"psi_{c}"] for c in TRIGGER_FEATURES + INFO_FEATURES) > PSI_WATCH:
        row["decision"], row["why"] = "WATCH", "inputs shifting, model still fine"
    else:
        row["decision"], row["why"] = "OK", ""
    return row


# ---- data ----------------------------------------------------------------------
feat = (pd.read_sql(text("""SELECT hour_utc, split, demand_kwh, lag_168,
                                   temperature_c, radiation_wm2
                            FROM gold.features ORDER BY hour_utc"""),
                    engine, parse_dates=["hour_utc"]).set_index("hour_utc"))
fc = (pd.read_sql(text("SELECT hour_utc, q10, q50, q90 FROM gold.demand_forecast"),
                  engine, parse_dates=["hour_utc"]).set_index("hour_utc"))
data = feat.join(fc, how="inner")            # validation and test months only
train = feat[feat["split"] == "train"]

valid = data[data["split"] == "valid"]
ref_mape = float((np.abs(valid["q50"] - valid["demand_kwh"]) / valid["demand_kwh"]).mean() * 100)

# ---- monthly monitoring ----------------------------------------------------------
rows = []
for (yr, mo), cur in data.groupby([data.index.year, data.index.month]):
    ref = train[train.index.month == mo]      # same calendar month in training years
    rows.append(evaluate(f"{yr}-{mo:02d}", ref, cur, ref_mape))
    last_ref, last_cur = ref, cur
report = pd.DataFrame(rows)

show = ["month", "psi_demand_kwh", "psi_lag_168", "psi_temperature_c",
        "psi_radiation_wm2", "mape_pct", "error_ratio", "coverage_pct", "decision", "why"]
print(f"reference error (validation period): MAPE {ref_mape:.2f}%")
print("\n=== Monthly drift report (each month vs the same month in 2012-2013) ===")
print(report[show].round(2).to_string(index=False))
print("\n" + report["decision"].value_counts().to_string())

# ---- stress test: prove the alarm fires --------------------------------------------
stress_cur = last_cur.copy()
stress_cur["demand_kwh"] = stress_cur["demand_kwh"] * 1.25
stress = evaluate("stress test (+25% demand)", last_ref, stress_cur, ref_mape)
print("\n=== Synthetic stress test: last month with demand scaled up 25% ===")
print(pd.DataFrame([stress])[show].round(2).to_string(index=False))

# ---- save ---------------------------------------------------------------------------
os.makedirs("docs", exist_ok=True)
report.to_csv(os.path.join("docs", "drift_report.csv"), index=False)
report.to_sql("drift_monitor", engine, schema="gold", if_exists="replace", index=False)
print("\nsaved docs/drift_report.csv and gold.drift_monitor")

# ---- optional automatic retraining ------------------------------------------------
if "--retrain" in sys.argv:
    latest = rows[-1]
    if latest["decision"] == "RETRAIN":
        print(f"\n{latest['month']} flagged ({latest['why']}) -> retraining the demand model")
        subprocess.run([sys.executable, os.path.join("models", "train_quantile.py")], check=True)
    else:
        print(f"\nlatest month is {latest['decision']}: no retraining needed")