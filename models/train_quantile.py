import os
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()
engine = create_engine(os.environ["DATABASE_URL"])
pd.set_option("display.width", 220)

QUANTILES = [0.1, 0.5, 0.9]
TARGET = "demand_kwh"
CALENDAR = ["hour_local", "dow_local", "month", "is_weekend", "is_holiday"]
HISTORY = ["lag_24", "lag_48", "lag_168", "roll_mean_24", "roll_std_24",
           "roll_mean_168", "trend_velocity", "ramp_yesterday"]
WEATHER = ["temperature_c", "cloud_cover_pct", "radiation_wm2"]
VARIANTS = {
    "A no weather": HISTORY + CALENDAR,
    "B perfect weather forecast": HISTORY + CALENDAR + WEATHER,
    "C noisy weather forecast": HISTORY + CALENDAR + WEATHER,
}


def pinball(y, q, a):
    d = y - q
    return float(np.mean(np.maximum(a * d, (a - 1) * d)))


def fit_quantiles(train, features):
    models = {}
    for a in QUANTILES:
        m = lgb.LGBMRegressor(
            objective="quantile", alpha=a, n_estimators=400, learning_rate=0.03,
            num_leaves=15, min_child_samples=40, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, random_state=42, verbose=-1)
        m.fit(train[features], train[TARGET])
        models[a] = m
    return models


def predict(models, d, features):
    q = np.column_stack([models[a].predict(d[features]) for a in QUANTILES])
    return np.sort(q, axis=1)          # guarantees P10 <= P50 <= P90


def evaluate(y, q, hour):
    y = y.to_numpy()
    err = q[:, 1] - y
    peak = hour.isin([18, 19, 20, 21]).to_numpy()
    return {
        "MAE_P50": float(np.abs(err).mean()),
        "peak_MAE": float(np.abs(err[peak]).mean()),
        "under10_%": float((q[:, 1] < 0.9 * y).mean() * 100),
        "avg_pinball": float(np.mean([pinball(y, q[:, i], a)
                                      for i, a in enumerate(QUANTILES)])),
        "coverage_%": float(((y >= q[:, 0]) & (y <= q[:, 2])).mean() * 100),
        "band_width": float((q[:, 2] - q[:, 0]).mean()),
    }


def conformal_shift(y, q, alpha=0.2):
    """Amount to widen (or narrow) the P10-P90 band so it covers 80 percent."""
    scores = np.maximum(q[:, 0] - y, y - q[:, 2])
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


def apply_shift(q, s):
    out = q.copy()
    out[:, 0] = np.maximum(np.minimum(out[:, 0] - s, out[:, 1]), 0)
    out[:, 2] = np.maximum(out[:, 2] + s, out[:, 1])
    return out


# ---- data -------------------------------------------------------------------
raw = (pd.read_sql(text("SELECT * FROM gold.features ORDER BY hour_utc"),
                   engine, parse_dates=["hour_utc"]).set_index("hour_utc"))

# ASSUMED weather-forecast errors for variant C (documented in the README)
rng = np.random.default_rng(42)
noisy = raw.copy()
noisy["temperature_c"] = noisy["temperature_c"] + rng.normal(0, 1.5, len(noisy))
noisy["cloud_cover_pct"] = (noisy["cloud_cover_pct"]
                            + rng.normal(0, 15, len(noisy))).clip(0, 100)
noisy["radiation_wm2"] = (noisy["radiation_wm2"]
                          + rng.normal(0, 100, len(noisy))).clip(lower=0)

# ---- train and evaluate every variant ----------------------------------------
results, fitted, preds = [], {}, {}
for name, features in VARIANTS.items():
    data = noisy if name.startswith("C") else raw
    models = fit_quantiles(data[data["split"] == "train"], features)
    fitted[name] = models
    for split in ("valid", "test"):
        d = data[data["split"] == split]
        q = predict(models, d, features)
        preds[(name, split)] = (d, q)
        row = {"variant": name, "split": split}
        row.update(evaluate(d[TARGET], q, d["hour_local"]))
        results.append(row)

res = pd.DataFrame(results)
print("\n=== Model results (raw quantile bands) ===")
print(res.round(2).to_string(index=False))

base = pd.read_csv(os.path.join("docs", "baseline_results.csv"))
base = base[base["baseline"] == "average of 3 days"]
print("\n=== Best baseline to beat ===")
print(base[["split", "MAE", "peak_MAE", "under10_%"]].to_string(index=False))

# ---- champion: best REALISTIC variant on validation pinball loss -------------
candidates = ["A no weather", "C noisy weather forecast"]
vs = res[(res["split"] == "valid") & res["variant"].isin(candidates)]
champion = vs.set_index("variant")["avg_pinball"].idxmin()
features = VARIANTS[champion]
print(f"\nChampion: {champion}")

# ---- calibration on validation, check on the untouched test period -----------
d_valid, q_valid = preds[(champion, "valid")]
shift = conformal_shift(d_valid[TARGET].to_numpy(), q_valid)
d_test, q_test = preds[(champion, "test")]
q_test_cal = apply_shift(q_test, shift)
y_test = d_test[TARGET].to_numpy()
for label, q in (("raw", q_test), ("calibrated", q_test_cal)):
    cov = ((y_test >= q[:, 0]) & (y_test <= q[:, 2])).mean() * 100
    print(f"test coverage of P10-P90 band, {label:<10}: {cov:5.1f}%  "
          f"(target 80%), width {(q[:, 2] - q[:, 0]).mean():6.1f} kWh")
print(f"conformal shift applied: {shift:+.1f} kWh")

# ---- feature importance ------------------------------------------------------
imp = pd.Series(fitted[champion][0.5].booster_.feature_importance(importance_type="gain"),
                index=features)
print("\nTop features for the P50 model (% of gain):")
print((100 * imp / imp.sum()).sort_values(ascending=False).head(8).round(1).to_string())

# ---- publish forecasts and save the champion ---------------------------------
frames = []
for split in ("valid", "test"):
    d, q = preds[(champion, split)]
    q = apply_shift(q, shift)
    frames.append(pd.DataFrame(
        {"split": split, "demand_kwh": d[TARGET].to_numpy(),
         "q10": q[:, 0], "q50": q[:, 1], "q90": q[:, 2]}, index=d.index))
fc = pd.concat(frames)
fc.to_sql("demand_forecast", engine, schema="gold", if_exists="replace",
          index=True, index_label="hour_utc", method="multi", chunksize=1000)
with engine.begin() as conn:
    conn.execute(text("ALTER TABLE gold.demand_forecast ADD PRIMARY KEY (hour_utc)"))

os.makedirs("models/artifacts", exist_ok=True)
for a, m in fitted[champion].items():
    m.booster_.save_model(f"models/artifacts/demand_q{int(a * 100)}.txt")
with open("models/artifacts/demand_meta.json", "w") as f:
    json.dump({"champion": champion, "features": features,
               "conformal_shift_kwh": shift, "quantiles": QUANTILES}, f, indent=2)
print(f"\nwrote {len(fc):,} rows to gold.demand_forecast and saved models")
# ---- experiment tracking with MLflow -------------------------------------------
import mlflow
import mlflow.lightgbm
from mlflow import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("gridpulse-demand")
client = MlflowClient()
METRICS = ["MAE_P50", "peak_MAE", "under10_%", "avg_pinball", "coverage_%", "band_width"]

for variant, feats in VARIANTS.items():
    with mlflow.start_run(run_name=variant):
        mlflow.log_param("variant", variant)
        mlflow.log_param("features", ",".join(feats))
        mlflow.log_param("is_champion", variant == champion)
        for _, r in res[res["variant"] == variant].iterrows():
            for m in METRICS:
                mlflow.log_metric(f"{r['split']}_{m.replace('%', 'pct')}", float(r[m]))
        if variant == champion:
            mlflow.log_metric("conformal_shift_kwh", float(shift))
            mlflow.log_artifact("docs/baseline_results.csv")
            mlflow.log_artifact("models/artifacts/demand_meta.json")
            for a, model in fitted[champion].items():
                reg_name = f"gridpulse-demand-q{int(a * 100)}"
                mlflow.lightgbm.log_model(model.booster_,
                                          artifact_path=f"model_q{int(a * 100)}",
                                          registered_model_name=reg_name)
                versions = client.search_model_versions(f"name='{reg_name}'")
                latest = max(int(v.version) for v in versions)
                client.set_registered_model_alias(reg_name, "champion", latest)

print("\nlogged runs and registered the champion demand models in MLflow")