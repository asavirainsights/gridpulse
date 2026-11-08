import os
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()
engine = create_engine(os.environ["DATABASE_URL"])
pd.set_option("display.width", 200)

QUANTILES = [0.1, 0.5, 0.9]
TARGET = "pv_kwh"
FEATURES = ["hour_of_day", "doy", "pv_lag_24", "pv_mean_3d", "cloud_lag_24", "temp_lag_24"]

# ---- data and leak-free features (everything is at least 24 h old) -----------
df = (pd.read_sql(text("""SELECT hour_utc, split, pv_kwh, cloud_lag_24, temp_lag_24
                          FROM gold.features ORDER BY hour_utc"""),
                  engine, parse_dates=["hour_utc"]).set_index("hour_utc"))
df["pv_kwh"] = df["pv_kwh"].astype(float)
pv = df["pv_kwh"]
df["pv_lag_24"] = pv.shift(24)
df["pv_mean_3d"] = (pv.shift(24) + pv.shift(48) + pv.shift(72)) / 3
df["hour_of_day"] = df.index.hour          # UTC hour: the sun follows UTC, not clocks
df["doy"] = df.index.dayofyear
df = df.dropna()

# hours of the year in which the sun is ever up (learned from training data only)
train_all = df[df["split"] == "train"]
daylight = train_all.groupby([train_all.index.month, train_all.index.hour])[TARGET].max() > 0


def is_day(d):
    key = pd.MultiIndex.from_arrays([d.index.month, d.index.hour])
    return daylight.reindex(key, fill_value=False).to_numpy(dtype=bool)


def pinball(y, q, a):
    d = y - q
    return float(np.mean(np.maximum(a * d, (a - 1) * d)))


def fit_quantiles(train):
    models = {}
    for a in QUANTILES:
        m = lgb.LGBMRegressor(
            objective="quantile", alpha=a, n_estimators=400, learning_rate=0.03,
            num_leaves=15, min_child_samples=40, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, random_state=42, verbose=-1)
        m.fit(train[FEATURES], train[TARGET])
        models[a] = m
    return models


def predict(models, d):
    q = np.column_stack([models[a].predict(d[FEATURES]) for a in QUANTILES])
    q = np.sort(q, axis=1).clip(min=0)
    q[~is_day(d), :] = 0.0                 # no sun, no solar
    return q


def conformal_shift(y, q, alpha=0.2):
    scores = np.maximum(q[:, 0] - y, y - q[:, 2])
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


def apply_shift(q, s, day):
    out = q.copy()
    out[:, 0] = np.maximum(np.minimum(out[:, 0] - s, out[:, 1]), 0)
    out[:, 2] = np.maximum(out[:, 2] + s, out[:, 1])
    out[~day, :] = 0.0
    return out


def evaluate(y, q, day):
    y, q = y[day], q[day]
    err = q[:, 1] - y
    return {
        "MAE_P50": float(np.abs(err).mean()),
        "MAE_%_of_mean": float(100 * np.abs(err).mean() / y.mean()),
        "bias": float(err.mean()),
        "avg_pinball": float(np.mean([pinball(y, q[:, i], a)
                                      for i, a in enumerate(QUANTILES)])),
        "coverage_%": float(((y >= q[:, 0]) & (y <= q[:, 2])).mean() * 100),
        "band_width": float((q[:, 2] - q[:, 0]).mean()),
    }


# ---- train on daylight training hours ----------------------------------------
train = df[df["split"] == "train"]
models = fit_quantiles(train[is_day(train)])

rows, preds = [], {}
for split in ("valid", "test"):
    d = df[df["split"] == split]
    day = is_day(d)
    y = d[TARGET].to_numpy()
    q = predict(models, d)
    preds[split] = (d, q, day, y)
    r = {"method": "LightGBM P10/P50/P90", "split": split}
    r.update(evaluate(y, q, day))
    rows.append(r)
    for name, p in {"same hour yesterday": d["pv_lag_24"],
                    "average of 3 days": d["pv_mean_3d"]}.items():
        err = p.to_numpy()[day] - y[day]
        rows.append({"method": "baseline: " + name, "split": split,
                     "MAE_P50": float(np.abs(err).mean()),
                     "MAE_%_of_mean": float(100 * np.abs(err).mean() / y[day].mean()),
                     "bias": float(err.mean())})

print("\n=== Solar results (daylight hours only) ===")
print(pd.DataFrame(rows).round(2).to_string(index=False))

# ---- calibrate on validation, check on the untouched test period -------------
_, q_valid, day_valid, y_valid = preds["valid"]
shift = conformal_shift(y_valid[day_valid], q_valid[day_valid])
d_test, q_test, day_test, y_test = preds["test"]
q_test_cal = apply_shift(q_test, shift, day_test)
for label, q in (("raw", q_test), ("calibrated", q_test_cal)):
    qq, yy = q[day_test], y_test[day_test]
    cov = ((yy >= qq[:, 0]) & (yy <= qq[:, 2])).mean() * 100
    print(f"test coverage of P10-P90 band, {label:<10}: {cov:5.1f}%  "
          f"(target 80%), width {(qq[:, 2] - qq[:, 0]).mean():6.1f} kWh")
print(f"conformal shift applied: {shift:+.1f} kWh")

# ---- publish and save ----------------------------------------------------------
frames = []
for split in ("valid", "test"):
    d, q, day, y = preds[split]
    q = apply_shift(q, shift, day)
    frames.append(pd.DataFrame({"split": split, "pv_kwh": y, "q10": q[:, 0],
                                "q50": q[:, 1], "q90": q[:, 2]}, index=d.index))
fc = pd.concat(frames)
fc.to_sql("solar_forecast", engine, schema="gold", if_exists="replace",
          index=True, index_label="hour_utc", method="multi", chunksize=1000)
with engine.begin() as conn:
    conn.execute(text("ALTER TABLE gold.solar_forecast ADD PRIMARY KEY (hour_utc)"))

os.makedirs("models/artifacts", exist_ok=True)
for a, m in models.items():
    m.booster_.save_model(f"models/artifacts/solar_q{int(a * 100)}.txt")
with open("models/artifacts/solar_meta.json", "w") as f:
    json.dump({"features": FEATURES, "conformal_shift_kwh": shift,
               "quantiles": QUANTILES}, f, indent=2)
print(f"\nwrote {len(fc):,} rows to gold.solar_forecast and saved models")
# ---- experiment tracking with MLflow -------------------------------------------
import mlflow
import mlflow.lightgbm
from mlflow import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("gridpulse-solar")
client = MlflowClient()

with mlflow.start_run(run_name="solar quantile model"):
    mlflow.log_param("features", ",".join(FEATURES))
    mlflow.log_metric("conformal_shift_kwh", float(shift))
    for r in rows:
        if r["method"].startswith("LightGBM"):
            tag = "model"
        else:
            tag = "naive_" + r["method"].split(": ")[1].replace(" ", "_")
        for key, value in r.items():
            if key in ("method", "split"):
                continue
            name = f"{tag}_{r['split']}_{key}".replace("%", "pct")
            mlflow.log_metric(name, float(value))
    for a, model in models.items():
        reg_name = f"gridpulse-solar-q{int(a * 100)}"
        mlflow.lightgbm.log_model(model.booster_,
                                  artifact_path=f"model_q{int(a * 100)}",
                                  registered_model_name=reg_name)
        versions = client.search_model_versions(f"name='{reg_name}'")
        latest = max(int(v.version) for v in versions)
        client.set_registered_model_alias(reg_name, "champion", latest)

print("\nlogged the solar run and registered the models in MLflow")