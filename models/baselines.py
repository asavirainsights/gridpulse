import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()
engine = create_engine(os.environ["DATABASE_URL"])
pd.set_option("display.width", 200)

df = (pd.read_sql(text("SELECT * FROM gold.features ORDER BY hour_utc"),
                  engine, parse_dates=["hour_utc"])
      .set_index("hour_utc"))

train = df[df["split"] == "train"]
# typical demand for each (day of week, hour), learned from TRAINING data only
profile = train.groupby(["dow_local", "hour_local"])["demand_kwh"].mean()


def score(y, p, hour):
    err = p - y
    peak = hour.isin([18, 19, 20, 21])
    return {
        "MAE": err.abs().mean(),
        "RMSE": float(np.sqrt((err ** 2).mean())),
        "MAPE_%": (err.abs() / y).mean() * 100,
        "peak_MAE": err[peak].abs().mean(),
        "bias": err.mean(),
        "under10_%": (p < 0.9 * y).mean() * 100,
    }


rows = []
for split in ("valid", "test"):
    d = df[df["split"] == split]
    key = pd.MultiIndex.from_frame(d[["dow_local", "hour_local"]])
    preds = {
        "same hour yesterday": d["lag_24"],
        "same hour last week": d["lag_168"],
        "average of 3 days": (d["lag_24"] + d["lag_48"] + d["lag_168"]) / 3,
        "hour-of-week profile": pd.Series(profile.reindex(key).to_numpy(), index=d.index),
    }
    for name, p in preds.items():
        row = {"split": split, "baseline": name}
        row.update(score(d["demand_kwh"], p, d["hour_local"]))
        rows.append(row)

res = pd.DataFrame(rows).round(1)
print(res.to_string(index=False))
os.makedirs("docs", exist_ok=True)
res.to_csv(os.path.join("docs", "baseline_results.csv"), index=False)
print("\nsaved docs/baseline_results.csv")