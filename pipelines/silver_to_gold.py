import os
import sys
import pandas as pd
import holidays
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()
engine = create_engine(os.environ["DATABASE_URL"])

SOLAR_KWP = 500            # ASSUMPTION: installed solar capacity in kW peak
PERFORMANCE_RATIO = 0.8    # ASSUMPTION: system losses (wiring, inverter, dirt)
TRAIN_END = "2013-12-31 23:00"
VALID_END = "2014-06-30 23:00"

# ---- 1. hourly demand from Silver (complete hours only) -------------------
HOURLY_SQL = """
SELECT date_trunc('hour', ts_utc) AS hour_utc,
       sum(kw) * 0.25 AS demand_kwh
FROM silver.meter_clean
GROUP BY 1
HAVING count(*) = 4 * (SELECT count(DISTINCT meter_id) FROM silver.meter_clean)
ORDER BY 1
"""
WEATHER_SQL = """
SELECT ts_utc, temperature_c, cloud_cover_pct, radiation_wm2
FROM silver.weather_clean ORDER BY 1
"""

demand = (pd.read_sql(text(HOURLY_SQL), engine, parse_dates=["hour_utc"])
          .set_index("hour_utc")["demand_kwh"].astype(float))
weather = (pd.read_sql(text(WEATHER_SQL), engine, parse_dates=["ts_utc"])
           .set_index("ts_utc").astype(float))
print(f"complete hourly rows from Silver: {len(demand):,}", flush=True)

# ---- 2. continuous hourly index; fill the clock-change gaps ---------------
full_index = pd.date_range(demand.index.min(), demand.index.max(), freq="h")
demand = demand.reindex(full_index)
filled = int(demand.isna().sum())
demand = demand.interpolate(method="linear", limit=3)
if demand.isna().any():
    sys.exit("GOLD AUDIT FAILED: a demand gap is longer than 3 hours")
print(f"hours filled by interpolation: {filled}", flush=True)

# ---- 3. weather and simulated solar ---------------------------------------
# Open-Meteo radiation is the average of the PRECEDING hour, so the value
# stamped H+1:00 describes demand hour H:00-H+1:00. Shift it back one hour.
rad = weather["radiation_wm2"].shift(-1).ffill(limit=1)
df = pd.DataFrame({"demand_kwh": demand})
df["temperature_c"] = weather["temperature_c"].reindex(full_index)
df["cloud_cover_pct"] = weather["cloud_cover_pct"].reindex(full_index)
df["radiation_wm2"] = rad.reindex(full_index)
# W/m2 over one hour -> kWh/m2; times capacity at 1000 W/m2, times losses
df["pv_kwh"] = SOLAR_KWP * (df["radiation_wm2"] / 1000.0) * PERFORMANCE_RATIO

# ---- 4. leak-free features (everything is at least 24 hours old) ----------
for h in (24, 48, 168):
    df[f"lag_{h}"] = df["demand_kwh"].shift(h)

past = df["demand_kwh"].shift(24)          # window below ends 24 h before target
df["roll_mean_24"] = past.rolling(24).mean()
df["roll_std_24"] = past.rolling(24).std()
df["roll_mean_168"] = past.rolling(168).mean()
# day-over-day change in demand level (trend velocity)
df["trend_velocity"] = df["roll_mean_24"] - df["demand_kwh"].shift(48).rolling(24).mean()
# how fast demand was ramping at this time of day yesterday
df["ramp_yesterday"] = df["lag_24"] - df["demand_kwh"].shift(27)

# weather as known 24 h earlier (for a solar forecast without peeking)
df["temp_lag_24"] = df["temperature_c"].shift(24)
df["cloud_lag_24"] = df["cloud_cover_pct"].shift(24)
df["rad_lag_24"] = df["radiation_wm2"].shift(24)

# calendar features use LOCAL time, because people follow the local clock
local = df.index.tz_localize("UTC").tz_convert("Europe/Lisbon")
df["hour_local"] = local.hour
df["dow_local"] = local.dayofweek
df["month"] = local.month
df["is_weekend"] = (local.dayofweek >= 5).astype(int)
pt_holidays = holidays.Portugal(years=range(2012, 2015))
df["is_holiday"] = [int(d in pt_holidays) for d in local.date]

# chronological split, never shuffled
df["split"] = "test"
df.loc[df.index <= VALID_END, "split"] = "valid"
df.loc[df.index <= TRAIN_END, "split"] = "train"

before = len(df)
df = df.dropna()                           # drops the warm-up rows at the start
df.index.name = "hour_utc"
print(f"warm-up rows dropped: {before - len(df):,}", flush=True)

# ---- audit gate -------------------------------------------------------------
problems = []
if df.index.to_series().diff().dropna().ne(pd.Timedelta(hours=1)).any():
    problems.append("hourly index has gaps")
if (df["demand_kwh"] <= 0).any():
    problems.append("non-positive demand found")
if (df["pv_kwh"] < 0).any():
    problems.append("negative solar found")
if problems:
    sys.exit("GOLD AUDIT FAILED, nothing published:\n  " + "\n  ".join(problems))

# ---- publish ----------------------------------------------------------------
df.to_sql("features", engine, schema="gold", if_exists="replace",
          index=True, index_label="hour_utc", method="multi", chunksize=1000)
with engine.begin() as conn:
    conn.execute(text("ALTER TABLE gold.features ADD PRIMARY KEY (hour_utc)"))

print(f"\nfeature rows written: {len(df):,}")
print(f"range: {df.index.min()} to {df.index.max()}")
for name, n in df["split"].value_counts().items():
    print(f"  {name:<6}{n:>8,}")
print(f"average demand: {df['demand_kwh'].mean():,.0f} kWh/hour, "
      f"average solar: {df['pv_kwh'].mean():,.0f} kWh/hour")