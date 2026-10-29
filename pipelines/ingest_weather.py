import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()
engine = create_engine(os.environ["DATABASE_URL"])

URL = "https://archive-api.open-meteo.com/v1/archive"
BASE = {
    "latitude": 38.7223,
    "longitude": -9.1393,
    "hourly": "temperature_2m,cloud_cover,shortwave_radiation",
    "timezone": "UTC",
}

# Cache folder: finished pieces are saved, so a rerun skips them
cache_dir = os.path.join(os.environ["DATA_DIR"], "weather_cache")
os.makedirs(cache_dir, exist_ok=True)

# Build the list of 3-month periods from 2011 to 2014
quarters = [
    ("01-01", "03-31"),
    ("04-01", "06-30"),
    ("07-01", "09-30"),
    ("10-01", "12-31"),
]
periods = []
for year in range(2011, 2015):
    for s, e in quarters:
        periods.append((f"{year}-{s}", f"{year}-{e}"))

frames = []
for start, end in periods:
    cache_file = os.path.join(cache_dir, f"{start}.csv")

    if os.path.exists(cache_file):
        print(f"{start} already downloaded, skipping", flush=True)
        frames.append(pd.read_csv(cache_file, dtype=str))
        continue

    params = {**BASE, "start_date": start, "end_date": end}

    for attempt in range(1, 6):
        try:
            print(f"Downloading {start} to {end} (try {attempt})...", flush=True)
            resp = requests.get(URL, params=params, timeout=(15, 120))
            resp.raise_for_status()
            part = pd.DataFrame(resp.json()["hourly"])
            part.to_csv(cache_file, index=False)
            frames.append(part.astype(str))
            print("  done", flush=True)
            break
        except (requests.RequestException, ValueError) as err:
            print(f"  failed: {type(err).__name__}", flush=True)
            time.sleep(5 * attempt)
    else:
        raise SystemExit(f"Could not download {start}. Rerun later; finished pieces are kept.")

df = pd.concat(frames, ignore_index=True)

# Bronze = raw as received, plus audit metadata
df["ingested_at"] = pd.Timestamp.now(tz="UTC")
df["source"] = "open-meteo-archive"

df.to_sql("weather_raw", engine, schema="bronze", if_exists="replace", index=False)
print(f"{len(df):,} weather rows loaded into bronze.weather_raw")