import os
import requests
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()
engine = create_engine(os.environ["DATABASE_URL"])

URL = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 51.5072,
    "longitude": -0.1276,
    "start_date": "2012-01-01",
    "end_date": "2013-12-31",
    "hourly": "temperature_2m,cloud_cover,shortwave_radiation",
    "timezone": "UTC",
}

resp = requests.get(URL, params=params, timeout=120)
resp.raise_for_status()
df = pd.DataFrame(resp.json()["hourly"])

# Bronze = raw as received, plus audit metadata
df["ingested_at"] = pd.Timestamp.now(tz="UTC")
df["source"] = "open-meteo-archive"

df.to_sql("weather_raw", engine, schema="bronze", if_exists="replace", index=False)
print(f"{len(df):,} weather rows loaded into bronze.weather_raw")