import os
import glob
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()
engine = create_engine(os.environ["DATABASE_URL"])
data_dir = os.environ["DATA_DIR"]

files = sorted(glob.glob(os.path.join(data_dir, "halfhourly_dataset", "block_*.csv")))
if not files:
    raise SystemExit("No block_*.csv files found. Check DATA_DIR in .env")

# Start clean so re-running does not create duplicates
with engine.begin() as conn:
    conn.execute(text("DROP TABLE IF EXISTS bronze.meter_raw"))

total = 0
for path in files:
    name = os.path.basename(path)
    for chunk in pd.read_csv(path, dtype=str, chunksize=200_000):
        chunk = chunk.rename(columns={"energy(kWh/hh)": "energy_raw"})
        chunk["ingested_at"] = pd.Timestamp.now(tz="UTC")
        chunk["source_file"] = name
        chunk.to_sql("meter_raw", engine, schema="bronze",
                     if_exists="append", index=False, method="multi", chunksize=5000)
        total += len(chunk)
    print(f"loaded {name} (running total: {total:,})")

print(f"Done: {total:,} rows in bronze.meter_raw")