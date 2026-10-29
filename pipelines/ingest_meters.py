import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()
engine = create_engine(os.environ["DATABASE_URL"])
path = os.path.join(os.environ["DATA_DIR"], "LD2011_2014.txt")
if not os.path.exists(path):
    raise SystemExit(f"File not found: {path}")

N_METERS = 10                      # raise to 50 later
cols = list(range(0, N_METERS + 1))  # column 0 is the timestamp

# Start clean so re-running does not create duplicates
with engine.begin() as conn:
    conn.execute(text("DROP TABLE IF EXISTS bronze.meter_raw"))

total = 0
for chunk in pd.read_csv(path, sep=";", usecols=cols, dtype=str, chunksize=10_000):
    chunk = chunk.rename(columns={chunk.columns[0]: "ts_raw"})
    # wide -> long: one row per (timestamp, meter)
    long = chunk.melt(id_vars="ts_raw", var_name="meter_id", value_name="kw_raw")
    long["ingested_at"] = pd.Timestamp.now(tz="UTC")
    long["source_file"] = "LD2011_2014.txt"
    long.to_sql("meter_raw", engine, schema="bronze",
                if_exists="append", index=False, method="multi", chunksize=5000)
    total += len(long)
    print(f"running total: {total:,}")

print(f"Done: {total:,} rows in bronze.meter_raw")