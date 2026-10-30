import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()
engine = create_engine(os.environ["DATABASE_URL"])
pd.set_option("display.width", 200)

# Meter values are text with a decimal comma (for example 12,5); convert safely
VAL = (
    "CASE WHEN kw_raw ~ '^[0-9]+([.,][0-9]+)?$' "
    "THEN CAST(replace(kw_raw, ',', '.') AS numeric) END"
)
V = f"WITH v AS (SELECT meter_id, ts_raw, {VAL} AS kw FROM bronze.meter_raw) "


def show(title, sql):
    print(f"\n=== {title} ===", flush=True)
    print(pd.read_sql(text(sql), engine).to_string(index=False), flush=True)


show("1. Rows and meters", """
SELECT count(*) AS total_rows, count(DISTINCT meter_id) AS meters
FROM bronze.meter_raw
""")

show("2. Null, negative and non-numeric meter values", """
SELECT count(*) FILTER (WHERE kw_raw IS NULL) AS null_values,
       count(*) FILTER (WHERE kw_raw IS NOT NULL AND left(kw_raw, 1) = '-') AS negative,
       count(*) FILTER (WHERE kw_raw IS NOT NULL
                        AND kw_raw !~ '^-?[0-9]+([.,][0-9]+)?$') AS non_numeric
FROM bronze.meter_raw
""")

show("3. Duplicate (timestamp, meter) pairs", """
SELECT count(*) AS duplicate_groups FROM (
    SELECT ts_raw, meter_id FROM bronze.meter_raw
    GROUP BY ts_raw, meter_id HAVING count(*) > 1
) d
""")

show("4. Zero readings per meter and first non-zero reading", V + """
SELECT meter_id,
       count(*) AS total_rows,
       count(*) FILTER (WHERE kw = 0) AS zero_rows,
       round(100.0 * count(*) FILTER (WHERE kw = 0) / count(*), 1) AS zero_pct,
       min(ts_raw) FILTER (WHERE kw > 0) AS first_nonzero
FROM v GROUP BY meter_id ORDER BY meter_id
""")

show("5. Reading statistics (non-zero values, kW per 15 min)", V + """
SELECT min(kw) AS min_kw,
       round(avg(kw), 2) AS avg_kw,
       round(CAST(percentile_cont(0.99) WITHIN GROUP (ORDER BY kw) AS numeric), 2) AS p99_kw,
       max(kw) AS max_kw
FROM v WHERE kw > 0
""")

show("6. Average demand by hour of day (the daily curve)", V + """
SELECT extract(hour FROM CAST(ts_raw AS timestamp)) AS hour,
       round(avg(kw), 1) AS avg_kw
FROM v WHERE kw > 0 GROUP BY 1 ORDER BY 1
""")

show("7. Weather: missing values and ranges", """
SELECT count(*) AS weather_rows,
       count(*) FILTER (WHERE temperature_2m IS NULL
                        OR temperature_2m IN ('nan', 'None', '')) AS missing_temp,
       count(*) FILTER (WHERE shortwave_radiation IS NULL
                        OR shortwave_radiation IN ('nan', 'None', '')) AS missing_radiation,
       min(CASE WHEN temperature_2m ~ '^-?[0-9.]+$'
                THEN CAST(temperature_2m AS numeric) END) AS min_temp,
       max(CASE WHEN temperature_2m ~ '^-?[0-9.]+$'
                THEN CAST(temperature_2m AS numeric) END) AS max_temp,
       max(CASE WHEN shortwave_radiation ~ '^-?[0-9.]+$'
                THEN CAST(shortwave_radiation AS numeric) END) AS max_radiation
FROM bronze.weather_raw
""")