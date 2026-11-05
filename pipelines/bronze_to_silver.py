import os
import sys
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()
engine = create_engine(os.environ["DATABASE_URL"])

KW_CEILING = 1000       # kW per 15 min; the largest value in the profile was 552
MAX_BAD_SHARE = 0.01    # gate: unexpected rejects must stay under 1 percent


def run(conn, sql):
    return conn.execute(text(sql))


def log_run(pipeline, rows_in, rows_clean, rows_quarantined, status):
    with engine.begin() as conn:
        run(conn, """
            CREATE TABLE IF NOT EXISTS silver.pipeline_runs (
                run_ts timestamptz DEFAULT now(),
                pipeline text, rows_in bigint, rows_clean bigint,
                rows_quarantined bigint, status text)
        """)
        conn.execute(
            text("""INSERT INTO silver.pipeline_runs
                    (pipeline, rows_in, rows_clean, rows_quarantined, status)
                    VALUES (:p, :i, :c, :q, :s)"""),
            {"p": pipeline, "i": rows_in, "c": rows_clean,
             "q": rows_quarantined, "s": status},
        )


def print_reasons(reasons):
    for reason, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"  {reason:<26}{n:>12,}", flush=True)


# ---------------------------------------------------------------- WEATHER
WEATHER_STAGING = """
CREATE TABLE silver.weather_staging AS
WITH typed AS (
    SELECT CASE WHEN "time" ~ '^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}$'
                THEN CAST("time" AS timestamp) END AS ts_utc,
           CASE WHEN temperature_2m ~ '^-?[0-9]+([.][0-9]+)?$'
                THEN CAST(temperature_2m AS numeric) END AS temperature_c,
           CASE WHEN cloud_cover ~ '^-?[0-9]+([.][0-9]+)?$'
                THEN CAST(cloud_cover AS numeric) END AS cloud_cover_pct,
           CASE WHEN shortwave_radiation ~ '^-?[0-9]+([.][0-9]+)?$'
                THEN CAST(shortwave_radiation AS numeric) END AS radiation_wm2
    FROM bronze.weather_raw
)
SELECT ts_utc, temperature_c, cloud_cover_pct, radiation_wm2,
       CASE
           WHEN ts_utc IS NULL THEN 'bad_timestamp'
           WHEN temperature_c IS NULL OR cloud_cover_pct IS NULL
                OR radiation_wm2 IS NULL THEN 'missing_value'
           WHEN temperature_c NOT BETWEEN -30 AND 50 THEN 'temperature_out_of_range'
           WHEN cloud_cover_pct NOT BETWEEN 0 AND 100 THEN 'cloud_out_of_range'
           WHEN radiation_wm2 NOT BETWEEN 0 AND 1500 THEN 'radiation_out_of_range'
       END AS reject_reason
FROM typed
"""

WEATHER_PUBLISH = [
    """CREATE TABLE IF NOT EXISTS silver.weather_clean (
           ts_utc timestamp PRIMARY KEY,
           temperature_c numeric NOT NULL,
           cloud_cover_pct numeric NOT NULL,
           radiation_wm2 numeric NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS quarantine.weather_rejects (
           run_ts timestamptz, ts_utc timestamp, temperature_c numeric,
           cloud_cover_pct numeric, radiation_wm2 numeric, reject_reason text)""",
    "TRUNCATE silver.weather_clean",
    """INSERT INTO silver.weather_clean
       SELECT ts_utc, temperature_c, cloud_cover_pct, radiation_wm2
       FROM silver.weather_staging WHERE reject_reason IS NULL""",
    "TRUNCATE quarantine.weather_rejects",
    """INSERT INTO quarantine.weather_rejects
       SELECT now(), ts_utc, temperature_c, cloud_cover_pct, radiation_wm2, reject_reason
       FROM silver.weather_staging WHERE reject_reason IS NOT NULL""",
]


def process_weather():
    print("\n=== WEATHER: write ===", flush=True)
    with engine.begin() as conn:
        run(conn, "DROP TABLE IF EXISTS silver.weather_staging")
        run(conn, WEATHER_STAGING)

    print("=== WEATHER: audit ===", flush=True)
    with engine.connect() as conn:
        bronze_rows = run(conn, "SELECT count(*) FROM bronze.weather_raw").scalar()
        staged_rows = run(conn, "SELECT count(*) FROM silver.weather_staging").scalar()
        reasons = dict(run(conn, """
            SELECT coalesce(reject_reason, 'clean'), count(*)
            FROM silver.weather_staging GROUP BY 1""").all())
        n, d, expected = run(conn, """
            SELECT count(*), count(DISTINCT ts_utc),
                   CAST(extract(epoch FROM max(ts_utc) - min(ts_utc)) / 3600 AS bigint) + 1
            FROM silver.weather_staging WHERE reject_reason IS NULL""").one()
    print_reasons(reasons)

    clean = reasons.get("clean", 0)
    rejected = staged_rows - clean
    problems = []
    if staged_rows != bronze_rows:
        problems.append(f"row count mismatch: bronze {bronze_rows:,} vs staged {staged_rows:,}")
    if n != d:
        problems.append("duplicate hours found")
    if n != expected:
        problems.append(f"gaps in hourly series: {n:,} rows, expected {expected:,}")
    if rejected / bronze_rows > MAX_BAD_SHARE:
        problems.append(f"{rejected:,} rejected rows exceed the {MAX_BAD_SHARE:.0%} gate")
    if problems:
        log_run("weather", bronze_rows, clean, rejected, "FAILED")
        sys.exit("WEATHER AUDIT FAILED, nothing published:\n  " + "\n  ".join(problems))

    print("=== WEATHER: publish ===", flush=True)
    with engine.begin() as conn:
        for sql in WEATHER_PUBLISH:
            run(conn, sql)
    log_run("weather", bronze_rows, clean, rejected, "PUBLISHED")
    print(f"  published {clean:,} rows, quarantined {rejected:,}", flush=True)


# ----------------------------------------------------------------- METERS
METER_STAGING = """
CREATE TABLE silver.meter_staging AS
WITH typed AS (
    SELECT meter_id,
           kw_raw,
           CASE WHEN ts_raw ~ '^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}$'
                THEN CAST(ts_raw AS timestamp) END AS ts_end,
           CASE WHEN kw_raw ~ '^-?[0-9]+([.,][0-9]+)?$'
                THEN CAST(replace(kw_raw, ',', '.') AS numeric) END AS kw
    FROM bronze.meter_raw
),
first_on AS (
    SELECT meter_id, min(ts_end) FILTER (WHERE kw > 0) AS first_nonzero
    FROM typed GROUP BY meter_id
),
dst_days AS (
    SELECT CAST(d AS date) AS day, CAST(extract(month FROM d) AS int) AS month
    FROM generate_series(timestamp '2011-01-01', timestamp '2015-01-01',
                         interval '1 day') AS d
    WHERE extract(dow FROM d) = 0
      AND extract(day FROM d) >= 25
      AND extract(month FROM d) IN (3, 10)
)
SELECT t.meter_id,
       t.ts_end,
       t.ts_end AS interval_start,
       t.kw,
       t.kw_raw,
       CASE
           WHEN t.ts_end IS NULL THEN 'bad_timestamp'
           WHEN t.kw IS NULL THEN 'bad_value'
           WHEN t.kw < 0 THEN 'negative_value'
           WHEN t.kw > __CEILING__ THEN 'above_ceiling'
           WHEN f.first_nonzero IS NULL OR t.ts_end < f.first_nonzero THEN 'pre_service_zero'
           WHEN dd.month = 3 THEN 'dst_gap_hour'
           WHEN dd.month = 10 THEN 'dst_doubled_hour'
       END AS reject_reason
FROM typed t
JOIN first_on f ON f.meter_id = t.meter_id
LEFT JOIN dst_days dd
       ON dd.day = CAST(t.ts_end AS date)
      AND CAST(t.ts_end AS time) >= time '01:00'
      AND CAST(t.ts_end AS time) < time '02:00'
"""

METER_PUBLISH = [
    """CREATE TABLE IF NOT EXISTS silver.meter_clean (
           meter_id text NOT NULL,
           ts_end timestamp NOT NULL,
           interval_start_local timestamp NOT NULL,
           ts_utc timestamp NOT NULL,
           kw numeric NOT NULL,
           PRIMARY KEY (meter_id, ts_end))""",
    """CREATE TABLE IF NOT EXISTS quarantine.meter_rejects (
           run_ts timestamptz, meter_id text, ts_end timestamp,
           kw_raw text, reject_reason text)""",
    "TRUNCATE silver.meter_clean",
    """INSERT INTO silver.meter_clean
       SELECT meter_id, ts_end, interval_start,
              (interval_start AT TIME ZONE 'Europe/Lisbon') AT TIME ZONE 'UTC',
              kw
       FROM silver.meter_staging WHERE reject_reason IS NULL""",
    "TRUNCATE quarantine.meter_rejects",
    """INSERT INTO quarantine.meter_rejects
       SELECT now(), meter_id, ts_end, kw_raw, reject_reason
       FROM silver.meter_staging WHERE reject_reason IS NOT NULL""",
]


def process_meters():
    print("\n=== METERS: write ===", flush=True)
    with engine.begin() as conn:
        run(conn, "DROP TABLE IF EXISTS silver.meter_staging")
        run(conn, METER_STAGING.replace("__CEILING__", str(KW_CEILING)))

    print("=== METERS: audit ===", flush=True)
    with engine.connect() as conn:
        bronze_rows = run(conn, "SELECT count(*) FROM bronze.meter_raw").scalar()
        staged_rows = run(conn, "SELECT count(*) FROM silver.meter_staging").scalar()
        reasons = dict(run(conn, """
            SELECT coalesce(reject_reason, 'clean'), count(*)
            FROM silver.meter_staging GROUP BY 1""").all())
        dupes = run(conn, """
            SELECT count(*) FROM (
                SELECT 1 FROM silver.meter_staging
                WHERE reject_reason IS NULL
                GROUP BY meter_id, ts_end HAVING count(*) > 1) d""").scalar()
    print_reasons(reasons)

    clean = reasons.get("clean", 0)
    quarantined = staged_rows - clean
    # pre_service_zero and dst_* are expected, by-design exclusions.
    # Only these reasons count as unexpected data-quality failures:
    unexpected = sum(reasons.get(r, 0) for r in
                     ("bad_timestamp", "bad_value", "negative_value", "above_ceiling"))
    problems = []
    if staged_rows != bronze_rows:
        problems.append(f"row count mismatch: bronze {bronze_rows:,} vs staged {staged_rows:,}")
    if dupes > 0:
        problems.append(f"{dupes:,} duplicate (meter, timestamp) pairs")
    if unexpected / bronze_rows > MAX_BAD_SHARE:
        problems.append(f"{unexpected:,} unexpected rejects exceed the {MAX_BAD_SHARE:.0%} gate")
    if problems:
        log_run("meters", bronze_rows, clean, quarantined, "FAILED")
        sys.exit("METER AUDIT FAILED, nothing published:\n  " + "\n  ".join(problems))

    print("=== METERS: publish ===", flush=True)
    with engine.begin() as conn:
        for sql in METER_PUBLISH:
            run(conn, sql)
    log_run("meters", bronze_rows, clean, quarantined, "PUBLISHED")
    print(f"  published {clean:,} rows, quarantined {quarantined:,}", flush=True)


if __name__ == "__main__":
    process_weather()
    process_meters()
    print("\nSilver layer published.", flush=True)