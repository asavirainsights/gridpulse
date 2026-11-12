# GridPulse

![CI](https://github.com/asavirainsights/gridpulse/actions/workflows/ci.yml/badge.svg)

Probabilistic demand and solar forecasts that drive a battery scheduler, tested by replaying 184 real days the models never saw.

![Architecture](docs/images/architecture.png)

## Headline results

Replay of July to December 2014 for 10 customers from a public Portuguese dataset. The battery size and tariff are assumed (see [Assumptions and limitations](#assumptions-and-limitations)).

- The forecast-driven battery plan cut the electricity bill by **5.2%** (about EUR 28,500 over 184 days) and the average daily peak by **8.7%**.
- A simple "charge at night, discharge in the evening" rule saved 4.8% but made the daily peak **4.7% worse**. The optimizer was cheaper than the rule on 70% of days.
- The demand forecast has **25% lower error** than the best naive baseline (MAE 32.9 vs 43.9 kWh on the test period) and **31% lower** error in the evening peak. Its P10 to P90 band covered 78.4% of real hours (target: 80%).
- A surprising finding: inside the scheduler, the ML forecast saved only about EUR 100 more than a naive 3-day-average forecast. For this site, battery size and tariff matter far more than forecast accuracy, so forecast value is measured in euros, not only in error scores.

## The problem

Electricity costs more in the evening peak, and large customers also pay for their single highest hour. A battery can shift energy and shave peaks, but only if it knows tomorrow's demand and solar output, and how uncertain they are. GridPulse forecasts both with uncertainty, turns the forecasts into an hourly battery schedule, and measures the result in euros on days the models never saw.

## Data

| Source | Use |
|---|---|
| UCI ElectricityLoadDiagrams20112014 | 15-minute readings from 370 customers in Portugal; the first 10 meters are used |
| Open-Meteo archive API | Hourly temperature, cloud cover and shortwave radiation for Lisbon |
| Simulated solar | 500 kWp system with an 80% performance ratio, derived from measured radiation (the dataset has no solar data) |

## Data pipeline: Bronze, Silver, Gold

- **Bronze:** data exactly as received, as text, with source and ingestion time.
- **Silver:** typed and validated with Write-Audit-Publish. Rows are staged with a reject reason, audited (row counts must reconcile, no duplicates, unexpected rejects under 1%), then published. Failed rows go to a quarantine schema instead of disappearing.
- **What Silver found** in 1,402,560 meter rows: no nulls, negatives or duplicates, but 350,400 rows (all of 2011) were zeros from customers not yet connected, plus 240 clock-change rows (120 in the spring gap hour, 120 in the autumn repeated hour). 1,051,920 clean rows remain, and the counts reconcile exactly with Bronze.
- **Gold:** hourly demand (sum of the 10 meters, complete hours only), 6 clock-change hours filled by interpolation, weather joined, simulated solar added, and leak-free features. 26,112 rows from 2012-01-09 to 2014-12-31.
- **Time handling:** meter timestamps are Lisbon local time and weather is UTC, so Silver stores both. Calendar features use local time.
- **Split by time, never shuffled:** train 2012 to 2013 (17,352 hours), validation January to June 2014 (4,344), test July to December 2014 (4,416).
- **No leakage:** every demand feature is at least 24 hours old (lags of 24, 48 and 168 hours, rolling means ending 24 hours before the target, day-over-day trend, yesterday's ramp).

## Forecasting

- Three LightGBM quantile models (P10, P50, P90) for demand and for solar. Bands are calibrated with a conformal correction fitted on the validation period and then checked on the untouched test period.
- Demand variants: **A** no weather, **B** perfect weather (an upper bound), **C** weather with synthetic forecast noise. The champion is chosen between A and C by validation pinball loss. Weather matters little for these customers: last week's and yesterday's demand carry the model.
- Solar uses only information known 24 hours ahead, trains on daylight hours, and forces night forecasts to zero. Results: `docs/solar_results.txt`.

| Demand, test period | Best baseline (3-day average) | Champion model |
|---|---|---|
| MAE (kWh per hour) | 43.9 | 32.9 |
| Evening-peak MAE | 65.0 | 44.9 |
| Hours under-forecast by more than 10% | 7.0% | 3.2% |
| P10 to P90 coverage (target 80%) | n/a | 78.4% |

| Solar, test period (daylight hours) | Best baseline (3-day average) | LightGBM |
|---|---|---|
| MAE (kWh per hour) | 21.90 | 20.68 |
| P10 to P90 coverage (target 80%) | n/a | 83.2% after calibration (73.8% raw) |

## Battery scheduling

A linear program (PuLP with CBC) plans each 24-hour window. It minimizes energy cost plus a daily peak charge plus a small wear cost, subject to battery capacity, reserve, power limit and energy balance, and the battery must end the day at least as full as it started. Surplus solar earns nothing.

## Replay simulation

Plans are made from forecasts, then judged on what really happened, over 184 test days.

| Policy | Bill (EUR) | Saved | Avg daily peak (kW) | Peak change | Cycles per day |
|---|---|---|---|---|---|
| 1. No battery | 546,426 | n/a | 1,079 | n/a | n/a |
| 2. Simple rule | 520,131 | 4.8% | 1,130 | +4.7% | 0.85 |
| 3. Optimizer + naive forecast | 518,005 | 5.2% | 989 | -8.4% | 0.70 |
| 4. Optimizer + ML forecast (P50) | 517,903 | 5.2% | 985 | -8.7% | 0.69 |
| 5. Optimizer + cautious ML (P90 demand, P10 solar) | 517,635 | 5.3% | 987 | -8.5% | 0.71 |
| 6. Optimizer + perfect foresight (upper bound) | 515,091 | 5.7% | 949 | -12.1% | 0.70 |

Perfect foresight is the cheapest, as it must be. The cautious plan is 0.05% cheaper than the P50 plan, which is within noise.

## MLOps

- **MLflow:** experiments for demand (three variants) and solar; champion models are registered with a `champion` alias.
- **Drift monitoring** (`monitoring/drift_check.py`): each 2014 month is compared with the same month in the training years using PSI on demand inputs, the forecast-error ratio and band coverage. It gives an OK, WATCH or RETRAIN verdict, and `--retrain` retrains if the latest month is flagged.
  - 2 of 12 months were flagged. July: coverage fell to 67.5% (bands too narrow). October: demand PSI was 0.27 but the error was normal (a false alarm).
  - Temperature drifted strongly (PSI up to 1.26) without hurting accuracy.
  - A synthetic +25% demand shock triggers retraining on all three criteria.
  - Lesson: input drift alone should not trigger retraining. The 0.10 PSI threshold flagged 10 of 12 months, which is too sensitive for monthly samples.
- **Tests and CI:** 11 pytest tests (battery physics, the battery never worsens the bill, peak shaving, local-time tariff, PSI) run on every push with GitHub Actions.
- **Dashboard** (Streamlit): results, forecast bands for any test day, a battery what-if panel (size, power, prices), and data quality.

![Results](docs/images/dashboard_results.png)
![Forecast](docs/images/dashboard_forecast.png)
![What-if](docs/images/dashboard_whatif.png)
![Data quality](docs/images/dashboard_quality.png)
![MLflow models](docs/images/mlflow_models.png)

## Assumptions and limitations

- Battery (1,200 kWh, 400 kW, 90% round trip, 10% reserve), tariff (EUR 0.09 / 0.15 / 0.28 by local hour), peak charge (EUR 0.40 per kW per day) and wear cost are assumptions. The savings hold only for them.
- Solar is simulated from measured radiation, not metered.
- Variant C uses synthetic noise on observed weather, not archived weather forecasts.
- Only 10 of 370 customers and a six-month test period.
- Each plan covers one 24-hour UTC window and starts at 50% charge (the simple rule carries its charge between days). This is not a rolling-horizon controller.
- Historical replay only. There is no live deployment or hardware.
- Spring clock-change zeros were confirmed in the data. The autumn repeated hour follows the dataset documentation and was not independently verified. Both are quarantined, and 6 hours are interpolated.
- Retraining reruns the same fixed training window.
- Developed against a local PostgreSQL install. The pipeline is not containerized or scheduled.

## Run it yourself

Requires Python 3.11 and PostgreSQL 16.

```bash
pip install -r requirements.txt
psql -U postgres -c "CREATE DATABASE gridpulse;"
psql -U postgres -d gridpulse -c "CREATE SCHEMA bronze; CREATE SCHEMA silver; CREATE SCHEMA gold; CREATE SCHEMA quarantine;"
```

Create a `.env` file:

```
DATABASE_URL=postgresql+psycopg2://USER:PASSWORD@localhost:5432/gridpulse
DATA_DIR=/path/to/data
```

Download `LD2011_2014.txt` from the UCI repository into `DATA_DIR`, then run:

```bash
python pipelines/ingest_weather.py
python pipelines/ingest_meters.py
python pipelines/bronze_to_silver.py
python pipelines/silver_to_gold.py
python models/baselines.py
python models/train_quantile.py
python models/train_solar.py
python simulator/replay.py
python monitoring/drift_check.py
streamlit run dashboard/app.py
```

Optional: `mlflow ui --backend-store-uri sqlite:///mlflow.db` and `pytest`.

## Repository layout

```
pipelines/   ingestion, Bronze to Silver (Write-Audit-Publish), Silver to Gold, profiling
models/      baselines, quantile demand model, solar model
optimizer/   battery dispatch (linear program)
simulator/   replay of real days under six policies
monitoring/  drift checks and retrain rule
dashboard/   Streamlit app
tests/       pytest suite
docs/        result files and images
.github/     CI workflow
docker-compose.yml   PostgreSQL service definition (development used a local install)
```

## Future work

- FastAPI service for forecasts and schedules
- Containerized deployment and scheduled orchestration
- Rolling-horizon planning, real tariffs and more customers
- Retraining on a moving window, and a rule that requires worse accuracy before retraining