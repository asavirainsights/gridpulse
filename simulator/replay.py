import os
import sys
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "optimizer"))
from dispatch import (plan_day, check_plan, tariff, load_forecasts,
                      E_MAX_KWH, P_MAX_KW, ROUND_TRIP_EFF, SOC_MIN_FRAC,
                      SOC_START_FRAC, DEMAND_CHARGE_EUR_PER_KW_DAY,
                      DEGRADATION_EUR_PER_KWH)

load_dotenv()
engine = create_engine(os.environ["DATABASE_URL"])
pd.set_option("display.width", 200)

# ---- forecasts plus the naive forecasts used as a comparison -----------------
fc = load_forecasts(engine)
feat = (pd.read_sql(text("""SELECT hour_utc, lag_24, lag_48, lag_168, pv_kwh AS pv_actual
                            FROM gold.features"""),
                    engine, parse_dates=["hour_utc"]).set_index("hour_utc").sort_index())
feat["naive_dem"] = (feat["lag_24"] + feat["lag_48"] + feat["lag_168"]) / 3
pv = feat["pv_actual"]
feat["naive_pv"] = (pv.shift(24) + pv.shift(48) + pv.shift(72)) / 3
fc = fc.join(feat[["naive_dem", "naive_pv"]], how="left")
test = fc[fc["split"] == "test"].dropna(subset=["naive_dem", "naive_pv"])
local_hour = test.index.tz_localize("UTC").tz_convert("Europe/Lisbon").hour.to_numpy()


# ---- the simple rule: charge at night, discharge at the evening peak ---------
def run_simple_rule(dem, sol, hour):
    eta = ROUND_TRIP_EFF ** 0.5
    s_min = SOC_MIN_FRAC * E_MAX_KWH
    soc = SOC_START_FRAC * E_MAX_KWH      # carried from day to day, never reset
    c, d = np.zeros(len(dem)), np.zeros(len(dem))
    for t in range(len(dem)):
        net = max(dem[t] - sol[t], 0.0)
        if 0 <= hour[t] < 8:
            c[t] = max(0.0, min(P_MAX_KW, (E_MAX_KWH - soc) / eta))
            soc += eta * c[t]
        elif 18 <= hour[t] < 21:
            d[t] = max(0.0, min(P_MAX_KW, (soc - s_min) * eta, net))
            soc -= d[t] / eta
    return c, d


rc, rd = run_simple_rule(test["demand_kwh"].to_numpy(), test["pv_kwh"].to_numpy(), local_hour)
rule_c = pd.Series(rc, index=test.index)
rule_d = pd.Series(rd, index=test.index)

# ---- replay every test day -----------------------------------------------------
rows = []
days = sorted(set(test.index.date))
for i, day in enumerate(days, 1):
    d = test.loc[str(day)]
    if len(d) != 24:
        continue
    price = tariff(d.index)
    dem = d["demand_kwh"].to_numpy()     # what REALLY happened
    sol = d["pv_kwh"].to_numpy()
    zeros = np.zeros(24)

    def record(policy, charge, discharge):
        grid = np.maximum(dem - sol + charge - discharge, 0.0)   # surplus solar is wasted
        rows.append({
            "day": day, "policy": policy,
            "energy_eur": float((price * grid).sum()),
            "peak_eur": DEMAND_CHARGE_EUR_PER_KW_DAY * float(grid.max()),
            "wear_eur": DEGRADATION_EUR_PER_KWH * float((charge + discharge).sum()),
            "peak_kw": float(grid.max()),
            "discharged_kwh": float(discharge.sum()),
        })

    record("1 no battery", zeros, zeros)
    record("2 simple rule", rule_c.loc[d.index].to_numpy(), rule_d.loc[d.index].to_numpy())

    plans = {
        "3 optimizer + naive forecast": (d["naive_dem"], d["naive_pv"]),
        "4 optimizer + ML forecast (P50)": (d["q50_dem"], d["q50_pv"]),
        "5 optimizer + ML cautious (P90/P10)": (d["q90_dem"], d["q10_pv"]),
        "6 optimizer + perfect foresight": (d["demand_kwh"], d["pv_kwh"]),
    }
    for name, (fd, fs) in plans.items():
        plan = plan_day(fd.to_numpy(), fs.to_numpy(), price)
        check_plan(plan)
        record(name, plan["charge_kwh"].to_numpy(), plan["discharge_kwh"].to_numpy())

    if i % 20 == 0 or i == len(days):
        print(f"replayed {i}/{len(days)} days", flush=True)

# ---- results ---------------------------------------------------------------------
res = pd.DataFrame(rows)
res["total_eur"] = res["energy_eur"] + res["peak_eur"] + res["wear_eur"]
n_days = res["day"].nunique()
summary = res.groupby("policy").agg(
    total_cost_eur=("total_eur", "sum"),
    energy_eur=("energy_eur", "sum"),
    peak_charge_eur=("peak_eur", "sum"),
    wear_eur=("wear_eur", "sum"),
    avg_daily_peak_kw=("peak_kw", "mean"),
    discharged_kwh=("discharged_kwh", "sum"),
)
base = summary.loc["1 no battery"]
summary["cost_saved_%"] = 100 * (1 - summary["total_cost_eur"] / base["total_cost_eur"])
summary["peak_cut_%"] = 100 * (1 - summary["avg_daily_peak_kw"] / base["avg_daily_peak_kw"])
summary["cycles_per_day"] = summary["discharged_kwh"] / E_MAX_KWH / n_days

print(f"\n=== Replay of {n_days} test days, July to December 2014 (real bills) ===")
cols = ["total_cost_eur", "cost_saved_%", "avg_daily_peak_kw", "peak_cut_%", "cycles_per_day"]
print(summary[cols].round(1).to_string())
print("\ncost breakdown (EUR):")
print(summary[["energy_eur", "peak_charge_eur", "wear_eur"]].round(0).to_string())

daily = res.pivot(index="day", columns="policy", values="total_eur")
ml, rule, naive = ("4 optimizer + ML forecast (P50)", "2 simple rule",
                   "3 optimizer + naive forecast")
print(f"\nML optimizer cheaper than the simple rule on {100 * (daily[ml] < daily[rule]).mean():.0f}% of days")
print(f"ML optimizer cheaper than the naive-forecast optimizer on {100 * (daily[ml] < daily[naive]).mean():.0f}% of days")

os.makedirs("docs", exist_ok=True)
summary.round(2).to_csv(os.path.join("docs", "simulation_results.csv"))
res.to_sql("simulation_daily", engine, schema="gold", if_exists="replace",
           index=False, method="multi", chunksize=500)
print("\nsaved docs/simulation_results.csv and gold.simulation_daily")