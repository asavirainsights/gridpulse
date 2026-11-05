import os
import numpy as np
import pandas as pd
import pulp
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# ---- ASSUMPTIONS (document these in the README) ------------------------------
E_MAX_KWH = 1200           # battery energy capacity
P_MAX_KW = 400             # maximum charge / discharge power
ROUND_TRIP_EFF = 0.90      # one-way efficiency is the square root of this
SOC_MIN_FRAC = 0.10        # reserve that is never used
SOC_START_FRAC = 0.50      # each planning day starts at 50 percent
DEMAND_CHARGE_EUR_PER_KW_DAY = 0.40
DEGRADATION_EUR_PER_KWH = 0.01


def tariff(index_utc):
    """Price in EUR/kWh for each UTC hour, based on the LOCAL clock hour."""
    hour = index_utc.tz_localize("UTC").tz_convert("Europe/Lisbon").hour
    price = np.full(len(hour), 0.15)
    price[(hour >= 0) & (hour < 8)] = 0.09
    price[(hour >= 18) & (hour < 21)] = 0.28
    return price


def day_cost(grid_kwh, price, demand_charge=DEMAND_CHARGE_EUR_PER_KW_DAY):
    """Electricity bill for one planning day: energy cost plus the peak charge."""
    grid = np.asarray(grid_kwh, dtype=float)
    return float((np.asarray(price) * grid).sum() + demand_charge * grid.max())


def plan_day(demand_kwh, solar_kwh, price, e_max=E_MAX_KWH, p_max=P_MAX_KW,
             eff=ROUND_TRIP_EFF, soc_min_frac=SOC_MIN_FRAC,
             soc_start_frac=SOC_START_FRAC,
             demand_charge=DEMAND_CHARGE_EUR_PER_KW_DAY,
             degradation=DEGRADATION_EUR_PER_KWH):
    """Best charge/discharge plan for one window of consecutive hours.

    Inputs are per-hour energy in kWh, which equals average kW over that hour.
    Surplus solar that cannot be used or stored is curtailed (no export revenue).
    """
    T = len(demand_kwh)
    eta = eff ** 0.5
    s0 = soc_start_frac * e_max
    s_min = soc_min_frac * e_max

    prob = pulp.LpProblem("battery_dispatch", pulp.LpMinimize)
    c = pulp.LpVariable.dicts("charge", range(T), 0, p_max)
    d = pulp.LpVariable.dicts("discharge", range(T), 0, p_max)
    s = pulp.LpVariable.dicts("soc", range(T), s_min, e_max)
    g = pulp.LpVariable.dicts("grid", range(T), 0)
    peak = pulp.LpVariable("peak", 0)

    # objective: energy bill + peak charge + a small wear cost on battery use
    prob += (pulp.lpSum(float(price[t]) * g[t] for t in range(T))
             + demand_charge * peak
             + degradation * pulp.lpSum(c[t] + d[t] for t in range(T)))

    for t in range(T):
        prev = s0 if t == 0 else s[t - 1]
        prob += s[t] == prev + eta * c[t] - d[t] / eta                 # battery energy balance
        prob += g[t] >= float(demand_kwh[t]) - float(solar_kwh[t]) + c[t] - d[t]
        prob += peak >= g[t]                                           # peak = highest grid import
    prob += s[T - 1] >= s0                                             # do not drain the battery

    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"optimizer failed: {pulp.LpStatus[status]}")

    val = lambda v: float(pulp.value(v) or 0.0)
    return pd.DataFrame({
        "charge_kwh": [val(c[t]) for t in range(T)],
        "discharge_kwh": [val(d[t]) for t in range(T)],
        "soc_kwh": [val(s[t]) for t in range(T)],
        "grid_kwh": [val(g[t]) for t in range(T)],
    })


def check_plan(plan, e_max=E_MAX_KWH, p_max=P_MAX_KW, eff=ROUND_TRIP_EFF,
               soc_min_frac=SOC_MIN_FRAC, soc_start_frac=SOC_START_FRAC):
    """Safety checks: a plan must obey the physics of the battery."""
    eta = eff ** 0.5
    s0 = soc_start_frac * e_max
    tol = 1e-3
    soc = plan["soc_kwh"].to_numpy()
    c = plan["charge_kwh"].to_numpy()
    d = plan["discharge_kwh"].to_numpy()
    prev = np.concatenate([[s0], soc[:-1]])
    assert (soc >= soc_min_frac * e_max - tol).all(), "battery went below its reserve"
    assert (soc <= e_max + tol).all(), "battery went above its capacity"
    assert (c <= p_max + tol).all() and (d <= p_max + tol).all(), "power limit exceeded"
    assert np.allclose(soc, prev + eta * c - d / eta, atol=tol), "energy balance broken"
    assert soc[-1] >= s0 - tol, "battery ends the day lower than it started"


def load_forecasts(engine):
    dem = pd.read_sql(text("""SELECT hour_utc, split, demand_kwh, q10, q50, q90
                              FROM gold.demand_forecast"""),
                      engine, parse_dates=["hour_utc"]).set_index("hour_utc")
    sol = pd.read_sql(text("""SELECT hour_utc, pv_kwh, q10, q50, q90
                              FROM gold.solar_forecast"""),
                      engine, parse_dates=["hour_utc"]).set_index("hour_utc")
    return dem.join(sol, lsuffix="_dem", rsuffix="_pv", how="inner").sort_index()


if __name__ == "__main__":
    load_dotenv()
    engine = create_engine(os.environ["DATABASE_URL"])
    fc = load_forecasts(engine)
    pd.set_option("display.width", 200)

    for day_str in ("2014-07-15", "2014-12-10"):          # a summer day and a winter day
        day = fc.loc[day_str]
        price = tariff(day.index)
        dem = day["q50_dem"].to_numpy()
        sol = day["q50_pv"].to_numpy()

        plan = plan_day(dem, sol, price)
        check_plan(plan)

        base_grid = np.maximum(dem - sol, 0)
        cost_no = day_cost(base_grid, price)
        cost_bat = day_cost(plan["grid_kwh"].to_numpy(), price)

        table = pd.DataFrame({
            "hour_local": day.index.tz_localize("UTC").tz_convert("Europe/Lisbon").hour,
            "price": price, "demand": dem, "solar": sol,
            "charge": plan["charge_kwh"].to_numpy(),
            "discharge": plan["discharge_kwh"].to_numpy(),
            "battery": plan["soc_kwh"].to_numpy(),
            "grid": plan["grid_kwh"].to_numpy(),
        }, index=day.index)
        print(f"\n===== {day_str} (plan built from the P50 forecasts) =====")
        print(table.round(2).to_string())
        print(f"\ncost without battery: EUR {cost_no:,.0f}")
        print(f"cost with battery:    EUR {cost_bat:,.0f}   "
              f"(saving {100 * (1 - cost_bat / cost_no):.1f}%)")
        print(f"peak grid import:     {base_grid.max():,.0f} -> "
              f"{plan['grid_kwh'].max():,.0f} kWh/h")
    print("\nall plan checks passed")