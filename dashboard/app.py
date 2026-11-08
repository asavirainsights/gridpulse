import os
import sys
from datetime import date

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "optimizer"))
from dispatch import (plan_day, load_forecasts, E_MAX_KWH, P_MAX_KW,
                      DEMAND_CHARGE_EUR_PER_KW_DAY, DEGRADATION_EUR_PER_KWH)

load_dotenv(os.path.join(ROOT, ".env"))
st.set_page_config(page_title="GridPulse", layout="wide")


# ---- data access ---------------------------------------------------------------
@st.cache_resource
def get_engine():
    return create_engine(os.environ["DATABASE_URL"])


@st.cache_data
def get_test_forecasts():
    fc = load_forecasts(get_engine())
    return fc[fc["split"] == "test"]


@st.cache_data
def get_results():
    return pd.read_csv(os.path.join(ROOT, "docs", "simulation_results.csv"))


@st.cache_data
def get_quality():
    eng = get_engine()
    rej = pd.read_sql(text("""SELECT reject_reason, count(*) AS n_rows
                              FROM quarantine.meter_rejects GROUP BY 1 ORDER BY 2 DESC"""), eng)
    runs = pd.read_sql(text("""SELECT run_ts, pipeline, rows_in, rows_clean,
                                      rows_quarantined, status
                               FROM silver.pipeline_runs ORDER BY run_ts DESC LIMIT 10"""), eng)
    return rej, runs


# ---- helpers -------------------------------------------------------------------
def local_time(index_utc):
    return index_utc.tz_localize("UTC").tz_convert("Europe/Lisbon").tz_localize(None)


def band_chart(d, actual, q10, q50, q90, unit):
    df = pd.DataFrame({
        "time": local_time(d.index),
        "actual": d[actual].to_numpy(),
        "low": d[q10].to_numpy(),
        "mid": d[q50].to_numpy(),
        "high": d[q90].to_numpy(),
    })
    base = alt.Chart(df).encode(x=alt.X("time:T", title="Local time"))
    band = base.mark_area(opacity=0.25, color="#2a9d8f").encode(
        y=alt.Y("low:Q", title=unit), y2=alt.Y2("high"))
    mid = base.mark_line(color="#2a9d8f").encode(y="mid:Q")
    act = base.mark_line(color="#d62728", strokeDash=[4, 3]).encode(y="actual:Q")
    return (band + mid + act).properties(height=280)


def make_price(index_utc, off, std, peak):
    hour = index_utc.tz_localize("UTC").tz_convert("Europe/Lisbon").hour
    price = np.full(len(hour), float(std))
    price[(hour >= 0) & (hour < 8)] = off
    price[(hour >= 18) & (hour < 21)] = peak
    return price


def bill(grid, price, demand_charge, wear=0.0):
    return float((price * grid).sum() + demand_charge * grid.max() + wear)


def run_day(d, mode, e_max, p_max, price, demand_charge):
    """Plan from forecasts, then judge the plan on what really happened."""
    dem, sol = d["demand_kwh"].to_numpy(), d["pv_kwh"].to_numpy()
    base_grid = np.maximum(dem - sol, 0.0)
    base_cost = bill(base_grid, price, demand_charge)
    zeros = np.zeros(len(dem))
    if e_max <= 0:
        return dict(base_cost=base_cost, cost=base_cost, base_grid=base_grid,
                    grid=base_grid, charge=zeros, discharge=zeros, soc=zeros)
    if mode.startswith("ML forecast"):
        fd, fs = d["q50_dem"], d["q50_pv"]
    elif mode.startswith("ML cautious"):
        fd, fs = d["q90_dem"], d["q10_pv"]
    else:
        fd, fs = d["demand_kwh"], d["pv_kwh"]
    plan = plan_day(fd.to_numpy(), fs.to_numpy(), price, e_max=e_max, p_max=p_max,
                    demand_charge=demand_charge)
    c = plan["charge_kwh"].to_numpy()
    dis = plan["discharge_kwh"].to_numpy()
    grid = np.maximum(dem - sol + c - dis, 0.0)
    wear = DEGRADATION_EUR_PER_KWH * float((c + dis).sum())
    return dict(base_cost=base_cost, cost=bill(grid, price, demand_charge, wear),
                base_grid=base_grid, grid=grid, charge=c, discharge=dis,
                soc=plan["soc_kwh"].to_numpy())


# ---- page ----------------------------------------------------------------------
st.title("GridPulse")
st.caption("Probabilistic demand and solar forecasts feeding a battery scheduler, "
           "tested on real days the models never saw.")

test = get_test_forecasts()
days = sorted(set(test.index.date))
day = st.sidebar.date_input("Day to inspect (test period)", value=date(2014, 12, 10),
                            min_value=days[0], max_value=days[-1])
st.sidebar.caption("Days are 24-hour UTC windows from July to December 2014. "
                   "The models never saw these days in training.")
d = test.loc[str(day)]
if len(d) != 24:
    st.error("This day has fewer than 24 hours of data. Pick another day.")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(["Results", "Forecasts", "Battery what-if", "Data quality"])

with tab1:
    res = get_results().set_index("policy")
    base = res.loc["1 no battery"]
    ml = res.loc["4 optimizer + ML forecast (P50)"]
    saved = base["total_cost_eur"] - ml["total_cost_eur"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Bill saved over 184 test days", f"EUR {saved:,.0f}", f"{ml['cost_saved_%']:.1f}%")
    c2.metric("Average daily peak cut", f"{ml['peak_cut_%']:.1f}%")
    c3.metric("Battery cycles per day", f"{ml['cycles_per_day']:.2f}")
    left, right = st.columns(2)
    left.subheader("Bill saved vs no battery (%)")
    left.bar_chart(res["cost_saved_%"])
    right.subheader("Average daily peak (kW)")
    right.bar_chart(res["avg_daily_peak_kw"])
    st.dataframe(res.round(2))
    with st.expander("Assumptions behind these numbers"):
        st.markdown(
            "- 10 customers from a public Portuguese dataset, July to December 2014 (184 days replayed)\n"
            "- Battery of 1,200 kWh and 400 kW with 90% round-trip efficiency\n"
            "- Tariff and peak charge are assumed, not real contracts\n"
            "- Solar is simulated from measured sunlight (500 kWp installation)\n"
            "- Plans are made from forecasts and judged on what really happened")

with tab2:
    inside = ((d["demand_kwh"] >= d["q10_dem"]) & (d["demand_kwh"] <= d["q90_dem"])).sum()
    st.subheader(f"Demand forecast, {day}")
    st.altair_chart(band_chart(d, "demand_kwh", "q10_dem", "q50_dem", "q90_dem", "kWh per hour"))
    st.caption(f"Green line = forecast, shaded = P10 to P90 range, red dashed = what really happened. "
               f"Actual demand fell inside the range in {inside} of 24 hours.")
    st.subheader(f"Solar forecast, {day}")
    st.altair_chart(band_chart(d, "pv_kwh", "q10_pv", "q50_pv", "q90_pv", "kWh per hour"))

with tab3:
    st.write("Change the battery or the tariff and see what the plan does to this day's bill. "
             "The plan is made from forecasts, then judged on what really happened.")
    c1, c2, c3 = st.columns(3)
    mode = c1.selectbox("Plan using", ["ML forecast (P50)",
                                       "ML cautious (P90 demand, P10 solar)",
                                       "Perfect foresight (upper bound)"])
    e_max = c1.slider("Battery size (kWh)", 0, 4800, int(E_MAX_KWH), 200)
    p_max = c1.slider("Max power (kW)", 100, 1600, int(P_MAX_KW), 50)
    off = c2.slider("Off-peak price, 00:00 to 08:00 (EUR/kWh)", 0.05, 0.20, 0.09, 0.01)
    std = c2.slider("Standard price (EUR/kWh)", 0.08, 0.30, 0.15, 0.01)
    peak = c3.slider("Evening price, 18:00 to 21:00 (EUR/kWh)", 0.15, 0.60, 0.28, 0.01)
    charge = c3.slider("Peak charge (EUR per kW per day)", 0.0, 1.0,
                       float(DEMAND_CHARGE_EUR_PER_KW_DAY), 0.05)

    price = make_price(d.index, off, std, peak)
    r = run_day(d, mode, e_max, p_max, price, charge)
    saving = r["base_cost"] - r["cost"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Bill without battery", f"EUR {r['base_cost']:,.0f}")
    m2.metric("Bill with battery", f"EUR {r['cost']:,.0f}")
    m3.metric("Saving", f"EUR {saving:,.0f}", f"{100 * saving / r['base_cost']:.1f}%")
    m4.metric("Peak grid import", f"{r['grid'].max():,.0f} kWh/h",
              f"{r['grid'].max() - r['base_grid'].max():,.0f}", delta_color="inverse")

    t = local_time(d.index)
    st.line_chart(pd.DataFrame({"Grid import, no battery": r["base_grid"],
                                "Grid import, with battery": r["grid"],
                                "Battery level": r["soc"]}, index=t))
    st.bar_chart(pd.DataFrame({"Charge": r["charge"], "Discharge": -r["discharge"]}, index=t))

    st.subheader("What if the battery were bigger?")
    st.caption("Same day and prices; power is set to one third of the battery size.")
    sizes = [0, 600, 1200, 2400, 3600]
    saved_pct = []
    for size in sizes:
        rr = run_day(d, mode, size, max(size / 3, 1.0), price, charge)
        saved_pct.append(100 * (rr["base_cost"] - rr["cost"]) / rr["base_cost"])
    study = pd.DataFrame({"size": sizes, "saved": saved_pct})
    st.altair_chart(alt.Chart(study).mark_bar(color="#2a9d8f").encode(
        x=alt.X("size:O", sort=None, title="Battery size (kWh)"),
        y=alt.Y("saved:Q", title="Bill saved (%)")).properties(height=260))

with tab4:
    rej, runs = get_quality()
    st.subheader("Rows quarantined by the Silver layer")
    st.bar_chart(rej.set_index("reject_reason")["n_rows"])
    st.caption("Most quarantined rows are 2011 readings of zero from customers not yet connected. "
               "The rest are clock-change hours where the source data is wrong.")
    st.subheader("Recent pipeline runs")
    st.dataframe(runs)