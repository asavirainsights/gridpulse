import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "optimizer"))

from dispatch import plan_day, check_plan, tariff, day_cost  # noqa: E402


def two_price_day():
    price = np.full(24, 0.15)
    price[:8] = 0.09
    price[18:21] = 0.28
    return price


def test_plan_obeys_battery_physics():
    rng = np.random.default_rng(0)
    price = two_price_day()
    for _ in range(5):
        demand = 700 + 250 * rng.random(24)
        solar = rng.random(24) * 100
        plan = plan_day(demand, solar, price)
        check_plan(plan)          # raises if any battery limit is broken


def test_battery_never_makes_the_bill_worse():
    rng = np.random.default_rng(1)
    price = two_price_day()
    for _ in range(5):
        demand = 600 + 400 * rng.random(24)
        solar = rng.random(24) * 80
        plan = plan_day(demand, solar, price)
        with_battery = day_cost(plan["grid_kwh"].to_numpy(), price)
        without = day_cost(np.maximum(demand - solar, 0), price)
        assert with_battery <= without + 1e-3


def test_flat_prices_and_flat_demand_mean_no_cycling():
    plan = plan_day(np.full(24, 800.0), np.zeros(24), np.full(24, 0.15))
    assert plan["charge_kwh"].sum() + plan["discharge_kwh"].sum() < 1e-3


def test_peak_shaving_lowers_the_highest_hour():
    demand = np.full(24, 600.0)
    demand[19] = 1500.0
    plan = plan_day(demand, np.zeros(24), np.full(24, 0.15))
    assert plan["grid_kwh"].max() < 1400.0


def test_arbitrage_charges_cheap_and_discharges_expensive():
    plan = plan_day(np.full(24, 800.0), np.zeros(24), two_price_day())
    assert plan["charge_kwh"].iloc[:8].sum() > 100
    assert plan["discharge_kwh"].iloc[18:21].sum() > 100


def test_tariff_uses_local_lisbon_time():
    summer = tariff(pd.date_range("2014-07-15", periods=24, freq="h"))
    assert summer[17] == 0.28      # 17:00 UTC is 18:00 in Lisbon in summer
    assert summer[20] == 0.15      # 21:00 local is back to the standard price
    assert summer[23] == 0.09      # 23:00 UTC is midnight local
    winter = tariff(pd.date_range("2014-12-10", periods=24, freq="h"))
    assert winter[18] == 0.28
    assert winter[7] == 0.09
    assert winter[8] == 0.15


def test_check_plan_rejects_an_impossible_plan():
    bad = pd.DataFrame({"charge_kwh": [0.0], "discharge_kwh": [500.0],
                        "soc_kwh": [100.0], "grid_kwh": [0.0]})
    with pytest.raises(AssertionError):
        check_plan(bad)


def test_day_cost_adds_energy_and_peak_charge():
    # 0.1*100 + 0.2*200 + 0.5*200 = 150
    assert day_cost([100, 200], [0.1, 0.2], demand_charge=0.5) == pytest.approx(150.0)