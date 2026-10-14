import os
import sys

import numpy as np


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "build"))
import dtfx_engine


DAY = int(86400e9)
HOUR = int(3600e9)
YEAR = 365.25 * DAY


def _run(boundary_gap_ns):
    entry = np.array([0, DAY], dtype=np.int64)
    exit_ = entry + HOUR
    code = np.zeros(2, dtype=np.int32)
    price = np.ones(2, dtype=np.float64)
    stop = np.ones(2, dtype=np.float64)
    r = np.ones(2, dtype=np.float64)
    side = np.array([1, 0], dtype=np.uint8)
    return dtfx_engine.run_two_account_mc(
        entry, exit_, code, price, stop, r, side,
        n_sims=1, block=2, max_trades=4, seed=7, risk_pct=1.0,
        start_per_side=100.0, max_leverage=50.0, cap=500000.0,
        target1=600.0, target2=1e12, year_ns=YEAR, gbp_usd_approx=1.36,
        ruin_floor=1.0, n_threads=1, boundary_gap_ns=boundary_gap_ns,
    )


def test_boundary_gap_is_counted_in_target_time():
    fast = _run(1)[0][0] * 365.25
    calendar = _run(10 * DAY)[0][0] * 365.25
    assert 2.0 < fast < 3.0
    assert 11.0 < calendar < 12.0
    assert calendar - fast > 8.9


def test_tail_exit_is_settled_after_trade_budget():
    entry = np.array([0, DAY], dtype=np.int64)
    exit_ = entry + HOUR
    out = dtfx_engine.run_two_account_mc(
        entry, exit_, np.zeros(2, np.int32), np.ones(2), np.ones(2), np.ones(2),
        np.array([1, 0], np.uint8), n_sims=1, block=2, max_trades=2, seed=9,
        risk_pct=1.0, start_per_side=100.0, max_leverage=50.0, cap=500000.0,
        target1=300.0, target2=1e12, year_ns=YEAR, gbp_usd_approx=1.36,
        ruin_floor=1.0, n_threads=1, boundary_gap_ns=DAY,
    )
    assert out[0][0] > 0.0
