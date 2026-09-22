import os
import sys

import numpy as np


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "build"))
import dtfx_engine


DAY = int(86400e9)
YEAR = 365.25 * DAY


def run_one(*, risk, cap_fraction, unit_r, offsets, ratios, target):
    width = len(unit_r)
    return dtfx_engine.run_portfolio_mc(
        np.array([0], np.int64), np.array([10 * DAY], np.int64),
        np.array([0], np.int32), np.array([risk], np.float64),
        np.array([cap_fraction], np.float64), np.array([width], np.int32),
        np.array([unit_r], np.float64), np.array([offsets], np.int64),
        np.array([ratios], np.float64), n_accounts=1, n_sims=1, block=1,
        max_trades=1, seed=1, start_per_account=200.0 if width == 1 else 100.0,
        cap=500000.0, target=target, leg_floor=1.0, total_floor=1.0,
        year_ns=YEAR, boundary_gap_ns=DAY, n_threads=1,
    )


def test_profit_is_realized_at_exit_and_tail_is_drained():
    out = run_one(risk=0.5, cap_fraction=1.0, unit_r=[1.0], offsets=[0], ratios=[1.0], target=300.0)
    assert 9.9 < out[0][0] * 365.25 < 10.1


def test_pyramid_addition_requires_incremental_margin():
    out = run_one(risk=0.2, cap_fraction=0.25, unit_r=[1.0, 10.0],
                  offsets=[0, DAY], ratios=[1.0, 1.0], target=300.0)
    assert out[0][0] == -1.0
    assert out[5][0] == 1

