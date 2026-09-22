"""Properly margin-aware MC: instead of pre-computing one fixed 'safe' risk_pct from a single
percentile of the stop-distance distribution (13.28, 13.34 -- both shown to be sensitive to that
arbitrary choice), track REAL currently-open positions' margin through the resampled sequence,
using each trade's actual entry_time/exit_time, and SKIP a trade if opening it would push
aggregate margin over 100% of equity -- mirroring what a real broker's margin check would do.

Bootstrap blocks are mapped onto a monotonic synthetic clock. Relative entry gaps inside each
contiguous block and every trade's real holding duration are preserved; a median inter-trade gap
joins unrelated blocks. This avoids comparing timestamps from unrelated historical years.

P&L is settled at exit, never at entry. Because the trade artifact currently records the final
pyramid-unit count but not every add timestamp, the simulator conservatively reserves margin for
all final units from entry until exit. This can reject a trade that would have been affordable if
units were added gradually, but it cannot create the prior impossible free-margin leverage.
"""
import sys
sys.path.insert(0, "/Users/brunoinzunza/Documents/GitHub/tradingResearch/dtfxStrategy/research_scripts")
sys.path.insert(0, "/Users/brunoinzunza/Documents/GitHub/tradingResearch/dtfxStrategy/research_scripts/beid_goal_hunt")
sys.path.insert(0, "/Users/brunoinzunza/Documents/GitHub/tradingResearch")
import numpy as np
import pandas as pd
import combined_fx_gold_mc as cmc
from mc_true_median import true_median_and_stats
from quant_engine.settlement import EventSettler


def simulate_path_margin_aware(account_keys, true_instruments, is_buy, stop_risk, entry_price,
                                net_r, entry_time_ns, exit_time_ns, dt_years, risk_pct,
                                instrument_list, start_per_inst=100.0, max_leverage=50.0,
                                gold_leverage=20.0, cap=500000.0, target1=1e6, leg_floor=10.0,
                                total_floor=50.0, btc_leverage=10.0, es_leverage=15.0,
                                eth_leverage=10.0, unit_count=None, exit_dt_years=None,
                                unit_rs=None, unit_add_offsets_ns=None, unit_prices=None):
    keys = [(a, s) for a in instrument_list for s in (True, False)]
    settler = EventSettler(keys, {k: start_per_inst / 2.0 for k in keys}, leg_floor=leg_floor)
    n = len(account_keys)
    if unit_count is None:
        unit_count = np.ones(n, dtype=float)
    if exit_dt_years is None:
        exit_dt_years = dt_years

    def on_settle(_key, _pnl, elapsed_years):
        total = settler.total_equity()
        if total >= target1:
            return elapsed_years, False
        if total <= total_floor or not settler.any_alive():
            return -1.0, True
        return None

    for i in range(n):
        settled = settler.settle_through(entry_time_ns[i], on_settle)
        if settled is not None:
            return settled[0], settled[1], settler.skipped
        acct = account_keys[i]; side = bool(is_buy[i])
        key = (acct, side)
        if not settler.alive[key]:
            continue
        sizing_base = settler.equity[key]
        avail = settler.available_margin(key)
        if avail <= 0:
            settler.skipped += 1
            continue
        _, risk_dollars, margin = cmc.size_and_margin(
            true_instruments[i], sizing_base, entry_price[i], stop_risk[i], risk_pct[i],
            max_leverage, gold_leverage, cap, btc_leverage=btc_leverage,
            es_leverage=es_leverage, eth_leverage=eth_leverage)
        if margin > avail:
            settler.skipped += 1
            continue  # real margin call: this trade could not have been taken, skip it
        row_rs = list(unit_rs[i]) if unit_rs is not None else [float(net_r[i])]
        row_prices = list(unit_prices[i]) if unit_prices is not None else [float(entry_price[i])]
        row_offsets = list(unit_add_offsets_ns[i]) if unit_add_offsets_ns is not None else [0]
        if not row_rs:
            row_rs = [float(net_r[i])]
        settler.open_position(
            key, entry_ns=entry_time_ns[i], exit_ns=exit_time_ns[i], margin=margin,
            risk_dollars=risk_dollars, entry_price=entry_price[i], unit_rs=row_rs,
            unit_prices=row_prices, unit_offsets_ns=row_offsets,
            exit_payload=float(exit_dt_years[i]))

    settled = settler.settle_through(np.iinfo(np.int64).max, on_settle)
    if settled is not None:
        return settled[0], settled[1], settler.skipped
    return -1.0, False, settler.skipped


def run_mc_margin_aware(trades, instrument_list, n_sims=1000, block=20, max_trades=10000, seed=1, **kw):
    trades = trades.sort_values("entry_time", kind="stable").reset_index(drop=True)
    account_keys = trades.get("account", trades.instrument).to_numpy()
    true_instruments = trades.instrument.to_numpy()
    is_buy = trades.is_buy.to_numpy()
    stop_risk = trades.stop_risk.to_numpy(float)
    entry_price = trades.entry_price.to_numpy(float)
    net_r = trades.net_r.to_numpy(float)
    risk_pct = trades.risk_pct.to_numpy(float)
    unit_count = trades.get("units", pd.Series(np.ones(len(trades)), index=trades.index)).to_numpy(float)
    if {"unit_rs", "unit_entry_times", "unit_entry_prices"}.issubset(trades.columns):
        unit_rs_all = trades.unit_rs.to_numpy(object)
        unit_prices_all = trades.unit_entry_prices.to_numpy(object)
        unit_offsets_all = np.array([
            (pd.to_datetime(list(ts)).values.astype("datetime64[ns]").astype(np.int64) - entry_ns_i)
            for ts, entry_ns_i in zip(trades.unit_entry_times, trades.entry_time.values.astype("datetime64[ns]").astype(np.int64))
        ], dtype=object)
    else:
        unit_rs_all = np.array([[x] for x in net_r], dtype=object)
        unit_prices_all = np.array([[x] for x in entry_price], dtype=object)
        unit_offsets_all = np.array([[0] for _ in range(len(trades))], dtype=object)
    entry_ns = trades.entry_time.values.astype("datetime64[ns]").astype(np.int64)
    exit_ns = trades.exit_time.values.astype("datetime64[ns]").astype(np.int64)
    n = len(trades)
    positive_gaps = np.diff(entry_ns)
    positive_gaps = positive_gaps[positive_gaps > 0]
    boundary_gap = int(np.median(positive_gaps)) if len(positive_gaps) else int(86400e9)
    durations = np.maximum(exit_ns - entry_ns, 1)
    rng = np.random.default_rng(seed)
    results = []; ruins = []; total_skipped = 0
    for _ in range(n_sims):
        idx = []
        while len(idx) < max_trades:
            start = rng.integers(0, n - block + 1) if n > block else 0
            idx.extend(range(start, min(start + block, n)))
        idx = np.array(idx[:max_trades])
        synthetic_entry = np.zeros(len(idx), dtype=np.int64)
        if len(idx) > 1:
            gaps = np.full(len(idx) - 1, boundary_gap, dtype=np.int64)
            adjacent = idx[1:] == idx[:-1] + 1
            real_gaps = entry_ns[idx[1:]] - entry_ns[idx[:-1]]
            gaps[adjacent] = np.maximum(real_gaps[adjacent], 1)
            synthetic_entry[1:] = np.cumsum(gaps)
        synthetic_exit = synthetic_entry + durations[idx]
        dt_years = synthetic_entry / 1e9 / 86400 / 365.25
        exit_dt_years = synthetic_exit / 1e9 / 86400 / 365.25
        yrs, ruined, skipped = simulate_path_margin_aware(
            account_keys[idx], true_instruments[idx], is_buy[idx], stop_risk[idx], entry_price[idx],
            net_r[idx], synthetic_entry, synthetic_exit, dt_years, risk_pct[idx], instrument_list,
            unit_count=unit_count[idx], exit_dt_years=exit_dt_years,
            unit_rs=unit_rs_all[idx], unit_add_offsets_ns=unit_offsets_all[idx],
            unit_prices=unit_prices_all[idx], **kw)
        results.append(yrs); ruins.append(ruined); total_skipped += skipped
    r = true_median_and_stats(np.array(results), np.array(ruins), n_sims)
    r["avg_skipped_per_path"] = total_skipped / n_sims
    return r
