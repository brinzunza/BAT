"""Rank every full-rigor strategy built in this investigation by REAL (not resampled/Monte-Carlo)
2026-YTD performance: replay each strategy's actual historical trades from 2026-01-01 through the
last available real data point (Aug/Sep 2026, ~8-8.5 months), starting from a common $200, using
the exact same margin-aware position-sizing/tracking logic as the rest of the investigation
(margin_aware_mc.py's real entry/exit-time margin tracking) -- but walked ONCE, in the real
historical order, with no bootstrap resampling. This answers a different, more concrete question
than every MC result in this report: not "what's the distribution of possible outcomes," but
"what actually would have happened this year, for real, if you had started each of these systems
on Jan 1 2026 with $200."
"""
import sys
sys.path.insert(0, "/Users/brunoinzunza/Documents/GitHub/tradingResearch/dtfxStrategy/research_scripts")
sys.path.insert(0, "/Users/brunoinzunza/Documents/GitHub/tradingResearch/dtfxStrategy/research_scripts/beid_goal_hunt")
sys.path.insert(0, "/Users/brunoinzunza/Documents/GitHub/tradingResearch")
import numpy as np
import pandas as pd
import gold_orb_m1_probe as gp
import btc_orb_m1_probe as bp
from donchian_concurrent_engine import build_concurrent_trades
import combined_fx_gold_mc as cmc
from feargreed_full_mc_test import compute_atr_ratio
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from beid_ml_metalabel import build_features
from quant_engine.settlement import EventSettler

ETH_CSV = "/Users/brunoinzunza/Documents/GitHub/trading-data/oanda_ethusd_m1_2021on_clean.csv"
START_2026 = pd.Timestamp("2026-01-01")


def replay_real_path(trades, start_map, leg_floor=2.0, total_floor=10.0,
                      max_leverage=50.0, gold_leverage=20.0, cap=500000.0,
                      btc_leverage=10.0, es_leverage=15.0, eth_leverage=10.0):
    """Single deterministic walk through the REAL trade sequence (sorted by entry_time, no
    resampling), tracking real open-position margin via the shared quant_engine.settlement
    core (also used by margin_aware_mc.py -- previously a copy-pasted duplicate of that file's
    settle_through closure, which is what let this function silently stop honoring an
    `account`-column override that margin_aware_mc.py respects; fixed by sharing the one
    implementation instead of two drifting copies), returning the full equity trajectory /
    summary stats instead of just a years-to-target."""
    t = trades.sort_values("entry_time", kind="stable").reset_index(drop=True)
    keys = list(start_map.keys())
    settler = EventSettler(keys, {k: start_map[k] / 2.0 for k in keys}, leg_floor=leg_floor)
    entry_ns = t.entry_time.values.astype("datetime64[ns]").astype(np.int64)
    exit_ns = t.exit_time.values.astype("datetime64[ns]").astype(np.int64)
    accounts = t.get("account", t.instrument)
    equity_curve = []
    n_taken = 0; n_wins = 0; gross_profit = 0.0; gross_loss = 0.0
    peak = sum(settler.equity.values()); max_dd = 0.0
    ruined = False

    def on_settle(_key, pnl, exit_time):
        nonlocal peak, max_dd, n_wins, gross_profit, gross_loss, ruined
        if pnl > 0:
            n_wins += 1; gross_profit += pnl
        else:
            gross_loss += -pnl
        total = settler.total_equity()
        peak = max(peak, total)
        max_dd = max(max_dd, (peak - total) / peak if peak > 0 else 0.0)
        equity_curve.append((exit_time, total))
        if total <= total_floor or not settler.any_alive():
            ruined = True
            return True
        return None

    for i in range(len(t)):
        settler.settle_through(entry_ns[i], on_settle)
        if ruined:
            break
        key = (accounts.iloc[i], bool(t.is_buy.iloc[i]))
        if key not in settler.equity or not settler.alive[key]:
            continue
        avail = settler.available_margin(key)
        if avail <= 0:
            settler.skipped += 1
            continue
        _, risk_dollars, margin = cmc.size_and_margin(
            t.instrument.iloc[i], settler.equity[key], t.entry_price.iloc[i], t.stop_risk.iloc[i],
            t.risk_pct.iloc[i], max_leverage, gold_leverage, cap, btc_leverage=btc_leverage,
            es_leverage=es_leverage, eth_leverage=eth_leverage)
        if margin > avail:
            settler.skipped += 1
            continue
        n_taken += 1
        if {"unit_rs", "unit_entry_times", "unit_entry_prices"}.issubset(t.columns):
            row_rs = list(t.unit_rs.iloc[i]); row_times = list(t.unit_entry_times.iloc[i])
            row_prices = list(t.unit_entry_prices.iloc[i])
        else:
            row_rs = [float(t.net_r.iloc[i])]; row_times = [t.entry_time.iloc[i]]
            row_prices = [float(t.entry_price.iloc[i])]
        row_offsets = [pd.Timestamp(rt).value - entry_ns[i] for rt in row_times]
        settler.open_position(
            key, entry_ns=entry_ns[i], exit_ns=exit_ns[i], margin=margin,
            risk_dollars=risk_dollars, entry_price=t.entry_price.iloc[i], unit_rs=row_rs,
            unit_prices=row_prices, unit_offsets_ns=row_offsets, exit_payload=t.exit_time.iloc[i])

    if not ruined:
        settler.settle_through(np.iinfo(np.int64).max, on_settle)
    final_equity = settler.total_equity() if not ruined else max(settler.total_equity(), 0.0)
    n_skipped = settler.skipped
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf") if gross_profit > 0 else float("nan")
    wr = 100 * n_wins / n_taken if n_taken > 0 else float("nan")
    return dict(final_equity=final_equity, n_taken=n_taken, n_skipped=n_skipped, win_rate=wr, pf=pf,
                max_dd=100 * max_dd, ruined=ruined, equity_curve=equity_curve)


def report(name, trades, start_map, **kw):
    sub = trades[trades.entry_time >= START_2026].copy()
    if len(sub) == 0:
        print(f"{name}: no trades in 2026 YTD window")
        return None
    r = replay_real_path(sub, start_map, **kw)
    total_start = sum(start_map.values()) / 2.0  # start_map values are per (instrument,side) key,
    # each key funded at start_map[k]/2 inside replay_real_path (matching margin_aware_mc.py's own
    # start_per_inst/2-per-side convention) -- so the real total capital is half the dict's sum
    ret_pct = 100 * (r["final_equity"] / total_start - 1)
    months = (sub.entry_time.max() - START_2026).days / 30.44
    print(f"{name}: start=${total_start:.0f} -> end=${r['final_equity']:.2f} ({ret_pct:+.1f}% over "
          f"{months:.1f}mo) | n_trades={r['n_taken']} (skipped={r['n_skipped']}) win_rate={r['win_rate']:.0f}% "
          f"PF={r['pf']:.2f} max_dd={r['max_dd']:.1f}% ruined={r['ruined']}")
    return dict(name=name, start=total_start, end=r["final_equity"], ret_pct=ret_pct, months=months,
                n_trades=r["n_taken"], win_rate=r["win_rate"], pf=r["pf"], max_dd=r["max_dd"],
                ruined=r["ruined"])


if __name__ == "__main__":
    results = []

    atr_ratio_btc = compute_atr_ratio(bp.BTC_CSV)
    atr_ratio_eth = compute_atr_ratio(ETH_CSV)
    cols = ["instrument", "entry_time", "exit_time", "entry_price", "stop_risk", "net_r", "is_buy",
            "risk_pct", "units", "unit_rs", "unit_entry_times", "unit_entry_prices"]

    gold_trades = build_concurrent_trades("XAUUSD", gp.XAUUSD_CSV, gp.PIP, gp.SPREAD_PIPS, entry_days=8,
                                           k_stop=2.0, add_step=1.0, max_units=5, max_concurrent=1, start_year=2021)
    gold_trades["risk_pct"] = 0.15

    btc_all = build_concurrent_trades("BTCUSD", bp.BTC_CSV, bp.PIP, bp.SPREAD_PIPS, entry_days=8, k_stop=4.0,
                                       add_step=1.0, max_units=5, max_concurrent=10, start_year=2021)
    btc_all["atr_ratio"] = btc_all.entry_time.dt.date.map(atr_ratio_btc)
    btc_vol = btc_all[btc_all.atr_ratio < 1.0].copy()
    btc_vol["risk_pct"] = 0.5

    eth_all = build_concurrent_trades("ETHUSD", ETH_CSV, 1.0, 3.0, entry_days=8, k_stop=2.0, add_step=1.0,
                                       max_units=5, max_concurrent=10, start_year=2021)
    eth_all["atr_ratio"] = eth_all.entry_time.dt.date.map(atr_ratio_eth)
    eth_vol = eth_all[eth_all.atr_ratio < 1.0].copy()
    eth_vol["risk_pct"] = 0.5

    fg = pd.read_csv("fear_greed_daily.csv", parse_dates=["date"])
    fg_map = dict(zip(fg.date.dt.date, fg.value))
    btc_fg_input = btc_vol.copy()
    btc_fg_input["fg"] = btc_fg_input.entry_time.dt.date.map(fg_map)
    aligned = ((btc_fg_input.is_buy) & (btc_fg_input.fg > 55)) | ((~btc_fg_input.is_buy) & (btc_fg_input.fg < 45))
    btc_sentiment = btc_fg_input[~aligned].copy()

    keys2 = [(a, s) for a in ["XAUUSD", "BTCUSD"] for s in (True, False)]
    keys3 = [(a, s) for a in ["XAUUSD", "BTCUSD", "ETHUSD"] for s in (True, False)]

    print("=== Trend-following strategies (gold/BTC/ETH Donchian pyramid, vol-filtered where applicable) ===")
    results.append(report("1. Gold only", gold_trades[cols], {k: 200.0 for k in [("XAUUSD", True), ("XAUUSD", False)]}))
    results.append(report("2. BTC only (vol-filtered)", btc_vol[cols], {k: 200.0 for k in [("BTCUSD", True), ("BTCUSD", False)]}))
    results.append(report("3. ETH only (vol-filtered)", eth_vol[cols], {k: 200.0 for k in [("ETHUSD", True), ("ETHUSD", False)]}))
    gb = pd.concat([gold_trades[cols], btc_vol[cols]], ignore_index=True)
    results.append(report("4. Gold+BTC (13.36 best-verified)", gb, {k: 100.0 for k in keys2}))
    gbe = pd.concat([gold_trades[cols], btc_vol[cols], eth_vol[cols]], ignore_index=True)
    results.append(report("5. Gold+BTC+ETH (14.6 new best)", gbe, {k: 200.0 / 3 for k in keys3}))
    gb_sent = pd.concat([gold_trades[cols], btc_sentiment[cols]], ignore_index=True)
    results.append(report("6. Gold+BTC w/ sentiment filter (14.2, shown worse)", gb_sent, {k: 100.0 for k in keys2}))

    print("\n=== BEID (FX mean-reversion) ===")
    df = pd.read_pickle("legit_dataset.pkl").sort_values("entry_time").reset_index(drop=True)
    df["y"] = (df.net_r > 0).astype(int)
    t60 = df.entry_time.quantile(0.60); t80 = df.entry_time.quantile(0.80)
    embargo = pd.Timedelta(days=3)
    train = df[df.entry_time < t60 - embargo]
    val = df[(df.entry_time >= t60 + embargo) & (df.entry_time < t80 - embargo)]
    holdout = df[df.entry_time >= t80 + embargo]
    Xtr, _ = build_features(train); Xho, _ = build_features(holdout)
    scaler = StandardScaler().fit(Xtr)
    lr = LogisticRegression(C=0.1, max_iter=2000).fit(scaler.transform(Xtr), train.y.values)
    p_ho = lr.predict_proba(scaler.transform(Xho))[:, 1]
    ml_filtered = holdout[p_ho >= 0.49].copy()
    ml_filtered["risk_pct"] = 0.01
    fx_instruments = sorted(df.instrument.unique().tolist())
    fx_keys = [(a, s) for a in fx_instruments for s in (True, False)]

    raw_beid = holdout.copy()
    raw_beid["risk_pct"] = 0.01
    beid_cols = [c for c in cols if c in raw_beid.columns]
    results.append(report("7. BEID raw/unfiltered (FX)", raw_beid[beid_cols], {k: 200.0 / len(fx_instruments) for k in fx_keys}))
    results.append(report("8. BEID + ML meta-label filter (14.4)", ml_filtered[beid_cols], {k: 200.0 / len(fx_instruments) for k in fx_keys}))

    print("\n=== RANKED BY 2026 YTD RETURN ===")
    valid = [r for r in results if r is not None]
    valid.sort(key=lambda r: r["ret_pct"], reverse=True)
    print(f"{'Rank':<5}{'Strategy':<45}{'Start':>8}{'End':>12}{'Return':>10}{'Trades':>8}{'WinRate':>9}{'PF':>7}{'MaxDD':>8}{'Ruined':>8}")
    for i, r in enumerate(valid, 1):
        print(f"{i:<5}{r['name']:<45}${r['start']:>6.0f} ${r['end']:>10.2f} {r['ret_pct']:>+8.1f}% "
              f"{r['n_trades']:>7} {r['win_rate']:>7.0f}% {r['pf']:>6.2f} {r['max_dd']:>6.1f}% {str(r['ruined']):>8}")
