"""Rerank all 15 STRATEGY_RANKINGS.md candidates and redo the race-to-$1M Monte Carlo
under a NEW concurrency assumption: at most 1 open position per side (buy sub-account,
sell sub-account) -- max 2 total, never 2 stacked on the same side. Matches the live
PRODLIVETRADING architecture (two separate, never-pooled OANDA sub-accounts) more
faithfully than the previous margin-only model.

This also fixes a real inconsistency in the existing methodology, as a side effect: until
now, "Speed"/"Margin capacity" were two SEPARATE, disjoint calculations --
goal12_strategy_score.py's speed/drawdown MC (run_mc_capped) resamples the raw net_r
stream completely blind to margin, while goal23's margin-acceptance % was a static
single-path walk with no MC at all. Here, both come from ONE real-timing block-bootstrap
MC per strategy (goal21/goal28's methodology) that enforces margin AND the 1-per-side cap
together on every simulated path -- so "how fast" and "how much survives realistic
capital constraints" are finally the same simulation, not two different assumptions.

WR, R:R, and raw trade frequency are unaffected by concurrency and keep coming from each
strategy's full (unconstrained) TRUE holdout trade stream, same as before -- only Speed
and Margin/Concurrency change.
"""
import os
import sys
import heapq
import time
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

import dtfx_engine
import goal20_speed_king_margin_report as g20
import goal23_margin_capacity_all_strategies as g23

rs, pg, TRADEABLE = g23.rs, g23.pg, g23.TRADEABLE
SPLIT = g23.SPLIT
START, RISK_PCT, MAX_LEVERAGE, CAP = g20.START, g20.RISK_PCT, g20.MAX_LEVERAGE, g20.CAP
YEAR_NS = 365.25 * 86400 * 1e9

# All 15 current STRATEGY_RANKINGS.md candidates: goal23's original 14, plus Apex-King
# (N=4.5, R:R=0.18, age=0.30h, unconditional re-arm, 8 touches, no session filter --
# same holdout convention as Speed-King/Quality-King, per the doc's own parameter table).
STRATEGIES = dict(g23.STRATEGIES)
STRATEGIES["Apex-King"] = dict(n=4.5, rr=0.18, age_h=0.30, bounce=False, touches=8,
                                session=None, holdout=True)
RR_BY_STRATEGY = {k: v["rr"] for k, v in STRATEGIES.items()}


def build_trades_with_side(n, rr, age_h, bounce, touches, session):
    """Exact copy of goal23.build_trades, plus an is_buy column -- needed for the
    per-side concurrency cap. Calls simulate_generic_ex separately per side (buy-only
    zones, then sell-only) instead of mixed, since the engine's output is per-trade with
    no zone-index back-reference, so there's no way to recover side from a mixed call."""
    frames = []
    for inst in TRADEABLE:
        z = rs._prep_zones(inst, pg.MIN_RETRACEMENT)
        times, lows, highs, df = rs.load_instrument(inst)
        times_ns = times.astype("datetime64[ns]").astype("int64")
        created_ns = z["created_at"].astype("datetime64[ns]").astype("int64")

        n_i = n[inst] if isinstance(n, dict) else n
        age_i = age_h[inst] if isinstance(age_h, dict) else age_h
        target_mult = n_i * rr

        risk_pct = (n_i * (z["top"] - z["bottom"]) / 2.0) / z["mid"]
        risk_ok = risk_pct >= pg.MIN_RISK_PCT

        for side_is_buy in (True, False):
            mask = risk_ok & (z["is_buy"] == side_is_buy)
            top, bottom, mid = z["top"][mask], z["bottom"][mask], z["mid"][mask]
            is_buy_arr, start_idx, c_ns = z["is_buy"][mask], z["start_idx"][mask], created_ns[mask]
            if len(top) == 0:
                continue
            touch_num, entry_idx, exit_idx, is_win, entry_mid, stop_risk = dtfx_engine.simulate_generic_ex(
                top, bottom, mid, is_buy_arr, start_idx, c_ns, times_ns, lows, highs,
                int(age_i * 3600 * 1e9), n_i, target_mult, bounce, pg.PROBE_DEPTH, 1.0,
                touches, pg.MAX_WINDOW_BARS)
            if len(entry_idx) == 0:
                continue

            pip = pg.PIP_SIZE[inst]
            spread_price = pg.SPREAD_PIPS_TYPICAL * pip
            r_gross = np.where(is_win, rr, -1.0)
            cost_in_r = np.where(is_win, 0.0, spread_price / stop_risk)
            net_r = r_gross - cost_in_r
            entry_time = pd.to_datetime(times[entry_idx])
            exit_time = pd.to_datetime(times[exit_idx])

            keep = np.ones(len(entry_idx), dtype=bool)
            if isinstance(session, dict):
                allowed = session.get(inst)
                if allowed is not None:
                    keep = np.isin(entry_time.hour, list(allowed))

            frames.append(pd.DataFrame({
                "instrument": inst, "entry_time": entry_time[keep], "exit_time": exit_time[keep],
                "entry_price": entry_mid[keep], "stop_risk": stop_risk[keep],
                "net_r": net_r[keep], "is_win": is_win[keep], "is_buy": side_is_buy,
            }))
    return pd.concat(frames, ignore_index=True).sort_values("entry_time").reset_index(drop=True)


def vote_filtered_trades_with_side(cfg):
    pools = {}
    for inst in TRADEABLE:
        t = g23.POOL_BUILDERS[cfg["pool_fn"]](inst)
        t["glog"] = np.log1p(0.05 * t.net_r)
        t["hour"] = t.entry_time.dt.hour // 6
        pools[inst] = t
    wf_hours, half = g23.get_wf_hours_and_split(pools)

    raw = build_trades_with_side(cfg["n"], cfg["rr"], cfg["age_h"], cfg["bounce"], cfg["touches"], session=None)
    block = raw.entry_time.dt.hour // 6
    keep = np.zeros(len(raw), dtype=bool)
    for inst in TRADEABLE:
        m = (raw.instrument.values == inst) & block.isin(wf_hours[inst]).values & (raw.entry_time >= half).values
        keep |= m
    return raw[keep].sort_values("entry_time").reset_index(drop=True)


def get_true_trades(label, cfg):
    if cfg["session"] == "VOTE":
        return vote_filtered_trades_with_side(cfg)
    t = build_trades_with_side(cfg["n"], cfg["rr"], cfg["age_h"], cfg["bounce"], cfg["touches"], cfg["session"])
    return t[t.entry_time >= SPLIT] if cfg["holdout"] else t


def mc_path_1_per_side(entry_ns, exit_ns, instrument, entry_price, stop_risk, net_r, is_buy,
                        rng, block, max_trades, start=START, risk_pct=RISK_PCT,
                        max_leverage=MAX_LEVERAGE, cap=CAP, target1=1_000_000.0, ruin_floor=1.0):
    """Real-timing block-bootstrap MC path (goal21/goal28 methodology): margin-constrained
    + max 1 open position per side. Returns (years_to_1m, max_dd_pct, ruined, processed,
    accepted_n)."""
    n = len(entry_ns)
    equity = start
    peak = start
    dd_worst = 0.0
    margin_used = 0.0
    open_buy, open_sell = 0, 0
    heap = []
    clock = 0
    t1 = -1.0
    ruined = False
    processed = 0
    accepted_n = 0

    while processed < max_trades and t1 < 0 and not ruined:
        start_idx = rng.integers(0, n)
        end_idx = min(start_idx + block, n)
        offset = clock - entry_ns[start_idx]
        block_exit_max = clock

        for idx in range(start_idx, end_idx):
            et = entry_ns[idx] + offset
            xt = exit_ns[idx] + offset
            if xt > block_exit_max:
                block_exit_max = xt

            while heap and heap[0][0] <= et:
                _, r_dollars, margin, r, buy_side = heapq.heappop(heap)
                equity += r_dollars * r
                margin_used -= margin
                if buy_side:
                    open_buy -= 1
                else:
                    open_sell -= 1
                if equity > peak:
                    peak = equity
                dd = 100 * (equity - peak) / peak
                if dd < dd_worst:
                    dd_worst = dd
                if equity <= ruin_floor:
                    ruined = True
            margin_used = max(margin_used, 0.0)
            if ruined:
                break

            _, risk_dollars, margin = g20.position_size_and_margin(
                instrument[idx], equity, entry_price[idx], stop_risk[idx], risk_pct,
                max_leverage, cap)
            avail = min(equity, cap) if cap is not None else equity
            side_is_buy = bool(is_buy[idx])
            side_open = open_buy if side_is_buy else open_sell

            if margin_used + margin <= avail and side_open < 1:
                heapq.heappush(heap, (xt, risk_dollars, margin, net_r[idx], side_is_buy))
                margin_used += margin
                if side_is_buy:
                    open_buy += 1
                else:
                    open_sell += 1
                accepted_n += 1

            processed += 1
            if t1 < 0 and equity >= target1:
                t1 = et / YEAR_NS
                break
            if processed >= max_trades:
                break

        if ruined or t1 >= 0:
            break
        clock = block_exit_max

    return t1, dd_worst, ruined, processed, accepted_n


def run_mc(t, n_paths, seed, block=g20.BLOCK, max_trades=200_000):
    t = t.sort_values("entry_time").reset_index(drop=True)
    entry_ns = t.entry_time.values.astype("datetime64[ns]").astype(np.int64)
    exit_ns = t.exit_time.values.astype("datetime64[ns]").astype(np.int64)
    instrument = t.instrument.values
    entry_price = t.entry_price.values
    stop_risk = t.stop_risk.values
    net_r = t.net_r.values
    is_buy = t.is_buy.values

    rng = np.random.default_rng(seed)
    t1s = np.empty(n_paths)
    dds = np.empty(n_paths)
    ruins = np.zeros(n_paths, dtype=bool)
    accept_rates = np.empty(n_paths)
    for p in range(n_paths):
        t1, dd, ruined, processed, accepted_n = mc_path_1_per_side(
            entry_ns, exit_ns, instrument, entry_price, stop_risk, net_r, is_buy,
            rng, block, max_trades)
        t1s[p] = t1
        dds[p] = dd
        ruins[p] = ruined
        accept_rates[p] = accepted_n / processed if processed else np.nan
    return t1s, dds, ruins, accept_rates


def minmax_norm(values, higher_is_better):
    vals = np.array(list(values.values()), dtype=float)
    lo, hi = vals.min(), vals.max()
    if hi == lo:
        return {k: 100.0 for k in values}
    out = {}
    for k, v in values.items():
        frac = (v - lo) / (hi - lo)
        out[k] = 100 * frac if higher_is_better else 100 * (1 - frac)
    return out


DEFAULT_WEIGHTS = dict(speed=0.35, wr=0.30, freq=0.12, margin=0.08, drawdown=0.10, rrr=0.05)


def main():
    for inst in TRADEABLE:
        rs.load_instrument(inst)
        rs._prep_zones(inst, pg.MIN_RETRACEMENT)

    print(f"Rebuilding TRUE entry/exit+side trade streams for all {len(STRATEGIES)} strategies...")
    raw = {}
    for label, cfg in STRATEGIES.items():
        t0 = time.perf_counter()
        t = get_true_trades(label, cfg)
        years = (t.entry_time.max() - t.entry_time.min()).days / 365.25
        tpy = len(t) / years
        wr = 100 * t.is_win.mean()

        # Fewer/shorter paths for strategies whose raw trade stream is far slower to
        # compound (goal21's precedent: precision where the real competition is, less
        # elsewhere) -- avoids one slow strategy dominating total runtime.
        n_paths = 300 if tpy >= 1000 else 150
        max_trades = 200_000 if tpy >= 1000 else 600_000
        t1s, dds, ruins, accept_rates = run_mc(t, n_paths, seed=abs(hash(label)) % 100000,
                                                max_trades=max_trades)
        reached = t1s >= 0
        median_yrs_1m = np.median(t1s[reached]) if reached.any() else float("inf")
        median_dd = np.median(dds)
        pct_ruined = 100 * ruins.mean()
        concurrency_pct = 100 * np.nanmedian(accept_rates)

        raw[label] = dict(trades=len(t), tpy=tpy, wr=wr, rr=RR_BY_STRATEGY[label],
                           median_yrs_1m=median_yrs_1m, median_dd=median_dd,
                           pct_ruined=pct_ruined, margin_pct=concurrency_pct,
                           pct_reached=100 * reached.mean())
        dt = time.perf_counter() - t0
        print(f"{label:22s} n={len(t):7,} ({tpy:6.0f}/yr) WR={wr:5.2f}% RR={RR_BY_STRATEGY[label]:.2f} "
              f"yrs->1M={median_yrs_1m:6.3f} medDD={median_dd:6.1f}% concurrency%={concurrency_pct:5.1f}% "
              f"ruin={pct_ruined:5.2f}% reached={raw[label]['pct_reached']:5.1f}%  [{dt:5.1f}s, {n_paths}p]")

    speed_n = minmax_norm({k: v["median_yrs_1m"] for k, v in raw.items()}, higher_is_better=False)
    dd_n = minmax_norm({k: v["median_dd"] for k, v in raw.items()}, higher_is_better=True)
    wr_n = minmax_norm({k: v["wr"] for k, v in raw.items()}, higher_is_better=True)
    freq_n = minmax_norm({k: v["tpy"] for k, v in raw.items()}, higher_is_better=True)
    margin_n = minmax_norm({k: v["margin_pct"] for k, v in raw.items()}, higher_is_better=True)
    rrr_n = minmax_norm({k: v["rr"] for k, v in raw.items()}, higher_is_better=True)

    w = DEFAULT_WEIGHTS
    final_scores = {k: (w["speed"] * speed_n[k] + w["drawdown"] * dd_n[k] + w["wr"] * wr_n[k] +
                         w["freq"] * freq_n[k] + w["margin"] * margin_n[k] + w["rrr"] * rrr_n[k])
                     for k in raw}

    rows = []
    for k in raw:
        rows.append(dict(strategy=k, **raw[k], speed_score=round(speed_n[k], 1),
                          dd_score=round(dd_n[k], 1), wr_score=round(wr_n[k], 1),
                          freq_score=round(freq_n[k], 1), margin_score=round(margin_n[k], 1),
                          rrr_score=round(rrr_n[k], 1), final_score=round(final_scores[k], 1)))
    df = pd.DataFrame(rows).sort_values("final_score", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    df.to_csv(os.path.join(DATA_DIR, "goal29_rerank_1_per_side.csv"), index=False)

    print("\n" + "=" * 100)
    print("RERANKED (1 open position per side, margin+concurrency-aware speed MC)")
    print("=" * 100)
    print(df[["rank", "strategy", "final_score", "speed_score", "wr_score", "freq_score",
              "margin_score", "dd_score", "rrr_score", "wr", "margin_pct", "tpy",
              "median_yrs_1m"]].to_string(index=False))
    print(f"\nSaved: goal29_rerank_1_per_side.csv")


if __name__ == "__main__":
    main()
