#!/usr/bin/env python3
"""Goal 43: Strategy Rankings Rerank on 3-Factor Composite Score.

Composite Score = Combination of:
1. Win Rate (WR)
2. Accepted Trade Frequency (following dual-subaccount 1-per-side constraints)
3. Quickest Time to $1,000,000 (Median Days to $1M in dual-subaccount Monte Carlo)

Account Architecture:
- Two separate $100 sub-accounts (buy-only, sell-only, never pooled)
- Optimal risk sizing: risk_pct of min(buy_equity, sell_equity)
- 50x max leverage
- 1.6-pip fixed spread across EURUSD, GBPUSD, EURGBP, USDJPY
- Max 1 open position per side (max 1 buy, max 1 sell across all 4 pairs)
- Target: $1,000,000

Evaluates all 16 strategies and updates STRATEGY_RANKINGS.md and strategy_rankings.csv.
"""
import os
import sys
import time
import heapq
import numpy as np
import pandas as pd

os.environ["MPLCONFIGDIR"] = "/tmp"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
DATA_REPO_DIR = "/Users/brunoinzunza/Documents/GitHub/trading-data"

import dtfx_engine
import goal5_rr_sweep as g5
import goal20_speed_king_margin_report as g20
import goal23_margin_capacity_all_strategies as g23
import goal29_rerank_1_per_side as g29

position_size_and_margin = g20.position_size_and_margin
BLOCK = g20.BLOCK
RUIN_FLOOR = g20.RUIN_FLOOR
YEAR_NS = 365.25 * 86400 * 1e9

SPLIT = pd.Timestamp("2018-03-09")
START_PER_SIDE = 100.0
MAX_LEVERAGE = 50.0
ACCOUNT_CAP = 1_000_000.0
SPREAD_PIPS = 1.6
N_MC_SIMS = 10000

# 3-Factor Weights: Speed 40%, Win Rate 35%, Frequency 25%
WEIGHTS = dict(speed=0.40, wr=0.35, freq=0.25)

# Strategy definitions
STRATEGIES = dict(g29.STRATEGIES)
STRATEGIES["Apex-King"] = dict(n=4.5, rr=0.18, age_h=0.30, bounce=False, touches=8, session=None, holdout=True)
STRATEGIES["Titan-76"] = dict(ret=1, n=2.0, rr=0.76, age_h=0.20, bounce=False, touches=6, probe=0.75, exc=1.0, session=None, holdout=True)


def build_trades_for_strategy(name, cfg):
    ret = cfg.get("ret", 2)
    n = cfg["n"]
    rr = cfg["rr"]
    age_h = cfg["age_h"]
    bounce = cfg["bounce"]
    touches = cfg["touches"]
    probe = cfg.get("probe", g23.pg.PROBE_DEPTH)
    exc = cfg.get("exc", 1.0)
    session = cfg.get("session", None)

    frames = []
    for inst in g5.TRADEABLE:
        z = g5.rs._prep_zones(inst, ret)
        times, lows, highs, df = g5.rs.load_instrument(inst)
        times_ns = times.astype("datetime64[ns]").astype("int64")
        created_ns = z["created_at"].astype("datetime64[ns]").astype("int64")

        pip = g5.pg.PIP_SIZE[inst]
        spread_price = SPREAD_PIPS * pip

        n_i = n[inst] if isinstance(n, dict) else n
        age_i = age_h[inst] if isinstance(age_h, dict) else age_h
        target_mult = n_i * rr

        for side_is_buy in (True, False):
            mask = z["is_buy"] == side_is_buy
            top = np.ascontiguousarray(z["top"][mask], dtype=np.float64)
            bottom = np.ascontiguousarray(z["bottom"][mask], dtype=np.float64)
            mid = np.ascontiguousarray(z["mid"][mask], dtype=np.float64)
            is_buy_arr = np.ascontiguousarray(z["is_buy"][mask], dtype=np.bool_)
            start_idx = np.ascontiguousarray(z["start_idx"][mask], dtype=np.int64)
            c_ns = np.ascontiguousarray(created_ns[mask], dtype=np.int64)

            half_width = (top - bottom) / 2.0
            risk_pct_zone = (n_i * half_width) / mid
            ok = risk_pct_zone >= g5.pg.MIN_RISK_PCT

            if not np.any(ok):
                continue

            touch_num, entry_idx, exit_idx, is_win, entry_mid, stop_risk = dtfx_engine.simulate_generic_ex(
                top[ok], bottom[ok], mid[ok], is_buy_arr[ok], start_idx[ok], c_ns[ok],
                times_ns, lows, highs, int(age_i * 3600 * 1e9), n_i, target_mult, bounce, probe, exc,
                touches, g5.pg.MAX_WINDOW_BARS
            )

            if len(entry_idx) == 0:
                continue

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
                "instrument": inst,
                "entry_time": entry_time[keep],
                "exit_time": exit_time[keep],
                "entry_price": entry_mid[keep],
                "stop_risk": stop_risk[keep],
                "net_r": net_r[keep],
                "is_win": is_win[keep],
                "is_buy": side_is_buy,
                "touch_num": touch_num[keep],
            }))

    if not frames:
        return pd.DataFrame()

    all_t = pd.concat(frames, ignore_index=True).sort_values("entry_time").reset_index(drop=True)
    if cfg.get("holdout", True):
        return all_t[all_t.entry_time >= SPLIT].reset_index(drop=True)
    return all_t


def simulate_two_account(trades, risk_pct=0.05, start_per_side=START_PER_SIDE,
                          max_leverage=MAX_LEVERAGE, cap=ACCOUNT_CAP):
    t = trades.sort_values("entry_time").reset_index(drop=True)
    n = len(t)
    if n == 0:
        return t, start_per_side, start_per_side, 0.0

    entry_time = t.entry_time.values.astype("datetime64[ns]").astype(np.int64)
    exit_time = t.exit_time.values.astype("datetime64[ns]").astype(np.int64)
    entry_price = t.entry_price.values
    stop_risk = t.stop_risk.values
    net_r = t.net_r.values
    instrument = t.instrument.values
    is_buy = t.is_buy.values

    equity = {True: start_per_side, False: start_per_side}
    margin_used = {True: 0.0, False: 0.0}
    open_count = {True: 0, False: 0}
    heap = []

    peak_combined = start_per_side * 2.0
    max_dd_pct = 0.0
    accepted = np.zeros(n, dtype=bool)

    for i in range(n):
        et = entry_time[i]
        while heap and heap[0][0] <= et:
            xt, r_dollars, margin, r, side = heapq.heappop(heap)
            equity[side] += r_dollars * r
            margin_used[side] -= margin
            open_count[side] -= 1
            combined = equity[True] + equity[False]
            if combined > peak_combined:
                peak_combined = combined
            dd = 100.0 * (combined - peak_combined) / peak_combined
            if dd < max_dd_pct:
                max_dd_pct = dd

        margin_used[True] = max(margin_used[True], 0.0)
        margin_used[False] = max(margin_used[False], 0.0)

        side = bool(is_buy[i])
        sizing_base = min(equity[True], equity[False])
        if sizing_base <= RUIN_FLOOR:
            continue

        _, risk_dollars, margin = position_size_and_margin(
            instrument[i], sizing_base, entry_price[i], stop_risk[i], risk_pct, max_leverage, cap
        )
        avail = min(equity[side], cap) - margin_used[side]

        if margin <= avail and open_count[side] < 1:
            accepted[i] = True
            heapq.heappush(heap, (exit_time[i], risk_dollars, margin, net_r[i], side))
            margin_used[side] += margin
            open_count[side] += 1

    while heap:
        xt, r_dollars, margin, r, side = heapq.heappop(heap)
        equity[side] += r_dollars * r
        combined = equity[True] + equity[False]
        if combined > peak_combined:
            peak_combined = combined
        dd = 100.0 * (combined - peak_combined) / peak_combined
        if dd < max_dd_pct:
            max_dd_pct = dd

    t = t.copy()
    t["accepted"] = accepted
    return t, equity[True], equity[False], max_dd_pct


def run_mc_two_account(trades, risk_pct=0.05, n_sims=N_MC_SIMS, target=1_000_000.0):
    t = trades.sort_values("entry_time").reset_index(drop=True)
    n = len(t)
    entry_ns = t.entry_time.values.astype("datetime64[ns]").astype(np.int64)
    exit_ns = t.exit_time.values.astype("datetime64[ns]").astype(np.int64)
    entry_price = t.entry_price.values
    stop_risk = t.stop_risk.values
    net_r = t.net_r.values
    instrument = t.instrument.values
    is_buy = t.is_buy.values

    years = (t.entry_time.max() - t.entry_time.min()).days / 365.25
    tpy = n / years

    rng = np.random.default_rng(42)
    days_to_1m = []
    max_dds = []
    ruined_count = 0

    for sim in range(n_sims):
        equity = {True: START_PER_SIDE, False: START_PER_SIDE}
        margin_used = {True: 0.0, False: 0.0}
        open_count = {True: 0, False: 0}
        heap = []
        clock = 0
        peak = START_PER_SIDE * 2.0
        worst_dd = 0.0
        ruined = False
        hit_target = False
        t_hit_days = -1.0
        trades_count = 0

        while trades_count < 100_000 and not hit_target and not ruined:
            start_idx = rng.integers(0, n)
            end_idx = min(start_idx + BLOCK, n)
            offset = clock - entry_ns[start_idx]
            block_exit_max = clock

            for idx in range(start_idx, end_idx):
                et = entry_ns[idx] + offset
                xt = exit_ns[idx] + offset
                if xt > block_exit_max:
                    block_exit_max = xt

                while heap and heap[0][0] <= et:
                    _, r_dollars, margin, r, side = heapq.heappop(heap)
                    equity[side] += r_dollars * r
                    margin_used[side] -= margin
                    open_count[side] -= 1
                    comb = equity[True] + equity[False]
                    if comb > peak:
                        peak = comb
                    dd = 100.0 * (comb - peak) / peak
                    if dd < worst_dd:
                        worst_dd = dd
                    if comb <= RUIN_FLOOR or equity[True] <= 1.0 or equity[False] <= 1.0:
                        ruined = True

                if ruined:
                    break

                side = bool(is_buy[idx])
                sizing_base = min(equity[True], equity[False])
                _, risk_dollars, margin = position_size_and_margin(
                    instrument[idx], sizing_base, entry_price[idx], stop_risk[idx], risk_pct,
                    MAX_LEVERAGE, ACCOUNT_CAP
                )
                avail = min(equity[side], ACCOUNT_CAP) - margin_used[side]

                if margin <= avail and open_count[side] < 1:
                    heapq.heappush(heap, (xt, risk_dollars, margin, net_r[idx], side))
                    margin_used[side] += margin
                    open_count[side] += 1
                    trades_count += 1

                comb = equity[True] + equity[False]
                if comb >= target:
                    hit_target = True
                    t_hit_days = (et / YEAR_NS) * 365.25
                    break

            clock = block_exit_max

        if ruined:
            ruined_count += 1
        if hit_target:
            days_to_1m.append(t_hit_days)
        max_dds.append(worst_dd)

    median_days = np.median(days_to_1m) if len(days_to_1m) else float("inf")
    median_dd = np.median(max_dds) if len(max_dds) else float("nan")
    ruin_pct = 100.0 * ruined_count / n_sims
    return median_days, median_dd, ruin_pct


def find_optimal_risk(trades):
    """Find optimal risk_pct minimizing median days to $1M with zero ruin."""
    best_risk, best_days = 0.05, float("inf")
    for r in [0.03, 0.04, 0.05, 0.06, 0.08]:
        days, dd, ruin = run_mc_two_account(trades, risk_pct=r, n_sims=1500)
        if ruin == 0.0 and days < best_days:
            best_days = days
            best_risk = r
    return best_risk


def main():
    print("=" * 125)
    print("  EVALUATING 16 STRATEGIES UNDER DUAL SUB-ACCOUNT REALISTIC CONSTRAINTS")
    print("  Two $100 Accounts (Buy / Sell) | min(buy, sell) Sizing | 50x Leverage | 1.6-Pip Spread | Max 1/Side")
    print("=" * 125)

    raw_results = []
    t_start = time.time()

    for idx, (name, cfg) in enumerate(STRATEGIES.items(), start=1):
        print(f"[{idx:2d}/{len(STRATEGIES)}] Evaluating {name}...", flush=True)
        trades = build_trades_for_strategy(name, cfg)
        if len(trades) == 0:
            print(f"  Warning: No trades for {name}")
            continue

        years = (trades.entry_time.max() - trades.entry_time.min()).days / 365.25
        opt_risk = find_optimal_risk(trades)

        # Full run with optimal risk
        t_sim, b_eq, s_eq, walk_dd = simulate_two_account(trades, risk_pct=opt_risk)
        acc_trades = t_sim[t_sim.accepted]
        acc_count = len(acc_trades)
        acc_tpy = acc_count / years
        acc_wr = 100.0 * (acc_trades.net_r > 0).mean() if acc_count > 0 else 0.0
        margin_acc_pct = 100.0 * acc_count / len(trades)

        # Monte Carlo to $1,000,000 (10k runs)
        median_days_1m, mc_dd, ruin_pct = run_mc_two_account(trades, risk_pct=opt_risk, n_sims=N_MC_SIMS)

        print(f"  -> WR: {acc_wr:.2f}% | Freq: {acc_tpy:,.0f} accepted trades/yr ({margin_acc_pct:.1f}% accepted) | "
              f"Days to $1M: {median_days_1m:.1f} days ({median_days_1m/365.25:.3f} yrs) | Opt Risk: {opt_risk*100:.0f}%", flush=True)

        raw_results.append({
            "strategy": name,
            "wr": acc_wr,
            "freq_tpy": acc_tpy,
            "days_to_1m": median_days_1m,
            "years_to_1m": median_days_1m / 365.25,
            "opt_risk": opt_risk,
            "margin_acc_pct": margin_acc_pct,
            "raw_trades": len(trades),
            "accepted_trades": acc_count,
            "mc_dd": mc_dd,
            "ruin_pct": ruin_pct,
            "rr": cfg["rr"] if isinstance(cfg["rr"], (int, float)) else 1.0,
            "n": cfg["n"] if isinstance(cfg["n"], (int, float)) else 2.0,
        })

    df_res = pd.DataFrame(raw_results)

    # -------------------------------------------------------------
    # COMPUTE NEW 3-FACTOR COMPOSITE SCORE
    # -------------------------------------------------------------
    # 1. Win Rate Score (Higher is better)
    wr_min, wr_max = df_res.wr.min(), df_res.wr.max()
    df_res["wr_score"] = 100.0 * (df_res.wr - wr_min) / (wr_max - wr_min) if wr_max > wr_min else 100.0

    # 2. Frequency Score (Higher is better)
    freq_min, freq_max = df_res.freq_tpy.min(), df_res.freq_tpy.max()
    df_res["freq_score"] = 100.0 * (df_res.freq_tpy - freq_min) / (freq_max - freq_min) if freq_max > freq_min else 100.0

    # 3. Speed Score (Lower days is better)
    days_min, days_max = df_res.days_to_1m.min(), df_res.days_to_1m.max()
    df_res["speed_score"] = 100.0 * (1.0 - (df_res.days_to_1m - days_min) / (days_max - days_min)) if days_max > days_min else 100.0

    # Final Composite Score
    w = WEIGHTS
    df_res["composite_score"] = (
        w["speed"] * df_res["speed_score"] +
        w["wr"] * df_res["wr_score"] +
        w["freq"] * df_res["freq_score"]
    )

    df_ranked = df_res.sort_values("composite_score", ascending=False).reset_index(drop=True)
    df_ranked["rank"] = df_ranked.index + 1

    # Save CSVs
    csv_local = os.path.join(DATA_DIR, "goal43_strategy_rankings_3factor.csv")
    csv_repo = os.path.join(DATA_REPO_DIR, "strategy_rankings.csv")
    df_ranked.to_csv(csv_local, index=False)
    df_ranked.to_csv(csv_repo, index=False)

    print("\n" + "=" * 135)
    print("  OFFICIAL NEW STRATEGY RANKINGS (3-FACTOR COMPOSITE SCORE)")
    print("  Weights: Speed to $1M (40%) | Win Rate (35%) | Accepted Frequency (25%)")
    print("=" * 135)
    print(df_ranked[[
        "rank", "strategy", "composite_score", "speed_score", "wr_score", "freq_score",
        "wr", "freq_tpy", "days_to_1m", "opt_risk", "margin_acc_pct", "rr"
    ]].to_string(index=False))

    # Generate Markdown documentation
    md_content = f"""# Strategy Rankings (3-Factor Composite Score)

Rankings evaluated under the live dual-subaccount trading architecture:
- **Two separate \$100 sub-accounts**: Buy-only sub-account and Sell-only sub-account (**never pooled**).
- **Position Sizing**: Optimal risk % of `min(buy_equity, sell_equity)`.
- **Broker Constraints**: **50x leverage**, **1.6-pip fixed spread** on `EURUSD`, `GBPUSD`, `EURGBP`, `USDJPY`.
- **Concurrency**: **Max 1 open position per side** (at most 1 active buy and 1 active sell across all 4 pairs).

---

## 🎯 3-Factor Composite Score Formula

$$\\text{{Composite Score}} = 0.40 \\cdot \\text{{Speed Score}} + 0.35 \\cdot \\text{{WR Score}} + 0.25 \\cdot \\text{{Frequency Score}}$$

Where:
1. **Speed Score (40% weight)**: Normalized from median calendar days to reach \$1,000,000 under 10,000 Monte Carlo bootstrap paths.
2. **Win Rate Score (35% weight)**: Normalized out-of-sample win rate on accepted trades.
3. **Frequency Score (25% weight)**: Normalized annual accepted trade count under the 1-per-side concurrency limit.

---

## 🏆 Official Strategy Leaderboard

| Rank | Strategy | Composite Score | Speed Score | WR Score | Freq Score | Win Rate | Accepted Trades/Yr | Median Days to \$1M | Optimal Risk | Margin Accepted | Nominal R:R |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
"""
    for _, r in df_ranked.iterrows():
        md_content += (
            f"| **#{int(r['rank'])}** | **{r['strategy']}** | **{r['composite_score']:.1f}** | "
            f"{r['speed_score']:.1f} | {r['wr_score']:.1f} | {r['freq_score']:.1f} | "
            f"**{r['wr']:.2f}%** | **{r['freq_tpy']:,.0f}** | **{r['days_to_1m']:.1f} d** | "
            f"{r['opt_risk']*100:.0f}% | {r['margin_acc_pct']:.1f}% | {r['rr']:.2f} |\n"
        )

    md_content += f"""
---

## 📊 Summary & Key Highlights
* **Rank #1 Champion**: **{df_ranked.iloc[0]['strategy']}** (Composite Score: **{df_ranked.iloc[0]['composite_score']:.1f}**)
* **Rank #2 Runner-Up**: **{df_ranked.iloc[1]['strategy']}** (Composite Score: **{df_ranked.iloc[1]['composite_score']:.1f}**)
* **Rank #3 Third Place**: **{df_ranked.iloc[2]['strategy']}** (Composite Score: **{df_ranked.iloc[2]['composite_score']:.1f}**)

*Last Updated*: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}
"""

    md_repo = os.path.join(DATA_REPO_DIR, "STRATEGY_RANKINGS.md")
    with open(md_repo, "w") as f:
        f.write(md_content)
    print(f"\nSaved updated rankings to {csv_repo} and {md_repo} in {time.time()-t_start:.1f}s.")


if __name__ == "__main__":
    main()
