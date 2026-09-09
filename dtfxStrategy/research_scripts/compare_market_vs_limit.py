import os
import sys
import time
import heapq
import numpy as np
import pandas as pd
from numba import njit

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

import goal95_every_swing_zones as g95
import goal99_dtfx_base_backtest as g99
import goal79_two_account_portfolio as g79
import goal104_optimal_risk_two_account as g104

INSTRUMENTS = {
    "EURUSD": ("lse_eurusd_1min_full.csv", 1e-4),
    "GBPUSD": ("lse_gbpusd_1min_full.csv", 1e-4),
    "EURGBP": ("lse_eurgbp_1min_full.csv", 1e-4),
    "USDJPY": ("lse_usdjpy_1min_2018on.csv", 1e-2),
}
SPREAD_PIPS = 1.6
MIN_STOP_PIPS = 3.0
MAX_WINDOW = 1440

@njit(cache=True)
def resolve_orders_comparison(z_created, z_top, z_bottom, z_is_buy, high, low,
                              spread_price, min_stop_price, max_window, min_zone_height,
                              order_mode):
    """
    order_mode:
      0 = MARKET ORDER ON TOUCH:
          - Enters immediately when mid reaches 50% zone
          - Filled at Ask (Buy: mid + spread/2) or Bid (Sell: mid - spread/2)
          - TP triggers when chart reaches top (Buy) or bottom (Sell), fills at Bid/Ask
          - SL triggers when chart reaches opposite edge, fills at Bid/Ask
      1 = RESTING LIMIT ORDER:
          - Resting limit sits at 50% zone mid
          - Fills ONLY when price penetrates deep enough for Bid/Ask to reach mid:
              Buy: Low <= mid - spread/2 (Ask <= mid) -> filled at exact mid
              Sell: High >= mid + spread/2 (Bid >= mid) -> filled at exact mid
          - TP sits at top/bottom limit -> fills when Mid reaches top + spread/2 (Buy) or bottom - spread/2 (Sell)
              Filled at exact top/bottom price
          - SL sits at opposite edge -> triggers when Mid reaches bottom + spread/2 (Buy) or top - spread/2 (Sell)
              Filled at exact opposite edge price
    """
    m = len(z_created)
    n_bars = len(high)
    out_entry_idx = np.full(m, -1, dtype=np.int64)
    out_exit_idx = np.full(m, -1, dtype=np.int64)
    out_outcome = np.zeros(m, dtype=np.int8)      # 0 = unresolved/no fill, 1 = win, 2 = loss
    out_net_r = np.zeros(m, dtype=np.float64)
    out_entry_price = np.zeros(m, dtype=np.float64)
    out_stop_risk = np.zeros(m, dtype=np.float64)
    
    half_spread = spread_price / 2.0

    for zi in range(m):
        top = z_top[zi]; bottom = z_bottom[zi]; is_buy = z_is_buy[zi]
        height = top - bottom
        if height <= 0 or height < min_zone_height:
            continue
        mid = (top + bottom) / 2.0
        c = z_created[zi]
        end = c + max_window
        if end > n_bars:
            end = n_bars

        # Stage 1: Overshoot gate
        overshoot_idx = -1
        if is_buy:
            if high[c] > top:
                overshoot_idx = c
        else:
            if low[c] < bottom:
                overshoot_idx = c
        if overshoot_idx == -1:
            for j in range(c + 1, end):
                if is_buy:
                    if high[j] > top:
                        overshoot_idx = j; break
                else:
                    if low[j] < bottom:
                        overshoot_idx = j; break
        if overshoot_idx == -1:
            continue

        # Stage 2: Entry gate
        tag_idx = -1
        if order_mode == 0:
            # Market order: triggers on chart mid touch
            for j in range(overshoot_idx, end):
                if is_buy:
                    if low[j] <= mid:
                        tag_idx = j; break
                else:
                    if high[j] >= mid:
                        tag_idx = j; break
            if tag_idx == -1:
                continue
            entry_price = mid + half_spread if is_buy else mid - half_spread
            stop_price = bottom - half_spread if is_buy else top + half_spread
            target_price = top - half_spread if is_buy else bottom + half_spread
            
        else:
            # Resting limit order: requires deep penetration so Ask/Bid reaches mid
            for j in range(overshoot_idx, end):
                if is_buy:
                    if low[j] <= mid - half_spread:
                        tag_idx = j; break
                else:
                    if high[j] >= mid + half_spread:
                        tag_idx = j; break
            if tag_idx == -1:
                continue
            entry_price = mid
            stop_price = bottom if is_buy else top
            target_price = top if is_buy else bottom

        eff_risk = abs(entry_price - stop_price)
        if eff_risk < min_stop_price:
            continue

        # Stage 3: Exit resolution
        exit_idx = -1; outcome = 0; net_r = 0.0
        for j in range(tag_idx, end):
            if order_mode == 0:
                # Market order exit checks on chart mid
                if is_buy:
                    hit_stop = low[j] <= bottom
                    hit_target = high[j] >= top
                else:
                    hit_stop = high[j] >= top
                    hit_target = low[j] <= bottom
            else:
                # Limit order exit checks: TP requires penetration, SL triggers at threshold
                if is_buy:
                    hit_stop = low[j] <= bottom + half_spread
                    hit_target = high[j] >= top + half_spread
                else:
                    hit_stop = high[j] >= top - half_spread
                    hit_target = low[j] <= bottom - half_spread

            if hit_stop:
                exit_idx = j; outcome = 2; net_r = -1.0
                break
            elif hit_target:
                exit_idx = j; outcome = 1
                net_r = abs(target_price - entry_price) / eff_risk
                break

        if exit_idx == -1:
            continue

        out_entry_idx[zi] = tag_idx
        out_exit_idx[zi] = exit_idx
        out_outcome[zi] = outcome
        out_net_r[zi] = net_r
        out_entry_price[zi] = entry_price
        out_stop_risk[zi] = eff_risk

    return out_entry_idx, out_exit_idx, out_outcome, out_net_r, out_entry_price, out_stop_risk

def build_trades_for_mode(cache, min_zone_pips, order_mode):
    frames = []
    for inst, d in cache.items():
        pip = d["pip"]
        spread_price = SPREAD_PIPS * pip
        min_stop_price = MIN_STOP_PIPS * pip
        min_zone_height = min_zone_pips * pip

        entry_idx, exit_idx, outcome, net_r, entry_price, stop_risk = resolve_orders_comparison(
            d["z_created"], d["z_top"], d["z_bottom"], d["z_is_buy"], d["high"], d["low"],
            spread_price, min_stop_price, MAX_WINDOW, min_zone_height, order_mode
        )

        resolved = outcome > 0
        ei = entry_idx[resolved]; xi = exit_idx[resolved]
        is_buy = d["z_is_buy"][resolved]
        
        frames.append(pd.DataFrame({
            "instrument": inst,
            "entry_time": d["times"].values[ei],
            "exit_time": d["times"].values[xi],
            "entry_price": entry_price[resolved],
            "stop_risk": stop_risk[resolved],
            "net_r": net_r[resolved],
            "is_buy": is_buy,
        }))
    trades = pd.concat(frames, ignore_index=True).sort_values("entry_time").reset_index(drop=True)
    return trades

def main():
    t0 = time.time()
    print("=" * 85)
    print("MARKET ORDER ON TOUCH  vs.  RESTING LIMIT ORDER (DEEP TOUCH) BACKTEST")
    print("All 4 pairs (EURUSD, GBPUSD, EURGBP, USDJPY) across full 17-year 1-minute history")
    print("=" * 85)
    
    print("\nLoading datasets and building swing zones...")
    cache = g104.load_and_build_zones()
    
    zone_list = [5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 50.0]
    comparison_rows = []
    
    for z in zone_list:
        print(f"\nEvaluating Zone Floor: {z} pips...", flush=True)
        # 1. Market Order
        t_mkt = build_trades_for_mode(cache, z, order_mode=0)
        n_mkt = len(t_mkt)
        wr_mkt = 100.0 * (t_mkt["net_r"] > 0).sum() / n_mkt if n_mkt > 0 else 0
        gw_mkt = t_mkt[t_mkt["net_r"] > 0]["net_r"].sum()
        gl_mkt = -t_mkt[t_mkt["net_r"] <= 0]["net_r"].sum()
        pf_mkt = gw_mkt / gl_mkt if gl_mkt > 0 else 0
        avg_r_mkt = t_mkt["net_r"].mean()
        tot_r_mkt = t_mkt["net_r"].sum()
        
        # 2. Resting Limit Order
        t_lmt = build_trades_for_mode(cache, z, order_mode=1)
        n_lmt = len(t_lmt)
        wr_lmt = 100.0 * (t_lmt["net_r"] > 0).sum() / n_lmt if n_lmt > 0 else 0
        gw_lmt = t_lmt[t_lmt["net_r"] > 0]["net_r"].sum()
        gl_lmt = -t_lmt[t_lmt["net_r"] <= 0]["net_r"].sum()
        pf_lmt = gw_lmt / gl_lmt if gl_lmt > 0 else 0
        avg_r_lmt = t_lmt["net_r"].mean()
        tot_r_lmt = t_lmt["net_r"].sum()
        
        fill_ratio = (n_lmt / n_mkt) * 100.0 if n_mkt > 0 else 0
        
        # Run 2-Account Monte Carlo to $1M at 3% risk for both
        g79.RISK_PCT = 0.03
        days_mkt, _, dds_mkt, ruin_mkt = g79.run_monte_carlo_two_account(t_mkt, 500, 1_000_000.0, max_trades=30_000, seed=42)
        days_lmt, _, dds_lmt, ruin_lmt = g79.run_monte_carlo_two_account(t_lmt, 500, 1_000_000.0, max_trades=30_000, seed=42)
        
        med_yr_mkt = np.median(days_mkt) / 365.25 if len(days_mkt) else float("nan")
        med_yr_lmt = np.median(days_lmt) / 365.25 if len(days_lmt) else float("nan")
        med_dd_mkt = np.median(dds_mkt) if len(dds_mkt) else float("nan")
        med_dd_lmt = np.median(dds_lmt) if len(dds_lmt) else float("nan")
        
        comparison_rows.append({
            "min_zone": z,
            "mkt_trades": n_mkt,
            "lmt_trades": n_lmt,
            "fill_rate_pct": fill_ratio,
            "mkt_wr": wr_mkt,
            "lmt_wr": wr_lmt,
            "mkt_pf": pf_mkt,
            "lmt_pf": pf_lmt,
            "mkt_avg_r": avg_r_mkt,
            "lmt_avg_r": avg_r_lmt,
            "mkt_tot_r": tot_r_mkt,
            "lmt_tot_r": tot_r_lmt,
            "mkt_med_yrs_1m": med_yr_mkt,
            "lmt_med_yrs_1m": med_yr_lmt,
            "mkt_med_dd": med_dd_mkt,
            "lmt_med_dd": med_dd_lmt,
        })
        
        print(f"  [Zone >= {z:>4.1f} pips]")
        print(f"    Market Order:  {n_mkt:>9,} trades | WR: {wr_mkt:>6.2f}% | PF: {pf_mkt:>5.3f} | Avg R: {avg_r_mkt:>+6.4f} | Med Time to $1M: {med_yr_mkt:>5.2f}y | MaxDD: {med_dd_mkt:>5.1f}%")
        print(f"    Resting Limit: {n_lmt:>9,} trades | WR: {wr_lmt:>6.2f}% | PF: {pf_lmt:>5.3f} | Avg R: {avg_r_lmt:>+6.4f} | Med Time to $1M: {med_yr_lmt:>5.2f}y | MaxDD: {med_dd_lmt:>5.1f}%")
        print(f"    Limit Fill Rate: {fill_ratio:>5.1f}% of market touches reach deep limit threshold", flush=True)

    df_comp = pd.DataFrame(comparison_rows)
    out_csv = os.path.join(DATA_DIR, "market_vs_limit_order_comparison.csv")
    df_comp.to_csv(out_csv, index=False)
    print(f"\nSaved comparison to: {out_csv}")
    print(f"Total comparison completed in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
