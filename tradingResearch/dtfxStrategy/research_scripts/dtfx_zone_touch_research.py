"""Research & Backtesting on Zone Touch Mechanics:
1. Touch Number Analysis: Is 1st touch into zone mid best, or 2nd, 3rd, 4th+?
2. Shallow Penetration (30% touch) Analysis: Does a shallow touch and bounce
   mitigate/weaken the zone, or does it confirm responsiveness? Should a 30% touch-and-exit
   invalidate the zone?

Runs across all major instruments using Numba for fast candle-level path simulation.
"""
import os, sys
import numpy as np
import pandas as pd
from numba import njit
from scipy import stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

from dtfx_robust_search import (
    _prep_zones, load_instrument, _indicator_cache, _price_cache,
    ANALYSIS_START, FILL_SEARCH_FAST_BARS, MAX_BARS_TO_RESOLVE
)
from dtfx_somega_params import SESSION_HOUR_FILTER

BASE_PARAMS = dict(
    min_retracement=2,
    n_multiplier=3.343496,
    min_risk_pct=0.000170,
    min_fill_age_hours=3.366682
)

INSTRUMENTS = {
    "EURUSD": "lse_eurusd_1min_full.csv",
    "GBPUSD": "lse_gbpusd_1min_full.csv",
    "EURGBP": "lse_eurgbp_1min_full.csv",
    "USDJPY": "lse_usdjpy_1min_2018on.csv",
}

_zone_cache = {}


@njit
def sim_multi_touch_zones(zone_top, zone_bottom, zone_mid, zone_is_buy, zone_start_idx,
                          lows, highs, n_multiplier, min_fill_bars, max_bars):
    """Tracks up to 4 consecutive touches of zone mid before invalidation.
    Returns:
      touch_entry_idx[i, k]: entry bar index for k-th touch (k=0..3)
      touch_status[i, k]: 1=stop, 2=target, 0=none
      shallow_touch_before_mid[i]: 1 if price touched 30% depth and bounced past outer edge before touching mid
      max_depth_before_mid[i]: max penetration fraction (0.0 to 1.0) reached before 1st mid touch
    """
    n = len(zone_top)
    n_times = len(lows)
    
    # Store up to 4 touches
    touch_entry_idx = np.full((n, 4), -1, dtype=np.int64)
    touch_status = np.zeros((n, 4), dtype=np.int8)
    
    shallow_30_bounce_before_1st_mid = np.zeros(n, dtype=np.int8)
    shallow_touch_count_before_1st_mid = np.zeros(n, dtype=np.int32)
    
    for i in range(n):
        is_buy = zone_is_buy[i]
        top, bottom, mid = zone_top[i], zone_bottom[i], zone_mid[i]
        zone_width = top - bottom
        risk = n_multiplier * zone_width / 2.0
        invalidation = bottom if is_buy else top
        outer_edge = top if is_buy else bottom
        
        # 30% shallow penetration level into the zone from the outer edge
        level_30 = (top - 0.30 * zone_width) if is_buy else (bottom + 0.30 * zone_width)
        
        start = zone_start_idx[i]
        if start >= n_times:
            continue
            
        earliest_valid_entry = start + min_fill_bars
        if earliest_valid_entry >= n_times:
            continue
            
        stop_level = mid - risk if is_buy else mid + risk
        target_level = mid + risk if is_buy else mid - risk
        
        # Phase 1: Track shallow 30% touches and bounces prior to the first valid 50% mid touch
        touched_30 = False
        bounce_after_30 = False
        num_30_touches = 0
        
        current_bar = start
        while current_bar < n_times:
            # Check invalidation
            if is_buy and lows[current_bar] <= invalidation:
                break
            elif (not is_buy) and highs[current_bar] >= invalidation:
                break
                
            # Check 30% penetration
            if is_buy and lows[current_bar] <= level_30:
                if not touched_30:
                    touched_30 = True
                    num_30_touches += 1
            elif (not is_buy) and highs[current_bar] >= level_30:
                if not touched_30:
                    touched_30 = True
                    num_30_touches += 1
                    
            # Check if bounced back past outer edge after touching 30%
            if touched_30:
                if is_buy and highs[current_bar] > outer_edge:
                    bounce_after_30 = True
                    touched_30 = False # reset to detect next bounce
                elif (not is_buy) and lows[current_bar] < outer_edge:
                    bounce_after_30 = True
                    touched_30 = False
                    
            # Check 50% mid touch
            mid_hit = (lows[current_bar] <= mid) if is_buy else (highs[current_bar] >= mid)
            if mid_hit:
                if current_bar >= earliest_valid_entry:
                    # Valid first mid touch!
                    shallow_30_bounce_before_1st_mid[i] = 1 if bounce_after_30 else 0
                    shallow_touch_count_before_1st_mid[i] = num_30_touches
                    break
                else:
                    # Touched mid too early (before min_fill_bars) -> invalidates 1st touch under Omega rule
                    break
                    
            current_bar += 1
            
        if current_bar >= n_times:
            continue
            
        # Phase 2: Simulate Touch 1, 2, 3, 4 sequentially
        touch_num = 0
        sim_bar = current_bar
        
        while touch_num < 4 and sim_bar < n_times:
            # sim_bar is the entry bar for touch_num
            touch_entry_idx[i, touch_num] = sim_bar
            
            # Resolve this trade outcome (SL vs TP vs Invalidation)
            after_start = sim_bar + 1
            after_end = min(after_start + max_bars, n_times)
            
            sl_bar, tp_bar = -1, -1
            resolved_bar = -1
            
            for j in range(after_start, after_end):
                # Check stop / invalidation
                if is_buy:
                    if sl_bar == -1 and lows[j] <= stop_level:
                        sl_bar = j
                    if tp_bar == -1 and highs[j] >= target_level:
                        tp_bar = j
                else:
                    if sl_bar == -1 and highs[j] >= stop_level:
                        sl_bar = j
                    if tp_bar == -1 and lows[j] <= target_level:
                        tp_bar = j
                        
                if sl_bar != -1 and tp_bar != -1:
                    break
                    
            if sl_bar == -1 and tp_bar == -1:
                # Unresolved within max_bars -> end
                break
                
            if sl_bar != -1 and (tp_bar == -1 or sl_bar <= tp_bar):
                touch_status[i, touch_num] = 1 # stopped out
                # If stopped out, zone is broken -> no further touches possible
                break
            else:
                touch_status[i, touch_num] = 2 # target hit!
                resolved_bar = tp_bar
                
            # If target hit, look for next re-entry into mid (next touch)
            touch_num += 1
            if touch_num >= 4 or resolved_bar == -1:
                break
                
            # Scan forward from resolved_bar for the NEXT mid touch
            next_mid_bar = -1
            for j in range(resolved_bar + 1, n_times):
                # If invalidation hit before next mid, zone dies
                if is_buy and lows[j] <= invalidation:
                    break
                elif (not is_buy) and highs[j] >= invalidation:
                    break
                    
                mid_hit = (lows[j] <= mid) if is_buy else (highs[j] >= mid)
                if mid_hit:
                    next_mid_bar = j
                    break
                    
            if next_mid_bar == -1:
                break
            sim_bar = next_mid_bar

    return touch_entry_idx, touch_status, shallow_30_bounce_before_1st_mid, shallow_touch_count_before_1st_mid


def load_instrument_custom(instrument_name, filename):
    if instrument_name in _price_cache:
        return _price_cache[instrument_name]
    path = os.path.join(DATA_DIR, filename)
    print(f"Loading {instrument_name} from {filename} ...", flush=True)
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    df.index = df.index.tz_localize(None)
    df = df[~df.index.duplicated(keep="first")]
    df = df[df.index >= ANALYSIS_START]
    df = df[["open", "high", "low", "close"]]
    
    times = df.index.values
    lows = np.ascontiguousarray(df["low"].values, dtype=np.float64)
    highs = np.ascontiguousarray(df["high"].values, dtype=np.float64)
    _price_cache[instrument_name] = (times, lows, highs, df)
    print(f"  {len(df):,} bars, {df.index.min()} to {df.index.max()}", flush=True)
    return _price_cache[instrument_name]


def prep_zones_custom(instrument_name, filename):
    key = (instrument_name, BASE_PARAMS["min_retracement"])
    if key in _zone_cache:
        return _zone_cache[key]
    times, lows, highs, df = load_instrument_custom(instrument_name, filename)
    from dtfx_eurusd_backtest_strict import find_pivots_strict
    from dtfx_eurusd_backtest import run_structure_fsm
    zigzag = find_pivots_strict(df, BASE_PARAMS["min_retracement"])
    _, zones = run_structure_fsm(zigzag)
    
    n = len(zones)
    top = np.array([z["top"] for z in zones], dtype=np.float64)
    bottom = np.array([z["bottom"] for z in zones], dtype=np.float64)
    mid = np.array([z["mid"] for z in zones], dtype=np.float64)
    is_buy = np.array([z["side"] == "buy" for z in zones], dtype=np.bool_)
    created_at = np.array([np.datetime64(z["created_at"]) for z in zones])
    start_idx = np.searchsorted(times, created_at, side="right").astype(np.int64)
    
    packed = dict(top=top, bottom=bottom, mid=mid, is_buy=is_buy, start_idx=start_idx, created_at=created_at)
    _zone_cache[key] = packed
    print(f"  {instrument_name} MIN_RETRACEMENT={BASE_PARAMS['min_retracement']}: {n:,} zones", flush=True)
    return packed


def run_research(instrument_name, filename):
    print(f"\nAnalyzing {instrument_name}...")
    times, lows, highs, df = load_instrument_custom(instrument_name, filename)
    z = prep_zones_custom(instrument_name, filename)
    
    min_fill_bars = int(BASE_PARAMS["min_fill_age_hours"] * 60)
    
    entry_indices, statuses, shallow_bounces, shallow_counts = sim_multi_touch_zones(
        z["top"], z["bottom"], z["mid"], z["is_buy"], z["start_idx"],
        lows, highs, float(BASE_PARAMS["n_multiplier"]), min_fill_bars, MAX_BARS_TO_RESOLVE
    )
    
    risk = BASE_PARAMS["n_multiplier"] * (z["top"] - z["bottom"]) / 2.0
    risk_pct = risk / z["mid"]
    
    # ── PART 1: TOUCH NUMBER ANALYSIS (1st vs 2nd vs 3rd vs 4th) ──────────────
    touch_records = []
    
    for touch_idx in range(4):
        entries = entry_indices[:, touch_idx]
        valid = (entries >= 0) & (risk_pct >= BASE_PARAMS["min_risk_pct"])
        
        # Apply SOMEGA session hour filter if applicable
        allowed_hours = SESSION_HOUR_FILTER.get(instrument_name)
        if allowed_hours is not None:
            entry_hours = pd.to_datetime(times[np.where(valid, entries, 0)]).hour
            hour_ok = np.isin(entry_hours, list(allowed_hours))
            valid = valid & hour_ok
            
        n_trades = valid.sum()
        if n_trades == 0:
            continue
            
        stat = statuses[:, touch_idx][valid]
        wins = (stat == 2).sum()
        losses = (stat == 1).sum()
        wr = wins / n_trades if n_trades > 0 else 0.0
        pf = wins / losses if losses > 0 else float("inf")
        sum_r = wins * 1.0 - losses * 1.0
        avg_r = sum_r / n_trades if n_trades > 0 else 0.0
        
        touch_records.append({
            "instrument": instrument_name,
            "touch_number": f"{touch_idx + 1}{'st' if touch_idx==0 else 'nd' if touch_idx==1 else 'rd' if touch_idx==2 else 'th'} Touch",
            "trades": n_trades,
            "wins": wins,
            "losses": losses,
            "win_rate": wr,
            "profit_factor": pf,
            "avg_r": avg_r,
            "sum_r": sum_r,
        })
        
    # ── PART 2: SHALLOW 30% PENETRATION & BOUNCE ANALYSIS ─────────────────────
    # For 1st mid touch: compare trades that had prior 30% shallow bounce vs clean entries
    entries_1st = entry_indices[:, 0]
    valid_1st = (entries_1st >= 0) & (risk_pct >= BASE_PARAMS["min_risk_pct"])
    
    allowed_hours = SESSION_HOUR_FILTER.get(instrument_name)
    if allowed_hours is not None:
        entry_hours = pd.to_datetime(times[np.where(valid_1st, entries_1st, 0)]).hour
        valid_1st = valid_1st & np.isin(entry_hours, list(allowed_hours))
        
    idx_1st = np.where(valid_1st)[0]
    df_shallow = pd.DataFrame({
        "status": statuses[idx_1st, 0],
        "had_30_bounce": shallow_bounces[idx_1st],
        "shallow_count": shallow_counts[idx_1st],
        "win": statuses[idx_1st, 0] == 2,
    })
    
    # Clean vs 30% Bounce Breakdown
    clean = df_shallow[df_shallow["had_30_bounce"] == 0]
    bounced = df_shallow[df_shallow["had_30_bounce"] == 1]
    
    shallow_records = [
        {
            "instrument": instrument_name,
            "condition": "Clean Entry (No prior 30% bounce)",
            "trades": len(clean),
            "win_rate": clean["win"].mean() if len(clean) else 0.0,
            "profit_factor": clean["win"].sum() / (len(clean) - clean["win"].sum()) if (len(clean) - clean["win"].sum()) > 0 else float("inf"),
            "avg_r": (clean["win"].sum() - (len(clean) - clean["win"].sum())) / len(clean) if len(clean) else 0.0,
        },
        {
            "instrument": instrument_name,
            "condition": "Prior 30% Shallow Bounce Happened",
            "trades": len(bounced),
            "win_rate": bounced["win"].mean() if len(bounced) else 0.0,
            "profit_factor": bounced["win"].sum() / (len(bounced) - bounced["win"].sum()) if (len(bounced) - bounced["win"].sum()) > 0 else float("inf"),
            "avg_r": (bounced["win"].sum() - (len(bounced) - bounced["win"].sum())) / len(bounced) if len(bounced) else 0.0,
        }
    ]
    
    return touch_records, shallow_records


def main():
    print("="*80)
    print("  RESEARCH: ZONE TOUCH NUMBER & SHALLOW 30% PENETRATION DYNAMICS")
    print("="*80)
    
    all_touches = []
    all_shallow = []
    
    for name, fname in INSTRUMENTS.items():
        t_recs, s_recs = run_research(name, fname)
        all_touches.extend(t_recs)
        all_shallow.extend(s_recs)
        
    df_touch = pd.DataFrame(all_touches)
    df_shallow = pd.DataFrame(all_shallow)
    
    print("\n" + "="*90)
    print("  1. TOUCH NUMBER PERFORMANCE (1st vs 2nd vs 3rd vs 4th Touch)")
    print("="*90)
    df_touch_disp = df_touch.copy()
    df_touch_disp["Win Rate"] = df_touch_disp["win_rate"].apply(lambda x: f"{x*100:.2f}%")
    df_touch_disp["Profit Factor"] = df_touch_disp["profit_factor"].apply(lambda x: f"{x:.3f}")
    df_touch_disp["Avg R"] = df_touch_disp["avg_r"].apply(lambda x: f"{x:+.4f}")
    df_touch_disp["Sum R"] = df_touch_disp["sum_r"].apply(lambda x: f"{x:+,.1f}")
    cols_t = ["instrument", "touch_number", "trades", "Win Rate", "Profit Factor", "Avg R", "Sum R"]
    print(df_touch_disp[cols_t].to_string(index=False))
    
    # Combined Touches Summary
    print("\n  COMBINED TOUCH NUMBER AGGREGATE:")
    agg_touch = df_touch.groupby("touch_number").agg(
        total_trades=("trades", "sum"),
        total_wins=("wins", "sum"),
        total_losses=("losses", "sum"),
        total_sum_r=("sum_r", "sum")
    )
    agg_touch["Win Rate"] = agg_touch["total_wins"] / agg_touch["total_trades"]
    agg_touch["Profit Factor"] = agg_touch["total_wins"] / agg_touch["total_losses"]
    agg_touch["Avg R"] = agg_touch["total_sum_r"] / agg_touch["total_trades"]
    print(agg_touch[["total_trades", "Win Rate", "Profit Factor", "Avg R", "total_sum_r"]].to_string())
    
    print("\n" + "="*90)
    print("  2. SHALLOW 30% PENETRATION & BOUNCE ANALYSIS")
    print("="*90)
    df_shallow_disp = df_shallow.copy()
    df_shallow_disp["Win Rate"] = df_shallow_disp["win_rate"].apply(lambda x: f"{x*100:.2f}%")
    df_shallow_disp["Profit Factor"] = df_shallow_disp["profit_factor"].apply(lambda x: f"{x:.3f}")
    df_shallow_disp["Avg R"] = df_shallow_disp["avg_r"].apply(lambda x: f"{x:+.4f}")
    cols_s = ["instrument", "condition", "trades", "Win Rate", "Profit Factor", "Avg R"]
    print(df_shallow_disp[cols_s].to_string(index=False))
    
    # Save CSVs
    df_touch.to_csv(os.path.join(DATA_DIR, "research_touch_number_results.csv"), index=False)
    df_shallow.to_csv(os.path.join(DATA_DIR, "research_shallow_penetration_results.csv"), index=False)
    print(f"\nSaved research CSVs to {DATA_DIR}")

if __name__ == "__main__":
    main()
