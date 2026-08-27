"""Re-runs the session/liquidity hour-filtering comparison (dtfx_omega_session_liquidity_
filter.py) under the CORRECTED asymmetric cost model established in
dtfx_omega_stop_cost_ratio.py: spread is charged only on stop-out trades (market-order
exit), never on limit-filled entries or take-profit exits. The original session analysis
used a uniform per-trade cost, which Part 1 of that follow-up showed was badly overstating
real cost (EUR/USD flipped from Sharpe -0.25 to +3.45 at the same 1.6-pip spread just from
this correction). This checks whether hour-filtering is still worth doing -- and how much
it's worth -- now that the "all hours" baseline is no longer artificially crippled.

Three cost scenarios compared side by side, same trade population each time:
  - flat        : 1.6 pips (this account's live-measured spread) on every stop-out, any hour
  - mild session: session-multiplier table from the original analysis, tightest at the
                  London/NY overlap, on every stop-out
  - aggressive session: same table, wider multipliers, on every stop-out
"""
import os
import numpy as np
import pandas as pd

from dtfx_robust_search import (
    _prep_zones, load_instrument, _simulate_zones_numba, _indicator_cache, score_trades, DATA_DIR
)

BASE_PARAMS = dict(min_retracement=2, n_multiplier=3.343496, min_fill_age_hours=3.366682)
LIVE_MIN_RISK_PCT = 0.000170
FILL_SEARCH_FAST_BARS = 60 * 24 * 5
MAX_BARS_TO_RESOLVE = 60 * 24 * 30
PIP_SIZE = 0.0001
FLAT_SPREAD_PIPS = 1.6  # this account's live-measured EUR/USD spread
BASELINE_SPREAD_PIPS = 1.2  # session-scenario floor, same as the original analysis
SESSION_MULTIPLIERS = {
    "mild":       {"overlap": 1.0, "single": 1.3, "asian": 1.6, "thin": 2.0},
    "aggressive": {"overlap": 1.0, "single": 1.6, "asian": 2.5, "thin": 4.0},
}
INSTRUMENTS = ["EURUSD", "GBPUSD", "EURGBP"]


def session_bucket(hour):
    if 12 <= hour < 16:
        return "overlap"
    if 7 <= hour < 12 or 16 <= hour < 21:
        return "single"
    if 0 <= hour < 7:
        return "asian"
    return "thin"


def build_trades_with_detail(instrument):
    z = _prep_zones(instrument, BASE_PARAMS["min_retracement"])
    times, lows, highs, df = load_instrument(instrument)
    rsi14, atr_ratio = _indicator_cache[instrument]

    entry_idx, status = _simulate_zones_numba(
        z["top"], z["bottom"], z["mid"], z["is_buy"], z["start_idx"],
        lows, highs, float(BASE_PARAMS["n_multiplier"]), FILL_SEARCH_FAST_BARS, MAX_BARS_TO_RESOLVE)

    filled = entry_idx >= 0
    risk = BASE_PARAMS["n_multiplier"] * (z["top"] - z["bottom"]) / 2.0
    risk_pct_of_price = risk / z["mid"]
    entry_times_ns = times[np.where(filled, entry_idx, 0)]
    fill_hours = (entry_times_ns - z["created_at"]) / np.timedelta64(1, "h")

    keep = (filled & (risk_pct_of_price >= LIVE_MIN_RISK_PCT)
            & (fill_hours >= BASE_PARAMS["min_fill_age_hours"]))
    r14 = rsi14[np.where(filled, entry_idx, 0)]
    ar = atr_ratio[np.where(filled, entry_idx, 0)]
    rule_ok = ~np.isnan(r14) & ~np.isnan(ar) & ~((r14 >= 45) & (r14 <= 55)) & ~(ar < 0.7)
    keep = keep & rule_ok

    idx = np.where(keep)[0]
    entry_times = pd.to_datetime(entry_times_ns[idx])
    out = pd.DataFrame({
        "entry_time": entry_times,
        "r_gross": np.where(status[idx] == 2, 1.0, -1.0),
        "is_stop": status[idx] == 1,
        "risk_price": risk[idx],
    }).sort_values("entry_time").reset_index(drop=True)
    out["hour"] = out["entry_time"].dt.hour
    out["session"] = out["hour"].apply(session_bucket)
    return out


def net_r_flat(trades, spread_pips):
    cost_price = spread_pips * PIP_SIZE
    cost_in_r = cost_price / trades["risk_price"]
    return trades["r_gross"] - np.where(trades["is_stop"], cost_in_r, 0.0)


def net_r_session(trades, scenario):
    mult = trades["session"].map(SESSION_MULTIPLIERS[scenario])
    spread_pips = mult * BASELINE_SPREAD_PIPS
    cost_price = spread_pips * PIP_SIZE
    cost_in_r = cost_price / trades["risk_price"]
    return trades["r_gross"] - np.where(trades["is_stop"], cost_in_r, 0.0)


def headline_ab(trades, r_col):
    variants = {
        "all_hours": trades,
        "overlap_only": trades[trades["session"] == "overlap"],
        "overlap_or_single": trades[trades["session"].isin(["overlap", "single"])],
    }
    rows = []
    for name, sub in variants.items():
        scored = sub.rename(columns={r_col: "net_r"})[["entry_time", "net_r"]]
        score, stats = score_trades(scored)
        row = dict(variant=name, n_trades=len(sub), score=score)
        row.update(stats or {})
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    all_rows = []
    for instrument in INSTRUMENTS:
        print(f"\n{'='*78}\n{instrument} -- asymmetric cost model (stops only)\n{'='*78}")
        trades = build_trades_with_detail(instrument)

        for cost_label, r_col in [
            ("flat_1.6pip", "net_r_flat"),
            ("mild_session", "net_r_mild"),
            ("aggressive_session", "net_r_aggr"),
        ]:
            if cost_label == "flat_1.6pip":
                trades[r_col] = net_r_flat(trades, FLAT_SPREAD_PIPS)
            elif cost_label == "mild_session":
                trades[r_col] = net_r_session(trades, "mild")
            else:
                trades[r_col] = net_r_session(trades, "aggressive")

            ab = headline_ab(trades, r_col)
            ab.insert(0, "cost_model", cost_label)
            ab.insert(0, "instrument", instrument)
            print(f"\n  --- {cost_label} ---")
            cols = ["variant", "n_trades", "score", "ann_return", "maxdd", "calmar",
                    "ann_sharpe", "profit_factor", "win_rate", "avg_r"]
            print(ab[cols].to_string(index=False))
            all_rows.append(ab)

    summary = pd.concat(all_rows, ignore_index=True)
    out_path = os.path.join(DATA_DIR, "dtfx_omega_session_filter_corrected_summary.csv")
    summary.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
