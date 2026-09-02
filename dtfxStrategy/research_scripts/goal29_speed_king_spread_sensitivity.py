"""Speed-King (N=7.0, R:R=0.20, age_h=0.175, unconditional re-arm, touches=4): backtest +
Monte Carlo across a spread sweep -- 0, 0.8, 1.6, 2.0, 2.5, 3.0, 4.0, 5.0 pips.

Zone detection and win/loss are spread-independent (the C++ engine's simulate_generic_ex
is run ONCE); only the cost model changes per spread level, exactly as everywhere else in
this project: cost_in_r = spread_price / stop_risk, charged on stop-outs only (matches
the live hybrid-stop design), where spread_price = spread_pips * pip_size[instrument].
1.6 pips is this project's standard modeled assumption (prod_filter_grid_search's
SPREAD_PIPS_TYPICAL) -- everything else here is a sensitivity check around it, not a
claim about what spread actually is.

For each spread level:
  1. Raw population PF/Sharpe (win rate is identical at every spread by construction --
     spread never flips a win into a loss, only makes losses deeper)
  2. Margin-constrained backtest (goal20's exact model): $200 start, 3% risk, 50x
     leverage, single continuous compounding stream, $2M sizing cap -- accept rate,
     final equity, realized max drawdown
  3. Forward Monte Carlo (25,000 sims, same 3% risk, $2M cap): median years to $1M,
     drawdown percentiles, ruin rate

Also solves for the exact BREAKEVEN spread (profit factor = 1.0) analytically: net_r is
linear in spread_pips for every losing trade (cost_in_r = spread_pips * pip_size /
stop_risk), so the aggregate loss total is linear in spread_pips too -- no need to
grid-search for it.
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

import dtfx_engine
import goal20_speed_king_margin_report as g20
from goal10_cana_nt_mc_capped import BLOCK, MAX_TRADES, CAP, T1, T10, RUIN_FLOOR

rs, pg, TRADEABLE = g20.rs, g20.pg, g20.TRADEABLE
SK_N, SK_RR, SK_AGE_H, SK_BOUNCE, SK_TOUCHES = g20.SK_N, g20.SK_RR, g20.SK_AGE_H, g20.SK_BOUNCE, g20.SK_TOUCHES

RISK_PCT = 0.03
SPREADS = [0.0, 0.8, 1.6, 2.0, 2.5, 3.0, 4.0, 5.0]


def build_raw():
    """Speed-King's spread-independent population: is_win, stop_risk, entry_price,
    entry_time, exit_time, instrument, pip_size -- net_r is computed fresh per spread
    level below, not baked in here."""
    frames = []
    for inst in TRADEABLE:
        z = rs._prep_zones(inst, pg.MIN_RETRACEMENT)
        times, lows, highs, df = rs.load_instrument(inst)
        times_ns = times.astype("datetime64[ns]").astype("int64")
        created_ns = z["created_at"].astype("datetime64[ns]").astype("int64")
        risk_pct = (SK_N * (z["top"] - z["bottom"]) / 2.0) / z["mid"]
        ok = risk_pct >= pg.MIN_RISK_PCT
        top, bottom, mid = z["top"][ok], z["bottom"][ok], z["mid"][ok]
        is_buy, start_idx, c_ns = z["is_buy"][ok], z["start_idx"][ok], created_ns[ok]
        touch_num, entry_idx, exit_idx, is_win, entry_mid, stop_risk = dtfx_engine.simulate_generic_ex(
            top, bottom, mid, is_buy, start_idx, c_ns, times_ns, lows, highs,
            int(SK_AGE_H * 3600 * 1e9), SK_N, SK_N * SK_RR, SK_BOUNCE,
            pg.PROBE_DEPTH, 1.0, SK_TOUCHES, pg.MAX_WINDOW_BARS)
        frames.append(pd.DataFrame({
            "instrument": inst,
            "entry_time": pd.to_datetime(times[entry_idx]),
            "exit_time": pd.to_datetime(times[exit_idx]),
            "entry_price": entry_mid,
            "stop_risk": stop_risk,
            "is_win": is_win,
            "pip_size": pg.PIP_SIZE[inst],
        }))
    return pd.concat(frames, ignore_index=True).sort_values("entry_time").reset_index(drop=True)


def with_spread(raw, spread_pips):
    t = raw.copy()
    spread_price = spread_pips * t.pip_size
    r_gross = np.where(t.is_win, SK_RR, -1.0)
    cost_in_r = np.where(t.is_win, 0.0, spread_price / t.stop_risk)
    t["net_r"] = r_gross - cost_in_r
    return t


def breakeven_spread(raw):
    """PF(spread) = 1 solved analytically: net_r is linear in spread_pips for every
    losing trade, so W (win total, spread-independent) and L(spread) = L0 + K*spread_pips
    are both linear/constant -- solve W = L0 + K*spread* directly."""
    losers = raw[~raw.is_win]
    W = raw.is_win.sum() * SK_RR  # every win pays exactly +RR, spread-independent
    L0 = len(losers) * 1.0        # every loss costs exactly -1.0R before spread
    K = (losers.pip_size / losers.stop_risk).sum()  # extra R of loss per pip of spread
    return (W - L0) / K


def main():
    for inst in TRADEABLE:
        rs.load_instrument(inst)
        rs._prep_zones(inst, pg.MIN_RETRACEMENT)

    print("Building Speed-King's spread-independent trade population (C++ engine)...", flush=True)
    raw = build_raw()
    years = (raw.entry_time.max() - raw.entry_time.min()).days / 365.25
    tpy = len(raw) / years
    wr = 100 * raw.is_win.mean()
    print(f"  {len(raw):,} trades ({tpy:,.0f}/yr), {raw.entry_time.min().date()} -> "
          f"{raw.entry_time.max().date()}, WR={wr:.2f}% (spread-independent, same at every level below)\n")

    be = breakeven_spread(raw)
    print(f"Breakeven spread (profit factor = 1.0, solved analytically): {be:.3f} pips")
    print(f"  (i.e. the strategy's raw edge survives spread widening up to ~{be:.1f}x today's "
          f"1.6-pip assumption before profit factor would fall to 1.0)\n")

    rows = []
    for sp in SPREADS:
        t = with_spread(raw, sp)
        w = t.loc[t.net_r > 0, "net_r"].sum(); l = -t.loc[t.net_r <= 0, "net_r"].sum()
        pf_raw = w / l if l > 0 else float("inf")

        tm, final_eq, max_dd_pct, accepted, trace = g20.simulate_margin_constrained(
            t, start=g20.START, risk_pct=RISK_PCT, max_leverage=g20.MAX_LEVERAGE,
            cap=g20.CAP, track_detail=True)
        acc = tm[tm.accepted]
        wr_acc = 100 * acc.is_win.mean() if len(acc) else float("nan")
        wA = acc.loc[acc.net_r > 0, "net_r"].sum(); lA = -acc.loc[acc.net_r <= 0, "net_r"].sum()
        pf_acc = wA / lA if lA > 0 else float("inf")
        accept_pct = 100 * len(acc) / len(tm) if len(tm) else float("nan")

        s = trace.set_index("close_time")["equity"].resample("W").last().ffill()
        wk_ret = s.pct_change().dropna()
        sharpe = (wk_ret.mean() / wk_ret.std()) * np.sqrt(52) if len(wk_ret) >= 10 and wk_ret.std() > 0 else float("nan")

        net_r_all = t.net_r.values.astype(np.float64)
        t1_raw, t10_raw, mc_dd, ruined = dtfx_engine.run_mc_capped(
            net_r_all, 25_000, BLOCK, MAX_TRADES, 23, start=g20.START, risk_frac=RISK_PCT,
            cap=CAP, target1=T1, target2=T10, ruin_floor=RUIN_FLOOR)
        t1_hit = t1_raw[t1_raw > 0]
        y1 = t1_hit / tpy
        dd_pct = mc_dd * 100

        print(f"{'='*94}\nSpread = {sp:.1f} pips\n{'='*94}")
        print(f"  Raw PF: {pf_raw:.3f}   |   Margin-constrained: accepted {len(acc):,}/{len(tm):,} "
              f"({accept_pct:.1f}%), WR={wr_acc:.2f}%, PF={pf_acc:.3f}, weekly Sharpe={sharpe:.2f}")
        print(f"  $200 start -> ${final_eq:,.2f} ({years:.1f}yr, $2M sizing cap); "
              f"realized max drawdown: {max_dd_pct:.1f}%")
        print(f"  MC (25,000 sims): median {np.median(y1):.3f}yr to $1M "
              f"(P10={np.percentile(y1,10):.3f}, P90={np.percentile(y1,90):.3f}); "
              f"max-DD median={np.median(dd_pct):.1f}% (P10={np.percentile(dd_pct,10):.1f}%, "
              f"P90={np.percentile(dd_pct,90):.1f}%, worst={dd_pct.min():.1f}%); "
              f"ruin={100*ruined.mean():.3f}%")

        rows.append(dict(
            spread_pips=sp, pf_raw=pf_raw, accept_pct=accept_pct, wr_accepted=wr_acc,
            pf_accepted=pf_acc, sharpe=sharpe, final_equity=final_eq, max_dd_realized=max_dd_pct,
            mc_median_yrs_1m=float(np.median(y1)), mc_p10_yrs_1m=float(np.percentile(y1, 10)),
            mc_p90_yrs_1m=float(np.percentile(y1, 90)), mc_median_dd=float(np.median(dd_pct)),
            mc_p10_dd=float(np.percentile(dd_pct, 10)), mc_p90_dd=float(np.percentile(dd_pct, 90)),
            mc_worst_dd=float(dd_pct.min()), mc_ruin_pct=float(100 * ruined.mean()),
        ))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DATA_DIR, "goal29_speed_king_spread_sweep.csv"), index=False)
    print(f"\n\n{'='*94}\nSUMMARY\n{'='*94}")
    print(df.to_string(index=False))
    print(f"\nBreakeven spread: {be:.3f} pips")


if __name__ == "__main__":
    main()
