"""Apex-Ultra (N=4.0, RR=0.15, age=0.08h, 20 touches), single-symbol (goal51's new
closest-price concurrency resolver), backtested standalone per symbol on 2026 YTD data,
starting fresh from two $100 sub-accounts -- as opposed to goal51's full-holdout numbers,
which start compounding from the beginning of each symbol's holdout window.

For each of the 4 symbols independently:
1. 2026 YTD backtest, $100/$100 fresh start -- trade acceptance rate, trade count, WR,
   PF, ending equity, max drawdown.
2. Monte Carlo (goal30's block-bootstrap, reused unmodified -- it operates generically on
   any entry/exit/price/outcome arrays, so it's agnostic to which concurrency-resolution
   method produced them) resampled from the 2026 trade population: years to $1M, given
   2026's own edge/frequency held.
3. Sharpe / excess kurtosis (goal31's project-standard fixed-1%-risk daily-curve
   convention) on the 2026 trade stream specifically.
4. Day/week/month/year positive-rate (goal31's period-reset methodology) computed over
   the FULL resolved holdout population, not just 2026 YTD -- 2026 alone is ~8 months,
   too few weeks/months and only one (partial) year to say anything about "% of periods
   positive" with statistical meaning. Labeled separately from the 2026-specific numbers
   above, not blended with them.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

import goal29_rerank_1_per_side as g29
import goal30_apex_king_deployment as g30
import goal31_apex_stats as g31
import goal51_single_symbol_closest_price_backtest as g51

APEX_ULTRA_CFG = dict(n=4.0, rr=0.15, age_h=0.08, bounce=False, touches=20)
RISK_PCT = 0.06
YEAR_START = pd.Timestamp("2026-01-01")
MC_PATHS = 1000
MC_SEED = 42


def get_symbol_trades(inst):
    """Full-history raw signal candidates and closest-price-resolved accepted trades for
    one symbol, both sides combined."""
    g29.rs.load_instrument(inst)
    g29.rs._prep_zones(inst, g29.pg.MIN_RETRACEMENT)
    price_at = g51._price_lookup(inst)
    raw_frames, accepted_frames = [], []
    for side_is_buy in (True, False):
        touches = g51.build_touches_with_metadata(
            inst, APEX_ULTRA_CFG["n"], APEX_ULTRA_CFG["rr"], APEX_ULTRA_CFG["age_h"],
            APEX_ULTRA_CFG["bounce"], APEX_ULTRA_CFG["touches"], side_is_buy)
        raw_frames.append(touches)
        accepted_frames.append(g51.resolve_closest_price(touches, APEX_ULTRA_CFG["age_h"], price_at))
    raw = pd.concat(raw_frames, ignore_index=True).sort_values("entry_time").reset_index(drop=True)
    accepted = pd.concat(accepted_frames, ignore_index=True).sort_values("entry_time").reset_index(drop=True)
    return raw, accepted


def backtest_2026(raw, accepted, risk_pct=RISK_PCT):
    raw_2026 = raw[raw.entry_time >= YEAR_START]
    acc_candidate_2026 = accepted[accepted.entry_time >= YEAR_START]
    if acc_candidate_2026.empty:
        return None

    t, eq_buy, eq_sell, max_dd, _ = g30.simulate_two_account(acc_candidate_2026, risk_pct)
    final = t[t.accepted].sort_values("entry_time").reset_index(drop=True)
    n = len(final)
    wins = final[final.net_r > 0]
    losses = final[final.net_r <= 0]
    wr = 100 * len(wins) / n if n else float("nan")
    gross_win = wins.net_r.sum()
    gross_loss = -losses.net_r.sum()
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
    span_days = max((final.entry_time.max() - YEAR_START).days, 1) if n else 1
    acceptance_rate = 100 * len(acc_candidate_2026) / len(raw_2026) if len(raw_2026) else float("nan")

    ann_sharpe, exk, calmar, maxdd_ref, n_obs = g31.sharpe_and_kurtosis(final) if n >= 10 else (
        float("nan"), float("nan"), float("nan"), float("nan"), 0)

    return dict(
        raw_signals_2026=len(raw_2026), accepted_trades_2026=n,
        acceptance_rate_pct=acceptance_rate, trades_per_day=n / span_days,
        win_rate_pct=wr, profit_factor=pf, end_buy=eq_buy, end_sell=eq_sell,
        end_combined=eq_buy + eq_sell, max_drawdown_pct=max_dd,
        ann_sharpe=ann_sharpe, excess_kurtosis=exk, calmar=calmar,
        acc_candidate_2026=acc_candidate_2026,
    )


def monte_carlo_2026(acc_candidate_2026, risk_pct=RISK_PCT, n_paths=MC_PATHS, seed=MC_SEED):
    if len(acc_candidate_2026) < 30:
        return None
    t1s, t10s, dds, ruins, accs = g30.run_mc(acc_candidate_2026, risk_pct, n_paths, seed)
    return dict(
        median_years_to_1m=np.nanmedian(t1s[t1s >= 0]) if (t1s >= 0).any() else float("nan"),
        p10_years_to_1m=np.nanpercentile(t1s[t1s >= 0], 10) if (t1s >= 0).any() else float("nan"),
        p90_years_to_1m=np.nanpercentile(t1s[t1s >= 0], 90) if (t1s >= 0).any() else float("nan"),
        pct_reaching_1m=100 * (t1s >= 0).mean(),
        median_max_drawdown_pct=np.median(dds),
        pct_ruined=100 * ruins.mean(),
    )


def period_positive_rates(accepted, risk_pct=RISK_PCT):
    """Full holdout history (not just 2026) -- see module docstring for why."""
    out = {}
    for label, freq in [("Day", "D"), ("Week", "W"), ("Month", "M"), ("Year", "Y")]:
        pct, n_periods, _ = g31.period_positive_pct(accepted, freq, risk_pct=risk_pct)
        out[label] = (pct, n_periods)
    return out


def main():
    rows = []
    for inst in g29.TRADEABLE:
        print(f"\n{'='*90}\n{inst}\n{'='*90}")
        t0 = time.time()
        raw, accepted = get_symbol_trades(inst)
        print(f"  {len(raw):,} raw signal candidates, {len(accepted):,} accepted "
              f"(full history, closest-price policy) [{time.time()-t0:.0f}s]")

        bt = backtest_2026(raw, accepted)
        if bt is None:
            print("  No 2026 trades -- skipping.")
            continue
        acc_2026 = bt.pop("acc_candidate_2026")
        print(f"  2026 YTD: {bt['accepted_trades_2026']:,} trades "
              f"(acceptance {bt['acceptance_rate_pct']:.1f}%), "
              f"WR={bt['win_rate_pct']:.2f}%, PF={bt['profit_factor']:.3f}, "
              f"end=${bt['end_combined']:,.2f}, DD={bt['max_drawdown_pct']:.1f}%, "
              f"Sharpe={bt['ann_sharpe']:.2f}, ExKurt={bt['excess_kurtosis']:+.2f}, "
              f"Calmar={bt['calmar']:.2f}")

        t0 = time.time()
        mc = monte_carlo_2026(acc_2026)
        if mc:
            print(f"  MC ({MC_PATHS} paths, resampled from 2026): median {mc['median_years_to_1m']:.2f}y "
                  f"to $1M (P10={mc['p10_years_to_1m']:.2f}y, P90={mc['p90_years_to_1m']:.2f}y), "
                  f"{mc['pct_reaching_1m']:.1f}% of paths reach it, "
                  f"median path max DD={mc['median_max_drawdown_pct']:.1f}%, "
                  f"ruined={mc['pct_ruined']:.1f}% [{time.time()-t0:.0f}s]")
        else:
            print("  Too few 2026 trades for a meaningful MC.")
            mc = {}

        t0 = time.time()
        periods = period_positive_rates(accepted)
        print("  Positive-period % (full holdout history, period-reset):")
        for label, (pct, n_periods) in periods.items():
            print(f"    {label:6s}: {pct:5.1f}% positive (n={n_periods})")
        print(f"    [{time.time()-t0:.0f}s]")

        row = dict(instrument=inst, **bt, **{f"mc_{k}": v for k, v in mc.items()},
                   **{f"pct_positive_{label.lower()}": pct for label, (pct, _) in periods.items()},
                   **{f"n_{label.lower()}s": n for label, (_, n) in periods.items()})
        rows.append(row)

    df = pd.DataFrame(rows)
    out_path = os.path.join(DATA_DIR, "goal52_apex_ultra_2026_per_symbol.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    return df


if __name__ == "__main__":
    main()
