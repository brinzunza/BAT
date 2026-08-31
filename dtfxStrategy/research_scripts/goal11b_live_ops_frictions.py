"""Live-operations friction checks for Hybrid-Ultra, using PRODLIVETRADING's ACTUAL
deployed guardrails (read directly from config.py / engine.py / bot.py), none of which
the backtest models at all:

  1. MAX_CONCURRENT_POSITIONS (bot.py) -- following the live-readiness audit, this cap
     was changed from a fixed headcount of 2 to a capital-based formula:
     floor(MAX_TOTAL_RISK_PCT / RISK_PCT) = floor(1.0 / 0.05) = 20 -- "unlimited,
     until 5%-risk positions would use up all the capital" instead of an arbitrary
     small number. Simulates this new cap against Hybrid-Ultra's TRUE entry/exit
     timestamps to confirm how many trades are still rejected live (real historical
     concurrency peaks at 9, comfortably under 20) and rescores WR/PF/frequency on
     the accepted subset.
  2. Rollover blackout (bot.py: 21:00-21:30 and 22:00-22:30 UTC daily) + weekend
     blackout (Fri 21:00 UTC -> Sun 21:00 UTC) -- suppresses new order placement.
     Hybrid-Ultra's backtest has no time-of-day filter at all, so this friction is
     completely unmodeled. Quantifies what fraction of trades fall in blocked windows.
  3. BACKSTOP_MULTIPLIER = 1.5 hybrid-stop design (engine.py/bot.py/oanda_client.py):
     the "real" 1R stop used in every backtest is a SOFT stop enforced by the bot's
     own software; the only broker-guaranteed stop is a native order at 1.5x that
     distance. A software failure, disconnect, or price gap means the realized loss
     on that trade is bounded by -1.5R (+ spread), not -1.0R. Sensitivity-tests how
     much PF would degrade if some fraction of losses actually realize at the
     backstop level instead of the modeled soft-stop level (there's no live trade
     history yet -- trades.db is empty -- so this can only be bounded, not measured).
"""
import os
import sys
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

import goal11_hybrid_ultra_live_readiness as g11
import goal5_rr_sweep as g5

RISK_PCT = 0.05
MAX_TOTAL_RISK_PCT = 1.0
MAX_CONCURRENT_POSITIONS = max(1, int(MAX_TOTAL_RISK_PCT / RISK_PCT))  # PRODLIVETRADING/config.py (post-fix: 20, was 2)
BACKSTOP_MULTIPLIER = 1.5     # PRODLIVETRADING/strategy/engine.py


def score(sub, label):
    if len(sub) < 2:
        print(f"  {label}: too few trades ({len(sub)})")
        return
    yrs = (sub.exit_time.max() - sub.exit_time.min()).days / 365.25
    wr = 100 * sub.is_win.mean()
    w = sub.loc[sub.net_r > 0, "net_r"].sum(); l = -sub.loc[sub.net_r <= 0, "net_r"].sum()
    pf = w / l if l > 0 else float("inf")
    print(f"  {label}: n={len(sub):,} ({len(sub)/yrs:,.0f}/yr) WR={wr:.2f}% PF={pf:.3f}")
    return dict(label=label, n=len(sub), tpy=len(sub) / yrs, wr=wr, pf=pf)


def test_concurrency_cap(t):
    print("\n" + "=" * 90)
    print(f"1. MAX_CONCURRENT_POSITIONS={MAX_CONCURRENT_POSITIONS} (live bot.py cap) vs. backtest's unlimited concurrency")
    print("=" * 90)
    t = t.sort_values("true_entry_time").reset_index(drop=True)
    import heapq
    open_exits = []  # min-heap of exit times for currently-accepted open trades
    accepted = np.zeros(len(t), dtype=bool)
    for i, row in enumerate(t.itertuples()):
        while open_exits and open_exits[0] <= row.true_entry_time:
            heapq.heappop(open_exits)
        if len(open_exits) < MAX_CONCURRENT_POSITIONS:
            heapq.heappush(open_exits, row.exit_time)
            accepted[i] = True
    t["accepted_live"] = accepted
    print(f"  Backtest trades (unlimited concurrency): {len(t):,}")
    print(f"  Would be ACCEPTED live under the cap:     {accepted.sum():,} ({100*accepted.mean():.1f}%)")
    print(f"  Would be SKIPPED live (signal dropped):   {(~accepted).sum():,} ({100*(~accepted).mean():.1f}%)")
    score(t[t.accepted_live], "Realistic live-achievable (cap-respecting) subset")
    score(t, "Backtest as reported (unlimited concurrency)          ")
    t.to_csv(os.path.join(DATA_DIR, "goal11b_concurrency_capped_trades.csv"), index=False)
    return t


def test_rollover_weekend_blackout(t):
    print("\n" + "=" * 90)
    print("2. Rollover (21:00-21:30 & 22:00-22:30 UTC daily) + weekend blackout -- unmodeled in backtest")
    print("=" * 90)
    ts = t.true_entry_time
    hour = ts.dt.hour
    minute = ts.dt.minute
    weekday = ts.dt.weekday  # 0=Mon .. 4=Fri, 5=Sat, 6=Sun
    is_rollover = ((hour == 21) & (minute < 30)) | ((hour == 22) & (minute < 30))
    is_weekend = (weekday == 5) | ((weekday == 4) & (hour >= 21)) | ((weekday == 6) & (hour < 21))
    is_blocked = is_rollover | is_weekend
    print(f"  Trades whose TRUE entry falls in the daily rollover window: {is_rollover.sum():,} ({100*is_rollover.mean():.2f}%)")
    print(f"  Trades whose TRUE entry falls in the weekend blackout:      {is_weekend.sum():,} ({100*is_weekend.mean():.2f}%)")
    print(f"  Total blocked live that the backtest silently counts:       {is_blocked.sum():,} ({100*is_blocked.mean():.2f}%)")
    t["blocked_by_ops_filters"] = is_blocked
    score(t[~is_blocked], "Ex-blackout subset (what live could actually take)")
    return t


def test_backstop_tail_risk(full_trades):
    print("\n" + "=" * 90)
    print(f"3. BACKSTOP_MULTIPLIER={BACKSTOP_MULTIPLIER} hybrid-stop -- soft-stop failure sensitivity")
    print("=" * 90)
    print("  The backtest assumes every loss costs exactly -1.0R (+ spread). Live, the only")
    print("  broker-GUARANTEED stop sits 1.5x further out; the -1.0R stop is enforced by the")
    print("  bot's own software and can miss (disconnect, crash, price gap). No live trade")
    print("  history exists yet to measure the real miss rate (trades.db is empty), so this")
    print("  bounds the sensitivity instead of measuring it:")
    t = full_trades.copy()
    losses = t.net_r <= 0
    base_pf_num = t.loc[~losses, "net_r"].sum()
    base_loss_r = -t.loc[losses, "net_r"].values  # positive R magnitudes of each loss, incl. spread cost already
    rng = np.random.default_rng(11)
    for miss_rate in [0.0, 0.01, 0.02, 0.05, 0.10]:
        miss_mask = rng.random(len(base_loss_r)) < miss_rate
        adj_loss_r = base_loss_r.copy()
        # a missed soft-stop realizes at ~BACKSTOP_MULTIPLIER x the original stop distance
        adj_loss_r[miss_mask] = base_loss_r[miss_mask] * BACKSTOP_MULTIPLIER
        pf = base_pf_num / adj_loss_r.sum()
        avg_loss = adj_loss_r.mean()
        print(f"    soft-stop miss rate {100*miss_rate:4.1f}%: PF={pf:.3f}  avg loss={-avg_loss:.3f}R"
              f"{'  <- BREAKS EVEN OR WORSE' if pf <= 1.0 else ''}")


def main():
    g11.warm_up()
    t = g11.build_hybrid_ultra_with_true_times()
    full_trades = g11.g10.build(g11.HU_N, g11.HU_RR, **g11.HU_KW)

    test_concurrency_cap(t.copy())
    test_rollover_weekend_blackout(t.copy())
    test_backstop_tail_risk(full_trades)

    print("\n" + "#" * 90)
    print("  LIVE-OPS FRICTION AUDIT COMPLETE")
    print("#" * 90)


if __name__ == "__main__":
    main()
