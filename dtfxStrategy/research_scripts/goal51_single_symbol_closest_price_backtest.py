"""Backtest architecture update for PRODLIVETRADING's 2026-09-03 concurrency change
(commits 2616aa4/9e42f40): a resting (unfilled) limit order no longer occupies the
MAX_CONCURRENT_PER_SIDE slot -- only a broker-confirmed fill does. When a symbol/side
already has its one resting order and a new zone matures, the bot now keeps whichever of
the two sits closer to current price (cancelling the other), instead of first-come-first-
served. The old goal29/goal30 pipeline (`simulate_two_account`) modeled the OLD policy: it
resolves each symbol's zone touches independently (ignoring concurrency at the touch level
entirely), then walks the combined multi-symbol trade stream in entry-time order accepting
at most one open position per side, permanently discarding anything that loses the race --
correct under the old policy, where "rejected" and "never placed" were the same thing.

Under the new policy those diverge: a rejected signal's order still gets placed and rests,
and can still fill later. Modeling that exactly across multiple symbols racing on real
price action (see the AskUserQuestion in this session) is a materially bigger rebuild --
the user chose instead to drop multi-symbol portfolio trading entirely and evaluate one
symbol only, which removes the cross-symbol racing/sibling-cancellation question outright:
with one symbol, "one resting order per side, closest price wins" is the WHOLE story.

This module still needs to get the "closest price wins, no first-come-first-served" part
right for a single symbol, though, and that requires knowing WHEN a zone's touch became
eligible to rest (its dispatch time), not just when it eventually filled -- data the
existing pipeline computes internally (dtfx_engine.simulate_generic_ex has no zone-index
output, so touch_num alone can't be regrouped by zone) but never keeps in its trade
DataFrame. `entry_price` (== a zone's own fixed `mid`) is reused here as a zone-identity
key instead of requiring an engine change: a re-armed zone always fills its later touches
at the exact same price, so grouping same-instrument/side/entry_price touches and sorting
by touch_num recovers each zone's own touch sequence without needing an explicit zone id.
Dispatch time is then: created_at + MIN_FILL_AGE for a zone's first touch, and the previous
touch's own exit_time for every touch after that (unconditional re-arm becomes eligible
again immediately once the prior touch resolves -- matches strategy/engine.py's actual
behavior in PRODLIVETRADING).

Zero lookahead: the resolver only ever uses already-known STRUCTURAL timestamps (created_at,
dispatch_time, entry_time, exit_time) to decide who wins a race -- never a touch's own
win/loss outcome, which stays unknown until after it's decided to actually happen.
"""
import os
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

import goal29_rerank_1_per_side as g29
import goal30_apex_king_deployment as g30

rs, pg = g29.rs, g29.pg
dtfx_engine = rs.dtfx_engine


def build_touches_with_metadata(inst, n, rr, age_h, bounce, touches, side_is_buy):
    """Like goal29.build_trades_with_side, scoped to one instrument/side, but also keeps
    touch_num and created_at -- both dropped by the original, both needed to derive
    dispatch times below."""
    z = rs._prep_zones(inst, pg.MIN_RETRACEMENT)
    times, lows, highs, df = rs.load_instrument(inst)
    times_ns = times.astype("datetime64[ns]").astype("int64")
    created_ns = z["created_at"].astype("datetime64[ns]").astype("int64")

    target_mult = n * rr
    risk_pct = (n * (z["top"] - z["bottom"]) / 2.0) / z["mid"]
    risk_ok = risk_pct >= pg.MIN_RISK_PCT
    mask = risk_ok & (z["is_buy"] == side_is_buy)
    top, bottom, mid = z["top"][mask], z["bottom"][mask], z["mid"][mask]
    is_buy_arr, start_idx, c_ns = z["is_buy"][mask], z["start_idx"][mask], created_ns[mask]
    if len(top) == 0:
        return pd.DataFrame(columns=["instrument", "is_buy", "touch_num", "created_at",
                                      "entry_time", "exit_time", "entry_price", "stop_risk",
                                      "net_r", "is_win"])

    touch_num, entry_idx, exit_idx, is_win, entry_mid, stop_risk = dtfx_engine.simulate_generic_ex(
        top, bottom, mid, is_buy_arr, start_idx, c_ns, times_ns, lows, highs,
        int(age_h * 3600 * 1e9), n, target_mult, bounce, pg.PROBE_DEPTH, 1.0,
        touches, pg.MAX_WINDOW_BARS)
    if len(entry_idx) == 0:
        return pd.DataFrame(columns=["instrument", "is_buy", "touch_num", "created_at",
                                      "entry_time", "exit_time", "entry_price", "stop_risk",
                                      "net_r", "is_win"])

    pip = pg.PIP_SIZE[inst]
    spread_price = pg.SPREAD_PIPS_TYPICAL * pip
    r_gross = np.where(is_win, rr, -1.0)
    cost_in_r = np.where(is_win, 0.0, spread_price / stop_risk)
    net_r = r_gross - cost_in_r

    # simulate_generic_ex has no zone-index output (per goal29's own docstring), so a
    # touch can't be traced back to its zone by output position. mid and created_at ARE
    # paired 1:1 in the input zone arrays (mid[k]/c_ns[k] are the same zone k, preserved
    # through the mask above) -- but mid alone is NOT a safe lookup key: with 100k+ zones
    # over a bounded price range, plenty of unrelated zones at genuinely different times
    # land on the same rounded price (confirmed: a naive last-write-wins dict on rounded
    # mid alone put ~39% of touches' created_at AFTER their own entry_time, sometimes by
    # years -- collisions, not a rare edge case here). A zone can never be touched before
    # it's created, so for each price level keep every candidate creation time and pick
    # the LATEST one that's still <= this touch's own entry_time -- exact whenever at most
    # one same-priced zone was ever alive at a time (the common case; ambiguity only
    # remains if two same-priced zones were BOTH already alive simultaneously).
    entry_time_all = times_ns[entry_idx]
    mid_key = np.round(mid, 8)
    created_by_price = {}
    for k in np.argsort(mid_key):
        created_by_price.setdefault(mid_key[k], []).append(c_ns[k])
    for lst in created_by_price.values():
        lst.sort()
    created_at_col = np.empty(len(entry_mid), dtype="int64")
    for i, (m, et) in enumerate(zip(np.round(entry_mid, 8), entry_time_all)):
        candidates = created_by_price.get(m)
        pos = np.searchsorted(candidates, et, side="right") - 1
        created_at_col[i] = candidates[pos] if pos >= 0 else candidates[0]

    return pd.DataFrame({
        "instrument": inst, "is_buy": side_is_buy, "touch_num": touch_num,
        "created_at": pd.to_datetime(created_at_col),
        "entry_time": pd.to_datetime(times[entry_idx]),
        "exit_time": pd.to_datetime(times[exit_idx]),
        "entry_price": np.round(entry_mid, 8), "stop_risk": stop_risk,
        "net_r": net_r, "is_win": is_win,
    })


def _price_lookup(inst):
    times, lows, highs, df = rs.load_instrument(inst)
    times_ns = times.astype("datetime64[ns]").astype("int64")
    close = df["close"].values

    def price_at(t_ns):
        idx = np.searchsorted(times_ns, t_ns, side="right") - 1
        idx = max(0, min(idx, len(close) - 1))
        return close[idx]
    return price_at


def resolve_closest_price(touches, age_h, price_at):
    """Event-driven, zero-lookahead resolver for one (instrument, side): at most one
    candidate ever rests at a time; when a new candidate's dispatch overlaps an already-
    resting one, keeps whichever is closer to price AT that dispatch moment (see module
    docstring). A dispatch that arrives while a confirmed fill is still open is blocked --
    matches _process_signals' MAX_CONCURRENT_PER_SIDE gate running after the closest-price
    resolution -- but isn't lost outright: it retries the instant the block clears (matches
    undo_dispatch's real retry-every-poll behavior), unless the block doesn't clear until
    after this touch's own price excursion would already be over (permanently moot then,
    same as a zone whose window has simply passed).

    Returns touches filtered to just the accepted rows (ready for g30.simulate_two_account,
    which still separately re-applies margin/equity availability on top of this)."""
    if touches.empty:
        return touches.assign(accepted=pd.Series(dtype=bool))
    t = touches.reset_index(drop=True)
    dispatch_time = pd.Series(pd.NaT, index=t.index, dtype="datetime64[ns]")

    # Group by (entry_price, created_at), not entry_price alone -- two genuinely different
    # zones (different created_at) can share the same price by coincidence (confirmed: not
    # rare with 100k+ zones over a bounded range -- see build_touches_with_metadata), and
    # created_at is now resolved correctly enough to tell them apart.
    for _, group in t.groupby(["entry_price", "created_at"]):
        g = group.sort_values("touch_num")
        idx = g.index.to_numpy()
        first = idx[0]
        dispatch_time.loc[first] = g.loc[first, "created_at"] + pd.Timedelta(hours=age_h)
        if len(idx) > 1:
            dispatch_time.loc[idx[1:]] = g["exit_time"].to_numpy()[:-1]
    t["dispatch_time"] = dispatch_time

    events = []
    for idx, row in t.iterrows():
        events.append((row.dispatch_time.value, 0, idx))
        events.append((row.entry_time.value, 1, idx))
    events.sort()

    resting = None
    confirmed_open_until = np.int64(-1)
    accepted = np.zeros(len(t), dtype=bool)
    entry_time_ns = t.entry_time.values.astype("int64")
    entry_price = t.entry_price.values
    exit_time_ns = t.exit_time.values.astype("int64")

    for t_ns, kind, idx in events:
        if kind == 0:  # dispatch
            if confirmed_open_until > entry_time_ns[idx]:
                continue  # block won't clear before this touch's own excursion is over
            effective_t = max(t_ns, confirmed_open_until)
            price = price_at(effective_t)
            if resting is None:
                resting = idx
            elif abs(entry_price[idx] - price) < abs(entry_price[resting] - price):
                resting = idx
        else:  # fill
            if resting == idx:
                accepted[idx] = True
                confirmed_open_until = exit_time_ns[idx]
                resting = None

    return t[accepted].drop(columns=["dispatch_time"])


def simulate_single_symbol(inst, cfg, risk_pct=0.06):
    """Builds both sides' touches for one instrument, resolves each side's closest-price
    concurrency independently (no cross-symbol interaction -- there's only one symbol),
    concatenates the survivors, and runs them through the unchanged margin/equity layer."""
    price_at = _price_lookup(inst)
    sides = []
    for side_is_buy in (True, False):
        touches = build_touches_with_metadata(
            inst, cfg["n"], cfg["rr"], cfg["age_h"], cfg["bounce"], cfg["touches"], side_is_buy)
        sides.append(resolve_closest_price(touches, cfg["age_h"], price_at))
    survivors = pd.concat(sides, ignore_index=True).sort_values("entry_time").reset_index(drop=True)
    if cfg.get("holdout"):
        survivors = survivors[survivors.entry_time >= g29.SPLIT]
    return g30.simulate_two_account(survivors, risk_pct)


def compare_symbols(cfg, risk_pct=0.06):
    """Quick per-symbol comparison under one strategy's parameters, to pick which single
    symbol the rebuilt single-symbol pipeline should evaluate going forward."""
    rows = []
    for inst in g29.TRADEABLE:
        rs.load_instrument(inst)
        rs._prep_zones(inst, pg.MIN_RETRACEMENT)
        t, eq_buy, eq_sell, max_dd, _ = simulate_single_symbol(inst, cfg, risk_pct)
        acc = t[t.accepted]
        wr = 100 * acc.is_win.mean() if len(acc) else float("nan")
        w = acc.loc[acc.net_r > 0, "net_r"].sum()
        l = -acc.loc[acc.net_r <= 0, "net_r"].sum()
        pf = w / l if l > 0 else float("inf")
        years = (t.entry_time.max() - t.entry_time.min()).days / 365.25 if len(t) else 0
        rows.append(dict(instrument=inst, trades=len(acc), win_rate=wr, profit_factor=pf,
                          years=years, trades_per_year=len(acc) / years if years else 0,
                          end_combined=eq_buy + eq_sell, max_drawdown_pct=max_dd))
    return pd.DataFrame(rows).sort_values("end_combined", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    cfg = g29.STRATEGIES["Velocity-King"] if "Velocity-King" in g29.STRATEGIES else dict(
        n=5.0, rr=0.20, age_h=0.20, bounce=False, touches=16, session=None, holdout=True)
    print("Per-symbol comparison under Velocity-King's exact deployed parameters "
          f"({cfg}), single-symbol (no portfolio), new closest-price concurrency:\n")
    result = compare_symbols(cfg)
    print(result.to_string(index=False))
    result.to_csv(os.path.join(DATA_DIR, "goal51_symbol_comparison.csv"), index=False)
