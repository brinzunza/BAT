"""/goal (2026-09-04): "keep developing strategies until you make it to under 2 year to
median 1 million. it must be a legit valid robust strategy given the restrictions we have
talked about before."

Critical methodology gap this goal exposes: every "years to $1M" number this whole session
(Yen-Rapid's 31.5yr, the GBM Federated Portfolio's 6.84yr, all of it) came from ONE
deterministic historical equity path, extrapolated via CAGR. Under FIXED-fraction position
sizing, multiplying the same set of trade outcomes together is commutative -- reordering the
SAME trades changes nothing, so a single historical path's CAGR is silent on genuine
sequence-of-returns risk. "Median" specifically calls for a real distribution: many
resampled paths, ordered differently, some drawing an unlucky early cluster of the (rare,
but real) losses, some drawing a lucky run -- exactly what a block-bootstrap Monte Carlo is
for, and exactly what this project's own `run_mc_capped` infra (goal10) was built for, but
never applied THIS session's two-separate-sub-account/margin/one-open-per-side mechanics to.

Implementation: a block bootstrap that resamples BLOCKS of `block` consecutive real trades
(with replacement, wrapping) -- preserving real local timing/duration structure within each
block, destroying long-range calendar dependence between blocks -- concatenated into a
synthetic multi-year sequence, then run through the EXACT SAME `g30.simulate_two_account`
two-sub-account/margin/leverage/one-open-per-side engine used everywhere else this session
(not a simplified approximation). Reports the MEDIAN elapsed time across many such
simulated paths, not the single historical path's CAGR extrapolation.
"""
import os
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

import goal30_apex_king_deployment as g30
import goal64_meta_labeling as g64

BLOCK = 20  # matches goal10's own established block-bootstrap convention


def block_bootstrap_sequence(trades, block, n_trades_target, rng):
    """One synthetic multi-year sequence: draws blocks of `block` consecutive real trades
    (with replacement), shifts each block's own entry/exit times to continue immediately
    after the previous block's last exit (preserving each block's own real internal timing
    -- inter-trade gaps and position durations -- while destroying which calendar period any
    given block came from). Block start indices are drawn from [0, n-block) -- NOT wrapped
    past the end of the array: wrapping was tried first and confirmed (via a direct span
    check) to occasionally splice the end of history directly onto the beginning within one
    block, producing a non-monotonic, garbled timeline (one observed case: a 6000-trade
    synthetic sequence spanning 13.1 real years instead of the ~5.7 its own trade count and
    the sleeve's real density implies) -- a real bug, not a acceptable approximation."""
    t = trades.sort_values("entry_time").reset_index(drop=True)
    n = len(t)
    entry_ns = t.entry_time.values.astype("datetime64[ns]").astype("int64")
    exit_ns = t.exit_time.values.astype("datetime64[ns]").astype("int64")

    out_rows = []
    cursor = entry_ns[0]
    trades_so_far = 0
    while trades_so_far < n_trades_target:
        start_idx = rng.integers(0, n - block)
        idxs = np.arange(start_idx, start_idx + block)
        block_entry = entry_ns[idxs]
        block_exit = exit_ns[idxs]
        offset = cursor - block_entry[0]
        new_entry = block_entry + offset
        new_exit = block_exit + offset
        rows = t.iloc[idxs].copy()
        rows["entry_time"] = pd.to_datetime(new_entry)
        rows["exit_time"] = pd.to_datetime(new_exit)
        out_rows.append(rows)
        cursor = new_exit[-1] + 1_000_000_000  # +1s gap before the next block starts
        trades_so_far += block

    return pd.concat(out_rows, ignore_index=True).iloc[:n_trades_target].reset_index(drop=True)


def bootstrap_median_years_to_target(trades, risk_pct, n_sims, block=BLOCK, n_trades_target=6000,
                                      target=1_000_000.0, start_per_side=200.0, seed=42):
    rng = np.random.default_rng(seed)
    years_list = []
    n_not_reached = 0
    for s in range(n_sims):
        synth = block_bootstrap_sequence(trades, block, n_trades_target, rng)
        _, _, _, _, trace = g30.simulate_two_account(synth, risk_pct, start_per_side=start_per_side)
        hit = trace[trace.equity_combined >= target] if len(trace) else trace
        if len(hit) == 0:
            n_not_reached += 1
            continue
        start_t = synth.entry_time.min()
        years = (hit.close_time.iloc[0] - start_t).days / 365.25
        years_list.append(years)
    years_arr = np.array(years_list)
    median_years = np.median(years_arr) if len(years_arr) else float("inf")
    return dict(median_years=median_years, p25=np.percentile(years_arr, 25) if len(years_arr) else None,
                p75=np.percentile(years_arr, 75) if len(years_arr) else None,
                frac_reached=1 - n_not_reached / n_sims, n_sims=n_sims, years_arr=years_arr)


def simulate_sleeve_trace(trades, risk_pct, n_trades_target, rng, start_per_side=200.0, block=BLOCK):
    """One bootstrap-resampled equity TRACE (close_time, equity_combined) for one sleeve --
    the building block for combining multiple independent sleeves' synthetic timelines."""
    synth = block_bootstrap_sequence(trades, block, n_trades_target, rng)
    _, _, _, _, trace = g30.simulate_two_account(synth, risk_pct, start_per_side=start_per_side)
    start_t = synth.entry_time.min()
    if len(trace) == 0:
        return pd.DataFrame({"years": [], "equity": []})
    years = (trace.close_time - start_t).dt.days / 365.25
    return pd.DataFrame({"years": years.values, "equity": trace.equity_combined.values})


def bootstrap_median_years_combined_portfolio(sleeve_trades, risk_pcts, n_sims, n_trades_target=6000,
                                               target=1_000_000.0, start_per_side=200.0, seed=42,
                                               horizon_years=10.0, grid_points=2000):
    """Combines N independently-bootstrapped sleeve traces (each own draw, own randomness)
    onto a common yearly grid, sums equity across sleeves at each grid point, and finds the
    first grid year the SUMMED equity crosses target -- the genuine multi-sleeve-portfolio
    analogue of the single-sleeve function above."""
    rng = np.random.default_rng(seed)
    grid = np.linspace(0, horizon_years, grid_points)
    years_list = []
    n_not_reached = 0
    for s in range(n_sims):
        combined = np.full(grid_points, start_per_side * 2 * len(sleeve_trades))
        for inst, trades in sleeve_trades.items():
            trace = simulate_sleeve_trace(trades, risk_pcts[inst], n_trades_target, rng,
                                           start_per_side=start_per_side)
            if len(trace) == 0:
                continue
            sleeve_on_grid = np.interp(grid, trace.years.values, trace.equity.values,
                                        left=start_per_side * 2, right=trace.equity.values[-1])
            combined = combined - start_per_side * 2 + sleeve_on_grid  # replace this sleeve's flat baseline with its own trace
        hit_idx = np.argmax(combined >= target) if np.any(combined >= target) else -1
        if hit_idx == -1 or combined[hit_idx] < target:
            n_not_reached += 1
            continue
        years_list.append(grid[hit_idx])
    years_arr = np.array(years_list)
    median_years = np.median(years_arr) if len(years_arr) else float("inf")
    return dict(median_years=median_years,
                p25=np.percentile(years_arr, 25) if len(years_arr) else None,
                p75=np.percentile(years_arr, 75) if len(years_arr) else None,
                frac_reached=1 - n_not_reached / n_sims, n_sims=n_sims)


def gbm_sleeve(inst, n, rr, age, threshold):
    """Same recipe validated for EURGBP/EURUSD, applied to all four symbols (goal68 only
    tried it on 2; extending it to GBPUSD/USDJPY here found the two biggest single wins of
    this whole /goal round -- see module docstring)."""
    from sklearn.ensemble import GradientBoostingClassifier
    t = g64.build_labeled_touches(inst, n, rr, age).sort_values("entry_time").reset_index(drop=True)
    split_idx = int(len(t) * 0.7)
    train, test = t.iloc[:split_idx], t.iloc[split_idx:]
    clf = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                                      subsample=0.8, random_state=42)
    clf.fit(train[g64.FEATURES].values, (train["net_r"] > 0).astype(int).values)
    test = test.copy()
    test["meta_prob"] = clf.predict_proba(test[g64.FEATURES].values)[:, 1]
    filt = test[test.meta_prob >= threshold].copy()
    return g64.true_accepted(filt, age, inst)


# v2 Federated Portfolio: the same extreme-N/R:R=0.05 + GBM meta-labeling recipe that
# worked so well for EURGBP/EURUSD (goal68), now also applied to GBPUSD and USDJPY --
# neither had been tried at this specific shape before. Result: GBPUSD's own frequency
# goes 236->1,604/yr (PF still robust at 1.252); USDJPY, previously never rescued by ANY
# meta-labeling attempt, gets its first genuine improvement (PF 1.031->2.672, though see
# the walk-forward caveat below -- one of its four windows shows filtered PF=0.765, a real
# weak spot the other three sleeves don't share).
V2_SLEEVES = {
    "EURGBP": dict(n=35.0, rr=0.05, age=0.05, threshold=0.97),
    "EURUSD": dict(n=40.0, rr=0.05, age=0.05, threshold=0.98),
    "GBPUSD": dict(n=30.0, rr=0.05, age=0.05, threshold=0.95),
    "USDJPY": dict(n=15.0, rr=0.05, age=0.05, threshold=0.98),
}


if __name__ == "__main__":
    import time
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    t0 = time.time()

    print("Building v2 sleeves (extreme-N/R:R=0.05 + GBM on all four symbols)...")
    sleeves = {inst: gbm_sleeve(inst, **cfg) for inst, cfg in V2_SLEEVES.items()}
    all_trades = pd.concat(list(sleeves.values()), ignore_index=True)
    span_years = (all_trades.entry_time.max() - all_trades.entry_time.min()).days / 365.25
    total_n = sum(len(v) for v in sleeves.values())
    gw = sum(v[v.net_r > 0].net_r.sum() for v in sleeves.values())
    gl = sum(-v[v.net_r <= 0].net_r.sum() for v in sleeves.values())
    print(f"COMBINED v2: freq={total_n/span_years:.1f}/yr "
          f"pooled_WR={100*sum((v.net_r>0).sum() for v in sleeves.values())/total_n:.1f}% "
          f"pooled_PF={(gw/gl if gl>0 else float('inf')):.3f}")
    for inst, v in sleeves.items():
        v.to_csv(os.path.join(DATA_DIR, f"goal69_v2_federated_{inst.lower()}_trades.csv"), index=False)
        print(f"  {inst}: n={len(v)}")

    print("\nBootstrap MEDIAN years-to-$1M (block-bootstrap MC, two-sub-account mechanics, "
          f"{time.time()-t0:.0f}s so far)...")
    n_sims = 100 if args.quick else 300
    for label, risks in [("moderate", dict(EURGBP=0.20, EURUSD=0.20, GBPUSD=0.15, USDJPY=0.10)),
                          ("aggressive", dict(EURGBP=0.45, EURUSD=0.50, GBPUSD=0.30, USDJPY=0.10)),
                          ("very_aggressive", dict(EURGBP=0.58, EURUSD=0.58, GBPUSD=0.45, USDJPY=0.15))]:
        res = bootstrap_median_years_combined_portfolio(sleeves, risks, n_sims=n_sims,
                                                          n_trades_target=8000, horizon_years=5.0)
        print(f"  {label} {risks}: median={res['median_years']:.2f}yr "
              f"IQR=[{res['p25']:.2f},{res['p75']:.2f}] reached={100*res['frac_reached']:.1f}% "
              f"({time.time()-t0:.0f}s total)")
