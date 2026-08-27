"""Walk-forward, purged, multi-instrument, robustness-aware parameter+rule search for DTFX.

Directly implements the methodology agreed on for this phase of research:

1. WALK-FORWARD, not a single train/test split. History is split into 8 sequential blocks;
   for each of the 7 transitions, Optuna searches using an EXPANDING train window (all data
   up to the boundary) and is then scored OUT-OF-SAMPLE on the next, never-before-seen block.
   No test window is ever reused across folds.

2. PURGED + EMBARGOED boundaries. Trades entered within EMBARGO_DAYS of a train/test boundary
   are dropped from whichever side they're closest to, so a trade whose holding period could
   straddle the boundary never leaks information across it. Standard practice from purged CV
   (Lopez de Prado); this is the same category of fix as the elapsed-time leak found earlier
   in the RL lifecycle work, applied here to boundary leakage instead of feature leakage.

3. ROBUSTNESS-AWARE OBJECTIVE. Never raw Sharpe or raw return. Each instrument's score is
   Calmar ratio penalized by excess kurtosis (fat tails = reliance on a few big days). Under-
   trading a window (< MIN_TRADES_FOR_SCORE) is scored as a hard failure, not "no data."

4. JOINT MULTI-INSTRUMENT SCORING. The objective a trial optimizes is the MINIMUM of the
   three instruments' scores (EUR/USD, GBP/USD, EUR/GBP -- all from the same vendor, London
   Strategic Edge, for apples-to-apples comparability). This bakes cross-instrument fragility
   into the search itself instead of discovering it after the fact, which is exactly what
   went wrong with the earlier RL overlay (trained on EUR/USD alone, found fragile elsewhere).

5. An optional, previously-validated RSI/ATR filter (see ../CHECKPOINT_NOTES.md section 4) is
   exposed as a searchable boolean rule, not hardcoded on or off -- this is the "can be
   modified, add other indicators/rules" dimension of the base DTFX strategy.

The per-zone fill/resolution search is numba-jitted (see _simulate_zones_numba) -- a plain
Python loop over ~50k-150k zones per instrument took ~12s per call, which would have made a
multi-fold Optuna search intractable (thousands of calls needed); JIT compiling the same loop
cuts this to a small fraction of a second, the same category of fix as last session's bounded-
window optimization, just one level deeper.

Outputs (all under ../data/, gitignored but persistent):
  - dtfx_robust_search_folds.csv       per-fold chosen params + in-sample and OOS stats
  - dtfx_robust_search_pareto.csv      every trial's params + score, for the Pareto frontier
"""
import os
import numpy as np
import pandas as pd
import optuna
from numba import njit
from scipy.stats import kurtosis as scipy_kurtosis, skew as scipy_skew

from dtfx_eurusd_backtest_strict import find_pivots_strict
from dtfx_eurusd_backtest import run_structure_fsm

optuna.logging.set_verbosity(optuna.logging.WARNING)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
INSTRUMENTS = {
    "EURUSD": "lse_eurusd_1min_full.csv",
    "GBPUSD": "lse_gbpusd_1min_full.csv",
    "EURGBP": "lse_eurgbp_1min_full.csv",
}
ANALYSIS_START = pd.Timestamp("2009-09-25")
MIN_RETRACEMENT_VALUES = [2, 3, 4, 5]
EMBARGO_DAYS = 5
N_WINDOWS = 8            # -> 7 walk-forward train/test transitions
N_TRIALS_PER_FOLD = 80
MIN_TRADES_FOR_SCORE = 30
RISK_PCT_FOR_SHARPE = 0.01
START_EQUITY = 100.0
FILL_SEARCH_FAST_BARS = 60 * 24 * 5
MAX_BARS_TO_RESOLVE = 60 * 24 * 30
IBKR_TYPICAL_PCT = 0.0054

_price_cache = {}      # instrument -> (times, lows, highs, df)
_indicator_cache = {}  # instrument -> (rsi14 array, atr_ratio array)
_zone_cache = {}       # (instrument, min_retracement) -> zone numpy arrays (see _prep_zones)


@njit(cache=True)
def _simulate_zones_numba(zone_top, zone_bottom, zone_mid, zone_is_buy, zone_start_idx,
                           lows, highs, n_multiplier, fast_bars, max_bars):
    n = len(zone_top)
    n_times = len(lows)
    entry_idx_out = np.full(n, -1, dtype=np.int64)
    status_out = np.zeros(n, dtype=np.int8)  # 0=unresolved, 1=stopped_out, 2=take_profit

    for i in range(n):
        is_buy = zone_is_buy[i]
        top, bottom, mid = zone_top[i], zone_bottom[i], zone_mid[i]
        risk = n_multiplier * (top - bottom) / 2.0
        invalidation = bottom if is_buy else top
        start = zone_start_idx[i]
        if start >= n_times:
            continue

        fast_end = start + fast_bars
        if fast_end > n_times:
            fast_end = n_times
        breach_idx, mid_idx = -1, -1
        for j in range(start, fast_end):
            if is_buy:
                if breach_idx == -1 and lows[j] <= invalidation:
                    breach_idx = j
                if mid_idx == -1 and lows[j] <= mid:
                    mid_idx = j
            else:
                if breach_idx == -1 and highs[j] >= invalidation:
                    breach_idx = j
                if mid_idx == -1 and highs[j] >= mid:
                    mid_idx = j
            if breach_idx != -1 and mid_idx != -1:
                break

        if mid_idx == -1 and breach_idx == -1 and fast_end < n_times:
            for j in range(fast_end, n_times):
                if is_buy:
                    if breach_idx == -1 and lows[j] <= invalidation:
                        breach_idx = j
                    if mid_idx == -1 and lows[j] <= mid:
                        mid_idx = j
                else:
                    if breach_idx == -1 and highs[j] >= invalidation:
                        breach_idx = j
                    if mid_idx == -1 and highs[j] >= mid:
                        mid_idx = j
                if breach_idx != -1 and mid_idx != -1:
                    break

        if mid_idx == -1:
            continue
        if breach_idx != -1 and breach_idx <= mid_idx:
            continue

        entry_idx = mid_idx
        after_start = entry_idx + 1
        after_end = after_start + max_bars
        if after_end > n_times:
            after_end = n_times
        if after_start >= n_times:
            continue

        stop_level = mid - risk if is_buy else mid + risk
        target_level = mid + risk if is_buy else mid - risk
        sl_idx, tp_idx = -1, -1
        for j in range(after_start, after_end):
            if is_buy:
                if sl_idx == -1 and lows[j] <= stop_level:
                    sl_idx = j
                if tp_idx == -1 and highs[j] >= target_level:
                    tp_idx = j
            else:
                if sl_idx == -1 and highs[j] >= stop_level:
                    sl_idx = j
                if tp_idx == -1 and lows[j] <= target_level:
                    tp_idx = j
            if sl_idx != -1 and tp_idx != -1:
                break

        if sl_idx == -1 and tp_idx == -1:
            continue
        entry_idx_out[i] = entry_idx
        if sl_idx != -1 and (tp_idx == -1 or sl_idx <= tp_idx):
            status_out[i] = 1  # stopped_out
        else:
            status_out[i] = 2  # take_profit

    return entry_idx_out, status_out


def _rsi(close, n=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _atr_ratio(df, n=14):
    prev_close = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - prev_close).abs(),
                     (df["low"] - prev_close).abs()], axis=1).max(axis=1)
    atr14 = tr.ewm(alpha=1 / n, adjust=False).mean()
    return atr14 / atr14.rolling(2000).mean()


def load_instrument(instrument):
    if instrument in _price_cache:
        return _price_cache[instrument]
    path = os.path.join(DATA_DIR, INSTRUMENTS[instrument])
    print(f"Loading {instrument} from {path} ...", flush=True)
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    df.index = df.index.tz_localize(None)
    df = df[~df.index.duplicated(keep="first")]
    df = df[df.index >= ANALYSIS_START]
    df = df[["open", "high", "low", "close"]]

    _indicator_cache[instrument] = (_rsi(df["close"]).values, _atr_ratio(df).values)
    times = df.index.values
    lows = np.ascontiguousarray(df["low"].values, dtype=np.float64)
    highs = np.ascontiguousarray(df["high"].values, dtype=np.float64)
    _price_cache[instrument] = (times, lows, highs, df)
    print(f"  {len(df):,} bars, {df.index.min()} to {df.index.max()}", flush=True)
    return _price_cache[instrument]


def _prep_zones(instrument, min_retracement):
    key = (instrument, min_retracement)
    if key in _zone_cache:
        return _zone_cache[key]
    times, lows, highs, df = load_instrument(instrument)
    zigzag = find_pivots_strict(df, min_retracement)
    _, zones = run_structure_fsm(zigzag)

    n = len(zones)
    top = np.array([z["top"] for z in zones], dtype=np.float64)
    bottom = np.array([z["bottom"] for z in zones], dtype=np.float64)
    mid = np.array([z["mid"] for z in zones], dtype=np.float64)
    is_buy = np.array([z["side"] == "buy" for z in zones], dtype=np.bool_)
    created_at = np.array([np.datetime64(z["created_at"]) for z in zones])
    start_idx = np.searchsorted(times, created_at, side="right").astype(np.int64)

    packed = dict(top=top, bottom=bottom, mid=mid, is_buy=is_buy, start_idx=start_idx,
                  created_at=created_at)
    _zone_cache[key] = packed
    print(f"  {instrument} MIN_RETRACEMENT={min_retracement}: {n:,} zones", flush=True)
    return packed


def build_trades(instrument, min_retracement, n_multiplier, min_risk_pct, min_fill_age_hours,
                  use_rsi_atr_filter):
    z = _prep_zones(instrument, min_retracement)
    times, lows, highs, df = load_instrument(instrument)
    rsi14, atr_ratio = _indicator_cache[instrument]

    entry_idx, status = _simulate_zones_numba(
        z["top"], z["bottom"], z["mid"], z["is_buy"], z["start_idx"],
        lows, highs, float(n_multiplier), FILL_SEARCH_FAST_BARS, MAX_BARS_TO_RESOLVE)

    filled = entry_idx >= 0
    if not filled.any():
        return pd.DataFrame(columns=["entry_time", "net_r"])

    risk = n_multiplier * (z["top"] - z["bottom"]) / 2.0
    risk_pct_of_price = risk / z["mid"]
    entry_times_ns = times[np.where(filled, entry_idx, 0)]
    fill_hours = (entry_times_ns - z["created_at"]) / np.timedelta64(1, "h")

    keep = filled & (risk_pct_of_price >= min_risk_pct) & (fill_hours >= min_fill_age_hours)
    if use_rsi_atr_filter:
        r14 = rsi14[np.where(filled, entry_idx, 0)]
        ar = atr_ratio[np.where(filled, entry_idx, 0)]
        rule_ok = ~np.isnan(r14) & ~np.isnan(ar) & ~((r14 >= 45) & (r14 <= 55)) & ~(ar < 0.7)
        keep = keep & rule_ok

    if not keep.any():
        return pd.DataFrame(columns=["entry_time", "net_r"])

    idx = np.where(keep)[0]
    r_gross = np.where(status[idx] == 2, 1.0, -1.0)
    cost_in_r = IBKR_TYPICAL_PCT / (risk_pct_of_price[idx] * 100)
    out = pd.DataFrame({
        "entry_time": pd.to_datetime(entry_times_ns[idx]),
        "net_r": r_gross - cost_in_r,
    }).sort_values("entry_time")
    return out


def score_trades(trades):
    """geomean(Sharpe, Calmar, profit factor) / (1 + max(0, excess_kurtosis)). None if too
    few trades or degenerate.

    Four metrics, one number, deliberately multiplicative rather than a weighted sum: a
    weighted sum lets a config coast on one great metric while another is mediocre (e.g. a
    huge profit factor from a handful of outsized wins masking a poor Sharpe) -- exactly the
    kind of one-dimensional strength this whole search exists to catch. A geometric mean
    collapses toward zero the moment any one of the three "higher is better" metrics is bad,
    the same worst-case-oriented philosophy already used for the joint EUR/USD+GBP/USD+EUR/GBP
    scoring (worst instrument, not average instrument). Excess kurtosis divides in as before
    (fat tails = reliance on a few big days) since it's a "lower/near-zero is better" risk
    metric, not a return metric, and doesn't belong inside the same mean as the other three.
    """
    if len(trades) < MIN_TRADES_FOR_SCORE:
        return None, None
    equity = START_EQUITY
    curve_v, curve_t = [], []
    for _, row in trades.iterrows():
        equity = max(equity * (1 + RISK_PCT_FOR_SHARPE * row["net_r"]), 0.0)
        curve_t.append(row["entry_time"]); curve_v.append(equity)
    curve = pd.Series(curve_v, index=pd.to_datetime(curve_t))
    daily = curve.resample("D").last().ffill()
    ret = daily.pct_change().dropna()
    ret_nz = ret[ret != 0]
    if len(ret_nz) < 10:
        return None, None
    n_years = max((daily.index[-1] - daily.index[0]).days / 365.25, 0.1)
    total_return = curve.iloc[-1] / START_EQUITY - 1
    ann_return = (1 + total_return) ** (1 / n_years) - 1
    dd = (daily / daily.cummax() - 1.0)
    maxdd = abs(dd.min())
    if maxdd < 1e-6:
        return None, None
    calmar = ann_return / maxdd
    exk = scipy_kurtosis(ret_nz)
    ann_sharpe = ret_nz.mean() / ret_nz.std() * np.sqrt(252) if ret_nz.std() > 0 else 0.0

    gross_win = trades.loc[trades["net_r"] > 0, "net_r"].sum()
    gross_loss = -trades.loc[trades["net_r"] <= 0, "net_r"].sum()
    profit_factor = min(gross_win / gross_loss, 10.0) if gross_loss > 1e-9 else 10.0

    geomean_input = max(ann_sharpe, 0) * max(calmar, 0) * profit_factor
    score = geomean_input ** (1 / 3) / (1 + max(0, exk))
    stats = dict(trades=len(trades), ann_return=ann_return, maxdd=maxdd, calmar=calmar,
                 excess_kurtosis=exk, ann_sharpe=ann_sharpe, profit_factor=profit_factor,
                 skew=scipy_skew(ret_nz), n_daily_obs=len(ret_nz),
                 win_rate=float((trades["net_r"] > 0).mean()), avg_r=float(trades["net_r"].mean()))
    return score, stats


def purge(trades, lo=None, hi=None, embargo_before=None, embargo_after=None):
    t = trades
    if lo is not None:
        t = t[t["entry_time"] >= lo]
    if hi is not None:
        t = t[t["entry_time"] < hi]
    if embargo_before is not None:
        t = t[t["entry_time"] < embargo_before - pd.Timedelta(days=EMBARGO_DAYS)]
    if embargo_after is not None:
        t = t[t["entry_time"] >= embargo_after + pd.Timedelta(days=EMBARGO_DAYS)]
    return t


def make_train_objective(train_end):
    def objective(trial):
        n_multiplier = trial.suggest_float("n_multiplier", 1.0, 5.0)
        min_risk_pct = trial.suggest_float("min_risk_pct", 0.00005, 0.001, log=True)
        min_fill_age_hours = trial.suggest_float("min_fill_age_hours", 0.1, 4.0)
        min_retracement = trial.suggest_categorical("min_retracement", MIN_RETRACEMENT_VALUES)
        use_rsi_atr_filter = trial.suggest_categorical("use_rsi_atr_filter", [False, True])

        scores = []
        for instrument in INSTRUMENTS:
            trades = build_trades(instrument, min_retracement, n_multiplier, min_risk_pct,
                                   min_fill_age_hours, use_rsi_atr_filter)
            train_trades = purge(trades, hi=train_end, embargo_before=train_end)
            score, _ = score_trades(train_trades)
            scores.append(score if score is not None else -10.0)
        return min(scores)
    return objective


def evaluate_config(params, lo=None, hi=None, embargo_after=None):
    result = {}
    for instrument in INSTRUMENTS:
        trades = build_trades(instrument, params["min_retracement"], params["n_multiplier"],
                               params["min_risk_pct"], params["min_fill_age_hours"],
                               params["use_rsi_atr_filter"])
        window_trades = purge(trades, lo=lo, hi=hi, embargo_after=embargo_after)
        score, stats = score_trades(window_trades)
        result[instrument] = dict(score=score, **(stats or {}))
    return result


def run_walk_forward():
    ends = [pd.Timestamp(load_instrument(i)[0][-1]) for i in INSTRUMENTS]
    common_end = min(ends)
    edges = pd.date_range(ANALYSIS_START, common_end, periods=N_WINDOWS + 1)
    print(f"\nWalk-forward windows ({N_WINDOWS}): {[e.date().isoformat() for e in edges]}\n", flush=True)

    # warm the zone cache for every MIN_RETRACEMENT value up front (one-time cost)
    for instrument in INSTRUMENTS:
        for mr in MIN_RETRACEMENT_VALUES:
            _prep_zones(instrument, mr)

    fold_rows = []
    all_trial_records = []
    for k in range(1, N_WINDOWS):
        train_end = edges[k]
        test_start, test_end = edges[k], edges[k + 1] if k + 1 < len(edges) else common_end
        print(f"=== FOLD {k}/{N_WINDOWS - 1}: train < {train_end.date()}  |  test [{test_start.date()}, {test_end.date()}) ===", flush=True)

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=k))
        study.enqueue_trial(dict(n_multiplier=2.0, min_risk_pct=0.00025, min_fill_age_hours=1.0,
                                  min_retracement=3, use_rsi_atr_filter=False))

        def cb(study, trial):
            all_trial_records.append(dict(fold=k, **trial.params, score=trial.value))

        study.optimize(make_train_objective(train_end), n_trials=N_TRIALS_PER_FOLD, callbacks=[cb])
        best_params = study.best_params
        print(f"  best train score={study.best_value:.4f}  params={best_params}", flush=True)

        oos = evaluate_config(best_params, lo=test_start, hi=test_end, embargo_after=None)
        oos_scores = [v["score"] for v in oos.values() if v["score"] is not None]
        oos_min_score = min(oos_scores) if len(oos_scores) == 3 else -10.0
        print(f"  OOS min-score={oos_min_score:.4f}  " +
              "  ".join(f"{inst}: score={v['score']}, trades={v.get('trades')}, win_rate={v.get('win_rate')}"
                        for inst, v in oos.items()), flush=True)

        row = dict(fold=k, train_end=train_end, test_start=test_start, test_end=test_end,
                   oos_min_score=oos_min_score)
        for inst, v in oos.items():
            for kk, vv in v.items():
                row[f"{inst}__{kk}"] = vv
        for p, v in best_params.items():
            row[f"param__{p}"] = v
        fold_rows.append(row)

    folds_df = pd.DataFrame(fold_rows)
    folds_df.to_csv(os.path.join(DATA_DIR, "dtfx_robust_search_folds.csv"), index=False)
    trials_df = pd.DataFrame(all_trial_records)
    trials_df.to_csv(os.path.join(DATA_DIR, "dtfx_robust_search_pareto.csv"), index=False)
    print(f"\nSaved fold results -> dtfx_robust_search_folds.csv ({len(folds_df)} folds)", flush=True)
    print(f"Saved all trials -> dtfx_robust_search_pareto.csv ({len(trials_df)} trials)", flush=True)
    return folds_df, trials_df


if __name__ == "__main__":
    run_walk_forward()
