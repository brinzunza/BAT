"""Block-bootstrap Monte Carlo race to $1M/$10M for all goal-2 contenders, same
methodology as the goal70 $200@5% report: 4,000 sims, 20-trade blocks resampled with
replacement from each strategy's own real trade history, $200 start, 5% fixed-
fractional risk per trade, ruin floor $1.
"""
import numpy as np
import pandas as pd

DATA = "/Users/brunoinzunza/Documents/GitHub/tradingResearch/dtfxStrategy/data"
START = 200.0
RISK_FRAC = 0.05
T1, T10 = 1_000_000.0, 10_000_000.0
BLOCK = 20
N_SIMS = 4000
MAX_TRADES = 30000
RUIN_FLOOR = 1.0

CONFIGS = [
    ("goal70 (original)", f"{DATA}/goal70_final_winner_age25.csv"),
    ("goal2 Candidate A (balanced)", f"{DATA}/goal2_candidateA_k45.csv"),
    ("goal2 Candidate B (max-speed)", f"{DATA}/goal2_hour_opt_trades.csv"),
    ("SOMEGA (live)", f"{DATA}/somega_faithful_trades.csv"),
    ("SOMEGA + opt-hour", f"{DATA}/goal2_somega_opthour_trades.csv"),
    ("CANA (faithful)", f"{DATA}/cana_faithful_trades.csv"),
]


def run_mc(net_r, trades_per_year, seed):
    rng = np.random.default_rng(seed)
    n = len(net_r)
    t1_list, t10_list = [], []
    ruin_before_1m = 0
    for sim in range(N_SIMS):
        eq = START
        trades = 0
        t1 = None
        t10 = None
        ruined = False
        while trades < MAX_TRADES:
            start_idx = rng.integers(0, n)
            idxs = (start_idx + np.arange(BLOCK)) % n
            block = net_r[idxs]
            done = False
            for r in block:
                risk = eq * RISK_FRAC
                eq += risk * r
                trades += 1
                if eq < RUIN_FLOOR:
                    ruined = True
                    done = True
                    break
                if t1 is None and eq >= T1:
                    t1 = trades
                if t1 is not None and eq >= T10:
                    t10 = trades
                    done = True
                    break
            if done:
                break
        t1_list.append(t1)
        t10_list.append(t10)
        if ruined and t1 is None:
            ruin_before_1m += 1
    t1_arr = np.array([x for x in t1_list if x is not None])
    t10_arr = np.array([x for x in t10_list if x is not None])
    return t1_arr, t10_arr, ruin_before_1m


def main():
    results = []
    for i, (label, path) in enumerate(CONFIGS):
        t = pd.read_csv(path, parse_dates=["entry_time"]).sort_values("entry_time")
        years = (t.entry_time.max() - t.entry_time.min()).days / 365.25
        tpy = len(t) / years
        t1_arr, t10_arr, ruined = run_mc(t.net_r.values, tpy, seed=1000 + i)
        row = dict(
            label=label, trades=len(t), tpy=tpy,
            hit_1m_pct=100 * len(t1_arr) / N_SIMS,
            ruin_pct=100 * ruined / N_SIMS,
            p10_1m=np.percentile(t1_arr, 10) / tpy if len(t1_arr) else None,
            p50_1m=np.percentile(t1_arr, 50) / tpy if len(t1_arr) else None,
            p90_1m=np.percentile(t1_arr, 90) / tpy if len(t1_arr) else None,
            mean_1m=t1_arr.mean() / tpy if len(t1_arr) else None,
            p10_10m=np.percentile(t10_arr, 10) / tpy if len(t10_arr) else None,
            p50_10m=np.percentile(t10_arr, 50) / tpy if len(t10_arr) else None,
            p90_10m=np.percentile(t10_arr, 90) / tpy if len(t10_arr) else None,
            mean_10m=t10_arr.mean() / tpy if len(t10_arr) else None,
        )
        results.append(row)
        print(f"{label:32s} trades/yr={tpy:6.0f}  hit1M={row['hit_1m_pct']:5.1f}%  ruin={row['ruin_pct']:4.1f}%  "
              f"yrs->1M[P10/P50/P90]={row['p10_1m']:.2f}/{row['p50_1m']:.2f}/{row['p90_1m']:.2f}  "
              f"yrs->10M[P50]={row['p50_10m']:.2f}" if row['p50_10m'] else "N/A")

    pd.DataFrame(results).to_csv(f"{DATA}/goal2_mc_race_results.csv", index=False)
    print(f"\nSaved: {DATA}/goal2_mc_race_results.csv")


if __name__ == "__main__":
    main()
