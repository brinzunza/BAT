"""/goal (2026-09-04): "find a profitable strategy that beats all in the composite score
with an emphasis on trade frequency." Tests directly whether chasing raw trade count
actually wins once frequency is weighted heavily, against every candidate goal51-57
already validated (Bedrock-King, Yen-Punch, Yen-Steady, Dollar-Wide, Cable-Wide).

Same corrected fill model (goal51: full 1.6-pip fill depth, universal spread cost) and
account rules (goal30: two $100 sub-accounts, 50x leverage, max 1 open position/side)
throughout.
"""
import os
import sys
import itertools

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

import goal30_apex_king_deployment as g30
import goal31_apex_stats as g31
import goal51_single_symbol_closest_price_backtest as g51

AGE_H, TOUCHES = 0.08, 20
SPLIT = pd.Timestamp("2018-01-01")


def build(inst, n, rr, mrp):
    parts = [g51.build_touches_with_metadata(inst, n, rr, AGE_H, False, TOUCHES, side,
                                              min_risk_pct=mrp)
             for side in (True, False)]
    return pd.concat(parts, ignore_index=True)


def high_frequency_screen():
    """Wider grid than goal54/57, explicitly reaching for higher N and tighter R:R to
    find the highest-frequency profitable configs across all 4 symbols."""
    n_grid = [15, 18, 20, 22, 25, 28, 30, 35]
    rr_grid = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    mrp_grid = [0.008, 0.010, 0.013, 0.016, 0.020, 0.025]
    rows = []
    for inst in ("EURUSD", "GBPUSD", "EURGBP", "USDJPY"):
        for n, rr, mrp in itertools.product(n_grid, rr_grid, mrp_grid):
            t = build(inst, n, rr, mrp)
            if len(t) < 200:
                continue
            gw, gl = t[t.net_r > 0].net_r.sum(), -t[t.net_r <= 0].net_r.sum()
            pf = gw / gl if gl > 0 else float("inf")
            if pf <= 1.03:
                continue
            years = (t.entry_time.max() - t.entry_time.min()).days / 365.25
            rows.append(dict(inst=inst, n=n, rr=rr, mrp=mrp, n_trades=len(t), pf=pf,
                              wr=100 * (t.net_r > 0).mean(), freq_tpy=len(t) / years))
    return pd.DataFrame(rows).sort_values("freq_tpy", ascending=False)


def full_stats(label, inst, n, rr, mrp):
    t = build(inst, n, rr, mrp).sort_values("entry_time").reset_index(drop=True)
    nn = len(t)
    years = (t.entry_time.max() - t.entry_time.min()).days / 365.25
    wins, losses = t[t.net_r > 0], t[t.net_r <= 0]
    wr = 100 * len(wins) / nn
    gw, gl = wins.net_r.sum(), -losses.net_r.sum()
    pf = gw / gl
    ann_sharpe, exk, calmar, maxdd, _ = g31.sharpe_and_kurtosis(t)
    _, eq_buy, eq_sell, dd6, _ = g30.simulate_two_account(t, 0.06)
    return dict(label=label, inst=inst, n=n, rr=rr, mrp=mrp, n_trades=nn, freq_tpy=nn / years,
                pf=pf, wr=wr, sharpe=ann_sharpe, exk=exk, calmar=calmar,
                maxdd_1pct=-100 * maxdd, maxdd_6pct=dd6)


def composite(df, freq_weight=0.40, other_weight=0.20):
    def norm(s):
        return 100 * (s - s.min()) / (s.max() - s.min())
    df = df.copy()
    df["freq_score"] = norm(df.freq_tpy)
    df["pf_score"] = norm(df.pf)
    df["wr_score"] = norm(df.wr)
    df["dd_score"] = norm(-df.maxdd_6pct.abs())
    df["composite_freq_emphasis"] = (freq_weight * df.freq_score + other_weight * df.pf_score
                                      + other_weight * df.wr_score + other_weight * df.dd_score)
    return df.sort_values("composite_freq_emphasis", ascending=False)


if __name__ == "__main__":
    screen = high_frequency_screen()
    out = os.path.join(DATA_DIR, "goal58_high_frequency_screen.csv")
    screen.to_csv(out, index=False)
    print(f"{len(screen)} profitable high-frequency configs, saved {out}")
    print(screen.head(15).to_string(index=False))

    candidates = [
        ("Frequent-Cable", "GBPUSD", 25.0, 0.30, 0.013),
        ("Bedrock-King", "EURGBP", 22.0, 0.75, 0.020),
        ("Yen-Punch", "USDJPY", 4.0, 0.75, 0.005),
        ("Yen-Steady", "USDJPY", 4.0, 0.15, 0.005),
        ("Dollar-Wide", "EURUSD", 15.0, 1.00, 0.020),
        ("Cable-Wide", "GBPUSD", 15.0, 1.50, 0.020),
        ("Balanced-Dollar", "EURUSD", 30.0, 0.50, 0.020),
    ]
    stats = pd.DataFrame([full_stats(*c) for c in candidates])
    ranked = composite(stats)
    out = os.path.join(DATA_DIR, "goal58_frequency_emphasis_composite.csv")
    ranked.to_csv(out, index=False)
    print(f"\nFrequency-emphasis composite (40% freq / 20% each PF, WR, DD), saved {out}:")
    print(ranked[["label", "inst", "freq_tpy", "pf", "wr", "maxdd_6pct",
                  "composite_freq_emphasis"]].to_string(index=False))
