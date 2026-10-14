"""python -m analytics.prodlive <sync|summary|trades|by|equity|events|open|weekly|expect|sql|plot> [options]"""
import argparse, json, sys
import pandas as pd
from . import queries as Q, sync as S
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40)
def fmt(v): return f"{v:.4g}" if isinstance(v, float) else str(v)
def main(argv=None):
    ap = argparse.ArgumentParser(prog="analytics.prodlive"); ap.add_argument("cmd"); ap.add_argument("arg", nargs="?"); ap.add_argument("--db"); ap.add_argument("--src"); ap.add_argument("--ssh")
    ap.add_argument("--symbol"); ap.add_argument("--account"); ap.add_argument("--since"); ap.add_argument("-n", type=int, default=30); ap.add_argument("--json", action="store_true")
    ap.add_argument("--win-rate", type=float); ap.add_argument("--avg-r", type=float); ap.add_argument("--freq", default="W"); ap.add_argument("--out", default="prodlive_equity.png")
    a = ap.parse_args(argv); flt = dict(account=a.account, symbol=a.symbol, since=a.since)
    if a.cmd == "sync": print("snapshot ->", S.sync(a.src, a.ssh)); return
    if a.cmd == "summary":
        s = Q.summary(a.db, **flt); print(json.dumps(s, default=str, indent=1) if a.json else "\n".join(f"{k:>18}: {fmt(v)}" for k, v in s.items()))
    elif a.cmd == "trades": print(Q.trades(a.db, **flt).tail(a.n)[["symbol", "side", "entry_time", "exit_time", "entry_price", "exit_price", "net_pnl", "r_multiple", "outcome"]].to_string())
    elif a.cmd == "by": print(Q.breakdown(a.arg or "symbol", a.db, **flt).round(3).to_string())
    elif a.cmd == "weekly": print(Q.period_returns(a.freq, a.db, **flt).round(2).to_string())
    elif a.cmd == "equity": print(Q.equity_curve(a.db).tail(a.n)[["timestamp", "combined_balance", "equity", "drawdown", "open_positions_count"]].round(4).to_string())
    elif a.cmd == "events": print(Q.events(a.db, a.n).to_string())
    elif a.cmd == "open": print(Q.open_positions(a.db).to_string()); print(Q.pending_orders(a.db).to_string())
    elif a.cmd == "expect": print(Q.vs_expectation(a.db, a.win_rate, a.avg_r, **flt))
    elif a.cmd == "sql": print(Q.sql(a.arg, a.db).to_string())
    elif a.cmd == "plot":
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        e = Q.equity_curve(a.db); f, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw=dict(height_ratios=[3, 1])); ax[0].plot(e.timestamp, e.equity); ax[0].set_title("PRODLIVETRADING equity"); ax[1].fill_between(e.timestamp, e.drawdown, 0, color="r", alpha=.4); ax[1].set_title("drawdown"); f.tight_layout(); f.savefig(a.out, dpi=120); print("wrote", a.out)
    else: sys.exit(f"unknown command {a.cmd}")
if __name__ == "__main__": main()
