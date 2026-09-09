import os
import glob
import pandas as pd
from backtest_beid_strategy import parse_candle_data, run_beid_state_machine_and_backtest

def run_comprehensive_suite():
    data_dirs = [
        "/Users/brunoinzunza/.gemini/antigravity/scratch/smc_strategy_backtest/data",
        "/Users/brunoinzunza/.gemini/antigravity/scratch/smc_strategy_backtest/data_ltf_deep"
    ]
    
    all_results = []
    
    for d in data_dirs:
        for f in sorted(glob.glob(f"{d}/*.csv")):
            fname = os.path.basename(f)
            candles = parse_candle_data(f)
            if len(candles) < 50:
                continue
            zones, trades = run_beid_state_machine_and_backtest(candles)
            if len(trades) == 0:
                continue
                
            wins = sum(1 for t in trades if t["result"] == "WIN")
            losses = sum(1 for t in trades if t["result"] == "LOSS")
            win_rate = (wins / len(trades)) * 100.0
            net_r = sum(t["r_mult"] for t in trades)
            pf = (wins * 1.0) / max(1, losses * 1.0)
            
            all_results.append({
                "file": fname,
                "path": f,
                "candles": len(candles),
                "zones": len(zones),
                "trades": len(trades),
                "wins": wins,
                "losses": losses,
                "win_rate": win_rate,
                "net_r": net_r,
                "profit_factor": pf
            })
            
    res_df = pd.DataFrame(all_results)
    print(res_df.to_string(index=False))
    
    total_trades = res_df["trades"].sum()
    total_wins = res_df["wins"].sum()
    total_losses = res_df["losses"].sum()
    overall_wr = (total_wins / total_trades) * 100.0 if total_trades > 0 else 0
    total_net_r = res_df["net_r"].sum()
    overall_pf = (total_wins * 1.0) / max(1, total_losses * 1.0)
    
    print("\n" + "=" * 60)
    print(f"OVERALL BEID PORTFOLIO PERFORMANCE ({len(res_df)} Datasets):")
    print(f"Total Trades: {total_trades} | Total Wins: {total_wins} | Total Losses: {total_losses}")
    print(f"Overall Win Rate: {overall_wr:.1f}%")
    print(f"Overall Profit Factor: {overall_pf:.2f}")
    print(f"Total Net Return: {total_net_r:+.1f} R")
    print("=" * 60)

if __name__ == "__main__":
    run_comprehensive_suite()
