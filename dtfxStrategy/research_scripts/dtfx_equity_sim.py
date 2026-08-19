import sys
sys.path.insert(0, '/private/tmp/claude-501/-Users-brunoinzunza-Documents-GitHub/3b4d2171-ccf5-4c75-8a5d-8dac2a9ede36/scratchpad')
import pandas as pd, numpy as np
from dtfx_2025_backtest import find_pivots, build_zigzag, run_structure_fsm, simulate_zone_vectorized, r_multiple

CACHE_DIR = '/private/tmp/claude-501/-Users-brunoinzunza-Documents-GitHub/3b4d2171-ccf5-4c75-8a5d-8dac2a9ede36/scratchpad'
RISK_PER_TRADE_PCT = 0.01     # risk 1% of current equity per trade (fixed-fractional sizing)
ROUND_TRIP_FEE_PCT = 0.001    # 0.1% round-trip taker fee, same assumption as before
STARTING_EQUITY = 10000.0

def get_trades(ticker, reversed_signal):
    df = pd.read_csv(f'{CACHE_DIR}/polygon_{ticker.replace("-","_")}_2025.csv', parse_dates=['date']).set_index('date').sort_index()
    df = df[['open','high','low','close']]
    swing_highs, swing_lows = find_pivots(df['high'], df['low'], window=3)
    zigzag = build_zigzag(swing_highs, swing_lows)
    _, zones = run_structure_fsm(zigzag)
    times, lows, highs = df.index.values, df['low'].values, df['high'].values
    trades = [simulate_zone_vectorized(z, times, lows, highs, reversed_signal=reversed_signal) for z in zones]
    completed = [tr for tr in trades if tr['status'] in ('take_profit','stopped_out')]
    completed.sort(key=lambda tr: tr['entry_time'])
    for tr in completed:
        tr['r'] = r_multiple(tr, reversed_signal=reversed_signal)
        is_buy_zone = tr['side'] == 'buy'
        going_long = is_buy_zone if not reversed_signal else not is_buy_zone
        risk_price = (tr['mid']-tr['bottom']) if is_buy_zone else (tr['top']-tr['mid'])
        if reversed_signal:
            risk_price = (tr['tp']-tr['mid']) if is_buy_zone else (tr['mid']-tr['tp'])
        tr['risk_pct_of_price'] = abs(risk_price) / tr['mid']
    return [tr for tr in completed if tr['r'] is not None]


def simulate_equity(trades, with_fees):
    equity = STARTING_EQUITY
    curve = [equity]
    for tr in trades:
        risk_dollars = equity * RISK_PER_TRADE_PCT
        pnl = tr['r'] * risk_dollars
        if with_fees:
            # fee is charged on notional traded; notional = risk_dollars / risk_pct_of_price
            notional = risk_dollars / tr['risk_pct_of_price'] if tr['risk_pct_of_price'] > 0 else 0
            pnl -= notional * ROUND_TRIP_FEE_PCT
        equity += pnl
        equity = max(equity, 0.0)  # can't go below zero (would be a margin call / account wipeout)
        curve.append(equity)
        if equity <= 0:
            break
    return equity, curve


print(f"{'ticker':10s} {'variant':9s} {'n':>6s} {'win%':>7s} {'avgWinR':>8s} {'avgLossR':>9s} "
      f"{'trades/day':>10s} {'$10k->gross':>14s} {'$10k->net(0.1%fee)':>19s}")
for ticker in ['BTC-USDT', 'ETH-USDT', 'SOL-USDT', 'XAUT-USDT']:
    for variant, rev in [('original', False), ('reversed', True)]:
        trades = get_trades(ticker, rev)
        wins = [tr for tr in trades if tr['status'] == 'take_profit']
        losses = [tr for tr in trades if tr['status'] == 'stopped_out']
        win_rate = len(wins) / len(trades) if trades else float('nan')
        avg_win_r = np.mean([tr['r'] for tr in wins]) if wins else float('nan')
        avg_loss_r = np.mean([tr['r'] for tr in losses]) if losses else float('nan')
        trades_per_day = len(trades) / 365

        final_gross, _ = simulate_equity(trades, with_fees=False)
        final_net, _ = simulate_equity(trades, with_fees=True)

        print(f"{ticker:10s} {variant:9s} {len(trades):6d} {win_rate*100:6.1f}% {avg_win_r:8.3f} {avg_loss_r:9.3f} "
              f"{trades_per_day:10.2f} {final_gross:14,.2f} {final_net:19,.2f}")
