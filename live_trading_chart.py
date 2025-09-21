import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from typing import Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')

from data_providers.alpaca_provider import AlpacaDataProvider, AlpacaBroker
from strategies.base_strategy import BaseStrategy
from engines.live_trading_engine import LiveTradingEngine


class LiveTradingChart:
    """Live trading chart with strategy indicators and Alpaca integration"""

    def __init__(self,
                 strategy: BaseStrategy,
                 api_key: str = None,
                 secret_key: str = None,
                 symbol: str = "BTC/USD",
                 paper_trading: bool = True,
                 quantity: float = 0.01):

        self.strategy = strategy
        self.symbol = symbol
        self.quantity = quantity
        self.paper_trading = paper_trading

        # Initialize data provider and broker
        self.data_provider = AlpacaDataProvider(api_key, secret_key)
        self.broker = AlpacaBroker(api_key, secret_key, paper_trading) if api_key else None

        # Get initial balance from Alpaca account
        initial_balance = self.broker.get_buying_power() if self.broker else 10000
        print(f"💰 Account Balance: ${initial_balance:,.2f}")

        # Initialize trading engine
        self.trading_engine = LiveTradingEngine(
            data_provider=self.data_provider,
            broker_interface=self.broker,
            initial_balance=initial_balance
        )

        # Chart setup
        self.fig, (self.main_ax, self.indicator_ax) = plt.subplots(2, 1,
                                                                  figsize=(15, 10),
                                                                  gridspec_kw={'height_ratios': [3, 1]})

        # Main chart setup
        self.main_ax.set_title(f'{symbol} Live Trading Chart - {strategy.name}')
        self.main_ax.set_ylabel('Price (USD)')
        self.main_ax.grid(True, alpha=0.3)

        # Indicator chart setup
        self.indicator_ax.set_xlabel('Time')
        self.indicator_ax.set_ylabel('Indicator Values')
        self.indicator_ax.grid(True, alpha=0.3)

        # Data storage
        self.data_history = pd.DataFrame()
        self.max_candles = 100
        self.min_data_points = max(50, getattr(strategy, 'window', 20) + 10)  # Ensure enough data for strategy

        # Trading state
        self.last_trade_time = None
        self.trade_cooldown = timedelta(minutes=5)  # Prevent rapid trades
        self.data_ready = False

        # Colors
        self.bull_color = '#2E8B57'  # Sea Green
        self.bear_color = '#DC143C'  # Crimson
        self.bg_color = '#F5F5F5'    # White Smoke

        plt.style.use('default')
        self.fig.patch.set_facecolor(self.bg_color)

    def fetch_and_process_data(self):
        """Fetch data - initial bulk load then live updates"""
        try:
            # Initial data loading phase
            if self.data_history.empty:
                print(f"🚀 Starting fast data collection for {self.symbol}...")

                # Use the public endpoint to get recent bars immediately
                initial_df = self.data_provider.get_recent_bars_public(self.symbol, limit=self.min_data_points + 10)

                if not initial_df.empty:
                    # Use the fetched data, keep only what we need
                    self.data_history = initial_df.tail(self.min_data_points).copy().reset_index(drop=True)
                    print(f"✅ Fast initialization complete! Loaded {len(self.data_history)} bars")

                    # Mark as ready since we have enough data
                    if len(self.data_history) >= self.min_data_points:
                        self.data_ready = True
                        print(f"🎯 Ready for trading immediately!")
                        print(f"📊 Now switching to live data updates...")
                else:
                    print(f"❌ Failed to get initial data, will try live updates")
                    return None

            # Live data update phase (after initial load)
            else:
                # Get latest bar for live updates
                latest_bar = self.data_provider.get_latest_bar(self.symbol)

                if not latest_bar:
                    print("⚠️ No live data received, using existing data")
                    # Continue with existing data
                else:
                    # Convert to DataFrame row
                    new_row = pd.DataFrame([latest_bar])

                    # Check if this is a new timestamp (avoid duplicates)
                    # Ensure both timestamps are timezone-aware for comparison
                    latest_ts = pd.to_datetime(latest_bar['timestamp'])
                    if latest_ts.tz is None:
                        latest_ts = latest_ts.tz_localize('UTC')

                    last_historical_ts = self.data_history['timestamp'].iloc[-1]
                    if last_historical_ts.tz is None:
                        last_historical_ts = last_historical_ts.tz_localize('UTC')

                    if latest_ts > last_historical_ts:
                        print(f"🔄 Adding new live bar: ${latest_bar['Close']:.2f} at {latest_bar['timestamp'].strftime('%H:%M:%S')}")

                        # Add new bar
                        self.data_history = pd.concat([self.data_history, new_row], ignore_index=True)

                        # Keep only max_candles
                        if len(self.data_history) > self.max_candles:
                            self.data_history = self.data_history.tail(self.max_candles).reset_index(drop=True)

            # Process data if we have enough
            if len(self.data_history) >= self.min_data_points:
                if not self.data_ready:
                    print(f"✅ Data collection complete! Ready for trading with {len(self.data_history)} bars")
                    self.data_ready = True

                # Generate signals with strategy
                try:
                    df_with_signals = self.strategy.generate_signals(self.data_history.copy())
                    self.data_history = df_with_signals

                    # Process trading signals
                    self._process_trading_signals()
                except Exception as e:
                    print(f"⚠️ Error generating signals: {e}")
                    # Continue with existing data
            else:
                remaining = self.min_data_points - len(self.data_history)
                print(f"🔄 Need more data... {len(self.data_history)}/{self.min_data_points} bars ({remaining} more needed)")

            return self.data_history

        except Exception as e:
            print(f"❌ Error in fetch_and_process_data: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _process_trading_signals(self):
        """Process trading signals and execute trades"""
        try:
            # Only process signals if we have enough data
            if not self.data_ready or len(self.data_history) < self.min_data_points:
                return

            current_time = datetime.now()

            # Check cooldown period
            if (self.last_trade_time and
                current_time - self.last_trade_time < self.trade_cooldown):
                return

            # Get signal names from strategy
            signal_names = self.strategy.get_signal_names()
            latest_row = self.data_history.iloc[-1]

            buy_signal = latest_row.get(signal_names['buy'], False)
            sell_signal = latest_row.get(signal_names['sell'], False)
            current_price = latest_row['Close']
            timestamp = latest_row['timestamp']

            # Log current market data
            print(f"\n📊 [{timestamp.strftime('%Y-%m-%d %H:%M:%S')}] Market Update:")
            print(f"   💰 {self.symbol}: ${current_price:.2f}")
            print(f"   📈 Buy Signal: {'🟢 YES' if buy_signal else '⚫ NO'}")
            print(f"   📉 Sell Signal: {'🔴 YES' if sell_signal else '⚫ NO'}")

            # Execute trades through trading engine
            if buy_signal or sell_signal:
                signal_type = "BUY" if buy_signal else "SELL"
                print(f"\n🚨 [{timestamp.strftime('%H:%M:%S')}] {signal_type} SIGNAL TRIGGERED!")
                print(f"   💵 Price: ${current_price:.2f}")
                print(f"   📊 Strategy: {self.strategy.name}")
                print(f"   🔄 Processing trade...")

                # Store previous position for logging
                previous_position = self.trading_engine.position
                previous_balance = self.trading_engine.current_balance

                self.trading_engine.process_signals(
                    self.data_history,
                    self.strategy,
                    self.symbol,
                    self.quantity
                )

                # Log trade execution results
                new_position = self.trading_engine.position
                new_balance = self.trading_engine.current_balance

                if new_position != previous_position:
                    position_change = "OPENED" if previous_position == 0 else "CLOSED/CHANGED"
                    balance_change = new_balance - previous_balance

                    print(f"   ✅ Trade {position_change}!")
                    print(f"   📍 New Position: {self._format_position(new_position)}")
                    if balance_change != 0:
                        sign = "+" if balance_change > 0 else ""
                        print(f"   💰 Balance Change: {sign}${balance_change:.2f}")
                    print(f"   🏦 New Balance: ${new_balance:.2f}")
                else:
                    print(f"   ⚠️  No position change (cooldown or other condition)")

                self.last_trade_time = current_time

            else:
                # Log when no signals
                position_str = self._format_position(self.trading_engine.position)
                unrealized_pnl = self._calculate_unrealized_pnl(current_price)
                print(f"   📊 Position: {position_str}")
                if unrealized_pnl != 0:
                    sign = "+" if unrealized_pnl > 0 else ""
                    print(f"   💹 Unrealized P&L: {sign}${unrealized_pnl:.2f}")

        except Exception as e:
            print(f"❌ Error processing trading signals: {e}")

    def _format_position(self, position):
        """Format position for display"""
        if position == 0:
            return "FLAT (No position)"
        elif position == 1:
            entry_price = getattr(self.trading_engine, 'entry_price', 0)
            return f"LONG @ ${entry_price:.2f}"
        elif position == -1:
            entry_price = getattr(self.trading_engine, 'entry_price', 0)
            return f"SHORT @ ${entry_price:.2f}"
        return "UNKNOWN"

    def _calculate_unrealized_pnl(self, current_price):
        """Calculate unrealized P&L"""
        if (self.trading_engine.position == 0 or
            not hasattr(self.trading_engine, 'entry_price') or
            self.trading_engine.entry_price == 0):
            return 0

        if self.trading_engine.position == 1:  # Long
            return (current_price - self.trading_engine.entry_price) * self.quantity
        elif self.trading_engine.position == -1:  # Short
            return (self.trading_engine.entry_price - current_price) * self.quantity
        return 0

    def draw_candlesticks(self):
        """Draw candlestick chart with indicators"""
        self.main_ax.clear()
        self.indicator_ax.clear()

        if self.data_history is None or len(self.data_history) < 1:
            # Show data collection status
            self.main_ax.text(0.5, 0.5, 'Collecting Market Data...',
                            horizontalalignment='center', verticalalignment='center',
                            transform=self.main_ax.transAxes, fontsize=16)
            return

        # Show data collection progress if not ready
        if not self.data_ready:
            progress = len(self.data_history) / self.min_data_points * 100
            progress_text = f'Building Data Window: {len(self.data_history)}/{self.min_data_points} bars ({progress:.1f}%)'
            self.main_ax.text(0.5, 0.9, progress_text,
                            horizontalalignment='center', verticalalignment='center',
                            transform=self.main_ax.transAxes, fontsize=12,
                            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

        df = self.data_history.copy()

        # Setup main chart
        self.main_ax.set_title(f'{self.symbol} Live Trading - {self.strategy.name}')
        self.main_ax.set_ylabel('Price (USD)')
        self.main_ax.grid(True, alpha=0.3)

        # Draw candlesticks
        for i, (idx, row) in enumerate(df.iterrows()):
            open_price = row['Open']
            high_price = row['High']
            low_price = row['Low']
            close_price = row['Close']

            # Determine candle color
            color = self.bull_color if close_price >= open_price else self.bear_color

            # Draw wick
            self.main_ax.plot([i, i], [low_price, high_price], color='black', linewidth=1)

            # Draw body
            body_height = abs(close_price - open_price)
            body_bottom = min(open_price, close_price)

            rect = Rectangle((i - 0.3, body_bottom), 0.6, body_height,
                           facecolor=color, edgecolor='black', alpha=0.8)
            self.main_ax.add_patch(rect)

        # Draw strategy-specific indicators
        self._draw_strategy_indicators(df)

        # Draw trading signals
        self._draw_trading_signals(df)

        # Setup x-axis
        self._setup_time_axis(df)

        # Auto-scale
        self.main_ax.relim()
        self.main_ax.autoscale_view()
        self.indicator_ax.relim()
        self.indicator_ax.autoscale_view()

        # Add performance summary
        self._add_performance_text()

    def _draw_strategy_indicators(self, df: pd.DataFrame):
        """Draw strategy-specific indicators"""
        try:
            x_axis = range(len(df))

            # Common indicators that most strategies use
            if 'SMA_20' in df.columns:
                self.main_ax.plot(x_axis, df['SMA_20'], label='SMA 20', color='blue', alpha=0.7)

            if 'SMA_50' in df.columns:
                self.main_ax.plot(x_axis, df['SMA_50'], label='SMA 50', color='orange', alpha=0.7)

            if 'EMA_12' in df.columns:
                self.main_ax.plot(x_axis, df['EMA_12'], label='EMA 12', color='purple', alpha=0.7)

            if 'EMA_26' in df.columns:
                self.main_ax.plot(x_axis, df['EMA_26'], label='EMA 26', color='brown', alpha=0.7)

            # Bollinger Bands
            if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
                self.main_ax.plot(x_axis, df['bb_upper'], label='BB Upper', color='gray', alpha=0.5, linestyle='--')
                self.main_ax.plot(x_axis, df['bb_lower'], label='BB Lower', color='gray', alpha=0.5, linestyle='--')
                self.main_ax.fill_between(x_axis, df['bb_upper'], df['bb_lower'], alpha=0.1, color='gray')

            if 'bb_middle' in df.columns:
                self.main_ax.plot(x_axis, df['bb_middle'], label='BB Middle', color='blue', alpha=0.7)

            # RSI in indicator subplot
            if 'RSI' in df.columns:
                self.indicator_ax.plot(x_axis, df['RSI'], label='RSI', color='purple')
                self.indicator_ax.axhline(y=70, color='red', linestyle='--', alpha=0.5)
                self.indicator_ax.axhline(y=30, color='green', linestyle='--', alpha=0.5)
                self.indicator_ax.set_ylim(0, 100)

            # MACD in indicator subplot
            elif 'MACD' in df.columns and 'MACD_Signal' in df.columns:
                self.indicator_ax.plot(x_axis, df['MACD'], label='MACD', color='blue')
                self.indicator_ax.plot(x_axis, df['MACD_Signal'], label='Signal', color='red')
                if 'MACD_Hist' in df.columns:
                    self.indicator_ax.bar(x_axis, df['MACD_Hist'], label='Histogram', alpha=0.3)

            # Add legends
            if self.main_ax.get_legend_handles_labels()[0]:
                self.main_ax.legend(loc='upper left', fontsize=8)
            if self.indicator_ax.get_legend_handles_labels()[0]:
                self.indicator_ax.legend(loc='upper left', fontsize=8)

        except Exception as e:
            print(f"Error drawing indicators: {e}")

    def _draw_trading_signals(self, df: pd.DataFrame):
        """Draw buy/sell signals on chart"""
        try:
            signal_names = self.strategy.get_signal_names()
            x_axis = range(len(df))

            # Buy signals
            if signal_names['buy'] in df.columns:
                buy_signals = df[df[signal_names['buy']] == True]
                if not buy_signals.empty:
                    buy_indices = [df.index.get_loc(idx) for idx in buy_signals.index]
                    buy_prices = buy_signals['Close'].values
                    self.main_ax.scatter(buy_indices, buy_prices, color='green', marker='^',
                                       s=100, label='Buy Signal', zorder=5)

            # Sell signals
            if signal_names['sell'] in df.columns:
                sell_signals = df[df[signal_names['sell']] == True]
                if not sell_signals.empty:
                    sell_indices = [df.index.get_loc(idx) for idx in sell_signals.index]
                    sell_prices = sell_signals['Close'].values
                    self.main_ax.scatter(sell_indices, sell_prices, color='red', marker='v',
                                       s=100, label='Sell Signal', zorder=5)

        except Exception as e:
            print(f"Error drawing signals: {e}")

    def _setup_time_axis(self, df: pd.DataFrame):
        """Setup time axis labels"""
        try:
            if len(df) > 0:
                timestamps = [ts.strftime('%H:%M') for ts in df['timestamp']]
                tick_positions = list(range(len(df)))

                # Show every nth timestamp to avoid crowding
                show_every = max(1, len(timestamps) // 8)
                tick_labels = [timestamps[i] if i % show_every == 0 else ''
                             for i in range(len(timestamps))]

                self.main_ax.set_xticks(tick_positions)
                self.main_ax.set_xticklabels(tick_labels, rotation=45)

                self.indicator_ax.set_xticks(tick_positions)
                self.indicator_ax.set_xticklabels(tick_labels, rotation=45)

        except Exception as e:
            print(f"Error setting up time axis: {e}")

    def _add_performance_text(self):
        """Add performance summary text to chart"""
        try:
            performance = self.trading_engine.get_performance_summary()

            # Get current price and position info
            current_price = 0
            unrealized_pnl = 0
            if len(self.data_history) > 0:
                current_price = self.data_history.iloc[-1]['Close']

                if (self.trading_engine.position != 0 and
                    hasattr(self.trading_engine, 'entry_price') and
                    self.trading_engine.entry_price > 0):

                    if self.trading_engine.position == 1:  # Long
                        unrealized_pnl = current_price - self.trading_engine.entry_price
                    elif self.trading_engine.position == -1:  # Short
                        unrealized_pnl = self.trading_engine.entry_price - current_price

            # Position status
            position_status = "FLAT"
            if self.trading_engine.position == 1:
                position_status = f"LONG @ ${self.trading_engine.entry_price:.2f}"
            elif self.trading_engine.position == -1:
                position_status = f"SHORT @ ${self.trading_engine.entry_price:.2f}"

            # Create text
            info_text = (
                f"Price: ${current_price:.2f} | "
                f"Position: {position_status} | "
                f"Balance: ${performance['current_balance']:.2f} | "
                f"P&L: ${performance['total_return'] + unrealized_pnl:.2f} | "
                f"Trades: {performance['total_trades']} | "
                f"Win Rate: {performance['win_rate']:.1f}%"
            )

            self.main_ax.text(0.02, 0.98, info_text, transform=self.main_ax.transAxes,
                            fontsize=10, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        except Exception as e:
            print(f"Error adding performance text: {e}")

    def animate(self, frame):
        """Animation function for live updates"""
        try:
            print(f"\n📊 Fetching data... (Frame {frame})")
            data = self.fetch_and_process_data()

            if data is not None:
                self.draw_candlesticks()
                print(f"✅ Chart updated with {len(data)} candles")
            else:
                print("❌ No data to display")

        except Exception as e:
            print(f"❌ Animation error: {e}")

        return []

    def start_live_trading(self, update_interval: int = 60000):
        """Start live trading with chart updates"""
        print(f"\n🚀 Starting live trading for {self.symbol}")
        print(f"Strategy: {self.strategy.name}")
        print(f"Update interval: {update_interval/1000} seconds")
        print(f"Paper trading: {self.paper_trading}")
        print("=" * 50)

        # Initial data fetch
        self.fetch_and_process_data()

        # Start animation
        ani = animation.FuncAnimation(
            self.fig,
            self.animate,
            interval=update_interval,
            blit=False,
            cache_frame_data=False
        )

        plt.tight_layout()
        plt.show()

        return ani

    def stop_trading(self):
        """Stop live trading"""
        self.trading_engine.stop()
        print("🛑 Live trading stopped")

    def get_trade_history(self):
        """Get trading history"""
        return self.trading_engine.get_trade_history()

    def get_performance_summary(self):
        """Get performance summary"""
        return self.trading_engine.get_performance_summary()