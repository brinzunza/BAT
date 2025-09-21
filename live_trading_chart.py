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
        if self.broker:
            account_info = self.broker.get_account_api() if hasattr(self.broker, 'get_account_api') else self.broker.get_account()
            initial_balance = float(account_info.get('equity', 10000))
            print(f"💰 Account Equity: ${initial_balance:,.2f}")
            print(f"💰 Buying Power: ${float(account_info.get('buying_power', 0)):,.2f}")
        else:
            initial_balance = 10000
            print(f"💰 Simulation Balance: ${initial_balance:,.2f}")

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
                print(f"🚀 Initializing {self.symbol} data...")

                # Use the public endpoint to get recent bars immediately
                initial_df = self.data_provider.get_recent_bars_public(self.symbol, limit=self.min_data_points + 10)

                if not initial_df.empty:
                    # Use the fetched data, keep only what we need
                    self.data_history = initial_df.tail(self.min_data_points).copy().reset_index(drop=True)

                    # Mark as ready since we have enough data
                    if len(self.data_history) >= self.min_data_points:
                        self.data_ready = True
                        print(f"✅ Ready for live trading with {len(self.data_history)} bars")
                else:
                    print(f"❌ Failed to get initial data")
                    return None

            # Live data update phase (after initial load)
            else:
                # Get latest bar for live updates
                latest_bar = self.data_provider.get_latest_bar(self.symbol)

                if not latest_bar:
                    # Continue with existing data silently
                    pass
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
                        # Display OHLCV data
                        print(f"\n📊 NEW DATA - {self.symbol} at {latest_bar['timestamp'].strftime('%H:%M:%S')}")
                        print(f"    O: ${latest_bar['Open']:.2f} | H: ${latest_bar['High']:.2f} | L: ${latest_bar['Low']:.2f} | C: ${latest_bar['Close']:.2f} | V: {latest_bar['Volume']:.0f}")

                        # Check for active position and display unrealized PnL
                        self._display_position_update(latest_bar['Close'])

                        # Add new bar
                        self.data_history = pd.concat([self.data_history, new_row], ignore_index=True)

                        # Keep only max_candles
                        if len(self.data_history) > self.max_candles:
                            self.data_history = self.data_history.tail(self.max_candles).reset_index(drop=True)

            # Process data if we have enough
            if len(self.data_history) >= self.min_data_points:
                if not self.data_ready:
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

            # Execute trades through trading engine (only show signals when they trigger)
            if buy_signal or sell_signal:
                self.trading_engine.process_signals(
                    self.data_history,
                    self.strategy,
                    self.symbol,
                    self.quantity
                )
                self.last_trade_time = current_time

        except Exception as e:
            print(f"❌ Error processing trading signals: {e}")

    def _display_position_update(self, current_price: float):
        """Display position and unrealized PnL update with new price data"""
        try:
            if self.broker and hasattr(self.broker, 'get_position_for_symbol'):
                position = self.broker.get_position_for_symbol(self.symbol)
                current_qty = float(position.get('qty', 0))

                if current_qty > 0:  # We have an active position
                    unrealized_pnl = float(position.get('unrealized_pl', 0))
                    avg_entry_price = float(position.get('avg_entry_price', 0))
                    market_value = float(position.get('market_value', 0))

                    # Get account info
                    account_info = self.broker.get_account_api() if hasattr(self.broker, 'get_account_api') else self.broker.get_account()
                    account_balance = float(account_info.get('equity', 0))
                    session_pnl = account_balance - self.trading_engine.initial_balance

                    print(f"    📈 POSITION: {current_qty} {self.symbol} @ ${avg_entry_price:.2f} | Current: ${current_price:.2f}")
                    print(f"    💰 Market Value: ${market_value:.2f} | Unrealized P&L: ${unrealized_pnl:.2f} | Session P&L: ${session_pnl:.2f}")

        except Exception as e:
            # Silently continue if position check fails
            pass

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
        """Get unrealized P&L from Alpaca"""
        try:
            if self.broker and hasattr(self.broker, 'get_position_for_symbol'):
                position = self.broker.get_position_for_symbol(self.symbol)
                return float(position.get('unrealized_pl', 0))
            return 0
        except Exception:
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
        """Draw strategy-specific indicators dynamically"""
        try:
            x_axis = range(len(df))

            # Get the strategy's indicators
            strategy_indicators = self.strategy.get_indicators()

            # Clear indicator axis first
            self.indicator_ax.clear()
            self.indicator_ax.set_xlabel('Time')
            self.indicator_ax.grid(True, alpha=0.3)

            # Draw indicators based on strategy type
            strategy_name = self.strategy.name.lower()

            # === BOLLINGER BANDS STRATEGY ===
            if 'bollinger' in strategy_name or any('bb_' in indicator for indicator in strategy_indicators):
                self._draw_bollinger_bands(df, x_axis)

            # === MEAN REVERSION STRATEGY ===
            elif 'mean reversion' in strategy_name or any(indicator in ['SMA', 'STD', 'Upper Band', 'Lower Band'] for indicator in strategy_indicators):
                self._draw_mean_reversion_indicators(df, x_axis)

            # === RSI STRATEGY ===
            elif 'rsi' in strategy_name or 'rsi' in strategy_indicators:
                self._draw_rsi_indicators(df, x_axis)

            # === MACD STRATEGY ===
            elif 'macd' in strategy_name or any('macd' in indicator.lower() for indicator in strategy_indicators):
                self._draw_macd_indicators(df, x_axis)

            # === GENERIC INDICATOR DRAWING ===
            else:
                self._draw_generic_indicators(df, x_axis, strategy_indicators)

            # Add legends
            if self.main_ax.get_legend_handles_labels()[0]:
                self.main_ax.legend(loc='upper left', fontsize=8)
            if self.indicator_ax.get_legend_handles_labels()[0]:
                self.indicator_ax.legend(loc='upper left', fontsize=8)

        except Exception as e:
            print(f"Error drawing indicators: {e}")

    def _draw_bollinger_bands(self, df: pd.DataFrame, x_axis):
        """Draw Bollinger Bands indicators"""
        if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
            self.main_ax.plot(x_axis, df['bb_upper'], label='BB Upper', color='red', alpha=0.6, linestyle='--')
            self.main_ax.plot(x_axis, df['bb_lower'], label='BB Lower', color='green', alpha=0.6, linestyle='--')
            self.main_ax.fill_between(x_axis, df['bb_upper'], df['bb_lower'], alpha=0.1, color='blue', label='BB Band')

        if 'bb_middle' in df.columns:
            self.main_ax.plot(x_axis, df['bb_middle'], label='BB Middle (SMA)', color='blue', alpha=0.8)

        self.indicator_ax.set_ylabel('Bollinger Band %')
        # Calculate BB percentage (position within bands)
        if all(col in df.columns for col in ['bb_upper', 'bb_lower', 'Close']):
            bb_percent = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower']) * 100
            self.indicator_ax.plot(x_axis, bb_percent, label='BB %', color='purple')
            self.indicator_ax.axhline(y=80, color='red', linestyle='--', alpha=0.5, label='Overbought')
            self.indicator_ax.axhline(y=20, color='green', linestyle='--', alpha=0.5, label='Oversold')
            self.indicator_ax.set_ylim(0, 100)

    def _draw_mean_reversion_indicators(self, df: pd.DataFrame, x_axis):
        """Draw Mean Reversion strategy indicators"""
        if 'SMA' in df.columns:
            self.main_ax.plot(x_axis, df['SMA'], label='SMA', color='blue', alpha=0.8, linewidth=2)

        if 'Upper Band' in df.columns and 'Lower Band' in df.columns:
            self.main_ax.plot(x_axis, df['Upper Band'], label='Upper Band (+2σ)', color='red', alpha=0.6, linestyle='--')
            self.main_ax.plot(x_axis, df['Lower Band'], label='Lower Band (-2σ)', color='green', alpha=0.6, linestyle='--')
            self.main_ax.fill_between(x_axis, df['Upper Band'], df['Lower Band'], alpha=0.1, color='gray', label='±2σ Range')

        # Draw standard deviation in indicator panel
        if 'STD' in df.columns:
            self.indicator_ax.plot(x_axis, df['STD'], label='Standard Deviation', color='orange', linewidth=2)
            self.indicator_ax.set_ylabel('Standard Deviation')
            # Add reference lines for volatility levels
            std_mean = df['STD'].mean()
            self.indicator_ax.axhline(y=std_mean, color='orange', linestyle=':', alpha=0.5, label=f'Avg STD ({std_mean:.2f})')

    def _draw_rsi_indicators(self, df: pd.DataFrame, x_axis):
        """Draw RSI strategy indicators"""
        if 'rsi' in df.columns:
            self.indicator_ax.plot(x_axis, df['rsi'], label='RSI', color='purple', linewidth=2)
            self.indicator_ax.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Overbought (70)')
            self.indicator_ax.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Oversold (30)')
            self.indicator_ax.axhline(y=50, color='gray', linestyle=':', alpha=0.5, label='Midline')
            self.indicator_ax.set_ylim(0, 100)
            self.indicator_ax.set_ylabel('RSI Value')

            # Add background coloring for zones
            self.indicator_ax.fill_between(x_axis, 70, 100, alpha=0.1, color='red', label='Overbought Zone')
            self.indicator_ax.fill_between(x_axis, 0, 30, alpha=0.1, color='green', label='Oversold Zone')

    def _draw_macd_indicators(self, df: pd.DataFrame, x_axis):
        """Draw MACD strategy indicators"""
        if 'macd_line' in df.columns:
            self.indicator_ax.plot(x_axis, df['macd_line'], label='MACD Line', color='blue', linewidth=2)

        if 'signal_line' in df.columns:
            self.indicator_ax.plot(x_axis, df['signal_line'], label='Signal Line', color='red', linewidth=2)

        if 'histogram' in df.columns:
            # Color histogram bars based on positive/negative values
            colors = ['green' if x >= 0 else 'red' for x in df['histogram']]
            self.indicator_ax.bar(x_axis, df['histogram'], label='MACD Histogram', alpha=0.6, color=colors)

        self.indicator_ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, label='Zero Line')
        self.indicator_ax.set_ylabel('MACD Values')

    def _draw_generic_indicators(self, df: pd.DataFrame, x_axis, indicators):
        """Draw generic indicators for unknown strategies"""
        main_indicators = []
        oscillator_indicators = []

        # Categorize indicators
        for indicator in indicators:
            if indicator in df.columns:
                # Check if it's likely an oscillator (bounded indicator)
                values = df[indicator].dropna()
                if len(values) > 0:
                    min_val, max_val = values.min(), values.max()
                    if 0 <= min_val and max_val <= 100:  # Likely an oscillator
                        oscillator_indicators.append(indicator)
                    else:
                        main_indicators.append(indicator)

        # Draw main chart indicators
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
        for i, indicator in enumerate(main_indicators):
            color = colors[i % len(colors)]
            self.main_ax.plot(x_axis, df[indicator], label=indicator, color=color, alpha=0.7)

        # Draw oscillator indicators
        for i, indicator in enumerate(oscillator_indicators):
            color = colors[i % len(colors)]
            self.indicator_ax.plot(x_axis, df[indicator], label=indicator, color=color, linewidth=2)

        if oscillator_indicators:
            self.indicator_ax.set_ylabel('Oscillator Values')

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

                # Get current price and position info from Alpaca
            current_price = 0
            unrealized_pnl = 0
            position_status = "FLAT"

            if len(self.data_history) > 0:
                current_price = self.data_history.iloc[-1]['Close']

                # Get position data from Alpaca
                if self.broker and hasattr(self.broker, 'get_position_for_symbol'):
                    try:
                        position = self.broker.get_position_for_symbol(self.symbol)
                        current_qty = float(position.get('qty', 0))
                        avg_entry_price = float(position.get('avg_entry_price', 0))
                        unrealized_pnl = float(position.get('unrealized_pl', 0))

                        # Format position status using Alpaca data
                        if current_qty > 0:
                            position_status = f"LONG {current_qty} @ ${avg_entry_price:.2f}"
                        elif current_qty < 0:
                            position_status = f"SHORT {abs(current_qty)} @ ${avg_entry_price:.2f}"
                        else:
                            position_status = "FLAT"
                    except Exception:
                        # Fallback to engine data
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
            data = self.fetch_and_process_data()

            if data is not None:
                self.draw_candlesticks()

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