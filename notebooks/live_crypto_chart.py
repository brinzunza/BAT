import requests
import time
import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime
import pandas as pd
from matplotlib.patches import Rectangle

class LiveCryptoChart:
    def __init__(self):
        self.url = "https://data.alpaca.markets/v1beta3/crypto/us/latest/bars?symbols=BTC%2FUSD"
        self.headers = {"accept": "application/json"}
        self.data = []

        # Set up the plot
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.ax.set_title('BTC/USD Live Candlestick Chart')
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('Price (USD)')
        self.ax.grid(True, alpha=0.3)

    def fetch_data(self):
        try:
            response = requests.get(self.url, headers=self.headers)
            response_data = json.loads(response.text)

            if 'bars' in response_data and 'BTC/USD' in response_data['bars']:
                bar_data = response_data['bars']['BTC/USD']
                timestamp = datetime.now()

                ohlc = {
                    'timestamp': timestamp,
                    'open': float(bar_data['o']),
                    'high': float(bar_data['h']),
                    'low': float(bar_data['l']),
                    'close': float(bar_data['c']),
                    'volume': float(bar_data['v'])
                }

                self.data.append(ohlc)

                # Keep only last 50 data points for better visualization
                if len(self.data) > 50:
                    self.data.pop(0)

                print(f"{timestamp.strftime('%H:%M:%S')} - O: ${ohlc['open']:.2f}, H: ${ohlc['high']:.2f}, L: ${ohlc['low']:.2f}, C: ${ohlc['close']:.2f}")

        except Exception as e:
            print(f"Error fetching data: {e}")

    def draw_candlesticks(self):
        self.ax.clear()
        self.ax.set_title('BTC/USD Live Candlestick Chart')
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('Price (USD)')
        self.ax.grid(True, alpha=0.3)

        if len(self.data) < 1:
            return

        for i, candle in enumerate(self.data):
            open_price = candle['open']
            high_price = candle['high']
            low_price = candle['low']
            close_price = candle['close']

            # Determine candle color (green for bullish, red for bearish)
            color = 'green' if close_price >= open_price else 'red'

            # Draw the wick (high-low line)
            self.ax.plot([i, i], [low_price, high_price], color='black', linewidth=1)

            # Draw the body (rectangle from open to close)
            body_height = abs(close_price - open_price)
            body_bottom = min(open_price, close_price)

            rect = Rectangle((i - 0.3, body_bottom), 0.6, body_height,
                           facecolor=color, edgecolor='black', alpha=0.8)
            self.ax.add_patch(rect)

        # Set x-axis labels to show timestamps
        if len(self.data) > 0:
            timestamps = [d['timestamp'].strftime('%H:%M:%S') for d in self.data]
            tick_positions = list(range(len(self.data)))

            # Show every 5th timestamp to avoid crowding
            show_every = max(1, len(timestamps) // 10)
            tick_labels = [timestamps[i] if i % show_every == 0 else '' for i in range(len(timestamps))]

            self.ax.set_xticks(tick_positions)
            self.ax.set_xticklabels(tick_labels, rotation=45)

        # Auto-scale the plot
        self.ax.relim()
        self.ax.autoscale_view()

    def animate(self, frame):
        self.fetch_data()
        self.draw_candlesticks()
        return []

    def start_live_plot(self):
        # Start the animation (updates every 60 seconds)
        ani = animation.FuncAnimation(
            self.fig, self.animate, interval=60000, blit=False, cache_frame_data=False
        )

        plt.tight_layout()
        plt.show()
        return ani

if __name__ == "__main__":
    chart = LiveCryptoChart()
    print("Starting live BTC/USD chart...")
    print("Fetching data every 60 seconds...")
    chart.start_live_plot()