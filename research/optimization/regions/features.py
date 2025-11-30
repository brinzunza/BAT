import pandas as pd

class Features:
    def __init__(self, df: pd.DataFrame, window: int = 50):
        self.df = df
        self.window = window

    # Trend rating is the number of times the price has changed direction in the last window periods
    def trend_rating(self):
        switches = 0

        # Reset index to ensure we can use iloc properly
        df = self.df.reset_index(drop=True)

        for i in range(1, len(df)):  # Start from 1 to avoid i-1 = -1
            if df['Close'].iloc[i] > df['Open'].iloc[i] and df['Close'].iloc[i-1] < df['Open'].iloc[i-1]:
                switches += 1
            elif df['Close'].iloc[i] < df['Open'].iloc[i] and df['Close'].iloc[i-1] > df['Open'].iloc[i-1]:
                switches += 1

        return switches / self.window

    # Volatility is the standard deviation of the price in the window
    def volatility(self):
        df = self.df.reset_index(drop=True)
        vol = df['Close'].std() / df['Close'].mean() if df['Close'].mean() > 0 else 0.0
        return vol if not pd.isna(vol) else 0.0

    # Volume normalized to window average
    def volume(self):
        df = self.df.reset_index(drop=True)
        vol = df['Volume'].mean()
        return vol if not pd.isna(vol) else 0.0

    # Momentum as the ratio of last close to first close in window
    def momentum(self):
        df = self.df.reset_index(drop=True)
        if len(df) > 0 and df['Close'].iloc[0] > 0:
            mom = df['Close'].iloc[-1] / df['Close'].iloc[0]
            return mom if not pd.isna(mom) else 1.0
        return 1.0