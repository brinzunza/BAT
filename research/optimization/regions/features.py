class Features:
    def __init__(self, df: pd.DataFrame, window: int = 50):
        self.df = df
        self.window = window

    # Trend rating is the number of times the price has changed direction in the last window periods
    def trend_rating(self):
        switches = 0
        
        for i in range(len(self.df)):
            if self.df['Close'][i] > self.df['Open'][i] and self.df['Close'][i-1] < self.df['Open'][i-1]:
                switches += 1
            elif self.df['Close'][i] < self.df['Open'][i] and self.df['Close'][i-1] > self.df['Open'][i-1]:
                switches += 1
                
        return switches / self.window

    # Volatility is the standard deviation of the price in the last window periods
    def volatility(self):
        return self.df['Close'].rolling(window=self.window).std() / self.df['Close'].mean()

    # Volume is the average volume in the last window periods
    def volume(self):
        return self.df['Volume'].rolling(window=self.window).mean() / self.df['Volume'].mean()

    # Momentum is the average momentum in the last window periods
    def momentum(self):
        return self.df['Close'].rolling(window=self.window).mean() / self.df['Close'].mean()