# BAT - Bruninvestor Algorithmic Trading

## Overview
BAT is an algorithmic trading project that implements two distinct trading strategies using Python:

1. **Mean Reversion Strategy**
2. **Moving Averages Strategy**

This project demonstrates the application of quantitative analysis and algorithmic decision-making in financial markets.

## Features

- Data collection via Polygon API
- Comprehensive data processing and analysis
- Strategy implementation and backtesting
- Performance analysis and visualization

## Strategies

### Mean Reversion Strategy
This strategy is based on the assumption that asset prices tend to revert to their historical average over time. The algorithm:
- Buys when prices fall below the mean*
- Sells when prices rise above the mean*

### Moving Averages Strategy
This strategy uses three moving averages (short, medium, and long-term) to identify trends and generate trading signals:
- Buy signals occur when shorter moving averages cross above longer ones*
- Sell signals occur when shorter moving averages cross below longer ones*

## Future Improvements
- Implement additional strategies (e.g., momentum, sentiment analysis)
- Optimize parameters using machine learning techniques
- Integrate real-time data feeds for live trading

## Disclaimer
This project is for educational purposes only. It is not financial advice. Always conduct thorough research or consult with a professional financial advisor before making investment decisions.

*Typical application of strategy