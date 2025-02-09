import alpaca_trade_api as tradeapi


ALPACA_API_KEY = 'api-key'
ALPACA_SECRET_KEY = 'secret-key'
ALPACA_BASE_URL = 'https://paper-api.alpaca.markets/'

trade_api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, api_version="v2")

def buy(symbol='BTC', qty=1):
    trade_api.submit_order(
        symbol=symbol,
        qty=qty,
        side='buy',
        type='market',
        time_in_force='gtc'
    )
    return(f"Bought {qty} shares of {symbol}")

def sell(symbol='BTC', qty=1):
    trade_api.submit_order(
        symbol=symbol,
        qty=qty,
        side='sell',
        type='market',
        time_in_force='gtc'
    )
    return(f"Sold {qty} shares of {symbol}")


if __name__ == "__main__": 
    buy()