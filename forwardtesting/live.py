from dotenv import load_dotenv
load_dotenv()

import os
import alpaca_trade_api as tradeapi
import pandas as pd
import requests
from polygon import BaseClient
import matplotlib.pyplot as  plt
from datetime import datetime, timedelta
import time

ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
ALPACA_BASE_URL = 'https://paper-api.alpaca.markets/'

trade_api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, api_version='v2')

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

def getData(ticker='X:BTCUSD', timespan='minute', limit=50000):
    api_key = os.getenv('POLYGON_API_KEY')
    to_date = datetime.now() 
    from_date = to_date - timedelta(days=1) 
    to_date = to_date.strftime('%Y-%m-%d') 
    from_date = from_date.strftime('%Y-%m-%d')
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/{timespan}/{from_date}/{to_date}?adjusted=true&sort=asc&limit={limit}&apiKey={api_key}"
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data['results'])
    df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
    df.rename(columns={'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume'}, inplace=True)
    df.drop(columns=['vw', 'n', 't'], inplace=True)
    return df

def getSignals(df, window=20, num_std=2):
    df['SMA'] = df['Close'].rolling(window).mean()
    df['STD'] = df['Close'].rolling(window).std()

    df['Upper Band'] = df['SMA'] + (df['STD'] * num_std)
    df['Lower Band'] = df['SMA'] - (df['STD'] * num_std)

    df['Buy Signal'] = (df['Close'] < df['Lower Band'])   
    df['Sell Signal'] = (df['Close'] > df['Upper Band']) 

    return df

def run_trading_bot():
    position = 0
    while(1):
        data = getData()
        signals = getSignals(data)
        sellAction = signals.iloc[-1]['Sell Signal']
        buyAction = signals.iloc[-1]['Buy Signal']
        print()
        if position == 0:
            if(buyAction == True):
                buy()
                position = 1
            elif(sellAction == True):
                sell()
                position = -1
            else:
                print ("No affects on positions / Current Position: " + str(position))
        elif position == 1:
            if(sellAction == True):
                sell()
                sell()
                position = -1
            else:
                print ("No affects on positions / Current Position: " + str(position))
        elif position == -1:
            if(buyAction == True):
                buy()
                buy()
                position = 1
            else:
                print ("No affects on positions / Current Position: " + str(position))
    
        time.sleep(60)

if __name__ == "__main__":
    run_trading_bot()