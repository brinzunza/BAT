from alpaca.data.live import StockDataStream

stream = StockDataStream('***REMOVED***', '***REMOVED***')

async def handle_trade(data):
    print(data)

stream.subscribe_trades(handle_trade, "AAPL")

stream.run()