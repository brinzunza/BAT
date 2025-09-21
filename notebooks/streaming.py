from alpaca.data.live import StockDataStream

stream = StockDataStream('PK5CWVEHZZDVMOO9PPRK', 'tqvXaTZYR4tvkxXJl2pueh2zFf2Yi3u5y1vhLb9f')

async def handle_trade(data):
    print(data)

stream.subscribe_trades(handle_trade, "AAPL")

stream.run()