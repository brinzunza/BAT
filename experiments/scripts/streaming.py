import os
from alpaca.data.live import StockDataStream

stream = StockDataStream(os.environ['APCA_API_KEY_ID'], os.environ['APCA_API_SECRET_KEY'])

async def handle_trade(data):
    print(data)

stream.subscribe_trades(handle_trade, "AAPL")

stream.run()