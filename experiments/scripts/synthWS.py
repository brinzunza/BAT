import asyncio
import websockets
import json

async def connect():
    async with websockets.connect('ws://localhost:3000/ws') as websocket:
        print('Connected to market data feed')

        # Subscribe to tickers
        await websocket.send(json.dumps({
            'type': 'subscribe',
            'tickers': ['SYNTH', 'TECH', 'FINANCE']
        }))

        # Listen for messages
        while True:
            message = json.loads(await websocket.recv())

            if message['type'] == 'tick':
                ticker = message['data']['ticker']
                price = message['data']['price']
                volume = message['data']['volume']
                print(f"{ticker}: ${price} (Volume: {volume})")

asyncio.run(connect())
