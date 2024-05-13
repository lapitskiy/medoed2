import asyncio
import aiohttp
import zmq
import json

async def fetch_klines(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()

async def send_data(url):
    klines = await fetch_klines(url)
    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    socket.connect("tcp://localhost:5555")  # адрес сервера ZeroMQ
    socket.send_json(klines)
    socket.close()
    context.term()

async def main(urls):
    tasks = [send_data(url) for url in urls]
    await asyncio.gather(*tasks)

if __name__ == '__main__':
    urls = [
        "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1h",
        "https://api.binance.com/api/v3/klines?symbol=ETHUSDT&interval=1h"
    ]
    asyncio.run(main(urls))