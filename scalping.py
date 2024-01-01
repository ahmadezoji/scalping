import asyncio
import time
from bingx import *

quantity = 0.00012
current_order_id = None
order_status_open = False


async def scalping_strategy(symbol, interval):
    global current_order_id
    global order_status_open
    while True:
        try:
            rsi = await get_rsi14(symbol, interval)
            print(f'Current RSI for {interval} interval: {rsi}')

            # if not order_status_open and rsi < 30:
            #     print('RSI is below 30. Initiating buy order...')
            #     response = open_long(symbol=symbol, quantity=quantity)
            #     print(f'Buy order executed: {response.text}')
            #     response.raise_for_status()
            #     data = response.json().get('data', [])
            #     order_info = data.get('order', {})
            #     current_order_id = order_info.get('orderId', None)
            #     if current_order_id is not None:
            #         order_status_open = True

            # elif order_status_open and rsi > 70:
            #     print('RSI is above 70. Initiating sell order...')
            #     close_long(symbol=symbol, quantity=quantity)
            #     print(f'Sell order executed: {response.text}')
            #     order_status_open = False

            await asyncio.sleep(5)

        except Exception as e:
            print(f'An error occurred: {e}')


async def main():
    symbol = "BTC-USDT"
    interval = "1w"
    await asyncio.gather(
        scalping_strategy(symbol, interval),
        # Add more strategies here if needed
    )

if __name__ == '__main__':
    asyncio.run(main())

