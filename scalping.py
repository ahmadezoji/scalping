import ccxt
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
            rsi = await get_rsi(symbol, interval)
            print(f'Current RSI for {interval} interval: {rsi}')
            
            if not order_status_open and rsi < 30:
                print('RSI is below 30. Initiating buy order...')
                current_order_id  = open_order(symbol="BTC-USDT",quantity=quantity)
                if current_order_id is not None:
                    order_status_open = True

            elif order_status_open and rsi > 70:
                print('RSI is above 70. Initiating sell order...')
                close_order(symbol=symbol,order_id=current_order_id)
                order_status_open = False

            await asyncio.sleep(5)

        except Exception as e:
            print(f'An error occurred: {e}')

async def main():
    symbol = "BTC-USDT"
    interval = "1m"
    await asyncio.gather(
        scalping_strategy(symbol, interval),
        # Add more strategies here if needed
    )

if __name__ == '__main__':
    asyncio.run(main())
    
    
   
