import pandas as pd
import numpy as np
import time
import datetime
import asyncio
from bingx import *
import matplotlib.pyplot as plt


global last_order_id
global order_type


# Function to calculate EMA based on the previous EMA value and the current price
def calculate_ema(previous_ema, current_price, span):
    alpha = 2 / (span + 1)
    new_ema = (current_price - previous_ema) * alpha + previous_ema
    return new_ema

async def ema_real_time():
    symbol = "BTC-USDT"
    interval = "1m"  # 5-minute interval for real-time
    span_short = 9  # Short-term EMA
    span_long = 21  # Long-term EMA
    treshhold = 0  # Threshold to avoid false signals
    order_type = OrderType.NONE

    # Initialize the EMA values
    last_ema_short = None
    last_ema_long = None
    previous_ema_short = None
    previous_ema_long = None
    buy_target_index = 0
    sell_target_index = 0

    _last_price = 0

    while True:
        try:
            current_price = last_price(symbol) 
            _last_price = current_price
        except Exception as e:
            current_price = _last_price
        
        if current_price is None:
            print("Could not fetch the current price. Retrying...")
            await asyncio.sleep(60)
            continue

        # Initialize EMA with the current price if not already set
        if last_ema_short is None or last_ema_long is None:
            last_ema_short = current_price
            last_ema_long = current_price
            previous_ema_short = current_price
            previous_ema_long = current_price
        else:
            # Store previous EMA values for crossover detection
            previous_ema_short = last_ema_short
            previous_ema_long = last_ema_long

            # Update EMA values with the current price
            last_ema_short = calculate_ema(last_ema_short, current_price, span_short)
            last_ema_long = calculate_ema(last_ema_long, current_price, span_long)

        print(f'Current Price = {current_price}, EMA Short = {last_ema_short}, EMA Long = {last_ema_long}')

        # Buy Signal: When the short-term EMA crosses above the long-term EMA
        # We check if the previous short EMA was below the long EMA, and now it's above
        buy_signal = previous_ema_short <= previous_ema_long and last_ema_short > last_ema_long
        
        # Sell Signal: When the short-term EMA crosses below the long-term EMA
        # We check if the previous short EMA was above the long EMA, and now it's below
        sell_signal = previous_ema_short >= previous_ema_long and last_ema_short < last_ema_long

        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())

        if buy_signal:
            sell_target_index = 0
            if buy_target_index >= treshhold:
                if order_type == OrderType.SHORT:
                    order_type = OrderType.NONE
                    print(f"Closed SHORT order at {current_time}")
                if order_type == OrderType.NONE:
                    order_type = OrderType.LONG
                    print(f"Buy signal at {current_time} - Price: {current_price}")
            buy_target_index += 1
            print("Buy")
        else:
            buy_target_index = 0

        if sell_signal:
            buy_target_index = 0
            if sell_target_index >= treshhold:
                if order_type == OrderType.LONG:
                    order_type = OrderType.NONE
                    print(f"Closed LONG order at {current_time}")
                if order_type == OrderType.NONE:
                    order_type = OrderType.SHORT
                    print(f"Sell signal at {current_time} - Price: {current_price}")
            sell_target_index += 1
            print("Sell")
        else:
            sell_target_index = 0

        # Wait for the next price update (5 minutes)
        await asyncio.sleep(60)

# Run the asyncio event loop
if __name__ == "__main__":
    asyncio.run(ema_real_time())