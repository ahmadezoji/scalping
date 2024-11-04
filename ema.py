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


def calculate_ema(current_price, previous_ema, span):
    alpha = 2 / (span + 1)
    new_ema = (current_price - previous_ema) * alpha + previous_ema
    return new_ema


def fetch_latest_data_and_process(symbol, interval="15m", span_short=9, span_long=21, buffer_percentage=0.003, treshhold=2):
    # Initialize variables to store previous EMA values
    previous_ema9 = None
    previous_ema21 = None
    order_type = OrderType.NONE
    buy_target_index = 0
    sell_target_index = 0

    while True:
        try:
            now = int(time.time() * 1000)
            minutes_ago = 15  # Fetch data for the last 15 minutes
            start_time = now - (minutes_ago * 60 * 1000)

            # Fetch the most recent data point using start time
            response = get_kline(symbol, interval, start=start_time)
            response.raise_for_status()
            response = response.json().get('data', [])
            df = pd.DataFrame(response, columns=[
                'time', 'open', 'high', 'low', 'close', 'volume'])
            df['time'] = pd.to_datetime(df['time'], unit='ms')
            df.set_index('time', inplace=True)
            df['close'] = df['close'].astype(float)

            latest_data = response[-1]  # The latest data point

            current_price = float(latest_data[4])  # Close price
            
            current_time = pd.to_datetime(
                latest_data[0], unit='ms').strftime("%Y-%m-%d %H:%M:%S")

            if previous_ema9 is None or previous_ema21 is None:
                # If first run, initialize EMA with current price
                previous_ema9 = current_price
                previous_ema21 = current_price

            # Calculate the new EMA values
            ema9 = calculate_ema(current_price, previous_ema9, span_short)
            ema21 = calculate_ema(current_price, previous_ema21, span_long)

            # Update previous EMA values for the next iteration
            previous_ema9 = ema9
            previous_ema21 = ema21

            # Calculate price difference and determine buy/sell signals
            price_difference = current_price - ema9
            buy_signal = (price_difference > (
                ema9 * buffer_percentage) and ema9 > ema21)
            sell_signal = (price_difference < -
                           (ema9 * buffer_percentage) and ema9 < ema21)

            if buy_signal:
                sell_target_index = 0
                if buy_target_index >= treshhold:
                    if order_type == OrderType.SHORT:
                        order_type = OrderType.NONE
                    if order_type == OrderType.NONE:
                        order_type = OrderType.LONG
                        print(f'Buy signal at {
                              current_time} | Price: {current_price}')
                buy_target_index += 1
            else:
                buy_target_index = 0

            if sell_signal:
                buy_target_index = 0
                if sell_target_index >= treshhold:
                    if order_type == OrderType.LONG:
                        order_type = OrderType.NONE
                    if order_type == OrderType.NONE:
                        order_type = OrderType.SHORT
                        print(f'Sell signal at {
                              current_time} | Price: {current_price}')
                sell_target_index += 1
            else:
                sell_target_index = 0

        except Exception as e:
            print(f"Error: {e}")

        # Sleep until the next 15-minute interval
        time.sleep(60)


async def main():
    await asyncio.gather(
        fetch_latest_data_and_process("BTC-USDT")
    )


if __name__ == '__main__':
    asyncio.run(main())
