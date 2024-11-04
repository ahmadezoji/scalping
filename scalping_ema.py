import pandas as pd
import numpy as np
import time
import datetime
import asyncio
from bingx import *
import matplotlib.pyplot as plt


global last_order_id
global order_type

def emaChart():
    symbol = "BTC-USDT"
    interval = "1m"
    span = 9
    now = int(time.time() * 1000)
    minutes_ago = 100

    durationTime = now - (minutes_ago * 60 * 1000)
    response = get_kline(symbol, interval, start=durationTime)
    response.raise_for_status()
    response = response.json().get('data', [])
    df = pd.DataFrame(response, columns=[
        'time', 'open', 'high', 'low', 'close', 'volume'])
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    df.set_index('time', inplace=True)


    df['close'] = df['close'].astype(float)
    

    # Calculate EMA9
    df['ema9'] = df['close'].ewm(span=span, adjust=False).mean()
    df['ema9_shifted'] = df['ema9'].shift(-8) 


    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['close'], label='Close Price', color='blue')
    plt.plot(df.index, df['ema9_shifted'], label='EMA9', color='red')
    plt.title('Close Price vs EMA9')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

    return df
def calculate_ema(current_close, previous_ema, span):
    alpha = 2 / (span + 1)
    return (current_close - previous_ema) * alpha + previous_ema

def emaFinal():
    symbol = "BTC-USDT"
    interval = "1h"
    
    amount = 0.014  # in btc = 72068 ==> 1000$
    order_type = OrderType.NONE

    span = 20  # Adjusted for 1-hour time frame
    previous_ema = None
    buy_target_index = 0
    sell_target_index = 0
    last_order_id = None
    order_type = OrderType.NONE
    now = int(time.time() * 1000)
    minutes_ago = 30
    hours_ago = 5
    treshhold = 4
    durationTime = now - ((hours_ago * 60) + minutes_ago * 60 * 1000)

    while True:
        try:
            response = get_kline(symbol, interval, start=durationTime)
            response.raise_for_status()
            response = response.json().get('data', [])
            df = pd.DataFrame(response, columns=[
                'time', 'open', 'high', 'low', 'close', 'volume'])
            df['time'] = pd.to_datetime(df['time'], unit='ms')
            df.set_index('time', inplace=True)
            df['close'] = df['close'].astype(float)
            df['ema20'] = df['close'].ewm(span=span, adjust=False).mean()

            last_valid_ema = df['ema20'].dropna().iloc[-1]
            current_price = last_price(symbol=symbol)
            
            buy_signal = current_price > last_valid_ema
            sell_signal = current_price < last_valid_ema
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            if buy_signal:
                sell_target_index = 0
                if (buy_target_index >= treshhold):
                    if (order_type == OrderType.SHORT):
                        result = close_short(symbol=symbol, quantity=amount)
                        if(result == 0):
                            order_type = OrderType.NONE
                            print(f'closed SHORT order id = {last_order_id} ')
                    if order_type == OrderType.NONE:
                        order_type = OrderType.LONG
                        last_order_id, order_type = open_long(
                            symbol=symbol, quantity=amount)
                        print(f'Buy signal at {current_time}')
                buy_target_index += 1

            if sell_signal:
                buy_target_index = 0
                if (sell_target_index >= treshhold):
                    if (order_type == OrderType.LONG):
                        result = close_long(symbol=symbol, quantity=amount)
                        if(result == 0):
                            order_type = OrderType.NONE
                            print(f'closed LONG order id = {last_order_id} ')
                    if order_type == OrderType.NONE:
                        last_order_id, order_type = open_short(
                            symbol=symbol, quantity=amount)
                        order_type = OrderType.SHORT
                        print(f'Sell signal at {current_time}') 
                sell_target_index += 1
            
            time.sleep(3600)  # Adjusted to match the 1-hour interval
        except StopIteration:
            break

async def main():
    await asyncio.gather(
        emaFinal()
    )


if __name__ == '__main__':
    # asyncio.run(main())
    emaChart()
#    