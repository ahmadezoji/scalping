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
    span = 15
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
    
    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['close'], label='Close Price', color='blue')
    plt.plot(df.index, df['ema9'], label='EMA9', color='red')
    plt.title('Close Price vs EMA9')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

    return df

def emaFinal():
    symbol = "BTC-USDT"
    interval = "1m"
    amount = 0.014  # in btc = 72068 ==> 1000$
    span = 9
    previous_ema = None
    buy_target_index = 0
    sell_target_index = 0
    last_order_id = None
    order_type = OrderType.NONE
    now = int(time.time() * 1000)
    minutes_ago = 10
    treshhold = 3
    durationTime = now - (minutes_ago * 60 * 1000)

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
            df['ema9'] = df['close'].ewm(span=span, adjust=False).mean()


            # Calculate RSI
            # delta = df['close'].diff()
            # gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            # loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            # RS = gain / loss
            # df['RSI'] = 100 - (100 / (1 + RS))

            # Generate signals
            # & (df['RSI'.iloc[-1]] < 31)
            # & (df['RSI'.iloc[-1]] > 69)

            ema9 = df['ema9'].iloc[-1]
            current_price = last_price(symbol=symbol)

            buy_signal = current_price > ema9
            sell_signal = current_price < ema9
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if buy_signal:
                print("buy")
            if sell_signal:
                print("sell")
            if buy_signal:
                sell_target_index = 0
                if (buy_target_index >= treshhold):
                    if (order_type == OrderType.SHORT):
                        result = close_short(symbol=symbol,quantity=amount)
                        if(result == 0):
                            order_type = OrderType.NONE
                            print(f'closed SHORT order id = {last_order_id} ')
                    if order_type == OrderType.NONE:
                        order_type = OrderType.LONG
                        last_order_id, order_type = open_long(
                            symbol=symbol, quantity=amount)
                        print(f'Buy signal at {current_time}')
                buy_target_index = buy_target_index + 1

            if sell_signal:
                buy_target_index = 0
                if (sell_target_index >= treshhold):
                    if (order_type == OrderType.LONG):
                        result = close_long(symbol=symbol,quantity=amount)
                        if(result == 0):
                            order_type = OrderType.NONE
                            print(f'closed LONG order id = {last_order_id} ')
                    if order_type == OrderType.NONE:
                        last_order_id, order_type = open_short(
                            symbol=symbol, quantity=amount)
                        order_type = OrderType.SHORT
                        print(f'Sell signal at {current_time}') 
                sell_target_index = sell_target_index + 1
            
            time.sleep(60)
       except StopIteration:
            break    
    

async def main():
    await asyncio.gather(
        emaFinal()
    )


if __name__ == '__main__':
    asyncio.run(main())
    # emaChart()
