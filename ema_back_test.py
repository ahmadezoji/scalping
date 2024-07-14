import pandas as pd
import numpy as np
import time
import datetime
import asyncio
from bingx import *
import matplotlib.pyplot as plt


global last_order_id
global order_type


# Main function to backtest the strategy
def ema_backtest():
    symbol = "BTC-USDT"
    interval = "1m"
    amount = 0.014  # in btc = 72068 ==> 1000$
    span = 9
    treshhold = 4
    order_type = OrderType.NONE
    buy_target_index = 0
    sell_target_index = 0
    last_order_id = None

    now = int(time.time() * 1000)
    minutes_ago = 60 * 11  # Backtest for the 5 hour ago
    durationTime = now - (minutes_ago * 60 * 1000)

    buy_signals = []
    sell_signals = []
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
        df['ema9_shifted'] = df['ema9'].shift(-8)
        for index, row in df.iterrows():
            if pd.isna(row['ema9_shifted']):
                continue

            current_price = row['close']
            last_valid_ema = row['ema9_shifted']

            buy_signal = current_price > last_valid_ema
            sell_signal = current_price < last_valid_ema
            current_time = index.strftime("%Y-%m-%d %H:%M:%S")

            if buy_signal:
                sell_target_index = 0
                if buy_target_index >= treshhold:
                    if order_type == OrderType.SHORT:
                        order_type = OrderType.NONE
                        # print(f'closed SHORT order id = {last_order_id} at {current_time}')
                    if order_type == OrderType.NONE:
                        order_type = OrderType.LONG
                        print(f'Buy signal at {current_time}')
                        
                        sell_signals.append(index)
                buy_target_index += 1

            if sell_signal:
                buy_target_index = 0
                if sell_target_index >= treshhold:
                    if order_type == OrderType.LONG:
                        order_type = OrderType.NONE
                        # print(f'closed LONG order id = {last_order_id} at {current_time}')
                    if order_type == OrderType.NONE:
                        order_type = OrderType.SHORT
                        print(f'Sell signal at {current_time}')
                        buy_signals.append(index)
                sell_target_index += 1

    except Exception as e:
        print(f"Error: {e}")
    
    # Plotting
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['close'], label='Close Price', color='blue')
    plt.plot(df.index, df['ema9'], label='EMA 9', color='orange')

    plt.scatter(buy_signals, df.loc[buy_signals]['close'], marker='^', color='blue', lw=3, label='Buy Signal')
    plt.scatter(sell_signals, df.loc[sell_signals]['close'], marker='v', color='red', lw=3, label='Sell Signal')

    plt.title(f'{symbol} Price with Buy/Sell Signals')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    ema_backtest()