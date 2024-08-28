import pandas as pd
import numpy as np
import time
import datetime
import asyncio
from bingx import *
import matplotlib.pyplot as plt


global last_order_id
global order_type



def ema_backtest():
    symbol = "BTC-USDT"
    interval = "1h"
    amount = 0.014  # in btc = 72068 ==> 1000$
    span_short = 20  # Short-term EMA for signal detection
    span_long = 50  # Long-term EMA for trend confirmation
    treshhold = 3  # Reduce threshold to allow more signals
    buffer_percentage = 0.005  # 0.5% buffer to avoid small fluctuations
    order_type = OrderType.NONE
    buy_target_index = 0
    sell_target_index = 0
    last_order_id = None

    now = int(time.time() * 1000)
    hours_ago = 240  # Backtest for the last 240 hours (~10 days)
    durationTime = now - (hours_ago * 60 * 60 * 1000)

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
    
        df['ema20'] = df['close'].ewm(span=span_short, adjust=False).mean()  # Short-term EMA
        df['ema50'] = df['close'].ewm(span=span_long, adjust=False).mean()  # Long-term EMA

        for index, row in df.iterrows():
            if pd.isna(row['ema20']) or pd.isna(row['ema50']):
                continue

            current_price = row['close']
            short_ema = row['ema20']
            long_ema = row['ema50']
            price_difference = current_price - short_ema

            # Determine buy and sell signals based on trend and price difference
            buy_signal = (
                price_difference > (short_ema * buffer_percentage) and
                short_ema > long_ema
            )
            sell_signal = (
                price_difference < -(short_ema * buffer_percentage) and
                short_ema < long_ema
            )
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
                        buy_signals.append(index)
                buy_target_index += 1

            else:
                buy_target_index = 0

            if sell_signal:
                buy_target_index = 0
                if sell_target_index >= treshhold:
                    if order_type == OrderType.LONG:
                        order_type = OrderType.NONE
                        # print(f'closed LONG order id = {last_order_id} at {current_time}')
                    if order_type == OrderType.NONE:
                        order_type = OrderType.SHORT
                        print(f'Sell signal at {current_time}')
                        sell_signals.append(index)
                sell_target_index += 1

            else:
                sell_target_index = 0

    except Exception as e:
        print(f"Error: {e}")
    
    # Plotting
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['close'], label='Close Price', color='blue')
    plt.plot(df.index, df['ema20'], label='EMA 20', color='orange')
    plt.plot(df.index, df['ema50'], label='EMA 50', color='green')

    plt.scatter(buy_signals, df.loc[buy_signals]['close'], marker='^', color='blue', lw=3, label='Buy Signal')
    plt.scatter(sell_signals, df.loc[sell_signals]['close'], marker='v', color='red', lw=3, label='Sell Signal')

    plt.title(f'{symbol} Price with Buy/Sell Signals (1 Hour Time Frame)')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

# Call the backtest function to run it
if __name__ == "__main__":
    ema_backtest()