import pandas as pd
import numpy as np
import time
import datetime
import asyncio
from bingx import *
import matplotlib.pyplot as plt
import talib as ta

global last_order_id
global order_type



def rsi_macd_backtest():
    symbol = "BTC-USDT"
    interval = "1h"
    amount = 0.014  # in btc = 72068 ==> 1000$
    rsi_period = 14
    rsi_upper = 60  # Lower RSI thresholds to capture smaller movements
    rsi_lower = 40
    macd_fast = 6  # Shorter MACD periods for quicker signals
    macd_slow = 13
    macd_signal = 5
    sma_short = 5  # Short-term moving average
    sma_long = 10  # Slightly longer moving average
    order_type = OrderType.NONE
    buy_signals = []
    sell_signals = []

    now = int(time.time() * 1000)
    hours_ago = 240  # Backtest for the last 240 hours (~10 days)
    durationTime = now - (hours_ago * 60 * 60 * 1000)

    try:
        response = get_kline(symbol, interval, start=durationTime)
        response.raise_for_status()
        response = response.json().get('data', [])
        df = pd.DataFrame(response, columns=[
            'time', 'open', 'high', 'low', 'close', 'volume'])
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        df.set_index('time', inplace=True)

        df['close'] = df['close'].astype(float)
        
        # Calculate RSI
        df['rsi'] = ta.RSI(df['close'], timeperiod=rsi_period)
        
        # Calculate MACD
        df['macd'], df['macd_signal'], _ = ta.MACD(df['close'], 
                                                    fastperiod=macd_fast, 
                                                    slowperiod=macd_slow, 
                                                    signalperiod=macd_signal)
        
        # Calculate SMAs
        df['sma_short'] = ta.SMA(df['close'], timeperiod=sma_short)
        df['sma_long'] = ta.SMA(df['close'], timeperiod=sma_long)

        for index, row in df.iterrows():
            if pd.isna(row['rsi']) or pd.isna(row['macd']) or pd.isna(row['macd_signal']):
                continue

            current_price = row['close']
            rsi = row['rsi']
            macd = row['macd']
            macd_signal = row['macd_signal']
            sma_short_val = row['sma_short']
            sma_long_val = row['sma_long']
            current_time = index.strftime("%Y-%m-%d %H:%M:%S")

            # Buy Signal: RSI < 40 and MACD crosses above the signal line and short SMA > long SMA
            if rsi < rsi_lower and macd > macd_signal and sma_short_val > sma_long_val:
                if order_type != OrderType.LONG:
                    print(f'Buy signal at {current_time}')
                    buy_signals.append(index)
                    order_type = OrderType.LONG

            # Sell Signal: RSI > 60 and MACD crosses below the signal line and short SMA < long SMA
            elif rsi > rsi_upper and macd < macd_signal and sma_short_val < sma_long_val:
                if order_type != OrderType.SHORT:
                    print(f'Sell signal at {current_time}')
                    sell_signals.append(index)
                    order_type = OrderType.SHORT

    except Exception as e:
        print(f"Error: {e}")

    # Plotting
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['close'], label='Close Price', color='blue')

    plt.scatter(buy_signals, df.loc[buy_signals]['close'], marker='^', color='blue', lw=3, label='Buy Signal')
    plt.scatter(sell_signals, df.loc[sell_signals]['close'], marker='v', color='red', lw=3, label='Sell Signal')

    plt.title(f'{symbol} Price with RSI and MACD Buy/Sell Signals (1 Hour Time Frame)')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()



# Call the backtest function to run it
if __name__ == "__main__":
    rsi_macd_backtest()