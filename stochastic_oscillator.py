import pandas as pd
import numpy as np
import time
import datetime
import asyncio
from bingx import *
import matplotlib.pyplot as plt


global last_order_id
global order_type

def stochastic_oscillator_chart():
    symbol = "BTC-USDT"
    interval = "1m"
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
    
    # Calculate Stochastic Oscillator
    length = 14
    d_length = 3
    close_prices = df['close']
    lows = df['low']
    highs = df['high']

    lowest_low = lows.rolling(window=length).min()
    highest_high = highs.rolling(window=length).max()
    k = 100 * ((close_prices - lowest_low) / (highest_high - lowest_low))

    d = k.rolling(window=d_length).mean()

    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))

    color = 'tab:red'
    ax1.set_ylabel('Close Price', color=color)
    ax1.plot(df.index, df['close'], color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    color = 'tab:blue'
    ax2.set_ylabel('Stochastic Oscillator', color=color)
    ax2.plot(df.index, k, color=color, label='%K')
    ax2.plot(df.index, d, color='orange', label='%D')
    ax2.axhline(y=70, color='r', linestyle='--', linewidth=0.5)  # Upper trigger line
    ax2.axhline(y=30, color='g', linestyle='--', linewidth=0.5)  # Lower trigger line
    ax2.tick_params(axis='y', labelcolor=color)

    # Adjust layout
    plt.subplots_adjust(hspace=0.4)
    plt.title(f'{symbol} Close Price and Stochastic Oscillator')
    plt.show()

    return df



if __name__ == '__main__':
    stochastic_oscillator_chart()
