import pandas as pd
import numpy as np
import time
import datetime
import asyncio
from bingx import *
import matplotlib.pyplot as plt



def calculate_sar( af_start=0.02, af_increment=0.02, af_maximum=0.2):
    symbol = "BTC-USDT"
    interval = "1m"
    now = int(time.time() * 1000)
    minutes_ago = 500

    durationTime = now - (minutes_ago * 60 * 1000)
    response = get_kline(symbol, interval, start=durationTime)
    response.raise_for_status()
    response = response.json().get('data', [])
    df = pd.DataFrame(response, columns=[
        'time', 'open', 'high', 'low', 'close', 'volume'])
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    df.set_index('time', inplace=False)

    df['close'] = df['close'].astype(float)
    df['low'] = df['low'].astype(float)
    df['high'] = df['high'].astype(float)
    sar = np.full(len(df), np.nan, dtype=float)  # Initialize sar array with nan values
    trend = 'up'
    af = af_start
    extreme_point = df['high'].iloc[0]
    sar[0] = df['low'].iloc[0]
    
    for i in range(1, len(df)):
        if trend == 'up':
            sar[i] = sar[i - 1] + af * (extreme_point - sar[i - 1])
            if df['low'].iloc[i] < sar[i]:
                trend = 'down'
                sar[i] = extreme_point
                extreme_point = df['low'].iloc[i]
                af = af_start
            else:
                extreme_point = max(extreme_point, df['high'].iloc[i])
                af = min(af + af_increment, af_maximum)
        else:
            sar[i] = sar[i - 1] + af * (extreme_point - sar[i - 1])
            if df['high'].iloc[i] > sar[i]:
                trend = 'up'
                sar[i] = extreme_point
                extreme_point = df['high'].iloc[i]
                af = af_start
            else:
                extreme_point = min(extreme_point, df['low'].iloc[i])
                af = min(af + af_increment, af_maximum)
    
    signals = generate_signals(df=df,sar=sar)
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['close'], label='Close Price', color='blue')
    
    # Plot buy signals
    plt.plot(signals.index[signals['Signal'] == 1], 
             signals['Price'][signals['Signal'] == 1], 
             '^', markersize=10, color='g', lw=0, label='Buy Signal')
    
    # Plot sell signals
    plt.plot(signals.index[signals['Signal'] == -1], 
             signals['Price'][signals['Signal'] == -1], 
             'v', markersize=10, color='r', lw=0, label='Sell Signal')
    
    plt.title('Close Price with Buy/Sell Signals')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show() 
    
    # plt.figure(figsize=(10, 6))
    # plt.plot(df['time'], df['close'], label='Close Price')
    # plt.plot(df['time'], sar, label='Parabolic SAR', color='orange')
    # plt.title('Parabolic SAR Indicator')
    # plt.xlabel('Time')
    # plt.ylabel('Price')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    return sar

def generate_signals(df, sar):
    signals = pd.DataFrame(index=df.index)
    signals['Price'] = df['close']
    signals['SAR'] = sar
    signals['Signal'] = 0
    
    for i in range(1, len(signals)):
        if signals['Price'].iloc[i] > signals['SAR'].iloc[i] and signals['Price'].iloc[i - 1] <= signals['SAR'].iloc[i - 1]:
            signals['Signal'].iloc[i] = 1  # Buy signal
        elif signals['Price'].iloc[i] < signals['SAR'].iloc[i] and signals['Price'].iloc[i - 1] >= signals['SAR'].iloc[i - 1]:
            signals['Signal'].iloc[i] = -1  # Sell signal
    
    return signals

if __name__ == '__main__':
    calculate_sar()
