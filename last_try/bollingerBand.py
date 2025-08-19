import pandas as pd
import numpy as np
import time
import datetime
import asyncio
from bingx import *
import matplotlib.pyplot as plt

# Function to calculate Bollinger Bands
def calculate_bollinger_bands(df, window_size, num_std_dev):
    df['MA'] = df['close'].rolling(window=window_size).mean()
    df['BB_upper'] = df['MA'] + (df['close'].rolling(window=window_size).std() * num_std_dev)
   
    df['BB_lower'] = df['MA'] - (df['close'].rolling(window=window_size).std() * num_std_dev)
    return df

symbol = "BTC-USDT"
interval = "5m"
span = 15
now = int(time.time() * 1000)
minutes_ago = 200

durationTime = now - (minutes_ago * 60 * 1000)
response = get_kline(symbol, interval, start=durationTime)
response.raise_for_status()
response = response.json().get('data', [])
df = pd.DataFrame(response, columns=[
    'time', 'open', 'high', 'low', 'close', 'volume'])
df['time'] = pd.to_datetime(df['time'], unit='ms')



df.set_index('time', inplace=True)

# Parameters
window_size = 20  # Window size for moving average
num_std_dev = 2   # Number of standard deviations for Bollinger Bands

# Calculate Bollinger Bands
df = calculate_bollinger_bands(df, window_size, num_std_dev)


# # Plotting
# plt.figure(figsize=(15, 7))
# plt.plot(df.index, df['close'], label='Close Price', color='blue')
# plt.plot(df.index, df['MA'], label='Moving Average', color='red')
# plt.plot(df.index, df['BB_upper'], label='Upper Bollinger Band', color='green')
# plt.plot(df.index, df['BB_lower'], label='Lower Bollinger Band', color='orange')
# plt.title('Bollinger Bands Scalping Strategy (5-min Timeframe)')
# plt.xlabel('Time')
# plt.ylabel('Price')
# plt.legend()
# plt.grid(True)
# plt.show()

# Scalping Strategy
def scalping_strategy(df):
    signals = []
    position = None
    for i in range(1, len(df)):
        if df['close'][i] > df['BB_upper'][i]:
            if position != 'SELL':
                signals.append('SELL')
                position = 'SELL'
            else:
                signals.append('HOLD')
        elif df['close'][i] < df['BB_lower'][i]:
            if position != 'BUY':
                signals.append('BUY')
                position = 'BUY'
            else:
                signals.append('HOLD')
        else:
            signals.append('HOLD')
    
    df['Signal'] = signals
    return df

# Apply scalping strategy
df = scalping_strategy(df)

# Print the last few rows with signals
print(df.tail(10))
