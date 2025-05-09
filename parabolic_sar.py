import pandas as pd
import numpy as np
import time
import datetime
import asyncio
from bingx import *
import matplotlib.pyplot as plt

def plot(df,sar,signals):

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
    return

def calculate_sar(df, af_start=0.02, af_increment=0.02, af_maximum=0.2):
    sar = np.full(len(df), np.nan, dtype=float) 
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
    
    
    return sar
def generate_signals(df, sar):
    signals = pd.DataFrame(index=df.index)
    signals['Price'] = df['close']
    signals['SAR'] = sar
    signals['Signal'] = 0
    
    # Define buy and sell conditions
    buy_condition = (signals['Price'] > signals['SAR']) & (signals['Price'].shift(1) <= signals['SAR'].shift(1))
    sell_condition = (signals['Price'] < signals['SAR']) & (signals['Price'].shift(1) >= signals['SAR'].shift(1))
    
    # Apply conditions
    signals.loc[buy_condition, 'Signal'] = 1
    signals.loc[sell_condition, 'Signal'] = -1
    
    return signals['Signal'].iloc[-1]

async def prepare():
    await asyncio.gather(
        startStrategy()
    )

def startStrategy():
    symbol = "BTC-USDT"
    interval = "1m"
    now = int(time.time() * 1000)
    minutes_ago = 100
    durationTime = now - (minutes_ago * 60 * 1000)
   
    amount = 0.016  # in btc = 63675 ==> 1000$
    last_order_id = None
    order_type = OrderType.NONE

    while True:
       try:
            durationTime = now - (minutes_ago * 60 * 1000)
            response = get_kline(symbol, interval, start=durationTime)
            response.raise_for_status()
            response = response.json().get('data', [])
            df = pd.DataFrame(response, columns=[
                'time', 'open', 'high', 'low', 'close', 'volume'])
            df['time'] = pd.to_datetime(df['time'], unit='ms')

            df.set_index('time', inplace=True)
            df['close'] = df['close'].astype(float)
            df['low'] = df['low'].astype(float)
            df['high'] = df['high'].astype(float)

            sar = calculate_sar(df)
            last_signal = generate_signals(df, sar)
            # plot(df=df,sar=sar,signals=signals)
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            
            if last_signal == 1:
                print("Buy")
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
            elif last_signal == -1:
                print("Sell")
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
            else:
                print("No signal")

            time.sleep(60)
       except StopIteration:
            break    
    

if __name__ == '__main__':
    asyncio.run(prepare())