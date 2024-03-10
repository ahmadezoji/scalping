import pandas as pd
import numpy as np
import time
import datetime
import asyncio
from bingx import *
import matplotlib.pyplot as plt


def ema_for_current_price(current_price, previous_ema, span):
    smoothing_factor = 2 / (span + 1)
    ema = (current_price - previous_ema) * smoothing_factor + previous_ema
    return ema


def scalping_strategy(df):
    # Calculate EMA(9)
    df['EMA_9'] = df['close'].ewm(span=9, min_periods=9, adjust=False).mean()

    # Calculate RSI
    df['close'] = df['close'].astype(float)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    RS = gain / loss
    df['RSI'] = 100 - (100 / (1 + RS))

    # Generate signals
    # & (df['RSI'] < 31)
    # & (df['RSI'] > 69)
    print(df['close'])
    print(df['EMA_9'])
    buy_signal = (df['close'] >= df['EMA_9'])
    sell_signal = (df['close'] <= df['EMA_9'])

    print(buy_signal)

    # # Plot close and EMA
    # plt.figure(figsize=(10, 5))
    # plt.plot(df.index, df['close'], label='Close Price', color='blue')
    # plt.plot(df.index, df['EMA_9'], label='EMA(9)', color='red')
    # plt.xlabel('Time')
    # plt.ylabel('Price')
    # plt.title('Close Price vs EMA(9)')
    # plt.legend()
    # plt.show()
    return buy_signal, sell_signal


global last_order_id
global order_type


def ema_strategy():
    symbol = "BTC-USDT"
    interval = "1m"
    amount = 0.0015  # in btc = 68380 ==> 100$
    span = 9
    previous_ema = None
    buy_target_index = 0
    sell_target_index = 0
    last_order_id = None
    order_type = OrderType.NONE

    now = int(time.time() * 1000)
    minutes_ago = 100

    timestamp_120_min_ago = now - (minutes_ago * 60 * 1000)

    def get_one(df):
        for index, row in df.iterrows():
            yield index, row['close']

    response = get_kline(symbol, interval, start=timestamp_120_min_ago)
    response.raise_for_status()
    response = response.json().get('data', [])
    df = pd.DataFrame(response, columns=[
        'time', 'open', 'high', 'low', 'close', 'volume'])
    generator = get_one(df)

    while True:
        try:
            index, current_price = next(generator)
            current_price = float(current_price)

            if previous_ema is not None:
                ema9 = ema_for_current_price(
                    current_price=current_price, previous_ema=previous_ema, span=span)

                buy_signal = current_price > ema9
                sell_signal = current_price < ema9
                current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                if buy_signal:
                    print("buy")
                if sell_signal:
                    print("sell")

                if buy_signal:
                    sell_target_index = 0
                    if (buy_target_index >= 3):
                        if (order_type == OrderType.SHORT):
                            # close_order(symbol=symbol,order_id=last_order_id)
                            order_type = OrderType.NONE
                            print(f'closed SHORT order id = {last_order_id} ')
                        if order_type == OrderType.NONE:
                            order_type = OrderType.LONG
                            # last_order_id, order_type = open_long(
                            #     symbol=symbol, quantity=amount)
                            print(f'Buy signal at {current_time}')
                    # if (buy_target_index == 2):
                    #     if (order_type == OrderType.SHORT):
                    #         # close_order(symbol=symbol,order_id=last_order_id)
                    #         print(f'closed SHORT order id = {last_order_id} ')
                    buy_target_index = buy_target_index + 1

                if sell_signal:
                    buy_target_index = 0
                    if (sell_target_index >= 3):
                        if (order_type == OrderType.LONG):
                            # close_order(symbol=symbol,order_id=last_order_id)
                            order_type = OrderType.NONE
                            print(f'closed LONG order id = {last_order_id} ')
                        if order_type == OrderType.NONE:
                            # last_order_id, order_type = open_short(
                            #     symbol=symbol, quantity=amount)
                            order_type = OrderType.SHORT
                            print(f'Sell signal at {current_time}')
                    # if (sell_target_index == 2):
                        # if (order_type == OrderType.LONG):
                        #     # close_order(symbol=symbol,order_id=last_order_id)
                        #     print(f'closed LONG order id = {last_order_id} ')
                    sell_target_index = sell_target_index + 1

            previous_ema = current_price

            time.sleep(1)

        except StopIteration:
            break


def plot_close_data_with_signals(close_data, buy_indices, sell_indices):
    """
    Plots 'close' data on a chart and highlights the points where buy and sell signals occur.

    Parameters:
        close_data (list or array-like): List or array of 'close' data.
        buy_indices (list or array-like): List or array of indices where buy signals occur.
        sell_indices (list or array-like): List or array of indices where sell signals occur.
    """
    # Plot 'close' data
    plt.plot(close_data, label='Close')

    # # Highlight buy signals
    # for idx in buy_indices:
    #     plt.scatter(idx, close_data[idx], color='green',
    #                 marker='^', label='Buy Signal')

    # # Highlight sell signals
    # for idx in sell_indices:
    #     plt.scatter(idx, close_data[idx], color='red',
    #                 marker='v', label='Sell Signal')

    # Add labels and legend
    plt.title('Close Data with Buy and Sell Signals')
    plt.xlabel('Index')
    plt.ylabel('Close Price')
    plt.legend()
    plt.grid(True)

    # Show plot
    plt.show()


async def main():

    # await asyncio.gather(
    ema_strategy()
    # )


if __name__ == '__main__':
    previous_ema = 48.0
    asyncio.run(main())
