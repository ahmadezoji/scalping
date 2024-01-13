import asyncio
import time
from bingx import *
# import matplotlib.pyplot as plt
import numpy as np
import datetime


quantity = 0.00012
current_order_id = None
order_status_open = False

buy = 0
sell = 0
async def rsi_strategy(symbol, interval):
    global current_order_id
    global order_status_open
    global buy
    global sell
    while True:
        try:
            # Calculate RSI
            rsi = get_and_calculate_rsi(symbol, interval)
            overbought_threshold = 70
            oversold_threshold = 30

            # Determine buy and sell signals
            buy_signal = np.where(rsi < oversold_threshold, 1, 0)
            sell_signal = np.where(rsi > overbought_threshold, 1, 0)
           
            # Process signals
            for i in range(len(buy_signal)):
                if buy_signal[i] == 1:
                    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    if buy >= 5:
                        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        print(f"Buy signal at {current_time} for {symbol} ({interval})")
                        buy = 0
                    buy = buy+1

                    # Implement your buy logic here

                elif sell_signal[i] == 1:
                    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    if sell >= 5:
                        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        print(f"Sell signal at {current_time} for {symbol} ({interval})")
                        sell = 0
                    sell = sell+1

            await asyncio.sleep(60)

        except Exception as e:
            print(f'An error occurred: {e}')


def supertrend_strategy():
    start = int(dt.datetime(2023, 1, 1).timestamp())
    interval = "4h"
    symbol = 'BTC-USDT'
    df = pd.DataFrame()

    while True:
        response = get_kline(
            symbol=symbol,
            start=start,
            interval=interval)
        response.raise_for_status()
        klines_data = response.json().get('data', [])
        latest = format_data(klines_data)

        if not isinstance(latest, pd.DataFrame):
            break

        start = get_last_timestamp(latest)
        time.sleep(0.01)

        # df = pd.concat([df, latest])
        # print(f'Collecting data starting {
        #       dt.datetime.fromtimestamp(start/1000)}')

        # if len(latest) == 1:
        #     break


async def moving_average_crossover_strategy(symbol, interval, short_window, long_window):
    while True:
        response = get_kline(symbol, interval)
        response.raise_for_status()
        response = response.json().get('data', [])
        df = pd.DataFrame(response, columns=[
                          'time', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['time'], unit='ms')
        df.set_index('time', inplace=True)

        signals = pd.DataFrame(index=df.index)
        signals['signal'] = 0.0

        signals['short_mavg'] = df['close'].rolling(
            window=short_window, min_periods=1, center=False).mean()
        signals['long_mavg'] = df['close'].rolling(
            window=long_window, min_periods=1, center=False).mean()

        min_required_data_points = max(short_window, long_window)

        if len(df) >= min_required_data_points:
            signals['signal'][short_window:] = np.where(
                signals['short_mavg'][short_window:] > signals['long_mavg'][short_window:], 1.0, 0.0)
            signals['positions'] = signals['signal'].diff()
        # signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:] > signals['long_mavg'][short_window:], 1.0, 0.0)
        # for i in range(short_window, len(signals)):
            # signals['signal'][i] = 1.0 if signals['short_mavg'][i] > signals['long_mavg'][i] else 0.0
        signals['positions'] = signals['signal'].diff()

        print(signals.tail(1))  # Display the latest signal
        # print(f"short_mavg: {signals['short_mavg'][0]}, long_mavg: {signals['long_mavg'][0]}, signal: {signals['signal'][0]}")

        # Sleep for 1 second before fetching the next data
        await asyncio.sleep(1)


async def main():
    symbol = "BTC-USDT"
    interval = "1m"
    await asyncio.gather(
        rsi_strategy(symbol, interval),
        # Add more strategies here if needed
    )
    # symbol = "BTC-USDT"
    # interval = "1m"
    # short_window = 40
    # long_window = 100

    # await asyncio.gather(
    #     moving_average_crossover_strategy(symbol, interval, short_window, long_window)
    #     # Add more strategies here if needed
    # )

if __name__ == '__main__':
    asyncio.run(main())
    # supertrend_strategy()
