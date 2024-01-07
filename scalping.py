import asyncio
import time
from bingx import *
import matplotlib.pyplot as plt
import numpy as np

quantity = 0.00012
current_order_id = None
order_status_open = False


async def rsi_strategy(symbol, interval):
    global current_order_id
    global order_status_open
    while True:
        try:
            rsi =  get_and_calculate_rsi(symbol, interval)
            # print(f'Current RSI for {interval} interval: {rsi}')

            # if not order_status_open and rsi < 30:
            #     print('RSI is below 30. Initiating buy order...')
            #     response = open_long(symbol=symbol, quantity=quantity)
            #     print(f'Buy order executed: {response.text}')
            #     response.raise_for_status()
            #     data = response.json().get('data', [])
            #     order_info = data.get('order', {})
            #     current_order_id = order_info.get('orderId', None)
            #     if current_order_id is not None:
            #         order_status_open = True

            # elif order_status_open and rsi > 70:
            #     print('RSI is above 70. Initiating sell order...')
            #     close_long(symbol=symbol, quantity=quantity)
            #     print(f'Sell order executed: {response.text}')
            #     order_status_open = False

            await asyncio.sleep(5)

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
        response =  get_kline(symbol, interval)
        response.raise_for_status()
        response = response.json().get('data', [])
        df = pd.DataFrame(response, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['time'], unit='ms')
        df.set_index('time', inplace=True)

        signals = pd.DataFrame(index=df.index)
        signals['signal'] = 0.0

        signals['short_mavg'] = df['close'].rolling(window=short_window, min_periods=1, center=False).mean()
        signals['long_mavg'] = df['close'].rolling(window=long_window, min_periods=1, center=False).mean()

        min_required_data_points = max(short_window, long_window)

        if len(df) >= min_required_data_points:
            signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:] > signals['long_mavg'][short_window:], 1.0, 0.0)
            signals['positions'] = signals['signal'].diff()
        # signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:] > signals['long_mavg'][short_window:], 1.0, 0.0)
        # for i in range(short_window, len(signals)):
            # signals['signal'][i] = 1.0 if signals['short_mavg'][i] > signals['long_mavg'][i] else 0.0
        signals['positions'] = signals['signal'].diff()

        print(signals.tail(1))  # Display the latest signal
        # print(f"short_mavg: {signals['short_mavg'][0]}, long_mavg: {signals['long_mavg'][0]}, signal: {signals['signal'][0]}")

        await asyncio.sleep(1)  # Sleep for 1 second before fetching the next data

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
