import pandas as pd
import numpy as np
from binance.client import Client
from index import get_klines_all
import logging
import json
from datetime import datetime, timedelta

logging.basicConfig(
    filename='scalping_bot.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)


def backtest_vwap_scalping(data, atr_period=14, stochastic_period=14, rsi_period=14, capital=100, lot_size=0.01):
    # Add RSI
    delta = data['close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(rsi_period).mean()
    avg_loss = loss.rolling(rsi_period).mean()
    rs = avg_gain / avg_loss
    data['rsi'] = 100 - (100 / (1 + rs))

    # Add EMA for trend
    data['ema_trend'] = data['close'].rolling(15).mean()

    # Calculate VWAP
    data['vwap'] = (data['close'] * data['volume']
                    ).cumsum() / data['volume'].cumsum()

    # Calculate ATR
    high_low = data['high'] - data['low']
    high_close = abs(data['high'] - data['close'].shift(1))
    low_close = abs(data['low'] - data['close'].shift(1))
    data['tr'] = pd.concat(
        [high_low, high_close, low_close], axis=1).max(axis=1)
    data['atr'] = data['tr'].rolling(atr_period).mean()

    # Calculate Stochastic Oscillator
    data['stoch_k'] = (
        (data['close'] - data['low'].rolling(stochastic_period).min()) /
        (data['high'].rolling(stochastic_period).max() -
         data['low'].rolling(stochastic_period).min())
    ) * 100
    data['stoch_d'] = data['stoch_k'].rolling(3).mean()

    # Initialize backtest variables
    position = None  # 'long' or 'short'
    entry_price = 0
    unrealized_pnl = []
    realized_pnl = 0
    pnl_history = []
    trades = 0

    # Backtest logic
    for i in range(len(data)):
        if i < max(atr_period, stochastic_period):  # Skip initial rows
            unrealized_pnl.append(0)
            continue

        row = data.iloc[i]

        # Entry logic
        if position is None:
            if row['close'] > row['vwap'] and row['stoch_k'] < 50 and row['rsi'] > 45 and row['close'] > row['ema_trend']:
                # Open long
                position = 'long'
                entry_price = row['close']
                trades += 1

            elif row['close'] < row['vwap'] and row['stoch_k'] > 50 and row['rsi'] < 55 and row['close'] < row['ema_trend']:
                # Open short
                position = 'short'
                entry_price = row['close']
                trades += 1

        # Adjust Stop-Loss and Take-Profit
        take_profit = entry_price * (1.01 if position == 'long' else 0.99)
        stop_loss = entry_price * (0.995 if position == 'long' else 1.005)

        # Exit logic
        if position == 'long' and (row['close'] >= take_profit or row['close'] <= stop_loss):
            realized_pnl += (row['close'] - entry_price) * lot_size
            position = None
        elif position == 'short' and (row['close'] <= take_profit or row['close'] >= stop_loss):
            realized_pnl += (entry_price - row['close']) * lot_size
            position = None

        # Unrealized PnL
        if position == 'long':
            unrealized_pnl.append((row['close'] - entry_price) * lot_size)
        elif position == 'short':
            unrealized_pnl.append((entry_price - row['close']) * lot_size)
        else:
            unrealized_pnl.append(0)

        # Record PnL
        pnl_history.append(realized_pnl + unrealized_pnl[-1])

    # Results
    final_balance = capital + realized_pnl
    profit_percentage = ((final_balance - capital) / capital) * 100
    results = {
        'initial_capital': capital,
        'final_balance': final_balance,
        'realized_pnl': realized_pnl,
        'profit_percentage': profit_percentage,
        # 'unrealized_pnl': unrealized_pnl,
        # 'pnl_history': pnl_history,
        'total_trades': trades
    }
    return results


if __name__ == "__main__":
    symbol = 'ETHUSDT'
    interval = '1m'
    _interval = 1
    day = 24
    limit = int((day * 60) / _interval)  # steps calculated with minutes in a day


    # # historical_data = get_klines_all(symbol, interval, limit)
    # start_time = '2023-12-21 00:00:00'  # Specify start datetime
    # end_time = '2023-12-23 00:00:00'    # Specify end datetime

    # historical_data = get_klines_all(symbol, interval, start_time=start_time, end_time=end_time, limit=500)


    # result = backtest_vwap_scalping(historical_data, capital=2000, lot_size=1)
    # print("Final Results:")
    # print(result)

    # Run the backtest day by day for the last month
    end_date = datetime.now() 
    start_date = end_date - timedelta(days=30)
    current_date = start_date

    results = {}

    while current_date < end_date:
        next_date = current_date + timedelta(days=1)
        historical_data = get_klines_all(
            symbol=symbol,
            interval=interval,
            start_time=current_date.strftime('%Y-%m-%d %H:%M:%S'),
            end_time=next_date.strftime('%Y-%m-%d %H:%M:%S'),
            limit= limit
        )

        result = backtest_vwap_scalping(historical_data, capital=100, lot_size=1)
        results[current_date.strftime('%Y-%m-%d')] = result
        current_date = next_date
    

    print(json.dumps(results, indent=4))

    # Calculate the win rate from the results.
    win_count = 0
    total_trades = 0

    for day, result in results.items():
        if result["realized_pnl"] > 0:  # Count as a win if realized PnL is positive
            win_count += 1
        total_trades += result["total_trades"]

    win_rate = (win_count / len(results)) * 100  # Percentage of winning days
    average_trades_per_day = total_trades / len(results) if len(results) > 0 else 0

    # Displaying win rate and average trades per day
    win_rate, average_trades_per_day
    print(f"win rate : {win_rate} % and avg trade : {average_trades_per_day} ")
