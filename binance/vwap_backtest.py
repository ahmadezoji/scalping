import pandas as pd
import numpy as np
import logging
from index import get_klines_all
from datetime import datetime, timedelta
from itertools import product

logging.basicConfig(
    filename='vwap_backtest.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

def log_and_print(message):
    print(message)
    logging.info(message)

def calculate_vwap(data, atr_period=14, stochastic_period=14, rsi_period=14):
    """ Perform VWAP strategy calculations. """
    delta = data['close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(rsi_period).mean()
    avg_loss = loss.rolling(rsi_period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    data['rsi'] = 100 - (100 / (1 + rs.fillna(0)))
    data['ema_trend'] = data['close'].rolling(15).mean()
    data['vwap'] = (data['close'] * data['volume']).cumsum() / data['volume'].cumsum()
    high_low = data['high'] - data['low']
    high_close = abs(data['high'] - data['close'].shift(1))
    low_close = abs(data['low'] - data['close'].shift(1))
    data['tr'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    data['atr'] = data['tr'].rolling(atr_period).mean()
    data['stoch_k'] = (
        (data['close'] - data['low'].rolling(stochastic_period).min()) /
        (data['high'].rolling(stochastic_period).max() - data['low'].rolling(stochastic_period).min())
    ) * 100
    data['stoch_d'] = data['stoch_k'].rolling(3).mean()
    return data

def backtest_vwap_strategy(symbol, start_date, end_date):
    """ Backtest VWAP strategy over historical data with simulated TP/SL. """
    total_pnl = 0
    daily_pnl = {}
    position = None
    entry_price = 0
    entry_quantity = 0
    capital = 500  # Initial capital in USDT
    position_size = 0.2  # Risking 20% per trade
    tp_percentage = 0.30  # Take Profit at 0.5%
    sl_percentage = 0.10  # Stop Loss at 0.3%

    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    for date in date_range:
        log_and_print(f"Backtesting {symbol} on {date.date()}")
        data = get_klines_all(symbol, '5m', start_time=date, end_time=date + timedelta(days=1))
        if data.empty:
            log_and_print(f"No data available for {date.date()}")
            continue

        data = calculate_vwap(data)
        daily_pnl[date.date()] = 0
        for i in range(1, len(data)):
            close_price = data['close'].iloc[i]
            vwap = data['vwap'].iloc[i]
            stoch_k = data['stoch_k'].iloc[i]
            rsi = data['rsi'].iloc[i]
            ema_trend = data['ema_trend'].iloc[i]
            slippage = 0.0002  # 0.02% slippage

            if position is None:
                if close_price > vwap * 0.995 and stoch_k < 70 and rsi > 50 and close_price > ema_trend:
                    position = 'LONG'
                    entry_price = close_price
                    entry_quantity = (capital * position_size) / entry_price
                    log_and_print(f"Opening LONG at {entry_price} with {entry_quantity} quantity")
                elif close_price < vwap * 1.005 and stoch_k > 30 and rsi < 50 and close_price < ema_trend:
                    position = 'SHORT'
                    entry_price = close_price
                    entry_quantity = (capital * position_size) / entry_price
                    log_and_print(f"Opening SHORT at {entry_price} with {entry_quantity} quantity")
            else:
                close_price = close_price * (1 - slippage) if position == 'LONG' else close_price * (1 + slippage)
                pnl = (close_price - entry_price) * entry_quantity if position == 'LONG' else (entry_price - close_price) * entry_quantity
                log_and_print(f"Current PnL: {pnl:.2f} USDT")
                
                # Simulated TP/SL Process
                if pnl >= tp_percentage * capital * position_size:
                    log_and_print(f"Closing {position} at {close_price} due to Take Profit with PnL: {pnl:.2f} USDT")
                    total_pnl += pnl
                    daily_pnl[date.date()] += pnl
                    position = None
                elif pnl <= -sl_percentage * capital * position_size:
                    log_and_print(f"Closing {position} at {close_price} due to Stop Loss with PnL: {pnl:.2f} USDT")
                    total_pnl += pnl
                    daily_pnl[date.date()] += pnl
                    position = None

    # Ensure last position is closed at the last available price
    if position is not None:
        final_price = data['close'].iloc[-1]
        final_pnl = (final_price - entry_price) * entry_quantity if position == 'LONG' else (entry_price - final_price) * entry_quantity
        log_and_print(f"Closing Final {position} at {final_price} with PnL: {final_pnl:.2f} USDT")
        total_pnl += final_pnl
        daily_pnl[date_range[-1].date()] += final_pnl
    
    log_and_print(f"Daily PnL Summary: {daily_pnl}")
    log_and_print(f"Total PnL over backtest period: {total_pnl:.2f} USDT")

    
def optimize_tp_sl(symbol, start_date, end_date):
    """ Optimize TP and SL values to find the most profitable combination. """
    best_pnl = float('-inf')
    best_tp, best_sl = 0, 0
    
    tp_values = np.arange(0.3, 2.1, 0.1)  # 0.3% to 2% TP range
    sl_values = np.arange(0.1, 1.1, 0.1)  # 0.1% to 1% SL range
    
    for tp, sl in product(tp_values, sl_values):
        pnl = backtest_vwap_strategy(symbol, start_date, end_date, tp, sl)
        if pnl > best_pnl:
            best_pnl, best_tp, best_sl = pnl, tp, sl
    
    log_and_print(f"Best TP: {best_tp:.2f}%, Best SL: {best_sl:.2f}%, Max PnL: {best_pnl:.2f} USDT")
    return best_tp, best_sl

if __name__ == "__main__":
    symbol = 'DOGEUSDT'
    start_date = '2025-01-01'
    end_date = '2025-01-30'
    backtest_vwap_strategy(symbol, start_date, end_date)
