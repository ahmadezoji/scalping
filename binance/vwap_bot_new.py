import asyncio
import logging
from index import *
from telegram import send_telegram_message
from datetime import datetime
import pandas as pd
import numpy as np
import time
from index import SYMBOL, tp_percentage, sl_percentage, entry_usdt, trade_interval, sleep_time, tp_sl_check_interval


logging.basicConfig(
    filename='vwap_bot.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

def log_and_print(message):
    print(message)
    logging.info(message)

# Global variables to store active position details
current_position = None
entry_price = 0
entry_quantity = 0

 


def calculate_vwap(data, atr_period=14, stochastic_period=14, rsi_period=14):
    """
    Perform VWAP strategy calculations.
    """
    delta = data['close'].diff()

    # Gain and Loss
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    # Calculate Initial Average Gain/Loss using SMA
    avg_gain = pd.Series(gain).rolling(window=rsi_period, min_periods=rsi_period).mean()
    avg_loss = pd.Series(loss).rolling(window=rsi_period, min_periods=rsi_period).mean()

    # Apply Wilder's Smoothing (Cumulative Moving Average)
    avg_gain = avg_gain.shift().fillna(0) * (rsi_period - 1) / rsi_period + gain / rsi_period
    avg_loss = avg_loss.shift().fillna(0) * (rsi_period - 1) / rsi_period + loss / rsi_period

    rs = avg_gain / avg_loss
    data['rsi'] = 100 - (100 / (1 + rs))

    # data['ema_trend'] = data['close'].rolling(15).mean()
    data['ema_trend'] = data['close'].ewm(span=15, adjust=False).mean()
    data['vwap'] = (data['close'] * data['volume']).cumsum() / data['volume'].cumsum()
    high_low = data['high'] - data['low']
    high_close = abs(data['high'] - data['close'].shift(1))
    low_close = abs(data['low'] - data['close'].shift(1))
    data['tr'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    data['atr'] = data['tr'].rolling(atr_period).mean()
    data['stoch_k'] = (
    (data['close'] - data['low'].rolling(stochastic_period).min()) /
    (data['high'].rolling(stochastic_period).max() - data['low'].rolling(stochastic_period).min() + 1e-6)
    ) * 100
    data['stoch_d'] = data['stoch_k'].rolling(3).mean()
    return data

async def trade_logic():
    global current_position, entry_price, entry_quantity
    symbol = SYMBOL
    interval = trade_interval
    interval_value = sleep_time
    current_entry_usdt = entry_usdt  # Track the current available USDT for trading

    while True:
        try:
            data = get_klines_all(symbol, interval, limit=20)
            if data.empty:
                log_and_print("No data retrieved from Binance API.")
                await asyncio.sleep(interval_value * 60)
                continue

            data = calculate_vwap(data)
            available_balance = get_futures_account_balance('USDT')
            
            # Update current_entry_usdt based on available balance
            if current_position is None:
                current_entry_usdt = min(entry_usdt, available_balance)
                log_and_print(f"Updated entry USDT to: {current_entry_usdt}")

            signal = None
            if data['close'].iloc[-1] > data['vwap'].iloc[-1] * 0.999 and data['stoch_k'].iloc[-1] < 70 and data['rsi'].iloc[-1] > 50 and data['close'].iloc[-1] > data['ema_trend'].iloc[-1]:
                signal = 'LONG'
            elif data['close'].iloc[-1] < data['vwap'].iloc[-1] * 1.001 and data['stoch_k'].iloc[-1] > 30 and data['rsi'].iloc[-1] < 50 and data['close'].iloc[-1] < data['ema_trend'].iloc[-1]:
                signal = 'SHORT'

            log_and_print(f"Signal Check: close={data['close'].iloc[-1]}, vwap={data['vwap'].iloc[-1]}, stoch_k={data['stoch_k'].iloc[-1]}, rsi={data['rsi'].iloc[-1]}, ema_trend={data['ema_trend'].iloc[-1]}")

            if signal:
                if current_position:
                    if current_position == signal:
                        await asyncio.sleep(interval_value * 60)
                        continue
                    if current_position != signal:
                        # Close existing position
                        latest_price = data['close'].iloc[-1]
                        pnl_percentage = ((latest_price - entry_price) / entry_price) * 100 if current_position == 'LONG' else ((entry_price - latest_price) / entry_price) * 100
                        pnl_usdt = (pnl_percentage / 100) * (entry_price * entry_quantity)
                        
                        # Update available balance after closing position
                        current_entry_usdt = available_balance
                        
                        log_and_print(f"Closing {current_position} position - PnL: {pnl_percentage:.2f}% ({pnl_usdt:.2f} USDT)")
                        send_telegram_message(
                            f"🔄 Position Closed\n🔹 Previous Position: {current_position}\n💰 Exit Price: {latest_price}\n📈 PnL: {pnl_percentage:.2f}% ({pnl_usdt:.2f} USDT)\n💵 Updated Balance: {current_entry_usdt:.2f} USDT"
                        )
                        current_position = None

                # Place new order with updated balance
                log_and_print(f"Signal Generated: {signal}")
                current_price = data['close'].iloc[-1]

                if available_balance < 20:
                    log_and_print("Insufficient balance. Stopping bot.")
                    send_telegram_message("Insufficient balance. Stopping bot.")
                    break

                leverage_info = client.futures_leverage_bracket(symbol=symbol)[0]
                leverage = leverage_info['brackets'][0]['initialLeverage']

                max_quantity = calculate_max_quantity(current_entry_usdt, leverage, current_price)
                quantity = min(current_entry_usdt / current_price, max_quantity)
                log_and_print(f"Adjusted Quantity: {quantity} (Using {current_entry_usdt:.2f} USDT)")

                order_side = 'BUY' if signal == 'LONG' else 'SELL'
                # order = place_order(symbol, order_side, quantity)
                order = order_side
                message = (
                    f"🚀 <b>New Order Placed</b> 🚀\n"
                    f"📈 <b>Symbol:</b> {symbol}\n"
                    f"🔁 <b>Action:</b> {order_side}\n"
                    f"💵 <b>Quantity:</b> {quantity}\n"
                    f"💰 <b>Entry USDT:</b> {current_entry_usdt:.2f}\n"
                    f"📊 <b>Available Balance:</b> {available_balance:.2f}"
                )
                send_telegram_message(message)

                if order:
                    current_position = signal
                    entry_price = current_price
                    entry_quantity = quantity

        except Exception as e:
            log_and_print(f"Error in trade logic: {e}")

        await asyncio.sleep(interval_value * 60)


def calculate_max_quantity(available_balance, leverage, current_price):
    max_quantity = (available_balance * leverage) / current_price
    return max_quantity
async def tp_sl_monitor():
    global current_position, entry_price, entry_quantity
    symbol = SYMBOL
    interval = tp_sl_check_interval

    while True:
        try:
            if current_position is not None:
                data = get_klines_all(symbol, '5m', limit=1)
                if data.empty:
                    await asyncio.sleep(interval)
                    continue
                available_balance = get_futures_account_balance('USDT')
                latest_price = data['close'].iloc[-1]
                pnl_percentage = ((latest_price - entry_price) / entry_price) * 100 if current_position == 'LONG' else ((entry_price - latest_price) / entry_price) * 100
                pnl_usdt = (pnl_percentage / 100) * (entry_price * entry_quantity)

                log_and_print(f"TP/SL Check - Latest Price: {latest_price}, Entry Price: {entry_price}, PnL Percentage: {pnl_percentage:.2f}%, PnL USDT: {pnl_usdt:.2f}, Position: {current_position}")

                if (current_position == 'LONG' and pnl_percentage >= tp_percentage) or (current_position == 'SHORT' and pnl_percentage >= tp_percentage):
                    log_and_print("Take Profit triggered.")
                    # close_futures_position(symbol, current_position, entry_quantity)
                    send_telegram_message(f"🚀 Take Profit hit!\n🔹 Symbol: {symbol}\n💰 Close Price: {latest_price}\n📈 PnL: {pnl_percentage:.2f}% ({pnl_usdt:.2f} USDT)\n Balance:{available_balance} ")
                    current_position = None
                elif (current_position == 'LONG' and pnl_percentage <= -sl_percentage) or (current_position == 'SHORT' and pnl_percentage <= -sl_percentage):
                    log_and_print("Stop Loss triggered.")
                    # close_futures_position(symbol, current_position, entry_quantity)
                    send_telegram_message(f"⚠️ Stop Loss hit!\n🔹 Symbol: {symbol}\n💰 Close Price: {latest_price}\n📉 PnL: {pnl_percentage:.2f}% ({pnl_usdt:.2f} USDT) \n Balance:{available_balance} ")
                    current_position = None

        except Exception as e:
            log_and_print(f"Error in TP/SL monitor: {e}")

        await asyncio.sleep(interval)
async def main():
    log_and_print("Starting VWAP bot with optimized TP/SL.")
    send_telegram_message("VWAP bot started with optimized TP/SL.")
    await asyncio.gather(
        trade_logic(),
        tp_sl_monitor()
    )

def backtest_strategy(symbol, timeframe, start_date, end_date, interval_minutes=5):
    """
    Backtest the VWAP strategy over historical data.
    
    Args:
        symbol (str): Trading pair symbol (e.g., 'BTCUSDT')
        timeframe (str): Kline interval (e.g., '1m', '5m', '1h')
        start_date (str): Start date in format 'YYYY-MM-DD HH:MM:SS'
        end_date (str): End date in format 'YYYY-MM-DD HH:MM:SS'
        interval_minutes (int): Time interval between trades in minutes
        
    Returns:
        dict: Dictionary containing backtest results
    """
    # Convert dates to milliseconds timestamps
    start_ts = int(datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S').timestamp() * 1000)
    end_ts = int(datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S').timestamp() * 1000)
    
    # Initialize results tracking
    trades = []
    current_position = None
    entry_price = 0
    total_trades = 0
    winning_trades = 0
    total_profit_percentage = 0
    max_drawdown = 0
    current_drawdown = 0
    peak_balance = entry_usdt
    current_balance = entry_usdt
    
    # Get historical klines data
    historical_data = client.futures_historical_klines(
        symbol=symbol,
        interval=timeframe,
        start_str=start_ts,
        end_str=end_ts
    )
    
    # Convert to DataFrame
    df = pd.DataFrame(historical_data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades_count',
        'taker_buy_volume', 'taker_buy_quote_volume', 'ignore'
    ])
    
    # Convert string values to float
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Calculate indicators
    df = calculate_vwap(df)
    
    # Simulate trading
    for i in range(len(df)):
        if i < 20:  # Skip first few candles to allow indicators to initialize
            continue
            
        row = df.iloc[i]
        
        # Generate signals
        signal = None
        if (row['close'] > row['vwap'] * 0.999 and 
            row['stoch_k'] < 70 and 
            row['rsi'] > 50 and 
            row['close'] > row['ema_trend']):
            signal = 'LONG'
        elif (row['close'] < row['vwap'] * 1.001 and 
              row['stoch_k'] > 30 and 
              row['rsi'] < 50 and 
              row['close'] < row['ema_trend']):
            signal = 'SHORT'
        
        # Handle position management
        if current_position:
            # Calculate current PnL
            current_price = row['close']
            pnl_percentage = ((current_price - entry_price) / entry_price) * 100 if current_position == 'LONG' else ((entry_price - current_price) / entry_price) * 100
            
            # Check for TP/SL
            if pnl_percentage >= tp_percentage or pnl_percentage <= -sl_percentage:
                total_trades += 1
                if pnl_percentage > 0:
                    winning_trades += 1
                
                total_profit_percentage += pnl_percentage
                current_balance *= (1 + pnl_percentage/100)
                
                # Track drawdown
                if current_balance > peak_balance:
                    peak_balance = current_balance
                current_drawdown = (peak_balance - current_balance) / peak_balance * 100
                max_drawdown = max(max_drawdown, current_drawdown)
                
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': row['timestamp'],
                    'position': current_position,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'pnl_percentage': pnl_percentage
                })
                
                current_position = None
                
        # Enter new position if signal and no current position
        elif signal and not current_position:
            current_position = signal
            entry_price = row['close']
            entry_time = row['timestamp']
    
    # Calculate final metrics
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    average_profit = total_profit_percentage / total_trades if total_trades > 0 else 0
    final_balance = current_balance
    total_return = ((final_balance - entry_usdt) / entry_usdt) * 100
    
    results = {
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'win_rate': win_rate,
        'average_profit_per_trade': average_profit,
        'total_return_percentage': total_return,
        'max_drawdown': max_drawdown,
        'final_balance': final_balance,
        'trades': trades
    }
    
    return results

def print_backtest_results(results):
    """
    Print formatted backtest results
    """
    print("\n=== Backtest Results ===")
    print(f"Total Trades: {results['total_trades']}")
    print(f"Winning Trades: {results['winning_trades']}")
    print(f"Win Rate: {results['win_rate']:.2f}%")
    print(f"Average Profit per Trade: {results['average_profit_per_trade']:.2f}%")
    print(f"Total Return: {results['total_return_percentage']:.2f}%")
    print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
    print(f"Final Balance: {results['final_balance']:.2f} USDT")
    
    print("\n=== Trade History ===")
    for trade in results['trades'][:5]:  # Show first 5 trades
        print(f"\nPosition: {trade['position']}")
        print(f"Entry Time: {trade['entry_time']}")
        print(f"Exit Time: {trade['exit_time']}")
        print(f"Entry Price: {trade['entry_price']:.2f}")
        print(f"Exit Price: {trade['exit_price']:.2f}")
        print(f"PnL: {trade['pnl_percentage']:.2f}%")
    
    if len(results['trades']) > 5:
        print("\n... and more trades ...")

if __name__ == "__main__":
    # asyncio.run(main())
    # Example usage
    symbol = 'BTCUSDT'
    timeframe = '15m'  # You can use: '1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d'
    start_date = '2025-04-1 00:00:00'
    end_date = '2025-04-10 00:00:00'

    # Run backtest
    results = backtest_strategy(symbol, timeframe, start_date, end_date)

    # Print results
    print_backtest_results(results)
    

   
   