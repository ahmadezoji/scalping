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

    while True:
        try:
            data = get_klines_all(symbol, interval, limit=20)
            if data.empty:
                log_and_print("No data retrieved from Binance API.")
                await asyncio.sleep(interval_value * 60)
                continue

            data = calculate_vwap(data)

            signal = None
            if data['close'].iloc[-1] > data['vwap'].iloc[-1] * 0.999 and data['stoch_k'].iloc[-1] < 70 and data['rsi'].iloc[-1] > 50 and data['close'].iloc[-1] > data['ema_trend'].iloc[-1]:
                signal = 'LONG'
            elif data['close'].iloc[-1] < data['vwap'].iloc[-1] * 1.001 and data['stoch_k'].iloc[-1] > 30 and data['rsi'].iloc[-1] < 50 and data['close'].iloc[-1] < data['ema_trend'].iloc[-1]:
                signal = 'SHORT'

            log_and_print(f"Signal Check: close={data['close'].iloc[-1]}, vwap={data['vwap'].iloc[-1]}, stoch_k={data['stoch_k'].iloc[-1]}, rsi={data['rsi'].iloc[-1]}, ema_trend={data['ema_trend'].iloc[-1]}")

            if signal:
                if current_position:
                    if current_position == signal:
                        # Skip the rest of the loop and go to the next iteration
                        await asyncio.sleep(interval_value * 60)
                        continue
                    if current_position != signal:
                        # Calculate PnL when closing previous position
                        latest_price = data['close'].iloc[-1]
                        pnl_percentage = ((latest_price - entry_price) / entry_price) * 100 if current_position == 'LONG' else ((entry_price - latest_price) / entry_price) * 100
                        pnl_usdt = (pnl_percentage / 100) * (entry_price * entry_quantity)

                        log_and_print(f"Closing {current_position} position - PnL: {pnl_percentage:.2f}% ({pnl_usdt:.2f} USDT)")
                        send_telegram_message(
                            f"🔄 Position Closed\n🔹 Previous Position: {current_position}\n💰 Exit Price: {latest_price}\n📈 PnL: {pnl_percentage:.2f}% ({pnl_usdt:.2f} USDT)"
                        )
                        current_position = None  # Reset current position

                # Place new order
                log_and_print(f"Signal Generated: {signal}")

                available_balance = get_futures_account_balance('USDT')

                current_price = data['close'].iloc[-1]

                leverage_info = client.futures_leverage_bracket(symbol=symbol)[0]
                leverage = leverage_info['brackets'][0]['initialLeverage']

                max_quantity = calculate_max_quantity(entry_usdt, leverage, current_price)
                quantity = min(entry_usdt, max_quantity)
                log_and_print(f"Adjusted Quantity: {quantity}")

                if available_balance < 20:
                    log_and_print("Insufficient balance. Stopping bot.")
                    send_telegram_message("Insufficient balance. Stopping bot.")
                    break

                order_side = 'BUY' if signal == 'LONG' else 'SELL'
                # order = place_order(symbol, order_side, quantity)
                order = order_side
                message = (
                    f"🚀 <b>New Order Placed</b> 🚀\n"
                    f"📈 <b>Symbol:</b> {symbol}\n"
                    f"🔁 <b>Action:</b> {order_side}\n"
                    f"💵 <b>Quantity:</b> {quantity}\n"
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

if __name__ == "__main__":
    asyncio.run(main())
    

   
   
def backtest_strategy(symbol, timeframe, start_date, end_date, interval_minutes=5):
    """
    Backtest the VWAP strategy with realistic conditions.
    """
    start_ts = int(datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S').timestamp() * 1000)
    end_ts = int(datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S').timestamp() * 1000)
    
    # Initialize results tracking
    trades = []
    current_position = None
    entry_price = 0
    entry_quantity = 0
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
    
    df = pd.DataFrame(historical_data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades_count',
        'taker_buy_volume', 'taker_buy_quote_volume', 'ignore'
    ])
    
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = calculate_vwap(df)
    
    def simulate_slippage(price, side, volatility_factor=0.0005):
        """Simulate realistic slippage based on volatility"""
        slippage = price * volatility_factor * (1 if np.random.random() > 0.5 else -1)
        return price + slippage if side == 'LONG' else price - slippage

    def calculate_realistic_quantity(balance, price, leverage):
        """Calculate quantity with realistic constraints"""
        max_quantity = (balance * leverage) / price
        # Add minimum quantity checks and round to valid decimals
        return round(min(max_quantity, balance / price), 6)
    
    # Simulate trading with realistic conditions
    for i in range(len(df)):
        if i < 20:  # Skip first few candles
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
        
        # Handle position management with realistic conditions
        if current_position:
            # Simulate market conditions and delays
            current_price = row['close']  # Use closing price for simulation
            high_price = row['high']
            low_price = row['low']
            
            # Calculate PnL with price range consideration
            if current_position == 'LONG':
                worst_price = low_price
                best_price = high_price
            else:
                worst_price = high_price
                best_price = low_price
            
            # Calculate PnL range
            best_pnl = ((best_price - entry_price) / entry_price) * 100 if current_position == 'LONG' else ((entry_price - best_price) / entry_price) * 100
            worst_pnl = ((worst_price - entry_price) / entry_price) * 100 if current_position == 'LONG' else ((entry_price - worst_price) / entry_price) * 100
            
            # Check if TP or SL was hit during the candle
            if best_pnl >= tp_percentage or worst_pnl <= -sl_percentage:
                total_trades += 1
                # Use realistic exit price with slippage
                exit_price = simulate_slippage(
                    best_price if best_pnl >= tp_percentage else worst_price,
                    current_position
                )
                
                pnl_percentage = ((exit_price - entry_price) / entry_price) * 100 if current_position == 'LONG' else ((entry_price - exit_price) / entry_price) * 100
                
                if pnl_percentage > 0:
                    winning_trades += 1
                
                # Update balance with commission consideration
                commission = 0.0004  # 0.04% commission per trade
                net_pnl = pnl_percentage - (commission * 100)  # Convert commission to percentage
                current_balance *= (1 + net_pnl/100)
                
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
                    'exit_price': exit_price,
                    'pnl_percentage': pnl_percentage,
                    'net_pnl': net_pnl
                })
                
                current_position = None
                
        # Enter new position with realistic conditions
        elif signal and not current_position:
            # Simulate entry with slippage
            entry_price = simulate_slippage(row['close'], signal)
            
            # Calculate quantity based on current balance
            leverage = 20  # Example leverage value
            entry_quantity = calculate_realistic_quantity(current_balance, entry_price, leverage)
            
            if entry_quantity > 0:  # Check if quantity is valid
                current_position = signal
                entry_time = row['timestamp']
                # Deduct entry commission
                current_balance *= (1 - 0.0004)  # 0.04% commission
    
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
        'trades': trades,
        'commission_paid': (entry_usdt - final_balance) * 0.0004  # Track total commission paid
    }
    
    return results
    

   
   
def print_backtest_results(results):
    """
    Print detailed backtest results with realistic metrics
    """
    print("\n=== Backtest Results with Realistic Conditions ===")
    print(f"Total Trades: {results['total_trades']}")
    print(f"Winning Trades: {results['winning_trades']}")
    print(f"Win Rate: {results['win_rate']:.2f}%")
    print(f"Average Profit per Trade (Before Fees): {results['average_profit_per_trade']:.2f}%")
    print(f"Total Return (After Fees): {results['total_return_percentage']:.2f}%")
    print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
    print(f"Initial Balance: {entry_usdt:.2f} USDT")
    print(f"Final Balance: {results['final_balance']:.2f} USDT")
    print(f"Total Commission Paid: {results['commission_paid']:.4f} USDT")
    
    print("\n=== Detailed Trade History ===")
    profitable_trades = [t for t in results['trades'] if t['net_pnl'] > 0]
    losing_trades = [t for t in results['trades'] if t['net_pnl'] <= 0]
    
    print(f"\nProfitable Trades: {len(profitable_trades)}")
    print(f"Average Profit (Winners): {np.mean([t['net_pnl'] for t in profitable_trades]):.2f}%")
    print(f"Losing Trades: {len(losing_trades)}")
    print(f"Average Loss (Losers): {np.mean([t['net_pnl'] for t in losing_trades]):.2f}%")
    
    print("\n=== Sample Trades ===")
    for trade in results['trades'][:5]:
        print(f"\nPosition: {trade['position']}")
        print(f"Entry Time: {trade['entry_time']}")
        print(f"Exit Time: {trade['exit_time']}")
        print(f"Entry Price: {trade['entry_price']:.2f}")
        print(f"Exit Price: {trade['exit_price']:.2f}")
        print(f"Gross PnL: {trade['pnl_percentage']:.2f}%")
        print(f"Net PnL (After Fees): {trade['net_pnl']:.2f}%")
    
    if len(results['trades']) > 5:
        print("\n... and more trades ...")
        
    print("\n=== Risk Metrics ===")
    print(f"Profit Factor: {abs(np.sum([t['net_pnl'] for t in profitable_trades]) / np.sum([t['net_pnl'] for t in losing_trades])):.2f}")
    print(f"Average Win/Loss Ratio: {abs(np.mean([t['net_pnl'] for t in profitable_trades]) / np.mean([t['net_pnl'] for t in losing_trades])):.2f}")
    print(f"Expectancy: {np.mean([t['net_pnl'] for t in results['trades']]):.2f}%")
    

   
   