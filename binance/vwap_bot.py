import asyncio
import logging
from index import *
from telegram import send_telegram_message
from datetime import datetime
import pandas as pd
import numpy as np
import time

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
SYMBOL='DOGEUSDT'

def calculate_vwap(data, atr_period=14, stochastic_period=14, rsi_period=14):
    """
    Perform VWAP strategy calculations.
    """
    # Add RSI
    delta = data['close'].diff(1)

    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(rsi_period).mean()
    avg_loss = loss.rolling(rsi_period).mean()

    # Prevent division by zero in RSI calculation
    rs = avg_gain / avg_loss.replace(0, np.nan)  # Replace zero losses with NaN temporarily
    data['rsi'] = 100 - (100 / (1 + rs.fillna(0)))  # Fill NaN with 0 to handle missing values

    # Add EMA for trend
    data['ema_trend'] = data['close'].rolling(15).mean()

    # Calculate VWAP
    data['vwap'] = (data['close'] * data['volume']).cumsum() / data['volume'].cumsum()

    # Calculate ATR
    high_low = data['high'] - data['low']
    high_close = abs(data['high'] - data['close'].shift(1))
    low_close = abs(data['low'] - data['close'].shift(1))
    data['tr'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    data['atr'] = data['tr'].rolling(atr_period).mean()

    # Calculate Stochastic Oscillator
    data['stoch_k'] = (
        (data['close'] - data['low'].rolling(stochastic_period).min()) /
        (data['high'].rolling(stochastic_period).max() - data['low'].rolling(stochastic_period).min())
    ) * 100
    data['stoch_d'] = data['stoch_k'].rolling(3).mean()

    return data

async def vwap_strategy(data):
    """
    Perform the VWAP trading strategy logic with reduced sensitivity.
    """
    data = calculate_vwap(data)
    log_and_print(f"Signal Check: close={data['close'].iloc[-1]}, vwap={data['vwap'].iloc[-1]}, stoch_k={data['stoch_k'].iloc[-1]}, rsi={data['rsi'].iloc[-1]}, ema_trend={data['ema_trend'].iloc[-1]}")

    signal = None
    vwap_tolerance = 0.001  # 1% tolerance (reduced sensitivity)
    if data['close'].iloc[-1] > data['vwap'].iloc[-1] * (1 - vwap_tolerance) and data['stoch_k'].iloc[-1] < 70 and data['rsi'].iloc[-1] > 50 and data['close'].iloc[-1] > data['ema_trend'].iloc[-1]:
        signal = 'LONG'
    elif data['close'].iloc[-1] < data['vwap'].iloc[-1] * (1 + vwap_tolerance) and data['stoch_k'].iloc[-1] > 30 and data['rsi'].iloc[-1] < 50 and data['close'].iloc[-1] < data['ema_trend'].iloc[-1]:
        signal = 'SHORT'
    
    return {'signal': signal}

async def trade_logic():
    global current_position, entry_price, entry_quantity
    symbol = SYMBOL
    interval = '1m'

    while True:
        try:
            # Fetch current kline data
            data = get_klines_all(symbol, interval, limit=20)
            if data.empty:
                log_and_print("No data retrieved from Binance API.")
                await asyncio.sleep(60)
                continue

            # Execute VWAP strategy
            result = await vwap_strategy(data)
            signal = result['signal']
           

            if signal and (current_position is None or current_position != signal):
                log_and_print(f"Signal Generated: {signal}")
                # Fetch available balance
                available_balance = get_futures_account_balance('USDT')
                current_price = data['close'].iloc[-1]

                # Fetch leverage
                leverage_info = client.futures_leverage_bracket(symbol=symbol)[0]
                leverage = leverage_info['brackets'][0]['initialLeverage']

                # Calculate required margin
                required_margin = (available_balance * current_price) / leverage

                if available_balance < 10:
                    log_and_print("Available USDT balance is less than 10 USDT. Stopping the bot.")
                    send_telegram_message("Your available USDT balance is less than 10 USDT. Stopping the bot.")
                    break
                elif available_balance < required_margin:
                    log_and_print(f"Insufficient margin. Required: {required_margin}, Available: {available_balance}")
                    send_telegram_message("Insufficient margin for placing order.")
                    continue

                # Place order
                order_side = 'BUY' if signal == 'LONG' else 'SELL'
                order = place_order(symbol, order_side, available_balance)
                log_and_print(f"Order Response: {order}")
                if order:
                    filled_order = wait_for_order_fill(order['orderId'], symbol)
                    if filled_order:
                        current_position = signal
                        entry_price = float(filled_order['avgPrice'])
                        entry_quantity = float(filled_order['executedQty'])
                    else:
                        log_and_print("Order was not filled within the timeout period.")

        except Exception as e:
            log_and_print(f"Error in trade logic: {e}")

        await asyncio.sleep(60)



def wait_for_order_fill(order_id, symbol, timeout=30):
    start_time = time.time()
    while time.time() - start_time < timeout:
        order_status = client.futures_get_order(symbol=symbol, orderId=order_id)
        if order_status['status'] in ['FILLED', 'PARTIALLY_FILLED']:
            return order_status
        time.sleep(1)  # Wait 1 second before retrying
    return None

async def tp_sl_monitor():
    global current_position, entry_price, entry_quantity
    interval = 10  # Check every 10 seconds
    symbol = SYMBOL
    tp_percentage = 0.2  # Take Profit: 0.2%
    sl_percentage = 0.1  # Stop Loss: 0.1%

    while True:
        try:
            if current_position is not None:
                data = get_klines_all(symbol, '1m', limit=1)
                if data.empty:
                    log_and_print("No data retrieved for TP/SL monitoring.")
                    await asyncio.sleep(interval)
                    continue

                latest_price = data['close'].iloc[-1]
                pnl_percentage = ((latest_price - entry_price) / entry_price) * 100 if current_position == 'LONG' else ((entry_price - latest_price) / entry_price) * 100

                log_and_print(f"TP/SL Check - Latest Price: {latest_price}, Entry Price: {entry_price}, PnL Percentage: {pnl_percentage}, Position: {current_position}")

                # Check for TP or SL
                if current_position == 'LONG' and (pnl_percentage >= tp_percentage or pnl_percentage <= -sl_percentage):
                    log_and_print("Closing LONG position due to TP/SL.")
                    close_response = close_futures_position(symbol, 'LONG', entry_quantity)
                    log_and_print(f"Close Response: {close_response}")
                    current_position = None
                elif current_position == 'SHORT' and (pnl_percentage >= tp_percentage or pnl_percentage <= -sl_percentage):
                    log_and_print("Closing SHORT position due to TP/SL.")
                    close_response = close_futures_position(symbol, 'SHORT', entry_quantity)
                    log_and_print(f"Close Response: {close_response}")
                    current_position = None

        except Exception as e:
            log_and_print(f"Error in TP/SL monitor: {e}")

        await asyncio.sleep(interval)


async def main():
    """Main entry point for the VWAP bot."""
    log_and_print("Starting VWAP bot.")
    send_telegram_message("VWAP bot started.")

    # Run the trade logic and TP/SL monitor concurrently
    await asyncio.gather(
        trade_logic(),
        tp_sl_monitor()
    )

if __name__ == "__main__":
    asyncio.run(main())
