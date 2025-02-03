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

async def trade_logic():
    global current_position, entry_price, entry_quantity
    symbol = SYMBOL
    interval = '5m'  # 5-minute timeframe
    interval_value = 5  # Adjusted sleep interval in minutes

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

            if signal and (current_position is None or current_position != signal):
                log_and_print(f"Signal Generated: {signal}")
                available_balance = get_futures_account_balance('USDT')
                current_price = data['close'].iloc[-1]

                leverage_info = client.futures_leverage_bracket(symbol=symbol)[0]
                leverage = leverage_info['brackets'][0]['initialLeverage']

                max_quantity = calculate_max_quantity(available_balance, leverage, current_price)
                quantity = min(available_balance, max_quantity)
                log_and_print(f"Adjusted Quantity: {quantity}")

                if available_balance < 20:
                    log_and_print("Insufficient balance. Stopping bot.")
                    send_telegram_message("Insufficient balance. Stopping bot.")
                    break

                order_side = 'BUY' if signal == 'LONG' else 'SELL'
                order = place_order(symbol, order_side, quantity)
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
    interval = 30  # Check every 30 seconds

    while True:
        try:
            if current_position is not None:
                data = get_klines_all(symbol, '5m', limit=1)
                if data.empty:
                    await asyncio.sleep(interval)
                    continue

                latest_price = data['close'].iloc[-1]
                pnl_percentage = ((latest_price - entry_price) / entry_price) * 100 if current_position == 'LONG' else ((entry_price - latest_price) / entry_price) * 100
                pnl_usdt = (pnl_percentage / 100) * (entry_price * entry_quantity)

                log_and_print(f"TP/SL Check - Latest Price: {latest_price}, Entry Price: {entry_price}, PnL Percentage: {pnl_percentage:.2f}%, PnL USDT: {pnl_usdt:.2f}, Position: {current_position}")

                if (current_position == 'LONG' and pnl_percentage >= tp_percentage) or (current_position == 'SHORT' and pnl_percentage >= tp_percentage):
                    log_and_print("Take Profit triggered.")
                    close_futures_position(symbol, current_position, entry_quantity)
                    send_telegram_message(f"🚀 Take Profit hit!\n🔹 Symbol: {symbol}\n💰 Close Price: {latest_price}\n📈 PnL: {pnl_percentage:.2f}% ({pnl_usdt:.2f} USDT)")
                    current_position = None
                elif (current_position == 'LONG' and pnl_percentage <= -sl_percentage) or (current_position == 'SHORT' and pnl_percentage <= -sl_percentage):
                    log_and_print("Stop Loss triggered.")
                    close_futures_position(symbol, current_position, entry_quantity)
                    send_telegram_message(f"⚠️ Stop Loss hit!\n🔹 Symbol: {symbol}\n💰 Close Price: {latest_price}\n📉 PnL: {pnl_percentage:.2f}% ({pnl_usdt:.2f} USDT)")
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
    # asyncio.run(main())

    # available_balance = get_futures_account_balance('USDT')
    # leverage = 1  # Default leverage to avoid division errors
    # current_price = 0.24934
    # risk_factor = 0.9
    # max_quantity = (available_balance * leverage * risk_factor) / current_price
    # _, _, step_size = get_lot_size(SYMBOL) 
    # adjusted_quantity = adjust_quantity(float(max_quantity), step_size)
    # if adjusted_quantity < step_size:
    #     raise ValueError(f"Calculated quantity {adjusted_quantity} is below the minimum lot size {step_size}.")
    # log_and_print(f"Final Adjusted Quantity: {adjusted_quantity}")
    # order = place_order(SYMBOL, "BUY", adjusted_quantity)

    # available_balance = get_futures_account_balance('USDT')
    available_balance  = 30
    current_price =  94189.0

    # leverage_info = client.futures_leverage_bracket(symbol=SYMBOL)[0]
    # leverage = leverage_info['brackets'][0]['initialLeverage']
    leverage = 1

    # max_quantity = calculate_max_quantity(available_balance, leverage, current_price)
    # quantity = min(available_balance, max_quantity)
    risk_factor = 0.9  # Use 90% of the available balance
    max_quantity = (available_balance * leverage * risk_factor) / current_price
    log_and_print(f"Adjusted Quantity: {max_quantity}")

    order = place_order(SYMBOL, "SELL", max_quantity)