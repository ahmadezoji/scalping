import asyncio
import logging
from index import *
from telegram import send_telegram_message
from datetime import datetime
import pandas as pd
import numpy as np

logging.basicConfig(
    filename='vwap_bot.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)



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
    data = calculate_vwap(data)
    logging.info(f"Signal Check: close={data['close'].iloc[-1]}, vwap={data['vwap'].iloc[-1]}, stoch_k={data['stoch_k'].iloc[-1]}, rsi={data['rsi'].iloc[-1]}, ema_trend={data['ema_trend'].iloc[-1]}")

    signal = None
    vwap_tolerance = 0.01  # 1% tolerance
    if data['close'].iloc[-1] > data['vwap'].iloc[-1] * (1 - vwap_tolerance) and data['stoch_k'].iloc[-1] < 70 and data['rsi'].iloc[-1] > 40 and data['close'].iloc[-1] > data['ema_trend'].iloc[-1]:
        signal = 'LONG'
    elif data['close'].iloc[-1] < data['vwap'].iloc[-1] * (1 + vwap_tolerance) and data['stoch_k'].iloc[-1] > 30 and data['rsi'].iloc[-1] < 60 and data['close'].iloc[-1] < data['ema_trend'].iloc[-1]:
        signal = 'SHORT'
    
    logging.info(f"Generated Signal: {signal}")
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
                logging.warning("No data retrieved from Binance API.")
                await asyncio.sleep(60)
                continue

            # Execute VWAP strategy
            result = await vwap_strategy(data)
            signal = result['signal']
            logging.info(f"Signal Generated: {signal}")
            logging.info(f"Current Position: {current_position}")

            if signal and (current_position is None or current_position != signal):
                # Fetch available balance
                available_balance = get_futures_account_balance('USDT')
                current_price = data['close'].iloc[-1]

                # Fetch leverage
                leverage_info = client.futures_leverage_bracket(symbol=symbol)[0]
                leverage = leverage_info['brackets'][0]['initialLeverage']

                # Calculate required margin
                required_margin = (available_balance * current_price) / leverage

                if available_balance < 10:
                    logging.warning("Available USDT balance is less than 10 USDT. Stopping the bot.")
                    send_telegram_message("Your available USDT balance is less than 10 USDT. Stopping the bot.")
                    break
                elif available_balance < required_margin:
                    logging.warning(f"Insufficient margin. Required: {required_margin}, Available: {available_balance}")
                    send_telegram_message("Insufficient margin for placing order.")
                    continue

                # Place order
                order_side = 'BUY' if signal == 'LONG' else 'SELL'
                order = place_order(symbol, order_side, available_balance)
                logging.info(f"Order Response: {order}")

                if order and order.get('status') in ['FILLED', 'PARTIALLY_FILLED']:
                    current_position = signal
                    entry_price = float(order['fills'][0]['price'])
                    entry_quantity = available_balance
                else:
                    logging.warning(f"Order not filled immediately: {order}")

        except Exception as e:
            logging.error(f"Error in trade logic: {e}")

        await asyncio.sleep(60)


async def tp_sl_monitor():
    """Monitor positions for TP and SL conditions."""
    global current_position, entry_price, entry_quantity
    interval = 10  # Check every 10 seconds
    symbol=SYMBOL
    tp_percentage = 0.02  # Example TP threshold
    sl_percentage = 0.01  # Example SL threshold

    while True:
        try:
            if current_position is not None:
                # Fetch the latest price
                data = get_klines_all(symbol, '1m', limit=1)
                if data.empty:
                    logging.warning("No data retrieved for TP/SL monitoring.")
                    await asyncio.sleep(interval)
                    continue

                latest_price = data['close'].iloc[-1]
                if current_position == 'LONG':
                    pnl = (latest_price - entry_price) * entry_quantity
                    if pnl >= entry_price * tp_percentage or pnl <= -entry_price * sl_percentage:
                        logging.info("Closing LONG position due to TP/SL.")
                        close_futures_position(symbol, 'LONG', entry_quantity)
                        current_position = None
                elif current_position == 'SHORT':
                    pnl = (entry_price - latest_price) * entry_quantity
                    if pnl >= entry_price * tp_percentage or pnl <= -entry_price * sl_percentage:
                        logging.info("Closing SHORT position due to TP/SL.")
                        close_futures_position(symbol, 'SHORT', entry_quantity)
                        current_position = None

        except Exception as e:
            logging.error(f"Error in TP/SL monitor: {e}")

        await asyncio.sleep(interval)

async def main():
    """Main entry point for the VWAP bot."""
    logging.info("Starting VWAP bot.")
    send_telegram_message("VWAP bot started.")

    # Run the trade logic and TP/SL monitor concurrently
    await asyncio.gather(
        trade_logic(),
        tp_sl_monitor()
    )

if __name__ == "__main__":
    # asyncio.run(main())
    close_all_positions()
   
