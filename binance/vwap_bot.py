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
    """Main loop for fetching kline data and executing the VWAP strategy."""
    global current_position, entry_price, entry_quantity
    symbol = 'DOGEUSDT'
    interval = '1m'
    quantity = 0.0
    
    while True:
        try:
            # Fetch current kline data
            data = get_klines_all(symbol, interval, limit=20)
            if data.empty:
                logging.warning("No data retrieved from Binance API.")
                await asyncio.sleep(60)  # Wait before retrying
                continue

            # Execute VWAP strategy
            result = await vwap_strategy(data)
            signal = result['signal']

            # if current_position is None:
            if signal == 'LONG':
                if current_position is None or current_position == 'SHORT':
                    quantity = get_futures_account_balance('USDT')
                    if quantity is None or quantity < 10:
                        logging.warning("Available USDT balance is less than 10 USDT. Stopping the bot.")
                        send_telegram_message("Your available USDT balance is less than 10 USDT. Stopping the bot.")
                        break
                    elif quantity >= 10 and quantity < 20:
                        logging.info("Placing LONG order.")
                        order = place_order(symbol, 'BUY', quantity)
                        if order:
                            current_position = 'LONG'
                            entry_price = float(order['fills'][0]['price'])
                            entry_quantity = quantity

            elif signal == 'SHORT':
                if current_position is None or current_position == 'LONG':
                    quantity = get_futures_account_balance('USDT')
                    if quantity is None or quantity < 10:
                        logging.warning("Available USDT balance is less than 10 USDT. Stopping the bot.")
                        send_telegram_message("Your available USDT balance is less than 10 USDT. Stopping the bot.")
                        break
                    elif quantity >= 10 and quantity < 20:
                        logging.info("Placing SHORT order.")
                        order = place_order(symbol, 'SELL', quantity)
                        if order:
                            current_position = 'SHORT'
                            entry_price = float(order['fills'][0]['price'])
                            entry_quantity = quantity

        except Exception as e:
            logging.error(f"Error in trade logic: {e}")

        await asyncio.sleep(60)  # Adjust this based on the kline interval

async def tp_sl_monitor():
    """Monitor positions for TP and SL conditions."""
    global current_position, entry_price, entry_quantity
    symbol = 'ETHUSDT'
    interval = 10  # Check every 10 seconds
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
    asyncio.run(main())
   
