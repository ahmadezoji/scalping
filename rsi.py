import asyncio
import matplotlib.pyplot as plt
import aiohttp
import logging
from bingx import *
from datetime import datetime, timedelta
import numpy as np
import threading


# Existing setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

stop_thread = False  # Control variable for the SL/TP thread
sl_tp_thread = None


SYMBOL = "BTC-USDT"
INTERVAL = "1m"
MIN = 1
LIMIT = 100
AMOUNT_USDT = 3000  # USDT

# SL = -0.01  # Stop loss percentage
# TP = 0.15   # Take profit percentage
SL = -0.05  # Stop Loss: 2% below the entry price
TP = 0.15   # Take Profit: 3% above the entry price
THRESHOLD_PERCENTAGE = 0.0001  # Sensitivity for SMA
ATR_PERIOD = 14  # ATR period
RSI_PERIOD = 14  # RSI period
RSI_OVERBOUGHT = 70  # RSI overbought threshold
RSI_OVERSOLD = 30    # RSI oversold threshold

order_type = OrderType.NONE
last_order_id = None
ordered_price = None
count_of_long = 1
count_of_short = 1
last_trade_amount = None

# Function to calculate RSI


def calculate_rsi(close_prices, period=RSI_PERIOD):
    deltas = np.diff(close_prices)
    seed = deltas[:period]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    rs = up / down
    rsi = np.zeros_like(close_prices)
    rsi[:period] = 100. - 100. / (1. + rs)

    for i in range(period, len(close_prices)):
        delta = deltas[i - 1]  # The diff is 1 shorter
        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up * (period - 1) + upval) / period
        down = (down * (period - 1) + downval) / period

        rs = up / down
        rsi[i] = 100. - 100. / (1. + rs)

    return rsi


def trade_amount_calculate(symbol):
    last_price_symbol = last_price(symbol=symbol)
    if (last_price_symbol is not None or last_price_symbol != 0.0):
        return AMOUNT_USDT / last_price_symbol
    return -1

# Function to calculate moving averages


def calculate_moving_averages(klines):
    close_prices = np.array([float(kline['close']) for kline in klines])
    sma5 = close_prices[-5:].mean()
    sma8 = close_prices[-8:].mean()
    sma13 = close_prices[-13:].mean()
    return sma5, sma8, sma13, close_prices

# Function to calculate ATR (Average True Range)


def calculate_atr(klines, period=ATR_PERIOD):
    high_prices = np.array([float(kline['high']) for kline in klines])
    low_prices = np.array([float(kline['low']) for kline in klines])
    close_prices = np.array([float(kline['close']) for kline in klines])

    tr_list = np.maximum(high_prices[1:] - low_prices[1:],
                         np.maximum(np.abs(high_prices[1:] - close_prices[:-1]),
                                    np.abs(low_prices[1:] - close_prices[:-1])))
    atr = np.mean(tr_list[-period:])
    return atr

# Function to close the last order based on the current order type


# Function to close the last order based on the current order type
def close_last():
    global order_type, last_order_id, count_of_long, count_of_short, ordered_price, stop_thread, sl_tp_thread

    # Set stop_thread to True to signal the SL/TP thread to stop
    stop_thread = True

    # If the SL/TP thread is running, wait for it to finish
    if sl_tp_thread and sl_tp_thread.is_alive():
        sl_tp_thread.join()
        sl_tp_thread = None  # Reset the thread variable after it has stopped

    if order_type == OrderType.SHORT:
        result = closeAllPosition(symbol=SYMBOL)
        if result == 200:
            order_type = OrderType.NONE
            logger.info(f"Closed SHORT order id = {last_order_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            count_of_short = 1  # Reset counter for short trades
    elif order_type == OrderType.LONG:
        result = closeAllPosition(symbol=SYMBOL)
        if result == 200:
            order_type = OrderType.NONE
            logger.info(f"Closed LONG order id = {last_order_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            count_of_long = 1  # Reset counter for long trades

    # Reset ordered price and stop_thread after closing the position
    ordered_price = None
    stop_thread = False  # Reset stop_thread for the next trade


# Function to decide action based on RSI zones
def get_rsi_zone_action(rsi):
    if 0 <= rsi < 20:
        return "Extremely Oversold", "BUY"
    elif 20 <= rsi < 40:
        return "Oversold", "BUY"
    elif 40 <= rsi < 60:
        return "Neutral", "WAIT"
    elif 60 <= rsi < 80:
        return "Overbought", "SELL"
    elif 80 <= rsi <= 100:
        return "Extremely Overbought", "SELL"
    return "Undefined", "WAIT"

def calculate_rolling_sma(close_prices, window):
    return np.convolve(close_prices, np.ones(window)/window, mode='valid')

# Function to make trading decisions based on RSI zones
def make_trade_decision_rsi_based(klines, close_prices):
    global order_type, last_order_id, count_of_long, count_of_short, ordered_price, last_trade_amount, stop_thread

    # Calculate RSI
    rsi = calculate_rsi(close_prices)[-1]  # Get the latest RSI value
    rsi_zone, action = get_rsi_zone_action(rsi)
    logger.info(f"RSI: {rsi} ({rsi_zone}) at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Take action based on the RSI zone
    if action == "BUY":
        if order_type == OrderType.SHORT:
            close_last()  # Close any existing short position
        if count_of_long == 1:  # Ensure only one long trade at a time
            amount = trade_amount_calculate(symbol=SYMBOL)
            last_order_id, order_type, quantity = open_long(symbol=SYMBOL, quantity=amount)
            logger.info(f"LONG opened at {close_prices[-1]} AMOUNT: {amount} RSI: {rsi} ({rsi_zone}) at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            count_of_long += 1
            ordered_price = close_prices[-1]
            stop_thread = False  # Reset stop control for SL/TP monitoring thread
            sl_tp_thread = threading.Thread(target=monitor_sl_tp)  # Start SL/TP monitoring thread
            sl_tp_thread.start()

    elif action == "SELL":
        if order_type == OrderType.LONG:
            close_last()  # Close any existing long position
        if count_of_short == 1:  # Ensure only one short trade at a time
            amount = trade_amount_calculate(symbol=SYMBOL)
            last_order_id, order_type, quantity = open_short(symbol=SYMBOL, quantity=amount)
            logger.info(f"SHORT opened at {close_prices[-1]} AMOUNT: {amount} RSI: {rsi} ({rsi_zone}) at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            count_of_short += 1
            ordered_price = close_prices[-1]
            stop_thread = False  # Reset stop control for SL/TP monitoring thread
            sl_tp_thread = threading.Thread(target=monitor_sl_tp)  # Start SL/TP monitoring thread
            sl_tp_thread.start()

    else:  # "WAIT" action
        logger.info(f"Waiting due to neutral RSI zone ({rsi_zone}). No action taken.")


async def main():
    async with aiohttp.ClientSession() as session:
        while True:
            try:
                past_time_ms = get_server_time() - (LIMIT * MIN * 60 * 1000)
                klines = get_kline(
                    symbol=SYMBOL, interval=INTERVAL, limit=LIMIT, start=past_time_ms)

                klines.raise_for_status()
                klines = klines.json().get('data', [])
                if len(klines) == 0:
                    return
                klines.reverse()
                if klines:
                    sma5, sma8, sma13, close_prices = calculate_moving_averages(
                        klines)
                    logger.info(f"PRICES: {close_prices[-1]} SMA5: {
                                sma5}, SMA8: {sma8}, SMA13: {sma13}")
                    
                    make_trade_decision_rsi_based(
                        klines, close_prices)
                else:
                    logger.warning(
                        "No Kline data received. Skipping this iteration.")
                await asyncio.sleep(MIN * 60)
                # await asyncio.sleep(30)
            except asyncio.CancelledError:
                logger.warning("Task was cancelled. Exiting.")
                break
            except Exception as e:
                logger.exception(
                    "Unexpected error occurred in main loop. Continuing...")


def monitor_sl_tp():
    global stop_thread, order_type, ordered_price

    while not stop_thread:
        if ordered_price is not None and order_type is not OrderType.NONE:
            # Check the current price
            current_price = last_price(symbol=SYMBOL)
            profit_percentage = (
                (current_price - ordered_price) / ordered_price) * 100
            unrealized_pnl = profit_percentage if order_type == OrderType.LONG else - \
                1 * profit_percentage
            logger.info(f"unrealized_pnl: {unrealized_pnl} ")
            # Check SL and TP conditions
            if unrealized_pnl <= SL:
                close_last()
                logger.info(f"Closing position due to Stop Loss at {
                            datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                stop_thread = True  # Stop the monitoring thread
                ordered_price = None
            elif unrealized_pnl >= TP:
                close_last()
                logger.info(f"Closing position due to Take Profit at {
                            datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                stop_thread = True  # Stop the monitoring thread
                ordered_price = None
        # Wait for 10 seconds before checking again
        time.sleep(10)


# Run the async loop
if __name__ == "__main__":
    try:
        # asyncio.run(back_test(symbol=SYMBOL))
        asyncio.run(main())
        # asyncio.run(backtest(symbol=SYMBOL))

    except KeyboardInterrupt:
        logger.info("Trading bot stopped manually.")
    except Exception as e:
        logger.exception("Critical error. Bot stopped.")
