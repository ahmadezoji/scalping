import asyncio
import matplotlib.pyplot as plt
import aiohttp
import logging
from bingx import *
from datetime import datetime, timedelta
import numpy as np


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SYMBOL = "BTC-USDT"
INTERVAL = "1m"
MIN = 1
LIMIT = 100
AMOUNT_USDT = 2000  # USDT

SL = -0.08 # % stop loss percentage
TP = 0.18  # % take profit percentage

# THRESHOLD_PERCENTAGE = 0.002 # Moderate Sensitivity
# THRESHOLD_PERCENTAGE = 0.001 # High Sensitivity
# THRESHOLD_PERCENTAGE = 0.0005 # Very High Sensitivity
# THRESHOLD_PERCENTAGE = 0.0001  # ultra High Sensitivity
THRESHOLD_PERCENTAGE = 0.0007
ATR_PERIOD = 14  # ATR period

order_type = OrderType.NONE
last_order_id = None
ordered_price = None
count_of_long = 1
count_of_short = 1
last_trade_amount = None


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

    # Calculate the True Range (TR) and then average it over the specified period to get the ATR
    tr_list = np.maximum(high_prices[1:] - low_prices[1:],
                         np.maximum(np.abs(high_prices[1:] - close_prices[:-1]),
                                    np.abs(low_prices[1:] - close_prices[:-1])))
    atr = np.mean(tr_list[-period:])
    return atr

# Function to close the last order based on the current order type


def close_last():
    global order_type, last_order_id, count_of_long, count_of_short, ordered_price

    if order_type == OrderType.SHORT:
        # result = close_short(symbol=SYMBOL, quantity=0.015)
        result = closeAllPosition(symbol=SYMBOL)
        if result == 200:
            order_type = OrderType.NONE
            logger.info(f"Closed SHORT order id = {last_order_id} at {datetime.now().strftime(
                        '%Y-%m-%d %H:%M:%S')}")
            count_of_short = 1  # Reset counter for short trades
    elif order_type == OrderType.LONG:
        result = closeAllPosition(symbol=SYMBOL)
        if result == 200:
            order_type = OrderType.NONE
            logger.info(f"Closed LONG order id = {last_order_id} at {datetime.now().strftime(
                        '%Y-%m-%d %H:%M:%S')}")
            count_of_long = 1  # Reset counter for long trades

# Function to make trading decisions with threshold and ATR checks


def make_trade_decision(sma5, sma8, sma13, klines, close_prices):
    global order_type, last_order_id, count_of_long, count_of_short, ordered_price, last_trade_amount

    # Calculate the ATR for volatility assessment
    atr = calculate_atr(klines)

    # Calculate profit/loss percentage based on last traded price
    if ordered_price is not None and order_type is not OrderType.NONE:
        profit = (close_prices[-1] - ordered_price) / \
            ordered_price  # Calculate profit
        profit_percentage = (
            (close_prices[-1] - ordered_price) / ordered_price) * 100
        unrealized_pnl = profit_percentage if order_type == OrderType.LONG else - \
            1 * profit_percentage

        logger.info(f"unrealized_pnl = {
                    unrealized_pnl} Calculated profit = {profit} ")

        # Check if the trade needs to be closed based on SL
        # if (order_type == OrderType.LONG and unrealized_pnl <= SL) or (order_type == OrderType.SHORT and unrealized_pnl >= -1 * SL):
        if unrealized_pnl <= SL:
            close_last()
            logger.info(f"Closing position due to Stop Loss condition at {datetime.now().strftime(
                        '%Y-%m-%d %H:%M:%S')}")
            ordered_price = None  # Reset the ordered price after closing
        # Check if the trade needs to be closed based on TP
        # if (order_type == OrderType.LONG and profit >= TP) or (order_type == OrderType.SHORT and profit <= -1 * TP):
        if unrealized_pnl >= TP:
            close_last()
            logger.info(f"Closing position due to Take Profit condition at {datetime.now().strftime(
                        '%Y-%m-%d %H:%M:%S')}")
            ordered_price = None  # Reset the ordered price after closing

    # Check for SMA trend conditions with ATR confirmation
    if abs(sma5 - sma8) / sma8 > THRESHOLD_PERCENTAGE and abs(sma8 - sma13) / sma13 > THRESHOLD_PERCENTAGE:
        if sma5 > sma8 > sma13 and atr > 0.005:  # Buy signal criteria
            if order_type == OrderType.SHORT:
                result = closeAllPosition(symbol=SYMBOL)
                if result == 200:
                    order_type = OrderType.NONE
                    logger.info(f"Closed SHORT order id = {last_order_id} at {datetime.now().strftime(
                        '%Y-%m-%d %H:%M:%S')}")
                    count_of_short = 1  # Reset counter for short trades
            if count_of_long == 1:  # Ensure only one long trade at a time
                # Confirmation: Last close price should be above SMA5
                if close_prices[-1] > sma5:
                    # setLeverage(symbol=SYMBOL,side="LONG",leverage=5)
                    amount = trade_amount_calculate(symbol=SYMBOL)
                    last_order_id, order_type, quantity = open_long(
                        symbol=SYMBOL, quantity=amount)
                    logger.info(f"LONG opened in this price : {close_prices[-1]} AMOUNT :{amount} at {datetime.now().strftime(
                        '%Y-%m-%d %H:%M:%S')}")
                    count_of_long += 1
                    # Store the entry price for profit calculation
                    ordered_price = close_prices[-1]
                    last_trade_amount = quantity
                    # setLeverage(symbol=SYMBOL,side="LONG",leverage=1)

        elif sma5 < sma8 < sma13 and atr > 0.005:  # Sell signal criteria
            if order_type == OrderType.LONG:
                result = closeAllPosition(symbol=SYMBOL)
                if result == 200:
                    order_type = OrderType.NONE
                    logger.info(f"Closed LONG order id = {last_order_id} at {datetime.now().strftime(
                        '%Y-%m-%d %H:%M:%S')}")
                    count_of_long = 1  # Reset counter for long trades
            if count_of_short == 1:  # Ensure only one short trade at a time
                # Confirmation: Last close price should be below SMA5
                if close_prices[-1] < sma5:
                    # setLeverage(symbol=SYMBOL,side="SHORT",leverage=5)
                    amount = trade_amount_calculate(symbol=SYMBOL)
                    last_order_id, order_type, quantity = open_short(
                        symbol=SYMBOL, quantity=amount)
                    logger.info(f"SHORT opened in this price : {close_prices[-1]} AMOUNT :{amount} at {datetime.now().strftime(
                        '%Y-%m-%d %H:%M:%S')}")
                    count_of_short += 1
                    # Store the entry price for profit calculation
                    ordered_price = close_prices[-1]
                    last_trade_amount = quantity
                    # setLeverage(symbol=SYMBOL,side="SHORT",leverage=1)
        else:
            logger.info("No buy or sell signal met the criteria.")
    else:
        logger.info(
            "No significant movement detected. Waiting for better conditions.")


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
                    make_trade_decision(sma5, sma8, sma13,
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


async def back_test(symbol, interval="1m"):
    # Function to calculate moving averages for backtesting
    def calculate_all_smas(klines):
        close_prices = [float(kline['close']) for kline in klines]
        sma5 = [sum(close_prices[i:i+5]) /
                5 for i in range(len(close_prices)-5+1)]
        sma8 = [sum(close_prices[i:i+8]) /
                8 for i in range(len(close_prices)-8+1)]
        sma13 = [sum(close_prices[i:i+13]) /
                 13 for i in range(len(close_prices)-13+1)]
        return close_prices, sma5, sma8, sma13

    last_min = 180
    past_time_ms = get_server_time() - (last_min * 60 * 1000)
    klines = get_kline(symbol=symbol, interval=interval,
                       limit=last_min, start=past_time_ms)
    klines.raise_for_status()
    klines = klines.json().get('data', [])
    klines.reverse()

    if not klines:
        logger.error("No data available for backtesting.")
        return

    close_prices, sma5, sma8, sma13 = calculate_all_smas(klines)

    # Plotting the results
    plt.figure(figsize=(14, 7))
    plt.plot(close_prices, label="Close Price", color="black")

    # Align SMA arrays to the corresponding close price indices
    plt.plot(range(4, len(close_prices)), sma5, label="SMA5", color="brown")
    plt.plot(range(7, len(close_prices)), sma8, label="SMA8", color="orange")
    plt.plot(range(12, len(close_prices)), sma13, label="SMA13", color="blue")

    # Formatting the plot
    plt.title(f"{symbol} Backtest - Close Prices and SMAs")
    plt.xlabel("Time (most recent to older)")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.show()


# Run the async loop
if __name__ == "__main__":
    try:
        # asyncio.run(back_test(symbol=SYMBOL))
        asyncio.run(main())

    except KeyboardInterrupt:
        logger.info("Trading bot stopped manually.")
    except Exception as e:
        logger.exception("Critical error. Bot stopped.")


# Function for backtesting and plotting the results
