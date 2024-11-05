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
INTERVAL = "5m"
MIN = 5
LIMIT  = 100 
AMOUNT_USDT = 100  # USDT

THRESHOLD_PERCENTAGE = 0.005  # 0.5% threshold for SMA difference
ATR_PERIOD = 14  # ATR period

order_type = OrderType.NONE
last_order_id = None
count_of_long = 1
count_of_short = 1

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

# Function to make trading decisions with threshold and ATR checks
def make_trade_decision(sma5, sma8, sma13, klines, close_prices):
    global order_type, last_order_id, count_of_long, count_of_short
    
    atr = calculate_atr(klines)
 
    # Check if there is a significant trend with added conditions for confirmation
    if abs(sma5 - sma8) / sma8 > THRESHOLD_PERCENTAGE and abs(sma8 - sma13) / sma13 > THRESHOLD_PERCENTAGE:
        if sma5 > sma8 > sma13 and close_prices[-1] > sma5 and atr > 0.01:
            if order_type == OrderType.SHORT:
                result = close_short(symbol=SYMBOL, quantity=0.015)
                if result == 0:
                    order_type = OrderType.NONE
                    logger.info(f"Closed SHORT order id = {last_order_id}")
                    count_of_short = 1
            if count_of_long == 1:
                # Add logic for confirmation (e.g., multiple consecutive closes above SMA)
                if all(cp > sma5 for cp in close_prices[-3:]):  # Last 3 close prices above SMA5
                    last_order_id, order_type = open_long(symbol=SYMBOL, quantity=0.015)
                    logger.info(f"LONG opened at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                count_of_long += 1

        elif sma5 < sma8 < sma13 and close_prices[-1] < sma5 and atr > 0.01:
            if order_type == OrderType.LONG:
                result = close_long(symbol=SYMBOL, quantity=0.015)
                if result == 0:
                    order_type = OrderType.NONE
                    logger.info(f"Closed LONG order id = {last_order_id}")
                    count_of_long = 1
            if count_of_short == 1:
                # Add logic for confirmation (e.g., multiple consecutive closes below SMA)
                if all(cp < sma5 for cp in close_prices[-3:]):  # Last 3 close prices below SMA5
                    last_order_id, order_type = open_short(symbol=SYMBOL, quantity=0.015)
                    logger.info(f"SHORT opened at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                count_of_short += 1
    else:
        logger.info("No significant movement detected. Waiting for better conditions.")

# Main async loop to fetch data and process trading
async def main():
    async with aiohttp.ClientSession() as session:
        while True:
            try:
                past_time_ms = int(time.time() * 1000) - (LIMIT * MIN * 60 * 1000)
                klines = get_kline(symbol=SYMBOL, interval=INTERVAL, limit=LIMIT, start=past_time_ms)
                klines.raise_for_status()
                klines = klines.json().get('data', [])
                last = last_price(symbol=SYMBOL)
                klines.reverse()
                if klines:
                    sma5, sma8, sma13, close_prices = calculate_moving_averages(klines)
                    logger.info(f"PRICES: {last} SMA5: {sma5}, SMA8: {sma8}, SMA13: {sma13}")
                    make_trade_decision(sma5, sma8, sma13, klines,close_prices)
                else:
                    logger.warning("No Kline data received. Skipping this iteration.")
                await asyncio.sleep(MIN * 60)
            except asyncio.CancelledError:
                logger.warning("Task was cancelled. Exiting.")
                break
            except Exception as e:
                logger.exception("Unexpected error occurred in main loop. Continuing...")

# Run the async loop
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Trading bot stopped manually.")
    except Exception as e:
        logger.exception("Critical error. Bot stopped.")



# Function for backtesting and plotting the results
async def back_test(symbol, interval="1m"):
    # Function to calculate moving averages for backtesting
    def calculate_all_smas(klines):
        close_prices = [float(kline['close']) for kline in klines]
        sma5 = [sum(close_prices[i:i+5])/5 for i in range(len(close_prices)-5+1)]
        sma8 = [sum(close_prices[i:i+8])/8 for i in range(len(close_prices)-8+1)]
        sma13 = [sum(close_prices[i:i+13])/13 for i in range(len(close_prices)-13+1)]
        return close_prices, sma5, sma8, sma13
    

    last_min = 180
    past_time_ms = get_server_time() - (last_min * 60 * 1000)
    klines =  get_kline(symbol=symbol, interval=interval, limit=last_min, start=past_time_ms)
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

