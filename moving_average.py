import asyncio
import matplotlib.pyplot as plt
import aiohttp
import logging
from bingx import *
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


KLINES_ENDPOINT = "/v1/klines"
SYMBOL = "BTC-USDT"
INTERVAL = "1m"
LIMIT = 50  # Fetch 50 data points


# Function to calculate moving averages
def calculate_moving_averages(klines):
    close_prices = [float(kline['close']) for kline in klines]  # Assuming the 5th item is the closing price
    sma5 = sum(close_prices[-5:]) / 5
    sma8 = sum(close_prices[-8:]) / 8
    sma13 = sum(close_prices[-13:]) / 13
    return sma5, sma8, sma13

# Function to make trading decisions
def make_trade_decision(sma5, sma8, sma13):
    # if sma5 > sma8 > sma13:
    if sma5  > sma13:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f'Buy signal generated. SMAs indicate a strong upward trend.in this time {current_time}')
        # Here, you would add code to place a buy order via the API
    # elif sma5 < sma8 < sma13:
    elif sma5  < sma13:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f'Sell signal generated. SMAs indicate a strong downward trend. {current_time}')
        # Here, you would add code to place a sell order via the API
    else:
        logger.info("No clear trend. Waiting for better conditions.")

# Main async loop to keep fetching data and processing
async def main():
    async with aiohttp.ClientSession() as session:
        while True:
            try:
                past_time_ms = int(time.time() * 1000) - (LIMIT * 60 * 1000)  # 50 minutes ago
                # Calculate the past time in UTC for 50 minutes ago
                # past_time_utc = datetime.utcnow() - timedelta(minutes=50)
                # past_time_ms = int(past_time_utc.timestamp() * 1000)  # Convert to milliseconds


                klines = get_kline(symbol=SYMBOL, interval=INTERVAL,limit=LIMIT,start=past_time_ms)
                klines.raise_for_status()
                klines = klines.json().get('data', [])
                last = last_price(symbol=SYMBOL)
                klines.reverse()
                if klines:
                    sma5, sma8, sma13 = calculate_moving_averages(klines)
                   
                    logger.info(f"PRICES:{last} SMA5: {sma5}, SMA8: {sma8}, SMA13: {sma13}")
                    make_trade_decision(sma5, sma8, sma13)
                else:
                    logger.warning("No Kline data received. Skipping this iteration.")
                
                await asyncio.sleep(60)  # Wait for 1 minute before fetching new data
            except asyncio.CancelledError:
                logger.warning("Task was cancelled. Exiting.")
                break
            except Exception as e:
                logger.exception("Unexpected error occurred in main loop. Continuing...")



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


# Run the async loop
if __name__ == "__main__":
    try:
        # asyncio.run(back_test(symbol=SYMBOL))
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Trading bot stopped manually.")
    except Exception as e:
        logger.exception("Critical error. Bot stopped.")
