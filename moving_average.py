import asyncio
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
    if sma5 > sma8 > sma13:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f'Buy signal generated. SMAs indicate a strong upward trend.in this time {current_time}')
        # Here, you would add code to place a buy order via the API
    elif sma5 < sma8 < sma13:
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

# Run the async loop
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Trading bot stopped manually.")
    except Exception as e:
        logger.exception("Critical error. Bot stopped.")
