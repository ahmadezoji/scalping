import pandas as pd
import numpy as np
import ta
import time
from bingx import *
import logging
import aiohttp
import asyncio
from datetime import datetime, timedelta


SYMBOL = "BTC-USDT"
INTERVAL = "1m"  # Changed to 1-minute timeframe
MIN = 1  # Adjust sleep time accordingly (1 minute)
LIMIT = 100
AMOUNT_USDT = 3000  # USDT




# Existing setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def rsi(df, period=14):
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ema_up = up.ewm(com=period - 1, adjust=False).mean()
    ema_down = down.ewm(com=period - 1, adjust=False).mean()
    rs = ema_up / ema_down

    rsi = 100 - (100 / (1 + rs))

    return rsi

async def main():
    async with aiohttp.ClientSession() as session:
        while True:
            try:
                # Fetch latest OHLC data using your broker's API (replace get_kline)
                past_time_ms = get_server_time() - (LIMIT * MIN * 60 * 1000)
                klines = get_kline(symbol=SYMBOL, interval=INTERVAL, limit=LIMIT, start=past_time_ms)

                klines.raise_for_status()
                klines = klines.json().get('data', [])
                if len(klines) == 0:
                    return
                klines.reverse()
                

                data = pd.DataFrame(klines)
                data.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Datetime']  # Assuming these columns exist
                data['Datetime'] = pd.to_datetime(data['Datetime'], utc=True)  # Convert timestamps

                data['Close'] = pd.to_numeric(data['Close'], errors='coerce')  # Convert 'Close' column to numeric


                # Calculate indicators
                data['RSI']  = rsi(data, period=14)
                data['SMA_fast'] = data['Close'].rolling(window=10).mean()
                data['SMA_slow'] = data['Close'].rolling(window=20).mean()

                logger.info(f"PRICES: {data['Close'].iloc[-1]} RSI :{data['RSI'].iloc[-1]}SMA_fast : { data['SMA_fast'].iloc[-1]} SMA_slow : {data['SMA_slow'].iloc[-1] }  at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ")

                # Define trading signals (customize as needed)
                buy_signal = (data['RSI'] < 40) & (data['SMA_fast'] > data['SMA_slow'])
                sell_signal = (data['RSI'] > 60) & (data['SMA_fast'] < data['SMA_slow'])

                # Execute trades (replace with your broker's API)
                if buy_signal.iloc[-1]:
                    print(f"Buy signal triggered at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}!")
                    # Place buy order using your broker's API
                elif sell_signal.iloc[-1]:
                    print(f"Sell signal triggered at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} !")
                    # Place sell order using your broker's API

                # Wait for the next candle interval
                time.sleep(int(MIN) * 60)

            except asyncio.CancelledError:
                    logger.warning("Task was cancelled. Exiting.")
                    break
            except Exception as e:
                logger.exception(
                    "Unexpected error occurred in main loop. Continuing...")


     

# Run the async loop
if __name__ == "__main__":
    try:
        asyncio.run(main())

    except KeyboardInterrupt:
        logger.info("Trading bot stopped manually.")
    except Exception as e:
        logger.exception("Critical error. Bot stopped.")
