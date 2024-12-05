import time
import logging
from datetime import datetime
from binance.client import Client
from binance.exceptions import BinanceAPIException
import numpy as np

# Set up logging
logging.basicConfig(
    filename='scalping_bot.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

# Initialize Binance client
API_KEY = 'PPRQJTGEPGtwNAg7C6OkuLvJgxtOlQTUdobh51g0or62iX4yP1y5QlmDzdcEW9Gh'
API_SECRET = 'NVpSuN9KxAkSMiS1teAHRbcFrajQRmXUVFBcfSDBfaHjwNvwi4mfcb1sCd94gIs8'
client = Client(API_KEY, API_SECRET)

def calculate_sma(data, window):
    """Calculate Simple Moving Average."""
    return np.mean(data[-window:])

def calculate_rsi(prices, window=14):
    """Calculate Relative Strength Index."""
    deltas = np.diff(prices)
    seed = deltas[:window]
    up = seed[seed >= 0].sum() / window
    down = -seed[seed < 0].sum() / window
    rs = up / down if down != 0 else 0
    rsi = np.zeros_like(prices)
    rsi[:window] = 50  # Initialize RSI to 50

    for i in range(window, len(prices)):
        delta = deltas[i - 1] if i > 0 else 0
        gain = max(delta, 0)
        loss = -min(delta, 0)
        up = (up * (window - 1) + gain) / window
        down = (down * (window - 1) + loss) / window
        rs = up / down if down != 0 else 0
        rsi[i] = 100 - (100 / (1 + rs))

    return rsi

def get_klines(symbol, interval, limit=100):
    """Fetch candlestick data."""
    try:
        klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
        return [float(kline[4]) for kline in klines]  # Closing prices
    except BinanceAPIException as e:
        logging.error(f"Binance API error: {e}")
        return []

def scalping_bot(symbol, interval='1m'):
    """Scalping bot using RSI and SMA strategy."""
    while True:
        try:
            # Fetch data
            prices = get_klines(symbol, interval)
            if len(prices) < 14:
                logging.warning("Not enough data to calculate indicators.")
                time.sleep(60)
                continue

            # Calculate indicators
            rsi = calculate_rsi(prices, 14)
            sma_short = calculate_sma(prices, 5)
            sma_long = calculate_sma(prices, 13)

            # Get current price and signal
            current_price = prices[-1]
            last_rsi = rsi[-1]
            signal = None

            if sma_short > sma_long and last_rsi < 30:
                signal = "BUY"
            elif sma_short < sma_long and last_rsi > 70:
                signal = "SELL"

            # Log the signal
            if signal:
                logging.info(f"Signal: {signal}, Price: {current_price}, RSI: {last_rsi:.2f}")
                print(f"{datetime.now()} - Signal: {signal}, Price: {current_price}, RSI: {last_rsi:.2f}")

            # Wait before next iteration
            time.sleep(60)  # 5 minutes

        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            time.sleep(60)

# Run the bot
if __name__ == "__main__":
    symbol = 'BTCUSDT' 
    scalping_bot(symbol)
