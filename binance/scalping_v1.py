import time
import logging
from datetime import datetime
from binance.client import Client
from binance.exceptions import BinanceAPIException
import numpy as np
import matplotlib.pyplot as plt


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
        # Ensure klines contains the expected data structure
        return [(float(kline[4]), int(kline[0])) for kline in klines]  # Closing price and timestamp
    except Exception as e:
        logging.error(f"Error fetching klines: {e}")
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
def back_test(symbol, interval='1m', limit=100):
    """Backtest strategy using historical data."""
    try:
        # Fetch historical data
        data = get_klines(symbol, interval, limit)
        if not data or len(data) < 14:
            logging.warning("Not enough data to perform backtest.")
            return
        
        prices = [item[0] for item in data]
        timestamps = [item[1] for item in data]

        # Calculate indicators
        rsi = calculate_rsi(prices, 14)
        sma_short = [calculate_sma(prices[max(0, i - 4):i + 1], 5) for i in range(len(prices))]
        sma_long = [calculate_sma(prices[max(0, i - 12):i + 1], 13) for i in range(len(prices))]

        # Find signals
        for i in range(14, len(prices)):
            if sma_short[i] is None or sma_long[i] is None:
                continue

            signal = None
            if sma_short[i] > sma_long[i] and rsi[i] < 40:
                signal = "BUY"
            elif sma_short[i] < sma_long[i] and rsi[i] > 60:
                signal = "SELL"

            if signal:
                price = prices[i]
                time_of_signal = datetime.fromtimestamp(timestamps[i] / 1000)
                log_message = f"{time_of_signal} - Signal: {signal}, Price: {price:.2f}, RSI: {rsi[i]:.2f}"
                logging.info(log_message)
                print(log_message)

    except Exception as e:
        logging.error(f"Unexpected error during backtest: {e}")
def fetch_and_plot_klines(symbol, interval, limit):
    """Fetch kline data from Binance and plot close prices."""
    try:
        # Fetch klines
        klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
        
        # Extract close prices and timestamps
        close_prices = [float(kline[4]) for kline in klines]
        timestamps = [datetime.fromtimestamp(int(kline[0]) / 1000) for kline in klines]
        
        # Plot close prices
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, close_prices, label='Close Price', color='blue')
        plt.title(f"{symbol} Close Prices ({interval} Interval)")
        plt.xlabel("Time")
        plt.ylabel("Close Price")
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error fetching or plotting klines: {e}")
# Run the bot
if __name__ == "__main__":
    symbol = 'BTCUSDT'  # Change to your desired trading pair
    interval = '5m'  # Change to desired interval ('1m', '5m', etc.)
    limit = 300  # Number of historical candles to fetch
    back_test(symbol, interval, limit)
    # fetch_and_plot_klines(symbol="BTCUSDT", interval=interval, limit=limit)

    # scalping_bot(symbol)
