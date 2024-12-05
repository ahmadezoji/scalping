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


def scalping_bot(symbol, interval='1m', usdt_amount=200):
    """Scalping bot using RSI and SMA strategy with real orders."""
    global current_position
    current_position = None  # None, 'LONG', or 'SHORT'
    open_price = None

    while True:
        try:
            # Fetch data
            klines = get_klines(symbol, interval)
            if len(klines) < 14:
                logging.warning("Not enough data to calculate indicators.")
                time.sleep(60)
                continue

            prices = [kline[0] for kline in klines]  # Extract close prices
            timestamps = [datetime.fromtimestamp(kline[1] / 1000).strftime('%Y-%m-%d %H:%M:%S') for kline in klines]

            # Calculate indicators
            rsi = calculate_rsi(prices, 14)
            sma_short = calculate_sma(prices, 5)
            sma_long = calculate_sma(prices, 13)

            # Log calculated indicators
            logging.info(f"Latest Data: RSI: {rsi[-1]:.2f}, SMA Short (5): {sma_short:.2f}, SMA Long (13): {sma_long:.2f}, Time: {timestamps[-1]}")
            print(f"{timestamps[-1]} - RSI: {rsi[-1]:.2f}, SMA Short (5): {sma_short:.2f}, SMA Long (13): {sma_long:.2f}")

            # Generate trading signal
            last_rsi = rsi[-1]
            current_price = prices[-1]
            signal = None

            if sma_short > sma_long and last_rsi < 37:
                signal = "BUY"
            elif sma_short < sma_long and last_rsi > 63:
                signal = "SELL"

            # Process the signal
            if signal == "BUY" and current_position != "LONG":
                # Close SHORT position if active
                if current_position == "SHORT":
                    realized_pnl = (open_price - current_price) / open_price * usdt_amount
                    logging.info(f"Closing SHORT at {current_price:.2f}, Realized PnL: {realized_pnl:.2f}")
                    print(f"{timestamps[-1]} - Closing SHORT at {current_price:.2f}, Realized PnL: {realized_pnl:.2f}")

                # Open a LONG position
                open_price = current_price
                execute_trade(symbol, "BUY", usdt_amount)
                current_position = "LONG"
                logging.info(f"Opening LONG at {current_price:.2f}")
                print(f"{timestamps[-1]} - Opening LONG at {current_price:.2f}")

            elif signal == "SELL" and current_position != "SHORT":
                # Close LONG position if active
                if current_position == "LONG":
                    realized_pnl = (current_price - open_price) / open_price * usdt_amount
                    logging.info(f"Closing LONG at {current_price:.2f}, Realized PnL: {realized_pnl:.2f}")
                    print(f"{timestamps[-1]} - Closing LONG at {current_price:.2f}, Realized PnL: {realized_pnl:.2f}")

                # Open a SHORT position
                open_price = current_price
                execute_trade(symbol, "SELL", usdt_amount)
                current_position = "SHORT"
                logging.info(f"Opening SHORT at {current_price:.2f}")
                print(f"{timestamps[-1]} - Opening SHORT at {current_price:.2f}")

            # Calculate unrealized PnL for the current open position
            if current_position == "LONG":
                unrealized_pnl = (current_price - open_price) / open_price * usdt_amount
                logging.info(f"LONG Unrealized PnL: {unrealized_pnl:.2f}")
                print(f"{timestamps[-1]} - LONG Unrealized PnL: {unrealized_pnl:.2f}")
            elif current_position == "SHORT":
                unrealized_pnl = (open_price - current_price) / open_price * usdt_amount
                logging.info(f"SHORT Unrealized PnL: {unrealized_pnl:.2f}")
                print(f"{timestamps[-1]} - SHORT Unrealized PnL: {unrealized_pnl:.2f}")

            # Wait before the next iteration
            time.sleep(300)

        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            print(f"Error: {e}")
            time.sleep(60)
def place_order(symbol, side, quantity):
    """Place a buy or sell order on Binance."""
    try:
        order = client.create_order(
            symbol=symbol,
            side=side,
            type='MARKET',
            quantity=quantity
        )
        logging.info(f"Order placed: {side} {quantity} of {symbol}. Order ID: {order['orderId']}")
        print(f"Order placed: {side} {quantity} of {symbol}.")
        return order
    except BinanceAPIException as e:
        logging.error(f"Binance API Exception: {e}")
        print(f"Error: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        print(f"Error: {e}")
        return None

def execute_trade(symbol, signal, usdt_amount=10):
    """Execute a trade based on the signal."""
    global current_position

    # Determine quantity of BTC to trade (using the last market price)
    try:
        ticker = client.get_ticker(symbol=symbol)
        last_price = float(ticker['lastPrice'])
        btc_quantity = round(usdt_amount / last_price, 6)  # Adjust precision as needed

        if signal == "BUY":
            if current_position == "BUY":
                logging.info("Already in a BUY position, skipping new BUY signal.")
                print("Already in a BUY position, skipping new BUY signal.")
            else:
                order = place_order(symbol, side="BUY", quantity=btc_quantity)
                if order:
                    current_position = "BUY"

        elif signal == "SELL":
            if current_position == "BUY":
                # Close the long position
                order = place_order(symbol, side="SELL", quantity=btc_quantity)
                if order:
                    current_position = None
            else:
                logging.info("No active BUY position to close, skipping SELL signal.")
                print("No active BUY position to close, skipping SELL signal.")

    except BinanceAPIException as e:
        logging.error(f"Binance API Exception: {e}")
        print(f"Error: {e}")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        print(f"Error: {e}")
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

def back_test_via_pnl(symbol, interval='1m', limit=300, usdt_amount=200):
    """Backtest scalping strategy with dummy trading."""
    # Simulated trading state
    current_position = None  # None, 'LONG', or 'SHORT'
    open_price = None
    total_pnl = 0  # Total PnL across all trades

    # Fetch historical data
    klines = get_klines(symbol, interval, limit)
    if len(klines) < 14:
        logging.warning("Not enough data to perform backtest.")
        return

    prices = [kline[0] for kline in klines]  # Extract close prices
    timestamps = [datetime.fromtimestamp(kline[1] / 1000).strftime('%Y-%m-%d %H:%M:%S') for kline in klines]

    # Calculate indicators
    rsi = calculate_rsi(prices, 14)
    sma_short = [calculate_sma(prices[max(0, i - 4):i + 1], 5) for i in range(len(prices))]
    sma_long = [calculate_sma(prices[max(0, i - 12):i + 1], 13) for i in range(len(prices))]

    # Iterate through data to simulate trades
    for i in range(14, len(prices)):
        last_rsi = rsi[i]
        current_price = prices[i]
        signal = None

        # Generate trading signal
        if sma_short[i] > sma_long[i] and last_rsi < 37:
            signal = "BUY"
        elif sma_short[i] < sma_long[i] and last_rsi > 63:
            signal = "SELL"

        # Process the signal
        if signal == "BUY" and current_position != "LONG":
            # Close existing position if SHORT
            if current_position == "SHORT":
                realized_pnl = (open_price - current_price) / open_price * usdt_amount
                total_pnl += realized_pnl
                logging.info(f"Closing SHORT at {current_price:.2f}, Realized PnL: {realized_pnl:.2f}")
                print(f"{timestamps[i]} - Closing SHORT at {current_price:.2f}, Realized PnL: {realized_pnl:.2f}")

            # Open a new LONG position
            current_position = "LONG"
            open_price = current_price
            logging.info(f"Opening LONG at {current_price:.2f}")
            print(f"{timestamps[i]} - Opening LONG at {current_price:.2f}")

        elif signal == "SELL" and current_position != "SHORT":
            # Close existing position if LONG
            if current_position == "LONG":
                realized_pnl = (current_price - open_price) / open_price * usdt_amount
                total_pnl += realized_pnl
                logging.info(f"Closing LONG at {current_price:.2f}, Realized PnL: {realized_pnl:.2f}")
                print(f"{timestamps[i]} - Closing LONG at {current_price:.2f}, Realized PnL: {realized_pnl:.2f}")

            # Open a new SHORT position
            current_position = "SHORT"
            open_price = current_price
            logging.info(f"Opening SHORT at {current_price:.2f}")
            print(f"{timestamps[i]} - Opening SHORT at {current_price:.2f}")

        # Calculate unrealized PnL for the current open position
        if current_position == "LONG":
            unrealized_pnl = (current_price - open_price) / open_price * usdt_amount
            logging.info(f"LONG Unrealized PnL: {unrealized_pnl:.2f}")
            print(f"{timestamps[i]} - LONG Unrealized PnL: {unrealized_pnl:.2f}")
        elif current_position == "SHORT":
            unrealized_pnl = (open_price - current_price) / open_price * usdt_amount
            logging.info(f"SHORT Unrealized PnL: {unrealized_pnl:.2f}")
            print(f"{timestamps[i]} - SHORT Unrealized PnL: {unrealized_pnl:.2f}")

    # Log total PnL after backtest
    logging.info(f"Backtest Complete. Total PnL: {total_pnl:.2f}")
    print(f"Backtest Complete. Total PnL: {total_pnl:.2f}")

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
    usdt_amount = 9
    # limit = 300  # Number of historical candles to fetch
    # back_test(symbol, interval, limit)
    # fetch_and_plot_klines(symbol="BTCUSDT", interval=interval, limit=limit)

    scalping_bot(symbol,interval,usdt_amount)
    # back_test_via_pnl(symbol=symbol,interval=interval)
