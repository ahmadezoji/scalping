import time
import logging
from datetime import datetime, timedelta
from binance.client import Client
from binance.exceptions import BinanceAPIException
import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal
import threading


# Set up logging
logging.basicConfig(
    filename='scalping_bot.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

# Initialize Binance client
API_KEY = 'OCfUCqdr9E3GeEzHy1zADhFTP4UHwh50nfJtS7m07EVOpFeVzMxsF6EnxUFQHrUK'
API_SECRET = 'VtXuwNQNq5dOjJefjjKIrxwjdeIo8wRt0FRjgfASqJZvNPChSRrvFDaV12G2COwH'
client = Client(API_KEY, API_SECRET)
# client.FUTURES_URL = 'https://testnet.binancefuture.com/fapi'


def calculate_sma(data, window):
    """Calculate Simple Moving Average."""
    if len(data) < window:  # Not enough data to calculate SMA
        return np.nan  # or you can return 0.0, depending on preference
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

def get_klines(symbol, interval, limit=100, start_date=None):
    try:

        if start_date is None:
           start_date = datetime.now()

        start_time = int(start_date.timestamp() * 1000)
        
        klines = client.get_klines(
            symbol=symbol, 
            interval=interval, 
            limit=limit, 
            startTime=start_time
        )

        # Ensure klines contains the expected data structure (Closing price and timestamp)
        return [(float(kline[4]), int(kline[0])) for kline in klines]
    except Exception as e:
        logging.error(f"Error fetching klines: {e}")
        return []


def check_sl_tp(position, symbol, tp_price, sl_price):
    """Thread function to check if SL or TP is hit."""
    global current_position, open_trade_thread, current_quantity
    while current_position:
        try:
            # Fetch the current price
            klines = get_klines(symbol, '1m', limit=1)
            current_price = klines[-1][0]

            if position == "LONG":
                if current_price >= tp_price:  # Take Profit
                    log_trade(position, 'CLOSE (TP)', current_price)
                    close_futures_position(symbol, 'LONG', current_quantity)
                    current_position = None
                    break
                elif current_price <= sl_price:  # Stop Loss
                    log_trade(position, 'CLOSE (SL)', current_price)
                    close_futures_position(symbol, 'LONG', current_quantity)
                    current_position = None
                    break

            elif position == "SHORT":
                if current_price <= tp_price:  # Take Profit
                    log_trade(position, 'CLOSE (TP)', current_price)
                    close_futures_position(symbol, 'SHORT', current_quantity)
                    current_position = None
                    break
                elif current_price >= sl_price:  # Stop Loss
                    log_trade(position, 'CLOSE (SL)', current_price)
                    close_futures_position(symbol, 'SHORT', current_quantity)
                    current_position = None
                    break

            time.sleep(10)  # Check every 10 seconds

        except Exception as e:
            logging.error(f"Error in SL/TP thread: {e}")
            break


def scalping_bot(symbol, interval='1m', limit=500, usdt_amount=200, tp_percentage=0.02, sl_percentage=0.01, rsi_buy_threshold=40, rsi_sell_threshold=60,):
    """Scalping bot using RSI and SMA strategy with real orders."""
    global current_position, current_quantity, open_trade_thread
    current_position = None  # None, 'LONG', or 'SHORT'
    current_quantity = 0.0
    open_price = None

    while True:
        try:
            # Fetch data
            klines = get_klines(symbol, interval, limit=limit)
            if len(klines) < 14:
                logging.warning("Not enough data to calculate indicators.")
                time.sleep(60)
                continue

            prices = [kline[0] for kline in klines]  # Extract close prices
            timestamps = [datetime.fromtimestamp(
                kline[1] / 1000).strftime('%Y-%m-%d %H:%M:%S') for kline in klines]

            # Calculate indicators
            rsi = calculate_rsi(prices, 14)
            sma_short = [calculate_sma(
                prices[max(0, i - 4):i + 1], 5) if i >= 4 else np.nan for i in range(len(prices))]
            sma_long = [calculate_sma(prices[max(
                0, i - 12):i + 1], 13) if i >= 12 else np.nan for i in range(len(prices))]

            print(f"{timestamps[-1]} - RSI: {rsi:.2f}, SMA Short (5): {
                  sma_short:.2f}, SMA Long (13): {sma_long:.2f}")

            # Generate trading signal
            signal = None
            if sma_short > sma_long and rsi < rsi_buy_threshold:
                signal = "BUY"
            elif sma_short < sma_long and rsi > rsi_sell_threshold:
                signal = "SELL"

            # Process the signal
            if signal == "BUY" and current_position != "LONG":
                current_position = "LONG"
                open_price = prices[-1]
                execute_trade(symbol, "BUY", usdt_amount)

                sl_tp = calculate_sl_tp(
                    open_price, 'LONG', interval, tp_percentage, sl_percentage
                )
                open_trade_thread = threading.Thread(
                    target=check_sl_tp, args=(
                        open_price, 'LONG', symbol, sl_tp['TP'], sl_tp['SL'])
                )
                open_trade_thread.start()
                log_trade('LONG', 'OPEN', open_price, 0.0, timestamps[-1])

            elif signal == "SELL" and current_position != "SHORT":
                current_position = "SHORT"
                open_price = prices[-1]
                execute_trade(symbol, "SELL", usdt_amount)

                sl_tp = calculate_sl_tp(
                    open_price, 'SHORT', interval, tp_percentage, sl_percentage)
                open_trade_thread = threading.Thread(
                    target=check_sl_tp, args=(
                        open_price, 'SHORT', symbol, sl_tp['TP'], sl_tp['SL'])
                )
                open_trade_thread.start()
                log_trade('SHORT', 'OPEN', open_price, 0.0, timestamps[-1])

            time.sleep(5*60)

        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            print(f"Error: {e}")
            time.sleep(60)


def close_futures_position(symbol, position, quantity):
    try:
        if position == 'LONG':
            order = client.futures_create_order(
                symbol=symbol,
                side="SELL",
                type='MARKET',
                quantity=quantity
            )
        elif position == 'SHORT':
            order = client.futures_create_order(
                symbol=symbol,
                side="BUY",
                type='MARKET',
                quantity=quantity
            )

        print(f"Order Closed successfully: {order}")
        return order

    except Exception as e:
        print(f"Error closing position: {e}")
        return None


def place_order(symbol, side, usdt_amount):
    """Place a buy or sell order on Binance Futures."""
    global current_quantity
    try:
        # Get the current price
        ticker = client.futures_symbol_ticker(symbol=symbol)
        last_price = float(ticker['price'])

        # Get LOT_SIZE and NOTIONAL filters
        min_qty, max_qty, step_size = get_lot_size(symbol)
        min_notional = get_notional_min(symbol)

        # Calculate quantity and adjust to the nearest step size
        raw_quantity = usdt_amount / last_price
        quantity = adjust_quantity(raw_quantity, step_size)

        # Format quantity to the allowed precision
        quantity = f"{quantity:.{
            len(str(step_size).split('.')[-1])}f}".rstrip('0').rstrip('.')

        # Calculate notional value
        notional_value = float(quantity) * last_price

        # Ensure quantity is within the allowed range
        if float(quantity) < min_qty or float(quantity) > max_qty:
            raise ValueError(
                f"Quantity {quantity} is out of range: [{min_qty}, {max_qty}]")

        # Ensure notional value meets the minimum
        if notional_value < min_notional:
            raise ValueError(f"Notional value {
                             notional_value} is below the minimum of {min_notional}.")

        # Place the order
        order = client.futures_create_order(
            symbol=symbol,
            side=side,
            type='MARKET',
            quantity=quantity
        )
        current_quantity = quantity
        logging.info(f"Order placed: {side} {quantity} of {
                     symbol}. Order ID: {order['orderId']}")
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
    global current_position, current_quantity

    # Determine quantity of BTC to trade (using the last market price)
    try:
        # ticker = client.get_ticker(symbol=symbol)
        # last_price = float(ticker['lastPrice'])
        # btc_quantity = round(usdt_amount / last_price, 6)  # Adjust precision as needed

        if signal == "BUY":
            if current_position == "BUY":
                logging.info(
                    "Already in a BUY position, skipping new BUY signal.")
                print("Already in a BUY position, skipping new BUY signal.")
            else:
                order = place_order(symbol, side="BUY", quantity=usdt_amount)
                if order:
                    current_position = "BUY"

        elif signal == "SELL":
            if current_position == "BUY":
                # Close the long position
                order = place_order(symbol, side="SELL", quantity=usdt_amount)
                if order:
                    current_position = None
            else:
                logging.info(
                    "No active BUY position to close, skipping SELL signal.")
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
        sma_short = [calculate_sma(prices[max(0, i - 4):i + 1], 5)
                     for i in range(len(prices))]
        sma_long = [calculate_sma(prices[max(0, i - 12):i + 1], 13)
                    for i in range(len(prices))]

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
                log_message = f"{
                    time_of_signal} - Signal: {signal}, Price: {price:.2f}, RSI: {rsi[i]:.2f}"
                logging.info(log_message)
                print(log_message)

    except Exception as e:
        logging.error(f"Unexpected error during backtest: {e}")


def back_test_via_pnl(symbol, interval='1m', limit=100, usdt_amount=200,
                      rsi_buy_threshold=40, rsi_sell_threshold=60,
                      tp_percentage=0.02, sl_percentage=0.01):
    """Backtest scalping strategy with TP and SL logic included."""

    try:
        current_position = None  # None, 'LONG', or 'SHORT'
        open_price = None
        total_pnl = 0  # Total PnL across all trades
        stop_loss_price = None
        take_profit_price = None

        # Fetch historical data
        start_date = datetime.now() - timedelta(days=1)
        klines = get_klines(symbol, interval, limit,start_date=start_date)
        if len(klines) < 14:
            logging.warning("Not enough data to perform backtest.")
            return

        prices = [kline[0] for kline in klines]  # Extract close prices
        timestamps = [datetime.fromtimestamp(
            kline[1] / 1000).strftime('%Y-%m-%d %H:%M:%S') for kline in klines]

        # Calculate indicators
        rsi = calculate_rsi(prices, 14)
        sma_short = [calculate_sma(
            prices[max(0, i - 4):i + 1], 5) if i >= 4 else np.nan for i in range(len(prices))]
        sma_long = [calculate_sma(prices[max(0, i - 12):i + 1], 13)
                    if i >= 12 else np.nan for i in range(len(prices))]

        for i in range(14, len(prices)):
            last_rsi = rsi[i]
            current_price = prices[i]
            signal = None

            # Generate trading signal
            if sma_short[i] > sma_long[i] and last_rsi < rsi_buy_threshold:
                signal = "BUY"
            elif sma_short[i] < sma_long[i] and last_rsi > rsi_sell_threshold:
                signal = "SELL"

            # Check TP and SL conditions
            if current_position == "LONG":
                if current_price >= take_profit_price:  # Take Profit
                    realized_pnl = (current_price - open_price) / \
                        open_price * usdt_amount
                    total_pnl += realized_pnl
                    log_trade('LONG', 'CLOSE (TP)', current_price,
                              realized_pnl, timestamps[i])
                    current_position = None
                    open_price = None
                    stop_loss_price = None
                    take_profit_price = None
                elif current_price <= stop_loss_price:  # Stop Loss
                    realized_pnl = (current_price - open_price) / \
                        open_price * usdt_amount
                    total_pnl += realized_pnl
                    log_trade('LONG', 'CLOSE (SL)', current_price,
                              realized_pnl, timestamps[i])
                    current_position = None
                    open_price = None
                    stop_loss_price = None
                    take_profit_price = None

            elif current_position == "SHORT":
                if current_price <= take_profit_price:  # Take Profit
                    realized_pnl = (open_price - current_price) / \
                        open_price * usdt_amount
                    total_pnl += realized_pnl
                    log_trade('SHORT', 'CLOSE (TP)', current_price,
                              realized_pnl, timestamps[i])
                    current_position = None
                    open_price = None
                    stop_loss_price = None
                    take_profit_price = None
                elif current_price >= stop_loss_price:  # Stop Loss
                    realized_pnl = (open_price - current_price) / \
                        open_price * usdt_amount
                    total_pnl += realized_pnl
                    log_trade('SHORT', 'CLOSE (SL)', current_price,
                              realized_pnl, timestamps[i])
                    current_position = None
                    open_price = None
                    stop_loss_price = None
                    take_profit_price = None

            # Process the signal
            if signal == "BUY" and current_position != "LONG":
                if current_position == "SHORT":  # Close SHORT position
                    realized_pnl = (open_price - current_price) / \
                        open_price * usdt_amount
                    total_pnl += realized_pnl
                    log_trade('SHORT', 'CLOSE', current_price,
                              realized_pnl, timestamps[i])

                current_position = "LONG"
                open_price = current_price
                sl_tp = calculate_sl_tp(
                    open_price, 'LONG', interval, tp_percentage, sl_percentage)
                stop_loss_price = sl_tp['SL']
                take_profit_price = sl_tp['TP']
                log_trade('LONG', 'OPEN', current_price, 0.0, timestamps[i])

            elif signal == "SELL" and current_position != "SHORT":
                if current_position == "LONG":  # Close LONG position
                    realized_pnl = (current_price - open_price) / \
                        open_price * usdt_amount
                    total_pnl += realized_pnl
                    log_trade('LONG', 'CLOSE', current_price,
                              realized_pnl, timestamps[i])

                current_position = "SHORT"
                open_price = current_price
                sl_tp = calculate_sl_tp(
                    open_price, 'SHORT', interval, tp_percentage, sl_percentage)
                stop_loss_price = sl_tp['SL']
                take_profit_price = sl_tp['TP']
                log_trade('SHORT', 'OPEN', current_price, 0.0, timestamps[i])

            # Calculate and log unrealized PnL
            if current_position:
                unrealized_pnl = calculate_unrealized_pnl(
                    current_position, open_price, current_price, usdt_amount)
                log_unrealized_pnl(
                    current_position, unrealized_pnl, timestamps[i])

        logging.info(f"Backtest Complete. Total PnL: {total_pnl:.2f}")
        print(f"Backtest Complete. Total PnL: {total_pnl:.2f}")
    except Exception as e:
        print(f"Error fetching or plotting klines: {e}")


def log_trade(position, action, price, pnl=None, timestamp=None):
    """Logs trade actions for opening and closing positions."""
    timestamp = timestamp or datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    pnl_info = f", Realized PnL: {pnl:.2f}" if pnl is not None else ""
    message = f"{timestamp} - {action} {position} at {price:.2f}{pnl_info}"

    print(message)
    logging.info(message)


def log_unrealized_pnl(position, unrealized_pnl, timestamp):
    """Logs the unrealized PnL for open positions."""
    if position in ['LONG', 'SHORT']:
        print(f"{timestamp} - {position} Unrealized PnL: {unrealized_pnl:.2f}")
        logging.info(
            f"{timestamp} - {position} Unrealized PnL: {unrealized_pnl:.2f}")


def calculate_unrealized_pnl(current_position, open_price, current_price, usdt_amount):
    """Calculates the unrealized PnL for the current position."""
    if current_position == 'LONG':
        return (current_price - open_price) / open_price * usdt_amount
    elif current_position == 'SHORT':
        return (open_price - current_price) / open_price * usdt_amount
    return 0


def fetch_and_plot_klines(symbol, interval, limit):
    """Fetch kline data from Binance and plot close prices."""
    try:
        # Fetch klines
        klines = client.get_klines(
            symbol=symbol, interval=interval, limit=limit)

        # Extract close prices and timestamps
        close_prices = [float(kline[4]) for kline in klines]
        timestamps = [datetime.fromtimestamp(
            int(kline[0]) / 1000) for kline in klines]

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


def get_notional_min(symbol):
    """Fetch NOTIONAL filter for the given symbol."""
    exchange_info = client.get_exchange_info()
    for symbol_info in exchange_info['symbols']:
        if symbol_info['symbol'] == symbol:
            for filter_info in symbol_info['filters']:
                if filter_info['filterType'] == 'NOTIONAL':
                    return float(filter_info['minNotional'])
    raise Exception(f"NOTIONAL filter not found for symbol {symbol}")


def get_lot_size(symbol):
    """Fetch LOT_SIZE filter for the given symbol."""
    exchange_info = client.get_exchange_info()
    for symbol_info in exchange_info['symbols']:
        if symbol_info['symbol'] == symbol:
            for filter_info in symbol_info['filters']:
                if filter_info['filterType'] == 'LOT_SIZE':
                    return (
                        float(filter_info['minQty']),
                        float(filter_info['maxQty']),
                        float(filter_info['stepSize'])
                    )
    raise Exception(f"LOT_SIZE not found for symbol {symbol}")


def adjust_quantity(quantity, step_size):
    """Adjust quantity to the nearest step size and format to the allowed precision."""
    precision = len(str(step_size).split(
        '.')[-1])  # Determine the number of decimal places
    return round(quantity, precision)


def calculate_sl_tp(entry_price, direction='LONG', interval='1h', tp_percentage=0.02, sl_percentage=0.01):

    # Set multiplier according to the interval
    interval_multipliers = {
        '1m': 0.5,
        '5m': 0.02,
        '15m': 0.9,
        '30m': 1.0,
        '1h': 1.0,
        '2h': 1.2,
        '4h': 1.5,
        '12h': 1.75,
        '1d': 2.0,
        '1w': 3.0,
    }

    # Use the multiplier based on the interval, default to 1.0 if the interval is not in the list
    multiplier = interval_multipliers.get(interval, 1.0)

    # Calculate TP and SL
    if direction.upper() == 'LONG':
        tp_price = entry_price * (1 + tp_percentage * multiplier)
        sl_price = entry_price * (1 - sl_percentage * multiplier)
    elif direction.upper() == 'SHORT':
        tp_price = entry_price * (1 - tp_percentage * multiplier)
        sl_price = entry_price * (1 + sl_percentage * multiplier)
    else:
        raise ValueError("Invalid direction. Must be 'LONG' or 'SHORT'.")

    return {
        'TP': round(tp_price, 2),  # Round to 2 decimal places
        'SL': round(sl_price, 2)   # Round to 2 decimal places
    }


# Run the bot
if __name__ == "__main__":
    # back_test(symbol, interval, limit)
    # fetch_and_plot_klines(symbol=symbol, interval=interval, limit=limit)
    # scalping_bot(symbol,interval,usdt_amount)
    usdt_amount = 2000
    direction = 'LONG'
    symbol = 'BTCUSDT'
    interval = '5m'
    limit = 300

    tp_percentage = 0.02  # 2% take profit
    sl_percentage = 0.01  # 1% stop loss
    rsi_buy_threshold = 40
    rsi_sell_threshold = 60

    # scalping_bot(symbol=symbol, interval=interval, limit=limit, usdt_amount=usdt_amount, tp_percentage=tp_percentage,
    #              sl_percentage=sl_percentage, rsi_buy_threshold=rsi_buy_threshold, rsi_sell_threshold=rsi_sell_threshold)
    back_test_via_pnl(symbol=symbol,
                      limit=limit,
                      interval=interval,
                      usdt_amount=usdt_amount,
                      rsi_buy_threshold=rsi_buy_threshold,
                      rsi_sell_threshold=rsi_sell_threshold,
                      tp_percentage=tp_percentage,
                      sl_percentage=sl_percentage)

    # result = calculate_sl_tp(usdt_amount, direction, interval, tp_percentage, sl_percentage)
    # print(result)
