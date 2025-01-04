from binance.client import Client
from binance.exceptions import BinanceAPIException
import logging
import pandas as pd
from telegram import send_telegram_message  


logging.basicConfig(
    filename='scalping_bot.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)


API_KEY = '81vXiGyDU5cAgMH5PB5xHem9V9sGw6E1QaXGxyVMm79p0Gk8E7OYMf2OnSrMVWom'
API_SECRET = 'gQWW6QW73sh9a83O4B5W6MzEmnXRxE55hsqM2Lvz1I27CtFog5EqBPI8moFIPICb'

try:
    client = Client(API_KEY, API_SECRET, testnet=False)
    # client.FUTURES_URL = 'https://testnet.binancefuture.com/fapi'

except Exception as e:
    print(f"Error during ping: {e}")




def get_klines_with_start(symbol, interval, limit=100, start_time=None, end_time=None):
    try:
        klines = client.get_klines(
            symbol=symbol,
            interval=interval,
            limit=limit,
            startTime=start_time,
            endTime=end_time
        )

        # Ensure klines contains the expected data structure (Closing price and timestamp)
        return [(float(kline[4]), int(kline[0])) for kline in klines]
    except Exception as e:
        logging.error(f"Error fetching klines: {e}")
        return []




def get_klines_all(symbol, interval, start_time=None, end_time=None, limit=100):
    try:
        # Convert start and end times to milliseconds
        start_time_ms = int(pd.Timestamp(start_time).timestamp() * 1000) if start_time else None
        end_time_ms = int(pd.Timestamp(end_time).timestamp() * 1000) if end_time else None

        klines = client.get_klines(
            symbol=symbol,
            interval=interval,
            limit=limit,
            startTime=start_time_ms,
            endTime=end_time_ms,
        )
        
        # Convert to DataFrame
        data = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Select and format relevant columns
        data = data[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
        data[['open', 'high', 'low', 'close', 'volume']] = data[['open', 'high', 'low', 'close', 'volume']].astype(float)
        
        return data
    except Exception as e:
        logging.error(f"Error fetching klines: {e}")
        return pd.DataFrame()  # Return an empty DataFrame in case of error


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
        # Send message to Telegram group
        message = f"🚀 <b>Close Order</b> 🚀\n" \
            f"📈 <b>Symbol:</b> {symbol}\n" \
            f"🔁 <b>Action:</b> {position}\n" \
            f"💵 <b>Quantity:</b> {quantity}\n"

        send_telegram_message(message)
        return order

    except Exception as e:
        print(f"Error closing position: {e}")
        return None

def place_order(symbol, side, usdt_amount):
    """Place a buy or sell order on Binance Futures."""
    global current_quantity
    try:
        # Fetch the latest price
        ticker = client.futures_symbol_ticker(symbol=symbol)
        last_price = float(ticker['price'])

        # Fetch LOT_SIZE filter dynamically
        min_qty, max_qty, step_size = get_lot_size(symbol)
        logging.info(f"LOT_SIZE for {symbol} - Min Qty: {min_qty}, Max Qty: {max_qty}, Step Size: {step_size}")

        # Calculate quantity and adjust to the nearest step size
        raw_quantity = usdt_amount / last_price
        precision = len(str(step_size).split('.')[-1])  # Determine the number of decimal places for step_size
        quantity = round(raw_quantity - (raw_quantity % step_size), precision)

        # Ensure quantity is within the allowed range
        if quantity < min_qty or quantity > max_qty:
            raise ValueError(
                f"Quantity {quantity} is out of range for {symbol}: Min {min_qty}, Max {max_qty}"
            )

        # Place the order
        order = client.futures_create_order(
            symbol=symbol,
            side=side,
            type='MARKET',
            quantity=quantity
        )
        current_quantity = quantity
        logging.info(f"Order placed: {side} {quantity} of {symbol}. Order ID: {order['orderId']}")

        # Send message to Telegram group
        message = (
            f"🚀 <b>New Order Placed</b> 🚀\n"
            f"📈 <b>Symbol:</b> {symbol}\n"
            f"🔁 <b>Action:</b> {side}\n"
            f"💵 <b>Quantity:</b> {quantity}\n"
            f"💰 <b>Price:</b> {last_price}\n"
        )
        send_telegram_message(message)

        return order

    except BinanceAPIException as e:
        logging.error(f"Binance API Exception: {e}")
        print(f"Error: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        print(f"Error: {e}")
        return None

def set_leverage(symbol, leverage=1):
    """
    Set leverage for the given symbol.

    Args:
        symbol (str): Trading pair, e.g., 'BTCUSDT'.
        leverage (int): The leverage to be set. Default is 1x.
    """
    try:
        response = client.futures_change_leverage(
            symbol=symbol,
            leverage=leverage
        )
        logging.info(f"Leverage set to {leverage}x for {
                     symbol}. Response: {response}")
        return response
    except BinanceAPIException as e:
        logging.error(f"Binance API Exception while setting leverage: {e}")
        print(f"Error setting leverage: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error while setting leverage: {e}")
        print(f"Unexpected error: {e}")
        return None


def set_margin_mode(symbol, margin_type="ISOLATED"):
    """
    Set margin mode for the given symbol.

    Args:
        symbol (str): Trading pair, e.g., 'BTCUSDT'.
        margin_type (str): Margin mode to set ('ISOLATED' or 'CROSSED'). Default is 'ISOLATED'.
    """
    try:
        response = client.futures_change_margin_type(
            symbol=symbol,
            marginType=margin_type
        )
        logging.info(f"Margin mode set to {margin_type} for {
                     symbol}. Response: {response}")
        return response
    except BinanceAPIException as e:
        if "No need to change margin type." in str(e):
            logging.info(f"Margin type for {
                         symbol} is already set to {margin_type}.")
            print(f"Margin type for {symbol} is already set to {margin_type}.")
        else:
            logging.error(
                f"Binance API Exception while setting margin mode: {e}")
            print(f"Error setting margin mode: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error while setting margin mode: {e}")
        print(f"Unexpected error: {e}")
        return None


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


import logging

def get_account_balance(asset):
    try:
        # Fetch account information
        account_info = client.get_account()
        
        for balance in account_info['balances']:
            if balance['asset'] == 'USDT':
                print(f"Free: {balance['free']}, Locked: {balance['locked']}")
        
        # If the asset is not found, log a warning and return 0
        logging.warning(f"Asset {asset} not found in account balances.")
        return 0.0
    except Exception as e:
        logging.error(f"Error fetching account balance for {asset}: {e}")
        return 0.0

def get_futures_account_balance(asset):
    try:
        # Fetch futures account balance
        account_info = client.futures_account_balance()
        
        for balance in account_info:
            if balance['asset'] == asset:
                available_balance = float(balance['balance'])
                print(f"Futures Balance - Free: {available_balance}")
                return available_balance
        
        # If the asset is not found, log a warning and return 0
        logging.warning(f"Asset {asset} not found in futures account balances.")
        return 0.0
    except Exception as e:
        logging.error(f"Error fetching futures account balance for {asset}: {e}")
        return 0.0


import logging

def close_all_positions():
    """
    Close all open futures positions.
    """
    try:
        # Fetch all open positions
        positions = client.futures_account()['positions']
        for position in positions:
            symbol = position['symbol']
            position_amt = float(position['positionAmt'])  # Positive for LONG, Negative for SHORT

            # Skip positions with zero quantity
            if position_amt == 0:
                continue

            side = 'SELL' if position_amt > 0 else 'BUY'  # Close LONG with SELL, SHORT with BUY
            quantity = abs(position_amt)  # Use absolute value for order quantity

            # Place market order to close position
            logging.info(f"Closing position for {symbol}: {side} {quantity}")
            try:
                order = client.futures_create_order(
                    symbol=symbol,
                    side=side,
                    type='MARKET',
                    quantity=quantity
                )
                logging.info(f"Closed position for {symbol}: {order}")
            except Exception as e:
                logging.error(f"Error closing position for {symbol}: {e}")

    except Exception as e:
        logging.error(f"Error fetching or closing positions: {e}")
