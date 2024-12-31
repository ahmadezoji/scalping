from binance.client import Client
from binance.exceptions import BinanceAPIException
import logging
import pandas as pd

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



def check_sl_tp(position, symbol, tp_price, sl_price, open_price, usdt_amount):
    """Thread function to check if SL or TP is hit."""
    global current_position, open_trade_thread, current_quantity
    while current_position:
        try:
            # Fetch the current price
            klines = get_klines(symbol, '1m', limit=1)

            prices = [kline[0] for kline in klines]
            current_price = prices[-1]
            # Calculate and log unrealized PnL
            unrealized_pnl = calculate_unrealized_pnl(
                position, open_price, current_price, usdt_amount)
            logging.info(f"Unrealized PnL for position {position} on {symbol}: {unrealized_pnl} USDT")

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

