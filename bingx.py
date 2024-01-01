
import time
import requests
import hmac
from hashlib import sha256
# import matplotlib.pyplot as plt
import pandas as pd


APIURL = "https://open-api.bingx.com"
APIKEY = 'HE9i0tkLC2oxhtjwr6F3R8VA6KMgd9gEoaedxNmiZnC9emlxdHlNkcZGltMv1U0YKPHOvTrQCYcI5qUdDEDA'
SECRETKEY = 'kcfaeHV6CklsCMRCZqXvC17fcnpg0dKcM6YgiwdkhCXCeB2I8GCVNkHqdlK0BWaUN7E2nFwCCUmSO2OXofA'


def positions(symbol):
    payload = {}
    path = '/openApi/swap/v2/user/positions'
    method = "GET"
    paramsMap = {
        "recvWindow": "0",
        "symbol": symbol,
        "timestamp": str(int(time.time() * 1000))
    }
    paramsStr = praseParam(paramsMap)
    return send_request(method, path, paramsStr, payload)


def open_long(symbol, quantity):
    payload = {}
    path = '/openApi/swap/v2/trade/order'
    method = "POST"
    paramsMap = {
        "symbol": symbol,
        "side": "BUY",
        "positionSide": "LONG",
        "type": "MARKET",
        "quantity": quantity,
        "timestamp": str(int(time.time() * 1000))
    }
    paramsStr = praseParam(paramsMap)
    response = send_request(method, path, paramsStr, payload)
    return response


def close_long(symbol, quantity):
    payload = {}
    path = '/openApi/swap/v2/trade/order'
    method = "POST"
    paramsMap = {
        "symbol": symbol,
        "side": "SELL",
        "positionSide": "LONG",
        "type": "MARKET",
        "quantity": quantity,
        "timestamp": str(int(time.time() * 1000))
    }
    paramsStr = praseParam(paramsMap)
    response = send_request(method, path, paramsStr, payload)
    return response


def close_all_order(symbol):
    payload = {}
    path = '/openApi/swap/v2/trade/allOpenOrders'
    method = "DELETE"
    paramsMap = {
        "recvWindow": "0",
        "symbol": symbol,
        "timestamp": str(int(time.time() * 1000))
    }
    paramsStr = praseParam(paramsMap)
    return send_request(method, path, paramsStr, payload)


def check_order_status(symbol, order_id):
    payload = {}
    path = '/openApi/swap/v2/trade/order'
    method = "GET"

    params_map = {
        "symbol": symbol,
        "orderId": order_id,
        "timestamp": str(int(time.time() * 1000))
    }

    params_str = praseParam(params_map)
    return send_request(method, path, params_str, payload)


def close_order(symbol, order_id):
    order_status_response = check_order_status(symbol, order_id)
    print(order_status_response.json())
    # Check if the order is still active before attempting to cancel
    if order_status_response.json().get('status') == 'NEW':
        # Continue with the cancel order logic
        payload = {}
        path = '/openApi/swap/v2/trade/order'
        method = "DELETE"
        params_map = {
            "orderId": order_id,
            "symbol": symbol,
            "timestamp": str(int(time.time() * 1000))
        }
        params_str = praseParam(params_map)
        return send_request(method, path, params_str, payload)
    else:
        print(f"Order {order_id} is not active and cannot be canceled.")


def price(symbol):
    payload = {}
    path = '/openApi/swap/v2/quote/price'
    method = "GET"
    paramsMap = {
        "timestamp": "1702718923479",
        "symbol": symbol
    }
    paramsStr = praseParam(paramsMap)
    return send_request(method, path, paramsStr, payload)

async def get_rsi14(symbol, interval, period=14):
    try:
        response = get_kline(symbol, interval)
        response.raise_for_status()
        klines_data = response.json().get('data', [])
        df = pd.DataFrame(klines_data, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        df.set_index('time', inplace=True)
        
        # Convert 'close' column to numeric
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        
        # Drop rows with missing values in 'close' column
        df = df.dropna(subset=['close'])
        # Calculate price differences and handle NaN values
        price_diff = df['close'].diff(1).fillna(0)
        price_diff = df['close'].diff(1)
       
        gain = price_diff.where(price_diff > 0, 0)
        loss = -price_diff.where(price_diff < 0, 0)

        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
     

        return rsi.iloc[-1]
        # return 10
    
    except requests.exceptions.RequestException as e:
        print(f"Error making API request: {e}")
        return None


async def get_rsi(symbol, interval, period=14):
    try:
        response = get_kline(symbol, interval)
        response.raise_for_status()
        klines_data = response.json().get('data', [])

        if klines_data is None:
            print("Error: Klines data is None.")
            return None

        if len(klines_data) < period + 1:
            print("Insufficient data to calculate RSI.")
            print(f"Available data: {klines_data}")
            return None

        closes = [float(kline['close']) for kline in klines_data]

        if len(closes) < period:
            print("Insufficient data to calculate RSI.")
            print(f"Available closes: {closes}")
            return None

        # Calculate daily price changes
        changes = [closes[i] - closes[i - 1] for i in range(1, len(closes))]

        # Separate gains and losses
        gains = [change for change in changes if change > 0]
        losses = [-change for change in changes if change < 0]

        # Calculate average gains and losses for the initial period
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period

        # Calculate initial RS and RSI
        rs = avg_gain / avg_loss if avg_loss != 0 else float('inf')
        rsi = 100 - (100 / (1 + rs))

        # Calculate RSI using SMA for the remaining data
        for i in range(period, len(closes)):
            change = closes[i] - closes[i - 1]
            avg_gain = (avg_gain * (period - 1) + max(0, change)) / period
            avg_loss = (avg_loss * (period - 1) + max(0, -change)) / period
            rs = avg_gain / avg_loss if avg_loss != 0 else float('inf')
            rsi = 100 - (100 / (1 + rs))

        return rsi

    except requests.exceptions.RequestException as e:
        print(f"Error making API request: {e}")
        return None


def get_server_time():
    payload = {}
    path = '/openApi/swap/v2/server/time'
    method = "GET"
    paramsMap = {
    }
    paramsStr = praseParam(paramsMap)
    response = send_request(method, path, paramsStr, payload)
    response.raise_for_status()
    data = response.json().get('data', [])
    return data.get('serverTime', {})


def get_kline(symbol, interval, period=14):
    server_time = get_server_time()
    payload = {}
    path = '/openApi/swap/v3/quote/klines'
    method = "GET"
    paramsMap = {
        "symbol": symbol,
        "interval": interval,
        "limit": 10,
        "timestamp": str(int(time.time() * 1000))
    }
    

    paramsStr = praseParam(paramsMap)
    return send_request(method, path, paramsStr, payload)


def get_sign(api_secret, payload):
    signature = hmac.new(api_secret.encode("utf-8"),
                         payload.encode("utf-8"), digestmod=sha256).hexdigest()
    # print("sign=" + signature)
    return signature


def send_request(method, path, urlpa, payload):
    url = "%s%s?%s&signature=%s" % (
        APIURL, path, urlpa, get_sign(SECRETKEY, urlpa))
    headers = {
        'X-BX-APIKEY': APIKEY,
    }
    response = requests.request(method, url, headers=headers, data=payload)
    return response


def praseParam(paramsMap):
    sortedKeys = sorted(paramsMap)
    paramsStr = "&".join(["%s=%s" % (x, paramsMap[x]) for x in sortedKeys])
    return paramsStr+"&timestamp="+str(int(time.time() * 1000))


# def showKlines(symbol, interval,):
#     rsi_values = []
#     for _ in range(5):
#         rsi_value = get_rsi(symbol, interval)
#         if rsi_value is not None:
#             rsi_values.append(rsi_value)
#             print(f"RSI for {symbol} ({interval}): {rsi_value}")
#             break
#         else:
#             print("Retrying...")

#     # Plotting
#     if rsi_values:
#         plt.plot(rsi_values, label='RSI')
#         plt.axhline(y=30, color='r', linestyle='--',
#                     label='Oversold (RSI < 30)')
#         plt.axhline(y=70, color='g', linestyle='--',
#                     label='Overbought (RSI > 70)')

#         # Highlight buy orders (you can customize this part based on your buy conditions)
#         for i, rsi_value in enumerate(rsi_values):
#             if rsi_value < 30:
#                 plt.scatter(i, rsi_value, color='b',
#                             label='Buy Order', marker='^', s=100)

#         plt.title(f"RSI Chart for {symbol} ({interval})")
#         plt.xlabel("Time")
#         plt.ylabel("RSI Value")
#         plt.legend()
#         plt.show()


# if __name__ == '__main__':
#     showKlines(symbol="BTC-USDT", interval="1m")
