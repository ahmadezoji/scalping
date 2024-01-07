
import time
import requests
import hmac
from hashlib import sha256
# import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import numpy as np



APIURL = "https://open-api.bingx.com"
APIKEY = 'HE9i0tkLC2oxhtjwr6F3R8VA6KMgd9gEoaedxNmiZnC9emlxdHlNkcZGltMv1U0YKPHOvTrQCYcI5qUdDEDA'
SECRETKEY = 'kcfaeHV6CklsCMRCZqXvC17fcnpg0dKcM6YgiwdkhCXCeB2I8GCVNkHqdlK0BWaUN7E2nFwCCUmSO2OXofA'

def get_last_timestamp(df):
    # return int(df.time[-1:].values[0])
    return int(df.index[-1].timestamp())


def format_data(data):
    # data = response.get('list', None)
    
    if not data:
        return 
    
    data = pd.DataFrame(data,
                        columns =[
                            'time',
                            'open',
                            'high',
                            'low',
                            'close',
                            'volume',
                            ],
                        )
    
    
    # Convert 'time' column to pandas Timestamp
    data['time'] = pd.to_datetime(data['time'], unit='ms')
    
    # Set 'time' column as the DataFrame index
    data.set_index('time', inplace=True)

    return data[::-1].apply(pd.to_numeric)

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

def get_and_calculate_rsi(symbol, interval, period=14):
    start_time_seconds = time.time() - (5 * 60)  # 5 minutes ago in seconds
    start_time_milliseconds = int(start_time_seconds * 1000)
    response = get_kline(symbol, interval, period,start=start_time_milliseconds)
    response.raise_for_status()
    klines_data = response.json().get('data', [])
    df = pd.DataFrame(klines_data, columns=['time', 'open', 'high', 'low', 'close', 'volume'])   
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    df['delta'] = df['close'].astype(float) - df['close'].astype(float).shift(1)    
    gains = np.where(df['delta'] > 0, df['delta'], 0)
    losses = np.where(df['delta'] < 0, -df['delta'], 0)
    avg_gains = pd.Series(gains).rolling(window=14, min_periods=1).mean()
    avg_losses = pd.Series(losses).rolling(window=14, min_periods=1).mean()
    rs = avg_gains / avg_losses
    rsi = 100 - (100 / (1 + rs))
    overbought_threshold = 70
    oversold_threshold = 30
    df['buy_signal'] = np.where(rsi < oversold_threshold, 1, 0)
    df['sell_signal'] = np.where(rsi > overbought_threshold, 1, 0)

    for index, row in df.iterrows():
        if row['buy_signal'] == 1:
            print(f"Buy signal at {row['time']} for {symbol} ({interval})")
            # Implement your buy logic here
            
        elif row['sell_signal'] == 1:
            print(f"Sell signal at {row['time']} for {symbol} ({interval})")
            # Implement your sell logic here
    return rsi


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


def get_kline(symbol, interval, period=14,start=str(int(time.time() * 1000))):
    server_time = get_server_time()
    payload = {}
    path = '/openApi/swap/v3/quote/klines'
    method = "GET"
    paramsMap = {
        "symbol": symbol,
        "interval": interval,
        "startTime":start
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
