
import time
import requests
import hmac
from hashlib import sha256
# import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import numpy as np
import json
from enum import Enum


OrderType = Enum('OrderType', ['LONG', 'SHORT'])


# APIURL = "https://open-api.bingx.com"
APIURL = "https://open-api-vst.bingx.com"
APIKEY = 'CGRMVGm69DTMVYB6nAUbgOzfnXVBz0dY2NLVd1jG0JaJpInitIzxxuciPArmFiFldTjR5WqHXQMY0FcFuQ'
SECRETKEY = 'cZg6feI0YVG6H7x8OQuR7gFv84rYHn9q5Gi38WgWQ7guy4l5mV9r2kijbN56Tt3ZnKByVtVuDO60x0taflMMw'



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
    return (response.json().get('data').get('order').get('orderId'), OrderType.LONG) if response.json().get('code') == 0 else (-1, OrderType.LONG)

def open_short(symbol, quantity):
    payload = {}
    path = '/openApi/swap/v2/trade/order'
    method = "POST"
    paramsMap = {
        "symbol": symbol,
        "side": "SELL",
        "positionSide": "SHORT",
        "type": "MARKET",
        "quantity": quantity,
        "timestamp": str(int(time.time() * 1000))
    }
    paramsStr = praseParam(paramsMap)
    response = send_request(method, path, paramsStr, payload)
    
    return (response.json().get('data').get('order').get('orderId'), OrderType.SHORT) if response.json().get('code') == 0 else (-1, OrderType.SHORT)




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




def last_price(symbol):
    payload = {}
    path = '/openApi/swap/v1/ticker/price'
    method = "GET"
    paramsMap = {
    "timestamp": str(int(time.time() * 1000)),
    "symbol": symbol
    }
    paramsStr = praseParam(paramsMap)
    response =  send_request(method, path, paramsStr, payload)


    last_price =  (json.loads(response.text))['data']['price']
    return float(last_price)
    
    
   
    



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


def get_kline(symbol, interval,start=str(int(time.time() * 1000))):
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

