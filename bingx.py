
import time
import requests
import hmac
from hashlib import sha256
# import matplotlib.pyplot as plt


APIURL = "https://open-api.bingx.com"
APIKEY = 'HE9i0tkLC2oxhtjwr6F3R8VA6KMgd9gEoaedxNmiZnC9emlxdHlNkcZGltMv1U0YKPHOvTrQCYcI5qUdDEDA'
SECRETKEY = 'kcfaeHV6CklsCMRCZqXvC17fcnpg0dKcM6YgiwdkhCXCeB2I8GCVNkHqdlK0BWaUN7E2nFwCCUmSO2OXofA'


def open_order(symbol, quantity):
    payload = {}
    path = '/openApi/swap/v2/trade/order'
    method = "POST"
    paramsMap = {
        "symbol": symbol,
        "side": "BUY",
        "positionSide": "LONG",
        "type": "MARKET",
        "quantity": quantity,
    }
    paramsStr = praseParam(paramsMap)
    response = send_request(method, path, paramsStr, payload)
    response.raise_for_status()
    data = response.json().get('data', [])
    order_info = data.get('order', {})
    return order_info.get('orderId', None)

def close_order(symbol,order_id):
    payload = {}
    path = '/openApi/swap/v2/trade/order'
    method = "DELETE"
    paramsMap = {
    "orderId": order_id,
    "symbol": symbol,
}
    paramsStr = praseParam(paramsMap)
    return send_request(method, path, paramsStr, payload)


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


async def get_rsi(symbol, interval, period=14):
    try:
        response = get_kline(symbol, interval)
        response.raise_for_status()
        data = response.json().get('data', [])

        if len(data) < period + 1:
            print("Insufficient data to calculate RSI.")
            return None

        closes = [float(kline['close']) for kline in data]

        if len(closes) < period:
            print("Insufficient data to calculate RSI.")
            return None

        changes = [closes[i] - closes[i - 1] for i in range(1, len(closes))]

        avg_gain = sum([change for change in changes if change > 0]) / period
        avg_loss = -sum([change for change in changes if change < 0]) / period
        rs = avg_gain / avg_loss if avg_loss != 0 else float('inf')
        rsi = 100 - (100 / (1 + rs))

        return rsi

    except requests.exceptions.RequestException as e:
        print(f"Error making API request: {e}")
        return None

def get_kline(symbol, interval, period=14):
    payload = {}
    path = '/openApi/swap/v3/quote/klines'
    method = "GET"
    paramsMap = {
        "symbol": symbol,
        "interval": interval,
        "limit": period + 1,
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


if __name__ == '__main__':
    showKlines(symbol="BTC-USDT", interval="1m")
