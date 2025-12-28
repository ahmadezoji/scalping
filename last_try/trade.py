import hmac
import time
import base64
import urllib

ORDER_AMOUNT = 70
LONG_LEVERAGE = 10
SHORT_LEVERAGE = 10

# Define a list of your favorite cryptocurrencies
from utils import SECRET_KEY, API_KEY, APIURL, getLatestPrice

coins = ['BTC', 'ETH', 'XRP']
lastOrderId = None


def genSignature(path, method, paramsMap):
    sortedKeys = sorted(paramsMap)
    paramsStr = "&".join(["%s=%s" % (x, paramsMap[x]) for x in sortedKeys])
    paramsStr = method + path + paramsStr
    return hmac.new(SECRET_KEY.encode("utf-8"), paramsStr.encode("utf-8"), digestmod="sha256").digest()


def post(url, body):
    req = urllib.request.Request(url, data=body.encode("utf-8"), headers={'User-Agent': 'Mozilla/5.0'})
    return urllib.request.urlopen(req).read()


def getBalance():
    paramsMap = {
        "apiKey": API_KEY,
        "timestamp": int(time.time() * 1000),
        "currency": "USDT",
    }
    sortedKeys = sorted(paramsMap)
    paramsStr = "&".join(["%s=%s" % (x, paramsMap[x]) for x in sortedKeys])
    paramsStr += "&sign=" + urllib.parse.quote(
        base64.b64encode(genSignature("/api/v1/user/getBalance", "POST", paramsMap)))
    url = "%s/api/v1/user/getBalance" % APIURL
    return post(url, paramsStr)


def getPositions(symbol):
    paramsMap = {
        "symbol": symbol,
        "apiKey": API_KEY,
        "timestamp": int(time.time() * 1000),
    }
    sortedKeys = sorted(paramsMap)
    paramsStr = "&".join(["%s=%s" % (x, paramsMap[x]) for x in sortedKeys])
    paramsStr += "&sign=" + urllib.parse.quote(
        base64.b64encode(genSignature("/api/v1/user/getPositions", "POST", paramsMap)))
    url = "%s/api/v1/user/getPositions" % APIURL
    return post(url, paramsStr)

def setupOrder(type, symbol,leverage,amount):
    amount = float(amount*leverage)  # USDT
    last_price = getLatestPrice(symbol)
    vol = amount / last_price
    if type == "short":
        takerProfitPrice = float(last_price - (last_price * 0.02))
        stopLossPrice = float(last_price + (last_price * 0.05))
        result = placeOrder(symbol, "Ask", last_price, vol, "Market", "Open", takerProfitPrice, stopLossPrice)
        print(result)
    elif type == "long":
        takerProfitPrice = float(last_price + (last_price * 0.05))
        stopLossPrice = float(last_price - (last_price * 0.02))
        result = placeOrder(symbol, "Bid", last_price, vol, "Market", "Open", takerProfitPrice, stopLossPrice)
        print(result)
def placeOrder(symbol, side, price, volume, tradeType, action,tf,sl):
    print(f'TF ={tf}')
    print(f'SL ={sl}')
    paramsMap = {
        "symbol": symbol,
        "apiKey": API_KEY,
        "side": side,
        "entrustPrice": price,
        "entrustVolume": volume,
        "tradeType": tradeType,
        "action": action,
        "timestamp": int(time.time() * 1000),
        "takerProfitPrice": tf,
        "stopLossPrice": sl
    }

    sortedKeys = sorted(paramsMap)
    paramsStr = "&".join(["%s=%s" % (x, paramsMap[x]) for x in sortedKeys])
    paramsStr += "&sign=" + urllib.parse.quote(base64.b64encode(genSignature("/api/v1/user/trade", "POST", paramsMap)))
    url = "%s/api/v1/user/trade" % APIURL
    return post(url, paramsStr)


def cancleOrder(symbol, orderId):
    paramsMap = {
        "symbol": symbol,
        "apiKey": API_KEY,
        "orderId": orderId,
        "timestamp": int(time.time() * 1000),
    }
    sortedKeys = sorted(paramsMap)
    paramsStr = "&".join(["%s=%s" % (x, paramsMap[x]) for x in sortedKeys])
    paramsStr += "&sign=" + urllib.parse.quote(base64.b64encode(genSignature("/api/v1/user/cancelOrder", "POST", paramsMap)))
    url = "%s/api/v1/user/cancelOrder" % APIURL
    return post(url, paramsStr)
def setMarginMode(symbol,marginMode):
    paramsMap = {
        "symbol": symbol,
        "apiKey": API_KEY,
        "marginMode": marginMode,
        "timestamp": int(time.time() * 1000),
    }
    sortedKeys = sorted(paramsMap)
    paramsStr = "&".join(["%s=%s" % (x, paramsMap[x]) for x in sortedKeys])
    paramsStr += "&sign=" + urllib.parse.quote(
        base64.b64encode(genSignature("/api/v1/user/setMarginMode", "POST", paramsMap)))
    url = "%s/api/v1/user/setMarginMode" % APIURL
    return post(url, paramsStr)
def setLeverage(symbol,side,leverage):
    paramsMap = {
        "symbol": symbol,
        "apiKey": API_KEY,
        "side": side,
        "leverage" : leverage,
        "timestamp": int(time.time() * 1000),
    }
    sortedKeys = sorted(paramsMap)
    paramsStr = "&".join(["%s=%s" % (x, paramsMap[x]) for x in sortedKeys])
    paramsStr += "&sign=" + urllib.parse.quote(
        base64.b64encode(genSignature("/api/v1/user/setLeverage", "POST", paramsMap)))
    url = "%s/api/v1/user/setLeverage" % APIURL
    return post(url, paramsStr)

def main():
    print(getLatestPrice("BTC-USDT"))
    # print(cancleOrder("ETH-USDT",'1649936112992391168'))
# print(getLatestPrice("BTC-USDT"))
# thr = threading.Thread(target=getLastMessage, args=(), kwargs={})
# thr.start()
# coro = getLastMessage()
# asyncio.run(coro)
# loop = asyncio.get_event_loop()

# print("getBalance:", getBalance())
# while True:
#     getLastMessage()
# print("getLastPrice :", getLatestPrice("BTC-USDT"))
# result = loop.run_until_complete(coro)
# time.sleep(3)

# placeOrder("ETH-USDT", "Bid", 0, 0.02, "Market", "Open")
# result = placeOrder("ETH-USDT", "Bid", 0, 0.01, "Market", "Open")
# my_json = result.decode('utf8').replace("'", '"')
# print(my_json[3])
# print("placeOpenOrder:", placeOrder("DOGE-USDT", "Bid", 0,2, "Market", "Open"))

# print("getPositions:", getPositions("BTC-USDT"))

# print("placeCloseOrder:", placeOrder("BTC-USDT", "Ask", 0, 0.0004, "Market", "Close"))

if __name__ == "__main__":
    main()
