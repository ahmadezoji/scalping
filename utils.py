import hmac
import base64
import requests
import json
import urllib

APIURL = "https://api-swap-rest.bingbon.pro"
API_KEY = 'h8tPF9Ub6Ko0dN4fu23aZO011RmFufcVdsXnEZmf7RIUIVGiK4x2Gt31SCcGfc61R1bmEaEHKXC5O1Qug'
SECRET_KEY = 'bmY2rfydc3zt0sP2Tp4u02Z8dTxjwvVdZddsBlQhySAJ2DBWl31kwFAG9Ib8lg7ZV944QXgXDwPT136VwQ'


def genSignature(path, method, paramsMap):
    sortedKeys = sorted(paramsMap)
    paramsStr = "&".join(["%s=%s" % (x, paramsMap[x]) for x in sortedKeys])
    paramsStr = method + path + paramsStr
    return hmac.new(SECRET_KEY.encode("utf-8"), paramsStr.encode("utf-8"), digestmod="sha256").digest()


def getLatestPrice(symbol):
    paramsMap = {
        "symbol": symbol,
    }
    paramsStr = "&sign=" + urllib.parse.quote(
        base64.b64encode(genSignature("/api/v1/market/getLatestPrice", "GET", paramsMap)))
    url = f'https://api-swap-rest.bingbon.pro/api/v1/market/getLatestPrice?&symbol={symbol}{paramsStr}'
    payload = {}
    headers = {
    }
    response = requests.request("GET", url, headers=headers, data=payload)
    last_price = (json.loads(response.text))['data']['tradePrice']
    last_price = float(last_price)
    return last_price
