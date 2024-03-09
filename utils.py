import hmac
import base64
import requests
import json
import urllib

APIURL = "https://open-api.bingx.com"
API_KEY = 'HE9i0tkLC2oxhtjwr6F3R8VA6KMgd9gEoaedxNmiZnC9emlxdHlNkcZGltMv1U0YKPHOvTrQCYcI5qUdDEDA'
SECRET_KEY = 'kcfaeHV6CklsCMRCZqXvC17fcnpg0dKcM6YgiwdkhCXCeB2I8GCVNkHqdlK0BWaUN7E2nFwCCUmSO2OXofA'


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
    url = f'https://open-api.bingx.com/api/v1/market/getLatestPrice?&symbol={symbol}{paramsStr}'
    payload = {}
    headers = {
    }
  
    response = requests.request("GET", url, headers=headers, data=payload)
    # last_price = (json.loads(response.text))['data']['tradePrice']
    # last_price = float(last_price)
    return 100