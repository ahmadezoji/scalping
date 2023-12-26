import ccxt
import time

# Replace with your Binance API key and secret
api_key = 'vy8QkSdxYaNdj4FH0DYn9XvWF06ceAzx8BSzyUH09PLGATTGHHBwJ9i1YwCO8pU9GvXRuNavUihx1ApuhzA'
api_secret = 'tmaRazQ7XAp8JaVf8E5slZXKTTspnuVMgzU7APP9Zb0jWMkJfCOlVsUFIhmbAhIX5ODGU6yyQOz858sWJTsw'
symbol = 'BTC/USDT'  # Change to the desired trading pair
quantity = 0.001  # Change to the desired trade quantity
interval = '1m'  # Change to the desired time interval

# Initialize Binance client
exchange = ccxt.binance({
    'apiKey': api_key,
    'secret': api_secret,
})

def get_rsi(symbol, interval, period=14):
    ohlcv = exchange.fetch_ohlcv(symbol, interval)
    closes = [tick[4] for tick in ohlcv]
    changes = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    avg_gain = sum([change for change in changes if change > 0]) / period
    avg_loss = -sum([change for change in changes if change < 0]) / period
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def scalping_strategy():
    while True:
        try:
            rsi = get_rsi(symbol, interval)
            print(f'Current RSI for {interval} interval: {rsi}')

            if rsi < 30:
                # RSI is below 30, indicating oversold condition
                print('RSI is below 30. Initiating buy order...')
                order = exchange.create_market_buy_order(symbol, quantity)
                print(f'Buy order executed: {order}')

            elif rsi > 70:
                # RSI is above 70, indicating overbought condition
                print('RSI is above 70. Initiating sell order...')
                order = exchange.create_market_sell_order(symbol, quantity)
                print(f'Sell order executed: {order}')

            # Wait for the next interval before checking RSI again
            time.sleep(60)

        except Exception as e:
            print(f'An error occurred: {e}')

if __name__ == '__main__':
    scalping_strategy()

