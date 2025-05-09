

from bingx import *
amount = 0.025  # in btc = 68380 ==> 100$

last_order_id, order_type = open_long(
                                symbol="ETH-USDT", quantity=amount)

print(f'last_order_id = {last_order_id}')

time.sleep(5)
response = close_long(symbol="ETH-USDT",quantity=amount)
print(response.text)
# close_order(symbol="ETH-USDT",order_id=last_order_id)
