import argparse
from datetime import datetime
import pandas as pd
from index import SYMBOL, entry_usdt, trade_interval, tp_percentage, sl_percentage, client
from vwap_bot_new import calculate_vwap


def backtest_strategy(symbol, timeframe, start_date, end_date):
    start_ts = int(datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S').timestamp() * 1000)
    end_ts = int(datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S').timestamp() * 1000)

    trades = []
    current_position = None
    entry_price = 0
    entry_time = None
    total_profit_percentage = 0
    total_trades = 0
    winning_trades = 0
    peak_balance = current_balance = entry_usdt
    max_drawdown = 0

    # Fetch and prepare data
    historical_data = client.futures_historical_klines(
        symbol=symbol,
        interval=timeframe,
        start_str=start_ts,
        end_str=end_ts
    )
    df = pd.DataFrame(historical_data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades_count',
        'taker_buy_volume', 'taker_buy_quote_volume', 'ignore'
    ])
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].astype(float)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = calculate_vwap(df)

    daily_returns = {}

    for i in range(20, len(df)):
        row = df.iloc[i]
        signal = None

        if (row['close'] > row['vwap'] * 0.999 and
            row['stoch_k'] < 70 and
            row['rsi'] > 50 and
            row['close'] > row['ema_trend']):
            signal = 'LONG'
        elif (row['close'] < row['vwap'] * 1.001 and
              row['stoch_k'] > 30 and
              row['rsi'] < 50 and
              row['close'] < row['ema_trend']):
            signal = 'SHORT'

        if current_position:
            pnl_pct = ((row['close'] - entry_price) / entry_price) * 100 if current_position == 'LONG' else \
                      ((entry_price - row['close']) / entry_price) * 100

            if pnl_pct >= tp_percentage or pnl_pct <= -sl_percentage:
                total_trades += 1
                if pnl_pct > 0:
                    winning_trades += 1

                current_balance *= (1 + pnl_pct / 100)
                total_profit_percentage += pnl_pct

                if current_balance > peak_balance:
                    peak_balance = current_balance
                drawdown = (peak_balance - current_balance) / peak_balance * 100
                max_drawdown = max(max_drawdown, drawdown)

                day = row['timestamp'].date()
                daily_returns[day] = daily_returns.get(day, 0) + pnl_pct

                trades.append({
                    'entry_time': entry_time,
                    'exit_time': row['timestamp'],
                    'position': current_position,
                    'entry_price': entry_price,
                    'exit_price': row['close'],
                    'pnl_percentage': pnl_pct
                })

                current_position = None

        elif signal:
            current_position = signal
            entry_price = row['close']
            entry_time = row['timestamp']

    final_balance = current_balance
    win_rate = (winning_trades / total_trades * 100) if total_trades else 0
    avg_profit = total_profit_percentage / total_trades if total_trades else 0
    total_return = ((final_balance - entry_usdt) / entry_usdt) * 100

    return {
        'trades': trades,
        'daily_returns': daily_returns,
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'win_rate': win_rate,
        'avg_profit_per_trade': avg_profit,
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'final_balance': final_balance
    }


def print_results(results):
    print("\n=== Backtest Summary ===")
    print(f"Total Trades: {results['total_trades']}")
    print(f"Winning Trades: {results['winning_trades']}")
    print(f"Win Rate: {results['win_rate']:.2f}%")
    print(f"Average Profit/Trade: {results['avg_profit_per_trade']:.2f}%")
    print(f"Total Return: {results['total_return']:.2f}%")
    print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
    print(f"Final Balance: {results['final_balance']:.2f} USDT")

    print("\n=== Daily Returns ===")
    for date, ret in sorted(results['daily_returns'].items()):
        print(f"{date}: {ret:.2f}%")

    print("\n=== Sample Trade History ===")
    for trade in results['trades'][:5]:
        print(f"{trade['entry_time']} → {trade['exit_time']} | {trade['position']} | "
              f"{trade['entry_price']} → {trade['exit_price']} | {trade['pnl_percentage']:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest EMA + VWAP Strategy")
    parser.add_argument('--symbol', default=SYMBOL, help="Trading pair (default from config.ini)")
    parser.add_argument('--start', required=True, help="Start datetime (e.g. 2025-04-01 00:00:00)")
    parser.add_argument('--end', required=True, help="End datetime (e.g. 2025-04-30 00:00:00)")
    parser.add_argument('--tf', default=trade_interval, help="Kline interval (default from config.ini)")
    args = parser.parse_args()

    results = backtest_strategy(args.symbol, args.tf, args.start, args.end)
    print_results(results)
