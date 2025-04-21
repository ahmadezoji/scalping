import asyncio
import logging
from datetime import datetime
import pandas as pd
import numpy as np

from index import (
    SYMBOL,
    tp_percentage,
    sl_percentage,
    entry_usdt,
    trade_interval,
    sleep_time,
    tp_sl_check_interval,
    client,
    get_klines_all,
    get_futures_account_balance,
    place_order,
    close_futures_position,
    calculate_max_quantity,
    set_leverage,
    set_margin_mode,
    log_and_print,
)
from telegram import send_telegram_message

# ------------------------
# Strategy parameters
# ------------------------
EMA_SPAN = 50  # look‑back for trend filter
VWAP_BUFFER = 0.002  # 0.2 % buffer around VWAP to define pull‑back zone
ATR_PERIOD = 14      # for volatility filter (optional)
RSI_PERIOD = 14      # for momentum confirmation (optional)

# Global state (kept in‑memory while bot runs)
current_position: str | None = None  # "LONG", "SHORT" or None
entry_price: float = 0.0
entry_quantity: float = 0.0

# -------------------------------------------------
# Indicator helper
# -------------------------------------------------

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add EMA, VWAP, ATR, RSI to the DataFrame (in‑place)."""
    # --- EMA (trend) ---
    df["ema_trend"] = df["close"].ewm(span=EMA_SPAN, adjust=False).mean()

    # --- VWAP (session cumulative) ---
    df["vwap"] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()

    # --- ATR (volatility) ---
    high_low = df["high"] - df["low"]
    high_close = np.abs(df["high"] - df["close"].shift(1))
    low_close = np.abs(df["low"] - df["close"].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = tr.rolling(ATR_PERIOD).mean()

    # --- RSI (momentum) ---
    delta = df["close"].diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    avg_gain = pd.Series(gain).rolling(RSI_PERIOD).mean()
    avg_loss = pd.Series(loss).rolling(RSI_PERIOD).mean()

    rs = avg_gain / (avg_loss + 1e-10)
    df["rsi"] = 100 - (100 / (1 + rs))

    return df

# -------------------------------------------------
# Signal logic
# -------------------------------------------------

def generate_signal(row: pd.Series) -> str | None:
    """Return "LONG", "SHORT" or None based on EMA–VWAP pull‑back rules."""
    price = row["close"]
    trend_ema = row["ema_trend"]
    vwap = row["vwap"]
    rsi = row["rsi"]

    # Up‑trend — price above EMA trend line
    if price > trend_ema:
        # Wait for bullish pull‑back to VWAP zone (within buffer below VWAP)
        if price <= vwap * (1 + VWAP_BUFFER) and price >= vwap * (1 - VWAP_BUFFER):
            # Optional momentum confirmation
            if rsi > 50:
                return "LONG"

    # Down‑trend — price below EMA trend line
    if price < trend_ema:
        # Wait for bearish pull‑back to VWAP zone (within buffer above VWAP)
        if price >= vwap * (1 - VWAP_BUFFER) and price <= vwap * (1 + VWAP_BUFFER):
            if rsi < 50:
                return "SHORT"

    return None

# -------------------------------------------------
# Live trading task (infinite loop)
# -------------------------------------------------

async def trade_logic():
    global current_position, entry_price, entry_quantity
    symbol = SYMBOL
    interval = trade_interval  # e.g. "5m"
    pause_minutes = sleep_time

    # Set margin preferences once at start (silent failures tolerated)
    set_leverage(symbol, leverage=3)
    set_margin_mode(symbol, margin_type="ISOLATED")

    log_and_print("Starting EMA‑VWAP pull‑back bot …")
    send_telegram_message("EMA‑VWAP bot launched 🚀")

    while True:
        try:
            df = get_klines_all(symbol, interval, limit=EMA_SPAN + 20)
            if df.empty:
                log_and_print("No data fetched; retrying …")
                await asyncio.sleep(pause_minutes * 60)
                continue

            df = compute_indicators(df)
            row = df.iloc[-1]
            signal = generate_signal(row)

            # ---------- manage existing position ----------
            if current_position is not None:
                latest_price = row["close"]
                pnl_pct = (
                    (latest_price - entry_price) / entry_price * 100
                    if current_position == "LONG"
                    else (entry_price - latest_price) / entry_price * 100
                )

                if (
                    (current_position == "LONG" and pnl_pct >= tp_percentage)
                    or (current_position == "SHORT" and pnl_pct >= tp_percentage)
                ):
                    # Take‑profit hit
                    close_futures_position(symbol, current_position, entry_quantity)
                    log_and_print(f"TP hit ({pnl_pct:.2f} %) — position closed")
                    send_telegram_message(
                        f"✅ TP ({pnl_pct:.2f} %) hit, {current_position} closed at {latest_price}"
                    )
                    current_position = None
                    continue  # evaluate again next loop

                if (
                    (current_position == "LONG" and pnl_pct <= -sl_percentage)
                    or (current_position == "SHORT" and pnl_pct <= -sl_percentage)
                ):
                    # Stop‑loss hit
                    close_futures_position(symbol, current_position, entry_quantity)
                    log_and_print(f"SL hit ({pnl_pct:.2f} %) — position closed")
                    send_telegram_message(
                        f"🛑 SL ({pnl_pct:.2f} %) hit, {current_position} closed at {latest_price}"
                    )
                    current_position = None
                    continue

            # ---------- open new position ----------
            if signal and current_position is None:
                balance = get_futures_account_balance("USDT")
                if balance < 20:
                    log_and_print("Insufficient balance; stopping bot.")
                    send_telegram_message("Insufficient balance — bot stopped.")
                    break

                # Use fixed entry amount or full balance if less
                entry_capital = min(entry_usdt, balance)
                last_price = row["close"]
                leverage_info = client.futures_leverage_bracket(symbol=symbol)[0]
                leverage = leverage_info["brackets"][0]["initialLeverage"]
                qty = calculate_max_quantity(entry_capital, leverage, last_price)

                side = "BUY" if signal == "LONG" else "SELL"
                order = place_order(symbol, side, entry_capital)
                if order:
                    current_position = signal
                    entry_price = last_price
                    entry_quantity = qty
                    log_and_print(f"{signal} opened @ {entry_price} qty {entry_quantity}")
                    send_telegram_message(
                        f"📈 New {signal} opened @ {entry_price:.2f} qty {entry_quantity:.4f}"
                    )
        except Exception as exc:
            log_and_print(f"Error in trade loop: {exc}")

        await asyncio.sleep(pause_minutes * 60)

# -------------------------------------------------
# Historical back‑tester
# -------------------------------------------------

def backtest_strategy(symbol: str, timeframe: str, start_date: str, end_date: str, starting_balance: float = 100.0) -> dict:
    """Run a vectorised back‑test of the EMA‑VWAP pull‑back strategy."""
    # Download historical klines
    start_ts = int(pd.Timestamp(start_date).timestamp() * 1000)
    end_ts = int(pd.Timestamp(end_date).timestamp() * 1000)

    raw = client.futures_historical_klines(symbol, timeframe, start_str=start_ts, end_str=end_ts)
    if not raw:
        raise RuntimeError("No historical data returned by Binance API")

    cols = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "trades_count",
        "taker_buy_volume",
        "taker_buy_quote_volume",
        "ignore",
    ]
    df = pd.DataFrame(raw, columns=cols)
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

    df = compute_indicators(df)

    position = None
    entry_price = 0.0
    balance = starting_balance
    equity_curve = []
    trades = []

    for i in range(EMA_SPAN + 20, len(df)):
        row = df.iloc[i]
        signal = generate_signal(row)

        # manage open position
        if position is not None:
            price = row["close"]
            pnl_pct = (
                (price - entry_price) / entry_price * 100
                if position == "LONG"
                else (entry_price - price) / entry_price * 100
            )

            if pnl_pct >= tp_percentage or pnl_pct <= -sl_percentage:
                # close
                balance *= 1 + pnl_pct / 100
                trades.append({
                    "entry_time": entry_time,
                    "exit_time": row["timestamp"],
                    "position": position,
                    "entry_price": entry_price,
                    "exit_price": price,
                    "pnl_pct": pnl_pct,
                })
                position = None

        # open new position
        if position is None and signal is not None:
            position = signal
            entry_price = row["close"]
            entry_time = row["timestamp"]

        equity_curve.append(balance)

    # metrics
    total_trades = len(trades)
    wins = sum(1 for t in trades if t["pnl_pct"] > 0)
    win_rate = wins / total_trades * 100 if total_trades else 0
    avg_profit = np.mean([t["pnl_pct"] for t in trades]) if trades else 0
    max_drawdown = 0
    peak = starting_balance
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak * 100
        max_drawdown = max(max_drawdown, dd)

    return {
        "total_trades": total_trades,
        "winning_trades": wins,
        "win_rate": win_rate,
        "average_profit_per_trade": avg_profit,
        "total_return_pct": (balance - starting_balance) / starting_balance * 100,
        "max_drawdown_pct": max_drawdown,
        "final_balance": balance,
        "trades": trades,
    }

# -------------------------------------------------
# Utility — nicely print results in CLI
# -------------------------------------------------

def print_backtest_results(results: dict):
    print("\n=== Back‑test Results ===")
    print(f"Total Trades       : {results['total_trades']}")
    print(f"Winning Trades     : {results['winning_trades']}")
    print(f"Win Rate           : {results['win_rate']:.2f}%")
    print(f"Avg Profit/Trade   : {results['average_profit_per_trade']:.2f}%")
    print(f"Total Return       : {results['total_return_pct']:.2f}%")
    print(f"Max Draw‑down      : {results['max_drawdown_pct']:.2f}%")
    print(f"Final Balance      : {results['final_balance']:.2f} USDT\n")

    if results["trades"]:
        print("First 5 trades …")
        for t in results["trades"][:5]:
            print(
                f"{t['position']:5} | {t['entry_time']} → {t['exit_time']} | {t['pnl_pct']:.2f}%"
            )

# -------------------------------------------------
# Calculate win rate for each 5-day period
# -------------------------------------------------

def calculate_winrate_per_5_days(symbol: str, timeframes: list[str], sl_tp_combinations: list[tuple[float, float]]):
    """Calculate win rate for each 5-day period in April 2025."""
    start_date = pd.Timestamp("2025-04-01")
    end_date = pd.Timestamp("2025-04-30")
    results = []

    while start_date < end_date:
        next_date = start_date + pd.Timedelta(days=5)
        for timeframe in timeframes:
            for sl, tp in sl_tp_combinations:
                global sl_percentage, tp_percentage
                sl_percentage = sl
                tp_percentage = tp

                try:
                    res = backtest_strategy(
                        symbol,
                        timeframe,
                        start_date.strftime("%Y-%m-%d %H:%M:%S"),
                        next_date.strftime("%Y-%m-%d %H:%M:%S"),
                        starting_balance=100.0,
                    )
                    results.append({
                        "start_date": start_date.strftime("%Y-%m-%d"),
                        "end_date": next_date.strftime("%Y-%m-%d"),
                        "timeframe": timeframe,
                        "sl_percentage": sl,
                        "tp_percentage": tp,
                        "win_rate": res["win_rate"],
                        "total_return_pct": res["total_return_pct"],
                        "total_trades": res["total_trades"],
                    })
                except Exception as e:
                    print(f"Error for {start_date} to {next_date}, {timeframe}, SL={sl}, TP={tp}: {e}")

        start_date = next_date

    # Print results
    print("\n=== Win Rate Results for April 2025 ===")
    for result in results:
        print(
            f"Period: {result['start_date']} to {result['end_date']} | "
            f"Timeframe: {result['timeframe']} | SL: {result['sl_percentage']}% | TP: {result['tp_percentage']}% | "
            f"Win Rate: {result['win_rate']:.2f}% | Total Return: {result['total_return_pct']:.2f}% | "
            f"Total Trades: {result['total_trades']}"
        )

# -----------------------------
# Quick manual test runner
# -----------------------------
if __name__ == "__main__":
    # example back‑test run (adjust dates & timeframe) — disabled for safety
    # res = backtest_strategy(SYMBOL, trade_interval, "2025-03-01 00:00:00", "2025-03-10 23:59:00")
    # print_backtest_results(res)

    # Example: Calculate win rate for each 5-day period in April 2025
    timeframes = ["5m", "15m"]
    sl_tp_combinations = [(0.5, 0.8), (1.0, 1.5), (1.5, 2.0)]  # SL and TP percentages to test
    calculate_winrate_per_5_days(SYMBOL, timeframes, sl_tp_combinations)

    # To activate live trading:
    # asyncio.run(trade_logic())
    pass
