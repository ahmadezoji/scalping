from __future__ import annotations
"""
EMA + VWAP Scalping Strategy (Config‑driven)
==========================================
This module back‑tests a short‑time‑frame scalping strategy that combines:
  • Exponential Moving Average (EMA)
  • Volume‑Weighted Average Price (VWAP)
  • RSI / Stochastic filters

All tunable parameters are read **only** from `config.ini` so you can tweak the
strategy without touching code.  The *only* hard‑coded values are the historical
period you want to back‑test over – edit the two lines in **main** at the bottom
of the file.

Quick run (nothing to pass on the CLI):
$ python ema_vwap.py

The script prints a concise performance summary and the first few trades.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List

import configparser
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# 1️⃣  Load configuration -----------------------------------------------------
# ---------------------------------------------------------------------------
CFG_FILE = Path(__file__).with_name("config.ini")
if not CFG_FILE.exists():
    raise FileNotFoundError("config.ini not found next to ema_vwap.py")

_cfg = configparser.ConfigParser()
_cfg.read(CFG_FILE)

TRADING = _cfg["TRADING"]
STRATEGY = _cfg["STRATEGY"] if "STRATEGY" in _cfg else {}

# ✨  Pull parameters ---------------------------------------------------------
SYMBOL        = TRADING.get("SYMBOL", "BTCUSDT")
TIMEFRAME     = TRADING.get("trade_interval", "1m")
ENTRY_USDT    = float(TRADING.get("entry_usdt", 100))
TP_PCT        = float(TRADING.get("tp_percentage", 0.8))
SL_PCT        = float(TRADING.get("sl_percentage", 0.5))

EMA_SPAN      = int(STRATEGY.get("ema_span", 50))
VWAP_BUFFER   = float(STRATEGY.get("vwap_buffer", 0.002))
RSI_GATE      = int(STRATEGY.get("rsi_gate", 50))

# ---------------------------------------------------------------------------
# 2️⃣  Indicator helpers ------------------------------------------------------
# ---------------------------------------------------------------------------

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute EMA, VWAP, ATR, Stoch‑%K/%D and RSI."""
    delta = df["close"].diff()
    gain  = np.where(delta > 0, delta, 0)
    loss  = np.where(delta < 0, -delta, 0)

    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean()
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean()
    avg_gain = avg_gain.shift().fillna(0) * 13/14 + gain/14
    avg_loss = avg_loss.shift().fillna(0) * 13/14 + loss/14
    rs = avg_gain / (avg_loss + 1e-9)
    df["rsi"] = 100 - 100 / (1 + rs)

    df["ema_trend"] = df["close"].ewm(span=EMA_SPAN, adjust=False).mean()
    df["vwap"]      = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()

    hl   = df["high"] - df["low"]
    hc   = (df["high"] - df["close"].shift()).abs()
    lc   = (df["low"]  - df["close"].shift()).abs()
    df["tr"]  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df["atr"] = df["tr"].rolling(14).mean()

    low_min  = df["low"].rolling(14).min()
    high_max = df["high"].rolling(14).max()
    df["stoch_k"] = (df["close"] - low_min) / (high_max - low_min + 1e-9) * 100
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()
    return df

# ---------------------------------------------------------------------------
# 3️⃣  Back‑test core ---------------------------------------------------------
# ---------------------------------------------------------------------------

def fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Lightweight wrapper around Binance REST – requires *python‑binance*."""
    from binance.client import Client  # local import keeps package optional

    # You can safely set dummy keys for historical data endpoints
    client = Client("x", "y", testnet=False)
    raw = client.get_historical_klines(symbol, interval, start_ms, end_ms)

    df = pd.DataFrame(raw, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_base", "taker_quote", "ignore",
    ])
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


def backtest(start: str, end: str) -> Dict[str, float | int | List[dict]]:
    start_ms = int(datetime.strptime(start, "%Y-%m-%d %H:%M:%S").timestamp() * 1000)
    end_ms   = int(datetime.strptime(end, "%Y-%m-%d %H:%M:%S").timestamp() * 1000)

    df = fetch_klines(SYMBOL, TIMEFRAME, start_ms, end_ms)
    df = calculate_indicators(df)

    trades: List[dict] = []
    balance           = ENTRY_USDT
    peak_balance      = ENTRY_USDT
    max_dd            = 0.0
    win, total        = 0, 0
    position          = None  # "LONG" / "SHORT"
    entry_price       = 0.0

    for i, row in df.iterrows():
        if i < 20:
            continue  # warm‑up

        price = row["close"]
        signal = None
        if (price > row["vwap"] * (1 - VWAP_BUFFER) and row["stoch_k"] < 70 and
                row["rsi"] > RSI_GATE and price > row["ema_trend"]):
            signal = "LONG"
        elif (price < row["vwap"] * (1 + VWAP_BUFFER) and row["stoch_k"] > 30 and
                  row["rsi"] < 100 - RSI_GATE and price < row["ema_trend"]):
            signal = "SHORT"

        # Manage open position ------------------------------------------------
        if position is not None:
            pnl_pct = ((price - entry_price) / entry_price * 100) if position == "LONG" else ((entry_price - price) / entry_price * 100)
            if pnl_pct >= TP_PCT or pnl_pct <= -SL_PCT:
                total += 1
                if pnl_pct > 0:
                    win += 1
                balance *= 1 + pnl_pct / 100
                peak_balance = max(peak_balance, balance)
                max_dd = max(max_dd, (peak_balance - balance) / peak_balance * 100)
                trades.append({
                    "side": position,
                    "entry": entry_time,
                    "exit": row["timestamp"],
                    "entry_px": entry_price,
                    "exit_px": price,
                    "pnl_pct": pnl_pct,
                })
                position = None

        # Enter new trade -----------------------------------------------------
        if position is None and signal is not None:
            position    = signal
            entry_price = price
            entry_time  = row["timestamp"]

    win_rate = win / total * 100 if total else 0
    return {
        "total_trades": total,
        "wins": win,
        "win_rate": win_rate,
        "final_balance": balance,
        "total_return_pct": (balance - ENTRY_USDT) / ENTRY_USDT * 100,
        "max_drawdown": max_dd,
        "trades": trades,
    }

# ---------------------------------------------------------------------------
# 4️⃣  Pretty print -----------------------------------------------------------
# ---------------------------------------------------------------------------

def print_report(r: Dict[str, float | int | List[dict]]):
    print("\n=== EMA + VWAP Back‑test ===")
    print(f"Symbol           : {SYMBOL}")
    print(f"Time‑frame       : {TIMEFRAME}")
    print("----------------------------------")
    print(f"Total trades     : {r['total_trades']}")
    print(f"Wins             : {r['wins']}")
    print(f"Win‑rate         : {r['win_rate']:.2f} %")
    print(f"Total return     : {r['total_return_pct']:.2f} %")
    print(f"Final balance    : {r['final_balance']:.2f} USDT")
    print(f"Max draw‑down    : {r['max_drawdown']:.2f} %")

    print("\nFirst few trades:")
    for t in r["trades"][:5]:
        print(f" {t['side']:<5}  {t['entry']:%Y-%m-%d %H:%M} -> {t['exit']:%Y-%m-%d %H:%M}  {t['pnl_pct']:+6.2f} %")
    if len(r["trades"]) > 5:
        print(" …and more …")

# ---------------------------------------------------------------------------
# 5️⃣  Main ------------------------------------------------------------------
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 🔴 Edit these two lines only – everything else comes from config.ini
    START_DATE = "2024-01-01 00:00:00"
    END_DATE   = "2024-01-28 23:59:00"

    report = backtest(START_DATE, END_DATE)
    print_report(report)
