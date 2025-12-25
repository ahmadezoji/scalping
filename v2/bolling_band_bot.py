#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bollinger Band Mean-Reversion Trading Bot
- Entry on EMA bullish/bearish crossover filtered by VWAP side
- Management with trailing stop, take-profit, and fail-safe stop
- Uses user's index.py helpers (client, place_order, get_klines_all, set_leverage, set_margin_mode, etc.)
- Designed to run as a long-lived service on a server

Requirements:
- python-binance installed (already required by your index.py)
- config.ini present (uses [TRADING] as in your project, optional [STRATEGY])

Author: You + ChatGPT
"""

import asyncio
import signal
import sys
import time
from datetime import datetime, timedelta
import configparser
import logging
from typing import Optional

import pandas as pd
import numpy as np

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from binance_helper import log_and_print, set_margin_mode, set_leverage, get_futures_account_balance, place_order, get_klines_all, client
from vwap_helper import calculate_vwap

# ---- import your helpers -----------------------------------------------------


# ------------------------- Logging -------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# ------------------------- Config --------------------------------------------
CFG = configparser.ConfigParser()
CFG.read("config.ini")

# Trading pair(s)
SYMBOLS = [s.strip() for s in CFG.get("TRADING", "SYMBOL", fallback="BTCUSDT").split(",")]
if not SYMBOLS:
    SYMBOLS = ["BTCUSDT"]

# Force 1-minute trading for this bot
TF = CFG.get("STRATEGY", "timeframe", fallback="1m")

# Strategy params with safe defaults
EMA_FAST = CFG.getint("STRATEGY", "ema_fast", fallback=12)
EMA_SLOW = CFG.getint("STRATEGY", "ema_slow", fallback=26)
USE_VWAP = CFG.get("STRATEGY", "use_vwap_filter", fallback="true").lower() == "true"
VOLUME_CONFIRM = CFG.getfloat("STRATEGY", "volume_confirm", fallback=0.0)  # 0 => off

TRAIL_PCT = CFG.getfloat("STRATEGY", "trail_percent", fallback=0.25) / 100.0   # 0.25%
TP_PCT    = CFG.getfloat("STRATEGY", "tp_percent",    fallback=0.75) / 100.0   # 0.75%
FS_PCT    = CFG.getfloat("STRATEGY", "failsafe_sl_percent", fallback=1.8) / 100.0  # 1.8%

LEVERAGE  = CFG.getint("STRATEGY", "leverage", fallback=2)
RISK_PER_TRADE_PCT = CFG.getfloat("STRATEGY", "risk_per_trade_pct", fallback=0.25) / 100.0
DAILY_LOSS_CAP_PCT = CFG.getfloat("STRATEGY", "daily_loss_cap_pct", fallback=2.0) / 100.0
COOLDOWN_MIN       = CFG.getint("STRATEGY", "cooldown_minutes", fallback=15)

# Management pacing
SIGNAL_LOOP_SEC = CFG.getint("STRATEGY", "signal_poll_seconds", fallback=5)   # check for new candle
RISK_LOOP_SEC   = CFG.getint("STRATEGY", "risk_poll_seconds",   fallback=1)   # manage trailing/TP/FS

# Telegram toggle (uses your index->telegram if needed); here we just reuse log_and_print
def notify(msg: str):
    try:
        log_and_print(msg)
    except Exception:
        logging.info(msg)


# (in momentum_trader_bot.py)
# REPLACE the old bollinger function with this one

# --- Add this helper function for RSI (or import it) ---
def rsi(s: pd.Series, period: int) -> pd.Series:
    delta = s.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# ------------------------- Backtest (Bollinger + RSI) ---------------------------
def backtest_bollinger_strategy(
    symbol=None,
    timeframe=None,
    start_date=None,
    end_date=None,
    entry_balance=None,
    fee_bps=4,
    slippage_bps=1,
    
    # --- NEW: Strategy-specific parameters ---
    bb_window=20,
    bb_std_dev=2.0,
    rsi_period=14,
    rsi_oversold=30,
    rsi_overbought=70,
    tp_pct_cfg=TP_PCT,
    sl_pct_cfg=FS_PCT
):
    """
    Backtest for a Bollinger Band + RSI mean-reversion strategy.
    
    - LONG entry: Price crosses *back inside* from below + RSI oversold.
    - SHORT entry: Price crosses *back inside* from above + RSI overbought.
    - Exits: Fixed TP/SL percentages.
    """
    
    # ---- Resolve config fallbacks ----
    symbol = symbol or SYMBOLS[0]
    timeframe = timeframe or TF
    entry_balance = float(entry_balance or 100.0)

    # ---- Basic parameter validation ----
    if bb_window < 5 or rsi_period < 5:
        return None
    if bb_std_dev <= 0:
        return None
    if rsi_oversold < 0 or rsi_overbought > 100:
        return None
    if rsi_oversold >= rsi_overbought:
        return None
    if tp_pct_cfg <= 0 or sl_pct_cfg <= 0:
        return None
    if sl_pct_cfg >= tp_pct_cfg:
        return None

    tp_pct = tp_pct_cfg * 100.0
    sl_pct = sl_pct_cfg * 100.0

    log_and_print(f"[Backtest] Running for {symbol} on {timeframe}...")
    log_and_print(f"[Backtest] Strategy: Bollinger ({bb_window}, {bb_std_dev}) + RSI ({rsi_period}, {rsi_overbought}/{rsi_oversold})")
    log_and_print(f"[Backtest] Exits: TP={tp_pct:.2f}%, SL={sl_pct:.2f}%")

    # ---- Fetch & prep data ----
    try:
        klines_list = client.get_historical_klines(
            symbol, timeframe, start_str=start_date, end_str=end_date
        )
        df = pd.DataFrame(klines_list, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
    except Exception as e:
        print(f"Error fetching klines: {e}")
        return None

    if df.empty:
        print("No data fetched for backtest.")
        return None

    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col])

    log_and_print(f"[Backtest] Fetched {len(df)} candles from {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")

    # ---- Calculate indicators ----
    df['bb_mid'] = df['close'].rolling(window=bb_window).mean()
    df['bb_std'] = df['close'].rolling(window=bb_window).std()
    df['bb_high'] = df['bb_mid'] + (df['bb_std'] * bb_std_dev)
    df['bb_low'] = df['bb_mid'] - (df['bb_std'] * bb_std_dev)
    df['rsi'] = rsi(df['close'], rsi_period)

    # ---- State ----
    balance = entry_balance
    position = None
    entry_price = 0.0
    trades = []
    wins = 0
    peak_balance = balance
    max_dd = 0.0

    # ---- Iterate candles ----
    for i in range(max(bb_window, rsi_period), len(df)): # Wait for indicators
        prev = df.iloc[i-1]
        cur = df.iloc[i]

        if pd.isna(cur['bb_low']) or pd.isna(cur['rsi']):
            continue

        # --- 1. Manage Open Position (Check Exits) ---
        if position:
            exit_price = None
            exit_reason = None
            
            if position == 'LONG':
                tp_price = entry_price * (1 + tp_pct / 100.0)
                sl_price = entry_price * (1 - sl_pct / 100.0)
                if cur['high'] >= tp_price: exit_price, exit_reason = tp_price, "TakeProfit"
                elif cur['low'] <= sl_price: exit_price, exit_reason = sl_price, "StopLoss"
                    
            elif position == 'SHORT':
                tp_price = entry_price * (1 - tp_pct / 100.0)
                sl_price = entry_price * (1 + sl_pct / 100.0)
                if cur['low'] <= tp_price: exit_price, exit_reason = tp_price, "TakeProfit"
                elif cur['high'] >= sl_price: exit_price, exit_reason = sl_price, "StopLoss"

            if exit_price:
                change_pct = ((exit_price - entry_price) / entry_price * 100.0) if position == 'LONG' else ((entry_price - exit_price) / entry_price * 100.0)
                fees_pct = (fee_bps / 100.0) * 2
                slip_pct = (slippage_bps / 100.0)
                net_pct = change_pct - fees_pct - slip_pct
                balance *= (1.0 + net_pct / 100.0)
                wins += 1 if net_pct > 0 else 0
                trades.append({ "net_pct": float(net_pct), "balance": float(balance) })
                position = None
                if balance > peak_balance: peak_balance = balance
                dd = (peak_balance - balance) / peak_balance * 100.0
                max_dd = max(max_dd, dd)
                continue

        # --- 2. Look for New Entry (if flat) ---
        if position is None:
            
            # --- IMPROVED "SNAP-BACK" + RSI LOGIC ---
            
            # LONG: Price crossed back *above* lower band AND RSI is oversold
            long_sig = (prev['close'] < prev['bb_low']) and \
                       (cur['close'] >= cur['bb_low']) and \
                       (cur['rsi'] < rsi_oversold)
            
            # SHORT: Price crossed back *below* upper band AND RSI is overbought
            short_sig = (prev['close'] > prev['bb_high']) and \
                        (cur['close'] <= cur['bb_high']) and \
                        (cur['rsi'] > rsi_overbought)
            
            # --- END OF SIGNAL LOGIC ---

            if long_sig:
                position = 'LONG'
                entry_price = cur['close']
            elif short_sig:
                position = 'SHORT'
                entry_price = cur['close']

    # ---- Final Report ----
    total_trades = len(trades)
    win_rate = (wins / total_trades * 100.0) if total_trades else 0.0
    total_return_pct = ((balance - entry_balance) / entry_balance * 100.0)
    avg_trade_pct = (np.mean([t["net_pct"] for t in trades]) if trades else 0.0)

    # Return the results *and* the parameters used
    return {
        "strategy_name": "BollingerRSI", # Renamed
        "symbol": symbol,
        "timeframe": timeframe,
        "total_return_pct": float(total_return_pct),
        "total_trades": total_trades,
        "win_rate_pct": float(win_rate),
        "max_drawdown_pct": float(max_dd),
        "avg_trade_pct": float(avg_trade_pct),
        "tp_pct": float(tp_pct),
        "sl_pct": float(sl_pct),
        "bb_window": bb_window,
        "bb_std_dev": bb_std_dev,
        "rsi_period": rsi_period,
        "rsi_oversold": rsi_oversold,
        "rsi_overbought": rsi_overbought,
    }
