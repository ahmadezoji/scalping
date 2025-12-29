#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bollinger Band Mean-Reversion Trading Bot
- Entry on snap-back inside Bollinger Bands + RSI confirmation
- Optional VWAP/volume filters, TP/FailSafe, and trailing stop
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
from datetime import datetime, timedelta, timezone
import configparser
import logging
from typing import Optional

import pandas as pd
import numpy as np

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from binance_helper import log_and_print, set_margin_mode, set_leverage, get_futures_account_balance, place_order, get_klines_all, client
# from vwap_helper import calculate_vwap

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

STRATEGY_SECTION = "STRATEGY_BOLLINGER"
if not CFG.has_section(STRATEGY_SECTION) and CFG.has_section("STRATEGY"):
    STRATEGY_SECTION = "STRATEGY"

# Trading pair(s)
SYMBOLS = [s.strip() for s in CFG.get("TRADING", "SYMBOL", fallback="BTCUSDT").split(",")]
if not SYMBOLS:
    SYMBOLS = ["BTCUSDT"]
ENTRY_USDT = CFG.getfloat("TRADING", "entry_usdt", fallback=0.0)

# Force 1-minute trading for this bot
TF = CFG.get(STRATEGY_SECTION, "timeframe", fallback="1m")

# Strategy params with safe defaults
USE_VWAP = CFG.get(STRATEGY_SECTION, "use_vwap_filter", fallback="false").lower() == "true"
VOLUME_CONFIRM = CFG.getfloat(STRATEGY_SECTION, "volume_confirm", fallback=0.0)  # 0 => off
USE_TRAILING_STOP = CFG.get(STRATEGY_SECTION, "use_trailing_stop", fallback="false").lower() == "true"

TRAIL_PCT = CFG.getfloat(STRATEGY_SECTION, "trail_percent", fallback=0.25) / 100.0   # 0.25%
TP_PCT    = CFG.getfloat(STRATEGY_SECTION, "tp_percent",    fallback=0.75) / 100.0   # 0.75%
FS_PCT    = CFG.getfloat(STRATEGY_SECTION, "failsafe_sl_percent", fallback=1.8) / 100.0  # 1.8%

BB_WINDOW = CFG.getint(STRATEGY_SECTION, "bb_window", fallback=20)
BB_STD_DEV = CFG.getfloat(STRATEGY_SECTION, "bb_std_dev", fallback=2.0)
RSI_PERIOD = CFG.getint(STRATEGY_SECTION, "rsi_period", fallback=14)
RSI_OVERSOLD = CFG.getint(STRATEGY_SECTION, "rsi_oversold", fallback=30)
RSI_OVERBOUGHT = CFG.getint(STRATEGY_SECTION, "rsi_overbought", fallback=70)

LEVERAGE  = CFG.getint(STRATEGY_SECTION, "leverage", fallback=2)
RISK_PER_TRADE_PCT = CFG.getfloat(STRATEGY_SECTION, "risk_per_trade_pct", fallback=0.25) / 100.0
DAILY_LOSS_CAP_PCT = CFG.getfloat(STRATEGY_SECTION, "daily_loss_cap_pct", fallback=2.0) / 100.0
COOLDOWN_MIN       = CFG.getint(STRATEGY_SECTION, "cooldown_minutes", fallback=15)

# Management pacing
SIGNAL_LOOP_SEC = CFG.getint(STRATEGY_SECTION, "signal_poll_seconds", fallback=5)   # check for new candle
RISK_LOOP_SEC   = CFG.getint(STRATEGY_SECTION, "risk_poll_seconds",   fallback=1)   # manage trailing/TP/FS

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

def compute_vwap(df: pd.DataFrame) -> pd.Series:
    if "timestamp" not in df.columns:
        raise ValueError("DataFrame must include 'timestamp'")
    day = df["timestamp"].dt.date
    pv = df["close"] * df["volume"]
    return pv.groupby(day).cumsum() / df["volume"].groupby(day).cumsum()

# ------------------------- Live Trading Helpers ------------------------------
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["bb_mid"] = out["close"].rolling(window=BB_WINDOW).mean()
    out["bb_std"] = out["close"].rolling(window=BB_WINDOW).std()
    out["bb_high"] = out["bb_mid"] + (out["bb_std"] * BB_STD_DEV)
    out["bb_low"] = out["bb_mid"] - (out["bb_std"] * BB_STD_DEV)
    out["rsi"] = rsi(out["close"], RSI_PERIOD)
    if USE_VWAP:
        out["vwap"] = compute_vwap(out)
    return out

# ------------------------- State per symbol -----------------------------------
class PositionState:
    def __init__(self):
        self.side: Optional[str] = None       # 'LONG' or 'SHORT'
        self.entry_price: Optional[float] = None
        self.qty_usdt: float = 0.0            # we pass usdt_amount to place_order
        self.qty: float = 0.0                 # base asset quantity
        self.high_since_entry: Optional[float] = None
        self.low_since_entry: Optional[float] = None
        self.last_candle_time: Optional[pd.Timestamp] = None
        self.daily_start_date: Optional[datetime] = None
        self.daily_pnl: float = 0.0
        self.sl_streak: int = 0
        self.cooldown_until: Optional[datetime] = None

    def reset_intraday_if_needed(self):
        today = datetime.now(timezone.utc).date()
        if self.daily_start_date is None or self.daily_start_date.date() != today:
            self.daily_start_date = datetime.now(timezone.utc)
            self.daily_pnl = 0.0
            self.sl_streak = 0
            self.cooldown_until = None

# ------------------------- Core Bot ------------------------------------------
class BollingerBot:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.state = PositionState()

    @staticmethod
    def _extract_avg_price(order, fallback_price: float) -> float:
        try:
            avg_price = float(order.get("avgPrice") or 0)
            if avg_price > 0:
                return avg_price
        except Exception:
            pass
        try:
            price = float(order.get("price") or 0)
            if price > 0:
                return price
        except Exception:
            pass
        return fallback_price

    async def setup(self):
        set_margin_mode(self.symbol, "ISOLATED")
        set_leverage(self.symbol, LEVERAGE)
        self._sync_position_from_exchange()
        notify(
            f"[{self.symbol}] Ready | TF={TF} BB={BB_WINDOW}/{BB_STD_DEV} RSI={RSI_PERIOD} "
            f"({RSI_OVERSOLD}/{RSI_OVERBOUGHT}) TP={TP_PCT*100:.2f}% FS={FS_PCT*100:.2f}% "
            f"VWAP={USE_VWAP} VolConfirm={VOLUME_CONFIRM} Trail={TRAIL_PCT*100:.2f}% "
            f"UseTrail={USE_TRAILING_STOP} LEV={LEVERAGE} Risk/Trade={RISK_PER_TRADE_PCT*100:.2f}%"
        )

    def _sync_position_from_exchange(self) -> None:
        try:
            positions = client.futures_position_information(symbol=self.symbol)
            if not positions:
                return
            pos = positions[0]
            position_amt = float(pos.get("positionAmt") or 0)
            entry_price = float(pos.get("entryPrice") or 0)
            if position_amt == 0 or entry_price == 0:
                return
            self.state.side = "LONG" if position_amt > 0 else "SHORT"
            self.state.entry_price = entry_price
            self.state.qty = abs(position_amt)
            self.state.qty_usdt = self.state.qty * entry_price
            self.state.high_since_entry = entry_price
            self.state.low_since_entry = entry_price
            notify(f"[{self.symbol}] Synced open position: {self.state.side} qty={self.state.qty} @ {entry_price:.2f}")
        except Exception as e:
            logging.warning(f"[{self.symbol}] Position sync failed: {e}")

    def _position_size_usdt(self, balance_usdt: float, entry_price: float) -> float:
        # Risk model: (position_value / LEVERAGE) * FS_PCT ~= RISK_PER_TRADE_PCT * balance
        if FS_PCT <= 0:
            return 0.0
        pos_value_usdt = (RISK_PER_TRADE_PCT * balance_usdt * LEVERAGE) / FS_PCT
        return max(pos_value_usdt, 0.0)

    async def signal_loop(self):
        """
        Candle-based loop: check for new candle, compute Bollinger + RSI, enter positions.
        """
        while True:
            try:
                self.state.reset_intraday_if_needed()

                account_balance = get_futures_account_balance("USDT")
                if self.state.daily_pnl <= -DAILY_LOSS_CAP_PCT * account_balance:
                    notify(f"[{self.symbol}] Daily loss cap reached. Pausing until next UTC day.")
                    await asyncio.sleep(30)
                    continue

                if self.state.cooldown_until and datetime.now(timezone.utc) < self.state.cooldown_until:
                    await asyncio.sleep(SIGNAL_LOOP_SEC)
                    continue

                df = get_klines_all(self.symbol, TF, limit=200)
                min_len = max(30, BB_WINDOW, RSI_PERIOD) + 2
                if df.empty or len(df) < min_len:
                    await asyncio.sleep(SIGNAL_LOOP_SEC)
                    logging.info(f"[{self.symbol}] Not enough data fetched (have {len(df)}, need {min_len}).")
                    continue

                data = compute_indicators(df.iloc[:-1])
                if len(data) < 2:
                    await asyncio.sleep(SIGNAL_LOOP_SEC)
                    logging.info(f"[{self.symbol}] Not enough data after indicators.")
                    continue

                last_candle_time = data["timestamp"].iloc[-1]
                if self.state.last_candle_time is not None and last_candle_time == self.state.last_candle_time:
                    await asyncio.sleep(SIGNAL_LOOP_SEC)
                    logging.info(f"[{self.symbol}] No new candle yet.")
                    continue
                self.state.last_candle_time = last_candle_time

                prev, cur = data.iloc[-2], data.iloc[-1]
                if pd.isna(cur["bb_low"]) or pd.isna(cur["rsi"]):
                    await asyncio.sleep(SIGNAL_LOOP_SEC)
                    continue

                base_long = (prev["close"] < prev["bb_low"]) and (cur["close"] >= cur["bb_low"]) and (cur["rsi"] < RSI_OVERSOLD)
                base_short = (prev["close"] > prev["bb_high"]) and (cur["close"] <= cur["bb_high"]) and (cur["rsi"] > RSI_OVERBOUGHT)
                if USE_VWAP and "vwap" in cur:
                    vwap_ok_long = cur["close"] <= cur["vwap"]
                    vwap_ok_short = cur["close"] >= cur["vwap"]
                else:
                    vwap_ok_long = True
                    vwap_ok_short = True
                if VOLUME_CONFIRM > 0:
                    vol_avg = data["volume"].tail(20).mean()
                    vol_ok = cur["volume"] >= VOLUME_CONFIRM * vol_avg
                else:
                    vol_avg = None
                    vol_ok = True
                long_sig = base_long and vwap_ok_long and vol_ok
                short_sig = base_short and vwap_ok_short and vol_ok

                logging.info(
                    f"[{self.symbol}] t={cur.name} close={cur.close:.2f} "
                    f"bb_low={cur.bb_low:.2f} bb_high={cur.bb_high:.2f} rsi={cur.rsi:.2f} "
                    f"vwap={getattr(cur,'vwap',float('nan')):.2f} vol={cur['volume']:.2f} "
                    f"vwap_ok_long={vwap_ok_long} vwap_ok_short={vwap_ok_short} vol_ok={vol_ok} "
                    f"(mult={VOLUME_CONFIRM}, avg20={vol_avg if vol_avg else 0}) "
                    f"base_long={base_long} base_short={base_short} long_sig={long_sig} short_sig={short_sig}"
                )

                if self.state.side is None and (long_sig or short_sig):
                    self._sync_position_from_exchange()
                    if self.state.side is not None:
                        await asyncio.sleep(SIGNAL_LOOP_SEC)
                        logging.info(f"[{self.symbol}] Position opened externally, skipping entry.")
                        continue
                    side = "BUY" if long_sig else "SELL"
                    entry_price = float(cur["close"])
                    if ENTRY_USDT and ENTRY_USDT > 0:
                        usdt_amt = ENTRY_USDT
                    else:
                        balance = get_futures_account_balance("USDT")
                        usdt_amt = self._position_size_usdt(balance, entry_price)

                    if usdt_amt <= 5:
                        notify(f"[{self.symbol}] Skip entry: size too small (usdt={usdt_amt:.2f})")
                        await asyncio.sleep(SIGNAL_LOOP_SEC)
                        logging.info(f"[{self.symbol}] Position size too small, skipping entry.")
                        continue

                    order = place_order(self.symbol, side, usdt_amount=usdt_amt)
                    if order:
                        entry_price = self._extract_avg_price(order, entry_price)
                        qty = float(order.get("executedQty") or order.get("origQty") or 0)
                        if qty <= 0:
                            qty = usdt_amt / entry_price
                        self.state.side = "LONG" if long_sig else "SHORT"
                        self.state.entry_price = entry_price
                        self.state.qty_usdt = usdt_amt
                        self.state.qty = qty
                        self.state.high_since_entry = entry_price
                        self.state.low_since_entry = entry_price
                        notify(f"[{self.symbol}] ENTER {self.state.side} @ {entry_price:.2f} "
                               f"(notional ~{usdt_amt:.2f} USDT)")
            except Exception as e:
                logging.exception(f"[{self.symbol}] signal_loop error: {e}")
            await asyncio.sleep(SIGNAL_LOOP_SEC)

    async def risk_loop(self):
        """
        Fast loop: manage take-profit and fail-safe.
        Matches the backtest exits (TP/SL only).
        """
        while True:
            try:
                if self.state.side is None:
                    await asyncio.sleep(RISK_LOOP_SEC)
                    continue
                if self.state.entry_price is None:
                    self._sync_position_from_exchange()
                    await asyncio.sleep(RISK_LOOP_SEC)
                    continue

                ticker = client.futures_symbol_ticker(symbol=self.symbol)
                last_price = float(ticker["price"])

                if self.state.side == "LONG":
                    if self.state.high_since_entry is None:
                        self.state.high_since_entry = last_price
                    self.state.high_since_entry = max(self.state.high_since_entry, last_price)
                    tp_price = self.state.entry_price * (1 + TP_PCT)
                    if last_price >= tp_price:
                        await self._exit_market("TakeProfit", last_price)
                        continue
                    if USE_TRAILING_STOP and TRAIL_PCT > 0:
                        trail_price = self.state.high_since_entry * (1 - TRAIL_PCT)
                        if last_price <= trail_price:
                            await self._exit_market("TrailingStop", last_price)
                            continue
                    sl_price = self.state.entry_price * (1 - FS_PCT)
                    if last_price <= sl_price:
                        await self._exit_market("FailSafe", last_price)
                        continue

                elif self.state.side == "SHORT":
                    if self.state.low_since_entry is None:
                        self.state.low_since_entry = last_price
                    self.state.low_since_entry = min(self.state.low_since_entry, last_price)
                    tp_price = self.state.entry_price * (1 - TP_PCT)
                    if last_price <= tp_price:
                        await self._exit_market("TakeProfit", last_price)
                        continue
                    if USE_TRAILING_STOP and TRAIL_PCT > 0:
                        trail_price = self.state.low_since_entry * (1 + TRAIL_PCT)
                        if last_price >= trail_price:
                            await self._exit_market("TrailingStop", last_price)
                            continue
                    sl_price = self.state.entry_price * (1 + FS_PCT)
                    if last_price >= sl_price:
                        await self._exit_market("FailSafe", last_price)
                        continue

            except Exception as e:
                logging.exception(f"[{self.symbol}] risk_loop error: {e}")

            await asyncio.sleep(RISK_LOOP_SEC)

    async def _exit_market(self, reason: str, last_price: float):
        out_side = "SELL" if self.state.side == "LONG" else "BUY"
        if self.state.qty > 0:
            usdt_amt = self.state.qty * last_price
        else:
            usdt_amt = self.state.qty_usdt
        order = place_order(self.symbol, out_side, usdt_amount=usdt_amt, reduce_only=True)
        if order:
            exit_price = self._extract_avg_price(order, last_price)
            pnl_pct = ((exit_price - self.state.entry_price) / self.state.entry_price) if self.state.side == "LONG" \
                      else ((self.state.entry_price - exit_price) / self.state.entry_price)
            pnl_pct *= 100.0
            self.state.daily_pnl += (pnl_pct / 100.0) * (usdt_amt / LEVERAGE)

            if reason in ("FailSafe", "TrailingStop") and pnl_pct < 0:
                self.state.sl_streak += 1
                if self.state.sl_streak >= 3:
                    self.state.cooldown_until = datetime.now(timezone.utc) + timedelta(minutes=COOLDOWN_MIN)
                    notify(f"[{self.symbol}] SL streak {self.state.sl_streak}. Cooldown until {self.state.cooldown_until}.")

            if reason == "TakeProfit":
                self.state.sl_streak = 0

            notify(f"[{self.symbol}] EXIT ({reason}) {self.state.side} @ {exit_price:.2f} | "
                   f"PnL≈{pnl_pct:.2f}%  DailyPnL≈{self.state.daily_pnl:.2f} USDT")

            self.state.side = None
            self.state.entry_price = None
            self.state.qty_usdt = 0.0
            self.state.qty = 0.0
            self.state.high_since_entry = None
            self.state.low_since_entry = None

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
    bb_window=BB_WINDOW,
    bb_std_dev=BB_STD_DEV,
    rsi_period=RSI_PERIOD,
    rsi_oversold=RSI_OVERSOLD,
    rsi_overbought=RSI_OVERBOUGHT,
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

# ------------------------- Runner --------------------------------------------
async def run():
    bots = [BollingerBot(sym) for sym in SYMBOLS]
    for b in bots:
        await b.setup()

    tasks = []
    for b in bots:
        tasks.append(asyncio.create_task(b.signal_loop(), name=f"{b.symbol}-signal"))
        tasks.append(asyncio.create_task(b.risk_loop(),   name=f"{b.symbol}-risk"))

    stop_event = asyncio.Event()

    def _stop(*_):
        notify("Shutdown signal received, cancelling tasks...")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            asyncio.get_running_loop().add_signal_handler(sig, _stop)
        except NotImplementedError:
            signal.signal(sig, _stop)

    await stop_event.wait()
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    notify("All tasks stopped. Bye.")

if __name__ == "__main__":
    try:
        RUN_LIVE = True
        if RUN_LIVE:
            asyncio.run(run())
    except KeyboardInterrupt:
        pass
