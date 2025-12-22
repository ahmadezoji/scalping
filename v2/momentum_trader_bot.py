#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Momentum Trader Bot (EMA12/EMA26 + VWAP filter) for Binance Futures
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
import random
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

# ------------------------- Utils ---------------------------------------------
def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def compute_vwap(df: pd.DataFrame) -> pd.Series:
    # Intraday VWAP reset each calendar day
    if "timestamp" not in df.columns:
        raise ValueError("DataFrame must include 'timestamp'")
    day = df["timestamp"].dt.date
    pv = df["close"] * df["volume"]
    return pv.groupby(day).cumsum() / df["volume"].groupby(day).cumsum()

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ema_fast"] = ema(out["close"], EMA_FAST)
    out["ema_slow"] = ema(out["close"], EMA_SLOW)
    if USE_VWAP:
        out["vwap"] = compute_vwap(out)
    return out

def is_bull_cross(prev_row, row) -> bool:
    return prev_row.ema_fast <= prev_row.ema_slow and row.ema_fast > row.ema_slow

def is_bear_cross(prev_row, row) -> bool:
    return prev_row.ema_fast >= prev_row.ema_slow and row.ema_fast < row.ema_slow

def now_ms() -> int:
    return int(time.time() * 1000)

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
        today = datetime.utcnow().date()
        if self.daily_start_date is None or self.daily_start_date.date() != today:
            self.daily_start_date = datetime.utcnow()
            self.daily_pnl = 0.0
            self.sl_streak = 0
            self.cooldown_until = None

# ------------------------- Core Bot ------------------------------------------
class MomentumBot:
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
        # Margin/leverage one time
        set_margin_mode(self.symbol, "ISOLATED")
        set_leverage(self.symbol, LEVERAGE)
        self._sync_position_from_exchange()
        notify(f"[{self.symbol}] Ready | TF={TF} EMA{EMA_FAST}/{EMA_SLOW} VWAP={USE_VWAP} "
               f"TRAIL={TRAIL_PCT*100:.2f}% TP={TP_PCT*100:.2f}% FS={FS_PCT*100:.2f}% "
               f"LEV={LEVERAGE} Risk/Trade={RISK_PER_TRADE_PCT*100:.2f}%")

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
        # => position_value_usdt ≈ (RISK_PER_TRADE_PCT * balance * LEVERAGE) / FS_PCT
        if FS_PCT <= 0:
            return 0.0
        pos_value_usdt = (RISK_PER_TRADE_PCT * balance_usdt * LEVERAGE) / FS_PCT
        # clip tiny positions
        return max(pos_value_usdt, 0.0)

   

    async def signal_loop(self):
        """
        Candle-based loop: check for new 1m candle, compute crossover + VWAP filter,
        enter positions if eligible.
        """
        while True:
            try:
                self.state.reset_intraday_if_needed()

                # Daily loss cap check
                account_balance = get_futures_account_balance("USDT")
                if self.state.daily_pnl <= -DAILY_LOSS_CAP_PCT * account_balance:
                    notify(f"[{self.symbol}] Daily loss cap reached. Pausing until next UTC day.")
                    await asyncio.sleep(30)
                    continue

                # Cooldown after SL streak
                if self.state.cooldown_until and datetime.utcnow() < self.state.cooldown_until:
                    await asyncio.sleep(SIGNAL_LOOP_SEC)
                    continue

                df = get_klines_all(self.symbol, TF, limit=200)
                min_len = max(30, EMA_SLOW) + 2  # warmup + last open + prev/cur
                if df.empty or len(df) < min_len:
                    await asyncio.sleep(SIGNAL_LOOP_SEC)
                    continue

                # Only act on NEWLY CLOSED candle
                data = compute_indicators(df.iloc[:-1])
                if len(data) < 2:
                    await asyncio.sleep(SIGNAL_LOOP_SEC)
                    continue

                last_candle_time = data["timestamp"].iloc[-1]
                if self.state.last_candle_time is not None and last_candle_time == self.state.last_candle_time:
                    await asyncio.sleep(SIGNAL_LOOP_SEC)
                    continue
                self.state.last_candle_time = last_candle_time

                prev, cur = data.iloc[-2], data.iloc[-1]

                # --- DEBUG: show all gatekeepers per candle ---
                cross_up   = is_bull_cross(prev, cur)
                cross_down = is_bear_cross(prev, cur)
                vwap_side  = (not USE_VWAP) or (('vwap' in cur) and ((cur.close > cur.vwap) if cross_up else (cur.close < cur.vwap)))
                if VOLUME_CONFIRM > 0:
                    vol_avg = data["volume"].tail(20).mean()
                    vol_ok  = cur["volume"] >= VOLUME_CONFIRM * vol_avg
                else:
                    vol_avg = None
                    vol_ok  = True

                logging.info(
                    f"[{self.symbol}] t={cur.name} "
                    f"close={cur.close:.2f} ema_fast={cur.ema_fast:.2f} ema_slow={cur.ema_slow:.2f} "
                    f"VWAP={getattr(cur,'vwap',float('nan')):.2f} vol={cur['volume']:.2f} "
                    f"cross_up={cross_up} cross_down={cross_down} "
                    f"vwap_ok={vwap_side} vol_ok={vol_ok} (mult={VOLUME_CONFIRM}, avg20={vol_avg if vol_avg else 0})"
                )

                # Signals using debug variables
                long_sig  = cross_up   and vwap_side and vol_ok
                short_sig = cross_down and vwap_side and vol_ok

                # Enter if flat
                if self.state.side is None and (long_sig or short_sig):
                    self._sync_position_from_exchange()
                    if self.state.side is not None:
                        await asyncio.sleep(SIGNAL_LOOP_SEC)
                        continue
                    side = "BUY" if long_sig else "SELL"
                    entry_price = float(cur["close"])
                    balance = get_futures_account_balance("USDT")
                    usdt_amt = self._position_size_usdt(balance, entry_price)

                    if usdt_amt <= 5:  # ignore dust
                        notify(f"[{self.symbol}] Skip entry: size too small (usdt={usdt_amt:.2f})")
                        await asyncio.sleep(SIGNAL_LOOP_SEC)
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
        
        NOTE: This is modified to *exactly match the backtest*.
        It ONLY uses the fixed TP_PCT and FS_PCT.
        The Trailing Stop (TRAIL_PCT) is disabled.
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

                # last price from ticker (fast)
                ticker = client.futures_symbol_ticker(symbol=self.symbol)
                last_price = float(ticker["price"])

                # --- Exit Logic (Matches Backtest) ---
                if self.state.side == "LONG":
                    # take profit: gain from entry TP_PCT
                    tp_price = self.state.entry_price * (1 + TP_PCT)
                    if last_price >= tp_price:
                        await self._exit_market("TakeProfit", last_price)
                        continue
                        
                    # fail-safe: drop from entry FS_PCT
                    sl_price = self.state.entry_price * (1 - FS_PCT)
                    if last_price <= sl_price:
                        await self._exit_market("FailSafe", last_price)
                        continue

                elif self.state.side == "SHORT":
                    # take profit: drop from entry TP_PCT
                    tp_price = self.state.entry_price * (1 - TP_PCT)
                    if last_price <= tp_price:
                        await self._exit_market("TakeProfit", last_price)
                        continue
                        
                    # fail-safe: rise from entry FS_PCT
                    sl_price = self.state.entry_price * (1 + FS_PCT)
                    if last_price >= sl_price:
                        await self._exit_market("FailSafe", last_price)
                        continue

            except Exception as e:
                logging.exception(f"[{self.symbol}] risk_loop error: {e}")

            await asyncio.sleep(RISK_LOOP_SEC)

    async def _exit_market(self, reason: str, last_price: float):
        # Close by sending reverse side order for the same notional
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
            # Update daily pnl approximation (notional based)
            self.state.daily_pnl += (pnl_pct / 100.0) * (usdt_amt / LEVERAGE)

            # SL streak logic
            if reason in ("TrailingStop", "FailSafe") and pnl_pct < 0:
                self.state.sl_streak += 1
                if self.state.sl_streak >= 3:
                    self.state.cooldown_until = datetime.utcnow() + timedelta(minutes=COOLDOWN_MIN)
                    notify(f"[{self.symbol}] SL streak {self.state.sl_streak}. Cooldown until {self.state.cooldown_until}.")

            if reason == "TakeProfit":
                self.state.sl_streak = 0  # reset streak on win

            notify(f"[{self.symbol}] EXIT ({reason}) {self.state.side} @ {exit_price:.2f} | "
                   f"PnL≈{pnl_pct:.2f}%  DailyPnL≈{self.state.daily_pnl:.2f} USDT")

            # flat
            self.state.side = None
            self.state.entry_price = None
            self.state.qty_usdt = 0.0
            self.state.qty = 0.0
            self.state.high_since_entry = None
            self.state.low_since_entry = None
# ------------------------- Backtest ---------------------------------------------
# ------------------------- Backtest ---------------------------------------------
# (in momentum_trader_bot.py)
# REPLACE your old function with this one

# ------------------------- Backtest ---------------------------------------------
def backtest_momentum_strategy(
    symbol=None,
    timeframe=None,
    start_date=None,
    end_date=None,
    entry_balance=None,
    fee_bps=4,               # 0.04% per side
    slippage_bps=1,          # 0.01% simulated slippage
    
    # --- NEW: Strategy parameters passed as arguments ---
    # We default to the global config values so it doesn't break
    # your old way of running it from if __name__ == "__main__"
    ema_fast=EMA_FAST,
    ema_slow=EMA_SLOW,
    use_vwap=USE_VWAP,
    volume_confirm=VOLUME_CONFIRM,
    tp_pct_cfg=TP_PCT,  # Renamed to avoid conflict
    sl_pct_cfg=FS_PCT   # Renamed to avoid conflict
):
    """
    Backtest for the live EMA+VWAP momentum strategy.
    
    (docstring remains the same)
    """

    # ---- Resolve config fallbacks ----
    symbol = symbol or SYMBOLS[0]
    timeframe = timeframe or TF
    entry_balance = float(entry_balance or 100.0)

    # Use the *passed-in* strategy's TP and SL values
    # (Converts 0.015 -> 1.5%)
    tp_pct = tp_pct_cfg * 100.0
    sl_pct = sl_pct_cfg * 100.0

    log_and_print(f"[Backtest] Running for {symbol} on {timeframe}...")
    # Use the passed-in arguments for logging
    log_and_print(f"[Backtest] Strategy: EMA{ema_fast}/{ema_slow}, VWAP={use_vwap}, VolConfirm={volume_confirm}")
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

    # ---- Calculate indicators (using passed-in args) ----
    df["ema_fast"] = ema(df["close"], ema_fast)
    df["ema_slow"] = ema(df["close"], ema_slow)
    if use_vwap:
        df["vwap"] = compute_vwap(df)
    if volume_confirm > 0:
        df['vol_avg_20'] = df['volume'].rolling(20).mean()

    # ---- State ----
    balance = entry_balance
    position = None
    entry_price = 0.0
    entry_time = None
    trades = []
    wins = 0
    peak_balance = balance
    max_dd = 0.0

    # ---- Iterate candles ----
    for i in range(max(30, ema_slow), len(df) - 1):  # Wait for slow EMA to warm up
        prev = df.iloc[i-1]
        cur = df.iloc[i]

        # --- 1. Manage Open Position (Check Exits) ---
        if position:
            exit_price = None
            exit_reason = None
            
            if position == 'LONG':
                tp_price = entry_price * (1 + tp_pct / 100.0)
                sl_price = entry_price * (1 - sl_pct / 100.0)
                
                if cur['high'] >= tp_price:
                    exit_price = tp_price
                    exit_reason = "TakeProfit"
                elif cur['low'] <= sl_price:
                    exit_price = sl_price
                    exit_reason = "StopLoss"
                    
            elif position == 'SHORT':
                tp_price = entry_price * (1 - tp_pct / 100.0)
                sl_price = entry_price * (1 + sl_pct / 100.0)

                if cur['low'] <= tp_price:
                    exit_price = tp_price
                    exit_reason = "TakeProfit"
                elif cur['high'] >= sl_price:
                    exit_price = sl_price
                    exit_reason = "StopLoss"

            if exit_price:
                change_pct = ((exit_price - entry_price) / entry_price * 100.0) if position == 'LONG' else ((entry_price - exit_price) / entry_price * 100.0)
                fees_pct = (fee_bps / 100.0) * 2
                slip_pct = (slippage_bps / 100.0)
                net_pct = change_pct - fees_pct - slip_pct

                balance *= (1.0 + net_pct / 100.0)
                wins += 1 if net_pct > 0 else 0

                trades.append({
                    "entry_time": entry_time,
                    "exit_time": cur['timestamp'],
                    "side": position,
                    "entry_price": float(entry_price),
                    "exit_price": float(exit_price),
                    "reason": exit_reason,
                    "net_pct": float(net_pct),
                    "balance": float(balance),
                })

                position = None
                entry_price = 0.0
                entry_time = None

                if balance > peak_balance:
                    peak_balance = balance
                dd = (peak_balance - balance) / peak_balance * 100.0
                max_dd = max(max_dd, dd)

                continue

        # --- 2. Look for New Entry (if flat) ---
        if position is None:
            # --- REPLICATE LIVE SIGNAL LOGIC (using passed-in args) ---
            cross_up   = is_bull_cross(prev, cur)
            cross_down = is_bear_cross(prev, cur)
            
            vwap_side  = (not use_vwap) or \
                         (use_vwap and 'vwap' in cur and not pd.isna(cur['vwap']) and (
                           (cur.close > cur.vwap and cross_up) or \
                           (cur.close < cur.vwap and cross_down)
                         ))
            
            if volume_confirm > 0:
                vol_ok = 'vol_avg_20' in cur and not pd.isna(cur['vol_avg_20']) and cur['volume'] >= volume_confirm * cur['vol_avg_20']
            else:
                vol_ok = True

            long_sig  = cross_up   and vwap_side and vol_ok
            short_sig = cross_down and vwap_side and vol_ok
            # --- END OF LIVE SIGNAL LOGIC ---

            if long_sig:
                position = 'LONG'
                next_candle = df.iloc[i+1]
                entry_price = next_candle['open']
                entry_time = next_candle['timestamp']

            elif short_sig:
                position = 'SHORT'
                next_candle = df.iloc[i+1]
                entry_price = next_candle['open']
                entry_time = next_candle['timestamp']

    # ---- Final Report ----
    total_trades = len(trades)
    win_rate = (wins / total_trades * 100.0) if total_trades else 0.0
    total_return_pct = ((balance - entry_balance) / entry_balance * 100.0)
    avg_trade_pct = (np.mean([t["net_pct"] for t in trades]) if trades else 0.0)

    # ---- Daily stats for scalping evaluation ----
    daily_stats = []
    if trades:
        day_start_balance = entry_balance
        day_end_balance = entry_balance
        cur_day = trades[0]["exit_time"].date()
        day_trades = 0
        day_wins = 0
        for t in trades:
            t_day = t["exit_time"].date()
            if t_day != cur_day:
                day_return_pct = ((day_end_balance - day_start_balance) / day_start_balance * 100.0)
                daily_stats.append({
                    "date": cur_day.isoformat(),
                    "trades": day_trades,
                    "wins": day_wins,
                    "win_rate_pct": (day_wins / day_trades * 100.0) if day_trades else 0.0,
                    "return_pct": float(day_return_pct),
                })
                cur_day = t_day
                day_start_balance = day_end_balance
                day_trades = 0
                day_wins = 0
            day_trades += 1
            day_wins += 1 if t["net_pct"] > 0 else 0
            day_end_balance = t["balance"]

        day_return_pct = ((day_end_balance - day_start_balance) / day_start_balance * 100.0)
        daily_stats.append({
            "date": cur_day.isoformat(),
            "trades": day_trades,
            "wins": day_wins,
            "win_rate_pct": (day_wins / day_trades * 100.0) if day_trades else 0.0,
            "return_pct": float(day_return_pct),
        })

    # Return the results *and* the parameters used
    return {
        "strategy_name": "momentum",
        "symbol": symbol,
        "timeframe": timeframe,
        "start": df['timestamp'].iloc[0],
        "end": df['timestamp'].iloc[-1],
        "starting_balance": float(entry_balance),
        "final_balance": float(balance),
        "total_return_pct": float(total_return_pct),
        "total_trades": total_trades,
        "win_rate_pct": float(win_rate),
        "avg_trade_net_pct": float(avg_trade_pct),
        "max_drawdown_pct": float(max_dd),
        "daily_stats": daily_stats,
        # --- Return parameters for analysis ---
        "tp_pct": float(tp_pct),
        "sl_pct": float(sl_pct),
        "ema_fast": ema_fast,
        "ema_slow": ema_slow,
        "use_vwap": use_vwap,
        "volume_confirm": volume_confirm,
        "fee_bps": float(fee_bps),
        "slippage_bps": float(slippage_bps)
    }

def run_random_daily_backtests(
    year: int,
    month: int,
    samples: int,
    symbol: str,
    timeframe: str,
    entry_balance: float,
):
    # Randomly pick days in the month and backtest each 24h slice.
    first_day = datetime(year, month, 1)
    if month == 12:
        next_month = datetime(year + 1, 1, 1)
    else:
        next_month = datetime(year, month + 1, 1)
    days_in_month = (next_month - first_day).days
    samples = max(1, min(samples, days_in_month))
    day_numbers = random.sample(range(1, days_in_month + 1), samples)
    day_numbers.sort()

    results_all = []
    for day in day_numbers:
        start_dt = datetime(year, month, day)
        end_dt = start_dt + timedelta(days=1)
        results = backtest_momentum_strategy(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_dt.strftime("%Y-%m-%d %H:%M:%S"),
            end_date=end_dt.strftime("%Y-%m-%d %H:%M:%S"),
            entry_balance=entry_balance,
        )
        if results:
            results_all.append(results)
            print(f"Symbol: {results['symbol']}")
            print(f"Timeframe: {results['timeframe']}")
            print(f"Period: {results['start']} to {results['end']}")
            print(f"Starting Balance: ${results['starting_balance']:.2f}")
            print(f"Final Balance: ${results['final_balance']:.2f}")
            print(f"Total Return: {results['total_return_pct']:.2f}%")
            print(f"Total Trades: {results['total_trades']}")
            print(f"Win Rate: {results['win_rate_pct']:.2f}%")
            print(f"Average Trade: {results['avg_trade_net_pct']:.2f}%")
            print(f"Max Drawdown: {results['max_drawdown_pct']:.2f}%")
            print(f"Take Profit: {results['tp_pct']:.2f}%")
            print(f"Stop Loss: {results['sl_pct']:.2f}%")
            print("-" * 40)
        else:
            print("No results to display")
            print("-" * 40)

    return results_all

# ------------------------- Runner --------------------------------------------
async def run():
    # Create a bot instance per symbol
    bots = [MomentumBot(sym) for sym in SYMBOLS]
    for b in bots:
        await b.setup()

    tasks = []
    for b in bots:
        tasks.append(asyncio.create_task(b.signal_loop(), name=f"{b.symbol}-signal"))
        tasks.append(asyncio.create_task(b.risk_loop(),   name=f"{b.symbol}-risk"))

    # graceful shutdown
    stop_event = asyncio.Event()

    def _stop(*_):
        notify("Shutdown signal received, cancelling tasks...")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            asyncio.get_running_loop().add_signal_handler(sig, _stop)
        except NotImplementedError:
            # Windows
            signal.signal(sig, _stop)

    await stop_event.wait()
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    notify("All tasks stopped. Bye.")

if __name__ == "__main__":
    try:
        RUN_LIVE = True
        RUN_RANDOM_DAILY_BACKTEST = False

        if RUN_LIVE:
            asyncio.run(run())
        elif RUN_RANDOM_DAILY_BACKTEST:
            run_random_daily_backtests(
                year=2025,
                month=11,
                samples=20,
                symbol="BTCUSDT",
                timeframe="15m",
                entry_balance=2000,
            )

    except KeyboardInterrupt:
        pass
   
