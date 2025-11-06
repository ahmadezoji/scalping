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

    async def setup(self):
        # Margin/leverage one time
        set_margin_mode(self.symbol, "ISOLATED")
        set_leverage(self.symbol, LEVERAGE)
        notify(f"[{self.symbol}] Ready | TF={TF} EMA{EMA_FAST}/{EMA_SLOW} VWAP={USE_VWAP} "
               f"TRAIL={TRAIL_PCT*100:.2f}% TP={TP_PCT*100:.2f}% FS={FS_PCT*100:.2f}% "
               f"LEV={LEVERAGE} Risk/Trade={RISK_PER_TRADE_PCT*100:.2f}%")

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
                if df.empty or len(df) < 30:
                    await asyncio.sleep(SIGNAL_LOOP_SEC)
                    continue

                # Only act on NEWLY CLOSED candle
                last_candle_time = df["timestamp"].iloc[-1]
                if self.state.last_candle_time is not None and last_candle_time == self.state.last_candle_time:
                    await asyncio.sleep(SIGNAL_LOOP_SEC)
                    continue
                self.state.last_candle_time = last_candle_time

                data = compute_indicators(df)
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
                        self.state.side = "LONG" if long_sig else "SHORT"
                        self.state.entry_price = entry_price
                        self.state.qty_usdt = usdt_amt
                        self.state.high_since_entry = entry_price
                        self.state.low_since_entry = entry_price
                        notify(f"[{self.symbol}] ENTER {self.state.side} @ {entry_price:.2f} "
                               f"(notional ~{usdt_amt:.2f} USDT)")
            except Exception as e:
                logging.exception(f"[{self.symbol}] signal_loop error: {e}")
            await asyncio.sleep(SIGNAL_LOOP_SEC)

    async def risk_loop(self):
        """
        Fast loop: manage trailing stop, take-profit, and fail-safe.
        """
        while True:
            try:
                if self.state.side is None:
                    await asyncio.sleep(RISK_LOOP_SEC)
                    continue

                # last price from ticker (fast)
                ticker = client.futures_symbol_ticker(symbol=self.symbol)
                last_price = float(ticker["price"])

                # update swing refs
                if self.state.side == "LONG":
                    self.state.high_since_entry = max(self.state.high_since_entry or last_price, last_price)
                else:
                    self.state.low_since_entry = min(self.state.low_since_entry or last_price, last_price)

                # trailing + tp + fail-safe
                if self.state.side == "LONG":
                    # trailing: drop from high by TRAIL_PCT
                    if last_price <= (self.state.high_since_entry * (1 - TRAIL_PCT)):
                        await self._exit_market("TrailingStop", last_price)
                        continue
                    # take profit: gain from entry TP_PCT
                    if last_price >= (self.state.entry_price * (1 + TP_PCT)):
                        await self._exit_market("TakeProfit", last_price)
                        continue
                    # fail-safe: drop from entry FS_PCT
                    if last_price <= (self.state.entry_price * (1 - FS_PCT)):
                        await self._exit_market("FailSafe", last_price)
                        continue

                elif self.state.side == "SHORT":
                    # trailing: bounce up from low by TRAIL_PCT
                    if last_price >= (self.state.low_since_entry * (1 + TRAIL_PCT)):
                        await self._exit_market("TrailingStop", last_price)
                        continue
                    # take profit: drop from entry TP_PCT
                    if last_price <= (self.state.entry_price * (1 - TP_PCT)):
                        await self._exit_market("TakeProfit", last_price)
                        continue
                    # fail-safe: rise from entry FS_PCT
                    if last_price >= (self.state.entry_price * (1 + FS_PCT)):
                        await self._exit_market("FailSafe", last_price)
                        continue

            except Exception as e:
                logging.exception(f"[{self.symbol}] risk_loop error: {e}")

            await asyncio.sleep(RISK_LOOP_SEC)

    async def _exit_market(self, reason: str, last_price: float):
        # Close by sending reverse side order for the same notional
        out_side = "SELL" if self.state.side == "LONG" else "BUY"
        usdt_amt = self.state.qty_usdt
        order = place_order(self.symbol, out_side, usdt_amount=usdt_amt)
        if order:
            pnl_pct = ((last_price - self.state.entry_price) / self.state.entry_price) if self.state.side == "LONG" \
                      else ((self.state.entry_price - last_price) / self.state.entry_price)
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

            notify(f"[{self.symbol}] EXIT ({reason}) {self.state.side} @ {last_price:.2f} | "
                   f"PnL≈{pnl_pct:.2f}%  DailyPnL≈{self.state.daily_pnl:.2f} USDT")

            # flat
            self.state.side = None
            self.state.entry_price = None
            self.state.qty_usdt = 0.0
            self.state.high_since_entry = None
            self.state.low_since_entry = None
# ------------------------- Backtest ---------------------------------------------
def backtest_momentum_strategy(
    symbol=None,
    timeframe=None,
    start_date=None,
    end_date=None,
    tp=None,                 # take profit in %
    sl=None,                 # stop loss in %
    entry_balance=None,      # starting USDT
    fee_bps=4,               # 0.04% per side
    slippage_bps=1           # 0.01% simulated slippage
):
    """
    Simple backtest for your EMA+VWAP momentum scalping logic.

    Uses your existing:
      - get_klines_all(symbol, timeframe, start_time, end_time)
      - calculate_vwap(df)

    Signals (same family as your live logic):
      LONG  if trend==BULL and close>VWAP and RSI>50
      SHORT if trend==BEAR and close<VWAP and RSI<50

    TP/SL are ACTUAL % moves from entry. Fees & slippage are applied on exit.
    """

    # ---- Resolve config fallbacks ----
    try:
        # if you imported from index like:
        # from index import SYMBOL, trade_interval, entry_usdt, tp_percentage, sl_percentage
        default_symbol = symbol
        default_tf = timeframe
        default_balance = entry_balance
        default_tp = tp
        default_sl = sl
    except Exception:
        # safe fallbacks if not imported
        default_symbol = "BTCUSDT"
        default_tf = "1m"
        default_balance = 100.0
        default_tp = 0.8
        default_sl = 0.5

    symbol = symbol or default_symbol
    timeframe = timeframe or default_tf
    entry_balance = float(entry_balance or default_balance)
    tp = float(tp if tp is not None else default_tp)
    sl = float(sl if sl is not None else default_sl)

    # ---- Fetch & prep data ----
    df = get_klines_all(symbol, timeframe, start_time=start_date, end_time=end_date, limit=50)
    if df.empty:
        print("No data fetched for backtest.")
        return None

    df = calculate_vwap(df)
    df['ema_fast'] = df['close'].ewm(span=8, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=21, adjust=False).mean()
    df['trend'] = np.where(df['ema_fast'] > df['ema_slow'], 'BULL', 'BEAR')

    # ---- State ----
    balance = entry_balance
    position = None      # 'LONG' | 'SHORT' | None
    entry_price = None
    entry_time = None

    trades = []
    wins = 0
    peak_balance = balance
    max_dd = 0.0

    # ---- Iterate candles ----
    for i in range(30, len(df)):  # allow indicators to warm up
        row = df.iloc[i]
        close = row['close']

        # manage open position
        if position:
            change_pct = ((close - entry_price) / entry_price * 100.0) if position == 'LONG' else ((entry_price - close) / entry_price * 100.0)

            # TP/SL hit?
            if change_pct >= tp or change_pct <= -sl:
                # apply slippage + fees (round-trip)
                gross_pct = change_pct
                fees_pct = (fee_bps / 100.0) * 2
                slip_pct = slippage_bps / 100.0
                net_pct = gross_pct - fees_pct - (slip_pct * 100.0)

                balance *= (1.0 + net_pct / 100.0)
                wins += 1 if net_pct > 0 else 0

                trades.append({
                    "entry_time": entry_time,
                    "exit_time": row['timestamp'],
                    "side": position,
                    "entry_price": float(entry_price),
                    "exit_price": float(close),
                    "gross_pct": float(gross_pct),
                    "net_pct": float(net_pct),
                    "balance": float(balance),
                })

                # reset position
                position = None
                entry_price = None
                entry_time = None

                # track drawdown
                if balance > peak_balance:
                    peak_balance = balance
                dd = (peak_balance - balance) / peak_balance * 100.0
                max_dd = max(max_dd, dd)

                continue  # evaluate next candle fresh

        # look for new entry only if flat
        if position is None:
            bull = (row['trend'] == 'BULL')
            bear = (row['trend'] == 'BEAR')
            above_vwap = close > row['vwap']
            below_vwap = close < row['vwap']
            rsi = row['rsi']

            # you can add volume/ATR filters here if you want to tighten
            if bull and above_vwap and rsi > 50:
                position = 'LONG'
                entry_price = close
                entry_time = row['timestamp']

            elif bear and below_vwap and rsi < 50:
                position = 'SHORT'
                entry_price = close
                entry_time = row['timestamp']

    total_trades = len(trades)
    win_rate = (wins / total_trades * 100.0) if total_trades else 0.0
    total_return_pct = ((balance - entry_balance) / entry_balance * 100.0)
    avg_trade_pct = (np.mean([t["net_pct"] for t in trades]) if trades else 0.0)

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "start": df['timestamp'].iloc[0],
        "end": df['timestamp'].iloc[-1],
        "starting_balance": float(entry_balance),
        "final_balance": float(balance),
        "total_return_pct": float(total_return_pct),
        "trades": trades,
        "total_trades": total_trades,
        "win_rate_pct": float(win_rate),
        "avg_trade_net_pct": float(avg_trade_pct),
        "max_drawdown_pct": float(max_dd),
        "tp_pct": float(tp),
        "sl_pct": float(sl),
        "fee_bps": float(fee_bps),
        "slippage_bps": float(slippage_bps),
    }


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
        # asyncio.run(run())
        results = backtest_momentum_strategy(
                symbol="BTCUSDT",
                timeframe="15m",
                start_date = "2024-10-01 00:00:00",
                end_date = "2024-11-01 00:00:00",
                tp=0.75,  # %
                sl=0.5,   # %
                entry_balance=100
            )
        
        # results = backtest_strategy("BTCUSDT", "15m", "2024-10-01 00:00:00", "2024-11-01 00:00:00")

    except KeyboardInterrupt:
        pass
   
