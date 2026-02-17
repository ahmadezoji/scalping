#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bollinger Band Mean-Reversion Trading Bot
- Entry on snap-back inside Bollinger Bands + RSI confirmation
- Optional VWAP/volume filters, TP/FailSafe, and trailing stop
- Winrate-focused optional filters: trend, ATR, session, and reentry strength
- Uses local binance_helper helpers
"""

import asyncio
import configparser
import logging
import os
import signal
import sys
from datetime import datetime, time as dt_time, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from binance_helper import (
    client,
    get_futures_account_balance,
    get_klines_all,
    log_and_print,
    place_order,
    set_leverage,
    set_margin_mode,
)


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

symbol_cfg = CFG.get(STRATEGY_SECTION, "SYMBOL", fallback=None)
if symbol_cfg is None:
    symbol_cfg = CFG.get("TRADING", "SYMBOL", fallback="BTCUSDT")
SYMBOLS = [s.strip() for s in symbol_cfg.split(",") if s.strip()]
if not SYMBOLS:
    SYMBOLS = ["BTCUSDT"]

ENTRY_USDT = CFG.getfloat("TRADING", "entry_usdt", fallback=0.0)
TF = CFG.get(STRATEGY_SECTION, "timeframe", fallback="1m")

# Baseline strategy params
USE_VWAP = CFG.get(STRATEGY_SECTION, "use_vwap_filter", fallback="false").lower() == "true"
VOLUME_CONFIRM = CFG.getfloat(STRATEGY_SECTION, "volume_confirm", fallback=0.0)
USE_TRAILING_STOP = CFG.get(STRATEGY_SECTION, "use_trailing_stop", fallback="false").lower() == "true"

TRAIL_PCT = CFG.getfloat(STRATEGY_SECTION, "trail_percent", fallback=0.25) / 100.0
TP_PCT = CFG.getfloat(STRATEGY_SECTION, "tp_percent", fallback=0.75) / 100.0
FS_PCT = CFG.getfloat(STRATEGY_SECTION, "failsafe_sl_percent", fallback=1.8) / 100.0

BB_WINDOW = CFG.getint(STRATEGY_SECTION, "bb_window", fallback=20)
BB_STD_DEV = CFG.getfloat(STRATEGY_SECTION, "bb_std_dev", fallback=2.0)
RSI_PERIOD = CFG.getint(STRATEGY_SECTION, "rsi_period", fallback=14)
RSI_OVERSOLD = CFG.getint(STRATEGY_SECTION, "rsi_oversold", fallback=30)
RSI_OVERBOUGHT = CFG.getint(STRATEGY_SECTION, "rsi_overbought", fallback=70)

LEVERAGE = CFG.getint(STRATEGY_SECTION, "leverage", fallback=2)
RISK_PER_TRADE_PCT = CFG.getfloat(STRATEGY_SECTION, "risk_per_trade_pct", fallback=0.25) / 100.0
DAILY_LOSS_CAP_PCT = CFG.getfloat(STRATEGY_SECTION, "daily_loss_cap_pct", fallback=2.0) / 100.0
COOLDOWN_MIN = CFG.getint(STRATEGY_SECTION, "cooldown_minutes", fallback=15)

SIGNAL_LOOP_SEC = CFG.getint(STRATEGY_SECTION, "signal_poll_seconds", fallback=5)
RISK_LOOP_SEC = CFG.getint(STRATEGY_SECTION, "risk_poll_seconds", fallback=1)

# Winrate-first upgrade params (all optional, non-breaking)
USE_TREND_FILTER = CFG.get(STRATEGY_SECTION, "use_trend_filter", fallback="false").lower() == "true"
TREND_EMA_PERIOD = CFG.getint(STRATEGY_SECTION, "trend_ema_period", fallback=200)
TREND_DISTANCE_MAX_PCT = CFG.getfloat(STRATEGY_SECTION, "trend_distance_max_pct", fallback=0.35) / 100.0

USE_ATR_FILTER = CFG.get(STRATEGY_SECTION, "use_atr_filter", fallback="false").lower() == "true"
ATR_PERIOD = CFG.getint(STRATEGY_SECTION, "atr_period", fallback=14)
ATR_MIN_PCT = CFG.getfloat(STRATEGY_SECTION, "atr_min_pct", fallback=0.08) / 100.0
ATR_MAX_PCT = CFG.getfloat(STRATEGY_SECTION, "atr_max_pct", fallback=0.45) / 100.0

USE_SESSION_FILTER = CFG.get(STRATEGY_SECTION, "use_session_filter", fallback="false").lower() == "true"
SESSION_UTC_START = CFG.get(STRATEGY_SECTION, "session_utc_start", fallback="12:00")
SESSION_UTC_END = CFG.get(STRATEGY_SECTION, "session_utc_end", fallback="20:00")

ENTRY_MODE = CFG.get(STRATEGY_SECTION, "entry_mode", fallback="next_open").strip().lower()
if ENTRY_MODE not in ("next_open", "close"):
    ENTRY_MODE = "next_open"

REQUIRE_BAND_REENTRY_STRENGTH = (
    CFG.get(STRATEGY_SECTION, "require_band_reentry_strength", fallback="false").lower() == "true"
)
REENTRY_MIN_BODY_PCT = CFG.getfloat(STRATEGY_SECTION, "reentry_min_body_pct", fallback=0.03) / 100.0

COOLDOWN_BARS_AFTER_LOSS = CFG.getint(STRATEGY_SECTION, "cooldown_bars_after_loss", fallback=3)
LIVE_ENTRY_SLIPPAGE_BPS = CFG.getfloat(STRATEGY_SECTION, "live_entry_slippage_bps", fallback=2.0)


def notify(msg: str):
    try:
        log_and_print(msg)
    except Exception:
        logging.info(msg)


# ------------------------- Utils ---------------------------------------------
def parse_hhmm(text: str) -> dt_time:
    try:
        hh, mm = text.split(":", 1)
        return dt_time(hour=int(hh), minute=int(mm))
    except Exception:
        return dt_time(hour=0, minute=0)


def is_in_utc_session(ts: pd.Timestamp, start_hhmm: str, end_hhmm: str) -> bool:
    start_t = parse_hhmm(start_hhmm)
    end_t = parse_hhmm(end_hhmm)
    t = ts.time()
    if start_t <= end_t:
        return start_t <= t <= end_t
    # overnight window
    return t >= start_t or t <= end_t


def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def rsi(s: pd.Series, period: int) -> pd.Series:
    delta = s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_vwap(df: pd.DataFrame) -> pd.Series:
    day = df["timestamp"].dt.date
    pv = df["close"] * df["volume"]
    return pv.groupby(day).cumsum() / df["volume"].groupby(day).cumsum()


def atr(df: pd.DataFrame, period: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr0 = (df["high"] - df["low"]).abs()
    tr1 = (df["high"] - prev_close).abs()
    tr2 = (df["low"] - prev_close).abs()
    tr = pd.concat([tr0, tr1, tr2], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean()


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["bb_mid"] = out["close"].rolling(window=BB_WINDOW).mean()
    out["bb_std"] = out["close"].rolling(window=BB_WINDOW).std()
    out["bb_high"] = out["bb_mid"] + (out["bb_std"] * BB_STD_DEV)
    out["bb_low"] = out["bb_mid"] - (out["bb_std"] * BB_STD_DEV)
    out["rsi"] = rsi(out["close"], RSI_PERIOD)
    if USE_VWAP:
        out["vwap"] = compute_vwap(out)
    if USE_TREND_FILTER:
        out["ema_trend"] = ema(out["close"], TREND_EMA_PERIOD)
    if USE_ATR_FILTER:
        out["atr"] = atr(out, ATR_PERIOD)
        out["atr_pct"] = out["atr"] / out["close"]
    return out


# ------------------------- State per symbol ----------------------------------
class PositionState:
    def __init__(self):
        self.side: Optional[str] = None
        self.entry_price: Optional[float] = None
        self.qty_usdt: float = 0.0
        self.qty: float = 0.0
        self.high_since_entry: Optional[float] = None
        self.low_since_entry: Optional[float] = None
        self.last_candle_time: Optional[pd.Timestamp] = None
        self.daily_start_date: Optional[datetime] = None
        self.daily_pnl: float = 0.0
        self.sl_streak: int = 0
        self.cooldown_until: Optional[datetime] = None
        self.cooldown_bars_remaining: int = 0

    def reset_intraday_if_needed(self):
        today = datetime.now(timezone.utc).date()
        if self.daily_start_date is None or self.daily_start_date.date() != today:
            self.daily_start_date = datetime.now(timezone.utc)
            self.daily_pnl = 0.0
            self.sl_streak = 0
            self.cooldown_until = None
            self.cooldown_bars_remaining = 0


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

    @staticmethod
    def _with_entry_slippage(reference_price: float, side: str, slippage_bps: float) -> float:
        slip = slippage_bps / 10000.0
        if side == "BUY":
            return reference_price * (1 + slip)
        return reference_price * (1 - slip)

    async def setup(self):
        set_margin_mode(self.symbol, "ISOLATED")
        set_leverage(self.symbol, LEVERAGE)
        self._sync_position_from_exchange()
        notify(
            f"[{self.symbol}] Ready | TF={TF} BB={BB_WINDOW}/{BB_STD_DEV} RSI={RSI_PERIOD} "
            f"({RSI_OVERSOLD}/{RSI_OVERBOUGHT}) TP={TP_PCT*100:.2f}% FS={FS_PCT*100:.2f}% "
            f"VWAP={USE_VWAP} VolConfirm={VOLUME_CONFIRM} Trend={USE_TREND_FILTER} "
            f"ATR={USE_ATR_FILTER} Session={USE_SESSION_FILTER} EntryMode={ENTRY_MODE}"
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

    @staticmethod
    def _position_size_usdt(balance_usdt: float) -> float:
        if FS_PCT <= 0:
            return 0.0
        pos_value_usdt = (RISK_PER_TRADE_PCT * balance_usdt * LEVERAGE) / FS_PCT
        return max(pos_value_usdt, 0.0)

    async def signal_loop(self):
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

                warmup = max(30, BB_WINDOW, RSI_PERIOD, TREND_EMA_PERIOD if USE_TREND_FILTER else 0, ATR_PERIOD if USE_ATR_FILTER else 0)
                df = get_klines_all(self.symbol, TF, limit=max(220, warmup + 5))
                if df.empty or len(df) < warmup + 2:
                    await asyncio.sleep(SIGNAL_LOOP_SEC)
                    logging.info(f"[{self.symbol}] Not enough data fetched (have {len(df)}, need {warmup + 2}).")
                    continue

                closed = compute_indicators(df.iloc[:-1])
                if len(closed) < 2:
                    await asyncio.sleep(SIGNAL_LOOP_SEC)
                    logging.info(f"[{self.symbol}] Not enough data after indicators.")
                    continue

                last_candle_time = closed["timestamp"].iloc[-1]
                if self.state.last_candle_time is not None and last_candle_time == self.state.last_candle_time:
                    await asyncio.sleep(SIGNAL_LOOP_SEC)
                    logging.info(f"[{self.symbol}] No new candle yet.")
                    continue
                self.state.last_candle_time = last_candle_time

                if self.state.cooldown_bars_remaining > 0:
                    self.state.cooldown_bars_remaining -= 1

                prev, cur = closed.iloc[-2], closed.iloc[-1]
                if pd.isna(cur["bb_low"]) or pd.isna(cur["rsi"]):
                    await asyncio.sleep(SIGNAL_LOOP_SEC)
                    continue

                base_long = (prev["close"] < prev["bb_low"]) and (cur["close"] >= cur["bb_low"]) and (cur["rsi"] < RSI_OVERSOLD)
                base_short = (prev["close"] > prev["bb_high"]) and (cur["close"] <= cur["bb_high"]) and (cur["rsi"] > RSI_OVERBOUGHT)

                if USE_VWAP and "vwap" in cur and not pd.isna(cur["vwap"]):
                    vwap_ok_long = cur["close"] <= cur["vwap"]
                    vwap_ok_short = cur["close"] >= cur["vwap"]
                else:
                    vwap_ok_long = True
                    vwap_ok_short = True

                if VOLUME_CONFIRM > 0:
                    vol_avg = closed["volume"].tail(20).mean()
                    vol_ok = cur["volume"] >= VOLUME_CONFIRM * vol_avg
                else:
                    vol_avg = None
                    vol_ok = True

                if USE_TREND_FILTER and "ema_trend" in cur and not pd.isna(cur["ema_trend"]):
                    trend_ok_long = cur["close"] >= cur["ema_trend"]
                    trend_ok_short = cur["close"] <= cur["ema_trend"]
                    trend_dist_ok = abs(cur["close"] - cur["ema_trend"]) / cur["close"] <= TREND_DISTANCE_MAX_PCT
                else:
                    trend_ok_long = True
                    trend_ok_short = True
                    trend_dist_ok = True

                if USE_ATR_FILTER and "atr_pct" in cur and not pd.isna(cur["atr_pct"]):
                    atr_ok = ATR_MIN_PCT <= cur["atr_pct"] <= ATR_MAX_PCT
                else:
                    atr_ok = True

                if USE_SESSION_FILTER:
                    session_ok = is_in_utc_session(cur["timestamp"], SESSION_UTC_START, SESSION_UTC_END)
                else:
                    session_ok = True

                candle_body_pct = abs(cur["close"] - cur["open"]) / cur["open"] if cur["open"] else 0.0
                if REQUIRE_BAND_REENTRY_STRENGTH:
                    reentry_ok_long = (cur["close"] > cur["open"]) and (candle_body_pct >= REENTRY_MIN_BODY_PCT)
                    reentry_ok_short = (cur["close"] < cur["open"]) and (candle_body_pct >= REENTRY_MIN_BODY_PCT)
                else:
                    reentry_ok_long = True
                    reentry_ok_short = True

                cooldown_bars_ok = self.state.cooldown_bars_remaining == 0

                long_sig = all([
                    base_long,
                    vwap_ok_long,
                    vol_ok,
                    trend_ok_long,
                    trend_dist_ok,
                    atr_ok,
                    session_ok,
                    reentry_ok_long,
                    cooldown_bars_ok,
                ])
                short_sig = all([
                    base_short,
                    vwap_ok_short,
                    vol_ok,
                    trend_ok_short,
                    trend_dist_ok,
                    atr_ok,
                    session_ok,
                    reentry_ok_short,
                    cooldown_bars_ok,
                ])

                logging.info(
                    f"[{self.symbol}] t={cur.name} close={cur.close:.2f} bb_low={cur.bb_low:.2f} bb_high={cur.bb_high:.2f} "
                    f"rsi={cur.rsi:.2f} vwap={getattr(cur, 'vwap', float('nan')):.2f} vol={cur['volume']:.2f} "
                    f"vwap_ok_L/S={vwap_ok_long}/{vwap_ok_short} vol_ok={vol_ok} trend_ok_L/S={trend_ok_long}/{trend_ok_short} "
                    f"trend_dist_ok={trend_dist_ok} atr_ok={atr_ok} session_ok={session_ok} reentry_ok_L/S={reentry_ok_long}/{reentry_ok_short} "
                    f"cooldown_bars={self.state.cooldown_bars_remaining} base_L/S={base_long}/{base_short} sig_L/S={long_sig}/{short_sig} "
                    f"(mult={VOLUME_CONFIRM}, avg20={vol_avg if vol_avg else 0}, body%={candle_body_pct*100:.3f})"
                )

                if self.state.side is None and (long_sig or short_sig):
                    self._sync_position_from_exchange()
                    if self.state.side is not None:
                        await asyncio.sleep(SIGNAL_LOOP_SEC)
                        logging.info(f"[{self.symbol}] Position opened externally, skipping entry.")
                        continue

                    side = "BUY" if long_sig else "SELL"
                    open_candle = df.iloc[-1]
                    if ENTRY_MODE == "next_open" and not pd.isna(open_candle["open"]):
                        entry_ref_price = float(open_candle["open"])
                    else:
                        entry_ref_price = float(cur["close"])

                    if ENTRY_USDT > 0:
                        usdt_amt = ENTRY_USDT
                    else:
                        balance = get_futures_account_balance("USDT")
                        usdt_amt = self._position_size_usdt(balance)

                    if usdt_amt <= 5:
                        notify(f"[{self.symbol}] Skip entry: size too small (usdt={usdt_amt:.2f})")
                        await asyncio.sleep(SIGNAL_LOOP_SEC)
                        continue

                    order = place_order(self.symbol, side, usdt_amount=usdt_amt, strategy_name="BollingerRSI")
                    if order:
                        fallback_entry = self._with_entry_slippage(entry_ref_price, side, LIVE_ENTRY_SLIPPAGE_BPS)
                        entry_price = self._extract_avg_price(order, fallback_entry)
                        qty = float(order.get("executedQty") or order.get("origQty") or 0)
                        if qty <= 0:
                            qty = usdt_amt / entry_price

                        self.state.side = "LONG" if long_sig else "SHORT"
                        self.state.entry_price = entry_price
                        self.state.qty_usdt = usdt_amt
                        self.state.qty = qty
                        self.state.high_since_entry = entry_price
                        self.state.low_since_entry = entry_price

                        notify(
                            f"[{self.symbol}] ENTER {self.state.side} @ {entry_price:.2f} "
                            f"(mode={ENTRY_MODE}, ref={entry_ref_price:.2f}, notional~{usdt_amt:.2f} USDT)"
                        )
            except Exception as e:
                logging.exception(f"[{self.symbol}] signal_loop error: {e}")

            await asyncio.sleep(SIGNAL_LOOP_SEC)

    async def risk_loop(self):
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

        order = place_order(self.symbol, out_side, usdt_amount=usdt_amt, reduce_only=True, strategy_name="BollingerRSI")
        if not order:
            return

        exit_price = self._extract_avg_price(order, last_price)
        pnl_pct = (
            ((exit_price - self.state.entry_price) / self.state.entry_price)
            if self.state.side == "LONG"
            else ((self.state.entry_price - exit_price) / self.state.entry_price)
        )
        pnl_pct *= 100.0
        self.state.daily_pnl += (pnl_pct / 100.0) * (usdt_amt / LEVERAGE)

        if reason in ("FailSafe", "TrailingStop") and pnl_pct < 0:
            self.state.sl_streak += 1
            if COOLDOWN_BARS_AFTER_LOSS > 0:
                self.state.cooldown_bars_remaining = COOLDOWN_BARS_AFTER_LOSS
            if self.state.sl_streak >= 3:
                self.state.cooldown_until = datetime.now(timezone.utc) + timedelta(minutes=COOLDOWN_MIN)
                notify(f"[{self.symbol}] SL streak {self.state.sl_streak}. Cooldown until {self.state.cooldown_until}.")
        elif reason == "TakeProfit":
            self.state.sl_streak = 0

        notify(
            f"[{self.symbol}] EXIT ({reason}) {self.state.side} @ {exit_price:.2f} | "
            f"PnL≈{pnl_pct:.2f}% DailyPnL≈{self.state.daily_pnl:.2f} USDT"
        )

        self.state.side = None
        self.state.entry_price = None
        self.state.qty_usdt = 0.0
        self.state.qty = 0.0
        self.state.high_since_entry = None
        self.state.low_since_entry = None


# ------------------------- Backtest ------------------------------------------
def backtest_bollinger_strategy(
    symbol=None,
    timeframe=None,
    start_date=None,
    end_date=None,
    entry_balance=None,
    fee_bps=4,
    slippage_bps=1,
    bb_window=BB_WINDOW,
    bb_std_dev=BB_STD_DEV,
    rsi_period=RSI_PERIOD,
    rsi_oversold=RSI_OVERSOLD,
    rsi_overbought=RSI_OVERBOUGHT,
    tp_pct_cfg=TP_PCT,
    sl_pct_cfg=FS_PCT,
    use_vwap_filter=USE_VWAP,
    volume_confirm=VOLUME_CONFIRM,
    use_trend_filter=USE_TREND_FILTER,
    trend_ema_period=TREND_EMA_PERIOD,
    trend_distance_max_pct=TREND_DISTANCE_MAX_PCT,
    use_atr_filter=USE_ATR_FILTER,
    atr_period=ATR_PERIOD,
    atr_min_pct=ATR_MIN_PCT,
    atr_max_pct=ATR_MAX_PCT,
    use_session_filter=USE_SESSION_FILTER,
    session_utc_start=SESSION_UTC_START,
    session_utc_end=SESSION_UTC_END,
    entry_mode=ENTRY_MODE,
    require_band_reentry_strength=REQUIRE_BAND_REENTRY_STRENGTH,
    reentry_min_body_pct=REENTRY_MIN_BODY_PCT,
    cooldown_bars_after_loss=COOLDOWN_BARS_AFTER_LOSS,
    use_public_mainnet_klines=False,
):
    symbol = symbol or SYMBOLS[0]
    timeframe = timeframe or TF
    entry_balance = float(entry_balance or 100.0)
    entry_mode = (entry_mode or "next_open").lower()
    if entry_mode not in ("next_open", "close"):
        entry_mode = "next_open"

    if bb_window < 5 or rsi_period < 5:
        return None
    if bb_std_dev <= 0 or tp_pct_cfg <= 0 or sl_pct_cfg <= 0:
        return None
    if rsi_oversold >= rsi_overbought:
        return None

    try:
        hist_client = client
        if use_public_mainnet_klines:
            from binance.client import Client as BinanceClient
            hist_client = BinanceClient()
        klines_list = hist_client.get_historical_klines(symbol, timeframe, start_str=start_date, end_str=end_date)
        df = pd.DataFrame(
            klines_list,
            columns=[
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_asset_volume",
                "number_of_trades",
                "taker_buy_base_asset_volume",
                "taker_buy_quote_asset_volume",
                "ignore",
            ],
        )
    except Exception as e:
        logging.error(f"Backtest data fetch failed: {e}")
        return None

    if df.empty:
        return None

    df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col])

    df["bb_mid"] = df["close"].rolling(window=bb_window).mean()
    df["bb_std"] = df["close"].rolling(window=bb_window).std()
    df["bb_high"] = df["bb_mid"] + (df["bb_std"] * bb_std_dev)
    df["bb_low"] = df["bb_mid"] - (df["bb_std"] * bb_std_dev)
    df["rsi"] = rsi(df["close"], rsi_period)
    if use_vwap_filter:
        df["vwap"] = compute_vwap(df)
    if use_trend_filter:
        df["ema_trend"] = ema(df["close"], trend_ema_period)
    if use_atr_filter:
        df["atr"] = atr(df, atr_period)
        df["atr_pct"] = df["atr"] / df["close"]
    if volume_confirm > 0:
        df["vol_avg_20"] = df["volume"].rolling(20).mean()

    balance = entry_balance
    position = None
    entry_price = 0.0
    entry_time = None
    trades = []
    wins = 0
    peak_balance = balance
    max_dd = 0.0
    cooldown_bars_remaining = 0

    warmup = max(30, bb_window, rsi_period, trend_ema_period if use_trend_filter else 0, atr_period if use_atr_filter else 0)

    # keep len(df)-1 so next_open is always available
    for i in range(warmup, len(df) - 1):
        prev = df.iloc[i - 1]
        cur = df.iloc[i]
        nxt = df.iloc[i + 1]

        if pd.isna(cur["bb_low"]) or pd.isna(cur["rsi"]):
            continue

        # 1) manage open position on current candle
        if position:
            exit_price = None
            exit_reason = None

            if position == "LONG":
                tp_price = entry_price * (1 + tp_pct_cfg)
                sl_price = entry_price * (1 - sl_pct_cfg)
                if cur["high"] >= tp_price:
                    exit_price, exit_reason = tp_price, "TakeProfit"
                elif cur["low"] <= sl_price:
                    exit_price, exit_reason = sl_price, "StopLoss"

            elif position == "SHORT":
                tp_price = entry_price * (1 - tp_pct_cfg)
                sl_price = entry_price * (1 + sl_pct_cfg)
                if cur["low"] <= tp_price:
                    exit_price, exit_reason = tp_price, "TakeProfit"
                elif cur["high"] >= sl_price:
                    exit_price, exit_reason = sl_price, "StopLoss"

            if exit_price is not None:
                change_pct = (
                    ((exit_price - entry_price) / entry_price * 100.0)
                    if position == "LONG"
                    else ((entry_price - exit_price) / entry_price * 100.0)
                )
                fees_pct = (fee_bps / 100.0) * 2
                slip_pct = slippage_bps / 100.0
                net_pct = change_pct - fees_pct - slip_pct

                balance *= (1.0 + net_pct / 100.0)
                wins += 1 if net_pct > 0 else 0
                trades.append(
                    {
                        "entry_time": entry_time,
                        "exit_time": cur["timestamp"],
                        "side": position,
                        "entry_price": float(entry_price),
                        "exit_price": float(exit_price),
                        "reason": exit_reason,
                        "net_pct": float(net_pct),
                        "balance": float(balance),
                    }
                )

                if net_pct < 0 and cooldown_bars_after_loss > 0:
                    cooldown_bars_remaining = cooldown_bars_after_loss

                position = None
                entry_price = 0.0
                entry_time = None

                if balance > peak_balance:
                    peak_balance = balance
                dd = (peak_balance - balance) / peak_balance * 100.0
                max_dd = max(max_dd, dd)
                continue

        # 2) flat: evaluate entry
        if position is None:
            if cooldown_bars_remaining > 0:
                cooldown_bars_remaining -= 1
                continue

            base_long = (prev["close"] < prev["bb_low"]) and (cur["close"] >= cur["bb_low"]) and (cur["rsi"] < rsi_oversold)
            base_short = (prev["close"] > prev["bb_high"]) and (cur["close"] <= cur["bb_high"]) and (cur["rsi"] > rsi_overbought)

            if use_vwap_filter and "vwap" in cur and not pd.isna(cur["vwap"]):
                vwap_ok_long = cur["close"] <= cur["vwap"]
                vwap_ok_short = cur["close"] >= cur["vwap"]
            else:
                vwap_ok_long = True
                vwap_ok_short = True

            if volume_confirm > 0 and "vol_avg_20" in cur and not pd.isna(cur["vol_avg_20"]):
                vol_ok = cur["volume"] >= volume_confirm * cur["vol_avg_20"]
            elif volume_confirm > 0:
                vol_ok = False
            else:
                vol_ok = True

            if use_trend_filter and "ema_trend" in cur and not pd.isna(cur["ema_trend"]):
                trend_ok_long = cur["close"] >= cur["ema_trend"]
                trend_ok_short = cur["close"] <= cur["ema_trend"]
                trend_dist_ok = abs(cur["close"] - cur["ema_trend"]) / cur["close"] <= trend_distance_max_pct
            else:
                trend_ok_long = True
                trend_ok_short = True
                trend_dist_ok = True

            if use_atr_filter and "atr_pct" in cur and not pd.isna(cur["atr_pct"]):
                atr_ok = atr_min_pct <= cur["atr_pct"] <= atr_max_pct
            else:
                atr_ok = not use_atr_filter

            if use_session_filter:
                session_ok = is_in_utc_session(cur["timestamp"], session_utc_start, session_utc_end)
            else:
                session_ok = True

            body_pct = abs(cur["close"] - cur["open"]) / cur["open"] if cur["open"] else 0.0
            if require_band_reentry_strength:
                reentry_ok_long = (cur["close"] > cur["open"]) and (body_pct >= reentry_min_body_pct)
                reentry_ok_short = (cur["close"] < cur["open"]) and (body_pct >= reentry_min_body_pct)
            else:
                reentry_ok_long = True
                reentry_ok_short = True

            long_sig = all([base_long, vwap_ok_long, vol_ok, trend_ok_long, trend_dist_ok, atr_ok, session_ok, reentry_ok_long])
            short_sig = all([base_short, vwap_ok_short, vol_ok, trend_ok_short, trend_dist_ok, atr_ok, session_ok, reentry_ok_short])

            if long_sig:
                position = "LONG"
                entry_price = float(nxt["open"] if entry_mode == "next_open" else cur["close"])
                entry_time = nxt["timestamp"] if entry_mode == "next_open" else cur["timestamp"]
            elif short_sig:
                position = "SHORT"
                entry_price = float(nxt["open"] if entry_mode == "next_open" else cur["close"])
                entry_time = nxt["timestamp"] if entry_mode == "next_open" else cur["timestamp"]

    total_trades = len(trades)
    win_rate = (wins / total_trades * 100.0) if total_trades else 0.0
    total_return_pct = ((balance - entry_balance) / entry_balance * 100.0)
    avg_trade_pct = np.mean([t["net_pct"] for t in trades]) if trades else 0.0

    gross_profit = sum(t["net_pct"] for t in trades if t["net_pct"] > 0)
    gross_loss = abs(sum(t["net_pct"] for t in trades if t["net_pct"] < 0))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)

    return {
        "strategy_name": "BollingerRSI",
        "symbol": symbol,
        "timeframe": timeframe,
        "start": df["timestamp"].iloc[0],
        "end": df["timestamp"].iloc[-1],
        "starting_balance": float(entry_balance),
        "final_balance": float(balance),
        "total_return_pct": float(total_return_pct),
        "total_trades": total_trades,
        "win_rate_pct": float(win_rate),
        "max_drawdown_pct": float(max_dd),
        "avg_trade_pct": float(avg_trade_pct),
        "profit_factor": float(profit_factor) if np.isfinite(profit_factor) else float(999.0),
        "entry_mode": entry_mode,
        "tp_pct": float(tp_pct_cfg * 100.0),
        "sl_pct": float(sl_pct_cfg * 100.0),
        "bb_window": bb_window,
        "bb_std_dev": bb_std_dev,
        "rsi_period": rsi_period,
        "rsi_oversold": rsi_oversold,
        "rsi_overbought": rsi_overbought,
        "use_vwap_filter": bool(use_vwap_filter),
        "volume_confirm": float(volume_confirm),
        "use_trend_filter": bool(use_trend_filter),
        "trend_ema_period": int(trend_ema_period),
        "trend_distance_max_pct": float(trend_distance_max_pct * 100.0),
        "use_atr_filter": bool(use_atr_filter),
        "atr_period": int(atr_period),
        "atr_min_pct": float(atr_min_pct * 100.0),
        "atr_max_pct": float(atr_max_pct * 100.0),
        "use_session_filter": bool(use_session_filter),
        "session_utc_start": session_utc_start,
        "session_utc_end": session_utc_end,
        "require_band_reentry_strength": bool(require_band_reentry_strength),
        "reentry_min_body_pct": float(reentry_min_body_pct * 100.0),
        "cooldown_bars_after_loss": int(cooldown_bars_after_loss),
        "trades": trades,
    }


# ------------------------- Runner --------------------------------------------
async def run():
    bots = [BollingerBot(sym) for sym in SYMBOLS]
    for b in bots:
        await b.setup()

    tasks = []
    for b in bots:
        tasks.append(asyncio.create_task(b.signal_loop(), name=f"{b.symbol}-signal"))
        tasks.append(asyncio.create_task(b.risk_loop(), name=f"{b.symbol}-risk"))

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
