from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from binance.client import Client

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CFG_FILE = Path("config.ini")
if not CFG_FILE.exists():
    raise FileNotFoundError("config.ini not found")

import configparser
GLOBAL_CFG = configparser.ConfigParser()
GLOBAL_CFG.read(CFG_FILE)

TRADING  = GLOBAL_CFG["TRADING"]
STRAT    = GLOBAL_CFG["STRATEGY"]
OPT_CFG  = GLOBAL_CFG["OPTIMIZER"]

# Trading essentials
SYMBOL        = TRADING.get("SYMBOL", "BTCUSDT")
TFRAME        = TRADING.get("trade_interval", "1m")
ENTRY_USDT    = float(TRADING.get("entry_usdt", 100))
TP_PCT        = float(TRADING.get("tp_percentage", 0.8))
SL_PCT        = float(TRADING.get("sl_percentage", 0.5))

# Default strategy knobs (can be overridden by optimiser)
EMA_SPAN      = int(STRAT.get("ema_span", 21))
VWAP_BUFFER   = float(STRAT.get("vwap_buffer", 0.001))
RSI_GATE      = int(STRAT.get("rsi_gate", 55))



# Set up Binance client (public endpoints only – no API keys needed for klines)
client = Client("", "", testnet=False)

# ---------------------------------------------------------------------------
# Indicator helpers
# ---------------------------------------------------------------------------

def calculate_indicators(df: pd.DataFrame, ema_span: int, rsi_period: int = 14,
                         stochastic_period: int = 14) -> pd.DataFrame:
    """Add EMA, VWAP, RSI, ATR, Stoch K/D columns."""
    delta = df["close"].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = pd.Series(gain).rolling(rsi_period).mean()
    avg_loss = pd.Series(loss).rolling(rsi_period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df["rsi"] = 100 - (100 / (1 + rs))

    df["ema_trend"] = df["close"].ewm(span=ema_span, adjust=False).mean()
    df["vwap"]      = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()

    hl = df["high"] - df["low"]
    hc = abs(df["high"] - df["close"].shift())
    lc = abs(df["low"]  - df["close"].shift())
    df["atr"] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()

    df["stoch_k"] = (df["close"] - df["low"].rolling(stochastic_period).min()) / (
        df["high"].rolling(stochastic_period).max() - df["low"].rolling(stochastic_period).min() + 1e-6) * 100
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()
    return df

# ---------------------------------------------------------------------------
# Back‑test core
# ---------------------------------------------------------------------------
@dataclass
class BtMetrics:
    total_trades: int = 0
    wins: int = 0
    total_return_pct: float = 0.0
    max_dd: float = 0.0

    @property
    def win_rate(self):
        return (self.wins / self.total_trades * 100) if self.total_trades else 0.0


def backtest(ema_span: int, vwap_buffer: float, rsi_gate: int,
             tp_pct: float = TP_PCT, sl_pct: float = SL_PCT) -> BtMetrics:
    """Single‑pass back‑test returning performance metrics."""
    # Fetch klines
    start_ms = int(datetime.strptime(START, "%Y-%m-%d %H:%M:%S").timestamp()*1000)
    end_ms   = int(datetime.strptime(END,   "%Y-%m-%d %H:%M:%S").timestamp()*1000)
    klines = client.futures_historical_klines(SYMBOL, TFRAME, start_ms, end_ms)

    if not klines:
        raise RuntimeError("No data pulled – check dates & symbol")

    df = pd.DataFrame(klines, columns=[
        "ts", "open", "high", "low", "close", "volume", "ct", "qv",
        "trades", "tb_base", "tb_quote", "ignore"])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")

    df = calculate_indicators(df, ema_span)

    # Back‑test loop
    in_position = False
    entry_price = 0
    balance = ENTRY_USDT
    peak = balance
    metrics = BtMetrics()

    for i in range(20, len(df)):
        row = df.iloc[i]
        price = row["close"]
        signal = None
        if (price > row["vwap"] * (1 - vwap_buffer) and row["stoch_k"] < 70 and
                row["rsi"] > rsi_gate and price > row["ema_trend"]):
            signal = "LONG"
        elif (price < row["vwap"] * (1 + vwap_buffer) and row["stoch_k"] > 30 and
                row["rsi"] < (100 - rsi_gate) and price < row["ema_trend"]):
            signal = "SHORT"

        if in_position:
            pnl_pct = ((price - entry_price)/entry_price*100) if pos == "LONG" else ((entry_price - price)/entry_price*100)
            if pnl_pct >= tp_pct or pnl_pct <= -sl_pct:
                metrics.total_trades += 1
                if pnl_pct > 0:
                    metrics.wins += 1
                metrics.total_return_pct += pnl_pct
                balance *= (1 + pnl_pct/100)
                peak = max(peak, balance)
                dd = (peak - balance)/peak*100
                metrics.max_dd = max(metrics.max_dd, dd)
                in_position = False
        elif signal:
            in_position = True
            pos = signal
            entry_price = price

    return metrics

# ---------------------------------------------------------------------------
# Optimiser
# ---------------------------------------------------------------------------

def _parse_range(raw: str, cast):
    """Convert comma‑/colon‑separated ranges to list[int|float]."""
    raw = str(raw).strip()
    if ":" in raw:                 # eg 0.0005:0.002:0.0005
        start, stop, step = map(float if cast is float else int, raw.split(":"))
        return [cast(start + i*step) for i in range(int((stop-start)/step)+1)]
    return [cast(v) for v in raw.split(",")]


def optimise() -> tuple[dict, BtMetrics]:
    spans   = _parse_range(OPT_CFG.get("ema_span", EMA_SPAN), int)
    buffers = _parse_range(OPT_CFG.get("vwap_buffer", VWAP_BUFFER), float)
    rgates  = _parse_range(OPT_CFG.get("rsi_gate", RSI_GATE), int)
    tps     = _parse_range(OPT_CFG.get("tp_percentage", TP_PCT), float)
    sls     = _parse_range(OPT_CFG.get("sl_percentage", SL_PCT), float)

    objective = OPT_CFG.get("objective", "rr")      # rr | win | return

    best_cfg, best_met, best_score = None, None, -9e9
    logging.info("Starting grid search …")
    for ema, buf, rg, tp, sl in itertools.product(spans, buffers, rgates, tps, sls):
        met = backtest(ema, buf, rg, tp, sl)
        if objective == "win":
            score = met.win_rate
        elif objective == "return":
            score = met.total_return_pct
        else:  # risk‑reward: reward – drawdown penalty
            score = met.total_return_pct - met.max_dd
        if score > best_score:
            best_cfg = dict(ema_span=ema, vwap_buffer=buf, rsi_gate=rg, tp=tp, sl=sl)
            best_met = met
            best_score = score
    return best_cfg, best_met

# ---------------------------------------------------------------------------
# Main – back‑test or optimise
# ---------------------------------------------------------------------------

# Back‑test window (hard‑coded per user’s request)
START = "2025-04-01 00:00:00"
END   = "2025-04-26 23:59:00"

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    if OPT_CFG.getboolean("enabled", False):
        cfg, met = optimise()
        print("\nBest parameters:")
        for k, v in cfg.items():
            print(f"  {k}: {v}")
    else:
        met = backtest(EMA_SPAN, VWAP_BUFFER, RSI_GATE)

    print("\n=== Results ===")
    print(f"Trades     : {met.total_trades}")
    print(f"Win rate   : {met.win_rate:.2f} %")
    print(f"Return     : {met.total_return_pct:.2f} %")
    print(f"Max DD     : {met.max_dd:.2f} %")
    if OPT_CFG.getboolean("enabled", False):
        print(f"Score      : {met.total_return_pct - met.max_dd:.2f}")
