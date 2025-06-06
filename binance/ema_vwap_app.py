import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
import configparser
import pandas as pd
import numpy as np
from binance.client import Client
from ema_vwap import calculate_indicators
from telegram import send_telegram_message

# Enhanced logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)

# Load config
CFG_FILE = Path("config.ini")
if not CFG_FILE.exists():
    raise FileNotFoundError("config.ini not found")

config = configparser.ConfigParser()
config.read(CFG_FILE)

TRADING = config["TRADING"]
STRATEGY = config["STRATEGY"]

# Config variables with validation
SYMBOL = TRADING.get("SYMBOL", "BTCUSDT")
TFRAME = TRADING.get("trade_interval", "5m")
ENTRY_USDT = float(TRADING.get("entry_usdt", 100))
LEVERAGE = int(TRADING.get("leverage", 1))
TP_PCT = float(TRADING.get("tp_percentage", 0.7)) / 100  # Convert to decimal
SL_PCT = float(TRADING.get("sl_percentage", 0.35)) / 100  # Convert to decimal
SLEEP_MINUTES = int(TRADING.get("sleep_time", 1))
TP_SL_INTERVAL = int(TRADING.get("tp_sl_check_interval", 5))

EMA_SPAN = int(STRATEGY.get("ema_span", 9))
VWAP_BUFFER = float(STRATEGY.get("vwap_buffer", 0.0005))
RSI_GATE = int(STRATEGY.get("rsi_gate", 50))
STOCH_OVERBOUGHT = int(STRATEGY.get("stoch_k_overbought", 70))
STOCH_OVERSOLD = int(STRATEGY.get("stoch_k_oversold", 30))
MAX_POS_TIME = int(STRATEGY.get("max_pos_time", 30))
COOLDOWN = int(STRATEGY.get("cooldown_minutes", 10))

client = Client(config['API']['API_KEY'], config['API']['API_SECRET'], testnet=False)

# Global state with additional metrics
current_position = None
entry_price = 0
entry_time = None
balance = ENTRY_USDT
entry_quantity = 0
last_trade_time = None
trade_count = 0
win_count = 0

def  fetch_latest_data(symbol: str, interval: str, limit=20, retries=3):
    for attempt in range(retries):
        try:
            klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                'quote_volume', 'trades_count', 'taker_buy_volume', 'taker_buy_quote_volume', 'ignore'])
            
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            if attempt == retries - 1:
                logging.error(f"❌ Failed to fetch data after {retries} attempts: {e}")
                return pd.DataFrame()
            # await asyncio.sleep(5)

def calculate_position_size(atr, price):
    """Calculate position size based on volatility"""
    risk_per_trade = balance * 0.01  # Risk 1% of balance per trade
    atr_multiplier = 1.5  # Adjust based on your risk appetite
    sl_distance = atr * atr_multiplier
    size = (risk_per_trade / sl_distance) * price
    return min(size, balance * LEVERAGE / price)

def generate_signal(df):
    if df.empty or 'vwap' not in df.columns:
        return None
        
    row = df.iloc[-1]
    prev_row = df.iloc[-2]
    price = row['close']
    
    # Enhanced signal conditions
    roc = (price - df['close'].iloc[-5]) / df['close'].iloc[-5] * 100
    atr = row['atr']
    
    # Volume filter - require at least 1.5x average volume
    if STRATEGY.getboolean('volume_filter_enabled'):
        avg_volume = df['volume'].rolling(10).mean().iloc[-1]
        if row['volume'] < avg_volume * 1.5:
            return None
    
    # Price must be outside VWAP buffer zone
    vwap_distance = abs(price - row['vwap']) / row['vwap']
    if vwap_distance < VWAP_BUFFER * 2:  # Increased buffer zone
        return None
    
    # Trend confirmation - require EMA sloping in direction
    ema_slope = row['ema_trend'] - prev_row['ema_trend']
    
    # LONG conditions
    if (price > row['vwap'] * (1 + VWAP_BUFFER) and
        row['stoch_k'] < STOCH_OVERBOUGHT and
        row['rsi'] > RSI_GATE and
        price > row['ema_trend'] and
        ema_slope > 0 and
        roc > 0.1 and  # Minimum ROC filter
        vwap_distance > VWAP_BUFFER):
        return 'LONG'
    
    # SHORT conditions
    elif (price < row['vwap'] * (1 - VWAP_BUFFER) and
          row['stoch_k'] > STOCH_OVERSOLD and
          row['rsi'] < (100 - RSI_GATE) and
          price < row['ema_trend'] and
          ema_slope < 0 and
          roc < -0.1 and  # Minimum ROC filter
          vwap_distance > VWAP_BUFFER):
        return 'SHORT'
    
    return None

async def tp_sl_monitor():
    global current_position, entry_price, entry_time, balance, entry_quantity, win_count
    
    while True:
        await asyncio.sleep(TP_SL_INTERVAL)
        if current_position is None:
            continue

        # Check if position has been open too long
        if datetime.now() - entry_time > timedelta(minutes=MAX_POS_TIME):
            df = fetch_latest_data(SYMBOL, TFRAME, limit=1)
            if not df.empty:
                current_price = df['close'].iloc[-1]
                pnl_pct = ((current_price - entry_price) / entry_price) * 100 if current_position == 'LONG' \
                    else ((entry_price - current_price) / entry_price) * 100
                profit = balance * pnl_pct / 100
                
                logging.info(f"MAX TIME EXIT | {current_position} | Exit: {current_price} | PnL: {pnl_pct:.2f}%")
                send_telegram_message(
                    f"MAX TIME EXIT | {current_position} | Exit: {current_price} | PnL: {pnl_pct:.2f}% | Time: {datetime.now()}"
                )
                balance += profit
                if pnl_pct > 0:
                    win_count += 1
                current_position = None
            continue

        df = fetch_latest_data(SYMBOL, TFRAME, limit=1)
        if df.empty:
            continue

        current_price = df['close'].iloc[-1]
        pnl_pct = ((current_price - entry_price) / entry_price) * 100 if current_position == 'LONG' \
            else ((entry_price - current_price) / entry_price) * 100
        
        # Dynamic trailing stop based on ATR
        atr = df['atr'].iloc[-1]
        if STRATEGY.getboolean('trailing_tp_enabled') and abs(pnl_pct) > TP_PCT * 100:
            trailing_dist = atr * 0.5  # Trail by 0.5 ATR
            if (current_position == 'LONG' and 
                current_price < (entry_price * (1 + TP_PCT) - trailing_dist)):
                profit = balance * (TP_PCT + (pnl_pct/100 - TP_PCT))
                logging.info(f"TRAILING TP EXIT | {current_position} | Exit: {current_price} | Profit: {profit:.2f} USDT")
                send_telegram_message(
                    f"TRAILING TP EXIT | {current_position} | Exit: {current_price} | Profit: {profit:.2f} USDT"
                )
                balance += profit
                win_count += 1
                current_position = None
                
            elif (current_position == 'SHORT' and 
                  current_price > (entry_price * (1 - TP_PCT) + trailing_dist)):
                profit = balance * (TP_PCT + (abs(pnl_pct)/100 - TP_PCT))
                logging.info(f"TRAILING TP EXIT | {current_position} | Exit: {current_price} | Profit: {profit:.2f} USDT")
                send_telegram_message(
                    f"TRAILING TP EXIT | {current_position} | Exit: {current_price} | Profit: {profit:.2f} USDT"
                )
                balance += profit
                win_count += 1
                current_position = None

        # Check SL
        if pnl_pct <= -SL_PCT * 100:
            loss = balance * (pnl_pct / 100)
            logging.info(f"SL HIT | {current_position} | Exit: {current_price} | Loss: {loss:.2f} USDT")
            send_telegram_message(
                f"SL HIT | {current_position} | Exit: {current_price} | Loss: {loss:.2f} USDT"
            )
            balance += loss
            current_position = None
def apply_htf_filter(df, symbol, interval='1h'):
    """
    Apply higher timeframe trend filter
    Returns:
        float: Positive if uptrend, negative if downtrend, 0 if neutral
    """
    higher_df = fetch_latest_data(symbol, interval, limit=50)
    if higher_df.empty:
        return 0
    
    # Calculate EMA and its slope
    higher_df['ema_htf'] = higher_df['close'].ewm(span=EMA_SPAN).mean()
    
    # Get the last 3 EMA values to determine trend
    ema_values = higher_df['ema_htf'].iloc[-3:].values
    if len(ema_values) < 3:
        return 0
    
    # Calculate slope (current - previous)
    slope = ema_values[-1] - ema_values[-2]
    
    # Additional confirmation - price above/below EMA
    last_close = higher_df['close'].iloc[-1]
    if slope > 0 and last_close > higher_df['ema_htf'].iloc[-1]:
        return 1  # Strong uptrend
    elif slope < 0 and last_close < higher_df['ema_htf'].iloc[-1]:
        return -1  # Strong downtrend
    return 0  # Neutral trend
async def strategy_loop():
    global current_position, entry_price, entry_time, balance, entry_quantity, last_trade_time, trade_count
    
    while True:
        # Cooldown period check
        if last_trade_time and (datetime.now() - last_trade_time) < timedelta(minutes=COOLDOWN):
            await asyncio.sleep(SLEEP_MINUTES * 60)
            continue
            
        df = fetch_latest_data(SYMBOL, TFRAME, limit=50)  # Increased for better indicators
        if df.empty:
            await asyncio.sleep(SLEEP_MINUTES * 60)
            continue

        df = calculate_indicators(df, ema_span=EMA_SPAN)
        signal = generate_signal(df)

        # HTF filter
        if STRATEGY.getboolean('htf_filter'):
            trend = apply_htf_filter(df, SYMBOL)
            if signal == 'LONG' and trend < 0:
                signal = None
                logging.debug("HTF filter blocked LONG signal")
            elif signal == 'SHORT' and trend > 0:
                signal = None
                logging.debug("HTF filter blocked SHORT signal")

        if current_position is None and signal:
            # Wait for confirmation candle
            await asyncio.sleep(int(TFRAME[:-1]) * 60)  # Wait one full candle
            
            confirm_df = fetch_latest_data(SYMBOL, TFRAME, limit=1)
            if confirm_df.empty:
                continue
                
            confirm_price = confirm_df['close'].iloc[-1]
            
            # Check if signal still valid
            if (signal == 'LONG' and confirm_price > df['close'].iloc[-1]) or \
               (signal == 'SHORT' and confirm_price < df['close'].iloc[-1]):
                
                entry_price = confirm_price
                entry_time = datetime.now()
                current_position = signal
                atr = df['atr'].iloc[-1]
                entry_quantity = calculate_position_size(atr, entry_price)
                trade_count += 1
                
                logging.info(
                    f"OPEN {signal} | Entry: {entry_price} | Size: {entry_quantity:.5f} | "
                    f"Risk: {SL_PCT*100:.1f}% | Reward: {TP_PCT*100:.1f}% | "
                    f"Balance: {balance:.2f} USDT"
                )
                last_trade_time = datetime.now()

        await asyncio.sleep(SLEEP_MINUTES * 60)

async def main():
    logging.info("Starting trading bot with enhanced strategy")
    logging.info(f"Initial Balance: {balance:.2f} USDT")
    
    # Print strategy summary
    logging.info(
        f"Strategy Parameters:\n"
        f"Symbol: {SYMBOL} | TF: {TFRAME}\n"
        f"EMA: {EMA_SPAN} | VWAP Buffer: {VWAP_BUFFER*100:.2f}%\n"
        f"TP: {TP_PCT*100:.1f}% | SL: {SL_PCT*100:.1f}%\n"
        f"RSI Gate: {RSI_GATE} | Stoch OB/OS: {STOCH_OVERBOUGHT}/{STOCH_OVERSOLD}"
    )
    
    await asyncio.gather(
        strategy_loop(),
        tp_sl_monitor()
    )

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Bot stopped by user")
    except Exception as e:
        logging.error(f"🔥 Bot crashed with error: {e}")
        send_telegram_message(f"🚨 Bot crashed: {str(e)}")