import pandas as pd
import itertools
import time
import numpy as np

# Import the modified backtest function from your bot
from momentum_trader_bot import backtest_momentum_strategy
# Import the logger from your helper
from binance_helper import log_and_print

# -----------------------------------------------------------------
# --- 1. DEFINE YOUR PARAMETER GRID TO TEST ---
# -----------------------------------------------------------------
# WARNING: The number of combinations explodes quickly!
# Total Runs = (timeframes) * (ema_fasts) * (ema_slows) * (tps) * (sls) * ...
# Start with a small grid.

param_grid = {
    "timeframe": ["5m", "15m"],
    "ema_fast": [5, 8],
    "ema_slow": [21, 30],
    
    # Note: These are percentages (0.015 = 1.5%)
    # These values come from your config.ini (tp_percent = 1.6, failsafe_sl_percent = 0.8)
    # We are testing around those values.
    "tp_pct_cfg": [0.01, 0.016, 0.02], # 1.0%, 1.6%, 2.0%
    "sl_pct_cfg": [0.005, 0.008, 0.012], # 0.5%, 0.8%, 1.2%
    
    "use_vwap": [True, False],
    # "volume_confirm": [0.0, 1.5] # Can add this later
}

# --- Backtest constants ---
SYMBOL = "BTCUSDT"
START_DATE = "2024-10-01 00:00:00"
END_DATE = "2024-11-01 00:00:00"
ENTRY_BALANCE = 100.0


def run_optimizer():
    """
    Main function to run the grid search.
    """
    
    # --- 2. CREATE ALL COMBINATIONS ---
    keys = param_grid.keys()
    values = param_grid.values()
    
    # Creates a list of tuples, e.g., [('5m', 5, 21, ...), ('5m', 8, 21, ...)]
    combinations = list(itertools.product(*values))
    
    total_runs = len(combinations)
    log_and_print(f"--- Starting Strategy Optimization ---")
    log_and_print(f"Total combinations to test: {total_runs}")
    
    all_results = []
    start_time = time.time()

    # --- 3. RUN BACKTEST FOR EACH COMBINATION ---
    for i, combo in enumerate(combinations):
        
        # Create a dictionary of parameters for this run
        # e.g., {'timeframe': '5m', 'ema_fast': 5, 'ema_slow': 21, ...}
        params = dict(zip(keys, combo))
        
        # --- Sanity checks ---
        # Skip if fast EMA is slower than slow EMA
        if params["ema_fast"] >= params["ema_slow"]:
            log_and_print(f"[{i+1}/{total_runs}] SKIP: EMA {params['ema_fast']}>={params['ema_slow']}")
            continue
            
        # Skip if SL is greater than TP (bad risk/reward)
        if params["sl_pct_cfg"] >= params["tp_pct_cfg"]:
            log_and_print(f"[{i+1}/{total_runs}] SKIP: SL {params['sl_pct_cfg']} >= TP {params['tp_pct_cfg']}")
            continue

        log_and_print(f"\n--- [{i+1}/{total_runs}] TESTING ---")
        param_str = ", ".join(f"{k}={v}" for k, v in params.items())
        log_and_print(f"Params: {param_str}")
        
        try:
            # Call your backtest function, passing all strategy params
            results = backtest_momentum_strategy(
                symbol=SYMBOL,
                timeframe=params["timeframe"],
                start_date=START_DATE,
                end_date=END_DATE,
                entry_balance=ENTRY_BALANCE,
                
                # Pass strategy params from the grid
                ema_fast=params["ema_fast"],
                ema_slow=params["ema_slow"],
                tp_pct_cfg=params["tp_pct_cfg"],
                sl_pct_cfg=params["sl_pct_cfg"],
                use_vwap=params["use_vwap"],
                volume_confirm=0.0 # Keeping this fixed for now
            )

            if results:
                all_results.append(results)
                
        except Exception as e:
            log_and_print(f"!! ERROR on combo {params}: {e}")
            # Continue to the next combo
            pass

    # --- 4. ANALYZE AND DISPLAY RESULTS ---
    elapsed = time.time() - start_time
    log_and_print(f"\n--- Optimization Complete in {elapsed:.2f} seconds ---")
    
    if not all_results:
        log_and_print("No results found. The backtest may have failed for all combinations.")
        return

    # Convert list of dicts to a Pandas DataFrame
    df = pd.DataFrame(all_results)
    
    # Set display options for Pandas
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    # --- Filter and Sort ---
    # As you requested: positive profit and good win rate
    # We also add a minimum number of trades to avoid "lucky" 1-win strategies
    
    filtered_df = df[
        (df['total_return_pct'] > 5.0) &   # At least 5% return
        (df['win_rate_pct'] > 50.0) &      # Win rate > 50%
        (df['total_trades'] > 10) &        # At least 10 trades
        (df['max_drawdown_pct'] < 15.0)    # Max Drawdown < 15%
    ].copy()
    
    # Sort by the best return
    filtered_df.sort_values(by='total_return_pct', ascending=False, inplace=True)

    if filtered_df.empty:
        log_and_print("\n--- No Strategies Met Your Criteria ---")
        log_and_print("Showing the top 10 best by return (even if negative)...")
        
        # Define columns to show
        cols = [
            'total_return_pct', 'win_rate_pct', 'total_trades', 'max_drawdown_pct',
            'timeframe', 'ema_fast', 'ema_slow', 'tp_pct', 'sl_pct', 'use_vwap'
        ]
        
        # Check if 'cols' are available in the DataFrame
        available_cols = [col for col in cols if col in df.columns]
        
        print(df.sort_values(by='total_return_pct', ascending=False)[available_cols].head(10).to_string())
    else:
        log_and_print("\n--- 🚀 Top Performing Strategies 🚀 ---")
        
        # Define columns to show
        cols = [
            'total_return_pct', 'win_rate_pct', 'total_trades', 'max_drawdown_pct',
            'timeframe', 'ema_fast', 'ema_slow', 'tp_pct', 'sl_pct', 'use_vwap'
        ]
        
        # Check if 'cols' are available in the DataFrame
        available_cols = [col for col in cols if col in filtered_df.columns]
        
        print(filtered_df[available_cols].to_string())


if __name__ == "__main__":
    run_optimizer()