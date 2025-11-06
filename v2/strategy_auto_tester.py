import pandas as pd
import itertools
import time
import numpy as np

# --- NEW: Import BOTH backtest functions ---
from momentum_trader_bot import backtest_momentum_strategy
from bolling_band_bot import backtest_bollinger_strategy


from binance_helper import log_and_print

# -----------------------------------------------------------------
# --- 1. DEFINE YOUR PARAMETER GRIDS ---
# -----------------------------------------------------------------

# --- Grid 1: Your original Momentum/EMA strategy ---
momentum_grid = {
    "timeframe": ["5m", "15m"],
    "ema_fast": [5, 8],
    "ema_slow": [21, 30],
    "tp_pct_cfg": [0.016, 0.02],  # 1.6%, 2.0%
    "sl_pct_cfg": [0.008, 0.012], # 0.8%, 1.2%
    "use_vwap": [True], # We know this is good
}

# --- Grid 2: The NEW Bollinger Band strategy ---
bollinger_grid = {
    "timeframe": ["5m", "15m"],
    "bb_window": [20, 30],
    "bb_std_dev": [2.0, 2.5],
    "tp_pct_cfg": [0.005, 0.01],  # Tighter TP: 0.5%, 1.0%
    "sl_pct_cfg": [0.003, 0.005], # Tighter SL: 0.3%, 0.5%
}

# --- Backtest constants ---
SYMBOL = "BTCUSDT"
START_DATE = "2025-01-08 00:00:00"
END_DATE = "2025-04-08 00:00:00"
ENTRY_BALANCE = 100.0


def get_combinations(grid_params: dict) -> list[dict]:
    """Helper function to expand a grid into a list of param dicts."""
    keys = grid_params.keys()
    values = grid_params.values()
    
    combinations = list(itertools.product(*values))
    
    param_list = []
    for combo in combinations:
        param_list.append(dict(zip(keys, combo)))
    return param_list


def run_optimizer():
    """
    Main function to run the multi-strategy grid search.
    """
    
    # --- 2. CREATE ALL COMBINATIONS ---
    
    # Get combos for each strategy
    momentum_combos = get_combinations(momentum_grid)
    bollinger_combos = get_combinations(bollinger_grid)
    
    # Add a 'strategy_name' tag to each
    # This is how we'll know which function to call
    for p in momentum_combos: p['strategy_name'] = 'momentum'
    for p in bollinger_combos: p['strategy_name'] = 'bollinger'
    
    # Combine them into one giant list of jobs
    all_combinations = bollinger_combos + momentum_combos
    # all_combinations = bollinger_combos
    
    total_runs = len(all_combinations)
    log_and_print(f"--- Starting Multi-Strategy Optimization ---")
    log_and_print(f"Total combinations to test: {total_runs}")
    
    all_results = []
    start_time = time.time()

    # --- 3. RUN BACKTEST FOR EACH COMBINATION ---
    for i, params in enumerate(all_combinations):
        
        log_and_print(f"\n--- [{i+1}/{total_runs}] TESTING {params['strategy_name']} ---")
        param_str = ", ".join(f"{k}={v}" for k, v in params.items())
        log_and_print(f"Params: {param_str}")
        
        results = None
        try:
            # --- NEW: CONDITIONAL LOGIC ---
            # Call the correct backtest function based on the 'strategy_name'
            
            if params['strategy_name'] == 'momentum':
                
                # Sanity checks for momentum
                if params["ema_fast"] >= params["ema_slow"]:
                    log_and_print("SKIP: EMA fast >= slow")
                    continue
                if params["sl_pct_cfg"] >= params["tp_pct_cfg"]:
                    log_and_print("SKIP: SL >= TP")
                    continue

                results = backtest_momentum_strategy(
                    symbol=SYMBOL,
                    start_date=START_DATE,
                    end_date=END_DATE,
                    entry_balance=ENTRY_BALANCE,
                    
                    # Pass strategy params from the grid
                    timeframe=params["timeframe"],
                    ema_fast=params["ema_fast"],
                    ema_slow=params["ema_slow"],
                    tp_pct_cfg=params["tp_pct_cfg"],
                    sl_pct_cfg=params["sl_pct_cfg"],
                    use_vwap=params["use_vwap"],
                )

            elif params['strategy_name'] == 'bollinger':

                # Sanity checks for bollinger
                if params["sl_pct_cfg"] >= params["tp_pct_cfg"]:
                    log_and_print("SKIP: SL >= TP")
                    continue

                results = backtest_bollinger_strategy(
                    symbol=SYMBOL,
                    start_date=START_DATE,
                    end_date=END_DATE,
                    entry_balance=ENTRY_BALANCE,
                    
                    # Pass strategy params from the grid
                    timeframe=params["timeframe"],
                    bb_window=params["bb_window"],
                    bb_std_dev=params["bb_std_dev"],
                    tp_pct_cfg=params["tp_pct_cfg"],
                    sl_pct_cfg=params["sl_pct_cfg"],
                )

            if results:
                all_results.append(results)
                
        except Exception as e:
            log_and_print(f"!! ERROR on combo {params}: {e}")
            pass

    # --- 4. ANALYZE AND DISPLAY RESULTS ---
    elapsed = time.time() - start_time
    log_and_print(f"\n--- Optimization Complete in {elapsed:.2f} seconds ---")
    
    if not all_results:
        log_and_print("No results found.")
        return

    df = pd.DataFrame(all_results)
    
    pd.set_option('display.max_rows', 200)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    # --- Filter and Sort ---
    # We use the same filters as before
    filtered_df = df[
        (df['total_return_pct'] > 5.0) &
        (df['win_rate_pct'] > 50.0) &
        (df['total_trades'] > 10) &
        (df['max_drawdown_pct'] < 15.0)
    ].copy()
    
    filtered_df.sort_values(by='total_return_pct', ascending=False, inplace=True)

    # --- Define columns to show ---
    # We show all relevant params, NaN for those that don't apply
    cols = [
        'strategy_name', 'total_return_pct', 'win_rate_pct', 'total_trades', 'max_drawdown_pct',
        'timeframe', 'tp_pct', 'sl_pct',
        'ema_fast', 'ema_slow', 'use_vwap',
        'bb_window', 'bb_std_dev'
    ]
    # Filter for columns that actually exist in the dataframe
    available_cols = [col for col in cols if col in df.columns]

    if filtered_df.empty:
        log_and_print("\n--- No Strategies Met Your Criteria ---")
        log_and_print("Showing the top 10 best by return (even if negative)...")
        print(df.sort_values(by='total_return_pct', ascending=False)[available_cols].head(10).to_string())
    else:
        log_and_print("\n--- 🚀 Top Performing Strategies (All Types) 🚀 ---")
        print(filtered_df[available_cols].to_string())


if __name__ == "__main__":
    run_optimizer()