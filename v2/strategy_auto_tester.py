import pandas as pd
import itertools
import time
import random
from datetime import datetime, timedelta

# --- NEW: Import BOTH backtest functions ---
from momentum_trader_bot import backtest_momentum_strategy
from bolling_band_bot import backtest_bollinger_strategy


from binance_helper import log_and_print

# -----------------------------------------------------------------
# --- 1. DEFINE YOUR PARAMETER GRIDS ---
# -----------------------------------------------------------------

# --- Grid 1: Your original Momentum/EMA strategy ---
momentum_grid = {
    "timeframe": ["1m", "3m", "5m"],
    "ema_fast": [3, 5, 8, 12],
    "ema_slow": [15, 21, 30, 50],
    "tp_pct_cfg": [0.004, 0.006, 0.008, 0.012],  # 0.4%..1.2%
    "sl_pct_cfg": [0.003, 0.005, 0.008],         # 0.3%..0.8%
    "use_vwap": [True],  # Keep on for now; test False later if needed
}

# --- Grid 2: The NEW Bollinger Band strategy ---
bollinger_grid = {
    "timeframe": ["5m", "15m"],
    "bb_window": [20, 30],
    "bb_std_dev": [2.0, 2.5],
    "tp_pct_cfg": [0.005, 0.01],  # Tighter TP: 0.5%, 1.0%
    "sl_pct_cfg": [0.003, 0.005], # Tighter SL: 0.3%, 0.5%
}

# --- Backtest constants (daily focus) ---
SYMBOL = "BTCUSDT"
ENTRY_BALANCE = 100.0
DAILY_YEAR = 2025
DAILY_MONTH = 11
DAILY_SAMPLES = 20


def _iter_random_days(year: int, month: int, samples: int):
    first_day = datetime(year, month, 1)
    if month == 12:
        next_month = datetime(year + 1, 1, 1)
    else:
        next_month = datetime(year, month + 1, 1)
    days_in_month = (next_month - first_day).days
    samples = max(1, min(samples, days_in_month))
    day_numbers = random.sample(range(1, days_in_month + 1), samples)
    day_numbers.sort()
    for day in day_numbers:
        start_dt = datetime(year, month, day)
        end_dt = start_dt + timedelta(days=1)
        yield start_dt, end_dt


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
    # bollinger_combos = get_combinations(bollinger_grid)
    
    # Add a 'strategy_name' tag to each
    # This is how we'll know which function to call
    for p in momentum_combos: p['strategy_name'] = 'momentum'
    # for p in bollinger_combos: p['strategy_name'] = 'bollinger'
    
    # Combine them into one giant list of jobs
    # all_combinations = bollinger_combos + momentum_combos
    all_combinations = momentum_combos
    
    total_runs = len(all_combinations)
    log_and_print(f"--- Starting Multi-Strategy Optimization ---")
    log_and_print(f"Total combinations to test: {total_runs}")
    
    all_results = []
    start_time = time.time()

    # --- 3. RUN BACKTEST FOR EACH COMBINATION (DAILY SLICES) ---
    for i, params in enumerate(all_combinations):
        
        log_and_print(f"\n--- [{i+1}/{total_runs}] TESTING {params['strategy_name']} ---")
        param_str = ", ".join(f"{k}={v}" for k, v in params.items())
        log_and_print(f"Params: {param_str}")
        
        daily_results = []
        try:
            # --- NEW: CONDITIONAL LOGIC ---
            # Call the correct backtest function based on the 'strategy_name'
            
            for start_dt, end_dt in _iter_random_days(DAILY_YEAR, DAILY_MONTH, DAILY_SAMPLES):
                if params['strategy_name'] == 'momentum':
                    if params["ema_fast"] >= params["ema_slow"]:
                        log_and_print("SKIP: EMA fast >= slow")
                        daily_results = []
                        break
                    if params["sl_pct_cfg"] >= params["tp_pct_cfg"]:
                        log_and_print("SKIP: SL >= TP")
                        daily_results = []
                        break

                    results = backtest_momentum_strategy(
                        symbol=SYMBOL,
                        start_date=start_dt.strftime("%Y-%m-%d %H:%M:%S"),
                        end_date=end_dt.strftime("%Y-%m-%d %H:%M:%S"),
                        entry_balance=ENTRY_BALANCE,
                        timeframe=params["timeframe"],
                        ema_fast=params["ema_fast"],
                        ema_slow=params["ema_slow"],
                        tp_pct_cfg=params["tp_pct_cfg"],
                        sl_pct_cfg=params["sl_pct_cfg"],
                        use_vwap=params["use_vwap"],
                    )
                elif params['strategy_name'] == 'bollinger':
                    if params["sl_pct_cfg"] >= params["tp_pct_cfg"]:
                        log_and_print("SKIP: SL >= TP")
                        daily_results = []
                        break

                    results = backtest_bollinger_strategy(
                        symbol=SYMBOL,
                        start_date=start_dt.strftime("%Y-%m-%d %H:%M:%S"),
                        end_date=end_dt.strftime("%Y-%m-%d %H:%M:%S"),
                        entry_balance=ENTRY_BALANCE,
                        timeframe=params["timeframe"],
                        bb_window=params["bb_window"],
                        bb_std_dev=params["bb_std_dev"],
                        tp_pct_cfg=params["tp_pct_cfg"],
                        sl_pct_cfg=params["sl_pct_cfg"],
                    )
                else:
                    results = None

                if results:
                    daily_results.append(results)

            if daily_results:
                days = len(daily_results)
                pos_days = sum(1 for r in daily_results if r["total_return_pct"] > 0)
                neg_days = sum(1 for r in daily_results if r["total_return_pct"] < 0)
                flat_days = days - pos_days - neg_days

                agg = daily_results[0].copy()
                agg["days_tested"] = days
                agg["avg_daily_return_pct"] = sum(r["total_return_pct"] for r in daily_results) / days
                agg["avg_trades_per_day"] = sum(r["total_trades"] for r in daily_results) / days
                agg["avg_win_rate_pct"] = sum(r["win_rate_pct"] for r in daily_results) / days
                agg["positive_days_pct"] = (pos_days / days) * 100.0
                agg["negative_days_pct"] = (neg_days / days) * 100.0
                agg["flat_days_pct"] = (flat_days / days) * 100.0
                all_results.append(agg)
                
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
        (df['avg_daily_return_pct'] > 0.2) &
        (df['positive_days_pct'] > 55.0) &
        (df['avg_trades_per_day'] >= 1.0)
    ].copy()
    filtered_df.sort_values(by='avg_daily_return_pct', ascending=False, inplace=True)

    # --- Define columns to show ---
    # We show all relevant params, NaN for those that don't apply
    cols = [
        'strategy_name', 'avg_daily_return_pct', 'avg_win_rate_pct', 'avg_trades_per_day',
        'positive_days_pct', 'negative_days_pct', 'flat_days_pct', 'days_tested',
        'timeframe', 'tp_pct', 'sl_pct',
        'ema_fast', 'ema_slow', 'use_vwap',
        'bb_window', 'bb_std_dev'
    ]
    # Filter for columns that actually exist in the dataframe
    available_cols = [col for col in cols if col in df.columns]

    if filtered_df.empty:
        log_and_print("\n--- No Strategies Met Your Criteria ---")
        log_and_print("Showing the top 10 best by avg daily return...")
        print(df.sort_values(by='avg_daily_return_pct', ascending=False)[available_cols].head(10).to_string())
    else:
        log_and_print("\n--- 🚀 Top Performing Strategies (All Types) 🚀 ---")
        print(filtered_df[available_cols].to_string())

    # --- Save filtered_df to a report.txt file ---
    with open("report.txt", "w") as report_file:
        if filtered_df.empty:
            report_file.write("No strategies met the criteria.\n")
            report_file.write("Top 10 best by avg daily return:\n")
            report_file.write(df.sort_values(by='avg_daily_return_pct', ascending=False)[available_cols].head(10).to_string())
        else:
            report_file.write("Top Performing Strategies (All Types):\n")
            report_file.write(filtered_df[available_cols].to_string())


if __name__ == "__main__":
    run_optimizer()
