# Bollinger Winrate-First Runbook

## Objective
Push BTCUSDT 1m Bollinger strategy toward higher win rate with controlled overfitting risk.

## 1) Baseline Checks
- Confirm `v2/config.ini` has the new `[STRATEGY_BOLLINGER]` keys.
- Keep `entry_mode = next_open` for live/backtest consistency.
- Start with current defaults before retuning.

## 2) Run Tuner
Example:

```bash
cd v2
python bollinger_winrate_tuner.py \
  --symbol BTCUSDT \
  --timeframe 1m \
  --start 2025-10-01 \
  --end 2026-02-01 \
  --entry-balance 1000 \
  --train-days 21 \
  --val-days 10 \
  --top-k 5 \
  --min-trades 40 \
  --use-public-mainnet-klines
```

Output:
- JSON report under `v2/reports/`
- Top coarse/fine candidates
- Walk-forward aggregate metrics
- Failure counters in report meta (`coarse_failed`, `fine_failed`) for debugging

## 3) Promotion Criteria
Promote a candidate only if all hold:
- Average validation win rate >= 60%
- Average validation trades >= 40 (or your defined minimum)
- Average validation return >= 0
- Average validation profit factor >= 1.0

## 4) Live Rollout
- Apply chosen parameters in `v2/config.ini`.
- Run live in testnet first for at least 3 full sessions.
- Verify logs include gate reasons (`trend`, `atr`, `session`, `reentry`, `cooldown`).

## 5) Disable Conditions
Disable strategy if any occurs over a rolling 7-day window:
- Win rate falls below 52%
- Profit factor < 0.95
- Consecutive daily losses > 3 days

## 6) Monthly Retune Cadence
- Retune once per month with latest 3-4 months of data.
- Keep at least one prior parameter set as fallback.
- Avoid changing more than one major filter family at once.
