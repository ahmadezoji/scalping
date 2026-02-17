#!/usr/bin/env python3
"""
Winrate-first tuner for v2/bolling_band_bot.py

Workflow:
1) Coarse grid search
2) Fine search around top coarse configs
3) Walk-forward validation on top candidates

This script writes a JSON report to:
  v2/reports/bollinger_winrate_report_<timestamp>.json
"""

import argparse
import configparser
import itertools
import json
import os
import sys
import time
from datetime import datetime, timedelta

# Ensure config.ini is resolved relative to this script.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from bolling_band_bot import backtest_bollinger_strategy


def dt(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def fmt(d: datetime) -> str:
    return d.strftime("%Y-%m-%d %H:%M:%S")


def month_splits(start_date: str, end_date: str, train_days: int, val_days: int):
    start = dt(start_date)
    end = dt(end_date)
    splits = []
    cur = start
    while cur + timedelta(days=train_days + val_days) <= end:
        train_start = cur
        train_end = cur + timedelta(days=train_days)
        val_start = train_end
        val_end = val_start + timedelta(days=val_days)
        splits.append((train_start, train_end, val_start, val_end))
        cur = cur + timedelta(days=val_days)
    return splits


def score_result(result: dict, min_trades: int):
    if not result or result["total_trades"] <= 0:
        return -1e9
    win = result["win_rate_pct"]
    pf = result.get("profit_factor", 0.0)
    ret = result.get("total_return_pct", 0.0)
    trades = result.get("total_trades", 0)
    trade_penalty = 0.0
    if trades < min_trades:
        trade_penalty = (min_trades - trades) * 0.75
    return (win * 2.0) + (pf * 15.0) + (ret * 0.5) - trade_penalty


def evaluate(symbol: str, timeframe: str, start: datetime, end: datetime, entry_balance: float, params: dict):
    return backtest_bollinger_strategy(
        symbol=symbol,
        timeframe=timeframe,
        start_date=fmt(start),
        end_date=fmt(end),
        entry_balance=entry_balance,
        **params,
    )


def load_use_testnet() -> bool:
    cfg = configparser.ConfigParser()
    cfg.read(os.path.join(SCRIPT_DIR, "config.ini"))
    return cfg.getboolean("TRADING", "use_testnet", fallback=False)


def coarse_grid():
    return list(
        itertools.product(
            [12, 14, 16],            # bb_window
            [1.7, 1.8, 1.9],         # bb_std_dev
            [20, 22, 24],            # rsi_oversold
            [66, 68, 70],            # rsi_overbought
            [1.1, 1.2, 1.3],         # volume_confirm
            [True],                  # use_vwap_filter
            [True],                  # use_trend_filter
            [True],                  # use_atr_filter
            [0.8 / 100.0, 0.9 / 100.0, 1.0 / 100.0],  # tp_pct_cfg
            [0.5 / 100.0],           # sl_pct_cfg
        )
    )


def fine_variants(base: dict):
    out = []
    for bb_std in [base["bb_std_dev"] - 0.05, base["bb_std_dev"], base["bb_std_dev"] + 0.05]:
        for vol in [max(0.9, base["volume_confirm"] - 0.1), base["volume_confirm"], base["volume_confirm"] + 0.1]:
            for tp in [max(0.006, base["tp_pct_cfg"] - 0.001), base["tp_pct_cfg"], min(0.011, base["tp_pct_cfg"] + 0.001)]:
                for dist in [0.30 / 100.0, 0.35 / 100.0, 0.40 / 100.0]:
                    p = dict(base)
                    p["bb_std_dev"] = round(bb_std, 3)
                    p["volume_confirm"] = round(vol, 3)
                    p["tp_pct_cfg"] = round(tp, 5)
                    p["trend_distance_max_pct"] = dist
                    out.append(p)
    return out


def candidate_dict(t):
    return {
        "bb_window": t[0],
        "bb_std_dev": t[1],
        "rsi_period": 14,
        "rsi_oversold": t[2],
        "rsi_overbought": t[3],
        "volume_confirm": t[4],
        "use_vwap_filter": t[5],
        "use_trend_filter": t[6],
        "trend_ema_period": 200,
        "trend_distance_max_pct": 0.35 / 100.0,
        "use_atr_filter": t[7],
        "atr_period": 14,
        "atr_min_pct": 0.08 / 100.0,
        "atr_max_pct": 0.45 / 100.0,
        "use_session_filter": True,
        "session_utc_start": "12:00",
        "session_utc_end": "20:00",
        "entry_mode": "next_open",
        "require_band_reentry_strength": True,
        "reentry_min_body_pct": 0.03 / 100.0,
        "cooldown_bars_after_loss": 3,
        "tp_pct_cfg": t[8],
        "sl_pct_cfg": t[9],
        "fee_bps": 4,
        "slippage_bps": 1,
    }


def aggregate_walk_forward(rows):
    if not rows:
        return {}
    n = len(rows)
    return {
        "folds": n,
        "avg_train_winrate": sum(r["train"]["win_rate_pct"] for r in rows) / n,
        "avg_val_winrate": sum(r["val"]["win_rate_pct"] for r in rows) / n,
        "avg_train_return": sum(r["train"]["total_return_pct"] for r in rows) / n,
        "avg_val_return": sum(r["val"]["total_return_pct"] for r in rows) / n,
        "avg_val_trades": sum(r["val"]["total_trades"] for r in rows) / n,
        "avg_val_pf": sum(r["val"].get("profit_factor", 0.0) for r in rows) / n,
    }


def main():
    parser = argparse.ArgumentParser(description="Winrate-first Bollinger tuner")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--timeframe", default="1m")
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--entry-balance", type=float, default=1000.0)
    parser.add_argument("--train-days", type=int, default=21)
    parser.add_argument("--val-days", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-trades", type=int, default=40)
    parser.add_argument("--sleep-ms", type=int, default=120, help="Throttle between API calls")
    parser.add_argument("--use-public-mainnet-klines", action="store_true")
    parser.add_argument("--allow-empty-report", action="store_true")
    parser.add_argument("--max-coarse-evals", type=int, default=0, help="0 means all")
    args = parser.parse_args()

    use_testnet = load_use_testnet()
    if use_testnet and not args.use_public_mainnet_klines:
        print(
            "WARNING: TRADING.use_testnet=true. Historical data may be sparse/empty for backtests.\n"
            "Recommendation: rerun with --use-public-mainnet-klines."
        )

    # Preflight: fail fast instead of generating silent empty reports.
    baseline_params = candidate_dict(coarse_grid()[0])
    if args.use_public_mainnet_klines:
        baseline_params["use_public_mainnet_klines"] = True
    preflight = evaluate(args.symbol, args.timeframe, dt(args.start), dt(args.end), args.entry_balance, baseline_params)
    if not preflight and not args.allow_empty_report:
        raise RuntimeError(
            "Preflight backtest returned no data/results. "
            "Likely causes: testnet historical gaps, API errors, or invalid date range. "
            "Try: --use-public-mainnet-klines and/or shorter date range."
        )

    print("== Coarse search ==")
    coarse_rows = []
    coarse_failed = 0
    for idx, tup in enumerate(coarse_grid(), 1):
        if args.max_coarse_evals > 0 and idx > args.max_coarse_evals:
            break
        params = candidate_dict(tup)
        if args.use_public_mainnet_klines:
            params["use_public_mainnet_klines"] = True
        result = evaluate(args.symbol, args.timeframe, dt(args.start), dt(args.end), args.entry_balance, params)
        if not result:
            coarse_failed += 1
            if args.sleep_ms > 0:
                time.sleep(args.sleep_ms / 1000.0)
            continue
        score = score_result(result, args.min_trades)
        coarse_rows.append({"params": params, "result": result, "score": score})
        if args.sleep_ms > 0:
            time.sleep(args.sleep_ms / 1000.0)

    coarse_rows.sort(key=lambda x: x["score"], reverse=True)
    top_coarse = coarse_rows[: args.top_k]
    coarse_attempted = (args.max_coarse_evals if args.max_coarse_evals > 0 else len(coarse_grid()))
    print(f"coarse attempted={coarse_attempted} ok={len(coarse_rows)} failed={coarse_failed}")
    for i, row in enumerate(top_coarse, 1):
        r = row["result"]
        print(
            f"[coarse #{i}] score={row['score']:.2f} win={r['win_rate_pct']:.2f}% "
            f"ret={r['total_return_pct']:.2f}% pf={r['profit_factor']:.2f} trades={r['total_trades']}"
        )

    print("== Fine search ==")
    fine_rows = []
    fine_failed = 0
    for coarse in top_coarse:
        for params in fine_variants(coarse["params"]):
            result = evaluate(args.symbol, args.timeframe, dt(args.start), dt(args.end), args.entry_balance, params)
            if not result:
                fine_failed += 1
                if args.sleep_ms > 0:
                    time.sleep(args.sleep_ms / 1000.0)
                continue
            score = score_result(result, args.min_trades)
            fine_rows.append({"params": params, "result": result, "score": score})
            if args.sleep_ms > 0:
                time.sleep(args.sleep_ms / 1000.0)

    fine_rows.sort(key=lambda x: x["score"], reverse=True)
    top_fine = fine_rows[: args.top_k]
    print(f"fine ok={len(fine_rows)} failed={fine_failed}")
    for i, row in enumerate(top_fine, 1):
        r = row["result"]
        print(
            f"[fine #{i}] score={row['score']:.2f} win={r['win_rate_pct']:.2f}% "
            f"ret={r['total_return_pct']:.2f}% pf={r['profit_factor']:.2f} trades={r['total_trades']}"
        )

    print("== Walk-forward validation ==")
    splits = month_splits(args.start, args.end, args.train_days, args.val_days)
    wf_table = []

    for i, row in enumerate(top_fine, 1):
        params = row["params"]
        fold_rows = []
        for (train_start, train_end, val_start, val_end) in splits:
            train_res = evaluate(args.symbol, args.timeframe, train_start, train_end, args.entry_balance, params)
            val_res = evaluate(args.symbol, args.timeframe, val_start, val_end, args.entry_balance, params)
            if not train_res or not val_res:
                continue
            fold_rows.append(
                {
                    "train_window": f"{train_start.date()} -> {train_end.date()}",
                    "val_window": f"{val_start.date()} -> {val_end.date()}",
                    "train": {
                        "win_rate_pct": train_res["win_rate_pct"],
                        "total_return_pct": train_res["total_return_pct"],
                        "total_trades": train_res["total_trades"],
                    },
                    "val": {
                        "win_rate_pct": val_res["win_rate_pct"],
                        "total_return_pct": val_res["total_return_pct"],
                        "total_trades": val_res["total_trades"],
                        "profit_factor": val_res.get("profit_factor", 0.0),
                    },
                }
            )

        agg = aggregate_walk_forward(fold_rows)
        wf_table.append({"rank": i, "params": params, "aggregate": agg, "folds": fold_rows})
        if agg:
            print(
                f"[wf #{i}] val_win={agg['avg_val_winrate']:.2f}% val_ret={agg['avg_val_return']:.2f}% "
                f"val_pf={agg['avg_val_pf']:.2f} val_trades={agg['avg_val_trades']:.1f} folds={agg['folds']}"
            )

    report = {
        "meta": {
            "generated_at_utc": datetime.utcnow().isoformat(),
            "symbol": args.symbol,
            "timeframe": args.timeframe,
            "start": args.start,
            "end": args.end,
            "entry_balance": args.entry_balance,
            "train_days": args.train_days,
            "val_days": args.val_days,
            "min_trades": args.min_trades,
            "top_k": args.top_k,
            "use_testnet_config": use_testnet,
            "use_public_mainnet_klines": bool(args.use_public_mainnet_klines),
            "sleep_ms": args.sleep_ms,
            "max_coarse_evals": args.max_coarse_evals,
            "coarse_ok": len(coarse_rows),
            "coarse_failed": coarse_failed,
            "fine_ok": len(fine_rows),
            "fine_failed": fine_failed,
        },
        "top_coarse": top_coarse,
        "top_fine": top_fine,
        "walk_forward": wf_table,
    }

    out_dir = os.path.join(os.path.dirname(__file__), "reports")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"bollinger_winrate_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"report written: {out_path}")


if __name__ == "__main__":
    main()
