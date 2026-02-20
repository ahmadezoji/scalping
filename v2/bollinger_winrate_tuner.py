#!/usr/bin/env python3
"""
Robust winrate-first tuner for v2/bolling_band_bot.py

Workflow:
1) Coarse search (with hard in-sample trade-count gate)
2) Fine search around top coarse candidates
3) Walk-forward validation with eligibility gates
"""

import argparse
import configparser
import itertools
import json
import os
import statistics
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


def load_use_testnet() -> bool:
    cfg = configparser.ConfigParser()
    cfg.read(os.path.join(SCRIPT_DIR, "config.ini"))
    return cfg.getboolean("TRADING", "use_testnet", fallback=False)


def evaluate(symbol: str, timeframe: str, start: datetime, end: datetime, entry_balance: float, params: dict, max_pf_cap: float):
    payload = dict(params)
    payload["max_profit_factor_cap"] = max_pf_cap
    return backtest_bollinger_strategy(
        symbol=symbol,
        timeframe=timeframe,
        start_date=fmt(start),
        end_date=fmt(end),
        entry_balance=entry_balance,
        **payload,
    )


def candidate_dict(t):
    return {
        "bb_window": t[0],
        "bb_std_dev": t[1],
        "rsi_period": 14,
        "rsi_oversold": t[2],
        "rsi_overbought": t[3],
        "volume_confirm": t[4],
        "use_vwap_filter": True,
        "use_trend_filter": True,
        "trend_ema_period": 200,
        "trend_distance_max_pct": t[5],
        "use_atr_filter": True,
        "atr_period": 14,
        "atr_min_pct": 0.08 / 100.0,
        "atr_max_pct": t[6],
        "use_session_filter": True,
        "session_utc_start": t[7],
        "session_utc_end": t[8],
        "entry_mode": "next_open",
        "require_band_reentry_strength": t[9],
        "reentry_min_body_pct": 0.03 / 100.0,
        "cooldown_bars_after_loss": 3,
        "tp_pct_cfg": t[10],
        "sl_pct_cfg": t[11],
        "fee_bps": 4,
        "slippage_bps": 1,
    }


def coarse_grid():
    # Conservative band
    conservative = itertools.product(
        [12, 14, 16],
        [1.7, 1.8, 1.9],
        [21, 22, 23, 24],
        [66, 67, 68, 69, 70],
        [1.0, 1.1, 1.2],
        [0.35 / 100.0],
        [0.45 / 100.0],
        ["12:00"],
        ["20:00"],
        [True],
        [0.008, 0.009, 0.010],
        [0.005],
    )
    # Balanced band
    balanced = itertools.product(
        [12, 14, 16],
        [1.7, 1.8, 1.9],
        [21, 22, 23, 24],
        [66, 67, 68, 69, 70],
        [1.0, 1.1, 1.2],
        [0.35 / 100.0, 0.45 / 100.0, 0.55 / 100.0],
        [0.45 / 100.0, 0.60 / 100.0, 0.75 / 100.0],
        ["10:00", "12:00"],
        ["22:00", "20:00"],
        [True, False],
        [0.008, 0.009, 0.010],
        [0.005, 0.006],
    )
    out = [candidate_dict(x) for x in conservative]
    out.extend(candidate_dict(x) for x in balanced)
    # Deduplicate by stable json key
    uniq = {}
    for p in out:
        uniq[json.dumps(p, sort_keys=True)] = p
    return list(uniq.values())


def fine_variants(base: dict):
    out = []
    for bb_std in [base["bb_std_dev"] - 0.05, base["bb_std_dev"], base["bb_std_dev"] + 0.05]:
        for vol in [max(0.9, base["volume_confirm"] - 0.1), base["volume_confirm"], base["volume_confirm"] + 0.1]:
            for tp in [max(0.008, base["tp_pct_cfg"] - 0.001), base["tp_pct_cfg"], min(0.010, base["tp_pct_cfg"] + 0.001)]:
                for sl in [max(0.005, base["sl_pct_cfg"] - 0.001), base["sl_pct_cfg"], min(0.006, base["sl_pct_cfg"] + 0.001)]:
                    for dist in [0.35 / 100.0, 0.45 / 100.0, 0.55 / 100.0]:
                        for atr_max in [0.45 / 100.0, 0.60 / 100.0, 0.75 / 100.0]:
                            p = dict(base)
                            p["bb_std_dev"] = round(bb_std, 3)
                            p["volume_confirm"] = round(vol, 3)
                            p["tp_pct_cfg"] = round(tp, 5)
                            p["sl_pct_cfg"] = round(sl, 5)
                            p["trend_distance_max_pct"] = dist
                            p["atr_max_pct"] = atr_max
                            out.append(p)
    uniq = {}
    for p in out:
        uniq[json.dumps(p, sort_keys=True)] = p
    return list(uniq.values())


def in_sample_eligibility(result: dict, min_trades: int):
    reasons = []
    if not result:
        reasons.append("missing_result")
        return False, reasons
    if result.get("total_trades", 0) < min_trades:
        reasons.append(f"train_trades_lt_{min_trades}")
    return len(reasons) == 0, reasons


def in_sample_score(result: dict):
    # Used only for coarse/fine ordering before walk-forward.
    win = result.get("win_rate_pct", 0.0)
    ret = result.get("total_return_pct", 0.0)
    pf = result.get("profit_factor_capped", result.get("profit_factor", 0.0))
    trades = result.get("total_trades", 0)
    return (win * 2.5) + (ret * 1.0) + (pf * 1.5) + min(trades, 200) * 0.03


def aggregate_walk_forward(rows):
    if not rows:
        return {}
    n = len(rows)
    train_wins = [r["train"]["win_rate_pct"] for r in rows]
    val_wins = [r["val"]["win_rate_pct"] for r in rows]
    train_rets = [r["train"]["total_return_pct"] for r in rows]
    val_rets = [r["val"]["total_return_pct"] for r in rows]
    val_trades = [r["val"]["total_trades"] for r in rows]
    val_pf = [r["val"].get("profit_factor_capped", 0.0) for r in rows]
    eligible_flags = [bool(r.get("fold_eligible", False)) for r in rows]
    eligible_rows = [r for r in rows if r.get("fold_eligible", False)]

    out = {
        "folds": n,
        "eligible_folds": sum(eligible_flags),
        "avg_train_winrate": sum(train_wins) / n,
        "avg_val_winrate": sum(val_wins) / n,
        "avg_train_return": sum(train_rets) / n,
        "avg_val_return": sum(val_rets) / n,
        "avg_val_trades": sum(val_trades) / n,
        "avg_val_pf_capped": sum(val_pf) / n,
        "val_winrate_std": statistics.pstdev(val_wins) if len(val_wins) > 1 else 0.0,
        "val_return_std": statistics.pstdev(val_rets) if len(val_rets) > 1 else 0.0,
    }
    if eligible_rows:
        m = len(eligible_rows)
        out["eligible_avg_val_winrate"] = sum(r["val"]["win_rate_pct"] for r in eligible_rows) / m
        out["eligible_avg_val_return"] = sum(r["val"]["total_return_pct"] for r in eligible_rows) / m
        out["eligible_avg_val_trades"] = sum(r["val"]["total_trades"] for r in eligible_rows) / m
        out["eligible_avg_val_pf_capped"] = sum(r["val"].get("profit_factor_capped", 0.0) for r in eligible_rows) / m
    else:
        out["eligible_avg_val_winrate"] = 0.0
        out["eligible_avg_val_return"] = -999.0
        out["eligible_avg_val_trades"] = 0.0
        out["eligible_avg_val_pf_capped"] = 0.0
    return out


def wf_eligibility(agg: dict, min_val_trades_per_fold: int, min_wf_folds: int):
    reasons = []
    if not agg:
        reasons.append("no_walkforward_data")
        return False, reasons
    if agg.get("eligible_folds", 0) < min_wf_folds:
        reasons.append(f"eligible_folds_lt_{min_wf_folds}")
    if agg.get("avg_val_trades", 0.0) < float(min_val_trades_per_fold):
        reasons.append(f"avg_val_trades_lt_{min_val_trades_per_fold}")
    if agg.get("avg_val_return", -999.0) < 0:
        reasons.append("avg_val_return_negative")
    return len(reasons) == 0, reasons


def robust_wf_score(agg: dict):
    win = agg.get("avg_val_winrate", 0.0)
    ret = agg.get("avg_val_return", 0.0)
    pf = agg.get("avg_val_pf_capped", 0.0)
    win_std = agg.get("val_winrate_std", 0.0)
    ret_std = agg.get("val_return_std", 0.0)
    # Prioritize winrate, then return, then PF; penalize instability.
    return (win * 4.0) + (ret * 2.0) + (pf * 1.0) - (win_std * 0.40) - (ret_std * 0.75)


def main():
    parser = argparse.ArgumentParser(description="Robust winrate-first Bollinger tuner")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--timeframe", default="1m")
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--entry-balance", type=float, default=1000.0)
    parser.add_argument("--train-days", type=int, default=21)
    parser.add_argument("--val-days", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-trades", type=int, default=40)
    parser.add_argument("--min-val-trades-per-fold", type=int, default=8)
    parser.add_argument("--min-wf-folds", type=int, default=6)
    parser.add_argument("--max-profit-factor-cap", type=float, default=5.0)
    parser.add_argument("--ranking-mode", default="wf_robust", choices=["wf_robust"])
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

    # Preflight
    cg = coarse_grid()
    if not cg:
        raise RuntimeError("coarse grid is empty")
    baseline_params = dict(cg[0])
    if args.use_public_mainnet_klines:
        baseline_params["use_public_mainnet_klines"] = True
    preflight = evaluate(args.symbol, args.timeframe, dt(args.start), dt(args.end), args.entry_balance, baseline_params, args.max_profit_factor_cap)
    if not preflight and not args.allow_empty_report:
        raise RuntimeError(
            "Preflight backtest returned no data/results. "
            "Likely causes: historical fetch errors, API issues, or invalid date range."
        )

    print("== Coarse search ==")
    coarse_rows = []
    coarse_failed = 0
    coarse_attempted = 0
    for idx, params0 in enumerate(cg, 1):
        if args.max_coarse_evals > 0 and idx > args.max_coarse_evals:
            break
        coarse_attempted += 1
        params = dict(params0)
        if args.use_public_mainnet_klines:
            params["use_public_mainnet_klines"] = True
        result = evaluate(args.symbol, args.timeframe, dt(args.start), dt(args.end), args.entry_balance, params, args.max_profit_factor_cap)
        if not result:
            coarse_failed += 1
            if args.sleep_ms > 0:
                time.sleep(args.sleep_ms / 1000.0)
            continue
        eligible, reasons = in_sample_eligibility(result, args.min_trades)
        score = in_sample_score(result)
        coarse_rows.append({
            "params": params,
            "result": result,
            "score": score,
            "eligible": eligible,
            "ineligibility_reasons": reasons,
        })
        if args.sleep_ms > 0:
            time.sleep(args.sleep_ms / 1000.0)

    coarse_rows.sort(key=lambda x: (x["eligible"], x["score"]), reverse=True)
    eligible_coarse = [r for r in coarse_rows if r["eligible"]]
    top_coarse = (eligible_coarse if eligible_coarse else coarse_rows)[: args.top_k]

    print(f"coarse attempted={coarse_attempted} ok={len(coarse_rows)} failed={coarse_failed} eligible={len(eligible_coarse)}")
    for i, row in enumerate(top_coarse, 1):
        r = row["result"]
        print(
            f"[coarse #{i}] eligible={row['eligible']} score={row['score']:.2f} "
            f"win={r['win_rate_pct']:.2f}% ret={r['total_return_pct']:.2f}% "
            f"pf_cap={r.get('profit_factor_capped', r.get('profit_factor', 0.0)):.2f} trades={r['total_trades']}"
        )

    print("== Fine search ==")
    fine_rows = []
    fine_failed = 0
    for coarse in top_coarse:
        for params0 in fine_variants(coarse["params"]):
            params = dict(params0)
            if args.use_public_mainnet_klines:
                params["use_public_mainnet_klines"] = True
            result = evaluate(args.symbol, args.timeframe, dt(args.start), dt(args.end), args.entry_balance, params, args.max_profit_factor_cap)
            if not result:
                fine_failed += 1
                if args.sleep_ms > 0:
                    time.sleep(args.sleep_ms / 1000.0)
                continue
            eligible, reasons = in_sample_eligibility(result, args.min_trades)
            score = in_sample_score(result)
            fine_rows.append({
                "params": params,
                "result": result,
                "score": score,
                "eligible": eligible,
                "ineligibility_reasons": reasons,
            })
            if args.sleep_ms > 0:
                time.sleep(args.sleep_ms / 1000.0)

    fine_rows.sort(key=lambda x: (x["eligible"], x["score"]), reverse=True)
    eligible_fine = [r for r in fine_rows if r["eligible"]]
    top_fine = (eligible_fine if eligible_fine else fine_rows)[: args.top_k]

    print(f"fine ok={len(fine_rows)} failed={fine_failed} eligible={len(eligible_fine)}")
    for i, row in enumerate(top_fine, 1):
        r = row["result"]
        print(
            f"[fine #{i}] eligible={row['eligible']} score={row['score']:.2f} "
            f"win={r['win_rate_pct']:.2f}% ret={r['total_return_pct']:.2f}% "
            f"pf_cap={r.get('profit_factor_capped', r.get('profit_factor', 0.0)):.2f} trades={r['total_trades']}"
        )

    print("== Walk-forward validation ==")
    splits = month_splits(args.start, args.end, args.train_days, args.val_days)
    wf_table = []

    for i, row in enumerate(top_fine, 1):
        params = row["params"]
        fold_rows = []
        for (train_start, train_end, val_start, val_end) in splits:
            train_res = evaluate(args.symbol, args.timeframe, train_start, train_end, args.entry_balance, params, args.max_profit_factor_cap)
            val_res = evaluate(args.symbol, args.timeframe, val_start, val_end, args.entry_balance, params, args.max_profit_factor_cap)
            if not train_res or not val_res:
                continue

            fold_reasons = []
            if train_res.get("total_trades", 0) < args.min_trades:
                fold_reasons.append(f"train_trades_lt_{args.min_trades}")
            if val_res.get("total_trades", 0) < args.min_val_trades_per_fold:
                fold_reasons.append(f"val_trades_lt_{args.min_val_trades_per_fold}")
            if val_res.get("total_return_pct", -999.0) < 0:
                fold_reasons.append("val_return_negative")
            fold_eligible = len(fold_reasons) == 0

            fold_rows.append(
                {
                    "train_window": f"{train_start.date()} -> {train_end.date()}",
                    "val_window": f"{val_start.date()} -> {val_end.date()}",
                    "fold_eligible": fold_eligible,
                    "fold_ineligibility_reasons": fold_reasons,
                    "train": {
                        "win_rate_pct": train_res["win_rate_pct"],
                        "total_return_pct": train_res["total_return_pct"],
                        "total_trades": train_res["total_trades"],
                        "profit_factor_capped": train_res.get("profit_factor_capped", train_res.get("profit_factor", 0.0)),
                    },
                    "val": {
                        "win_rate_pct": val_res["win_rate_pct"],
                        "total_return_pct": val_res["total_return_pct"],
                        "total_trades": val_res["total_trades"],
                        "profit_factor_capped": val_res.get("profit_factor_capped", val_res.get("profit_factor", 0.0)),
                    },
                }
            )

        agg = aggregate_walk_forward(fold_rows)
        eligible, reasons = wf_eligibility(agg, args.min_val_trades_per_fold, args.min_wf_folds)
        robust_score = robust_wf_score(agg) if eligible else -1e9

        wf_table.append(
            {
                "rank": i,
                "params": params,
                "aggregate": agg,
                "folds": fold_rows,
                "eligible": eligible,
                "ineligibility_reasons": reasons,
                "robust_score": robust_score,
            }
        )

        if agg:
            print(
                f"[wf #{i}] eligible={eligible} score={robust_score:.2f} "
                f"val_win={agg.get('avg_val_winrate',0):.2f}% val_ret={agg.get('avg_val_return',0):.2f}% "
                f"val_pf_cap={agg.get('avg_val_pf_capped',0):.2f} val_trades={agg.get('avg_val_trades',0):.1f} "
                f"eligible_folds={agg.get('eligible_folds',0)}/{agg.get('folds',0)}"
            )

    wf_table.sort(key=lambda x: (x["eligible"], x["robust_score"]), reverse=True)

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
            "min_val_trades_per_fold": args.min_val_trades_per_fold,
            "min_wf_folds": args.min_wf_folds,
            "max_profit_factor_cap": args.max_profit_factor_cap,
            "ranking_mode": args.ranking_mode,
            "top_k": args.top_k,
            "use_testnet_config": use_testnet,
            "use_public_mainnet_klines": bool(args.use_public_mainnet_klines),
            "sleep_ms": args.sleep_ms,
            "max_coarse_evals": args.max_coarse_evals,
            "coarse_ok": len(coarse_rows),
            "coarse_failed": coarse_failed,
            "coarse_eligible": len(eligible_coarse),
            "fine_ok": len(fine_rows),
            "fine_failed": fine_failed,
            "fine_eligible": len(eligible_fine),
        },
        "top_coarse": top_coarse,
        "top_fine": top_fine,
        "walk_forward": wf_table[: args.top_k],
    }

    out_dir = os.path.join(os.path.dirname(__file__), "reports")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"bollinger_winrate_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"report written: {out_path}")


if __name__ == "__main__":
    main()
