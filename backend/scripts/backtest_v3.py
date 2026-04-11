# backend/scripts/backtest_v3.py

import pandas as pd
import logging

from backend.services.strategy_engine import StrategyEngine
from backend.services.risk_manager import RiskManager
from backend.services.data_loader import load_all_timeframes
from backend.services.ict_strategy import run_ict_pipeline
from backend.services.state_machine import run_state_machine
from backend.services.multi_timeframe import merge_htf_into_ltf
from backend.services.feature_builder import build_features

logger = logging.getLogger(__name__)


def run_backtest():
    print("\n=== PRODUCTION BACKTEST (StrategyEngine) ===\n")

    # Load data
    data = load_all_timeframes()
    df = data["15m"].copy()

    # Pipeline
    df = run_ict_pipeline(df)
    df = run_state_machine(df)
    df = merge_htf_into_ltf(df, {"1h": data["1h"], "4h": data["4h"]})
    df = build_features(df)

    engine = StrategyEngine()
    rm = RiskManager(compound=False)

    trades = 0

    for i in range(50, len(df) - 30):
        window = df.iloc[:i]
        row = df.iloc[i]

        signal = engine.evaluate(window, row, timeframe="15m")

        if signal.signal == "NO TRADE" or not signal.executable:
            continue

        trade = rm.calculate_position(
            entry=signal.entry,
            sl=signal.sl,
            direction=signal.signal,
            ts=row.name,
            strategy_type=signal.strategy_type,
            rr_override=signal.rr,
            risk_pct_override=signal.risk_pct,
            be_threshold=signal.be_threshold,
        )

        if not trade:
            continue

        # Simulate outcome
        future = df.iloc[i + 1 : i + 30]
        outcome = "SL"

        for _, f in future.iterrows():
            if signal.signal == "BUY":
                if f["low"] <= trade.sl:
                    outcome = "SL"
                    break
                if f["high"] >= trade.tp:
                    outcome = "TP"
                    break
            else:
                if f["high"] >= trade.sl:
                    outcome = "SL"
                    break
                if f["low"] <= trade.tp:
                    outcome = "TP"
                    break

        rm.record_outcome(trade, outcome)
        trades += 1

    summary = rm.summary()

    print("Trades:", trades)
    print("Win Rate:", summary["win_rate_pct"], "%")
    print("PnL:", summary["net_pnl_pct"], "%")
    print("Max DD:", summary["max_drawdown_pct"], "%")

    return summary

if __name__ == "__main__":
    run_backtest()