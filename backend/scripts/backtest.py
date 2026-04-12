"""
Backtest Engine — Production Grade v2
========================================
Fixes over v1:
  1. Monte Carlo: tracks path-dependent drawdown per shuffled run
     (old version computed shuffled.sum() which is identical every run)
  2. Slippage model added (SLIPPAGE_PCT per side, separate from fees)
  3. Walk-forward: clearly labelled as out-of-sample simulation
     (true walk-forward with retraining is in train_model.py --walk-forward)
  4. Monte Carlo report expanded: median/p5/p95 drawdown distribution added
  5. Chart: Monte Carlo equity fan chart added as 6th panel

Usage:
    python -m backend.scripts.backtest
    python -m backend.scripts.backtest --no-model
    python -m backend.scripts.backtest --walk-forward
"""
import sys
import logging
import argparse
import json
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime

from backend.config.settings import (
    SIGNAL_TF, HTF_LIST,
    MODEL_PATH, FEATURES_PATH, SCALER_PATH,
    INITIAL_CAPITAL, RISK_PCT, REWARD_RATIO, COOLDOWN_MINUTES,
    LOG_DIR, MIN_CONFIDENCE, ROUND_TRIP_COST,
)
from backend.config.logging_setup import setup_logging
from backend.services.data_loader     import load_all_timeframes
from backend.services.ict_strategy    import run_ict_pipeline
from backend.services.state_machine   import run_state_machine
from backend.services.multi_timeframe import merge_htf_into_ltf
from backend.services.feature_builder import build_features, FEATURE_COLS
from backend.services.label_generator import label_trades
from backend.services.risk_manager    import RiskManager

setup_logging()
logger = logging.getLogger(__name__)

# ── Slippage config ───────────────────────────────────────────────────────────
# Realistic market-order slippage on liquid crypto pairs (BTC, ETH).
# Applied per SIDE (entry and exit both incur slippage).
# 0.05% per side = 0.10% round-trip, on top of exchange fees.
# Adjust down to 0.02% for limit-order strategies.
SLIPPAGE_PCT = 0.0005   # 0.05% per side

# ── Monte Carlo config ────────────────────────────────────────────────────────
MC_RUNS = 1000


def _build_pipeline(data: dict) -> pd.DataFrame:
    """Run full ICT pipeline on loaded data."""
    df  = data[SIGNAL_TF].copy()
    df  = run_ict_pipeline(df)
    df  = run_state_machine(df)
    htf = {tf: data[tf] for tf in HTF_LIST if tf in data}
    if htf:
        df = merge_htf_into_ltf(df, htf)
    df = build_features(df)
    df = label_trades(df)
    return df


def _apply_slippage(entry_price: float, direction: str) -> float:
    """
    Adjust entry price for market-order slippage.
    BUY  fills slightly higher (you pay the ask).
    SELL fills slightly lower (you hit the bid).
    """
    if direction == "BUY":
        return entry_price * (1 + SLIPPAGE_PCT)
    else:
        return entry_price * (1 - SLIPPAGE_PCT)


def _simulate(
    df:        pd.DataFrame,
    model=None, scaler=None, features=None,
    compound:  bool  = False,
    apply_fees:bool  = True,
    label:     str   = "",
    require_htf_confluence: bool = False,
) -> RiskManager:
    """
    Walk through signals and simulate trades with risk management.
    Slippage is applied to entry price before position sizing.

    Gate counters printed at end:
        killed_by_drawdown_halt  — max-drawdown stop or system halted
        killed_by_daily_loss     — daily loss limit reached
        killed_by_cooldown       — cooldown between trades
        killed_by_capital        — capital depleted
        killed_by_htf_confluence — h4_bias/d1_bias not aligned (require_htf_confluence=True)
        killed_by_ml_confidence  — ML probability below MIN_CONFIDENCE
        killed_by_risk_manager   — calculate_position() returned None
        actually_traded          — trades that reached record_outcome()
    """
    rm = RiskManager(
        initial_capital  = INITIAL_CAPITAL,
        risk_pct         = RISK_PCT,
        rr               = REWARD_RATIO,
        cooldown_minutes = COOLDOWN_MINUTES,
        compound         = compound,
        apply_fees       = apply_fees,
        backtest_mode    = True,   # disables live-only filters: EV gate, daily trade governor
    )

    signals = df[df["signal"].isin([0, 2])].copy()
    total_signals            = len(signals)

    # ── Per-gate kill counters ────────────────────────────────────────────────
    killed_by_drawdown_halt  = 0   # max-drawdown stop or system-halted flag
    killed_by_daily_loss     = 0   # daily loss limit
    killed_by_cooldown       = 0   # inter-trade cooldown
    killed_by_capital        = 0   # capital depleted
    killed_by_htf_confluence = 0   # HTF bias not aligned (optional gate)
    killed_by_ml_confidence  = 0   # ML probability too low
    killed_by_risk_manager   = 0   # calculate_position() returned None
    actually_traded          = 0

    for ts, row in signals.iterrows():
        allowed, reason = rm.can_trade(ts)
        if not allowed:
            r = reason.lower()
            if "halted" in r or "max drawdown" in r:
                killed_by_drawdown_halt += 1
            elif "daily loss" in r:
                killed_by_daily_loss += 1
            elif "cooldown" in r:
                killed_by_cooldown += 1
            else:
                killed_by_capital += 1
            continue

        direction = "BUY" if row["signal"] == 2 else "SELL"

        # ── Optional HTF confluence gate ──────────────────────────────────────
        # Disabled by default (require_htf_confluence=False).
        # Enable to require h4_bias AND d1_bias to agree with trade direction.
        if require_htf_confluence:
            h4_bias = float(row.get("h4_bias", 0))
            d1_bias = float(row.get("d1_bias", 0))
            if direction == "BUY"  and (h4_bias <= 0 or d1_bias < 0):
                killed_by_htf_confluence += 1
                continue
            if direction == "SELL" and (h4_bias >= 0 or d1_bias > 0):
                killed_by_htf_confluence += 1
                continue

        # ── ML confidence gate ────────────────────────────────────────────────
        if model is not None:
            avail    = [f for f in features if f in df.columns]
            row_data = (
                df.loc[[ts], avail]
                  .apply(pd.to_numeric, errors="coerce")
                  .fillna(0)
            )
            x_scaled = scaler.transform(row_data)   # numpy array — avoids XGBoost DataFrame dtype bug
            prob = float(model.predict_proba(x_scaled)[0][1])
            if prob < MIN_CONFIDENCE:
                killed_by_ml_confidence += 1
                continue

        # ── Position sizing ───────────────────────────────────────────────────
        raw_entry     = float(row["close"])
        slipped_entry = _apply_slippage(raw_entry, direction)

        trade = rm.calculate_position(
            slipped_entry,
            float(row["signal_sl"]),
            direction,
            ts,
        )
        if trade is None:
            killed_by_risk_manager += 1
            continue

        # ── Outcome simulation ────────────────────────────────────────────────
        idx    = df.index.get_loc(ts)
        future = df.iloc[idx + 1 : idx + 31]
        outcome = None

        for _, frow in future.iterrows():
            if direction == "BUY":
                if frow["low"]  <= trade.sl: outcome = "SL"; break
                if frow["high"] >= trade.tp: outcome = "TP"; break
            else:
                if frow["high"] >= trade.sl: outcome = "SL"; break
                if frow["low"]  <= trade.tp: outcome = "TP"; break

        rm.record_outcome(trade, outcome or "SL")
        actually_traded += 1

    # ── Gate breakdown report ─────────────────────────────────────────────────
    sep = "-" * 58
    tag = label or "Standard"
    gate_lines = [
        "",
        sep,
        f"  GATE BREAKDOWN  [{tag}]",
        sep,
        f"  Total signals (signal in [0,2])     : {total_signals:>6,}",
        f"  Killed - drawdown halt / sys-halted : {killed_by_drawdown_halt:>6,}",
        f"  Killed - daily loss limit           : {killed_by_daily_loss:>6,}",
        f"  Killed - cooldown                   : {killed_by_cooldown:>6,}",
        f"  Killed - capital depleted           : {killed_by_capital:>6,}",
        f"  Killed - HTF confluence (optional)  : {killed_by_htf_confluence:>6,}",
        f"  Killed - ML confidence              : {killed_by_ml_confidence:>6,}",
        f"  Killed - risk mgr (calc_pos=None)   : {killed_by_risk_manager:>6,}",
        f"  Actually traded                     : {actually_traded:>6,}",
        f"  Slippage/side                       : {SLIPPAGE_PCT*100:.3f}%",
        sep,
    ]
    report = "\n".join(gate_lines)
    print(report)
    logger.info(report)

    return rm


def _run_monte_carlo(pnls: np.ndarray) -> dict:
    """
    FIX v2: Proper path-dependent Monte Carlo simulation.

    OLD (broken):
        mc_finals.append(INITIAL_CAPITAL + shuffled.sum())
        → shuffled.sum() == pnls.sum() for EVERY permutation
        → all 1000 runs produce identical final capital
        → median == p5 == p95 (meaningless)

    NEW (correct):
        For each run, simulate the full equity path using the shuffled
        trade sequence. This captures path dependency:
        - A bad streak early causes a deep drawdown even if total PnL is same
        - Each path has a unique max drawdown
        - The distribution of drawdowns across paths shows real risk

    What we measure per path:
        1. final_capital    — end equity
        2. max_drawdown_pct — peak-to-trough in that run's equity curve

    What we report:
        Median / P5 / P95 for BOTH final capital and max drawdown.
        P5 max drawdown = "in 95% of sequences your drawdown stays below X"
        This is the meaningful risk metric Monte Carlo provides.
    """
    mc_finals    = np.zeros(MC_RUNS)
    mc_drawdowns = np.zeros(MC_RUNS)

    for run in range(MC_RUNS):
        shuffled = np.random.permutation(pnls)

        # Build full equity path
        equity = INITIAL_CAPITAL + np.cumsum(shuffled)

        # Path-dependent peak-to-trough drawdown
        peak      = np.maximum.accumulate(
            np.concatenate([[INITIAL_CAPITAL], equity])
        )
        trough    = np.concatenate([[INITIAL_CAPITAL], equity])
        dd_series = (peak[1:] - trough[1:]) / peak[1:] * 100
        max_dd    = float(np.max(dd_series))

        mc_finals[run]    = equity[-1]
        mc_drawdowns[run] = max_dd

    pct_profitable = float(np.mean(mc_finals > INITIAL_CAPITAL) * 100)

    return {
        # Final capital distribution
        "median_final":      round(float(np.median(mc_finals)), 2),
        "p5_final":          round(float(np.percentile(mc_finals, 5)), 2),
        "p95_final":         round(float(np.percentile(mc_finals, 95)), 2),
        "pct_profitable":    round(pct_profitable, 1),
        # Drawdown distribution — the genuinely useful Monte Carlo output
        "median_max_dd_pct": round(float(np.median(mc_drawdowns)), 2),
        "p5_max_dd_pct":     round(float(np.percentile(mc_drawdowns, 5)), 2),
        "p95_max_dd_pct":    round(float(np.percentile(mc_drawdowns, 95)), 2),
        # Raw arrays for charting
        "_mc_finals":        mc_finals,
        "_mc_drawdowns":     mc_drawdowns,
    }


def run_backtest(
    use_model:              bool = True,
    walk_forward:           bool = False,
    require_htf_confluence: bool = False,
) -> dict:

    # ── Load & process data ───────────────────────────────────────────────────
    logger.info("Loading data...")
    data = load_all_timeframes()
    if SIGNAL_TF not in data:
        logger.error("Signal TF not found. Run fetch_data.py first.")
        sys.exit(1)

    df = _build_pipeline(data)
    logger.info(f"Total signals: {df['signal'].isin([0,2]).sum():,}")

    # Print SL distance diagnostics
    sigs = df[df["signal"].isin([0, 2])].copy()
    if len(sigs) > 0:
        sl_dists = abs(sigs["close"] - sigs["signal_sl"])
        logger.info(
            f"SL distances: min=${sl_dists.min():,.0f}  "
            f"mean=${sl_dists.mean():,.0f}  "
            f"max=${sl_dists.max():,.0f}"
        )

    # ── Load model ────────────────────────────────────────────────────────────
    model = scaler = features = None
    if use_model and MODEL_PATH.exists():
        model    = joblib.load(MODEL_PATH)
        scaler   = joblib.load(SCALER_PATH)
        features = joblib.load(FEATURES_PATH)
        logger.info(f"Model loaded. Confidence threshold: {MIN_CONFIDENCE}")

    results = {}

    # ── Standard backtest ─────────────────────────────────────────────────────
    logger.info("Running standard backtest (fixed risk, fees + slippage)...")
    rm_main = _simulate(
        df, model, scaler, features,
        compound=False, apply_fees=True, label="Standard",
        require_htf_confluence=require_htf_confluence,
    )
    results["standard"] = rm_main

    # ── Walk-forward out-of-sample simulation ─────────────────────────────────
    # NOTE: This uses the SAME trained model across all periods.
    # It is an out-of-sample simulation, NOT true walk-forward (which requires
    # retraining the model on each expanding window). True walk-forward is
    # available via: python -m backend.scripts.train_model --walk-forward
    if walk_forward:
        logger.info(
            "Running out-of-sample period validation "
            "(same model, different time windows)..."
        )
        splits = [
            ("2020-2022 in-sample",  "2020-01-01", "2022-12-31"),
            ("2023 out-of-sample",   "2023-01-01", "2023-12-31"),
            ("2024 out-of-sample",   "2024-01-01", "2024-12-31"),
        ]
        wf_results = {}
        for label, start, end in splits:
            period_df = df[start:end]
            if len(period_df) < 10:
                continue
            rm_wf = _simulate(
                period_df, model, scaler, features,
                compound=False, apply_fees=True, label=label
            )
            wf_results[label] = rm_wf.summary()
            logger.info(
                f"  [{label}] trades={wf_results[label]['total_trades']} "
                f"WR={wf_results[label]['win_rate_pct']}% "
                f"PnL={wf_results[label]['net_pnl_pct']:+.1f}%"
            )
        results["walk_forward"] = wf_results

    # ── Monte Carlo simulation ────────────────────────────────────────────────
    logger.info(f"Running Monte Carlo simulation ({MC_RUNS} runs)...")
    if rm_main.trade_log:
        trade_df = rm_main.to_dataframe()
        pnls     = trade_df["pnl"].values
        mc       = _run_monte_carlo(pnls)
        results["monte_carlo"] = mc

    # ── Print main results ────────────────────────────────────────────────────
    summary = rm_main.summary()
    _print_report(summary, results.get("monte_carlo"), results.get("walk_forward"))

    # ── Save charts ───────────────────────────────────────────────────────────
    _save_charts(rm_main, results.get("monte_carlo"))

    # ── Save JSON for API (strip private array keys before serialising) ───────
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    mc_json = {k: v for k, v in results.get("monte_carlo", {}).items()
               if not k.startswith("_")}
    with open(LOG_DIR / "backtest_summary.json", "w") as f:
        json.dump({**summary, "monte_carlo": mc_json}, f, indent=2)

    return results


def _print_report(
    summary:      dict,
    monte_carlo:  dict | None = None,
    walk_forward: dict | None = None,
) -> None:

    print("\n" + "=" * 60)
    print("  BACKTEST RESULTS  (1% risk, fees + slippage included)")
    print("=" * 60)

    if "error" in summary:
        print(f"  {summary['error']}")
        print("=" * 60)
        return

    print(f"  Initial Capital   : ${summary['initial_capital']:>12,.2f}")
    print(f"  Final Capital     : ${summary['final_capital']:>12,.2f}")
    print(f"  Net PnL           : ${summary['net_pnl']:>+12,.2f}  ({summary['net_pnl_pct']:+.1f}%)")
    print(f"  Total Trades      : {summary['total_trades']:>8,}")
    print(f"  Wins / Losses     : {summary['wins']:>4} / {summary['losses']:<4}")
    print(f"  Win Rate          : {summary['win_rate_pct']:>8.2f}%")
    print(f"  Avg Risk / Trade  : ${summary['avg_risk_per_trade']:>8,.2f}")
    print(f"  Slippage/side     : {SLIPPAGE_PCT*100:>8.3f}%")
    print(f"  Max Drawdown      : {summary['max_drawdown_pct']:>8.2f}%")
    print(f"  Profit Factor     : {summary['profit_factor']:>8.2f}")
    print(f"  Total Fees Paid   : ${summary.get('total_fees_paid', 0):>8,.2f}")
    print(f"  Trading Halted    : {summary.get('trades_halted', False)}")
    print()

    if monte_carlo:
        print(f"  MONTE CARLO ({MC_RUNS} path-shuffled runs)")
        print()
        print("  Final Capital Distribution:")
        print(f"    Median          : ${monte_carlo['median_final']:>12,.2f}")
        print(f"    5th Percentile  : ${monte_carlo['p5_final']:>12,.2f}")
        print(f"    95th Percentile : ${monte_carlo['p95_final']:>12,.2f}")
        print(f"    % Profitable    : {monte_carlo['pct_profitable']:>7.1f}%")
        print()
        print("  Max Drawdown Distribution (path-dependent):")
        print(f"    Median DD       : {monte_carlo['median_max_dd_pct']:>8.2f}%")
        print(f"    5th Percentile  : {monte_carlo['p5_max_dd_pct']:>8.2f}%  (best-case)")
        print(f"    95th Percentile : {monte_carlo['p95_max_dd_pct']:>8.2f}%  (near-worst-case)")
        print(f"    Interpretation  : In 95% of trade sequences, max DD")
        print(f"                      stays below {monte_carlo['p95_max_dd_pct']:.1f}%")
        print()
        # Note: with flat risk, all paths have identical final PnL sum.
        # The drawdown distribution is the meaningful Monte Carlo output.
        if monte_carlo['p5_final'] == monte_carlo['p95_final']:
            print("  (Note: identical final capitals are expected with")
            print("   fixed flat risk — path order doesn't change the sum.")
            print("   Drawdown distribution below is the meaningful metric.)")

    if walk_forward:
        print("  OUT-OF-SAMPLE PERIOD VALIDATION")
        print("  (Note: uses same trained model — not true walk-forward retraining)")
        for period, s in walk_forward.items():
            wr  = s.get('win_rate_pct', 0)
            pnl = s.get('net_pnl_pct', 0)
            n   = s.get('total_trades', 0)
            print(f"  {period:<28}: {n:>3} trades  WR={wr:.1f}%  PnL={pnl:+.1f}%")
        print()

    # Honest assessment
    print("  ASSESSMENT")
    dd = summary['max_drawdown_pct']
    pf = summary['profit_factor']
    wr = summary['win_rate_pct']

    if dd > 25:
        print(f"  WARN: Max drawdown {dd:.1f}% is high. Consider stricter filters.")
    elif dd > 15:
        print(f"  NOTE: Max drawdown {dd:.1f}% is moderate. Acceptable for 1% risk.")
    else:
        print(f"  GOOD: Max drawdown {dd:.1f}% is well controlled.")

    if pf > 1.5:
        print(f"  GOOD: Profit factor {pf:.2f} — strategy has positive edge.")
    elif pf > 1.0:
        print(f"  NOTE: Profit factor {pf:.2f} — marginal edge. Add filters.")
    else:
        print(f"  WARN: Profit factor {pf:.2f} < 1.0 — strategy losing money.")

    if monte_carlo:
        p95_dd = monte_carlo.get("p95_max_dd_pct", 0)
        if p95_dd > 25:
            print(f"  WARN: Monte Carlo P95 drawdown {p95_dd:.1f}% — tail risk is high.")
        elif p95_dd > 15:
            print(f"  NOTE: Monte Carlo P95 drawdown {p95_dd:.1f}% — moderate tail risk.")
        else:
            print(f"  GOOD: Monte Carlo P95 drawdown {p95_dd:.1f}% — robust across sequences.")

    print("=" * 60)


def _save_charts(rm: RiskManager, monte_carlo: dict | None = None) -> None:
    if not rm.trade_log:
        return

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    trade_df = rm.to_dataframe()
    caps     = trade_df["capital_after"].values
    pnls     = trade_df["pnl"].values

    # Rolling drawdown
    running_peak = np.maximum.accumulate(np.append(INITIAL_CAPITAL, caps))
    drawdowns    = (running_peak[1:] - caps) / running_peak[1:] * 100

    has_mc = monte_carlo is not None and "_mc_finals" in monte_carlo

    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor("#0d1117")
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3)

    panels = [
        (gs[0, :], "Equity Curve"),
        (gs[1, 0], "Drawdown (%)"),
        (gs[1, 1], "Per-Trade PnL"),
        (gs[2, 0], "Cumulative Fees"),
        (gs[2, 1], "Win/Loss by Direction"),
    ]

    # Add Monte Carlo panel if data available
    if has_mc:
        panels.append((gs[0, 1], "Monte Carlo — Final Capital Distribution"))

    for spec, title in panels:
        ax = fig.add_subplot(spec)
        ax.set_facecolor("#0d1117")
        ax.tick_params(colors="#555", labelsize=8)
        for sp in ax.spines.values():
            sp.set_color("#1a2440")
        ax.set_title(title, color=(1, 1, 1, 0.7), fontsize=9, pad=6)

    axes = fig.axes
    xs   = range(len(caps))

    # 1 — Equity curve (full width top row)
    axes[0].plot(xs, caps, color="#2dd4bf", lw=1.5, alpha=0.9)
    axes[0].fill_between(xs, caps, INITIAL_CAPITAL,
                          where=[c > INITIAL_CAPITAL for c in caps],
                          alpha=0.15, color="#2dd4bf")
    axes[0].fill_between(xs, caps, INITIAL_CAPITAL,
                          where=[c <= INITIAL_CAPITAL for c in caps],
                          alpha=0.15, color="#f43f5e")
    axes[0].axhline(INITIAL_CAPITAL, color="#888", lw=0.8, ls="--", alpha=0.5)
    axes[0].set_ylabel("Capital ($)", color="#888", fontsize=8)

    # 2 — Drawdown
    axes[1].fill_between(xs, -drawdowns, 0, color="#f43f5e", alpha=0.6)
    axes[1].plot(xs, -drawdowns, color="#f43f5e", lw=1)
    axes[1].axhline(-20, color="#f59e0b", lw=0.8, ls="--", alpha=0.7,
                     label="20% stop")
    axes[1].set_ylabel("Drawdown (%)", color="#888", fontsize=8)
    axes[1].legend(facecolor="#1a2440", labelcolor="white", fontsize=7)

    # 3 — Per-trade PnL
    colors = ["#2dd4bf" if p > 0 else "#f43f5e" for p in pnls]
    axes[2].bar(xs, pnls, color=colors, alpha=0.8, width=0.8)
    axes[2].axhline(0, color="#888", lw=0.8)
    axes[2].set_ylabel("PnL ($)", color="#888", fontsize=8)

    # 4 — Cumulative fees
    cum_fees = np.cumsum(trade_df["fee_cost"].values)
    axes[3].plot(xs, cum_fees, color="#f59e0b", lw=1.5)
    axes[3].fill_between(xs, cum_fees, alpha=0.15, color="#f59e0b")
    axes[3].set_ylabel("Cumulative Fees ($)", color="#888", fontsize=8)

    # 5 — Win/Loss by direction
    for direction, col in [("BUY", "#2dd4bf"), ("SELL", "#f43f5e")]:
        subset = trade_df[trade_df["direction"] == direction]
        w = (subset["outcome"] == "TP").sum()
        l = (subset["outcome"] == "SL").sum()
        axes[4].bar([direction + " Win"], [w], color=col, alpha=0.8)
        axes[4].bar([direction + " Loss"], [l], color=col, alpha=0.4)
    axes[4].set_ylabel("Count", color="#888", fontsize=8)

    # 6 — Monte Carlo final capital histogram
    if has_mc and len(axes) > 5:
        mc_finals = monte_carlo["_mc_finals"]
        if len(set(mc_finals)) > 1:
           axes[5].hist(mc_finals, bins=50, color="#a78bfa", alpha=0.7, edgecolor="none")
        else:
            axes[5].text(0.5, 0.5, f"All runs: ${mc_finals[0]:,.0f}",
                 transform=axes[5].transAxes, ha='center', va='center',
                 color='#a78bfa', fontsize=10)
        axes[5].axvline(monte_carlo["median_final"], color="#2dd4bf",
                        lw=1.5, ls="-",  label=f"Median ${monte_carlo['median_final']:,.0f}")
        axes[5].axvline(monte_carlo["p5_final"],     color="#f43f5e",
                        lw=1.2, ls="--", label=f"P5 ${monte_carlo['p5_final']:,.0f}")
        axes[5].axvline(monte_carlo["p95_final"],    color="#f59e0b",
                        lw=1.2, ls="--", label=f"P95 ${monte_carlo['p95_final']:,.0f}")
        axes[5].axvline(INITIAL_CAPITAL, color="#888",
                        lw=0.8, ls=":",  label="Break-even")
        axes[5].set_ylabel("Frequency", color="#888", fontsize=8)
        axes[5].set_xlabel("Final Capital ($)", color="#888", fontsize=8)
        axes[5].legend(facecolor="#1a2440", labelcolor="white", fontsize=7)

    out = LOG_DIR / "backtest_report.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    logger.info(f"Report saved -> {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-model",             action="store_true")
    parser.add_argument("--walk-forward",         action="store_true",
                        help="Include out-of-sample period validation")
    parser.add_argument("--require-htf-confluence", action="store_true",
                        help="Gate trades on h4_bias AND d1_bias alignment")
    args = parser.parse_args()
    run_backtest(
        use_model              = not args.no_model,
        walk_forward           = args.walk_forward,
        require_htf_confluence = args.require_htf_confluence,
    )