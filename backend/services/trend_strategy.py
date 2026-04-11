"""
Trend Strategy — 2R Continuation System
==========================================
Captures continuation moves in strong trending markets.

Entry logic:
    1. ICT pattern complete (Swing → CISD → FVG from state_machine)
    2. HTF trend confirmation (4H or Daily bias aligned)
    3. Pullback to FVG / order block / 50% retracement
    4. Quality score ≥ TREND_QUALITY_MIN (0.55)
    5. ML ensemble confidence above threshold

Execution:
    R:R = 2.0 (wider TP for continuation)
    SL  = below swing low (wider — structural)
    TP  = 2R from entry

Trade management:
    At 1R unrealized → move SL to breakeven (handled by live_trader)
    No partial close — full position rides to 2R TP

Expected performance:
    Win rate: 30–45%
    Trade frequency: Low (selective)
    Holding time: 4–20 candles

Design:
    Stateless evaluator — takes a DataFrame row and returns a StrategySignal.
    Requires HTF confluence to be present in the row (merged via multi_timeframe.py).
"""
import logging
import numpy as np
import pandas as pd
from typing import Optional

from backend.config.settings import (
    TREND_RR,
    TREND_RISK_PCT,
    TREND_QUALITY_MIN,
    TREND_BE_THRESHOLD,
    MIN_CONFIDENCE,
)
from backend.services.scalp_strategy import StrategySignal

logger = logging.getLogger(__name__)


class TrendStrategy:
    """
    2R continuation-based trend strategy.

    Stricter entry conditions than ScalpStrategy:
    - Requires HTF confluence (4H or Daily bias)
    - Checks for pullback to FVG zone or 50% retracement
    - Higher quality threshold
    - Greater per-trade risk (1%)
    - Breakeven at 1R (vs 0.5R for scalp)
    """

    def __init__(self):
        self.rr = TREND_RR
        self.risk_pct = TREND_RISK_PCT
        self.be_threshold = TREND_BE_THRESHOLD
        self.quality_min = TREND_QUALITY_MIN

    def evaluate(
        self,
        df: pd.DataFrame,
        row: pd.Series,
        regime: str = "TRENDING",
    ) -> StrategySignal:
        """
        Evaluate whether to take a trend continuation trade.

        Parameters:
            df     : Full DataFrame with ICT + feature + HTF columns
            row    : The signal candle row
            regime : Current regime (for metadata)

        Returns:
            StrategySignal with trend-specific parameters
        """
        raw_signal = int(row.get("signal", 1))

        if raw_signal not in (0, 2):
            return StrategySignal.no_trade("No ICT pattern", regime)

        direction = "BUY" if raw_signal == 2 else "SELL"
        is_buy = direction == "BUY"

        # ── HTF Confluence Check (MANDATORY for trend trades) ─────
        htf_ok, htf_reason = self._check_htf_confluence(row, is_buy)
        if not htf_ok:
            return StrategySignal.no_trade(
                f"Trend rejected: {htf_reason}", regime,
            )

        # ── Pullback Check ────────────────────────────────────────
        pullback_ok, pullback_reason = self._check_pullback(df, row, is_buy)
        if not pullback_ok:
            return StrategySignal.no_trade(
                f"Trend rejected: {pullback_reason}", regime,
            )

        # ── Quality Score ─────────────────────────────────────────
        quality = self._compute_trend_quality(row, is_buy)

        if quality < self.quality_min:
            return StrategySignal.no_trade(
                f"Trend quality {quality:.2f} < {self.quality_min}",
                regime,
            )

        # ── Entry / SL / TP ───────────────────────────────────────
        entry = float(row["close"])
        sl_raw = row.get("signal_sl", np.nan)
        sl = float(sl_raw) if not pd.isna(sl_raw) else None

        if sl is None:
            return StrategySignal.no_trade("No SL level from swing", regime)

        # Validate SL direction
        if is_buy and sl >= entry:
            return StrategySignal.no_trade("Invalid SL for trend BUY", regime)
        if not is_buy and sl <= entry:
            return StrategySignal.no_trade("Invalid SL for trend SELL", regime)

        risk = abs(entry - sl)
        if risk < entry * 0.001:
            return StrategySignal.no_trade("SL too tight for trend", regime)

        tp = (entry + risk * self.rr) if is_buy else (entry - risk * self.rr)

        return StrategySignal(
            signal=direction,
            strategy_type="trend",
            entry=round(entry, 2),
            sl=round(sl, 2),
            tp=round(tp, 2),
            rr=self.rr,
            risk_pct=self.risk_pct,
            be_threshold=self.be_threshold,
            quality_score=round(quality, 4),
            confidence=0.0,  # filled later by ML ensemble
            regime=regime,
            reason=(
                f"Trend {direction}: quality={quality:.2f}, RR={self.rr}, "
                f"HTF aligned"
            ),
            executable=True,
        )

    # ── Internal checks ──────────────────────────────────────────────

    def _check_htf_confluence(self, row: pd.Series, is_buy: bool) -> tuple[bool, str]:
        """
        Verify higher timeframe structural bias agrees with signal direction.
        Requires at least one of: h4_bias or d1_bias aligned.
        Full confluence (both aligned) is preferred but not required.
        """
        h4_bias = int(row.get("h4_bias", 0))
        d1_bias = int(row.get("d1_bias", 0))
        full_bull = bool(row.get("full_bull_confluence", 0))
        full_bear = bool(row.get("full_bear_confluence", 0))

        if is_buy:
            if full_bull:
                return True, "Full bullish HTF confluence"
            if h4_bias == 1:
                return True, "4H bullish bias"
            if d1_bias == 1:
                return True, "Daily bullish bias"
            return False, f"No bullish HTF (h4={h4_bias}, d1={d1_bias})"
        else:
            if full_bear:
                return True, "Full bearish HTF confluence"
            if h4_bias == -1:
                return True, "4H bearish bias"
            if d1_bias == -1:
                return True, "Daily bearish bias"
            return False, f"No bearish HTF (h4={h4_bias}, d1={d1_bias})"

    def _check_pullback(
        self,
        df: pd.DataFrame,
        row: pd.Series,
        is_buy: bool,
    ) -> tuple[bool, str]:
        """
        Check if price has pulled back to a structural zone before the signal.

        A valid pullback is one of:
        1. Price touched FVG zone (fvg_bot to fvg_top)
        2. Price at 50% retracement of the last swing range
        3. Price at range_position_20 between 0.3–0.7 (mid-range)

        For trend trades, we want entries on pullbacks, not at extremes.
        """
        range_pos = float(row.get("range_position_20", 0.5))
        fvg_top = row.get("fvg_top", np.nan)
        fvg_bot = row.get("fvg_bot", np.nan)
        close = float(row["close"])

        # Check 1: Price in FVG zone
        if not pd.isna(fvg_top) and not pd.isna(fvg_bot):
            fvg_t = float(fvg_top)
            fvg_b = float(fvg_bot)
            if fvg_b <= close <= fvg_t:
                return True, "Price in FVG zone"

        # Check 2: Mid-range position (pullback, not at extreme)
        if is_buy and 0.2 <= range_pos <= 0.55:
            return True, f"Pullback to lower range (pos={range_pos:.2f})"
        if not is_buy and 0.45 <= range_pos <= 0.8:
            return True, f"Pullback to upper range (pos={range_pos:.2f})"

        # Check 3: Bollinger band position indicates pullback
        bb_pos = float(row.get("bb_position", 0.5))
        if is_buy and bb_pos < 0.4:
            return True, f"Pullback towards lower BB (bb={bb_pos:.2f})"
        if not is_buy and bb_pos > 0.6:
            return True, f"Pullback towards upper BB (bb={bb_pos:.2f})"

        return False, f"No pullback detected (range_pos={range_pos:.2f})"

    def _compute_trend_quality(self, row: pd.Series, is_buy: bool) -> float:
        """
        Quality score for trend trades.
        Trend-weighted: emphasizes HTF confluence and trend strength.
        """
        # HTF confluence (0.30 weight — most important for trends)
        if is_buy:
            htf_conf = float(row.get("htf_bull_confluence", 0))
        else:
            htf_conf = float(row.get("htf_bear_confluence", 0))
        full_conf = bool(row.get("full_bull_confluence", 0) or row.get("full_bear_confluence", 0))
        htf_ok = htf_conf >= 1  # at least one HTF agrees
        htf_bonus = float(full_conf)  # bonus if all agree

        # Trend strength (0.25 weight)
        trend_str = float(row.get("trend_strength", 0))
        trend_ok = abs(trend_str) >= 0.5

        # Session timing (0.15 weight — less important for trends)
        is_optimal = float(row.get("is_optimal_window", 0)) == 1.0

        # Volume (0.15 weight)
        vol_ratio = float(row.get("volume_ratio", 0))
        vol_ok = vol_ratio > 1.1

        # ADX strength (0.15 weight — trends need direction)
        adx = float(row.get("adx_14", 0))
        adx_ok = adx > 25

        score = (
            0.30 * (float(htf_ok) * 0.7 + htf_bonus * 0.3)
            + 0.25 * float(trend_ok)
            + 0.15 * float(is_optimal)
            + 0.15 * float(vol_ok)
            + 0.15 * float(adx_ok)
        )
        return round(score, 4)
