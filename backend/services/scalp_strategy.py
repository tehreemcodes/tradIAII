"""
Scalp Strategy — 1R Reaction System
======================================
Captures short-term reaction moves in ranging/volatile markets.

Entry logic:
    1. ICT pattern complete (Swing → CISD → FVG from state_machine)
    2. Quality score ≥ SCALP_QUALITY_MIN (0.45)
    3. Killzone timing (London or NY session)
    4. ML ensemble confidence above threshold

Execution:
    R:R = 1.0 (tight TP)
    SL  = structural (swing level — tight)
    TP  = 1R from entry

Trade management:
    At 0.5R unrealized → move SL to breakeven (handled by live_trader)

Expected performance:
    Win rate: 45–60%
    Trade frequency: High
    Holding time: 1–4 candles

Design:
    Stateless evaluator — takes a DataFrame row and returns a StrategySignal.
    No side effects, no state mutation. Fully testable.
"""
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

from backend.config.settings import (
    SCALP_RR,
    SCALP_RISK_PCT,
    SCALP_QUALITY_MIN,
    SCALP_BE_THRESHOLD,
    MIN_CONFIDENCE,
)

logger = logging.getLogger(__name__)


@dataclass
class StrategySignal:
    """Unified signal output from any strategy."""
    signal: str              # "BUY" | "SELL" | "NO TRADE"
    strategy_type: str       # "scalp" | "trend" | "none"
    entry: float
    sl: Optional[float]
    tp: Optional[float]
    rr: float
    risk_pct: float
    be_threshold: float      # unrealized R at which to move SL to breakeven
    quality_score: float
    confidence: float        # ML confidence
    regime: str              # current regime label
    reason: str              # human-readable explanation
    executable: bool = False

    @classmethod
    def no_trade(cls, reason: str, regime: str = "UNKNOWN") -> "StrategySignal":
        return cls(
            signal="NO TRADE", strategy_type="none",
            entry=0.0, sl=None, tp=None, rr=0.0,
            risk_pct=0.0, be_threshold=0.0,
            quality_score=0.0, confidence=0.0,
            regime=regime, reason=reason, executable=False,
        )


class ScalpStrategy:
    """
    1R reaction-based scalping strategy.

    Uses the existing ICT pipeline signals (Swing → CISD → FVG) but with:
    - Lower quality threshold (0.45 vs 0.55 for trend)
    - Tighter R:R (1:1 vs 1:2 for trend)
    - Lower per-trade risk (0.5% vs 1%)
    - Breakeven at 0.5R (vs 1R for trend)
    """

    def __init__(self):
        self.rr = SCALP_RR
        self.risk_pct = SCALP_RISK_PCT
        self.be_threshold = SCALP_BE_THRESHOLD
        self.quality_min = SCALP_QUALITY_MIN

    def evaluate(
        self,
        df: pd.DataFrame,
        row: pd.Series,
        regime: str = "RANGING",
    ) -> StrategySignal:
        """
        Evaluate whether to take a scalp trade at this candle.

        Parameters:
            df     : Full DataFrame with ICT + feature columns
            row    : The signal candle row
            regime : Current regime (for metadata)

        Returns:
            StrategySignal with scalp-specific parameters
        """
        raw_signal = int(row.get("signal", 1))

        # Only process BUY (2) or SELL (0) signals from state machine
        if raw_signal not in (0, 2):
            return StrategySignal.no_trade("No ICT pattern", regime)

        direction = "BUY" if raw_signal == 2 else "SELL"

        # ── Quality checks ────────────────────────────────────────
        quality = float(row.get("quality_score", 0.0))

        # If quality_score isn't pre-computed, compute a lightweight version
        if quality == 0.0:
            quality = self._compute_scalp_quality(row, direction == "BUY")

        if quality < self.quality_min:
            return StrategySignal.no_trade(
                f"Scalp quality {quality:.2f} < {self.quality_min}",
                regime,
            )

        # ── Entry / SL / TP ───────────────────────────────────────
        entry = float(row["close"])
        sl_raw = row.get("signal_sl", np.nan)
        sl = float(sl_raw) if not pd.isna(sl_raw) else None

        if sl is None:
            return StrategySignal.no_trade("No SL level from swing", regime)

        # Validate SL direction
        if direction == "BUY" and sl >= entry:
            return StrategySignal.no_trade("Invalid SL (above entry for BUY)", regime)
        if direction == "SELL" and sl <= entry:
            return StrategySignal.no_trade("Invalid SL (below entry for SELL)", regime)

        risk = abs(entry - sl)
        if risk < entry * 0.001:  # SL too tight
            return StrategySignal.no_trade("SL too tight for scalp", regime)

        tp = (entry + risk * self.rr) if direction == "BUY" else (entry - risk * self.rr)

        return StrategySignal(
            signal=direction,
            strategy_type="scalp",
            entry=round(entry, 2),
            sl=round(sl, 2),
            tp=round(tp, 2),
            rr=self.rr,
            risk_pct=self.risk_pct,
            be_threshold=self.be_threshold,
            quality_score=round(quality, 4),
            confidence=0.0,  # filled later by ML ensemble
            regime=regime,
            reason=f"Scalp {direction}: quality={quality:.2f}, RR={self.rr}",
            executable=True,
        )

    def _compute_scalp_quality(self, row: pd.Series, is_buy: bool) -> float:
        """
        Lightweight quality score for scalp trades.
        Scalp-weighted: emphasizes session timing and displacement size.
        """
        # Session quality (0.30 weight — scalps love killzones)
        is_optimal = float(row.get("is_optimal_window", 0)) == 1.0

        # Displacement quality (0.25 weight)
        cisd_body = float(row.get("cisd_body_ratio", 0))
        displacement_ok = cisd_body > 1.8

        # Volume confirmation (0.20 weight)
        vol_ratio = float(row.get("volume_ratio", 0))
        vol_ok = vol_ratio > 1.1

        # Volatility — scalps need SOME vol but not too much (0.15 weight)
        atr_pct = float(row.get("atr_percentile", 0))
        atr_ok = 0.25 < atr_pct < 0.85  # sweet spot

        # FVG quality (0.10 weight)
        fvg_atr = float(row.get("fvg_to_atr_ratio", 0))
        fvg_ok = fvg_atr > 0.3

        score = (
            0.30 * float(is_optimal)
            + 0.25 * float(displacement_ok)
            + 0.20 * float(vol_ok)
            + 0.15 * float(atr_ok)
            + 0.10 * float(fvg_ok)
        )
        return round(score, 4)
