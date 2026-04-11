# backend/services/market_regime.py

import enum
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

from backend.config.settings import (
    REGIME_ADX_TRENDING,
    REGIME_ADX_RANGING,
    REGIME_VOL_HIGH,
    REGIME_VOL_LOW,
    REGIME_EMA_SLOPE_MIN,
    REGIME_LOOKBACK,
)

logger = logging.getLogger(__name__)


class Regime(enum.Enum):
    TRENDING = "TRENDING"
    RANGING = "RANGING"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"


@dataclass
class RegimeResult:
    regime: Regime
    confidence: float
    adx: float
    atr_percentile: float
    vol_percentile: float
    trend_strength: float
    structure_score: float
    reason: str

    @property
    def strategy_type(self) -> str:
        if self.regime == Regime.TRENDING:
            return "trend"
        elif self.regime in (Regime.RANGING, Regime.HIGH_VOLATILITY):
            return "scalp"
        return "none"


class MarketRegimeDetector:

    def classify(self, df: pd.DataFrame, row: Optional[pd.Series] = None) -> RegimeResult:
        if row is None:
            row = df.iloc[-1]

        adx = float(row.get("adx_14", 0))
        atr_pct = float(row.get("atr_percentile", 0.5))
        vol_pct = float(row.get("regime_volatility", 0.5))
        trend_str = float(row.get("trend_strength", 0))

        structure_score = self._compute_structure_score(df)
        ema_slope = self._compute_ema_slope(df)

        # LOW VOL
        if atr_pct < REGIME_VOL_LOW and adx < REGIME_ADX_RANGING:
            return RegimeResult(
                Regime.LOW_VOLATILITY,
                0.9,
                adx, atr_pct, vol_pct,
                trend_str, structure_score,
                "Low volatility market"
            )

        # HIGH VOL
        if atr_pct > REGIME_VOL_HIGH and adx < REGIME_ADX_TRENDING:
            return RegimeResult(
                Regime.HIGH_VOLATILITY,
                0.8,
                adx, atr_pct, vol_pct,
                trend_str, structure_score,
                "High volatility"
            )

        # TRENDING
        if (
            adx > REGIME_ADX_TRENDING and
            abs(trend_str) > 0.4 and
            abs(structure_score) > 0.2 and
            abs(ema_slope) > REGIME_EMA_SLOPE_MIN
        ):
            return RegimeResult(
                Regime.TRENDING,
                0.8,
                adx, atr_pct, vol_pct,
                trend_str, structure_score,
                "Trending market"
            )

        return RegimeResult(
            Regime.RANGING,
            0.6,
            adx, atr_pct, vol_pct,
            trend_str, structure_score,
            "Ranging market"
        )

    def _compute_structure_score(self, df: pd.DataFrame) -> float:
        recent = df.tail(REGIME_LOOKBACK)

        if len(recent) < 5:
            return 0.0

        # SAFE fallback (FIXED)
        if "swing_high" in recent.columns:
            sh = recent[recent["swing_high"].astype(bool)]
        else:
            sh = pd.DataFrame()

        if "swing_low" in recent.columns:
            sl = recent[recent["swing_low"].astype(bool)]
        else:
            sl = pd.DataFrame()

        bullish = bearish = 0

        if len(sh) >= 2 and "swing_high_price" in sh:
            prices = sh["swing_high_price"].dropna().values
            bullish += np.sum(np.diff(prices) > 0)
            bearish += np.sum(np.diff(prices) < 0)

        if len(sl) >= 2 and "swing_low_price" in sl:
            prices = sl["swing_low_price"].dropna().values
            bullish += np.sum(np.diff(prices) > 0)
            bearish += np.sum(np.diff(prices) < 0)

        total = bullish + bearish
        return (bullish - bearish) / total if total > 0 else 0.0

    def _compute_ema_slope(self, df: pd.DataFrame) -> float:
        if len(df) < 10:
            return 0.0

        ema = df["close"].ewm(span=8).mean()

        return (ema.iloc[-1] - ema.iloc[-3]) / max(df["close"].iloc[-1], 1e-8)