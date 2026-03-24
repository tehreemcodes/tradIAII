"""
Feature Builder
================
Single source of truth for ALL ML features.

Adding a feature:
  1. Add name to FEATURE_COLS list
  2. Compute it in build_features()
  3. Done — training, backtesting, and live prediction all
     read from FEATURE_COLS automatically

Feature categories:
  - ICT pattern flags (binary)
  - ICT numeric context (ratios, distances, sizes)
  - Candle geometry (body, wicks, range)
  - Volume indicators
  - Price context (range position, volatility regime)
  - Session / killzone timing
  - Multi-timeframe bias (4H, Daily)
  - HTF confluence scores
"""
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


# ── Master Feature List ───────────────────────────────────────
# This is the single source of truth. Preserve order — LightGBM
# is order-sensitive when features are passed by position.
FEATURE_COLS: list[str] = [

    # ICT pattern flags
    "swing_high",
    "swing_low",
    "bull_cisd",
    "bear_cisd",
    "bull_fvg",
    "bear_fvg",

    # ICT numeric context
    "cisd_body_ratio",        # body size / avg body (last 20)
    "cisd_vol_ratio",       # volume / avg volume (last 20)
    "cisd_swept_level_pct",   # swept swing level distance from close / close
    "fvg_size_pct",           # FVG gap / close price
    "sl_distance_pct",        # |entry - SL| / close
    "pattern_duration",       # candles from swing to signal
    "fvg_to_atr_ratio",       # FVG size / ATR14 (gap significance)

    # Candle geometry
    "body_pct",               # |close - open| / close
    "candle_range_pct",       # (high - low) / close
    "upper_wick_pct",         # upper wick / close
    "lower_wick_pct",         # lower wick / close
    "is_bullish_candle",      # close > open (1/0)
    "body_to_range_ratio",    # body / range (candle quality)

    # Volume
    "volume_ratio",           # volume / 20-period avg volume
    "volume_zscore",          # (volume - mean) / std

    # Price context
    "close",
    "atr_14",                 # 14-period ATR
    "atr_percentile",         # ATR percentile rank (volatility regime)
    "range_position_20",      # where price sits in 20-bar range [0, 1]
    "dist_swing_high_pct",    # distance from last swing high / close
    "dist_swing_low_pct",     # distance from last swing low / close

    # Session / killzone timing
    "hour",
    "day_of_week",
    "is_london_kz",           # 07:00-10:00 UTC
    "is_ny_kz",               # 12:00-15:00 UTC
    "is_asia_kz",             # 00:00-03:00 UTC
    "is_optimal_window",      # london OR ny open killzone

    # 4H bias (merged from multi_timeframe.py)
    "h4_bias",
    "h4_bull_fvg",
    "h4_bear_fvg",
    "h4_bull_cisd",
    "h4_bear_cisd",
    "h4_fvg_size",
    "h4_cisd_body_ratio",

    # Daily bias
    "d1_bias",
    "d1_bull_fvg",
    "d1_bear_fvg",
    "d1_bull_cisd",
    "d1_bear_cisd",
    "d1_fvg_size",
    "d1_cisd_body_ratio",

    # HTF confluence
    "htf_bull_confluence",
    "htf_bear_confluence",
    "full_bull_confluence",
    "full_bear_confluence",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all feature columns from the processed DataFrame.
    Safe to call after run_ict_pipeline() + run_state_machine()
    + merge_htf_into_ltf().

    Returns a new DataFrame (does not mutate input).
    """
    df = df.copy()
    c  = df["close"].replace(0, np.nan)    # safe denominator

    # ── Candle Geometry ──────────────────────────────────────
    body       = (df["close"] - df["open"]).abs()
    rng        = df["high"] - df["low"]
    upper_wick = df["high"] - df[["open", "close"]].max(axis=1)
    lower_wick = df[["open", "close"]].min(axis=1) - df["low"]

    df["body_pct"]          = body / c
    df["candle_range_pct"]  = rng / c
    df["upper_wick_pct"]    = upper_wick / c
    df["lower_wick_pct"]    = lower_wick / c
    df["is_bullish_candle"] = (df["close"] > df["open"]).astype(int)
    df["body_to_range_ratio"]= body / rng.replace(0, np.nan)

    # ── Volume ───────────────────────────────────────────────
    vol_ma = df["volume"].rolling(20, min_periods=1).mean()
    vol_sd = df["volume"].rolling(20, min_periods=1).std().replace(0, 1)
    df["volume_ratio"]  = df["volume"] / vol_ma.replace(0, 1)
    df["volume_zscore"] = (df["volume"] - vol_ma) / vol_sd

    # ── ATR & Volatility Regime ───────────────────────────────
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"]  - df["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr14 = tr.rolling(14, min_periods=1).mean()
    df["atr_14"] = atr14

    # ATR percentile rank over rolling 100 periods (volatility regime)
    df["atr_percentile"] = (
        atr14.rolling(100, min_periods=10)
             .rank(pct=True)
    )

    # ── SL Distance ──────────────────────────────────────────
    if "signal_sl" in df.columns:
        df["sl_distance_pct"] = (
            (df["close"] - df["signal_sl"]).abs() / c
        ).fillna(0.0)
    else:
        df["sl_distance_pct"] = 0.0

    # ── CISD Swept Level ──────────────────────────────────────────
    if "cisd_swept_level" in df.columns:
        df["cisd_swept_level_pct"] = (
            (df["close"] - df["cisd_swept_level"]).abs() / c
        ).fillna(0.0)
    else:
        df["cisd_swept_level_pct"] = 0.0

    # ── FVG to ATR Ratio ─────────────────────────────────────
    if "fvg_size_pct" in df.columns:
        fvg_abs = df["fvg_size_pct"] * c
        df["fvg_to_atr_ratio"] = (
            fvg_abs / atr14.replace(0, np.nan)
        ).fillna(0.0)
    else:
        df["fvg_to_atr_ratio"] = 0.0

    # ── Range Position ───────────────────────────────────────
    hi20 = df["high"].rolling(20, min_periods=1).max()
    lo20 = df["low"].rolling(20, min_periods=1).min()
    span = (hi20 - lo20).replace(0, np.nan)
    df["range_position_20"] = ((df["close"] - lo20) / span).fillna(0.5)

    # ── Distance from Swing Levels ───────────────────────────
    last_sh = df.get("swing_high_price", pd.Series(np.nan, index=df.index)).ffill()
    last_sl = df.get("swing_low_price",  pd.Series(np.nan, index=df.index)).ffill()
    df["dist_swing_high_pct"] = ((df["close"] - last_sh).abs() / c).fillna(0.0)
    df["dist_swing_low_pct"]  = ((df["close"] - last_sl).abs() / c).fillna(0.0)

    # ── Killzone / Session Timing ────────────────────────────
    if hasattr(df.index, "hour"):
        h = df.index.hour
        df["hour"]             = h
        df["day_of_week"]      = df.index.dayofweek
        df["is_london_kz"]     = ((h >= 7)  & (h < 10)).astype(int)
        df["is_ny_kz"]         = ((h >= 12) & (h < 15)).astype(int)
        df["is_asia_kz"]       = ((h >= 0)  & (h < 3)).astype(int)
        df["is_optimal_window"]= (
            ((h >= 7) & (h < 10)) | ((h >= 12) & (h < 15))
        ).astype(int)
    else:
        for col in ["hour", "day_of_week", "is_london_kz",
                    "is_ny_kz", "is_asia_kz", "is_optimal_window"]:
            df[col] = 0

    # ── Ensure HTF columns exist (filled 0 if MTF not merged) ─
    htf_defaults = [
        "h4_bias", "h4_bull_fvg", "h4_bear_fvg",
        "h4_bull_cisd", "h4_bear_cisd",
        "h4_fvg_size", "h4_cisd_body_ratio",
        "d1_bias", "d1_bull_fvg", "d1_bear_fvg",
        "d1_bull_cisd", "d1_bear_cisd",
        "d1_fvg_size", "d1_cisd_body_ratio",
        "htf_bull_confluence", "htf_bear_confluence",
        "full_bull_confluence", "full_bear_confluence",
    ]
    for col in htf_defaults:
        if col not in df.columns:
            df[col] = 0

    # ── Ensure pattern_duration exists ───────────────────────
    if "pattern_duration" not in df.columns:
        df["pattern_duration"] = np.nan

    return df


def get_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build features then return ONLY the FEATURE_COLS columns.
    Missing columns are filled with 0.
    Feature names are preserved as column names (required for LightGBM).

    This is the function called at training and inference time.
    """
    df        = build_features(df)
    available = [c for c in FEATURE_COLS if c in df.columns]
    missing   = [c for c in FEATURE_COLS if c not in df.columns]

    if missing:
        logger.warning(f"Missing features (filled 0): {missing}")

    X = df[available].copy()
    for col in missing:
        X[col] = 0.0

    # Return with exact column order from FEATURE_COLS
    return X[FEATURE_COLS].fillna(0.0)
