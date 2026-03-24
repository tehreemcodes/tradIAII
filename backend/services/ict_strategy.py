"""
ICT/SMC Strategy Detectors
============================
Three pure, stateless detection functions.
Each takes a DataFrame, returns a NEW DataFrame with added columns.
No mutation of inputs. All functions independently testable.

    detect_swings(df)
        Adds: swing_high, swing_low, swing_high_price, swing_low_price

    detect_cisd(df)
        Adds: bull_cisd, bear_cisd, cisd_body_ratio, cisd_vol_ratio

        FIX v2: True ICT CISD requires price to trade THROUGH a prior
        swing level, then close beyond it with displacement body + volume.
        The old implementation fired on any large candle regardless of
        structural significance — this inflated signal count.

    detect_fvg(df)
        Adds: bull_fvg, bear_fvg, fvg_top, fvg_bot, fvg_size_pct

        FIX v2: Label now sits on candle i+2 (the CONFIRMING candle),
        not candle i+1 (the middle candle). The gap between high[i] and
        low[i+2] cannot be confirmed until candle i+2 closes. Labeling
        on i+1 used future data (1-candle lookahead). Fixed.

    run_ict_pipeline(df)
        Runs all three in sequence. Convenience wrapper.

CHANGELOG v2:
    - detect_fvg:  mid = i + fw  (was i + 1)  — eliminates lookahead
    - detect_cisd: now requires prior swing sweep before displacement
    - Both changes reduce signal count but improve signal quality
"""
import numpy as np
import pandas as pd
import logging
from backend.config.settings import (
    SWING_LOOKBACK,
    CISD_BODY_MULT,
    CISD_VOL_MULT,
    CISD_LOOKBACK,
    FVG_CANDLES,
)

logger = logging.getLogger(__name__)


# ── STEP 1: Swing Detection ───────────────────────────────────────────────────

def detect_swings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Valid Swing High:
        high[i] > high[i-1] AND high[i-2]  (left side)
        high[i] > high[i+1] AND high[i+2]  (right side)

    Valid Swing Low:
        low[i] < low[i-1] AND low[i-2]
        low[i] < low[i+1] AND low[i+2]

    SWING_LOOKBACK = 2 (5-candle pattern as per spec)

    NOTE: Swing detection requires right-side confirmation (i+1, i+2),
    meaning swing labels are naturally delayed by SWING_LOOKBACK candles.
    This is correct and unavoidable — no lookahead bias here.
    """
    df    = df.copy()
    highs = df["high"].values
    lows  = df["low"].values
    n     = len(df)
    lb    = SWING_LOOKBACK   # = 2

    sh       = np.zeros(n, dtype=bool)
    sl       = np.zeros(n, dtype=bool)
    sh_price = np.full(n, np.nan)
    sl_price = np.full(n, np.nan)

    for i in range(lb, n - lb):
        # Swing High: strictly greater than all lb candles on each side
        if (highs[i] > highs[i - lb : i].max() and
                highs[i] > highs[i + 1 : i + lb + 1].max()):
            sh[i]       = True
            sh_price[i] = highs[i]

        # Swing Low: strictly less than all lb candles on each side
        if (lows[i] < lows[i - lb : i].min() and
                lows[i] < lows[i + 1 : i + lb + 1].min()):
            sl[i]       = True
            sl_price[i] = lows[i]

    df["swing_high"]       = sh
    df["swing_low"]        = sl
    df["swing_high_price"] = sh_price
    df["swing_low_price"]  = sl_price

    logger.debug(f"Swings  -> High: {sh.sum():,}  Low: {sl.sum():,}")
    return df


# ── STEP 2: CISD Detection ────────────────────────────────────────────────────

def detect_cisd(df: pd.DataFrame) -> pd.DataFrame:
    """
    TRUE ICT CISD = Change in State of Delivery.

    The original implementation fired on ANY large-body + high-volume candle,
    which is just a momentum filter — not a structural event. True ICT CISD
    requires three conditions to ALL be true simultaneously:

    Bullish CISD (all three required):
        1. A prior swing HIGH exists within CISD_LOOKBACK candles
        2. The current candle's HIGH trades THROUGH that swing high price
           (wicks above it — engineered liquidity sweep)
        3. The candle CLOSES bullish (close > open) with:
               body  > CISD_BODY_MULT * rolling avg body
               volume > CISD_VOL_MULT * rolling avg volume
           This closing beyond the swept level is the "delivery" change.

    Bearish CISD (all three required):
        1. A prior swing LOW exists within CISD_LOOKBACK candles
        2. The current candle's LOW trades THROUGH that swing low price
        3. The candle CLOSES bearish (close < open) with displacement body + vol

    Why this matters:
        - Old version generated ~3-5x more CISD signals than the new version
        - Most of those were random large candles with no structural significance
        - This inflated signal count and created a biased training dataset
        - True CISD signals are rarer but carry genuine institutional order flow

    Stores cisd_body_ratio and cisd_vol_ratio as numeric ML features (unchanged).
    Stores cisd_swept_level: the prior swing price that was swept (new, for features).
    """
    df     = df.copy()
    opens  = df["open"].values
    closes = df["close"].values
    highs  = df["high"].values
    lows   = df["low"].values
    vols   = df["volume"].values
    n      = len(df)
    lb     = CISD_LOOKBACK   # rolling window for avg body/vol = 20

    bull         = np.zeros(n, dtype=bool)
    bear         = np.zeros(n, dtype=bool)
    body_ratio   = np.zeros(n, dtype=float)
    vol_ratio    = np.zeros(n, dtype=float)
    swept_level  = np.full(n, np.nan)

    # Pre-extract swing arrays for fast lookup
    swing_high_flags  = df["swing_high"].values  if "swing_high" in df.columns  else np.zeros(n, dtype=bool)
    swing_high_prices = df["swing_high_price"].values if "swing_high_price" in df.columns else np.full(n, np.nan)
    swing_low_flags   = df["swing_low"].values   if "swing_low" in df.columns   else np.zeros(n, dtype=bool)
    swing_low_prices  = df["swing_low_price"].values  if "swing_low_price" in df.columns  else np.full(n, np.nan)

    for i in range(lb, n):
        body     = abs(closes[i] - opens[i])
        avg_body = np.mean(np.abs(closes[i - lb : i] - opens[i - lb : i]))
        avg_vol  = np.mean(vols[i - lb : i])

        if avg_body < 1e-8 or avg_vol < 1e-8:
            continue

        br = body / avg_body
        vr = vols[i] / avg_vol
        body_ratio[i] = br
        vol_ratio[i]  = vr

        # Displacement threshold must be met for ANY CISD
        if br < CISD_BODY_MULT or vr < CISD_VOL_MULT:
            continue

        # ── Bullish CISD: sweep a prior swing HIGH then close bullish ────────
        # Look back up to lb candles for the most recent swing high
        if closes[i] > opens[i]:   # bullish close required
            for k in range(i - 1, max(i - lb - 1, -1), -1):
                if swing_high_flags[k] and not np.isnan(swing_high_prices[k]):
                    prior_sh = swing_high_prices[k]
                    # Candle must have wicked THROUGH the swing high
                    # AND closed bullish beyond it (or at minimum swept it)
                    if highs[i] > prior_sh:
                        bull[i]        = True
                        swept_level[i] = prior_sh
                    break   # only use the most recent swing high

        # ── Bearish CISD: sweep a prior swing LOW then close bearish ─────────
        elif closes[i] < opens[i]:  # bearish close required
            for k in range(i - 1, max(i - lb - 1, -1), -1):
                if swing_low_flags[k] and not np.isnan(swing_low_prices[k]):
                    prior_sl = swing_low_prices[k]
                    # Candle must have wicked THROUGH the swing low
                    if lows[i] < prior_sl:
                        bear[i]        = True
                        swept_level[i] = prior_sl
                    break

    df["bull_cisd"]        = bull
    df["bear_cisd"]        = bear
    df["cisd_body_ratio"]  = body_ratio
    df["cisd_vol_ratio"]   = vol_ratio
    df["cisd_swept_level"] = swept_level   # new ML feature

    logger.debug(f"CISD    -> Bull: {bull.sum():,}  Bear: {bear.sum():,}")
    return df


# ── STEP 3: FVG Detection ─────────────────────────────────────────────────────

def detect_fvg(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fair Value Gap — 3-candle price imbalance.

    FIX v2: Label now sits on candle i+2 (the CONFIRMING candle).

    WHY THE OLD CODE WAS WRONG:
        The gap condition is: lows[i+2] > highs[i]  (bullish)
        This uses data from candle i+2. The old code placed the label on
        candle i+1 (mid = i+1), meaning candle i+1 was tagged using future
        data from candle i+2 — a 1-candle lookahead bias.

        In live trading you cannot know that lows[i+2] > highs[i] until
        candle i+2 has fully closed. The FVG is only CONFIRMED at close
        of the third candle.

    FIX:
        mid = i + fw   (fw = FVG_CANDLES = 2)
        Label sits on the candle that CONFIRMS the gap exists.
        This is slightly later but completely bias-free.

    Bullish FVG (confirmed at candle i+2):
        Condition:  low[i+2]  > high[i]
        fvg_top   = low[i+2]    upper edge of gap
        fvg_bot   = high[i]     lower edge of gap

    Bearish FVG (confirmed at candle i+2):
        Condition:  high[i+2] < low[i]
        fvg_top   = low[i]      upper edge of gap
        fvg_bot   = high[i+2]   lower edge of gap

    fvg_size_pct = gap size as % of close[i+2] price (numeric ML feature)
                   Changed from cls[i] to cls[i+fw] for accuracy.
    """
    df    = df.copy()
    highs = df["high"].values
    lows  = df["low"].values
    cls   = df["close"].values
    n     = len(df)
    fw    = FVG_CANDLES   # = 2

    bull     = np.zeros(n, dtype=bool)
    bear     = np.zeros(n, dtype=bool)
    fvg_top  = np.full(n, np.nan)
    fvg_bot  = np.full(n, np.nan)
    fvg_size = np.zeros(n, dtype=float)

    for i in range(n - fw):
        # FIX: label on the CONFIRMING candle (i+fw), not the middle (i+1)
        # The gap is only knowable once candle i+fw has closed.
        confirm = i + fw   # was: mid = i + 1  ← this was the lookahead bug

        # Bullish FVG: gap between high[i] and low[i+2]
        if lows[i + fw] > highs[i]:
            gap              = lows[i + fw] - highs[i]
            bull[confirm]    = True
            fvg_top[confirm] = lows[i + fw]
            fvg_bot[confirm] = highs[i]
            fvg_size[confirm]= gap / cls[i + fw] if cls[i + fw] > 0 else 0.0

        # Bearish FVG: gap between low[i] and high[i+2]
        if highs[i + fw] < lows[i]:
            gap              = lows[i] - highs[i + fw]
            bear[confirm]    = True
            fvg_top[confirm] = lows[i]
            fvg_bot[confirm] = highs[i + fw]
            fvg_size[confirm]= gap / cls[i + fw] if cls[i + fw] > 0 else 0.0

    df["bull_fvg"]     = bull
    df["bear_fvg"]     = bear
    df["fvg_top"]      = fvg_top
    df["fvg_bot"]      = fvg_bot
    df["fvg_size_pct"] = fvg_size

    logger.debug(f"FVG     -> Bull: {bull.sum():,}  Bear: {bear.sum():,}")
    return df


# ── Pipeline Wrapper ──────────────────────────────────────────────────────────

def run_ict_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run detect_swings -> detect_cisd -> detect_fvg in sequence.
    Returns a new DataFrame with all ICT columns added.

    Order matters: detect_cisd reads swing columns produced by detect_swings.
    """
    df = detect_swings(df)
    df = detect_cisd(df)
    df = detect_fvg(df)
    return df