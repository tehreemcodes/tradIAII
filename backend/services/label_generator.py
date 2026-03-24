"""
Trade Label Generator
======================
Binary labeling for each signal candle.
Candle-by-candle walk-forward simulation — no vectorized lookahead.

For every signal:
  1. Entry = close at signal candle
  2. SL = swing price (below swing low for BUY, above swing high for SELL)
  3. TP = entry +/- (risk * REWARD_RATIO)
  4. Scan forward LABEL_FORWARD candles:
       First TP touch -> WIN  (label = 1)
       First SL touch -> LOSS (label = 0)
       Neither        -> NaN  (excluded from training)

No lookahead bias: each label only uses data after the signal candle.
jab aap aik khaas trend line se upar closing de detay hain to phir app sell nahi detay is liye ke trend line ko cross kar lia hai, 
"""
import numpy as np
import pandas as pd
import logging
from backend.config.settings import (
    LABEL_FORWARD,
    REWARD_RATIO,
    LABEL_WIN,
    LABEL_LOSS,
)

logger = logging.getLogger(__name__)


def label_trades(
    df:              pd.DataFrame,
    rr:              float = REWARD_RATIO,
    forward_candles: int   = 30,
) -> pd.DataFrame:
    """
    Add ml_label column (1=WIN, 0=LOSS, NaN=no outcome).
    Also adds tp_price for reference.

    Parameters:
        df              : DataFrame after run_state_machine()
        rr              : risk:reward ratio (default 2.0 = 1:2)
        forward_candles : max candles to scan forward
    """
    df      = df.copy()
    n       = len(df)
    labels  = np.full(n, np.nan)
    tp_arr  = np.full(n, np.nan)

    # Only process signal candles
    sig_mask = df["signal"].isin([0, 2])
    sig_idxs = np.where(sig_mask.values)[0]
    logger.info(f"Labeling {len(sig_idxs):,} signals...")

    for i in sig_idxs:
        if i >= n - 2:
            continue

        row   = df.iloc[i]
        entry = float(row["close"])
        sl_raw= row.get("signal_sl", np.nan)
        sl    = float(sl_raw) if not pd.isna(sl_raw) else np.nan

        # Skip if SL is missing or invalid
        if np.isnan(sl) or sl <= 0:
            continue

        risk = abs(entry - sl)
        if risk < 1e-8:
            continue

        is_buy = (int(row["signal"]) == 2)

        # Validate SL is on the correct side of entry
        if is_buy  and sl >= entry: continue
        if not is_buy and sl <= entry: continue

        tp = entry + risk * rr if is_buy else entry - risk * rr
        tp_arr[i] = tp

        # Candle-by-candle forward scan
        end     = min(i + forward_candles + 1, n)
        outcome = None

        for j in range(i + 1, end):
            frow = df.iloc[j]
            if is_buy:
                if frow["low"]  <= sl: outcome = LABEL_LOSS; break
                if frow["high"] >= tp: outcome = LABEL_WIN;  break
            else:
                if frow["high"] >= sl: outcome = LABEL_LOSS; break
                if frow["low"]  <= tp: outcome = LABEL_WIN;  break

        if outcome is not None:
            labels[i] = float(outcome)

    df["ml_label"] = labels
    df["tp_price"] = tp_arr

    wins     = int((labels == LABEL_WIN).sum())
    losses   = int((labels == LABEL_LOSS).sum())
    timeouts = int(np.isnan(labels[sig_idxs]).sum())
    total    = wins + losses

    logger.info(
        f"Labels -> WIN: {wins}  LOSS: {losses}  "
        f"Timeout: {timeouts}  "
        f"Win rate: {wins/total*100:.1f}%" if total > 0 else
        f"Labels -> no outcomes labeled"
    )
    return df
