# backend/services/label_generator.py

import pandas as pd
import numpy as np
from backend.config.settings import LABEL_FORWARD, LABEL_WIN, LABEL_LOSS


def label_trades(df: pd.DataFrame, timeframe: str = "15m") -> pd.DataFrame:
    """
    Production-grade labeling:
    - PURE outcome-based (TP/SL hit)
    - NO quality influence (critical fix)
    - Forward window simulation
    """

    df = df.copy()
    forward = LABEL_FORWARD.get(timeframe, 30)

    labels = []

    for i in range(len(df)):
        row = df.iloc[i]

        # Only label real signals
        if row.get("signal") not in [0, 2]:
            labels.append(np.nan)
            continue

        entry = row["close"]
        sl = row.get("signal_sl")

        if pd.isna(sl):
            labels.append(np.nan)
            continue

        direction = "BUY" if row["signal"] == 2 else "SELL"
        risk = abs(entry - sl)

        # Safety check
        if risk <= 0:
            labels.append(np.nan)
            continue

        # Define TP based on strategy-neutral RR=1 (ML learns probability, not RR)
        tp = entry + risk if direction == "BUY" else entry - risk

        future = df.iloc[i + 1 : i + 1 + forward]

        outcome = None

        for _, frow in future.iterrows():
            if direction == "BUY":
                if frow["low"] <= sl:
                    outcome = LABEL_LOSS
                    break
                if frow["high"] >= tp:
                    outcome = LABEL_WIN
                    break
            else:
                if frow["high"] >= sl:
                    outcome = LABEL_LOSS
                    break
                if frow["low"] <= tp:
                    outcome = LABEL_WIN
                    break

        labels.append(outcome if outcome is not None else LABEL_LOSS)

    df["ml_label"] = labels
    return df