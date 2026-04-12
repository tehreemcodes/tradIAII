"""
Multi-Timeframe (MTF) Bias Merger
===================================
Runs the full ICT pipeline independently on each higher timeframe.
Merges the resulting bias columns into the signal timeframe (1H)
using pd.merge_asof with direction='backward' — zero lookahead bias.

Architecture:
    1. Run full ICT pipeline on 4H -> compute structure bias
    2. Run full ICT pipeline on Daily -> compute structure bias
    3. Forward-fill HTF values onto 1H index (backward merge)
    4. Compute confluence scores

Structure bias definition:
    +1 = bullish (last swing was a swing LOW  -> bullish higher low)
    -1 = bearish (last swing was a swing HIGH -> bearish lower high)
     0 = unclear / no swings detected yet
"""
import numpy as np
import pandas as pd
import logging
from backend.services.ict_strategy import run_ict_pipeline
from backend.services.state_machine import run_state_machine
from backend.config.settings import TF_PREFIX

logger = logging.getLogger(__name__)


def _compute_structure_bias(df: pd.DataFrame) -> np.ndarray:
    """
    Derives rolling structure bias from swing detections.
    Carries the last known bias forward (forward-fill).
    Never looks ahead — bias[i] only uses swings at or before candle i.
    """
    n    = len(df)
    bias = np.zeros(n, dtype=float)

    for i in range(1, n):
        if bool(df["swing_low"].iloc[i]):
            bias[i] = 1.0      # bullish: higher low detected
        elif bool(df["swing_high"].iloc[i]):
            bias[i] = -1.0     # bearish: lower high detected
        else:
            bias[i] = bias[i - 1]   # carry forward last known bias

    return bias


def _build_htf_frame(df_htf: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    Run full ICT pipeline on a higher timeframe DataFrame.
    Return a slim DataFrame with prefixed columns for merging into LTF.

    Columns produced:
        {prefix}_bias         structural bias (+1/-1/0)
        {prefix}_bull_fvg     bool
        {prefix}_bear_fvg     bool
        {prefix}_bull_cisd    bool
        {prefix}_bear_cisd    bool
        {prefix}_signal       0/1/2
        {prefix}_fvg_size     float
        {prefix}_cisd_body_ratio float
    """
    df = run_ict_pipeline(df_htf.copy())
    df = run_state_machine(df)

    out = pd.DataFrame(index=df.index)
    out[f"{prefix}_bias"]            = _compute_structure_bias(df)
    out[f"{prefix}_bull_fvg"]        = df["bull_fvg"].astype(int)
    out[f"{prefix}_bear_fvg"]        = df["bear_fvg"].astype(int)
    out[f"{prefix}_bull_cisd"]       = df["bull_cisd"].astype(int)
    out[f"{prefix}_bear_cisd"]       = df["bear_cisd"].astype(int)
    out[f"{prefix}_signal"]          = df["signal"].astype(int)
    out[f"{prefix}_fvg_size"]        = df["fvg_size_pct"].fillna(0.0)
    out[f"{prefix}_cisd_body_ratio"] = df["cisd_body_ratio"].fillna(0.0)

    return out


def merge_htf_into_ltf(
    df_ltf:   pd.DataFrame,
    htf_data: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Merge higher-timeframe bias frames into the LTF signal DataFrame.

    Uses pd.merge_asof with direction='backward' — each 1H candle
    receives the LAST KNOWN HTF value at or before its timestamp.
    This is the only correct approach: no lookahead bias possible.

    Parameters:
        df_ltf   : 1H signal DataFrame (already through ICT pipeline)
        htf_data : {"4h": df_4h, "1d": df_1d}

    Returns:
        df_ltf with HTF columns appended + confluence scores
    """
    df = df_ltf.sort_index().copy()
    merged_prefixes: list[str] = []

    for tf, df_htf in htf_data.items():
        prefix = TF_PREFIX.get(tf, tf)
        logger.info(f"Merging [{tf}] bias (prefix='{prefix}')...")

        htf_frame = _build_htf_frame(df_htf, prefix).sort_index()

        df = pd.merge_asof(
            df,
            htf_frame,
            left_index  = True,
            right_index = True,
            direction   = "backward",   # CRITICAL: no lookahead
        )
        merged_prefixes.append(prefix)

    # ── Confluence Scores ────────────────────────────────────
    bias_cols = [f"{p}_bias" for p in merged_prefixes
                 if f"{p}_bias" in df.columns]
    n_htf     = len(bias_cols)

    if n_htf > 0:
        df["htf_bull_confluence"] = sum(
            (df[c] == 1).astype(int) for c in bias_cols
        )
        df["htf_bear_confluence"] = sum(
            (df[c] == -1).astype(int) for c in bias_cols
        )
        df["full_bull_confluence"] = (
            df["htf_bull_confluence"] == n_htf
        ).astype(int)
        df["full_bear_confluence"] = (
            df["htf_bear_confluence"] == n_htf
        ).astype(int)
    else:
        for col in ["htf_bull_confluence", "htf_bear_confluence",
                    "full_bull_confluence", "full_bear_confluence"]:
            df[col] = 0

    logger.info(
        f"MTF merge complete -> "
        f"Full bull: {df['full_bull_confluence'].sum():,}  "
        f"Full bear: {df['full_bear_confluence'].sum():,}"
    )

    # ── d1_bias diagnostic: confirm real values were merged ───────────────────
    for col in ["d1_bias", "h4_bias"]:
        if col in df.columns:
            s = df[col]
            non_zero = int((s != 0).sum())
            logger.info(
                f"[MTF-diag] {col}: "
                f"non-zero={non_zero:,} ({non_zero/max(len(df),1)*100:.1f}%)  "
                f"mean={s.mean():.3f}  "
                f"+1={(s == 1).sum():,}  "
                f"-1={(s == -1).sum():,}  "
                f"0={(s == 0).sum():,}"
            )
        else:
            logger.warning(f"[MTF-diag] {col} NOT present after merge!")

    return df
