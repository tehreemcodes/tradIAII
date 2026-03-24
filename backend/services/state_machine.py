"""
ICT Pattern State Machine
==========================
Enforces STRICT sequential detection:

    STATE 1 -> Swing detected     (spawn PatternState)
    STATE 2 -> CISD confirmed     (within PATTERN_WINDOW candles)
    STATE 3 -> FVG formed         (after CISD, within window)
    COMPLETE -> Signal emitted

Rules:
    - Multiple patterns tracked simultaneously (one per swing)
    - Each expires after PATTERN_WINDOW candles from swing detection
    - Patterns can only advance states, never go backwards
    - Expired and completed patterns pruned every candle (no memory leak)
    - Signal direction must match swing direction throughout
"""
import numpy as np
import pandas as pd
import logging
from dataclasses import dataclass, field
from backend.config.settings import PATTERN_WINDOW

logger = logging.getLogger(__name__)


@dataclass
class PatternState:
    """Tracks one active Swing -> CISD -> FVG attempt."""
    direction:    str          # "BULL" | "BEAR"
    swing_index:  int
    swing_price:  float
    window:       int = field(default=PATTERN_WINDOW)

    # State flags
    cisd_found:   bool  = False
    cisd_index:   int   = -1

    fvg_found:    bool  = False
    fvg_index:    int   = -1
    fvg_top:      float = np.nan
    fvg_bot:      float = np.nan

    complete:     bool  = False

    def is_expired(self, current_idx: int) -> bool:
        return (current_idx - self.swing_index) > self.window

    def tick(self, i: int, row: pd.Series) -> None:
        """
        Advance state machine by one candle.
        Must be called for every candle AFTER the pattern spawns.
        """
        if self.complete or self.is_expired(i):
            return

        # ── STATE 2: wait for matching CISD ─────────────────
        if not self.cisd_found:
            col = "bear_cisd" if self.direction == "BEAR" else "bull_cisd"
            if bool(row.get(col, False)):
                self.cisd_found = True
                self.cisd_index = i
            # Do NOT fall through — FVG must come after CISD
            return

        # ── STATE 3: wait for matching FVG after CISD ───────
        col = "bear_fvg" if self.direction == "BEAR" else "bull_fvg"
        if bool(row.get(col, False)):
            self.fvg_found = True
            self.fvg_index = i
            self.fvg_top   = float(row.get("fvg_top", np.nan))
            self.fvg_bot   = float(row.get("fvg_bot", np.nan))
            self.complete  = True


def run_state_machine(
    df:     pd.DataFrame,
    window: int = PATTERN_WINDOW,
) -> pd.DataFrame:
    """
    Walk every candle. Spawn a PatternState on each swing detection.
    Emit signal when a pattern reaches COMPLETE.

    Output columns added to df:
        signal              int   2=BUY  0=SELL  1=NO TRADE
        signal_sl           float stop-loss price (swing level)
        signal_fvg_top      float FVG upper edge
        signal_fvg_bot      float FVG lower edge
        signal_swing_price  float the originating swing price
        pattern_duration    float candles from swing -> signal
    """
    df  = df.copy()
    n   = len(df)

    signal        = np.ones(n, dtype=int)
    signal_sl     = np.full(n, np.nan)
    fvg_top_out   = np.full(n, np.nan)
    fvg_bot_out   = np.full(n, np.nan)
    swing_px_out  = np.full(n, np.nan)
    duration_out  = np.full(n, np.nan)

    active: list[PatternState] = []

    for i in range(n):
        row = df.iloc[i]

        # Spawn new pattern on swing detection
        if bool(row.get("swing_high", False)):
            active.append(PatternState(
                direction   = "BEAR",
                swing_index = i,
                swing_price = float(row["swing_high_price"]),
                window      = window,
            ))

        if bool(row.get("swing_low", False)):
            active.append(PatternState(
                direction   = "BULL",
                swing_index = i,
                swing_price = float(row["swing_low_price"]),
                window      = window,
            ))

        # Tick all active patterns forward
        for p in active:
            p.tick(i, row)

        # Collect newly completed signals
        for p in active:
            if p.complete and p.fvg_index == i:
                signal[i]       = 2 if p.direction == "BULL" else 0
                signal_sl[i]    = p.swing_price
                fvg_top_out[i]  = p.fvg_top
                fvg_bot_out[i]  = p.fvg_bot
                swing_px_out[i] = p.swing_price
                duration_out[i] = float(i - p.swing_index)

        # Prune expired and completed patterns (prevent memory growth)
        active = [
            p for p in active
            if not p.complete and not p.is_expired(i)
        ]

    df["signal"]             = signal
    df["signal_sl"]          = signal_sl
    df["signal_fvg_top"]     = fvg_top_out
    df["signal_fvg_bot"]     = fvg_bot_out
    df["signal_swing_price"] = swing_px_out
    df["pattern_duration"]   = duration_out

    buys  = (signal == 2).sum()
    sells = (signal == 0).sum()
    logger.info(
        f"State machine -> BUY: {buys:,}  SELL: {sells:,}  "
        f"Total: {buys + sells:,}"
    )
    return df
