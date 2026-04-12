"""
Live Signal Generator v3 — Dual-Strategy Hybrid
==================================================
Fetches latest CLOSED candles, runs full ICT pipeline, then applies the
StrategyEngine to detect market regime and route to the appropriate
strategy (Scalp 1R or Trend 2R) with ML ensemble gating.

v3 Changes:
    - Integrated MarketRegimeDetector + StrategyEngine
    - Returns strategy_type ("scalp" | "trend") and regime info
    - Falls back to legacy single-model if ensemble not trained
    - Breakeven threshold included in response for live_trader

Returns:
    {
        "signal":         "BUY" | "SELL" | "NO TRADE"
        "strategy_type":  "scalp" | "trend" | "none"
        "regime":         "TRENDING" | "RANGING" | "HIGH_VOLATILITY" | "LOW_VOLATILITY"
        "confidence":     0.0 - 1.0
        "entry":          float
        "sl":             float | None
        "tp":             float | None
        "rr":             "1:1.0" | "1:2.0"
        "risk_amount":    float
        "position_size":  float
        "be_threshold":   float   (R-multiple for breakeven move)
        "risk_pct":       float   (per-trade risk %)
        "timestamp":      ISO string
        "candle_time":    ISO string
        "pair":           "BTC/USDT"
        "timeframe":      "1h"
        "htf_bias":       {...}
        "pattern":        {...}
        "error":          str | None
    }
"""
import ccxt
import pandas as pd
import numpy as np
import joblib
import logging
from datetime import datetime, timezone

from backend.config.settings import (
    SYMBOL, EXCHANGE, SIGNAL_TF, HTF_LIST,
    MODEL_PATH, FEATURES_PATH, SCALER_PATH,
    PATTERN_WINDOW, MIN_CONFIDENCE, REWARD_RATIO,
    INITIAL_CAPITAL, RISK_PCT,
)
from backend.services.ict_strategy    import run_ict_pipeline
from backend.services.state_machine   import run_state_machine
from backend.services.multi_timeframe import merge_htf_into_ltf
from backend.services.feature_builder import build_features, FEATURE_COLS
from backend.services.risk_manager    import RiskManager
from backend.services.strategy_engine import StrategyEngine

logger = logging.getLogger(__name__)


# ── Module-level singletons ───────────────────────────────────────────────────

# Exchange — one connection object reused across all API calls
_exchange = None

# Model artifacts — loaded once on first call, cached until reload_model()
# is explicitly called (e.g. after retraining)
# Now cached per-timeframe, e.g. _models["1h"]
_models   = {}
_scalers  = {}
_features = {}

# Strategy engine — single instance reused across calls
_strategy_engine: StrategyEngine = None


# ── Exchange ──────────────────────────────────────────────────────────────────

def _get_exchange() -> ccxt.Exchange:
    """
    Return the cached exchange connection.
    Tries EXCHANGE from settings first, then falls back to bybit/kucoin.
    Raises RuntimeError if no exchange is reachable.
    """
    global _exchange
    if _exchange is not None:
        return _exchange

    for name in [EXCHANGE, "bybit", "kucoin"]:
        try:
            ex = getattr(ccxt, name)({"enableRateLimit": True})
            ex.load_markets()
            _exchange = ex
            logger.info(f"Exchange connected: {name}")
            return ex
        except Exception as e:
            logger.warning(f"Exchange [{name}] unavailable: {e}")

    raise RuntimeError(
        "No exchange available. Check EXCHANGE setting and network."
    )


def _fetch_closed_candles(timeframe: str, limit: int) -> pd.DataFrame:
    """
    Fetch `limit` CLOSED candles for SYMBOL on the given timeframe.

    FIX: We request limit+1 candles then drop the LAST one.
    The final candle from the exchange is always the currently-forming
    candle — its OHLCV is incomplete and changes tick by tick.
    Dropping it ensures every candle that enters the ICT pipeline
    has a final, settled close price, correct body size, and real volume.

    Raises ValueError if fewer than `limit` closed candles are returned.
    """
    ex  = _get_exchange()

    # Request one extra so after dropping the forming candle we still
    # have `limit` fully closed candles
    raw = ex.fetch_ohlcv(SYMBOL, timeframe, limit=limit + 1)

    df  = pd.DataFrame(
        raw, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()
    df = df[df["close"] > 0].dropna()

    # Drop the last (forming) candle — this is the critical fix
    df = df.iloc[:-1]

    if len(df) < limit // 2:
        raise ValueError(
            f"[{timeframe}] Only {len(df)} closed candles returned "
            f"(expected ~{limit}). Exchange may be rate-limiting."
        )

    logger.debug(
        f"[{timeframe}] Fetched {len(df)} closed candles. "
        f"Last closed: {df.index[-1]}"
    )
    return df


# ── Model artifact cache ──────────────────────────────────────────────────────

def reload_model(timeframe: str = None) -> None:
    """
    Clear the in-memory model cache.
    If timeframe is given, clears only that TF. Otherwise clears all.
    Also clears the StrategyEngine ensemble cache.
    """
    global _models, _scalers, _features, _strategy_engine
    if timeframe:
        _models.pop(timeframe, None)
        _scalers.pop(timeframe, None)
        _features.pop(timeframe, None)
        logger.info(f"Model artifact cache cleared for {timeframe}.")
    else:
        _models.clear()
        _scalers.clear()
        _features.clear()
        logger.info("All model artifact caches cleared.")

    # Also clear strategy engine's ensemble cache
    if _strategy_engine is not None:
        _strategy_engine.clear_model_cache()
        logger.info("StrategyEngine ensemble cache cleared.")


def _get_strategy_engine() -> StrategyEngine:
    """Return the cached StrategyEngine singleton."""
    global _strategy_engine
    if _strategy_engine is None:
        _strategy_engine = StrategyEngine()
    return _strategy_engine


def _load_model_artifacts(timeframe: str) -> tuple:
    """
    Load model, scaler, and feature list from disk for a specific TF.
    Results are cached after the first load.
    Returns (model, scaler, features) tuple.
    Raises FileNotFoundError if pkl files are missing.
    Raises RuntimeError if loading fails.
    """
    global _models, _scalers, _features

    # Return cached versions if already loaded
    if timeframe in _models:
        return _models[timeframe], _scalers[timeframe], _features[timeframe]

    from backend.config.settings import MODEL_DIR
    model_path    = MODEL_DIR / f"ict_model_{timeframe}.pkl"
    scaler_path   = MODEL_DIR / f"scaler_{timeframe}.pkl"
    features_path = MODEL_DIR / f"features_{timeframe}.pkl"

    # Validate all three files exist before attempting any load
    for path, name in [
        (model_path,    f"ict_model_{timeframe}.pkl"),
        (scaler_path,   f"scaler_{timeframe}.pkl"),
        (features_path, f"features_{timeframe}.pkl"),
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"{name} not found at {path}. "
                f"Run: python -m backend.scripts.train_model --timeframe {timeframe}"
            )

    try:
        _models[timeframe]    = joblib.load(model_path)
        _scalers[timeframe]   = joblib.load(scaler_path)
        _features[timeframe]  = joblib.load(features_path)
        logger.info(
            f"Model artifacts loaded for {timeframe}. "
            f"Features: {len(_features[timeframe])}"
        )
    except Exception as e:
        # Reset globals so the next call retries the load
        _models.pop(timeframe, None)
        _scalers.pop(timeframe, None)
        _features.pop(timeframe, None)
        raise RuntimeError(f"Failed to load model artifacts for {timeframe}: {e}") from e

    return _models[timeframe], _scalers[timeframe], _features[timeframe]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_float(val) -> float | None:
    """Convert a value to float, returning None if NaN or unconvertible."""
    try:
        f = float(val)
        return round(f, 2) if not np.isnan(f) else None
    except Exception:
        return None


def _no_trade(reason: str, log_level: str = "warning", regime: str = "UNKNOWN") -> dict:
    """Build a standardised NO TRADE response with an error message."""
    getattr(logger, log_level)(f"NO TRADE: {reason}")
    return {
        "signal":         "NO TRADE",
        "strategy_type":  "none",
        "regime":         regime,
        "confidence":     0.0,
        "entry":          None,
        "sl":             None,
        "tp":             None,
        "rr":             f"1:{REWARD_RATIO}",
        "risk_amount":    0.0,
        "position_size":  0.0,
        "be_threshold":   0.0,
        "risk_pct":       0.0,
        "timestamp":      datetime.now(timezone.utc).isoformat(),
        "candle_time":    None,
        "pair":           SYMBOL,
        "timeframe":      SIGNAL_TF,
        "htf_bias":       {"h4": 0, "d1": 0, "full_confluence": False},
        "pattern":        {"swing_price": None, "fvg_top": None, "fvg_bot": None},
        "error":          reason,
        "executable":     False,
        "reject_reason":  reason,
    }


# ── Main prediction function ──────────────────────────────────────────────────

def get_live_signal(
    capital: float = INITIAL_CAPITAL,
    timeframe: str = SIGNAL_TF,
) -> dict:
    """
    Full live prediction pipeline on closed candles only.

    Parameters:
        capital   : current trading capital for position sizing.
        timeframe : signal timeframe ("15m", "1h", "4h", "1d").
                    HTF bias timeframes are selected dynamically.
    """
    # ── Dynamic HTF list based on signal timeframe ────────────────────────────
    # Each TF uses higher timeframes for structural bias:
    #   15m → 1h + 4h,  1h → 4h + 1d,  4h → 1d,  1d → none
    HTF_BY_SIGNAL_TF = {
        "15m": ["1h", "4h", "1d"],
        "1h":  HTF_LIST,      # default: ["4h", "1d"]
        "4h":  ["1d"],
        "1d":  [],
    }
    htf_list = HTF_BY_SIGNAL_TF.get(timeframe, HTF_LIST)

    # ── Step 1: Load model artifacts (cached after first load) ───────────────
    try:
        model, scaler, features = _load_model_artifacts(timeframe)
    except FileNotFoundError as e:
        return _no_trade(str(e), log_level="error")
    except RuntimeError as e:
        return _no_trade(str(e), log_level="error")

    # ── Step 2: Fetch CLOSED candles ─────────────────────────────────────────
    try:
        logger.info(f"Fetching closed [{timeframe}] candles...")
        df_ltf = _fetch_closed_candles(timeframe, limit=300)

        htf_data = {}
        for tf in htf_list:
            htf_data[tf] = _fetch_closed_candles(tf, limit=200)

    except Exception as e:
        return _no_trade(f"Data fetch failed: {e}", log_level="error")

    # ── Step 3: Full ICT pipeline ─────────────────────────────────────────────
    try:
        df = run_ict_pipeline(df_ltf)
        df = run_state_machine(df)
        df = merge_htf_into_ltf(df, htf_data)
        df = build_features(df)
    except Exception as e:
        return _no_trade(f"Pipeline error: {e}", log_level="error")

    # ── Step 4: Data sufficiency check ───────────────────────────────────────
    min_required = PATTERN_WINDOW + 20   # pattern window + feature warmup
    if len(df) < min_required:
        return _no_trade(
            f"Insufficient closed candles: {len(df)} < {min_required} required"
        )

    # ── Step 5: Scan recent candles for signals ──────────────────────────────
    SIGNAL_SCAN_WINDOW = 10
    recent      = df.iloc[-SIGNAL_SCAN_WINDOW:]
    signal_mask = recent["signal"].isin([0, 2])
    signal_candles = recent[signal_mask]

    is_stale = False
    if len(signal_candles) > 0:
        last       = signal_candles.iloc[-1]
        sig_ts     = signal_candles.index[-1]
        candle_ts  = sig_ts
        raw_signal = int(last["signal"])

        # Strict stale check: signal must come from the most recent closed candle.
        # Any signal older than 1 candle is stale and must not be traded.
        last_closed_ts = df.index[-1]
        if sig_ts < last_closed_ts:
            is_stale = True

        logger.info(
            f"SIGNAL FOUND at {sig_ts}  |  raw_signal={raw_signal}  |  "
            f"signal close={float(last['close']):,.2f}  |  "
            f"current close={float(df.iloc[-1]['close']):,.2f}  |  "
            f"stale={is_stale}"
        )
    else:
        last       = df.iloc[-1]
        sig_ts     = df.index[-1]
        candle_ts  = df.index[-1]
        raw_signal = 1
        logger.info(
            f"No signal in last {SIGNAL_SCAN_WINDOW} candles  |  "
            f"last candle: {candle_ts}  |  "
            f"close={float(last['close']):,.2f}"
        )

    # ── Step 6: Strategy Engine — Regime + Strategy + ML Gate ─────────────────
    engine = _get_strategy_engine()

    # Get regime classification (always, even for NO TRADE)
    regime_result = engine.regime_detector.classify(df, last)
    regime_str = regime_result.regime.value

    if raw_signal == 1:  # No ICT pattern
        return _no_trade("No ICT pattern in scan window", regime=regime_str)

    # Run full strategy engine evaluation
    strat_signal = engine.evaluate(df, last, timeframe=timeframe)

    signal          = strat_signal.signal
    strategy_type   = strat_signal.strategy_type
    win_prob        = strat_signal.confidence
    executable      = strat_signal.executable
    reject_reason   = strat_signal.reason if not executable else None
    rr_used         = strat_signal.rr
    risk_pct        = strat_signal.risk_pct
    be_threshold    = strat_signal.be_threshold

    logger.info(
        f"Strategy: {strategy_type} | Regime: {regime_str} | "
        f"Signal: {signal} | Confidence: {win_prob:.4f} | "
        f"Executable: {executable}"
    )

    # Hard block stale signals
    if executable and is_stale:
        executable = False
        reject_reason = "Signal is stale (not from current candle close)"

    # ── Step 7: Legacy model inference (backward compat) ─────────────────────
    # Also run the legacy single model for comparison logging
    legacy_confidence = 0.0
    try:
        avail    = [f for f in features if f in df.columns]
        x_raw    = df.loc[[sig_ts], avail].fillna(0)
        x_scaled = scaler.transform(x_raw)          # pass numpy array -- avoids XGBoost DataFrame bug
        proba    = model.predict_proba(x_scaled)[0]
        legacy_confidence = float(proba[1])
        logger.debug(f"Legacy model confidence: {legacy_confidence:.4f}")
    except Exception as e:
        logger.debug(f"Legacy model inference skipped: {e}")

    # Use ensemble confidence if available, else fall back to legacy
    if win_prob == 0.0 and legacy_confidence > 0:
        win_prob = legacy_confidence

    # ── Step 8: Position sizing ───────────────────────────────────────────────
    entry         = float(df.iloc[-1]["close"])
    sl            = strat_signal.sl
    tp            = strat_signal.tp
    risk_amount   = 0.0
    position_size = 0.0

    if signal != "NO TRADE" and sl is not None:
        rm    = RiskManager(initial_capital=capital, compound=True)
        trade = rm.calculate_position(
            entry=entry,
            sl=sl,
            direction=signal,
            ts=candle_ts,
            strategy_type=strategy_type,
            rr_override=rr_used,
            risk_pct_override=risk_pct,
            be_threshold=be_threshold,
        )
        if trade:
            risk_amount   = trade.risk_amount
            position_size = trade.position_size
            tp            = trade.tp  # use RM-calculated TP
        else:
            reject_reason = "Risk rejected (SL too tight or invalid)"
            logger.warning(f"RiskManager rejected: entry={entry}, sl={sl}")
            executable = False

    # ── Step 9: Build response ────────────────────────────────────────────────
    return {
        "signal":         signal,
        "strategy_type":  strategy_type,
        "regime":         regime_str,
        "executable":     executable,
        "reject_reason":  reject_reason,
        "confidence":     round(win_prob, 4),
        "entry":          round(entry, 2),
        "sl":             round(sl, 2) if sl is not None else None,
        "tp":             round(tp, 2) if tp is not None else None,
        "rr":             f"1:{rr_used}",
        "risk_amount":    round(risk_amount, 2),
        "position_size":  round(position_size, 6),
        "be_threshold":   be_threshold,
        "risk_pct":       risk_pct,
        "timestamp":      datetime.now(timezone.utc).isoformat(),
        "candle_time":    candle_ts.isoformat(),
        "pair":           SYMBOL,
        "timeframe":      timeframe,
        "htf_bias": {
            "h4":              int(last.get("h4_bias", 0)),
            "d1":              int(last.get("d1_bias", 0)),
            "full_confluence": bool(
                last.get("full_bull_confluence", 0) or
                last.get("full_bear_confluence", 0)
            ),
        },
        "pattern": {
            "swing_price": _safe_float(last.get("signal_swing_price")),
            "fvg_top":     _safe_float(last.get("signal_fvg_top")),
            "fvg_bot":     _safe_float(last.get("signal_fvg_bot")),
        },
        "error": None,
    }


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    from backend.config.logging_setup import setup_logging
    setup_logging()
    result = get_live_signal()
    print(json.dumps(result, indent=2, default=str))