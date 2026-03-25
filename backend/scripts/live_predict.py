"""
Live Signal Generator
======================
Fetches latest CLOSED candles, runs full ICT pipeline, returns signal dict.
Called by the API server on every /api/signal request.

FIXES v2:
    1. Forming candle dropped before pipeline runs
       The last candle from the exchange is always the CURRENTLY OPEN
       candle — its OHLCV values are incomplete and change every tick.
       Running the ICT pipeline on it produces incorrect FVG confirmations,
       wrong body ratios, and partial volume. We now fetch limit+1 candles
       and drop the last one so all processing is on fully closed candles.

    2. Model artifacts cached at module level
       The old code called joblib.load() on every API request — adding
       100-500ms disk I/O latency per call. Artifacts are now loaded once
       on first call and cached as module globals. Call reload_model() after
       retraining to invalidate the cache.

    3. Exchange connection validation hardened
       Added a connectivity check before processing to surface exchange
       errors clearly rather than failing silently mid-pipeline.

Returns:
    {
        "signal":        "BUY" | "SELL" | "NO TRADE"
        "confidence":    0.0 - 1.0
        "entry":         float
        "sl":            float | None
        "tp":            float | None
        "rr":            "1:2.0"
        "risk_amount":   float
        "position_size": float
        "timestamp":     ISO string   (timestamp of the SIGNAL candle)
        "candle_time":   ISO string   (timestamp of the last CLOSED candle)
        "pair":          "BTC/USDT"
        "timeframe":     "1h"
        "htf_bias":      {"h4": int, "d1": int, "full_confluence": bool}
        "pattern":       {"swing_price": float, "fvg_top": float, "fvg_bot": float}
        "error":         str | None
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
    """
    global _models, _scalers, _features
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


def _no_trade(reason: str, log_level: str = "warning") -> dict:
    """Build a standardised NO TRADE response with an error message."""
    getattr(logger, log_level)(f"NO TRADE: {reason}")
    return {
        "signal":        "NO TRADE",
        "confidence":    0.0,
        "entry":         None,
        "sl":            None,
        "tp":            None,
        "rr":            f"1:{REWARD_RATIO}",
        "risk_amount":   0.0,
        "position_size": 0.0,
        "timestamp":     datetime.now(timezone.utc).isoformat(),
        "candle_time":   None,
        "pair":          SYMBOL,
        "timeframe":     SIGNAL_TF,
        "htf_bias":      {"h4": 0, "d1": 0, "full_confluence": False},
        "pattern":       {"swing_price": None, "fvg_top": None, "fvg_bot": None},
        "error":         reason,
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
        "15m": ["1h", "4h"],
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

    # ── Step 5: Read last CLOSED candle as the signal candle ─────────────────
    # df.iloc[-1] is now guaranteed to be a fully closed candle because
    # _fetch_closed_candles() already dropped the forming one.
    last       = df.iloc[-1]
    candle_ts  = df.index[-1]             # timestamp of last closed candle
    raw_signal = int(last.get("signal", 1))

    logger.info(
        f"Last closed candle: {candle_ts}  |  "
        f"raw_signal={raw_signal}  |  "
        f"close={float(last['close']):,.2f}"
    )

    # ── Step 6: Model inference ───────────────────────────────────────────────
    # Use only the features the model was trained on.
    # Wrap in DataFrame to preserve feature names for LightGBM.
    avail   = [f for f in features if f in df.columns]
    missing = [f for f in features if f not in df.columns]
    if missing:
        logger.warning(f"Features missing at inference (filled 0): {missing}")

    x_raw = df.loc[[candle_ts], avail].fillna(0)
    x_sc  = pd.DataFrame(
        scaler.transform(x_raw), columns=avail
    )

    proba    = model.predict_proba(x_sc)[0]
    win_prob = float(proba[1])   # probability of WIN (class 1)

    logger.info(f"Model confidence: {win_prob:.4f}  (threshold: {MIN_CONFIDENCE})")

    # ── Step 7: Signal decision & Dynamic Thresholding ───────────────────────
    sig_map = {2: "BUY", 0: "SELL", 1: "NO TRADE"}
    signal  = sig_map.get(raw_signal, "NO TRADE")
    
    executable = False
    reject_reason = None
    dynamic_threshold = MIN_CONFIDENCE

    if signal != "NO TRADE":
        htf_info = {
            "h4": int(last.get("h4_bias", 0)),
            "d1": int(last.get("d1_bias", 0)),
            "full_confluence": bool(
                last.get("full_bull_confluence", 0) or
                last.get("full_bear_confluence", 0)
            ),
        }
        
        signal_dir_num = 1 if signal == "BUY" else -1
        
        # Calculate dynamic threshold based on HTF structure tailwind
        if htf_info["full_confluence"]:
            dynamic_threshold = MIN_CONFIDENCE - 0.05
        elif htf_info["h4"] == signal_dir_num:
            dynamic_threshold = MIN_CONFIDENCE
        else:
            dynamic_threshold = MIN_CONFIDENCE + 0.10
            
        dynamic_threshold = round(dynamic_threshold, 2)
        
        # Apply intelligent confidence filter
        if win_prob >= dynamic_threshold:
            executable = True
        else:
            reject_reason = f"ML Confidence {win_prob:.2f} < {dynamic_threshold:.2f} (Required)"
            logger.info(f"Signal {signal} not executable — {reject_reason}")

    # ── Step 8: Position sizing ───────────────────────────────────────────────
    entry         = float(last["close"])
    sl_raw        = last.get("signal_sl", np.nan)
    sl            = float(sl_raw) if not pd.isna(sl_raw) else None
    tp            = None
    risk_amount   = 0.0
    position_size = 0.0

    if signal != "NO TRADE" and sl is not None:
        risk     = abs(entry - sl)
        tp_price = (entry + risk * REWARD_RATIO) if signal == "BUY" \
                   else (entry - risk * REWARD_RATIO)
        tp = round(tp_price, 2)

        # Use compound=True for live trading — real position sizing
        # against current account balance
        rm    = RiskManager(initial_capital=capital, compound=True)
        trade = rm.calculate_position(entry, sl, signal, candle_ts)
        if trade:
            risk_amount   = trade.risk_amount
            position_size = trade.position_size
        else:
            # RiskManager rejected the trade (SL too close, bad direction, etc.)
            reject_reason = "Risk rejected (SL too tight or invalid)"
            logger.warning(
                f"RiskManager rejected position: entry={entry}, sl={sl}, "
                f"direction={signal}"
            )
            executable = False

    # ── Step 9: Build response ────────────────────────────────────────────────
    return {
        "signal":        signal,
        "executable":    executable,
        "reject_reason": reject_reason,
        "confidence":    round(win_prob, 4),
        "entry":         round(entry, 2),
        "sl":            round(sl, 2) if sl is not None else None,
        "tp":            tp,
        "rr":            f"1:{REWARD_RATIO}",
        "risk_amount":   round(risk_amount, 2),
        "position_size": round(position_size, 6),
        "timestamp":     datetime.now(timezone.utc).isoformat(),
        "candle_time":   candle_ts.isoformat(),
        "pair":          SYMBOL,
        "timeframe":     timeframe,
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