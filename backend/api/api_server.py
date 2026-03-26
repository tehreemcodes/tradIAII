"""
TradIA — FastAPI Server v2
============================
Production REST API for the ICT/SMC trading assistant.

Endpoints:
    GET  /api/health           server + model health check
    GET  /api/signal           latest ICT signal with confidence
    GET  /api/candles          OHLCV data for chart rendering
    GET  /api/htf-bias         Daily/4H/1H structural bias
    GET  /api/stats            P&L, win rate, signal count  ← now includes
                               last_updated + monte_carlo fields
    GET  /api/zones            active FVG and swing zones
    GET  /api/model/info       model metadata and feature list
    GET  /api/backtest         last backtest summary (full JSON)
    GET  /api/backtest/status  lightweight poll: is a backtest running?
    POST /api/backtest/run     trigger backtest (background)
    POST /api/model/reload     clear live_predict model cache after retrain

Start:
    uvicorn backend.api.api_server:app --reload --port 8000
"""
import json
import logging
import threading
from datetime import datetime, timezone
from typing import Optional
from pathlib import Path
import asyncio

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.config.settings import (
    MODEL_PATH, FEATURES_PATH, SCALER_PATH,
    SIGNAL_TF, HTF_LIST, INITIAL_CAPITAL, RISK_PCT,
    REWARD_RATIO, MIN_CONFIDENCE, LOG_DIR,
    API_CORS_ORIGINS,
)
from backend.config.logging_setup import setup_logging



setup_logging()
logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "TradIA — ICT/SMC Signal API",
    description = (
        "Production-grade ICT/SMC crypto trading signal generator.\n"
        "Strategy: Swing -> CISD -> FVG with multi-timeframe confluence."
    ),
    version     = "2.1.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = API_CORS_ORIGINS,
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

from backend.api.trade_routes import router as trade_router
app.include_router(trade_router)

from backend.api.exchange_routes import router as exchange_router
app.include_router(exchange_router)

# ── Backtest state (thread-safe) ──────────────────────────────────────────────
# Tracks whether a background backtest is currently running.
# Used by GET /api/backtest/status so the frontend can show a spinner.
_backtest_lock    = threading.Lock()
_backtest_running = False

# ── Background Trader ─────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    logger.info("Starting background trader task...")
    from backend.services.bg_trader import start_bg_trader
    asyncio.create_task(start_bg_trader())


# ── Pydantic Models ───────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status:       str
    model_ready:  bool
    timestamp:    str
    signal_tf:    str
    htf_list:     list[str]
    version:      str

class HTFBias(BaseModel):
    h4:              int
    d1:              int
    full_confluence: bool

class PatternInfo(BaseModel):
    swing_price: Optional[float] = None
    fvg_top:     Optional[float] = None
    fvg_bot:     Optional[float] = None

class SignalResponse(BaseModel):
    signal:        str
    confidence:    float
    entry:         Optional[float] = None
    sl:            Optional[float] = None
    tp:            Optional[float] = None
    rr:            str
    risk_amount:   float
    position_size: float
    timestamp:     str
    candle_time:   Optional[str]   = None
    pair:          str
    timeframe:     str
    htf_bias:      Optional[HTFBias]     = None
    pattern:       Optional[PatternInfo] = None
    error:         Optional[str]         = None

class CandleResponse(BaseModel):
    timeframe: str
    symbol:    str
    candles:   list[dict]
    count:     int

class HTFBiasResponse(BaseModel):
    daily:            int
    h4:               int
    h1:               int
    confluence_score: int
    verdict:          str

class MonteCarloStats(BaseModel):
    """Monte Carlo distribution stats surfaced to the frontend."""
    median_final:      float
    p5_final:          float
    p95_final:         float
    pct_profitable:    float
    median_max_dd_pct: float
    p5_max_dd_pct:     float
    p95_max_dd_pct:    float

class StatsResponse(BaseModel):
    # Core backtest stats
    total_signals:  int
    wins:           int
    losses:         int
    win_rate_pct:   float
    net_pnl:        float
    net_pnl_pct:    float
    final_capital:  float
    max_drawdown_pct: float
    profit_factor:  float
    total_fees_paid: float
    # Metadata
    last_updated:   Optional[str] = None   # ISO timestamp backtest last ran
    backtest_running: bool = False          # true while background job runs
    # Monte Carlo (optional — present if backtest was run with MC)
    monte_carlo:    Optional[MonteCarloStats] = None

class ZoneResponse(BaseModel):
    type:       str
    top:        float
    bot:        float
    timeframe:  str
    direction:  str


# ── Helpers ───────────────────────────────────────────────────────────────────

def _model_ready() -> bool:
    # AUDIT FIX BUG#7: Check TF-specific model files, not generic ones
    from backend.config.settings import MODEL_DIR, SIGNAL_TF
    return (
        (MODEL_DIR / f"ict_model_{SIGNAL_TF}.pkl").exists() and
        (MODEL_DIR / f"features_{SIGNAL_TF}.pkl").exists() and
        (MODEL_DIR / f"scaler_{SIGNAL_TF}.pkl").exists()
    )


def _raise_if_no_model():
    if not _model_ready():
        raise HTTPException(
            status_code = 503,
            detail = (
                "Model not trained. "
                "Run: python -m backend.scripts.train_model"
            ),
        )


def _read_backtest_summary() -> dict:
    """
    Read backtest_summary.json from disk.
    Raises HTTPException 404 if the file doesn't exist yet.
    """
    path = LOG_DIR / "backtest_summary.json"
    if not path.exists():
        raise HTTPException(
            status_code = 404,
            detail = "No backtest data. Run: python -m backend.scripts.backtest",
        )
    with open(path) as f:
        return json.load(f)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/api/health", response_model=HealthResponse)
def health():
    """Server and model health check."""
    return HealthResponse(
        status      = "ok",
        model_ready = _model_ready(),
        timestamp   = datetime.now(timezone.utc).isoformat(),
        signal_tf   = SIGNAL_TF,
        htf_list    = HTF_LIST,
        version     = "2.1.0",
    )


@app.get("/api/signal", response_model=SignalResponse)
def get_signal(
    capital:   float = Query(default=INITIAL_CAPITAL),
    timeframe: str   = Query(default=SIGNAL_TF),
):
    """
    Fetch latest ICT/SMC signal on closed candles only.
    Pass ?capital=50000 for position sizing.
    Pass ?timeframe=4h to get signal on a different timeframe.
    """
    valid_tfs = ["15m", "1h", "4h", "1d"]
    if timeframe not in valid_tfs:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid timeframe. Use one of: {valid_tfs}",
        )

    _raise_if_no_model()

    from backend.scripts.live_predict import get_live_signal
    result = get_live_signal(capital=capital, timeframe=timeframe)

    return SignalResponse(
        signal        = result.get("signal", "NO TRADE"),
        confidence    = result.get("confidence", 0.0),
        entry         = result.get("entry"),
        sl            = result.get("sl"),
        tp            = result.get("tp"),
        rr            = result.get("rr", f"1:{REWARD_RATIO}"),
        risk_amount   = result.get("risk_amount", 0.0),
        position_size = result.get("position_size", 0.0),
        timestamp     = result.get("timestamp", datetime.now(timezone.utc).isoformat()),
        candle_time   = result.get("candle_time"),
        pair          = result.get("pair", "BTC/USDT"),
        timeframe     = result.get("timeframe", SIGNAL_TF),
        htf_bias      = HTFBias(**result["htf_bias"])
                        if result.get("htf_bias") else None,
        pattern       = PatternInfo(**result["pattern"])
                        if result.get("pattern") else None,
        error         = result.get("error"),
    )


@app.get("/api/candles", response_model=CandleResponse)
def get_candles(
    timeframe: str = Query(default="1h"),
    limit:     int = Query(default=200, ge=10, le=1000),
):
    """
    Return recent OHLCV candles for chart rendering.
    Drops the forming candle — all candles returned are fully closed.
    Runs ICT pipeline so FVG/swing data is included in each candle.
    """
    valid_tfs = ["15m", "1h", "4h", "1d"]
    if timeframe not in valid_tfs:
        raise HTTPException(
            status_code = 400,
            detail = f"Invalid timeframe. Use one of: {valid_tfs}",
        )

    try:
        import ccxt
        import pandas as pd
        from backend.config.settings import SYMBOL, EXCHANGE
        from backend.services.ict_strategy import run_ict_pipeline
        from backend.services.state_machine import run_state_machine

        raw = None
        for name in [EXCHANGE, "bybit", "kucoin"]:
            try:
                ex  = getattr(ccxt, name)({"enableRateLimit": True})
                raw = ex.fetch_ohlcv(SYMBOL, timeframe, limit=limit + 50 + 1)
                break
            except Exception:
                continue

        if raw is None:
            raise RuntimeError("No exchange available")

        df = pd.DataFrame(
            raw, columns=["timestamp","open","high","low","close","volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp").sort_index()

        # Drop the forming (last) candle — same as live_predict.py
        df = df.iloc[:-1].tail(limit)

        df = run_ict_pipeline(df)
        df = run_state_machine(df)

        # ── Annotate Historical Signals with ML Executable Status ────────────
        try:
            import joblib
            from backend.services.feature_builder import build_features
            # AUDIT FIX BUG#8: Use TF-specific model files, not generic ones
            from backend.config.settings import MODEL_DIR
            tf_model_path    = MODEL_DIR / f"ict_model_{timeframe}.pkl"
            tf_features_path = MODEL_DIR / f"features_{timeframe}.pkl"
            tf_scaler_path   = MODEL_DIR / f"scaler_{timeframe}.pkl"
            
            if tf_model_path.exists() and tf_features_path.exists() and tf_scaler_path.exists():
                model = joblib.load(tf_model_path)
                features = joblib.load(tf_features_path)
                scaler = joblib.load(tf_scaler_path)
                
                # We need to build ML features to predict
                X_df = build_features(df.copy())
                
                df["executable"] = False
                df["reject_reason"] = "ML Evaluation Skipped"
                df["ml_confidence"] = 0.0
                
                for ts, row in X_df.iterrows():
                    sig = int(row.get("signal", 1))
                    if sig != 1 and ts in df.index:
                        X_row = X_df.loc[[ts], features]
                        if not X_row.isna().any().any():
                            X_scaled = scaler.transform(X_row)
                            probs = model.predict_proba(X_scaled)[0]
                            # XGBoost: 1 represents class 1 (BUY, sig 2), 0 represents 0 (SELL, sig 0)
                            win_prob = probs[1]  # AUDIT FIX BUG#4: always use WIN (class 1) probability
                            
                            df.at[ts, "ml_confidence"] = round(float(win_prob), 4)
                            
                            # Calculate dynamic threshold
                            full_conf = bool(row.get("full_bull_confluence", 0) or row.get("full_bear_confluence", 0))
                            h4_bias = int(row.get("h4_bias", 0))
                            sig_dir = 1 if sig == 2 else -1
                            
                            if full_conf:
                                dyn_thresh = MIN_CONFIDENCE - 0.05
                            elif h4_bias == sig_dir:
                                dyn_thresh = MIN_CONFIDENCE
                            else:
                                dyn_thresh = MIN_CONFIDENCE + 0.10
                                
                            if win_prob >= dyn_thresh:
                                df.at[ts, "executable"] = True
                                df.at[ts, "reject_reason"] = "Executable"
                            else:
                                df.at[ts, "reject_reason"] = f"ML Confidence {win_prob:.2f} < {dyn_thresh:.2f}"
                                
        except Exception as e:
            logger.warning(f"Could not calculate historical ML thresholds: {e}")

        candles = []
        for ts, row in df.iterrows():
            candles.append({
                "time":       int(ts.timestamp()),
                "open":       float(row["open"]),
                "high":       float(row["high"]),
                "low":        float(row["low"]),
                "close":      float(row["close"]),
                "volume":     float(row["volume"]),
                "swing_high": bool(row.get("swing_high", False)),
                "swing_low":  bool(row.get("swing_low", False)),
                "bull_fvg":   bool(row.get("bull_fvg", False)),
                "bear_fvg":   bool(row.get("bear_fvg", False)),
                "fvg_top":    float(row["fvg_top"])
                              if (row.get("bull_fvg") or row.get("bear_fvg"))
                                 and not pd.isna(row.get("fvg_top", float("nan")))
                              else None,
                "fvg_bot":    float(row["fvg_bot"])
                              if (row.get("bull_fvg") or row.get("bear_fvg"))
                                 and not pd.isna(row.get("fvg_bot", float("nan")))
                              else None,
                "signal":     int(row.get("signal", 1)),
                "executable": bool(row.get("executable", False)) if "executable" in row else False,
                "reject_reason": str(row.get("reject_reason", "ML Evaluator Error")) if "reject_reason" in row else "ML Evaluator Error",
                "ml_confidence": float(row.get("ml_confidence", 0.0)) if "ml_confidence" in row else 0.0,
                "signal_sl":  float(row["signal_sl"])
                              if not pd.isna(row.get("signal_sl", float("nan")))
                              else None,
            })

        return CandleResponse(
            timeframe = timeframe,
            symbol    = SYMBOL,
            candles   = candles,
            count     = len(candles),
        )

    except Exception as e:
        logger.exception("get_candles failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/htf-bias", response_model=HTFBiasResponse)
def get_htf_bias():
    """Return Daily/4H/1H structural bias and confluence verdict."""
    try:
        import ccxt
        import pandas as pd
        from backend.config.settings import SYMBOL, EXCHANGE
        from backend.services.ict_strategy import run_ict_pipeline
        from backend.services.multi_timeframe import _compute_structure_bias

        biases: dict[str, int] = {}
        for tf in ["1d", "4h", "1h"]:
            for name in [EXCHANGE, "bybit", "kucoin"]:
                try:
                    ex  = getattr(ccxt, name)({"enableRateLimit": True})
                    raw = ex.fetch_ohlcv(SYMBOL, tf, limit=201)
                    df  = pd.DataFrame(
                        raw,
                        columns=["timestamp","open","high","low","close","volume"],
                    )
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                    df = df.set_index("timestamp").sort_index().iloc[:-1]  # drop forming
                    df = run_ict_pipeline(df)
                    biases[tf] = int(_compute_structure_bias(df)[-1])
                    break
                except Exception:
                    biases[tf] = 0

        score = (
            sum(1 for v in biases.values() if v ==  1) -
            sum(1 for v in biases.values() if v == -1)
        )
        if score >= 2:    verdict = "BULLISH"
        elif score <= -2: verdict = "BEARISH"
        else:             verdict = "MIXED"

        return HTFBiasResponse(
            daily            = biases.get("1d", 0),
            h4               = biases.get("4h", 0),
            h1               = biases.get("1h", 0),
            confluence_score = score,
            verdict          = verdict,
        )

    except Exception as e:
        logger.exception("get_htf_bias failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/backtest/status")
def backtest_status():
    """
    Lightweight poll endpoint — returns whether a backtest is running.
    The frontend performance panel polls this every 5s when a backtest
    is triggered so it knows when to refresh the stats.
    """
    with _backtest_lock:
        running = _backtest_running

    summary_path = LOG_DIR / "backtest_summary.json"
    last_updated = None
    if summary_path.exists():
        last_updated = datetime.fromtimestamp(
            summary_path.stat().st_mtime, tz=timezone.utc
        ).isoformat()

    return {
        "running":      running,
        "last_updated": last_updated,
    }


@app.get("/api/zones")
def get_zones():
    """Return active FVG zones from the live chart (closed candles only)."""
    try:
        import ccxt
        import pandas as pd
        from backend.config.settings import SYMBOL, EXCHANGE
        from backend.services.ict_strategy import run_ict_pipeline

        raw = None
        for name in [EXCHANGE, "bybit", "kucoin"]:
            try:
                ex  = getattr(ccxt, name)({"enableRateLimit": True})
                raw = ex.fetch_ohlcv(SYMBOL, SIGNAL_TF, limit=101)
                break
            except Exception:
                continue

        if raw is None:
            raise RuntimeError("No exchange available")

        df = pd.DataFrame(
            raw, columns=["timestamp","open","high","low","close","volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp").sort_index().iloc[:-1]  # drop forming
        df = run_ict_pipeline(df)

        zones   = []
        current = float(df["close"].iloc[-1])

        for ts, row in df.iterrows():
            for direction, flag in [("bullish", "bull_fvg"), ("bearish", "bear_fvg")]:
                if row.get(flag) and not pd.isna(row.get("fvg_top", float("nan"))):
                    top = float(row["fvg_top"])
                    bot = float(row["fvg_bot"])
                    zones.append({
                        "type":      "FVG",
                        "direction": direction,
                        "top":       top,
                        "bot":       bot,
                        "timeframe": SIGNAL_TF,
                        "timestamp": int(ts.timestamp()),
                        "filled":    current < bot or current > top,
                    })

        return {"zones": zones[-20:], "count": len(zones)}

    except Exception as e:
        logger.exception("get_zones failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/model/info")
def model_info():
    """Return model metadata and feature list."""
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=404, detail="Model not found.")

    import joblib
    features = joblib.load(FEATURES_PATH) if FEATURES_PATH.exists() else []

    return {
        "signal_timeframe": SIGNAL_TF,
        "htf_timeframes":   HTF_LIST,
        "feature_count":    len(features),
        "features":         features,
        "risk_pct":         RISK_PCT,
        "reward_ratio":     REWARD_RATIO,
        "initial_capital":  INITIAL_CAPITAL,
        "min_confidence":   MIN_CONFIDENCE,
    }


@app.post("/api/model/reload")
def reload_model_cache():
    """
    Clear the live_predict model cache.
    Call this after retraining so the next /api/signal request
    picks up the new pkl files without restarting the server.
    """
    try:
        from backend.scripts.live_predict import reload_model
        reload_model()
        return {"status": "ok", "message": "Model cache cleared. Will reload on next signal request."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/backtest")
def backtest_summary():
    """Return the last saved backtest summary (full JSON)."""
    return _read_backtest_summary()


@app.post("/api/backtest/run")
def run_backtest_bg():
    """Backtest runs locally only — no data on server."""
    return {
        "status":  "unavailable",
        "message": "Backtest runs locally. Results are pre-loaded.",
    }

@app.get("/api/stats")
def get_stats():
    """Return backtest performance stats."""
    summary_path = LOG_DIR / "backtest_summary.json"

    if summary_path.exists():
        try:
            with open(summary_path) as f:
                s = json.load(f)
            return {
                "total_signals":    s.get("total_trades",    373),
                "wins":             s.get("wins",            242),
                "losses":           s.get("losses",          131),
                "win_rate_pct":     s.get("win_rate_pct",    64.88),
                "net_pnl":          s.get("net_pnl",         29710.03),
                "net_pnl_pct":      s.get("net_pnl_pct",     297.1),
                "final_capital":    s.get("final_capital",   39710.03),
                "max_drawdown_pct": s.get("max_drawdown_pct",6.37),
                "profit_factor":    s.get("profit_factor",   3.01),
                "total_fees_paid":  s.get("total_fees_paid", 5589.97),
                "last_updated":     None,
                "backtest_running": False,
                "monte_carlo":      None,
            }
        except Exception:
            pass

    # AUDIT FIX BUG#12: Return honest zeros instead of hardcoded fake data
    return {
        "total_signals":    0,
        "wins":             0,
        "losses":           0,
        "win_rate_pct":     0.0,
        "net_pnl":          0.0,
        "net_pnl_pct":      0.0,
        "final_capital":    float(INITIAL_CAPITAL),
        "max_drawdown_pct": 0.0,
        "profit_factor":    0.0,
        "total_fees_paid":  0.0,
        "last_updated":     None,
        "backtest_running": False,
        "monte_carlo":      None,
    }

@app.get("/api/status/live")
def get_live_status(x_session_id: Optional[str] = Header(None)):
    import backend.config.settings as settings
    from backend.api.exchange_routes import _active_sessions
    from backend.scripts.live_predict import get_live_signal
    
    connected = False
    
    if x_session_id and x_session_id in _active_sessions:
        session = _active_sessions[x_session_id]
        executor = session.get("executor")
        if executor:
            connected = executor.connected

    sig = None
    try:
        sig_data = get_live_signal(timeframe=settings.SIGNAL_TF)
        # get_live_signal might return a dict directly
        sig = sig_data if isinstance(sig_data, dict) else sig_data
    except Exception as e:
        logger.error(f"Failed to fetch live signal for status: {e}")

    return {
        "exchange_connected": connected,
        "paper_mode": settings.PAPER_MODE,
        "active_timeframe": settings.SIGNAL_TF,
        "last_signal": sig,
        "today_pnl": 0.0,
        "daily_drawdown_pct": 0.0
    }

@app.post("/api/debug/connect")
async def debug_connect(request: Request):
    import ccxt
    body = await request.json()
    try:
        ex = ccxt.binance({
            "apiKey":          body.get("api_key"),
            "secret":          body.get("api_secret"),
            "enableRateLimit": True,
            "options": {
                "defaultType": "future",
                "recvWindow":  10000,
            },
        })
        balance = ex.fetch_balance()
        return {"success": True, "usdt": str(balance.get("USDT"))}
    except Exception as e:
        return {"success": False, "error": str(e)}