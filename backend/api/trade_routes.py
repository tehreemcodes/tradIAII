"""
Trade Routes — Live Trading API Endpoints
==========================================
Mount these routes on the main FastAPI app in api_server.py:

    from backend.api.trade_routes import router as trade_router
    app.include_router(trade_router)

Endpoints:
    GET  /api/trades              open + closed trades + live stats
    GET  /api/trades/open         open positions only
    GET  /api/trades/closed       closed trade history (last 50)
    GET  /api/trades/stats        live performance stats
    POST /api/trades/close-all    emergency close all paper positions
    POST /api/trades/clear        wipe all trades (start fresh demo run)
    GET  /api/trades/exchange     real-time exchange position data
    GET  /api/analytics/summary   aggregated performance metrics
    GET  /api/analytics/trades    detailed audit logs
"""
import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.config.settings import (
    LIVE_TRADING_ENABLED, INITIAL_CAPITAL, EXCHANGE_TESTNET,
)
from backend.services.trade_tracker  import TradeTracker
from backend.services.trade_executor import TradeExecutor

logger = logging.getLogger(__name__)
router = APIRouter()

# ── Module-level singletons ───────────────────────────────────────────────────
# Shared with live_trader.py via import — both use the same JSON file
# so data is always consistent.
_tracker  = TradeTracker()
_executor = TradeExecutor()


# ── Pydantic response models ──────────────────────────────────────────────────

class LiveStats(BaseModel):
    total_trades:     int   = 0
    wins:             int   = 0
    losses:           int   = 0
    win_rate_pct:     float = 0.0
    total_pnl:        float = 0.0
    total_pnl_pct:    float = 0.0
    running_capital:  float = INITIAL_CAPITAL
    profit_factor:    float = 0.0
    max_drawdown_pct: float = 0.0
    last_trade_at:    Optional[str] = None

class TradesResponse(BaseModel):
    open:    list[dict]
    closed:  list[dict]
    stats:   LiveStats
    mode:    str          # "PAPER" or "LIVE"
    testnet: bool


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/api/trades", response_model=TradesResponse)
def get_trades():
    """
    Return all open positions, recent closed trades, and live stats.
    This is the primary endpoint for the dashboard trade panel.
    """
    data  = _tracker.get_all()
    stats = data.get("stats", {})

    return TradesResponse(
        open    = data["open"],
        closed  = data["closed"],
        stats   = LiveStats(
            total_trades     = stats.get("total_trades",    0),
            wins             = stats.get("wins",            0),
            losses           = stats.get("losses",          0),
            win_rate_pct     = stats.get("win_rate_pct",    0.0),
            total_pnl        = stats.get("total_pnl",       0.0),
            total_pnl_pct    = stats.get("total_pnl_pct",   0.0),
            running_capital  = stats.get("running_capital", INITIAL_CAPITAL),
            profit_factor    = stats.get("profit_factor",   0.0),
            max_drawdown_pct = stats.get("max_drawdown_pct",0.0),
            last_trade_at    = stats.get("last_trade_at"),
        ),
        mode    = "LIVE" if LIVE_TRADING_ENABLED else "PAPER",
        testnet = EXCHANGE_TESTNET,
    )


@router.get("/api/trades/open")
def get_open_trades():
    """Return only open positions."""
    return {
        "open":  _tracker.get_open_trades(),
        "count": len(_tracker.get_open_trades()),
        "mode":  "LIVE" if LIVE_TRADING_ENABLED else "PAPER",
    }


@router.get("/api/trades/closed")
def get_closed_trades(limit: int = 50):
    """Return last N closed trades."""
    data = _tracker.get_all()
    closed = data["closed"][-limit:]
    return {
        "closed": closed,
        "count":  len(closed),
    }


@router.get("/api/trades/stats", response_model=LiveStats)
def get_live_stats():
    """Return live performance stats computed from closed trades."""
    stats = _tracker.get_stats()
    return LiveStats(
        total_trades     = stats.get("total_trades",    0),
        wins             = stats.get("wins",            0),
        losses           = stats.get("losses",          0),
        win_rate_pct     = stats.get("win_rate_pct",    0.0),
        total_pnl        = stats.get("total_pnl",       0.0),
        total_pnl_pct    = stats.get("total_pnl_pct",   0.0),
        running_capital  = stats.get("running_capital", INITIAL_CAPITAL),
        profit_factor    = stats.get("profit_factor",   0.0),
        max_drawdown_pct = stats.get("max_drawdown_pct",0.0),
        last_trade_at    = stats.get("last_trade_at"),
    )


@router.get("/api/trades/exchange")
def get_exchange_positions():
    """
    Fetch real-time position data directly from the exchange.
    Only meaningful when LIVE_TRADING_ENABLED=True.
    Returns empty list in paper mode.
    """
    if not LIVE_TRADING_ENABLED:
        return {
            "positions": [],
            "balance":   0.0,
            "mode":      "PAPER",
            "note":      "Connect exchange and set LIVE_TRADING_ENABLED=True for live data.",
        }

    if not _executor.connected:
        _executor.connect()

    return {
        "positions": _executor.get_open_positions(),
        "balance":   _executor.get_balance(),
        "mode":      "LIVE",
        "testnet":   EXCHANGE_TESTNET,
    }


@router.post("/api/trades/close-all")
def close_all_paper_trades():
    """
    Emergency close all open PAPER positions.
    For live positions, use the exchange directly.
    """
    open_trades = _tracker.get_open_trades()
    if not open_trades:
        return {"status": "ok", "message": "No open trades to close.", "closed": 0}

    closed_count = 0
    for trade in open_trades:
        if trade.get("paper"):
            # Close at current signal price (approximation for paper mode)
            try:
                from backend.scripts.live_predict import get_live_signal
                sig           = get_live_signal()
                current_price = sig.get("entry", trade["entry_price"])
            except Exception:
                current_price = trade["entry_price"]

            entry     = trade["entry_price"]
            size      = trade["size"]
            direction = trade["direction"]
            gross_pnl = ((current_price - entry) * size
                         if direction == "BUY"
                         else (entry - current_price) * size)
            fee      = entry * size * 0.00075 * 2
            net_pnl  = round(gross_pnl - fee, 4)

            _tracker.close_trade(
                order_id    = trade["id"],
                outcome     = "manual",
                close_price = current_price,
                pnl         = net_pnl,
            )
            closed_count += 1

    return {
        "status":  "ok",
        "message": f"Closed {closed_count} paper position(s).",
        "closed":  closed_count,
    }


@router.post("/api/trades/clear")
def clear_all_trades():
    """
    Wipe all open and closed trades and reset stats.
    Use this to start a fresh demo run.
    """
    _tracker.clear_all()
    return {
        "status":  "ok",
        "message": "All trades cleared. Ready for a fresh demo run.",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ── Analytics Endpoints ───────────────────────────────────────────────────────

@router.get("/api/analytics/summary")
def get_analytics_summary():
    """Fetch aggregated PnL, fee, and win rate metrics from the analytics DB."""
    from backend.services.analytics_db import AnalyticsDB
    adb = AnalyticsDB()
    return adb.get_summary()


@router.get("/api/analytics/trades")
def get_analytics_trades(limit: int = 100):
    """Fetch detailed trade-by-trade analytics from the DB."""
    from backend.services.analytics_db import AnalyticsDB
    adb = AnalyticsDB()
    return {
        "trades": adb.get_all_trades(limit=limit),
        "count":  limit
    }