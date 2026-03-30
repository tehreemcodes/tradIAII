"""
Exchange Routes — Connect, Trade, Monitor
==========================================
All endpoints for exchange connection and auto-trading.

Mount in api_server.py:
    from backend.api.exchange_routes import router as exchange_router
    app.include_router(exchange_router)

Endpoints:
    POST /api/exchange/connect         validate keys + create session
    POST /api/exchange/disconnect      remove session
    GET  /api/exchange/status          connection status + balance
    GET  /api/exchange/positions       live positions from exchange
    GET  /api/trades                   open + closed trades + stats
    GET  /api/trades/open              open positions
    GET  /api/trades/closed            trade history
    GET  /api/trades/stats             live performance stats
    POST /api/trades/close-all         emergency close all positions
    POST /api/trades/clear             reset all trades (fresh demo)
    POST /api/trading/enable           enable auto-trading for session
    POST /api/trading/disable          disable auto-trading for session
    GET  /api/trading/status           is auto-trading active?
"""
import logging
from datetime import datetime, timezone
from typing import Optional

import ccxt
from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel

from backend.config.settings import (
    INITIAL_CAPITAL, EXCHANGE_TESTNET, SYMBOL, RISK_PCT, DEFAULT_LEVERAGE,
)
from backend.services.credential_store import CredentialStore
from backend.services.trade_executor   import TradeExecutor, ExecutorError
from backend.services.trade_tracker    import TradeTracker

logger = logging.getLogger(__name__)
router = APIRouter()

# ── Module-level trade tracker (shared across all sessions) ───────────────────
_tracker = TradeTracker()

# ── Active trading sessions ───────────────────────────────────────────────────
# session_id → {"executor": TradeExecutor, "enabled": bool, "risk_pct": float}
_active_sessions: dict[str, dict] = {}


# ── Request / Response models ─────────────────────────────────────────────────

class ConnectRequest(BaseModel):
    api_key:    str
    api_secret: str
    exchange:   str  = "binance"
    testnet:    bool = False

class ConnectResponse(BaseModel):
    connected:    bool
    session_id:   str
    exchange:     str
    testnet:      bool
    balance:      float
    connected_at: str
    message:      str

class StatusResponse(BaseModel):
    connected:    bool
    exchange:     Optional[str]   = None
    testnet:      Optional[bool]  = None
    balance:      Optional[float] = None
    connected_at: Optional[str]   = None
    expires_at:   Optional[str]   = None

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
    mode:    str
    testnet: bool


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_session_id(x_session_id: Optional[str] = Header(None)) -> Optional[str]:
    return x_session_id


def _require_session(session_id: Optional[str]) -> dict:
    """Validate session and return session info. Raises 401 if invalid."""
    if not session_id:
        raise HTTPException(
            status_code=401,
            detail="No session. Connect your exchange first via POST /api/exchange/connect"
        )
    info = CredentialStore.get_session_info(session_id)
    if not info:
        raise HTTPException(
            status_code=401,
            detail="Session expired or not found. Please reconnect."
        )
    return info


def _get_executor(session_id: str) -> TradeExecutor:
    """Get or create a connected TradeExecutor for a session."""
    if session_id in _active_sessions:
        return _active_sessions[session_id]["executor"]

    creds = CredentialStore.get_credentials(session_id)
    if not creds:
        raise HTTPException(status_code=401, detail="Session credentials not found.")

    executor = TradeExecutor(
        api_key    = creds["api_key"],
        api_secret = creds["api_secret"],
        exchange   = creds["exchange"],
        testnet    = creds["testnet"],
        leverage   = DEFAULT_LEVERAGE,
    )
    ok = executor.connect()
    if not ok:
        raise HTTPException(
            status_code=502,
            detail="Could not connect to exchange. Check your API keys."
        )

    _active_sessions[session_id] = {
        "executor":  executor,
        "enabled":   False,
        "risk_pct":  RISK_PCT,
        "leverage":  DEFAULT_LEVERAGE,
    }
    return executor


# ── Exchange Connection ───────────────────────────────────────────────────────

@router.post("/api/exchange/connect", response_model=ConnectResponse)
def connect_exchange(req: ConnectRequest):
    """
    Validate API keys by attempting a real balance fetch.
    On success, creates an encrypted session and returns a session_id.
    The session_id is what the frontend stores — raw keys are never returned.

    Security:
        - Keys are encrypted with AES-256-GCM before storage
        - Withdrawals cannot be enabled via this bot
        - Keys are validated against the exchange before storing
    """
    if not req.api_key or not req.api_secret:
        raise HTTPException(status_code=400, detail="API key and secret are required.")

    if len(req.api_key) < 10 or len(req.api_secret) < 10:
        raise HTTPException(status_code=400, detail="API key or secret appears invalid.")

    # Validate by attempting connection
    try:
        executor = TradeExecutor(
            api_key    = req.api_key,
            api_secret = req.api_secret,
            exchange   = req.exchange,
            testnet    = req.testnet,
        )
        ok = executor.connect()
        if not ok:
            raise HTTPException(
                status_code=401,
                detail=(
                    "Could not authenticate with exchange. "
                    "Check your API key and secret are correct and have "
                    "Futures trading permissions enabled."
                ),
            )

        balance = executor.get_balance()

    except ccxt.AuthenticationError:
        raise HTTPException(
            status_code=401,
            detail="Invalid API credentials. Check key and secret."
        )
    except ccxt.NetworkError:
        raise HTTPException(
            status_code=503,
            detail="Exchange unreachable. Check your internet connection."
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Connection error: {str(e)}")

    # Store encrypted credentials
    session_id = CredentialStore.create_session(
        api_key    = req.api_key,
        api_secret = req.api_secret,
        exchange   = req.exchange,
        testnet    = req.testnet,
        balance    = balance,
    )

    # Cache the live executor for this session
    _active_sessions[session_id] = {
        "executor": executor,
        "enabled":  False,
        "risk_pct": RISK_PCT,
        "leverage": DEFAULT_LEVERAGE,
    }

    now = datetime.now(timezone.utc).isoformat()
    logger.info(
        f"Exchange connected: {req.exchange} "
        f"testnet={req.testnet} balance={balance:.2f} "
        f"session={session_id[:8]}..."
    )

    return ConnectResponse(
        connected    = True,
        session_id   = session_id,
        exchange     = req.exchange,
        testnet      = req.testnet,
        balance      = balance,
        connected_at = now,
        message      = f"Connected to {req.exchange.upper()}. Balance: ${balance:,.2f} USDT",
    )


@router.post("/api/exchange/disconnect")
def disconnect_exchange(
    x_session_id: Optional[str] = Header(None)
):
    """Remove session and disconnect from exchange."""
    if not x_session_id:
        return {"status": "ok", "message": "No active session."}

    _active_sessions.pop(x_session_id, None)
    existed = CredentialStore.delete_session(x_session_id)

    return {
        "status":  "ok",
        "message": "Disconnected." if existed else "Session not found.",
    }


@router.get("/api/exchange/status", response_model=StatusResponse)
def exchange_status(x_session_id: Optional[str] = Header(None)):
    """
    Return connection status for the current session.
    Safe to call without a session — returns connected=False.
    """
    if not x_session_id:
        return StatusResponse(connected=False)

    info = CredentialStore.get_session_info(x_session_id)
    if not info:
        return StatusResponse(connected=False)

    # Refresh balance if executor is available
    if x_session_id in _active_sessions:
        try:
            executor = _active_sessions[x_session_id]["executor"]
            balance  = executor.get_balance()
            CredentialStore.update_balance(x_session_id, balance)
            info["balance"] = balance
        except Exception:
            pass

    return StatusResponse(
        connected    = True,
        exchange     = info["exchange"],
        testnet      = info["testnet"],
        balance      = info["balance"],
        connected_at = info["connected_at"],
        expires_at   = info["expires_at"],
    )


@router.get("/api/exchange/positions")
def get_live_positions(x_session_id: Optional[str] = Header(None)):
    """Fetch real-time positions directly from the exchange."""
    _require_session(x_session_id)
    executor = _get_executor(x_session_id)
    return {
        "positions": executor.get_open_positions(),
        "balance":   executor.get_balance(),
        "symbol":    SYMBOL,
    }


# ── Auto-Trading Control ──────────────────────────────────────────────────────

@router.post("/api/trading/enable")
def enable_trading(
    risk_pct:     float = RISK_PCT,
    x_session_id: Optional[str] = Header(None),
):
    """
    Enable auto-trading for this session.
    The live_trader.py loop checks this flag before placing orders.
    """
    _require_session(x_session_id)
    _get_executor(x_session_id)   # ensure connected

    if x_session_id in _active_sessions:
        _active_sessions[x_session_id]["enabled"]  = True
        _active_sessions[x_session_id]["risk_pct"] = min(risk_pct, 0.25)  # cap 25%

    logger.info(f"Auto-trading ENABLED: session={x_session_id[:8]}... risk={risk_pct:.0%}")
    return {
        "status":   "enabled",
        "risk_pct": risk_pct,
        "message":  f"Auto-trading enabled at {risk_pct*100:.0f}% risk per trade.",
    }


@router.post("/api/trading/disable")
def disable_trading(x_session_id: Optional[str] = Header(None)):
    """Disable auto-trading. Open positions are NOT closed."""
    if x_session_id and x_session_id in _active_sessions:
        _active_sessions[x_session_id]["enabled"] = False

    return {"status": "disabled", "message": "Auto-trading disabled. Open positions unchanged."}


@router.get("/api/trading/status")
def trading_status(x_session_id: Optional[str] = Header(None)):
    """Is auto-trading currently active for this session?"""
    if not x_session_id or x_session_id not in _active_sessions:
        return {"enabled": False, "connected": False}

    session = _active_sessions[x_session_id]
    return {
        "enabled":   session["enabled"],
        "connected": session["executor"].connected,
        "risk_pct":  session["risk_pct"],
    }


@router.get("/api/trading/paper-mode")
def get_paper_mode():
    import backend.config.settings as settings
    return {"paper_mode": settings.PAPER_MODE}


class PaperModeRequest(BaseModel):
    enabled: bool

@router.post("/api/trading/paper-mode")
def set_paper_mode(req: PaperModeRequest):
    import backend.config.settings as settings
    settings.PAPER_MODE = req.enabled
    logger.info(f"Global PAPER_MODE set to {req.enabled}")
    return {"status": "ok", "paper_mode": req.enabled}


# ── Trade Data ────────────────────────────────────────────────────────────────

@router.get("/api/trades", response_model=TradesResponse)
def get_trades(x_session_id: Optional[str] = Header(None)):
    """All open + closed trades with live stats."""
    data  = _tracker.get_all()
    stats = data.get("stats", {})

    info = CredentialStore.get_session_info(x_session_id) if x_session_id else None
    mode = "LIVE" if (info and x_session_id in _active_sessions
                      and _active_sessions[x_session_id]["enabled"]) else "PAPER"

    return TradesResponse(
        open   = data["open"],
        closed = data["closed"],
        stats  = LiveStats(
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
        mode    = mode,
        testnet = info["testnet"] if info else True,
    )


@router.get("/api/trades/open")
def get_open_trades():
    return {"open": _tracker.get_open_trades(), "count": len(_tracker.get_open_trades())}


@router.get("/api/trades/closed")
def get_closed_trades(limit: int = 50):
    data   = _tracker.get_all()
    closed = data["closed"][-limit:]
    return {"closed": closed, "count": len(closed)}


@router.get("/api/trades/stats", response_model=LiveStats)
def get_live_stats():
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


@router.post("/api/trades/close-all")
def close_all(x_session_id: Optional[str] = Header(None)):
    """Emergency close all open positions."""
    open_trades  = _tracker.get_open_trades()
    if not open_trades:
        return {"status": "ok", "message": "No open trades.", "closed": 0}

    executor     = None
    if x_session_id and x_session_id in _active_sessions:
        executor = _active_sessions[x_session_id]["executor"]

    closed_count = 0
    for trade in open_trades:
        try:
            if executor and executor.connected and not trade.get("paper"):
                executor.close_position(trade["direction"], trade["size"], "emergency")

            # Get approximate close price
            close_price = trade["entry_price"]
            if executor and executor.connected:
                positions = executor.get_open_positions()
                if positions:
                    close_price = positions[0].get("mark_price", close_price)

            direction = trade["direction"]
            entry     = trade["entry_price"]
            size      = trade["size"]
            gross_pnl = ((close_price - entry) * size if direction == "BUY"
                         else (entry - close_price) * size)
            net_pnl   = round(gross_pnl - entry * size * 0.00075 * 2, 4)

            _tracker.close_trade(
                order_id    = trade["id"],
                outcome     = "manual",
                close_price = close_price,
                pnl         = net_pnl,
            )
            closed_count += 1
        except Exception as e:
            logger.error(f"Failed to close trade {trade['id']}: {e}")

    return {"status": "ok", "closed": closed_count}


@router.post("/api/trades/clear")
def clear_trades():
    """Wipe all trade history. Use to start a fresh demo run."""
    _tracker.clear_all()
    return {"status": "ok", "message": "All trades cleared.", "timestamp": datetime.now(timezone.utc).isoformat()}