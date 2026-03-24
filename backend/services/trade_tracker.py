"""
Trade Tracker
==============
Persists all live/paper trades to a JSON file so nothing is lost
if the server restarts mid-trade.

Responsibilities:
    - Record newly opened trades
    - Mark trades as closed with outcome and P&L
    - Compute running performance stats (live win rate, P&L, etc.)
    - Serve data to the API for the dashboard

File format (live_trades.json):
    {
        "open":   [ {trade dict}, ... ],
        "closed": [ {trade dict}, ... ],
        "stats":  { running stats dict }
    }

Thread safety:
    All read/write operations acquire _lock.
    Safe to call from the background trading loop and API simultaneously.
"""
import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from backend.config.settings import TRADE_LOG_PATH, INITIAL_CAPITAL

logger = logging.getLogger(__name__)


class TradeTracker:

    def __init__(self, path: Path = TRADE_LOG_PATH):
        self._path  = path
        self._lock  = threading.Lock()
        self._data  = self._load()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self) -> dict:
        """Load trades from disk. Creates empty structure if file missing."""
        if self._path.exists():
            try:
                with open(self._path) as f:
                    data = json.load(f)
                    logger.info(
                        f"TradeTracker: loaded {len(data.get('open', []))} open, "
                        f"{len(data.get('closed', []))} closed trades"
                    )
                    return data
            except Exception as e:
                logger.error(f"TradeTracker: failed to load {self._path}: {e}")

        return {"open": [], "closed": [], "stats": {}}

    def _save(self) -> None:
        """Write current state to disk. Always called inside _lock."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "w") as f:
                json.dump(self._data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"TradeTracker: failed to save: {e}")

    # ── Open Trade Management ─────────────────────────────────────────────────

    def has_open_trade(self) -> bool:
        """True if there is at least one open position."""
        with self._lock:
            return len(self._data["open"]) > 0

    def get_open_trades(self) -> list[dict]:
        """Return a copy of all open trades."""
        with self._lock:
            return list(self._data["open"])

    def open_trade(self, order: dict) -> None:
        """
        Record a newly placed order as an open trade.
        order dict comes directly from TradeExecutor.place_order().
        """
        with self._lock:
            self._data["open"].append(order)
            self._save()
            logger.info(
                f"TradeTracker: opened {order['direction']} "
                f"id={order['id']} entry={order['entry_price']}"
            )

    def close_trade(
        self,
        order_id:    str,
        outcome:     str,    # "TP" | "SL" | "manual"
        close_price: float,
        pnl:         float,  # net P&L in USDT after fees
        closed_at:   Optional[str] = None,
    ) -> Optional[dict]:
        """
        Move a trade from open → closed and record the outcome.
        Returns the closed trade dict, or None if order_id not found.
        """
        with self._lock:
            idx = next(
                (i for i, t in enumerate(self._data["open"]) if t["id"] == order_id),
                None,
            )
            if idx is None:
                logger.warning(f"TradeTracker: close_trade — id {order_id} not found in open trades")
                return None

            trade = self._data["open"].pop(idx)
            trade.update({
                "outcome":     outcome,
                "close_price": close_price,
                "pnl":         pnl,
                "closed_at":   closed_at or datetime.now(timezone.utc).isoformat(),
            })
            self._data["closed"].append(trade)
            self._data["stats"] = self._compute_stats()
            self._save()

            logger.info(
                f"TradeTracker: closed {trade['direction']} id={order_id} "
                f"outcome={outcome} pnl={pnl:+.2f}"
            )
            return trade

    # ── Paper trade outcome simulation ────────────────────────────────────────

    def check_paper_outcomes(self, current_high: float, current_low: float) -> None:
        """
        For paper trades, check if the current candle's high/low
        has hit TP or SL and close them accordingly.

        Called each candle close from live_trader.py.
        Real trades are managed by exchange-native SL/TP orders.
        """
        with self._lock:
            to_close = []
            for trade in self._data["open"]:
                if not trade.get("paper"):
                    continue

                direction = trade["direction"]
                tp        = trade["tp_price"]
                sl        = trade["sl_price"]
                entry     = trade["entry_price"]

                if direction == "BUY":
                    if current_low  <= sl: to_close.append((trade["id"], "SL", sl))
                    elif current_high >= tp: to_close.append((trade["id"], "TP", tp))
                else:
                    if current_high >= sl: to_close.append((trade["id"], "SL", sl))
                    elif current_low  <= tp: to_close.append((trade["id"], "TP", tp))

        for order_id, outcome, close_price in to_close:
            # Find the trade to compute P&L
            trade = next((t for t in self._data["open"] if t["id"] == order_id), None)
            if trade is None:
                continue

            entry     = trade["entry_price"]
            size      = trade["size"]
            direction = trade["direction"]

            # Gross P&L
            if direction == "BUY":
                gross_pnl = (close_price - entry) * size
            else:
                gross_pnl = (entry - close_price) * size

            # Estimate round-trip fee (0.075% taker × 2 sides)
            fee       = entry * size * 0.00075 * 2
            net_pnl   = round(gross_pnl - fee, 4)

            self.close_trade(order_id, outcome, close_price, net_pnl)

    # ── Stats ─────────────────────────────────────────────────────────────────

    def _compute_stats(self) -> dict:
        """
        Compute running live performance stats from closed trades.
        Called internally every time a trade is closed.
        """
        closed = self._data["closed"]
        if not closed:
            return {}

        wins   = [t for t in closed if t.get("outcome") == "TP"]
        losses = [t for t in closed if t.get("outcome") == "SL"]
        total  = len(closed)

        total_pnl   = sum(t.get("pnl", 0) for t in closed)
        win_pnl     = sum(t.get("pnl", 0) for t in wins   if t.get("pnl", 0) > 0)
        loss_pnl    = sum(abs(t.get("pnl", 0)) for t in losses if t.get("pnl", 0) < 0)
        profit_factor = round(win_pnl / loss_pnl, 2) if loss_pnl > 0 else 0.0

        # Running capital (starts from INITIAL_CAPITAL)
        running_capital = INITIAL_CAPITAL + total_pnl

        # Max drawdown from running capital curve
        caps     = []
        cap      = INITIAL_CAPITAL
        for t in closed:
            cap += t.get("pnl", 0)
            caps.append(cap)

        import numpy as np
        if caps:
            arr      = np.array(caps)
            peak     = np.maximum.accumulate(
                np.concatenate([[INITIAL_CAPITAL], arr])
            )
            dd       = (peak[1:] - arr) / peak[1:]
            max_dd   = float(np.max(dd)) * 100
        else:
            max_dd = 0.0

        return {
            "total_trades":    total,
            "wins":            len(wins),
            "losses":          len(losses),
            "win_rate_pct":    round(len(wins) / total * 100, 2) if total else 0,
            "total_pnl":       round(total_pnl, 2),
            "total_pnl_pct":   round(total_pnl / INITIAL_CAPITAL * 100, 2),
            "running_capital": round(running_capital, 2),
            "profit_factor":   profit_factor,
            "max_drawdown_pct":round(max_dd, 2),
            "last_trade_at":   closed[-1].get("closed_at"),
        }

    def get_stats(self) -> dict:
        """Return the latest computed live stats."""
        with self._lock:
            return dict(self._data.get("stats", {}))

    def get_all(self) -> dict:
        """Return full trade data for the API."""
        with self._lock:
            return {
                "open":   list(self._data["open"]),
                "closed": list(self._data["closed"][-50:]),  # last 50
                "stats":  dict(self._data.get("stats", {})),
            }

    def clear_all(self) -> None:
        """
        Wipe all trades and reset stats.
        Use this to start a fresh 10-day demo run.
        """
        with self._lock:
            self._data = {"open": [], "closed": [], "stats": {}}
            self._save()
            logger.info("TradeTracker: all trades cleared.")