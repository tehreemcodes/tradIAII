import sqlite3
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from backend.config.settings import ANALYTICS_DB_PATH

logger = logging.getLogger(__name__)

class AnalyticsDB:
    def __init__(self, db_path: Path = ANALYTICS_DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        """Initialize the database and create tables if they don't exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS trades_analytics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        trade_id TEXT UNIQUE,
                        symbol TEXT,
                        direction TEXT,
                        side TEXT,
                        entry_price REAL,
                        exit_price REAL,
                        expected_profit REAL,
                        expected_loss REAL,
                        actual_pnl REAL,
                        fees REAL,
                        slippage REAL,
                        order_type TEXT,
                        leverage INTEGER,
                        position_size REAL,
                        rr_ratio REAL,
                        opened_at TEXT,
                        closed_at TEXT,
                        status TEXT,
                        outcome TEXT
                    )
                """)
                conn.commit()
                logger.info(f"AnalyticsDB: Initialized at {self.db_path}")
        except Exception as e:
            logger.error(f"AnalyticsDB: Initialization failed: {e}")

    def record_open_trade(self, trade_data: Dict[str, Any]):
        """Record the expected metrics when a trade is opened."""
        try:
            # Calculated metrics
            entry = float(trade_data.get("entry_price", 0))
            sl = float(trade_data.get("sl_price", 0))
            tp = float(trade_data.get("tp_price", 0))
            size = float(trade_data.get("size", 0))
            direction = trade_data.get("direction", "BUY")

            sl_dist = abs(entry - sl)
            tp_dist = abs(tp - entry)
            
            expected_loss = sl_dist * size
            expected_profit = tp_dist * size
            rr_ratio = tp_dist / sl_dist if sl_dist > 0 else 0

            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO trades_analytics (
                        trade_id, symbol, direction, entry_price, 
                        expected_profit, expected_loss, order_type, 
                        leverage, position_size, rr_ratio, opened_at, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    trade_data.get("id"),
                    trade_data.get("symbol"),
                    direction,
                    entry,
                    round(expected_profit, 4),
                    round(expected_loss, 4),
                    trade_data.get("order_type", "market"),
                    trade_data.get("leverage", 1),
                    size,
                    round(rr_ratio, 2),
                    trade_data.get("opened_at"),
                    "open"
                ))
                conn.commit()
                logger.info(f"AnalyticsDB: Recorded open trade {trade_data.get('id')}")
        except Exception as e:
            logger.error(f"AnalyticsDB: Failed to record open trade: {e}")

    def record_close_trade(self, trade_id: str, close_data: Dict[str, Any]):
        """Update trade record with actual performance metrics on close."""
        try:
            actual_pnl = float(close_data.get("pnl", 0))
            exit_price = float(close_data.get("close_price", 0))
            closed_at = close_data.get("closed_at", datetime.now(timezone.utc).isoformat())
            outcome = close_data.get("outcome")

            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Fetch original entry price to calculate slippage and fees
                cursor.execute("SELECT entry_price, position_size, direction FROM trades_analytics WHERE trade_id = ?", (trade_id,))
                row = cursor.fetchone()
                if not row:
                    logger.warning(f"AnalyticsDB: Trade {trade_id} not found for closing.")
                    return

                expected_entry, size, direction = row
                
                # Slippage calculation (expected vs executed entry is handled in trade_executor,
                # but here 'entry_price' in analytics is 'intended_entry' from Predict, 
                # and 'close_price' is actual exit).
                # Actually, in live_trader, TradeExecutor returns 'average' price from Binance.
                # So we need to compare 'intended' vs 'actual' fill.
                # For now, let's store what we have.
                
                # Estimate total fees (Binance typical ~0.1% round trip)
                # But actual fees should be fetched from records if possible.
                # In live_trader, matched_records has 'fee'.
                fees = float(close_data.get("fees", 0))

                cursor.execute("""
                    UPDATE trades_analytics SET
                        exit_price = ?,
                        actual_pnl = ?,
                        fees = ?,
                        closed_at = ?,
                        status = ?,
                        outcome = ?
                    WHERE trade_id = ?
                """, (
                    exit_price,
                    round(actual_pnl, 4),
                    round(fees, 4),
                    closed_at,
                    "closed",
                    outcome,
                    trade_id
                ))
                conn.commit()
                logger.info(f"AnalyticsDB: Recorded closed trade {trade_id} with PnL {actual_pnl}")
        except Exception as e:
            logger.error(f"AnalyticsDB: Failed to record close trade: {e}")

    def get_summary(self) -> Dict[str, Any]:
        """Fetch aggregated metrics for the dashboard."""
        try:
            with self._get_connection() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute("SELECT COUNT(*) as total, SUM(actual_pnl) as net_pnl, SUM(fees) as total_fees FROM trades_analytics WHERE status = 'closed'")
                summary = cursor.fetchone()
                
                cursor.execute("SELECT COUNT(*) as wins FROM trades_analytics WHERE outcome = 'TP'")
                row_wins = cursor.fetchone()
                wins = row_wins['wins'] if row_wins else 0
                
                # Equity curve data
                cursor.execute("SELECT closed_at, actual_pnl FROM trades_analytics WHERE status = 'closed' ORDER BY closed_at ASC")
                history = cursor.fetchall()
                
                return {
                    "total_trades": summary['total'] or 0,
                    "net_pnl": summary['net_pnl'] or 0.0,
                    "total_fees": summary['total_fees'] or 0.0,
                    "win_rate_pct": (wins / summary['total'] * 100) if summary['total'] and summary['total'] > 0 else 0,
                    "history": [{"at": h['closed_at'], "pnl": h['actual_pnl']} for h in history]
                }
        except Exception as e:
            logger.error(f"AnalyticsDB: Failed to get summary: {e}")
            return {}

    def get_all_trades(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Fetch list of all trades for the analytics table."""
        try:
            with self._get_connection() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM trades_analytics ORDER BY opened_at DESC LIMIT ?", (limit,))
                rows = cursor.fetchall()
                return [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"AnalyticsDB: Failed to get all trades: {e}")
            return []
