# ── bg_trader.py ──
import asyncio
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Optional

from backend.config.settings import SIGNAL_TF, MIN_CONFIDENCE
from backend.scripts.live_predict import get_live_signal
from backend.services.trade_tracker import TradeTracker
from backend.api.exchange_routes import _active_sessions
from backend.services.trade_executor import ExecutorError

logger = logging.getLogger(__name__)

# Timeframe ? seconds map for sleep calculation
TF_SECONDS = {
    "1m": 60, "5m": 300, "15m": 900,
    "30m": 1800, "1h": 3600, "4h": 14400, "1d": 86400,
}


def _seconds_to_next_close(tf: str = "1h") -> float:
    now        = datetime.now(timezone.utc)
    tf_secs    = TF_SECONDS.get(tf, 3600)
    epoch_secs = now.timestamp()
    current_candle_open = (epoch_secs // tf_secs) * tf_secs
    next_close = current_candle_open + tf_secs
    remaining  = next_close - epoch_secs
    return max(remaining + 5, 5)   # at least 5 seconds buffer





async def start_bg_trader():
    """
    Background trading loop integrated into the FastAPI server.
    Iterates over all enabled _active_sessions and places trades.
    """
    logger.info("=" * 60)
    logger.info("  TradIA Background Live Trader Started")
    logger.info("  Checks active dashboard sessions for auto-trading flags.")
    logger.info("=" * 60)

    tracker = TradeTracker()
    last_signal_ts = None
    loop_count = 0

    while True:
        try:
            loop_count += 1
            now = datetime.now(timezone.utc)
            logger.info(f"[BG_TRADER] Loop {loop_count} | {now.strftime('%Y-%m-%d %H:%M UTC')}")

            active_enabled_sessions = {
                sid: data for sid, data in _active_sessions.items()
                if data.get("enabled", False)
            }

            if not active_enabled_sessions:
                logger.debug("[BG_TRADER] No active auto-trading sessions. Skipping checks.")
            else:
                # ── Step 1: Manage Open Trades ────────────────────────────────
                if tracker.has_open_trade():
                    open_trades = tracker.get_open_trades()
                    for sid, session_data in active_enabled_sessions.items():
                        executor = session_data.get("executor")
                        if not executor or not executor.connected:
                            continue
                        
                        try:
                            exchange_positions = executor.get_open_positions()
                            if not exchange_positions:
                                closed_records = executor.get_closed_pnl(limit=5)
                                for trade in open_trades:
                                    if trade.get("exchange") == executor._exchange_name:
                                        for record in closed_records:
                                            if record.get("order_id") == trade.get("id"):
                                                pnl = float(record.get("pnl", 0))
                                                outcome = "TP" if pnl > 0 else "SL"
                                                tracker.close_trade(
                                                    order_id    = trade["id"],
                                                    outcome     = outcome,
                                                    close_price = float(record.get("price", 0)),
                                                    pnl         = pnl,
                                                )
                        except Exception as e:
                            logger.error(f"[BG_TRADER] Error managing open trades for session {sid[:8]}: {e}")

                # ── Step 2: Get Signal Once ──────────────────────────────────
                logger.info("[BG_TRADER] Fetching live signal from model...")
                # We call get_live_signal with a dummy capital just to get the raw ML prediction (entry/sl/tp/confidence/direction)
                # The sizing will be re-calculated per user account.
                signal = get_live_signal(capital=1000)

                candle_ts = signal.get("candle_time", "")
                
                if candle_ts and candle_ts != last_signal_ts:
                    logger.info(f"[BG_TRADER] Signal: {signal.get('signal')} | Confidence: {signal.get('confidence', 0):.4f} | Executable: {signal.get('executable')} | Reason: {signal.get('reject_reason')} | Candle: {candle_ts}")
                    
                    if signal.get("signal") not in ["NO TRADE", None] and signal.get("executable", False) and signal.get("sl") is not None:
                        # ── Step 3: Iterate Over Users and Execute ───────────
                        for sid, session_data in active_enabled_sessions.items():
                            executor = session_data.get("executor")
                            risk_pct = session_data.get("risk_pct", 0.01)

                            if not executor or not executor.connected:
                                continue

                            try:
                                # Fetch actual balance for this user
                                user_balance = executor.get_balance()
                                if user_balance <= 0:
                                    logger.warning(f"[BG_TRADER] Session {sid[:8]} has 0 balance. Skipping.")
                                    continue

                                # Recalculate position size using user's real balance
                                # Re-import RiskManager here to avoid circular imports if needed, but we can do it directly
                                from backend.services.risk_manager import RiskManager
                                rm = RiskManager(initial_capital=user_balance, risk_pct=risk_pct, compound=True)
                                
                                # Convert TS strings back to pd.Timestamp for RiskManager
                                import pandas as pd
                                ts_obj = pd.Timestamp(candle_ts) if candle_ts else pd.Timestamp(now)
                                
                                trade = rm.calculate_position(
                                    entry=signal["entry"],
                                    sl=signal["sl"],
                                    direction=signal["signal"],
                                    ts=ts_obj
                                )
                                
                                if trade and trade.position_size > 0:
                                    logger.info(f"[BG_TRADER] Placing {trade.direction} for session {sid[:8]} | size={trade.position_size:.6f}")
                                    order = executor.place_order(
                                        direction     = trade.direction,
                                        position_size = trade.position_size,
                                        entry_price   = trade.entry,
                                        sl_price      = trade.sl,
                                        tp_price      = trade.tp,
                                        signal_ts     = candle_ts,
                                    )
                                    if order:
                                        tracker.open_trade(order)
                            except ExecutorError as e:
                                logger.error(f"[BG_TRADER] Order failed for session {sid[:8]}: {e}")
                            except Exception as e:
                                logger.error(f"[BG_TRADER] Error processing signal for session {sid[:8]}: {e}")
                                
                        last_signal_ts = candle_ts

        except asyncio.CancelledError:
            logger.info("[BG_TRADER] Background trader cancelled.")
            break
        except Exception as e:
            logger.exception(f"[BG_TRADER] Unexpected error: {e}")
            await asyncio.sleep(60)
            continue

        # ── Step 4: Sleep ────────────────────────────────────────────────────
        sleep_secs = _seconds_to_next_close(SIGNAL_TF)
        wake_time  = datetime.now(timezone.utc) + timedelta(seconds=sleep_secs)
        logger.info(f"[BG_TRADER] Sleeping {sleep_secs/60:.1f} min -> next check at {wake_time.strftime('%H:%M:%S UTC')}")
        await asyncio.sleep(sleep_secs)
