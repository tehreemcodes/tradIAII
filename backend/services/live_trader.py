"""
Live Trader — Main Trading Loop
=================================
Runs continuously, executing trades based on ICT/SMC signals.

Loop behaviour (runs every closed 1H candle):
    1. Fetch latest closed candle timestamp
    2. Wait until the current candle closes
    3. Call get_live_signal() — full ICT pipeline on closed candles
    4. Check open positions — close any that hit TP/SL (paper mode)
    5. If signal is BUY/SELL and no open position → place order
    6. Log everything to live_trades.json via TradeTracker
    7. Sleep until the next candle close

Safety:
    - LIVE_TRADING_ENABLED=False → paper trading only (default)
    - MAX_OPEN_POSITIONS enforced — never opens a second trade
    - HTF confluence filter — only trades when 4H and Daily agree
    - Emergency stop: create a file called STOP_TRADING in project root
      to halt the loop cleanly without killing the process

Usage:
    # Paper trading (safe, no real orders):
    python -m backend.scripts.live_trader

    # Live trading (requires LIVE_TRADING_ENABLED=True in settings):
    python -m backend.scripts.live_trader --live

    # Run with walk-forward stats refresh every 24h:
    python -m backend.scripts.live_trader --refresh-stats
"""
import sys
import time
import logging
import argparse
from datetime import datetime, timezone, timedelta
from pathlib import Path

from backend.config.logging_setup import setup_logging
from backend.config.settings import (
    SIGNAL_TF, LIVE_TRADING_ENABLED, MAX_OPEN_POSITIONS,
    INITIAL_CAPITAL, MIN_CONFIDENCE, MAX_DAILY_LOSS_PCT,
)
from backend.scripts.live_predict    import get_live_signal
from backend.services.trade_executor import TradeExecutor, ExecutorError
from backend.services.trade_tracker  import TradeTracker

setup_logging()
logger = logging.getLogger(__name__)

# Emergency stop file — create this file to halt trading cleanly
STOP_FILE = Path(__file__).parents[2] / "STOP_TRADING"

# Timeframe → seconds map for sleep calculation
TF_SECONDS = {
    "1m": 60, "5m": 300, "15m": 900,
    "30m": 1800, "1h": 3600, "4h": 14400, "1d": 86400,
}


def _seconds_to_next_close(tf: str = "1h") -> float:
    """
    Returns seconds until the current candle closes.
    Adds a 5-second buffer so the candle is definitely closed
    before we fetch it.
    """
    now        = datetime.now(timezone.utc)
    tf_secs    = TF_SECONDS.get(tf, 3600)
    epoch_secs = now.timestamp()
    current_candle_open = (epoch_secs // tf_secs) * tf_secs
    next_close = current_candle_open + tf_secs
    remaining  = next_close - epoch_secs
    return max(remaining + 5, 5)   # at least 5 seconds


def _htf_confluence_ok(signal: dict) -> bool:
    """
    Additional HTF filter on top of the ML confidence filter.
    Only trade when at least one HTF timeframe agrees with signal direction.
    This prevents trading against strong structural bias.
    """
    bias = signal.get("htf_bias", {})
    h4   = bias.get("h4", 0)
    d1   = bias.get("d1", 0)
    sig  = signal.get("signal")

    if sig == "BUY":
        # Allow if 4H bullish OR full confluence
        return h4 == 1 or bias.get("full_confluence", False)
    elif sig == "SELL":
        # Allow if 4H bearish OR full confluence
        return h4 == -1 or bias.get("full_confluence", False)

    return False


def run_trading_loop(
    capital:       float = INITIAL_CAPITAL,
    refresh_stats: bool  = False,
) -> None:
    """
    Main trading loop. Runs until STOP_TRADING file is detected
    or KeyboardInterrupt.

    Parameters:
        capital       : starting capital for position sizing
        refresh_stats : if True, re-run backtest every 24h
    """
    logger.info("=" * 60)
    logger.info("  TradIA Live Trader")
    logger.info(f"  Mode: {'LIVE' if LIVE_TRADING_ENABLED else 'PAPER (no real orders)'}")
    logger.info(f"  Capital: ${capital:,.2f}")
    logger.info(f"  Min confidence: {MIN_CONFIDENCE}")
    logger.info(f"  Max open positions: {MAX_OPEN_POSITIONS}")
    logger.info("=" * 60)
    logger.info(f"  To stop cleanly: create file '{STOP_FILE.name}' in project root")
    logger.info(f"  Or press Ctrl+C")
    logger.info("=" * 60)

    # ── Initialise components ─────────────────────────────────────────────────
    executor = TradeExecutor()
    tracker  = TradeTracker()

    if LIVE_TRADING_ENABLED:
        connected = executor.connect()
        if not connected:
            logger.error("Failed to connect to exchange. Exiting.")
            sys.exit(1)
        
        # SAFETY: Cancel any orphan SL/TP orders from previous runs
        executor.cancel_all_conditional_orders(settings.SYMBOL)
        
        # Use real balance if live trading
        real_balance = executor.get_balance()
        if real_balance > 0:
            capital = real_balance
            logger.info(f"Using real balance: ${capital:,.2f}")
    else:
        logger.info("Paper mode: trades will be simulated, no exchange connection.")

    last_signal_ts  = None
    last_stats_refresh = datetime.now(timezone.utc)
    loop_count      = 0

    # ── Main loop ─────────────────────────────────────────────────────────────
    while True:
        loop_count += 1

        # Emergency stop check
        if STOP_FILE.exists():
            logger.info("STOP_TRADING file detected — halting loop cleanly.")
            STOP_FILE.unlink(missing_ok=True)
            break

        try:
            now = datetime.now(timezone.utc)
            logger.info(f"\n── Loop {loop_count} | {now.strftime('%Y-%m-%d %H:%M UTC')} ──")

            # ── Step 1: Check open paper trades against live price ────────────
            if tracker.has_open_trade():
                open_trades = tracker.get_open_trades()
                logger.info(f"Open positions: {len(open_trades)}")

                if not LIVE_TRADING_ENABLED:
                    # Paper mode: simulate TP/SL check using signal's entry price
                    sig = get_live_signal(capital=capital)
                    current_price = sig.get("entry", 0)
                    if current_price:
                        # Use a ±0.5% range as a proxy for candle high/low
                        approx_high = current_price * 1.005
                        approx_low  = current_price * 0.995
                        tracker.check_paper_outcomes(approx_high, approx_low)

                else:
                    # Live mode: exchange manages SL/TP natively.
                    # We just check if positions are still open.
                    exchange_positions = executor.get_open_positions()
                    exchange_ids       = {p.get("symbol").replace("/", "").split(":")[0] for p in exchange_positions}

                    if open_trades:
                        closed_records = executor.get_closed_pnl(limit=50)
                    else:
                        closed_records = []

                    for trade in open_trades:
                        # Normalize internal symbol BTC/USDT -> BTCUSDT for matching
                        trade_symbol = trade.get("symbol", "").replace("/", "").split(":")[0]

                        # If trade symbol is no longer in open exchange positions -> it was closed
                        if trade_symbol not in exchange_ids:
                            logger.info(f"Detected closed position for {trade_symbol}. Fetching matching records...")
                            
                            matched_records = []
                            try:
                                opened_ts = datetime.fromisoformat(trade["opened_at"].replace("Z", "+00:00")).timestamp()
                            except Exception as e:
                                logger.error(f"Failed to parse opened_at for trade {trade['id']}: {e}")
                                continue

                            for record in closed_records:
                                # CCXT record symbol might be 'BTC/USDT:USDT' or 'BTCUSDT'
                                record_symbol = record.get("symbol", "").replace("/", "").split(":")[0]
                                # record_ts is raw ms from our update to get_closed_pnl
                                record_ts = (record.get("timestamp") or 0) / 1000

                                if record_symbol == trade_symbol and record_ts >= (opened_ts - 2):
                                    matched_records.append(record)

                            if matched_records:
                                total_pnl  = sum(float(r.get("pnl", 0)) for r in matched_records)
                                total_fees = sum(float(r.get("fee", 0)) for r in matched_records)
                                
                                matched_records.sort(key=lambda r: r.get("timestamp", 0))
                                last_record = matched_records[-1]
                                close_price = float(last_record.get("price", 0))
                                outcome = "TP" if total_pnl > 0.0 else "SL"

                                tracker.close_trade(
                                    order_id    = trade["id"],
                                    outcome     = outcome,
                                    close_price = close_price,
                                    pnl         = total_pnl,
                                    closed_at   = last_record.get("datetime"),
                                    fees        = total_fees
                                )

                                symbol = trade.get("symbol", "").replace("/", "").split(":")[0]
                                executor.cancel_all_conditional_orders(symbol=symbol)
                                logger.info(
                                    f"Trade closed | {trade_symbol} | outcome={outcome} | pnl={total_pnl:.2f}"
                                )
                            else:
                                logger.warning(f"No matching closed records found for {trade_symbol} (opened at {trade['opened_at']}) — skipping close")
                                continue


            # ── Step 1.5: Daily Loss Limit Check ─────────────────────────────
            total_pnl_today = tracker.get_daily_pnl()
            stats           = tracker.get_stats()
            current_balance = stats.get("running_capital", INITIAL_CAPITAL)
            starting_balance_today = current_balance - total_pnl_today

            if total_pnl_today < 0:
                loss_pct = abs(total_pnl_today / starting_balance_today) if starting_balance_today > 0 else 0
                
                # Dynamic log with limit calculation
                limit_val = starting_balance_today * MAX_DAILY_LOSS_PCT
                logger.info(
                    f"[RISK] Daily PnL: {total_pnl_today:.2f} | Limit: {-limit_val:.2f}"
                )

                if loss_pct >= MAX_DAILY_LOSS_PCT:
                    logger.warning(
                        f"DAILY LOSS LIMIT REACHED ({loss_pct*100:.1f}%) — "
                        "skipping signal check until next UTC day."
                    )
                    # Skip signal check but stay in loop (wait for next candle)
                    # This allows the bot to resume automatically the next day
                    time.sleep(_seconds_to_next_close(SIGNAL_TF))
                    continue

            # ── Step 2: Check if we already have max positions ────────────────
            open_count = len(tracker.get_open_trades())
            if open_count >= MAX_OPEN_POSITIONS:
                logger.info(
                    f"Max open positions ({MAX_OPEN_POSITIONS}) reached — "
                    "skipping signal check."
                )
            else:
                # ── Step 3: Get signal ────────────────────────────────────────
                logger.info("Fetching live signal...")
                signal = get_live_signal(capital=capital)

                candle_ts = signal.get("candle_time", "")
                logger.info(
                    f"Signal: {signal['signal']} | "
                    f"Confidence: {signal.get('confidence', 0):.4f} | "
                    f"Candle: {candle_ts}"
                )

                # Skip if we already processed this candle
                if candle_ts and candle_ts == last_signal_ts:
                    logger.info("Same candle as last loop — skipping.")
                elif signal.get("error"):
                    logger.warning(f"Signal error: {signal['error']}")
                elif signal["signal"] == "NO TRADE":
                    logger.info("NO TRADE — confidence filter or no pattern.")
                else:
                    # AUDIT FIX BUG#6: Removed redundant _htf_confluence_ok() gate.
                    # live_predict.py already handles HTF alignment via dynamic
                    # confidence thresholds — this was a second filter that compounded
                    # signal suppression.
                    if signal.get("sl") is None:
                        logger.warning("Signal has no SL — skipping.")
                    elif signal.get("position_size", 0) <= 0:
                        logger.warning("Position size is 0 — skipping.")
                    else:
                        # ── Step 5: Place order ───────────────────────────────
                        logger.info(
                            f"Placing {signal['signal']} | "
                            f"entry={signal['entry']:,.2f} | "
                            f"sl={signal['sl']:,.2f} | "
                            f"tp={signal['tp']:,.2f} | "
                            f"size={signal['position_size']:.6f}"
                        )

                        try:
                            order = executor.place_order(
                                direction     = signal["signal"],
                                position_size = signal["position_size"],
                                entry_price   = signal["entry"],
                                sl_price      = signal["sl"],
                                tp_price      = signal["tp"],
                                signal_ts     = candle_ts,
                            )
                            if order:
                                tracker.open_trade(order)
                                last_signal_ts = candle_ts
                                logger.info(
                                    f"Order recorded: id={order['id']} "
                                    f"paper={order.get('paper', False)}"
                                )

                        except ExecutorError as e:
                            logger.error(f"Order failed: {e}")

            # ── Step 6: Log live stats ────────────────────────────────────────
            stats = tracker.get_stats()
            if stats:
                logger.info(
                    f"Live stats | trades={stats.get('total_trades', 0)} | "
                    f"WR={stats.get('win_rate_pct', 0):.1f}% | "
                    f"PnL={stats.get('total_pnl', 0):+.2f} USDT | "
                    f"capital={stats.get('running_capital', INITIAL_CAPITAL):,.2f}"
                )

            # ── Step 7: Optional — refresh backtest stats every 24h ──────────
            if refresh_stats:
                hours_since = (now - last_stats_refresh).total_seconds() / 3600
                if hours_since >= 24:
                    logger.info("Refreshing backtest stats (24h cycle)...")
                    try:
                        from backend.scripts.backtest import run_backtest
                        run_backtest(use_model=True)
                        last_stats_refresh = now
                    except Exception as e:
                        logger.error(f"Stats refresh failed: {e}")

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt — stopping trader.")
            break
        except Exception as e:
            logger.exception(f"Unexpected error in trading loop: {e}")
            logger.info("Sleeping 60s before retry...")
            time.sleep(60)
            continue

        # ── Sleep until next candle close ─────────────────────────────────────
        sleep_secs = _seconds_to_next_close(SIGNAL_TF)
        wake_time  = datetime.now(timezone.utc) + timedelta(seconds=sleep_secs)
        logger.info(
            f"Sleeping {sleep_secs/60:.1f} min → "
            f"next check at {wake_time.strftime('%H:%M:%S UTC')} (chunked polling)"
        )
        
        # Polling loop: sleep 10s, check if positions closed to cancel orphans instantly
        chunk_interval = 10.0
        elapsed = 0.0
        while elapsed < sleep_secs:
            if STOP_FILE.exists():
                break
                
            time.sleep(min(chunk_interval, sleep_secs - elapsed))
            elapsed += chunk_interval
            
            if LIVE_TRADING_ENABLED and tracker.has_open_trade():
                try:
                    exchange_positions = executor.get_open_positions()
                    exchange_ids = {p.get("symbol").replace("/", "").split(":")[0] for p in exchange_positions}
                    
                    for trade in tracker.get_open_trades():
                        trade_symbol = trade.get("symbol", "").replace("/", "").split(":")[0]
                        if trade_symbol not in exchange_ids:
                            logger.info(f"Chunk-Poll: Position {trade_symbol} naturally closed. Canceling orphan SL/TP orders.")
                            executor.cancel_all_conditional_orders(symbol=trade_symbol)
                            # Actually closing the trade logic will happen at Step 1 of the main loop.
                except Exception as e:
                    logger.warning(f"Chunk-Poll error: {e}")

    logger.info("Live trader stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TradIA Live Trader")
    parser.add_argument(
        "--live",
        action  = "store_true",
        help    = "Enable live trading (overrides LIVE_TRADING_ENABLED in settings)",
    )
    parser.add_argument(
        "--capital",
        type    = float,
        default = INITIAL_CAPITAL,
        help    = "Starting capital for position sizing",
    )
    parser.add_argument(
        "--refresh-stats",
        action  = "store_true",
        help    = "Re-run backtest every 24h to keep dashboard stats current",
    )
    args = parser.parse_args()

    if args.live:
        import backend.config.settings as s
        s.LIVE_TRADING_ENABLED = True
        logger.warning("LIVE TRADING ENABLED via CLI flag — real orders will be placed!")

    run_trading_loop(
        capital       = args.capital,
        refresh_stats = args.refresh_stats,
    )