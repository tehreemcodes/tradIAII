"""
Data Fetcher
=============
Downloads historical OHLCV data for all 4 timeframes.
Primary: Bybit (works from Pakistan, no geo-restrictions)
Fallback: KuCoin

PAGINATION FIX: Only stops when exchange returns empty response
or timestamp stops advancing. Never stops on partial batches.

Usage:
    python -m backend.scripts.fetch_data
"""
import ccxt
import pandas as pd
import time
import logging
from backend.config.settings import (
    DATA_DIR, SYMBOL, EXCHANGE,
    TIMEFRAMES, FETCH_START, DATA_FILES,
)
from backend.config.logging_setup import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def get_exchange():
    """Connect to Bybit, fall back to KuCoin automatically."""
    for name in [EXCHANGE, "bybit", "kucoin"]:
        try:
            ex = getattr(ccxt, name)({"enableRateLimit": True})
            ex.load_markets()
            logger.info(f"Connected to {name.upper()}")
            return ex
        except Exception as e:
            logger.warning(f"{name} failed: {e}")
    raise RuntimeError(
        "All exchanges failed. Check your internet connection."
    )


def fetch_ohlcv(
    exchange,
    symbol:    str,
    timeframe: str,
    start:     str,
    limit:     int = 1000,
) -> pd.DataFrame:
    """
    Paginated historical fetch from start date to present.

    Stops ONLY when:
      - Exchange returns empty response (reached present)
      - Timestamp stops advancing (stall detection)
      - Max retries exceeded
    """
    since       = exchange.parse8601(start)
    all_candles = []
    retries     = 0
    last_ts     = -1

    logger.info(f"[{timeframe}] Fetching {symbol} from {start}...")

    while True:
        try:
            candles = exchange.fetch_ohlcv(
                symbol, timeframe, since=since, limit=limit
            )
            retries = 0
        except ccxt.RateLimitExceeded:
            logger.warning("[{timeframe}] Rate limited, waiting 15s...")
            time.sleep(15)
            continue
        except Exception as e:
            retries += 1
            logger.error(f"[{timeframe}] Error ({retries}/5): {e}")
            if retries >= 5:
                logger.error(f"[{timeframe}] Max retries — stopping.")
                break
            time.sleep(5 * retries)
            continue

        # Empty = reached present day
        if not candles:
            logger.info(f"[{timeframe}] Reached end of available data.")
            break

        # Stall detection — timestamp not advancing
        if candles[-1][0] == last_ts:
            logger.warning(f"[{timeframe}] Timestamp stalled — stopping.")
            break

        last_ts = candles[-1][0]
        all_candles.extend(candles)
        since = candles[-1][0] + 1

        logger.info(f"[{timeframe}] {len(all_candles):,} candles...")
        time.sleep(exchange.rateLimit / 1000)

        # NOTE: Do NOT break on partial batches.
        # Bybit returns partial batches mid-history.
        # Only stop on empty response.

    if not all_candles:
        return pd.DataFrame()

    df = pd.DataFrame(
        all_candles,
        columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df = df[df["close"] > 0].dropna()

    logger.info(
        f"[{timeframe}] DONE: {len(df):,} candles "
        f"({df.index[0].date()} to {df.index[-1].date()})"
    )
    return df


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    exchange = get_exchange()

    for tf in TIMEFRAMES:
        out_path = DATA_FILES[tf]

        if out_path.exists():
            logger.info(
                f"[{tf}] Already exists — skipping. "
                f"Delete {out_path.name} to re-fetch."
            )
            continue

        df = fetch_ohlcv(exchange, SYMBOL, tf, FETCH_START[tf])

        if df.empty:
            logger.error(f"[{tf}] No data returned — skipping.")
            continue

        df.to_csv(out_path)
        logger.info(f"[{tf}] Saved {len(df):,} rows -> {out_path.name}")

    logger.info("All timeframes complete.")


if __name__ == "__main__":
    main()
