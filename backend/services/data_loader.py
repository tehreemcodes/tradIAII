"""
Data Loader
============
Loads OHLCV data from CSV files.
Handles all timestamp formats, normalises column names,
validates required columns, and cleans bad rows.

All functions are stateless and return new DataFrames.
"""
import pandas as pd
import logging
from pathlib import Path
from backend.config.settings import DATA_FILES, TIMEFRAMES

logger = logging.getLogger(__name__)

REQUIRED_COLS = {"open", "high", "low", "close", "volume"}


def load_ohlcv(timeframe: str) -> pd.DataFrame:
    """
    Load and validate OHLCV data for a single timeframe.

    Raises:
        ValueError:      Unknown timeframe or missing columns.
        FileNotFoundError: CSV file does not exist.
    """
    if timeframe not in DATA_FILES:
        raise ValueError(
            f"Unknown timeframe '{timeframe}'. "
            f"Valid: {list(DATA_FILES.keys())}"
        )

    path = DATA_FILES[timeframe]
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Data file not found: {path}\n"
            f"Run: python -m backend.scripts.fetch_data"
        )

    df = pd.read_csv(path)

    # Normalise column names: strip whitespace, lowercase
    df.columns = df.columns.str.strip().str.lower()

    # Find timestamp column (any column containing 'time' or 'date')
    ts_col = next(
        (c for c in df.columns if "time" in c or "date" in c), None
    )
    if ts_col is None:
        raise ValueError(f"No timestamp column found in {path}")

    # Parse timestamp — handles Unix epoch (int/float) and ISO strings
    raw = df[ts_col]
    if pd.api.types.is_numeric_dtype(raw):
        unit = "ms" if raw.iloc[0] > 1e12 else "s"
        df[ts_col] = pd.to_datetime(raw, unit=unit)
    else:
        df[ts_col] = pd.to_datetime(raw, utc=True).dt.tz_localize(None)

    df = df.rename(columns={ts_col: "timestamp"})
    df = df.set_index("timestamp").sort_index()

    # Validate required columns exist
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {missing} in {path}")

    # Clean rows: remove zero prices, NaNs, duplicates
    df = df[df["close"] > 0]
    df = df.dropna(subset=list(REQUIRED_COLS))
    df = df[~df.index.duplicated(keep="first")]

    logger.info(
        f"[{timeframe:>3s}] Loaded {len(df):>7,} candles  "
        f"{df.index[0].date()} to {df.index[-1].date()}"
    )
    return df


def load_all_timeframes() -> dict[str, pd.DataFrame]:
    """
    Load every configured timeframe.
    Missing files produce a warning but do not crash the system.
    Returns: {"1h": df_1h, "4h": df_4h, ...}
    """
    result: dict[str, pd.DataFrame] = {}
    for tf in TIMEFRAMES:
        try:
            result[tf] = load_ohlcv(tf)
        except FileNotFoundError as e:
            logger.warning(str(e))
        except Exception as e:
            logger.error(f"[{tf}] Failed to load: {e}")
    return result
