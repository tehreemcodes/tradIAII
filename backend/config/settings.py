"""
TradIA — Central Configuration
================================
Single source of truth. Zero hardcoded values anywhere else.
"""
from dotenv import load_dotenv
load_dotenv()
from pathlib import Path
import os

# ── Directory Layout ─────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parents[1]
DATA_DIR   = BASE_DIR / "data"
MODEL_DIR  = BASE_DIR / "models"
LOG_DIR    = BASE_DIR / "logs"

# ── Logging ──────────────────────────────────────────────────
LOG_FILE   = LOG_DIR / "tradia.log"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_LEVEL  = "INFO"

# ── Exchange & Symbol ────────────────────────────────────────
SYMBOL     = "BTC/USDT"
# AUDIT FIX BUG#1: Removed dead EXCHANGE = "bybit" — env override on L145 is the real value

# ── Timeframes ───────────────────────────────────────────────
TIMEFRAMES = ["15m", "1h", "4h", "1d"]
SIGNAL_TF  = "15m"
HTF_LIST   = ["1h", "4h", "1d"]
LTF        = "1m"

TF_LABELS  = {
    "15m": "15 Minute",
    "1h":  "1 Hour",
    "4h":  "4 Hour",
    "1d":  "Daily",
}

TF_PREFIX  = {
    "15m": "m15",
    "1h":  "h1",
    "4h":  "h4",
    "1d":  "d1",
}

# ── Data File Paths ───────────────────────────────────────────
DATA_FILES = {
    "15m": DATA_DIR / "btc_15m.csv",
    "1h":  DATA_DIR / "btc_1h.csv",
    "4h":  DATA_DIR / "btc_4h.csv",
    "1d":  DATA_DIR / "btc_1d.csv",
}

FETCH_START = {
    "15m": "2022-01-01T00:00:00Z",
    "1h":  "2020-01-01T00:00:00Z",
    "4h":  "2019-01-01T00:00:00Z",
    "1d":  "2018-01-01T00:00:00Z",
}

# ── ICT Strategy Parameters ──────────────────────────────────
SWING_LOOKBACK  = 2
CISD_BODY_MULT  = 2.0      # was 1.5 — require 2x+ avg body for true ICT displacement
CISD_VOL_MULT   = 1.5      # was 1.3 — require higher volume on CISD candles
CISD_LOOKBACK   = 20
PATTERN_WINDOW  = 15       # was 30 — max 15 candles (3.75h on 15m) for Swing→CISD→FVG
FVG_CANDLES     = 2

# ── Risk Management ──────────────────────────────────────────
INITIAL_CAPITAL    = 10_000.0

# FIX: Reduced from 10% to 1% — professional standard
# 10% risk per trade is reckless; 0.5-2% is industry standard
RISK_PCT           = 0.01      # was 0.02 — 1% risk per trade (professional standard)

REWARD_RATIO       = 2.0       # 1:3 minimum R:R (legacy default, overridden per strategy)
COOLDOWN_MINUTES   = 5
MAX_OPEN_TRADES    = 3         # was 1 — allow concurrent scalp + trend
MAX_NOTIONAL_MULT  = 10.0
DEFAULT_LEVERAGE   = 3         # was 5 — reduce leverage until win rate is proven

# FIX: Minimum SL distance as % of price
# Prevents degenerate signals with near-zero SL (causes position size explosion)
MIN_SL_PCT         = 0.001     # SL must be >= 0.1% of entry price away
SL_BUFFER_PCT = 0.0008  # 0.08% beyond swing — adjust per asset

# FIX: Slippage and fees (realistic crypto exchange costs)
SLIPPAGE_PCT       = 0.0005    # 0.05% slippage on entry and exit
FEE_PCT            = 0.0006    # 0.06% taker fee per side (Bybit standard)
ROUND_TRIP_COST    = (SLIPPAGE_PCT + FEE_PCT) * 2   # both sides

# FIX: Equity protection rules
MAX_DRAWDOWN_STOP  = 0.40      # was 0.15 — raised for realistic backtesting; halt at 40% drawdown
MAX_DAILY_LOSS_PCT = 0.15      # was 0.05 — raised for backtest diagnosis; stop day at 15% loss

# ── Dual-Strategy Configuration ──────────────────────────────
# Scalp Strategy (1R — short reaction trades)
SCALP_RR              = 1.0       # 1:1 risk:reward
SCALP_RISK_PCT        = 0.005     # 0.5% risk per scalp trade (half of trend)
SCALP_BE_THRESHOLD    = 0.5       # move SL to breakeven at 0.5R profit
SCALP_QUALITY_MIN     = 0.40      # lower quality bar → more trades
SCALP_MAX_CONCURRENT  = 2         # up to 2 simultaneous scalp positions
SCALP_COOLDOWN_MIN    = 3         # minutes between scalp entries

# Trend Strategy (2R — continuation trades)
TREND_RR              = 2.0       # 1:2 risk:reward
TREND_RISK_PCT        = 0.01      # 1% risk per trend trade
TREND_BE_THRESHOLD    = 1.0       # move SL to breakeven at 1R profit
TREND_QUALITY_MIN     = 0.50      # higher quality bar → fewer, better trades
TREND_MAX_CONCURRENT  = 1         # max 1 trend position at a time
TREND_COOLDOWN_MIN    = 10        # minutes between trend entries

# ── Market Regime Detection Thresholds ───────────────────────
REGIME_ADX_TRENDING   = 25        # ADX above this = trending
REGIME_ADX_RANGING    = 20        # ADX below this = ranging
REGIME_VOL_HIGH       = 0.75      # ATR/vol percentile above = high volatility
REGIME_VOL_LOW        = 0.25      # ATR/vol percentile below = low volatility
REGIME_EMA_SLOPE_MIN  = 0.0005    # EMA8 slope threshold for trend confirmation
REGIME_LOOKBACK       = 20        # candles to evaluate for HH/HL structure

# ── Ensemble Model Configuration ─────────────────────────────
ENSEMBLE_WEIGHTS = {
    "xgboost":             0.50,
    "random_forest":       0.30,
    "logistic_regression": 0.20,
}

# ── Trade Labeling ───────────────────────────────────────────
LABEL_FORWARD  = {
    "15m": 60,
    "1h":  30,
    "4h":  20,
    "1d":  10,
}
LABEL_WIN      = 1
LABEL_LOSS     = 0

# ── Model Artifacts ──────────────────────────────────────────
MODEL_PATH    = MODEL_DIR / f"ict_model_{SIGNAL_TF}.pkl"   # ict_model_15m.pkl
FEATURES_PATH = MODEL_DIR / f"features_{SIGNAL_TF}.pkl"   # features_15m.pkl
SCALER_PATH   = MODEL_DIR / f"scaler_{SIGNAL_TF}.pkl"     # scaler_15m.pkl
MIN_CONFIDENCE = 0.35          # was 0.50 — lowered for diagnosis; model confidence threshold

# Per-strategy model paths (used by ensemble gate)
SCALP_MODEL_DIR  = MODEL_DIR   # scalp models: scalp_ensemble_{tf}.pkl
TREND_MODEL_DIR  = MODEL_DIR   # trend models: trend_ensemble_{tf}.pkl

# ── XGBoost Hyperparameters ───────────────────────────────────
# FIX: Renamed from LGBM_PARAMS. Now uses XGBoost-compatible keys.
# scale_pos_weight is dynamically calculated during training (neg/pos ratio).
XGBOOST_PARAMS = {
    "n_estimators":       300,
    "learning_rate":      0.05,
    "max_depth":          6,
    "subsample":          0.8,
    "colsample_bytree":   0.8,
    "reg_alpha":          0.1,
    "reg_lambda":         0.1,
    "scale_pos_weight":   1.0,    # placeholder — calculated at train time
    "random_state":       42,
    "verbosity":          0,
    "eval_metric":        "logloss",
}
XGBOOST_EARLY_STOPPING = 20
TRAIN_SPLIT             = 0.80

# ── Signal Quality Gate Constants (Task 4) ────────────────────
# Minimum conditions for any trade to be considered
MIN_VOLUME_RATIO     = 1.1     # volume must be 10% above 20-period average
MIN_ATR_PERCENTILE   = 0.30    # market must have some volatility present
MIN_ADX              = 20      # some trend must exist
MIN_HTF_CONFLUENCE   = 1       # at least one HTF timeframe must agree

# ── API ───────────────────────────────────────────────────────
API_HOST         = "0.0.0.0"
API_PORT         = 8000

_frontend_url = os.getenv("FRONTEND_URL", "")
API_CORS_ORIGINS = list(filter(None, [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://187.127.103.154",
    "http://187.127.103.154:3000",
    "http://187.127.103.154:8000",
    _frontend_url,
    "*",   # fallback — restrict in production by removing this line
]))

# ── Live Trading ──────────────────────────────────────────────
LIVE_TRADING_ENABLED  = True
PAPER_MODE            = os.getenv("PAPER_MODE", "true").lower() == "true"
BINANCE_TESTNET       = os.getenv("BINANCE_TESTNET", "false").lower() == "true"
EXCHANGE              = os.getenv("EXCHANGE", "binance")
EXCHANGE_API_KEY      = os.getenv("EXCHANGE_API_KEY", "")
EXCHANGE_API_SECRET   = os.getenv("EXCHANGE_API_SECRET", "")
EXCHANGE_TESTNET      = False
TRADE_LOG_PATH        = LOG_DIR / "live_trades.json"
ANALYTICS_DB_PATH     = DATA_DIR / "analytics.db"
ORDER_TYPE            = "market"   # "market" or "limit"
MAX_OPEN_POSITIONS    = 1

# ── Binance Demo API ──────────────────────────────────────────
# Always points to demo-fapi — do NOT switch to mainnet without explicit override
BINANCE_BASE_URL = os.getenv("BINANCE_BASE_URL", "https://demo-fapi.binance.com")
assert "demo-fapi.binance.com" in BINANCE_BASE_URL, (
    f"Safety: BINANCE_BASE_URL must point to demo-fapi, got: {BINANCE_BASE_URL}"
)

# ── Limit Entry Orders ────────────────────────────────────────
USE_LIMIT_ENTRY          = True    # Use limit orders at book price; fall back to market on timeout
ENTRY_LIMIT_TIMEOUT_SEC  = 30      # Seconds to wait for limit fill before cancelling and going market

# ── Daily Trade Governor ──────────────────────────────────────
MAX_TRADES_PER_DAY = 6             # Maximum trades allowed in a single calendar day

# ── EV Gate ──────────────────────────────────────────────────
EXPECTED_WIN_RATE = 0.35           # Conservative win-rate assumption for EV filtering
FEE_RATE_TAKER    = 0.0005         # 0.05% taker fee per side
FEE_RATE_MAKER    = 0.0002         # 0.02% maker fee per side