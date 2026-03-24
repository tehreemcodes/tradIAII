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
EXCHANGE   = "bybit"

# ── Timeframes ───────────────────────────────────────────────
TIMEFRAMES = ["15m", "1h", "4h", "1d"]
SIGNAL_TF  = "1h"
HTF_LIST   = ["4h", "1d"]
LTF        = "15m"

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
CISD_BODY_MULT  = 1.5
CISD_VOL_MULT   = 1.3
CISD_LOOKBACK   = 20
PATTERN_WINDOW  = 20
FVG_CANDLES     = 2

# ── Risk Management ──────────────────────────────────────────
INITIAL_CAPITAL    = 10_000.0

# FIX: Reduced from 10% to 1% — professional standard
# 10% risk per trade is reckless; 0.5-2% is industry standard
RISK_PCT           = 0.01      # 1% risk per trade

REWARD_RATIO       = 2.0       # 1:2 minimum R:R
COOLDOWN_MINUTES   = 5
MAX_OPEN_TRADES    = 1
MAX_NOTIONAL_MULT  = 10.0

# FIX: Minimum SL distance as % of price
# Prevents degenerate signals with near-zero SL (causes position size explosion)
MIN_SL_PCT         = 0.003     # SL must be >= 0.3% of entry price away
SL_BUFFER_PCT = 0.0008  # 0.08% beyond swing — adjust per asset

# FIX: Slippage and fees (realistic crypto exchange costs)
SLIPPAGE_PCT       = 0.0005    # 0.05% slippage on entry and exit
FEE_PCT            = 0.0006    # 0.06% taker fee per side (Bybit standard)
ROUND_TRIP_COST    = (SLIPPAGE_PCT + FEE_PCT) * 2   # both sides

# FIX: Equity protection rules
MAX_DRAWDOWN_STOP  = 0.20      # Stop trading if drawdown exceeds 20%
MAX_DAILY_LOSS_PCT = 0.03      # Stop trading for the day after 3% daily loss

# ── Trade Labeling ───────────────────────────────────────────
LABEL_FORWARD  = {
    "15m": 30,
    "1h":  30,
    "4h":  20,
    "1d":  10,
}
LABEL_WIN      = 1
LABEL_LOSS     = 0

# ── Model Artifacts ──────────────────────────────────────────
MODEL_PATH    = MODEL_DIR / "ict_model.pkl"
FEATURES_PATH = MODEL_DIR / "features.pkl"
SCALER_PATH   = MODEL_DIR / "scaler.pkl"
MIN_CONFIDENCE = 0.60

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

# ── API ───────────────────────────────────────────────────────
API_HOST         = "0.0.0.0"
API_PORT         = 8000

_frontend_url = os.getenv("FRONTEND_URL", "")
API_CORS_ORIGINS = list(filter(None, [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    _frontend_url,                      # Set this to your Vercel URL in Railway env vars
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
MAX_OPEN_POSITIONS    = 1