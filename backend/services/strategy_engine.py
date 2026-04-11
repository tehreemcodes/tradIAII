# backend/services/strategy_engine.py

import logging
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from backend.config.settings import MIN_CONFIDENCE, MODEL_DIR
from backend.services.market_regime import MarketRegimeDetector
from backend.services.scalp_strategy import ScalpStrategy, StrategySignal
from backend.services.trend_strategy import TrendStrategy

logger = logging.getLogger(__name__)


class StrategyEngine:

    def __init__(self):
        self.regime_detector = MarketRegimeDetector()
        self.scalp_strategy = ScalpStrategy()
        self.trend_strategy = TrendStrategy()
        self.model = None
        self.scaler = None
        self.features = None

        self._load_model()

    def _load_model(self):
        try:
            self.model = joblib.load(MODEL_DIR / "ict_model_15m.pkl")
            self.scaler = joblib.load(MODEL_DIR / "scaler_15m.pkl")
            self.features = joblib.load(MODEL_DIR / "features_15m.pkl")
            logger.info("ML model loaded successfully")
        except Exception as e:
            logger.warning(f"ML model not loaded: {e}")

    def evaluate(self, df: pd.DataFrame, row: pd.Series, timeframe="15m") -> StrategySignal:

        # 1. Regime
        regime_result = self.regime_detector.classify(df, row)
        strategy_type = regime_result.strategy_type

        if strategy_type == "none":
            return StrategySignal.no_trade("Low volatility", regime_result.regime.value)

        # 2. Strategy
        if strategy_type == "trend":
            signal = self.trend_strategy.evaluate(df, row, regime_result.regime.value)
        else:
            signal = self.scalp_strategy.evaluate(df, row, regime_result.regime.value)

        if signal.signal == "NO TRADE":
            return signal

        # 3. ML Prediction (FIXED)
        try:
            if self.model is not None:
                X = row[self.features].to_frame().T
                X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
                X_scaled = self.scaler.transform(X)

                prob = self.model.predict_proba(X_scaled)[0][1]
                signal.confidence = prob

                if prob < MIN_CONFIDENCE:
                    signal.executable = False
                    signal.reason = f"{regime_result.regime.value} | {regime_result.reason} | ML={signal.confidence:.2f}"
                else:
                    signal.executable = True
            else:
                signal.executable = True
                signal.confidence = 0.5

        except Exception as e:
            logger.warning(f"ML failed: {e}")
            signal.executable = True
            signal.confidence = 0.5

        return signal