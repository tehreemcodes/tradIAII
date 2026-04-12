# backend/services/strategy_engine.py

import logging
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from backend.config.settings import MIN_CONFIDENCE, MODEL_DIR, RISK_PCT
from backend.services.market_regime import MarketRegimeDetector
from backend.services.scalp_strategy import ScalpStrategy, StrategySignal
from backend.services.trend_strategy import TrendStrategy

logger = logging.getLogger(__name__)

# Module-level gate result -- populated by every evaluate() call,
# read by GET /api/gate-status without re-running the pipeline.
_last_gate_result: dict = {
    "timestamp":     None,
    "regime":        "UNKNOWN",
    "strategy_type": "none",
    "gate_regime":   "UNKNOWN",
    "gate_strategy": "ok",
    "gate_ml":       "ok",
    "signal":        "NO TRADE",
    "confidence":    0.0,
    "executable":    False,
    "killed":        True,
    "reason":        "",
}


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

        gate_regime = "UNKNOWN"
        gate_strategy = "ok"
        gate_ml = "ok"
        killed = False

        # 1. Regime
        regime_result = self.regime_detector.classify(df, row)
        strategy_type = regime_result.strategy_type
        gate_regime = regime_result.regime.value

        if strategy_type == "none":
            killed = True
            gate_strategy = "NO TRADE"
            signal = StrategySignal.no_trade("Low volatility", regime_result.regime.value)
            logger.info(f"[GATE] ICT=ok | regime={gate_regime} | strategy={gate_strategy} | ML=skipped | KILLED")
            import backend.services.strategy_engine as _self_mod
            from datetime import datetime, timezone
            _self_mod._last_gate_result = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "regime": gate_regime, "strategy_type": strategy_type,
                "gate_regime": gate_regime, "gate_strategy": gate_strategy,
                "gate_ml": "skipped", "signal": "NO TRADE",
                "confidence": 0.0, "executable": False, "killed": True,
                "reason": "Low volatility / regime filter",
            }
            return signal

        # 2. Strategy
        if strategy_type == "trend":
            signal = self.trend_strategy.evaluate(df, row, regime_result.regime.value)
        else:
            signal = self.scalp_strategy.evaluate(df, row, regime_result.regime.value)

        if signal.signal == "NO TRADE":
            killed = True
            gate_strategy = "NO TRADE"
            logger.info(f"[GATE] ICT=ok | regime={gate_regime} | strategy={gate_strategy} | ML=skipped | KILLED")
            import backend.services.strategy_engine as _self_mod
            from datetime import datetime, timezone
            _self_mod._last_gate_result = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "regime": gate_regime, "strategy_type": strategy_type,
                "gate_regime": gate_regime, "gate_strategy": gate_strategy,
                "gate_ml": "skipped", "signal": "NO TRADE",
                "confidence": 0.0, "executable": False, "killed": True,
                "reason": signal.reason or "Strategy gate rejected",
            }
            return signal

        # 3. ML Prediction (FIXED)
        try:
            if self.model is not None:
                X = row[self.features].to_frame().T
                X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
                X_scaled = self.scaler.transform(X)

                prob = self.model.predict_proba(X_scaled)[0][1]
                signal.confidence = prob

                if prob >= MIN_CONFIDENCE * 1.2:
                    signal.executable = True
                    signal.risk_pct_override = RISK_PCT
                elif prob >= MIN_CONFIDENCE:
                    signal.executable = True
                    signal.risk_pct_override = RISK_PCT * 0.7
                elif prob >= MIN_CONFIDENCE * 0.8:
                    signal.executable = True
                    signal.risk_pct_override = RISK_PCT * 0.5
                else:
                    signal.executable = False
                    signal.reason = f"{regime_result.regime.value} | {regime_result.reason} | ML={prob:.2f} (below {MIN_CONFIDENCE * 0.8})"
                    gate_ml = f"{prob:.2f} (below {MIN_CONFIDENCE * 0.8})"
                    killed = True
            else:
                # Explicit path for missing model
                signal.executable = True
                signal.confidence = 0.5
                gate_ml = "skipped (no model loaded)"
                logger.info(f"Model not loaded, skipping ML check. Executable allowed from strategy alone.")

        except Exception as e:
            logger.warning(f"ML failed: {e}")
            signal.executable = True
            signal.confidence = 0.5
            gate_ml = f"error ({e})"

        if killed:
            logger.info(f"[GATE] ICT=ok | regime={gate_regime} | strategy=ok | ML={gate_ml} | KILLED")
        else:
            logger.info(f"[GATE] ICT=ok | regime={gate_regime} | strategy=ok | ML={gate_ml} | PASSED")

        # Update module-level gate result for /api/gate-status
        import backend.services.strategy_engine as _self_mod
        from datetime import datetime, timezone
        _self_mod._last_gate_result = {
            "timestamp":     datetime.now(timezone.utc).isoformat(),
            "regime":        gate_regime,
            "strategy_type": strategy_type,
            "gate_regime":   gate_regime,
            "gate_strategy": gate_strategy,
            "gate_ml":       gate_ml,
            "signal":        signal.signal,
            "confidence":    round(float(signal.confidence), 4),
            "executable":    signal.executable,
            "killed":        killed,
            "reason":        signal.reason or "",
        }

        return signal

    def dry_run(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        results = []
        for idx in range(len(df)):
            row = df.iloc[idx]
            # Use data up to the current candle to avoid look-ahead bias in regime classification
            sub_df = df.iloc[:idx + 1]
            signal = self.evaluate(sub_df, row, timeframe)
            
            kill_reason = signal.reason if (not signal.executable or signal.signal == "NO TRADE") else ""

            results.append({
                "timestamp": row.name if df.index.name else idx,
                "regime": getattr(self.regime_detector, "_confirmed_regime", "UNKNOWN"),
                "strategy_signal": signal.signal,
                "ml_prob": getattr(signal, "confidence", 0.0),
                "executable": signal.executable,
                "kill_reason": kill_reason
            })
            
        return pd.DataFrame(results)