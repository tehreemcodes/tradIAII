import joblib
import numpy as np
import pandas as pd
import sys

sys.path.insert(0, 'C:/Users/User/Downloads/tradIA_complete')

# Load 15m model artifacts
model    = joblib.load('backend/models/ict_model_15m.pkl')
scaler   = joblib.load('backend/models/scaler_15m.pkl')
features = joblib.load('backend/models/features_15m.pkl')

print("=== MODEL INSPECTION ===")
print(f"Model type       : {type(model).__name__}")
print(f"Feature count    : {len(features)}")
params = model.get_params()
print(f"scale_pos_weight : {params.get('scale_pos_weight', 'N/A')}")
print(f"n_estimators     : {params.get('n_estimators', 'N/A')}")
print(f"best_iteration   : {getattr(model, 'best_iteration', 'N/A')}")
print(f"Classes          : {model.classes_}")

# Probe 1: Random Gaussian inputs (simulate any market context)
n_probes = 1000
np.random.seed(42)
X_probe = pd.DataFrame(np.random.randn(n_probes, len(features)), columns=features)
X_probe_sc = pd.DataFrame(scaler.transform(X_probe), columns=features)
probas = model.predict_proba(X_probe_sc)[:, 1]

print("\n=== PROBABILITY DISTRIBUTION (random Gaussian inputs, n=1000) ===")
print(f"Min    : {probas.min():.4f}")
print(f"Max    : {probas.max():.4f}")
print(f"Mean   : {probas.mean():.4f}")
print(f"Median : {np.median(probas):.4f}")
print(f"Std    : {probas.std():.4f}")
for p in [5, 10, 25, 75, 90, 95, 99]:
    print(f"P{p:<3}    : {np.percentile(probas, p):.4f}")

print("\n=== THRESHOLD PASS RATE (random inputs) ===")
for t in [0.05, 0.08, 0.10, 0.12, 0.15, 0.17, 0.18, 0.19, 0.20, 0.22, 0.25, 0.30, 0.40, 0.50]:
    pct = (probas >= t).mean() * 100
    print(f"  >= {t:.2f} : {pct:5.1f}% pass")

# Probe 2: All-zero inputs (neutral / no pattern context)
X_zero = pd.DataFrame(np.zeros((1, len(features))), columns=features)
X_zero_sc = pd.DataFrame(scaler.transform(X_zero), columns=features)
p_zero = model.predict_proba(X_zero_sc)[0, 1]
print(f"\n=== ZERO VECTOR PROBE ===")
print(f"Confidence on all-zeros input: {p_zero:.4f}")

# Probe 3: Positive signal features (simulate a real BUY signal)
X_bull = pd.DataFrame(np.zeros((1, len(features))), columns=features)
bull_feature_overrides = {
    'swing_low': 1.0,
    'bull_cisd': 1.0,
    'bull_fvg':  1.0,
    'h4_bias':   1.0,
    'd1_bias':   1.0,
    'full_bull_confluence': 1.0,
    'htf_bull_confluence': 1.0,
    'is_bullish_candle': 1.0,
    'cisd_body_ratio': 2.0,
    'cisd_vol_ratio': 1.5,
    'is_optimal_window': 1.0,
}
for k, v in bull_feature_overrides.items():
    if k in X_bull.columns:
        X_bull[k] = v
X_bull_sc = pd.DataFrame(scaler.transform(X_bull), columns=features)
p_bull = model.predict_proba(X_bull_sc)[0, 1]
print(f"\n=== IDEAL BULL SIGNAL PROBE ===")
print(f"Confidence on strong BUY input: {p_bull:.4f}")

# Probe 4: Negative signal features (simulate a real SELL signal)
X_bear = pd.DataFrame(np.zeros((1, len(features))), columns=features)
bear_feature_overrides = {
    'swing_high': 1.0,
    'bear_cisd': 1.0,
    'bear_fvg':  1.0,
    'h4_bias':   -1.0,
    'd1_bias':   -1.0,
    'full_bear_confluence': 1.0,
    'htf_bear_confluence': 1.0,
    'is_bullish_candle': 0.0,
    'cisd_body_ratio': 2.0,
    'cisd_vol_ratio': 1.5,
}
for k, v in bear_feature_overrides.items():
    if k in X_bear.columns:
        X_bear[k] = v
X_bear_sc = pd.DataFrame(scaler.transform(X_bear), columns=features)
p_bear = model.predict_proba(X_bear_sc)[0, 1]
print(f"Confidence on strong SELL input: {p_bear:.4f}")

print("\nDone.")
