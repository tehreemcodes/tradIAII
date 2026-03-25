import pandas as pd
import ccxt
import joblib
from backend.services.ict_strategy import run_ict_pipeline
from backend.services.state_machine import run_state_machine
from backend.services.feature_builder import build_features
from backend.config.settings import MODEL_PATH, FEATURES_PATH, SCALER_PATH, MIN_CONFIDENCE

print("Fetching data...")
ex = ccxt.binance()
raw = ex.fetch_ohlcv("BTC/USDT", "15m", limit=300)
df = pd.DataFrame(raw, columns=["timestamp","open","high","low","close","volume"])
df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
df = df.set_index("timestamp").sort_index()

df = run_ict_pipeline(df)
df = run_state_machine(df)

print("Running ML annotation...")
if MODEL_PATH.exists() and FEATURES_PATH.exists() and SCALER_PATH.exists():
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)
    scaler = joblib.load(SCALER_PATH)

    X_df = build_features(df.copy())
    
    df["executable"] = False
    df["reject_reason"] = "No signal"
    df["ml_confidence"] = 0.0

    print("Iterating", len(X_df), "rows in X_df")
    for ts, row in X_df.iterrows():
        sig = int(row.get("signal", 1))
        if sig != 1 and ts in df.index:
            X_row = X_df.loc[[ts], features]
            if not X_row.isna().any().any():
                X_scaled = scaler.transform(X_row)
                probs = model.predict_proba(X_scaled)[0]
                win_prob = probs[1] if sig == 2 else probs[0]
                
                df.at[ts, "ml_confidence"] = round(float(win_prob), 4)
                
                full_conf = bool(row.get("full_bull_confluence", 0) or row.get("full_bear_confluence", 0))
                h4_bias = int(row.get("h4_bias", 0))
                sig_dir = 1 if sig == 2 else -1
                
                if full_conf:
                    dyn_thresh = MIN_CONFIDENCE - 0.05
                elif h4_bias == sig_dir:
                    dyn_thresh = MIN_CONFIDENCE
                else:
                    dyn_thresh = MIN_CONFIDENCE + 0.10
                    
                if win_prob >= dyn_thresh:
                    df.at[ts, "executable"] = True
                    df.at[ts, "reject_reason"] = "Executable"
                else:
                    df.at[ts, "reject_reason"] = f"ML Confidence {win_prob:.2f} < {dyn_thresh:.2f}"
            else:
                print(f"{ts}: NaN found in features")

print("Formatting response...")
candles = []
for ts, row in df.iterrows():
    if int(row.get("signal", 1)) != 1:
        c = {
            "time": ts,
            "signal": int(row.get("signal", 1)),
            "executable": bool(row.get("executable", False)) if "executable" in row else False,
            "reject_reason": str(row.get("reject_reason", "No signal")) if "reject_reason" in row else "No signal",
        }
        candles.append(c)

print("Output candles with signals:")
for c in candles:
    print(c)
    
