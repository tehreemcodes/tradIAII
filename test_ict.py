import pandas as pd
import ccxt
from backend.services.ict_strategy import run_ict_pipeline
from backend.services.state_machine import run_state_machine

print("Fetching data...")
ex = ccxt.binance()
raw = ex.fetch_ohlcv("BTC/USDT", "15m", limit=300)
df = pd.DataFrame(raw, columns=["timestamp","open","high","low","close","volume"])
df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
df = df.set_index("timestamp").sort_index()

print("Running pipeline...")
df = run_ict_pipeline(df)
df = run_state_machine(df)

signals = df[df["signal"].isin([0, 2])]
print(f"Total signals generated: {len(signals)}")
for ts, row in signals.iterrows():
    print(f"{ts}: signal={row['signal']}")
