# save as backend/scripts/diagnose_signals.py
from backend.services.data_loader import load_all_timeframes
from backend.services.ict_strategy import run_ict_pipeline
from backend.services.state_machine import run_state_machine
from backend.services.multi_timeframe import merge_htf_into_ltf
from backend.services.feature_builder import build_features
from backend.config.settings import HTF_LIST, SIGNAL_TF
import pandas as pd

data = load_all_timeframes()
df = data[SIGNAL_TF].copy()
df = run_ict_pipeline(df)
df = run_state_machine(df)
htf = {tf: data[tf] for tf in HTF_LIST if tf in data}
if htf:
    df = merge_htf_into_ltf(df, htf)
df = build_features(df)

signals = df[df["signal"].isin([0, 2])].copy()

lines = []
lines.append(f"Total candles: {len(df)}")
lines.append(f"Date range: {df.index[0]} to {df.index[-1]}")
lines.append(f"Total signals: {len(signals)}")
lines.append(f"Signal rate: {len(signals)/len(df):.2%}")
lines.append(f"BUY signals:  {(signals['signal'] == 2).sum()}")
lines.append(f"SELL signals: {(signals['signal'] == 0).sum()}")

lines.append(f"\nAvg ADX at signal: {signals['adx_14'].mean():.1f}")
lines.append(f"Avg ATR percentile: {signals['atr_percentile'].mean():.2f}")
lines.append(f"In killzone: {signals['is_optimal_window'].mean():.1%}")

lines.append(f"\nSL validity check:")
lines.append(f"  Signals with missing SL: {signals['signal_sl'].isna().sum()}")
lines.append(f"  Signals with SL == 0:    {(signals['signal_sl'] == 0).sum()}")
lines.append(f"  Signals with valid SL:   {signals['signal_sl'].notna().sum()}")

lines.append(f"\nHTF bias at BUY signals:")
buys = signals[signals['signal'] == 2]
if len(buys):
    lines.append(f"  h4_bias mean:  {buys['h4_bias'].mean():.2f}")
    lines.append(f"  d1_bias mean:  {buys['d1_bias'].mean():.2f}")
    lines.append(f"  htf_bull_confluence mean: {buys['htf_bull_confluence'].mean():.2f}")
else:
    lines.append("  No BUY signals found")

lines.append(f"\nHTF bias at SELL signals:")
sells = signals[signals['signal'] == 0]
if len(sells):
    lines.append(f"  h4_bias mean:  {sells['h4_bias'].mean():.2f}")
    lines.append(f"  d1_bias mean:  {sells['d1_bias'].mean():.2f}")
    lines.append(f"  htf_bear_confluence mean: {sells['htf_bear_confluence'].mean():.2f}")
else:
    lines.append("  No SELL signals found")

lines.append(f"\nRegime at signal time:")
lines.append(f"  regime_class distribution:\n{signals['regime_class'].value_counts().to_string()}")

lines.append(f"\nFirst 5 signals:")
cols = ['signal', 'close', 'signal_sl', 'h4_bias', 'd1_bias', 'adx_14', 'is_optimal_window']
available = [c for c in cols if c in signals.columns]
lines.append(signals[available].head(5).to_string())

lines.append(f"\nLast 5 signals:")
lines.append(signals[available].tail(5).to_string())

# Write to file
output = "\n".join(lines)
print(output)

with open("signal_diagnosis.txt", "w") as f:
    f.write(output)

print("\n--- Saved to signal_diagnosis.txt ---")