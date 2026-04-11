"""
TradIA Signal Quality Diagnostic
==================================
Run from your project root:
    python diagnose.py

What this tells you:
  1. Raw signal count and label distribution
  2. Which gate conditions are most/least restrictive
  3. Win rate when each individual condition is met
  4. Whether quality_score actually predicts wins
"""
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '.')

print("Loading data and running pipeline...")

from backend.services.data_loader import load_all_timeframes
from backend.services.ict_strategy import run_ict_pipeline
from backend.services.state_machine import run_state_machine
from backend.services.multi_timeframe import merge_htf_into_ltf
from backend.services.feature_builder import build_features
from backend.services.label_generator import label_trades
from backend.config.settings import SIGNAL_TF, HTF_LIST

data = load_all_timeframes()
df   = data[SIGNAL_TF].copy()
df   = run_ict_pipeline(df)
df   = run_state_machine(df)
htf  = {tf: data[tf] for tf in HTF_LIST if tf in data}
if htf:
    from backend.services.multi_timeframe import merge_htf_into_ltf
    df = merge_htf_into_ltf(df, htf)
df = build_features(df)
df = label_trades(df)

sigs    = df[df["signal"].isin([0, 2])].copy()
labeled = sigs[sigs["ml_label"].notna()].copy()

print()
print("=" * 60)
print("  1. RAW SIGNAL & LABEL SUMMARY")
print("=" * 60)
total       = len(sigs)
n_labeled   = labeled["ml_label"].notna().sum()
n_wins      = (labeled["ml_label"] == 1).sum()
n_losses    = (labeled["ml_label"] == 0).sum()
n_timeout   = total - n_labeled
raw_wr      = n_wins / n_labeled * 100 if n_labeled > 0 else 0

print(f"  Total raw signals        : {total}")
print(f"  Labeled (TP or SL hit)   : {n_labeled}")
print(f"  Timed out (no outcome)   : {n_timeout}")
print(f"  WIN labels               : {n_wins}")
print(f"  LOSS labels              : {n_losses}")
print(f"  Raw label win rate       : {raw_wr:.1f}%")
print()

print("=" * 60)
print("  2. GATE CONDITION PASS RATES  (of labeled signals)")
print("=" * 60)

gate_conditions = [
    ("is_optimal_window", 1,    "eq"),
    ("volume_ratio",      1.1,  "gt"),
    ("atr_percentile",    0.30, "gt"),
    ("adx_14",            20,   "gt"),
    ("htf_bull_confluence", 1,  "gte"),
    ("htf_bear_confluence", 1,  "gte"),
]

for col, thresh, op in gate_conditions:
    if col not in labeled.columns:
        print(f"  {col:<30}: COLUMN MISSING")
        continue
    if op == "eq":
        mask = labeled[col] == thresh
    elif op == "gt":
        mask = labeled[col] > thresh
    else:
        mask = labeled[col] >= thresh
    n   = mask.sum()
    pct = n / len(labeled) * 100
    print(f"  {col:<30}: {n:>4} / {len(labeled)}  ({pct:.1f}%)")

print()

# All gate conditions combined (BUY direction)
if all(c in labeled.columns for c in ["is_optimal_window","volume_ratio","atr_percentile","adx_14","htf_bull_confluence"]):
    buy_sigs  = labeled[labeled["signal"] == 2]
    sell_sigs = labeled[labeled["signal"] == 0]

    def gate_mask(df_subset, direction):
        htf_col = "htf_bull_confluence" if direction == "BUY" else "htf_bear_confluence"
        return (
            (df_subset["is_optimal_window"] == 1) &
            (df_subset["volume_ratio"]      > 1.1) &
            (df_subset["atr_percentile"]    > 0.30) &
            (df_subset["adx_14"]            > 20) &
            (df_subset.get(htf_col, pd.Series(0, index=df_subset.index)) >= 1)
        )

    buy_pass  = gate_mask(buy_sigs,  "BUY").sum()
    sell_pass = gate_mask(sell_sigs, "SELL").sum()
    total_pass = buy_pass + sell_pass
    print(f"  ALL gates combined (BUY)   : {buy_pass} signals pass")
    print(f"  ALL gates combined (SELL)  : {sell_pass} signals pass")
    print(f"  ALL gates combined (TOTAL) : {total_pass} signals pass")
    print()

print("=" * 60)
print("  3. WIN RATE WHEN EACH CONDITION IS MET")
print("=" * 60)

check_cols = [
    ("is_optimal_window", 1,    "eq"),
    ("volume_ratio",      1.1,  "gt"),
    ("atr_percentile",    0.30, "gt"),
    ("adx_14",            20,   "gt"),
]

for col, thresh, op in check_cols:
    if col not in labeled.columns:
        print(f"  {col:<30}: COLUMN MISSING")
        continue
    mask   = (labeled[col] == thresh) if op == "eq" else (labeled[col] > thresh)
    subset = labeled[mask]
    if len(subset) == 0:
        print(f"  {col:<30}: no signals pass this condition")
        continue
    wr = (subset["ml_label"] == 1).sum() / len(subset) * 100
    print(f"  {col:<30}: {len(subset):>4} trades  WR={wr:.1f}%")

# Win rate with NO gate (baseline)
baseline_wr = (labeled["ml_label"] == 1).mean() * 100
print(f"  {'(no gate — baseline)':<30}: {len(labeled):>4} trades  WR={baseline_wr:.1f}%")
print()

print("=" * 60)
print("  4. QUALITY SCORE ANALYSIS")
print("=" * 60)

if "quality_score" in labeled.columns:
    qs = labeled["quality_score"]
    print(f"  Min    : {qs.min():.3f}")
    print(f"  Mean   : {qs.mean():.3f}")
    print(f"  Median : {qs.median():.3f}")
    print(f"  Max    : {qs.max():.3f}")
    print(f"  >= 0.40: {(qs >= 0.40).sum()} / {len(labeled)} ({(qs >= 0.40).mean()*100:.1f}%)")
    print()
    print("  Win rate by quality score bucket:")
    labeled["qs_bucket"] = pd.cut(
        qs, bins=[0, 0.20, 0.40, 0.60, 0.80, 1.01],
        labels=["0.00-0.20", "0.20-0.40", "0.40-0.60", "0.60-0.80", "0.80-1.00"]
    )
    for bucket, group in labeled.groupby("qs_bucket", observed=True):
        wr = (group["ml_label"] == 1).mean() * 100
        bar = "█" * int(wr / 5)
        print(f"    {bucket} : {len(group):>4} trades  WR={wr:5.1f}%  {bar}")
else:
    print("  quality_score column NOT FOUND — label_generator.py may not have been updated")

print()
print("=" * 60)
print("  5. DIRECTION BREAKDOWN")
print("=" * 60)

for direction, sig_val in [("BUY", 2), ("SELL", 0)]:
    subset = labeled[labeled["signal"] == sig_val]
    if len(subset) == 0:
        print(f"  {direction}: no signals")
        continue
    wr = (subset["ml_label"] == 1).mean() * 100
    print(f"  {direction:<6}: {len(subset):>4} trades  WR={wr:.1f}%")

print()
print("=" * 60)
print("  6. HTF CONFLUENCE DISTRIBUTION")
print("=" * 60)

for col in ["htf_bull_confluence", "htf_bear_confluence", "full_bull_confluence", "full_bear_confluence"]:
    if col in labeled.columns:
        vc = labeled[col].value_counts().sort_index()
        print(f"  {col}:")
        for val, cnt in vc.items():
            print(f"    {val}: {cnt} signals")
    else:
        print(f"  {col}: MISSING")

print()
print("Diagnostic complete.")
print("Share the full output above for analysis.")