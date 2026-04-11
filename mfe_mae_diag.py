import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, '.')
from backend.services.data_loader import load_all_timeframes
from backend.services.ict_strategy import run_ict_pipeline
from backend.services.state_machine import run_state_machine
from backend.services.multi_timeframe import merge_htf_into_ltf
from backend.services.feature_builder import build_features
from backend.services.label_generator import label_trades
from backend.config.settings import SIGNAL_TF, HTF_LIST, REWARD_RATIO

def run():
    print("Loading data...")
    data = load_all_timeframes()
    df = data[SIGNAL_TF].copy()
    df = run_ict_pipeline(df)
    df = run_state_machine(df)
    htf = {tf: data[tf] for tf in HTF_LIST if tf in data}
    if htf:
        from backend.services.multi_timeframe import merge_htf_into_ltf
        df = merge_htf_into_ltf(df, htf)
    df = build_features(df)
    df = label_trades(df)

    sigs = df[df["signal"].isin([0, 2])].copy()
    trades = []
    
    # Simulate backtest gate checks
    for ts, row in sigs.iterrows():
        is_optimal = row.get("is_optimal_window", 0) == 1
        q_score    = row.get("quality_score", 0.0)
        passes_gate = is_optimal and q_score >= 0.55
        if not passes_gate:
            continue
            
        direction = "BUY" if row["signal"] == 2 else "SELL"
        entry = float(row["close"])
        raw_sl = row.get("signal_sl", np.nan)
        if np.isnan(raw_sl) or raw_sl <= 0:
            continue
            
        if direction == "BUY":
            sl = raw_sl * (1 - 0.001)  # SL_BUFFER_PCT = 0.001
            if sl >= entry: continue
        else:
            sl = raw_sl * (1 + 0.001)
            if sl <= entry: continue
            
        sl_dist = abs(entry - sl)
        if sl_dist < 1e-8:
            continue
            
        trades.append({
            "ts": ts,
            "direction": direction,
            "entry": entry,
            "sl": sl,
            "sl_dist": sl_dist,
            "idx": df.index.get_loc(ts)
        })

    print(f"Found {len(trades)} trades passing the hard gate.")

    mfe_list = []
    mae_list = []
    
    for t in trades[:13]: # only taking the exactly 13 trades we simulated
        direction = t["direction"]
        entry = t["entry"]
        sl = t["sl"]
        sl_dist = t["sl_dist"]
        idx = t["idx"]
        
        max_favorable = entry
        max_adverse = entry
        
        # Scan forward for MFE and MAE before SL is hit
        for forward_idx in range(idx + 1, min(idx + 300, len(df))): 
            frow = df.iloc[forward_idx]
            high = frow["high"]
            low = frow["low"]
            
            if direction == "BUY":
                if low <= sl: # Hit SL
                    max_adverse = min(max_adverse, sl)
                    if high > max_favorable: max_favorable = high
                    break
                else:
                    if high > max_favorable: max_favorable = high
                    if low < max_adverse: max_adverse = low
            else:
                if high >= sl: # Hit SL
                    max_adverse = max(max_adverse, sl)
                    if low < max_favorable: max_favorable = low
                    break
                else:
                    if low < max_favorable: max_favorable = low
                    if high > max_adverse: max_adverse = high

        if direction == "BUY":
            best_pnl = max_favorable - entry
            worst_pnl = entry - max_adverse
        else:
            best_pnl = entry - max_favorable
            worst_pnl = max_adverse - entry
            
        mfe_r_trade = best_pnl / sl_dist if sl_dist > 0 else 0
        mae_r_trade = worst_pnl / sl_dist if sl_dist > 0 else 0
        
        mfe_list.append(mfe_r_trade)
        mae_list.append(mae_r_trade)
        
    print("\n=== MFE/MAE ANALYSIS (R-Multiples) ===")
    if mfe_list:
        print(f"Average MFE (R): {np.mean(mfe_list):.2f}R")
        print(f"Max MFE (R): {np.max(mfe_list):.2f}R")
        print(f"Average MAE (R): {np.mean(mae_list):.2f}R")
        
        reaching_05r = sum(1 for x in mfe_list if x >= 0.5)
        reaching_10r = sum(1 for x in mfe_list if x >= 1.0)
        reaching_15r = sum(1 for x in mfe_list if x >= 1.5)
        reaching_20r = sum(1 for x in mfe_list if x >= 2.0)
        reaching_30r = sum(1 for x in mfe_list if x >= 3.0)
        
        print(f"\nTrades reaching >= 0.5R: {reaching_05r}/{len(mfe_list)} ({(reaching_05r/len(mfe_list)*100):.1f}%)")
        print(f"Trades reaching >= 1.0R: {reaching_10r}/{len(mfe_list)} ({(reaching_10r/len(mfe_list)*100):.1f}%)")
        print(f"Trades reaching >= 1.5R: {reaching_15r}/{len(mfe_list)} ({(reaching_15r/len(mfe_list)*100):.1f}%)")
        print(f"Trades reaching >= 2.0R: {reaching_20r}/{len(mfe_list)} ({(reaching_20r/len(mfe_list)*100):.1f}%)")
        print(f"Trades reaching >= 3.0R: {reaching_30r}/{len(mfe_list)} ({(reaching_30r/len(mfe_list)*100):.1f}%)")
        
        print("\nIndividual Trade Breakdown:")
        for i, (mfe, mae, t) in enumerate(zip(mfe_list, mae_list, trades[:13])):
            print(f"Trade {i+1}: {t['direction']} @ {t['entry']:.2f} | SL_Dist: {t['sl_dist']:.2f} | MFE: {mfe:.2f}R | MAE: {mae:.2f}R")

run()
