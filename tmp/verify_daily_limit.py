import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
import json

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parents[1]
sys.path.append(str(PROJECT_ROOT))

from backend.services.trade_tracker import TradeTracker
from backend.config.settings import INITIAL_CAPITAL, MAX_DAILY_LOSS_PCT

def test_daily_pnl():
    print("Testing get_daily_pnl()...")
    test_log = Path("test_trades.json")
    if test_log.exists(): os.remove(test_log)
    
    tracker = TradeTracker(path=test_log)
    
    # Add a closed trade for today
    now_utc = datetime.now(timezone.utc).isoformat()
    tracker.open_trade({
        "id": "trade1", "direction": "BUY", "entry_price": 50000, 
        "sl_price": 49000, "tp_price": 52000, "size": 0.1, "paper": True
    })
    tracker.close_trade("trade1", "SL", 49000, -100.0, closed_at=now_utc)
    
    # Add a closed trade for yesterday
    yesterday_utc = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    tracker.open_trade({
        "id": "trade2", "direction": "BUY", "entry_price": 50000, 
        "sl_price": 49000, "tp_price": 52000, "size": 0.1, "paper": True
    })
    tracker.close_trade("trade2", "SL", 49000, -50.0, closed_at=yesterday_utc)
    
    daily_pnl = tracker.get_daily_pnl()
    print(f"Daily PnL (should be -100): {daily_pnl}")
    assert daily_pnl == -100.0
    
    stats = tracker.get_stats()
    current_balance = stats.get("running_capital", INITIAL_CAPITAL)
    starting_balance_today = current_balance - daily_pnl
    print(f"Starting balance today: {starting_balance_today}")
    
    loss_pct = abs(daily_pnl / starting_balance_today)
    print(f"Daily Loss %: {loss_pct*100:.2f}%")
    print(f"Limit %: {MAX_DAILY_LOSS_PCT*100:.2f}%")
    
    if loss_pct >= MAX_DAILY_LOSS_PCT:
        print("Daily loss limit would be triggered.")
    else:
        print("Daily loss limit not triggered.")

    if test_log.exists(): os.remove(test_log)

if __name__ == "__main__":
    test_daily_pnl()
