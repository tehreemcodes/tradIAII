"""
Test Live Execution on Binance Testnet
======================================
Places a dummy trade to verify the execution pipeline end-to-end.

Usage:
    # Make sure your API keys are set in your .env
    python -m backend.scripts.test_live
"""
import logging
import time
from backend.config.logging_setup import setup_logging
from backend.services.trade_executor import TradeExecutor

setup_logging()
logger = logging.getLogger(__name__)

def run_test_trade():
    print("=" * 60)
    print("  TradIA Execution Layer Test - Binance Futures")
    print("=" * 60)
    print("WARNING: This will attempt to place an order on your connected account.")
    print("It is pre-configured to enforce TESTNET connection.")
    ans = input("Proceed? (y/n): ")
    if ans.lower() != 'y':
        return

    executor = TradeExecutor(testnet=True)
    
    if not executor.connect():
        print("Failed to connect!")
        return
        
    print(f"\nConnected! Balance: {executor.get_balance()} USDT")
    
    try:
        res = executor._request("GET", "/fapi/v1/ticker/price", {"symbol": executor.symbol}, signed=False)
        current_price = float(res['price'])
        print(f"Current {executor.symbol} mark price: ${current_price:,.2f}")
        
        # Calculate tiny test trade
        size = 0.005 # 0.005 BTC
        
        # Long trade params
        sl = current_price * 0.99
        tp = current_price * 1.01
        
        print(f"\nAttempting to orchestrate BUY order for {size} BTC at MARKET.")
        print(f"Setting conditional SL: ${sl:,.2f} | TP: ${tp:,.2f}")
        print("Placing order...")
        time.sleep(1)
        
        order = executor.place_order(
            direction="BUY",
            position_size=size,
            entry_price=current_price,
            sl_price=sl,
            tp_price=tp,
            signal_ts="TEST_RUN"
        )
        
        print("\nSUCCESS! Order flow completed cleanly.")
        print(f"Return payload: {order}")
        print("\nCheck your Binance Testnet dashboard to verify the Limit and SL/TP Conditional resting orders!")
        
    except Exception as e:
        print(f"\n[ERROR] Test trade failed: {e}")

if __name__ == "__main__":
    run_test_trade()
