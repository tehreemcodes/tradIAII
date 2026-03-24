"""
Quick exchange connection test.
Run from project root:
    python test_connection.py
"""
from dotenv import load_dotenv
load_dotenv()
import backend.config.settings as s

# Override for this test only
s.LIVE_TRADING_ENABLED = True

from backend.services.trade_executor import TradeExecutor

print("=" * 50)
print(f"  Exchange : {s.EXCHANGE}")
print(f"  Testnet  : {s.EXCHANGE_TESTNET}")
print(f"  Symbol   : {s.SYMBOL}")
print(f"  API Key  : {s.EXCHANGE_API_KEY[:8]}..." if s.EXCHANGE_API_KEY else "  API Key  : NOT SET")
print("=" * 50)

ex = TradeExecutor()
ok = ex.connect()

print(f"\nConnected : {ok}")

if ok:
    balance = ex.get_balance()
    print(f"Balance   : {balance:,.2f} USDT")

    positions = ex.get_open_positions()
    print(f"Open pos  : {len(positions)}")

    print("\nConnection test PASSED — ready to trade.")
else:
    print("\nConnection test FAILED.")
    print("Check:")
    print("  1. EXCHANGE_API_KEY and EXCHANGE_API_SECRET are set in .env")
    print("  2. Keys are from testnet.binancefuture.com (not mainnet)")
    print("  3. Keys have Futures trading permissions enabled")