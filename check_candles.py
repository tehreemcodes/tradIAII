import requests

try:
    res = requests.get("http://localhost:8000/api/candles?timeframe=1h&limit=200")
    if res.ok:
        data = res.json()
        candles = data.get("candles", [])
        signals = [c for c in candles if c.get("signal") in [0, 2]]
        print(f"Total candles: {len(candles)}")
        print(f"Total signals found: {len(signals)}")
        for s in signals[-5:]:
            print(f"Signal: {s['signal']}, Executable: {s.get('executable')}, Reason: {s.get('reject_reason')}, Confidence: {s.get('ml_confidence')}")
    else:
        print("API Error:", res.text)
except Exception as e:
    print("Error connecting to API:", e)
