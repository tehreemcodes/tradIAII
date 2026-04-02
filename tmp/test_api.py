import requests
import sys

try:
    res = requests.get('http://localhost:8000/api/analytics/summary')
    print(f"Summary Status: {res.status_code}")
    if res.status_code == 200:
        print(f"Summary Data: {res.json()}")
    
    res = requests.get('http://localhost:8000/api/analytics/trades')
    print(f"Trades Status: {res.status_code}")
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
