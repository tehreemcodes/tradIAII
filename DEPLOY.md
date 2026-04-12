# TradIA -- Deployment Guide

## Requirements
- Python 3.11+
- Node.js 18+
- Binance Demo API credentials (demo-fapi.binance.com)

## 1. Backend Setup

```bash
cd tradIAII
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env: set EXCHANGE_API_KEY, EXCHANGE_API_SECRET, BINANCE_BASE_URL
```

## 2. Train the Model

```bash
python -m backend.scripts.fetch_data
python -m backend.scripts.train_model
```

## 3. Run Backtest

```bash
python -m backend.scripts.backtest          # with ML model
python -m backend.scripts.backtest --no-model  # raw ICT signals only
```

## 4. Start Backend

```bash
uvicorn backend.api.api_server:app --host 0.0.0.0 --port 8000
```

API docs: http://localhost:8000/docs  
Dashboard: http://localhost:8000/dashboard

## 5. Frontend Setup

```bash
cd frontend
npm install
# .env.local is already created -- edit NEXT_PUBLIC_API_URL if needed
npm run dev       # development
npm run build     # production build
npm start         # serve production build
```

## 6. Production Server (187.127.103.154)

```bash
# Backend (port 8000)
uvicorn backend.api.api_server:app --host 0.0.0.0 --port 8000 --workers 1

# Frontend (port 3000)
cd frontend && npm run build && npm start
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| PAPER_MODE | Paper trading mode | true |
| BINANCE_BASE_URL | Must be demo-fapi.binance.com | required |
| EXCHANGE_API_KEY | Binance Demo API key | - |
| EXCHANGE_API_SECRET | Binance Demo API secret | - |
| FRONTEND_URL | CORS origin for production | - |
| LOG_LEVEL | Logging verbosity | INFO |

## Key Safety Assertions

- `BINANCE_BASE_URL` must contain `demo-fapi.binance.com` (enforced at startup)
- `PAPER_MODE=true` by default -- set to false only with valid API credentials
- `MAX_DRAWDOWN_STOP=0.40` halts trading if equity drops 40%
