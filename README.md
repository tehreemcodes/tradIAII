# TradIA — ICT/SMC AI Crypto Trading Assistant

## Strategy
```
Daily Bias + 4H Bias + 1H Signal
         ↓
Swing High/Low  →  CISD  →  FVG   (within 20 candles)
         ↓
LightGBM confidence filter  (≥ 60%)
         ↓
BUY / SELL signal with Entry, SL, TP, Position Size
         ↓
10% compounding risk management
```

## Project Structure
```
tradIA/
├── backend/
│   ├── config/
│   │   ├── settings.py          ← ALL config here
│   │   └── logging_setup.py     ← Windows-safe UTF-8 logging
│   ├── services/
│   │   ├── data_loader.py
│   │   ├── ict_strategy.py      ← Swing / CISD / FVG
│   │   ├── state_machine.py     ← Sequential pattern engine
│   │   ├── multi_timeframe.py   ← HTF bias, zero lookahead
│   │   ├── feature_builder.py   ← 45 ML features
│   │   ├── label_generator.py   ← WIN / LOSS labeling
│   │   └── risk_manager.py      ← 10% risk, compounding
│   ├── scripts/
│   │   ├── fetch_data.py        ← Bybit + KuCoin fallback
│   │   ├── train_model.py       ← Full training pipeline
│   │   ├── backtest.py          ← Historical simulation
│   │   └── live_predict.py      ← Real-time inference
│   ├── api/
│   │   └── api_server.py        ← FastAPI endpoints
│   └── tests/
│       └── test_strategy.py     ← Unit tests
└── frontend/
    ├── app/
    │   └── page.tsx             ← 3-column dashboard
    ├── components/
    │   ├── Chart.tsx            ← Candlestick + overlays
    │   ├── SignalCard.tsx       ← Signal + confidence
    │   ├── HTFBiasPanel.tsx     ← HTF bias + killzones
    │   ├── ConfluenceChecklist.tsx
    │   └── RiskPanel.tsx        ← P&L stats
    └── lib/
        └── api.ts               ← Typed API client
```

## Setup & Run

### Backend

```bash
cd tradIA/backend
pip install -r requirements.txt
```

**Step 1 — Fetch Data** (10-20 minutes, ~37k 1H candles)
```bash
python -m backend.scripts.fetch_data
```

**Step 2 — Train Model**
```bash
python -m backend.scripts.train_model
```
Target: Cohen's Kappa ≥ 0.35

**Step 3 — Run Backtest**
```bash
python -m backend.scripts.backtest
```

**Step 4 — Start API**
```bash
uvicorn backend.api.api_server:app --reload --port 8000
```

**Run Tests**
```bash
cd tradIA && python -m pytest backend/tests/ -v
```

### Frontend

```bash
cd tradIA/frontend
npm install
npm run dev
```
Open: http://localhost:3000

## API Endpoints

| Method | Endpoint            | Description                  |
|--------|---------------------|------------------------------|
| GET    | /api/health         | Server + model status        |
| GET    | /api/signal         | Latest ICT signal            |
| GET    | /api/candles        | OHLCV + ICT overlays         |
| GET    | /api/htf-bias       | Daily/4H/1H structure        |
| GET    | /api/stats          | Backtest performance         |
| GET    | /api/zones          | Active FVG zones             |
| GET    | /api/model/info     | Model metadata               |
| POST   | /api/backtest/run   | Trigger backtest             |

Interactive docs: http://localhost:8000/docs

## Configuration
All settings in `backend/config/settings.py`:
- `SIGNAL_TF = "1h"`         — signal generation timeframe
- `HTF_LIST  = ["4h", "1d"]` — bias timeframes
- `RISK_PCT  = 0.10`         — 10% max risk per trade
- `REWARD_RATIO = 2.0`       — 1:2 R:R minimum
- `MIN_CONFIDENCE = 0.60`    — model threshold
- `PATTERN_WINDOW = 20`      — max candles swing→FVG

## Kappa Score Guide
```
< 0.20   Not ready — re-fetch data with full history
0.20-0.35  Weak — usable with strict confidence filter
0.35-0.50  Good — ready for paper trading
> 0.50   Strong — production ready
```
