/**
 * TradIA Backend API Client
 * Typed fetch wrappers for all FastAPI endpoints.
 */

const BASE = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000'

// ── Types ──────────────────────────────────────────────────────

export type SignalType = 'BUY' | 'SELL' | 'NO TRADE'

export interface HTFBias {
  h4: number              // +1 bullish | -1 bearish | 0 neutral
  d1: number
  full_confluence: boolean
}

export interface PatternInfo {
  swing_price: number | null
  fvg_top:     number | null
  fvg_bot:     number | null
}

export interface Signal {
  signal:        SignalType
  confidence:    number
  entry:         number | null
  sl:            number | null
  tp:            number | null
  rr:            string
  risk_amount:   number
  position_size: number
  timestamp:     string
  candle_time:   string | null
  pair:          string
  timeframe:     string
  htf_bias:      HTFBias | null
  pattern:       PatternInfo | null
  error:         string | null
}

export interface Candle {
  time:        number   // Unix seconds
  open:        number
  high:        number
  low:         number
  close:       number
  volume:      number
  swing_high:  boolean
  swing_low:   boolean
  bull_fvg:    boolean
  bear_fvg:    boolean
  fvg_top:     number | null
  fvg_bot:     number | null
  signal:      number   // 2=BUY 0=SELL 1=NO TRADE
  signal_sl:   number | null
}

export interface CandleResponse {
  timeframe: string
  symbol:    string
  candles:   Candle[]
  count:     number
}

export interface HTFBiasResponse {
  daily:            number
  h4:               number
  h1:               number
  confluence_score: number
  verdict:          'BULLISH' | 'BEARISH' | 'MIXED'
}

export interface StatsResponse {
  total_signals:    number
  wins:             number
  losses:           number
  win_rate_pct:     number
  net_pnl:          number
  net_pnl_pct:      number
  final_capital:    number
  max_drawdown_pct: number
  profit_factor:    number
  total_fees_paid:  number
  last_updated:     string | null
  backtest_running: boolean
  monte_carlo:      {
    median_final:      number
    p5_final:          number
    p95_final:         number
    pct_profitable:    number
    median_max_dd_pct: number
    p5_max_dd_pct:     number
    p95_max_dd_pct:    number
  } | null
}

export interface FVGZone {
  type:       string
  direction:  'bullish' | 'bearish'
  top:        number
  bot:        number
  timeframe:  string
  timestamp:  number
  filled:     boolean
}

export interface ZonesResponse {
  zones: FVGZone[]
  count: number
}

export interface HealthResponse {
  status:       string
  model_ready:  boolean
  timestamp:    string
  signal_tf:    string
  htf_list:     string[]
  version:      string
}

export interface ModelInfo {
  signal_timeframe: string
  htf_timeframes:   string[]
  feature_count:    number
  features:         string[]
  risk_pct:         number
  reward_ratio:     number
  initial_capital:  number
  min_confidence:   number
}

// ── Fetch Helpers ─────────────────────────────────────────────

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    cache: 'no-store',
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || `HTTP ${res.status}`)
  }
  return res.json()
}

async function post<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    body ? JSON.stringify(body) : undefined,
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || `HTTP ${res.status}`)
  }
  return res.json()
}

// ── API Functions ─────────────────────────────────────────────

export const api = {
  /** Server + model health check */
  health: (): Promise<HealthResponse> =>
    get('/api/health'),

  /** Latest ICT/SMC signal with confidence and levels */
  signal: (capital?: number, timeframe: string = '1h'): Promise<Signal> => {
    const params = new URLSearchParams()
    if (capital) params.append('capital', capital.toString())
    if (timeframe) params.append('timeframe', timeframe)
    const qs = params.toString()
    return get(`/api/signal${qs ? `?${qs}` : ''}`)
  },

  /** OHLCV candles with ICT overlays for chart */
  candles: (timeframe = '1h', limit = 200): Promise<CandleResponse> =>
    get(`/api/candles?timeframe=${timeframe}&limit=${limit}`),

  /** Daily / 4H / 1H structural bias */
  htfBias: (): Promise<HTFBiasResponse> =>
    get('/api/htf-bias'),

  /** Backtest performance stats */
  stats: (): Promise<StatsResponse> =>
    get('/api/stats'),

  /** Active FVG zones */
  zones: (): Promise<ZonesResponse> =>
    get('/api/zones'),

  /** Model metadata */
  modelInfo: (): Promise<ModelInfo> =>
    get('/api/model/info'),

  /** Last backtest summary */
  backtest: () =>
    get('/api/backtest'),

  /** Trigger background backtest */
  runBacktest: () =>
    post('/api/backtest/run'),

  /** Backtest running status */
  backtestStatus: () =>
    get('/api/backtest/status'),
}

// ── Formatting Helpers ────────────────────────────────────────

export function fmtPrice(n: number | null | undefined): string {
  if (n == null) return '—'
  return n.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })
}

export function fmtPct(n: number | null | undefined, decimals = 1): string {
  if (n == null) return '—'
  return `${n >= 0 ? '+' : ''}${n.toFixed(decimals)}%`
}

export function fmtConfidence(n: number): string {
  return `${(n * 100).toFixed(1)}%`
}

export function signalColor(signal: SignalType): string {
  if (signal === 'BUY')  return '#2dd4bf'
  if (signal === 'SELL') return '#f43f5e'
  return '#64748b'
}

export function biasColor(bias: number): string {
  if (bias === 1)  return '#2dd4bf'
  if (bias === -1) return '#f43f5e'
  return '#64748b'
}

export function biasLabel(bias: number): string {
  if (bias === 1)  return 'Bullish'
  if (bias === -1) return 'Bearish'
  return 'Neutral'
}