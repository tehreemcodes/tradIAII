'use client'

/**
 * ExchangePanel
 * ==============
 * Shows exchange connection status, auto-trade toggle,
 * open positions, and recent trade history.
 *
 * Session management:
 *   - session_id stored in localStorage
 *   - Sent as X-Session-Id header on all API requests
 *   - Raw keys never touch this component
 */

import { useEffect, useState, useCallback, useRef } from 'react'
import ConnectExchangeModal from './ConnectExchangeModal'

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000'

// ── Types ─────────────────────────────────────────────────────────────────────

interface ExchangeStatus {
  connected:    boolean
  exchange?:    string
  testnet?:     boolean
  balance?:     number
  connected_at?: string
}

interface TradingStatus {
  enabled:   boolean
  connected: boolean
  risk_pct:  number
}

interface OpenTrade {
  id:          string
  direction:   'BUY' | 'SELL'
  entry_price: number
  sl_price:    number
  tp_price:    number
  size:        number
  opened_at:   string
  paper:       boolean
}

interface ClosedTrade {
  id:          string
  direction:   'BUY' | 'SELL'
  entry_price: number
  close_price: number
  pnl:         number
  outcome:     'TP' | 'SL' | 'manual'
  opened_at:   string
  closed_at:   string
  paper:       boolean
}

interface LiveStats {
  total_trades:     number
  wins:             number
  losses:           number
  win_rate_pct:     number
  total_pnl:        number
  total_pnl_pct:    number
  running_capital:  number
  profit_factor:    number
  max_drawdown_pct: number
}

// ── Helpers ───────────────────────────────────────────────────────────────────

const fmt$ = (n: number) =>
  '$' + Math.abs(n).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })

const timeAgo = (iso: string) => {
  const s = Math.floor((Date.now() - new Date(iso).getTime()) / 1000)
  if (s < 60) return `${s}s ago`
  if (s < 3600) return `${Math.floor(s/60)}m ago`
  return `${Math.floor(s/3600)}h ago`
}

const authHeaders = (sessionId: string | null) =>
  sessionId ? { 'X-Session-Id': sessionId } : {}

// ── Sub-components ────────────────────────────────────────────────────────────

function StatusBadge({ connected, testnet }: { connected: boolean; testnet?: boolean }) {
  return (
    <div className="flex items-center gap-1.5">
      <div
        className="w-1.5 h-1.5 rounded-full"
        style={{
          background: connected ? '#2dd4bf' : 'rgba(255,255,255,0.2)',
          boxShadow:  connected ? '0 0 6px #2dd4bf' : 'none',
        }}
      />
      <span
        className="text-[10px] font-semibold"
        style={{ color: connected ? '#2dd4bf' : 'rgba(255,255,255,0.3)' }}
      >
        {connected ? 'CONNECTED' : 'NOT CONNECTED'}
      </span>
      {connected && testnet && (
        <span
          className="text-[9px] px-1.5 py-0.5 rounded font-bold"
          style={{ background: 'rgba(59,130,246,0.12)', color: '#60a5fa' }}
        >
          TESTNET
        </span>
      )}
    </div>
  )
}

function TradeRow({ trade }: { trade: ClosedTrade }) {
  const win = trade.outcome === 'TP'
  return (
    <div className="flex items-center justify-between py-1.5 border-b border-white/[0.03] last:border-0">
      <div className="flex items-center gap-2">
        <div className="w-0.5 h-4 rounded-full" style={{ background: trade.direction === 'BUY' ? '#2dd4bf' : '#f43f5e' }} />
        <span className="text-[10px] font-semibold" style={{ color: trade.direction === 'BUY' ? '#2dd4bf' : '#f43f5e' }}>
          {trade.direction}
        </span>
        <span className="text-[10px] text-white/25">{timeAgo(trade.closed_at)}</span>
        {trade.paper && <span className="text-[9px] text-yellow-500/50">PAPER</span>}
      </div>
      <div className="flex items-center gap-2">
        <span className="mono text-[10px] font-semibold" style={{ color: win ? '#2dd4bf' : '#f43f5e' }}>
          {trade.pnl >= 0 ? '+' : ''}{fmt$(trade.pnl)}
        </span>
        <span
          className="text-[9px] px-1.5 py-0.5 rounded font-bold"
          style={{
            background: win ? 'rgba(45,212,191,0.10)' : 'rgba(244,63,94,0.10)',
            color:      win ? '#2dd4bf' : '#f43f5e',
          }}
        >
          {trade.outcome}
        </span>
      </div>
    </div>
  )
}

function OpenTradeCard({ trade }: { trade: OpenTrade }) {
  const rr = Math.abs((trade.tp_price - trade.entry_price) / (trade.entry_price - trade.sl_price))
  return (
    <div
      className="rounded-lg p-3 mb-2"
      style={{
        background: trade.direction === 'BUY' ? 'rgba(45,212,191,0.06)' : 'rgba(244,63,94,0.06)',
        border:     `1px solid ${trade.direction === 'BUY' ? 'rgba(45,212,191,0.15)' : 'rgba(244,63,94,0.15)'}`,
      }}
    >
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span
            className="text-[10px] font-bold px-1.5 py-0.5 rounded"
            style={{
              background: trade.direction === 'BUY' ? 'rgba(45,212,191,0.15)' : 'rgba(244,63,94,0.15)',
              color:      trade.direction === 'BUY' ? '#2dd4bf' : '#f43f5e',
            }}
          >
            {trade.direction}
          </span>
          {trade.paper && <span className="text-[9px] text-yellow-500/50">PAPER</span>}
        </div>
        <span className="text-[10px] text-white/25">{timeAgo(trade.opened_at)}</span>
      </div>
      <div className="grid grid-cols-3 gap-1 text-center">
        {[
          { l: 'Entry', v: fmt$(trade.entry_price), c: 'rgba(255,255,255,0.6)' },
          { l: 'SL',    v: fmt$(trade.sl_price),    c: '#f43f5e' },
          { l: 'TP',    v: fmt$(trade.tp_price),     c: '#2dd4bf' },
        ].map(p => (
          <div key={p.l}>
            <p className="text-[9px] text-white/20 mb-0.5">{p.l}</p>
            <p className="mono text-[10px] font-semibold" style={{ color: p.c }}>{p.v}</p>
          </div>
        ))}
      </div>
      <p className="text-[9px] text-white/20 text-right mt-1.5">RR 1:{rr.toFixed(1)}</p>
    </div>
  )
}

// ── Main Component ────────────────────────────────────────────────────────────

export default function ExchangePanel() {
  const [sessionId,      setSessionId]      = useState<string | null>(null)
  const [showModal,      setShowModal]      = useState(false)
  const [exStatus,       setExStatus]       = useState<ExchangeStatus>({ connected: false })
  const [tradingStatus,  setTradingStatus]  = useState<TradingStatus>({ enabled: false, connected: false, risk_pct: 0.1 })
  const [openTrades,     setOpenTrades]     = useState<OpenTrade[]>([])
  const [closedTrades,   setClosedTrades]   = useState<ClosedTrade[]>([])
  const [stats,          setStats]          = useState<LiveStats | null>(null)
  const [riskPct,        setRiskPct]        = useState(10)
  const [togglingTrade,  setTogglingTrade]  = useState(false)
  const [paperMode,      setPaperMode]      = useState<boolean | null>(null)
  const [togglingPaper,  setTogglingPaper]  = useState(false)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  // Load session from localStorage on mount
  useEffect(() => {
    const stored = localStorage.getItem('tradia_session_id')
    if (stored) setSessionId(stored)
  }, [])

  // Fetch exchange status
  const fetchStatus = useCallback(async (sid: string | null) => {
    if (!sid) return
    try {
      const res  = await fetch(`${API_BASE}/api/exchange/status`, { headers: authHeaders(sid) })
      const data = await res.json()
      setExStatus(data)
    } catch {}
  }, [])

  // Fetch trading status
  const fetchTradingStatus = useCallback(async (sid: string | null) => {
    if (!sid) return
    try {
      const res  = await fetch(`${API_BASE}/api/trading/status`, { headers: authHeaders(sid) })
      const data = await res.json()
      setTradingStatus(data)
    } catch {}
  }, [])

  // Fetch trades
  const fetchTrades = useCallback(async (sid: string | null) => {
    try {
      const res  = await fetch(`${API_BASE}/api/trades`, { headers: authHeaders(sid) })
      const data = await res.json()
      setOpenTrades(data.open ?? [])
      setClosedTrades(data.closed ?? [])
      setStats(data.stats ?? null)
    } catch {}
  }, [])

  // Fetch paper mode
  const fetchPaperMode = useCallback(async () => {
    try {
      const res  = await fetch(`${API_BASE}/api/trading/paper-mode`)
      const data = await res.json()
      setPaperMode(data.paper_mode ?? true)
    } catch {}
  }, [])

  const fetchAll = useCallback((sid: string | null) => {
    fetchStatus(sid)
    fetchTradingStatus(sid)
    fetchTrades(sid)
    fetchPaperMode()
  }, [fetchStatus, fetchTradingStatus, fetchTrades, fetchPaperMode])

  useEffect(() => {
    fetchAll(sessionId)
  }, [sessionId, fetchAll])

  // Poll faster when open trades exist
  useEffect(() => {
    if (pollRef.current) clearInterval(pollRef.current)
    const ms = openTrades.length > 0 ? 10_000 : 60_000
    pollRef.current = setInterval(() => fetchAll(sessionId), ms)
    return () => { if (pollRef.current) clearInterval(pollRef.current) }
  }, [sessionId, openTrades.length, fetchAll])

  const handleConnected = (sid: string, balance: number, exchange: string) => {
    setSessionId(sid)
    setExStatus({ connected: true, exchange, balance, testnet: false })
    fetchAll(sid)
  }

  const handleDisconnect = async () => {
    if (!sessionId) return
    await fetch(`${API_BASE}/api/exchange/disconnect`, {
      method: 'POST', headers: authHeaders(sessionId)
    })
    localStorage.removeItem('tradia_session_id')
    setSessionId(null)
    setExStatus({ connected: false })
    setTradingStatus({ enabled: false, connected: false, risk_pct: 0.1 })
  }

  const toggleTrading = async () => {
    if (!sessionId) return
    setTogglingTrade(true)
    try {
      const endpoint = tradingStatus.enabled ? 'disable' : 'enable'
      const url      = `${API_BASE}/api/trading/${endpoint}${!tradingStatus.enabled ? `?risk_pct=${riskPct/100}` : ''}`
      await fetch(url, { method: 'POST', headers: authHeaders(sessionId) })
      await fetchTradingStatus(sessionId)
    } finally {
      setTogglingTrade(false)
    }
  }

  const togglePaperMode = async () => {
    setTogglingPaper(true)
    try {
      await fetch(`${API_BASE}/api/trading/paper-mode`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json', ...authHeaders(sessionId) },
        body:    JSON.stringify({ enabled: !paperMode }),
      })
      await fetchPaperMode()
    } finally {
      setTogglingPaper(false)
    }
  }

  const closeAll = async () => {
    if (!confirm('Close all open positions?')) return
    await fetch(`${API_BASE}/api/trades/close-all`, {
      method: 'POST', headers: authHeaders(sessionId)
    })
    await fetchTrades(sessionId)
  }

  // ── Render ──────────────────────────────────────────────────────────────────

  return (
    <>
      {showModal && (
        <ConnectExchangeModal
          onConnected={handleConnected}
          onClose={() => setShowModal(false)}
        />
      )}

      <div className="rounded-xl border border-white/5 bg-white/[0.03] p-4">

        {/* Header */}
        <div className="flex items-center justify-between mb-4">
          <div>
            <p className="text-[10px] font-semibold tracking-widest text-white/30 uppercase">
              Exchange
            </p>
            <div className="mt-1">
              <StatusBadge connected={exStatus.connected} testnet={exStatus.testnet} />
            </div>
          </div>

          {exStatus.connected ? (
            <button
              onClick={handleDisconnect}
              className="text-[9px] px-2.5 py-1 rounded-lg transition-all font-semibold"
              style={{
                background: 'rgba(244,63,94,0.08)',
                border:     '1px solid rgba(244,63,94,0.15)',
                color:      '#f43f5e',
                cursor:     'pointer',
              }}
            >
              Disconnect
            </button>
          ) : (
            <button
              onClick={() => setShowModal(true)}
              className="text-[9px] px-2.5 py-1 rounded-lg transition-all font-bold"
              style={{
                background: 'linear-gradient(135deg, rgba(45,212,191,0.15), rgba(59,130,246,0.15))',
                border:     '1px solid rgba(45,212,191,0.25)',
                color:      '#2dd4bf',
                cursor:     'pointer',
              }}
            >
              Connect Exchange
            </button>
          )}
        </div>

        {/* Paper-mode pill */}
        <div className="mb-4">
          <div className="flex items-center justify-between">
            <p className="text-[10px] font-semibold tracking-widest text-white/30 uppercase">Mode</p>
            <button
              onClick={togglePaperMode}
              disabled={togglingPaper}
              className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg transition-all"
              style={{
                background: paperMode ? 'rgba(245,158,11,0.10)' : 'rgba(244,63,94,0.08)',
                border:     paperMode ? '1px solid rgba(245,158,11,0.20)' : '1px solid rgba(244,63,94,0.15)',
                cursor: togglingPaper ? 'not-allowed' : 'pointer',
                opacity: togglingPaper ? 0.5 : 1,
              }}
            >
              <span
                className="text-[9px] font-bold uppercase tracking-wider"
                style={{ color: paperMode ? '#f59e0b' : '#f43f5e' }}
              >
                {paperMode == null ? '…' : paperMode ? '📋 Paper Mode' : '⚡ Live Mode'}
              </span>
            </button>
          </div>
        </div>

        {/* Connected info */}
        {exStatus.connected && (
          <div
            className="rounded-lg p-3 mb-4"
            style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.05)' }}
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-[9px] text-white/30 uppercase tracking-wider">Balance</p>
                <p className="mono text-base font-bold text-white mt-0.5">
                  {fmt$(exStatus.balance ?? 0)} <span className="text-[10px] text-white/30">USDT</span>
                </p>
              </div>
              <div className="text-right">
                <p className="text-[9px] text-white/30 uppercase tracking-wider">Exchange</p>
                <p className="text-xs font-semibold text-white/60 mt-0.5">
                  {exStatus.exchange?.toUpperCase()}
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Auto-trading toggle */}
        {exStatus.connected && (
          <div className="mb-4">
            <p className="text-[10px] font-semibold tracking-widest text-white/30 uppercase mb-2">
              Auto Trading
            </p>

            {/* Risk selector */}
            {!tradingStatus.enabled && (
              <div className="mb-2">
                <div className="flex items-center justify-between mb-1">
                  <span className="text-[10px] text-white/30">Risk per trade</span>
                  <span className="mono text-[11px] font-bold" style={{ color: '#f59e0b' }}>
                    {riskPct}%
                  </span>
                </div>
                <input
                  type="range"
                  min={1} max={25} step={1}
                  value={riskPct}
                  onChange={e => setRiskPct(Number(e.target.value))}
                  className="w-full h-1 rounded-full appearance-none cursor-pointer"
                  style={{ accentColor: '#2dd4bf' }}
                />
                <div className="flex justify-between text-[9px] text-white/20 mt-0.5">
                  <span>1% (safe)</span>
                  <span>10% (demo)</span>
                  <span>25% (high)</span>
                </div>
              </div>
            )}

            <button
              onClick={toggleTrading}
              disabled={togglingTrade}
              className="w-full py-2.5 rounded-xl text-xs font-bold transition-all"
              style={{
                background: tradingStatus.enabled
                  ? 'rgba(244,63,94,0.10)'
                  : 'linear-gradient(135deg, #2dd4bf, #3b82f6)',
                border: tradingStatus.enabled ? '1px solid rgba(244,63,94,0.20)' : 'none',
                color:  tradingStatus.enabled ? '#f43f5e' : '#0d1117',
                cursor: togglingTrade ? 'not-allowed' : 'pointer',
                opacity: togglingTrade ? 0.6 : 1,
              }}
            >
              {togglingTrade ? '...' : tradingStatus.enabled ? '⏹ Stop Auto Trading' : '▶ Start Auto Trading'}
            </button>

            {tradingStatus.enabled && (
              <div
                className="mt-2 px-3 py-2 rounded-lg text-center"
                style={{ background: 'rgba(45,212,191,0.06)', border: '1px solid rgba(45,212,191,0.12)' }}
              >
                <p className="text-[10px] text-teal-400">
                  🟢 Trading active · {(tradingStatus.risk_pct * 100).toFixed(0)}% risk per trade
                </p>
              </div>
            )}
          </div>
        )}

        {/* Live stats */}
        {stats && stats.total_trades > 0 && (
          <>
            <p className="text-[10px] font-semibold tracking-widest text-white/30 uppercase mb-2">
              Live Performance
            </p>
            <div className="grid grid-cols-3 gap-1.5 mb-3">
              {[
                { l: 'P&L',      v: `${stats.total_pnl >= 0 ? '+' : ''}${stats.total_pnl_pct.toFixed(1)}%`,
                  c: stats.total_pnl >= 0 ? '#2dd4bf' : '#f43f5e' },
                { l: 'Win Rate', v: `${stats.win_rate_pct.toFixed(1)}%`,
                  c: stats.win_rate_pct >= 50 ? '#2dd4bf' : '#f43f5e' },
                { l: 'Trades',   v: String(stats.total_trades),
                  c: 'rgba(255,255,255,0.6)' },
              ].map(m => (
                <div
                  key={m.l}
                  className="rounded-lg p-2 text-center"
                  style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.04)' }}
                >
                  <p className="text-[9px] text-white/20 mb-0.5">{m.l}</p>
                  <p className="mono text-[11px] font-bold" style={{ color: m.c }}>{m.v}</p>
                </div>
              ))}
            </div>
          </>
        )}

        {/* Open positions */}
        <div className="mb-1">
          <div className="flex items-center justify-between mb-2">
            <p className="text-[10px] font-semibold tracking-widest text-white/30 uppercase">
              Open ({openTrades.length})
            </p>
            {openTrades.length > 0 && (
              <button
                onClick={closeAll}
                className="text-[9px] px-2 py-0.5 rounded transition-all font-semibold"
                style={{
                  background: 'rgba(244,63,94,0.08)',
                  border:     '1px solid rgba(244,63,94,0.15)',
                  color:      '#f43f5e',
                  cursor:     'pointer',
                }}
              >
                Close All
              </button>
            )}
          </div>

          {openTrades.length === 0 ? (
            <p className="text-[11px] text-white/15 text-center py-3">No open positions</p>
          ) : (
            openTrades.map(t => <OpenTradeCard key={t.id} trade={t} />)
          )}
        </div>

        {/* Recent closed trades */}
        {closedTrades.length > 0 && (
          <>
            <p className="text-[10px] font-semibold tracking-widest text-white/30 uppercase mb-2 mt-3">
              Recent Trades
            </p>
            {[...closedTrades].reverse().slice(0, 8).map(t => (
              <TradeRow key={t.id + t.closed_at} trade={t} />
            ))}
          </>
        )}

        {/* No connection prompt */}
        {!exStatus.connected && (
          <div className="text-center py-4 mt-2">
            <p className="text-[11px] text-white/20 mb-1">No exchange connected</p>
            <p className="text-[10px] text-white/12">
              Connect your Binance account to enable auto trading
            </p>
          </div>
        )}
      </div>
    </>
  )
}