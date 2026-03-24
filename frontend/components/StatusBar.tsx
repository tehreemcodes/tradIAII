'use client'

/**
 * StatusBar
 * =========
 * A top-of-dashboard status strip showing live system health:
 *  - Exchange connection state
 *  - Paper / Live mode
 *  - Active signal timeframe
 *  - Last signal summary
 *  - Today's PnL + daily drawdown
 *
 * Polls /api/status/live every 30 s.
 */

import { useEffect, useRef, useState } from 'react'

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000'

interface LiveStatus {
  exchange_connected:  boolean
  paper_mode:          boolean
  active_timeframe:    string
  last_signal:         { signal?: string; confidence?: number; entry?: number } | null
  today_pnl:           number
  daily_drawdown_pct:  number
}

const authHeaders = () => {
  const sid = typeof window !== 'undefined' ? localStorage.getItem('tradia_session_id') : null
  return sid ? { 'X-Session-Id': sid } : {}
}

function Pill({
  label, value, color,
}: { label: string; value: string; color: string }) {
  return (
    <div
      className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg"
      style={{
        background: `${color}14`,
        border: `1px solid ${color}28`,
      }}
    >
      <span className="text-[9px] uppercase tracking-wider font-semibold" style={{ color: `${color}80` }}>
        {label}
      </span>
      <span className="text-[11px] font-bold" style={{ color }}>
        {value}
      </span>
    </div>
  )
}

export default function StatusBar() {
  const [status, setStatus] = useState<LiveStatus | null>(null)
  const [togglingPaper, setTogglingPaper] = useState(false)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const fetchStatus = async () => {
    try {
      const res  = await fetch(`${API_BASE}/api/status/live`, { headers: authHeaders() })
      const data = await res.json()
      setStatus(data)
    } catch {}
  }

  useEffect(() => {
    fetchStatus()
    pollRef.current = setInterval(fetchStatus, 30_000)
    return () => { if (pollRef.current) clearInterval(pollRef.current) }
  }, [])

  const togglePaperMode = async () => {
    if (!status) return
    setTogglingPaper(true)
    try {
      await fetch(`${API_BASE}/api/trading/paper-mode`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json', ...authHeaders() },
        body:    JSON.stringify({ enabled: !status.paper_mode }),
      })
      await fetchStatus()
    } finally {
      setTogglingPaper(false)
    }
  }

  if (!status) {
    return (
      <div
        className="rounded-xl px-4 py-2.5 mb-4 flex items-center gap-2"
        style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.05)' }}
      >
        <div className="w-1.5 h-1.5 rounded-full bg-white/10 animate-pulse" />
        <span className="text-[10px] text-white/20">Loading system status...</span>
      </div>
    )
  }

  const pnlColor = status.today_pnl >= 0 ? '#2dd4bf' : '#f43f5e'
  const sigColor = status.last_signal?.signal === 'BUY' ? '#2dd4bf'
    : status.last_signal?.signal === 'SELL' ? '#f43f5e'
    : 'rgba(255,255,255,0.4)'

  return (
    <div
      className="rounded-xl px-4 py-3 mb-4 flex items-center justify-between gap-3 flex-wrap"
      style={{
        background: 'rgba(255,255,255,0.02)',
        border: '1px solid rgba(255,255,255,0.05)',
      }}
    >
      {/* Left — connection + mode */}
      <div className="flex items-center gap-2 flex-wrap">

        {/* Exchange dot */}
        <div className="flex items-center gap-1.5">
          <div
            className="w-1.5 h-1.5 rounded-full"
            style={{
              background: status.exchange_connected ? '#2dd4bf' : 'rgba(255,255,255,0.2)',
              boxShadow:  status.exchange_connected ? '0 0 6px #2dd4bf' : 'none',
            }}
          />
          <span
            className="text-[10px] font-semibold"
            style={{ color: status.exchange_connected ? '#2dd4bf' : 'rgba(255,255,255,0.25)' }}
          >
            {status.exchange_connected ? 'Exchange Connected' : 'No Exchange'}
          </span>
        </div>

        <span className="text-white/10 text-xs">|</span>

        {/* Paper / Live mode pill */}
        <button
          onClick={togglePaperMode}
          disabled={togglingPaper}
          title="Click to toggle Paper / Live mode"
          className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg transition-all"
          style={{
            background: status.paper_mode ? 'rgba(245,158,11,0.12)' : 'rgba(244,63,94,0.10)',
            border:     status.paper_mode ? '1px solid rgba(245,158,11,0.25)' : '1px solid rgba(244,63,94,0.20)',
            cursor: togglingPaper ? 'not-allowed' : 'pointer',
            opacity: togglingPaper ? 0.5 : 1,
          }}
        >
          <span
            className="text-[9px] uppercase tracking-wider font-bold"
            style={{ color: status.paper_mode ? '#f59e0b' : '#f43f5e' }}
          >
            {status.paper_mode ? '📋 PAPER MODE' : '⚡ LIVE MODE'}
          </span>
        </button>

        <span className="text-white/10 text-xs">|</span>

        {/* Active TF */}
        <Pill label="TF" value={status.active_timeframe?.toUpperCase() ?? '—'} color="#60a5fa" />

        {/* Last signal */}
        {status.last_signal && (
          <Pill
            label="Signal"
            value={`${status.last_signal.signal ?? '—'}${
              status.last_signal.confidence != null
                ? ` ${(status.last_signal.confidence * 100).toFixed(0)}%`
                : ''
            }`}
            color={sigColor}
          />
        )}
      </div>

      {/* Right — daily PnL */}
      <div className="flex items-center gap-2">
        <div
          className="px-2.5 py-1 rounded-lg"
          style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.05)' }}
        >
          <p className="text-[9px] text-white/20 uppercase tracking-wider">Today P&L</p>
          <p className="mono text-[11px] font-bold" style={{ color: pnlColor }}>
            {status.today_pnl >= 0 ? '+' : ''}${Math.abs(status.today_pnl).toFixed(2)}
          </p>
        </div>

        {status.daily_drawdown_pct > 0 && (
          <div
            className="px-2.5 py-1 rounded-lg"
            style={{ background: 'rgba(244,63,94,0.06)', border: '1px solid rgba(244,63,94,0.12)' }}
          >
            <p className="text-[9px] text-white/20 uppercase tracking-wider">DD</p>
            <p className="mono text-[11px] font-bold text-rose-400">
              -{status.daily_drawdown_pct.toFixed(2)}%
            </p>
          </div>
        )}
      </div>
    </div>
  )
}
