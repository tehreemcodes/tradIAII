'use client'

/**
 * RiskPanel — Backtest Performance Panel
 * ========================================
 * Fetches live stats from GET /api/stats on mount and every 5 minutes.
 * When a backtest is running (backtest_running=true), polls every 5s
 * and shows a spinner until the job completes, then refreshes.
 *
 * Displays:
 *   - Net P/L, Win Rate, Profit Factor, Max Drawdown
 *   - Total signals, Wins/Losses, Total fees
 *   - Monte Carlo P95 drawdown (if available)
 *   - Last updated timestamp
 *   - Trigger backtest button → POST /api/backtest/run
 */

import { useEffect, useState, useCallback, useRef } from 'react'

// ── Types ─────────────────────────────────────────────────────────────────────

interface MonteCarlo {
  median_final:      number
  p5_final:          number
  p95_final:         number
  pct_profitable:    number
  median_max_dd_pct: number
  p5_max_dd_pct:     number
  p95_max_dd_pct:    number
}

interface Stats {
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
  monte_carlo:      MonteCarlo | null
}

// ── Constants ─────────────────────────────────────────────────────────────────

const API_BASE         = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000'
const POLL_IDLE_MS     = 5 * 60 * 1000   // refresh every 5 min when idle
const POLL_RUNNING_MS  = 15 * 1000       // poll every 15s while backtest runs (min 15s)

// ── Helpers ───────────────────────────────────────────────────────────────────

function fmt$(n: number): string {
  return '$' + n.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })
}

function fmtPct(n: number, sign = false): string {
  return (sign && n > 0 ? '+' : '') + n.toFixed(2) + '%'
}

function relativeTime(iso: string | null): string {
  if (!iso) return '—'
  const diff = Math.floor((Date.now() - new Date(iso).getTime()) / 1000)
  if (diff < 60)   return `${diff}s ago`
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`
  if (diff < 86400)return `${Math.floor(diff / 3600)}h ago`
  return `${Math.floor(diff / 86400)}d ago`
}

// ── Sub-components ────────────────────────────────────────────────────────────

function StatRow({
  label,
  value,
  color,
  sub,
}: {
  label: string
  value: string
  color?: string
  sub?:  string
}) {
  return (
    <div className="flex items-center justify-between py-1.5 border-b border-white/[0.04] last:border-0">
      <span className="text-white/30 text-xs">{label}</span>
      <div className="text-right">
        <span
          className="mono text-xs font-semibold"
          style={{ color: color ?? 'rgba(255,255,255,0.7)' }}
        >
          {value}
        </span>
        {sub && (
          <span className="block text-[10px] text-white/20">{sub}</span>
        )}
      </div>
    </div>
  )
}

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <p className="text-[10px] font-semibold tracking-widest text-white/30 uppercase mb-2 mt-3 first:mt-0">
      {children}
    </p>
  )
}

function Spinner() {
  return (
    <div
      className="w-3 h-3 rounded-full border-2 border-white/10 border-t-teal-400 animate-spin"
      style={{ borderTopColor: '#2dd4bf' }}
    />
  )
}

// ── Main Component ────────────────────────────────────────────────────────────

export default function RiskPanel() {
  const [stats,         setStats]         = useState<Stats | null>(null)
  const [loading,       setLoading]       = useState(true)
  const [error,         setError]         = useState<string | null>(null)
  const [triggerMsg,    setTriggerMsg]    = useState<string | null>(null)
  const [relTime,       setRelTime]       = useState<string>('—')
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  // ── Fetch stats ────────────────────────────────────────────────────────────
  const fetchStats = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/stats`)
      if (!res.ok) {
        const detail = await res.json().catch(() => ({ detail: res.statusText }))
        throw new Error(detail?.detail ?? res.statusText)
      }
      const data: Stats = await res.json()
      setStats(data)
      setError(null)
    } catch (e: any) {
      setError(e.message ?? 'Failed to load stats')
    } finally {
      setLoading(false)
    }
  }, [])

  // ── Polling logic ──────────────────────────────────────────────────────────
  // When backtest is running: poll every 5s.
  // When idle: poll every 5 min.
  useEffect(() => {
    fetchStats()
  }, [fetchStats])

  useEffect(() => {
    if (pollRef.current) clearInterval(pollRef.current)

    const interval = stats?.backtest_running ? POLL_RUNNING_MS : POLL_IDLE_MS
    pollRef.current = setInterval(fetchStats, interval)

    return () => {
      if (pollRef.current) clearInterval(pollRef.current)
    }
  }, [stats?.backtest_running, fetchStats])

  // ── Relative time ticker ───────────────────────────────────────────────────
  useEffect(() => {
    const tick = () => setRelTime(relativeTime(stats?.last_updated ?? null))
    tick()
    const iv = setInterval(tick, 30_000)
    return () => clearInterval(iv)
  }, [stats?.last_updated])

  // ── Trigger backtest ───────────────────────────────────────────────────────
  const triggerBacktest = async () => {
    try {
      setTriggerMsg('Starting...')
      const res  = await fetch(`${API_BASE}/api/backtest/run`, { method: 'POST' })
      const data = await res.json()
      if (data.status === 'already_running') {
        setTriggerMsg('Already running')
      } else {
        setTriggerMsg('Running...')
        // Kick off fast polling immediately
        fetchStats()
      }
    } catch {
      setTriggerMsg('Failed to start')
    } finally {
      setTimeout(() => setTriggerMsg(null), 4000)
    }
  }

  // ── Render ─────────────────────────────────────────────────────────────────

  const tile = (
    <div
      className="rounded-xl border border-white/5 bg-white/[0.03] p-4"
      style={{ minWidth: 0 }}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <p className="text-[10px] font-semibold tracking-widest text-white/30 uppercase">
          Performance
        </p>
        <div className="flex items-center gap-2">
          {stats?.backtest_running && <Spinner />}
          <button
            onClick={triggerBacktest}
            disabled={stats?.backtest_running}
            className="text-[9px] font-semibold tracking-wider px-2 py-0.5 rounded transition-all"
            style={{
              background:  stats?.backtest_running
                ? 'rgba(255,255,255,0.04)'
                : 'rgba(45,212,191,0.10)',
              border:      `1px solid ${stats?.backtest_running
                ? 'rgba(255,255,255,0.06)'
                : 'rgba(45,212,191,0.20)'}`,
              color:       stats?.backtest_running
                ? 'rgba(255,255,255,0.20)'
                : '#2dd4bf',
              cursor:      stats?.backtest_running ? 'not-allowed' : 'pointer',
            }}
          >
            {triggerMsg ?? (stats?.backtest_running ? 'RUNNING' : 'RUN')}
          </button>
        </div>
      </div>

      {/* Loading state */}
      {loading && (
        <div className="flex items-center justify-center py-8">
          <Spinner />
        </div>
      )}

      {/* Error state */}
      {!loading && error && (
        <div
          className="rounded-lg p-3 text-center"
          style={{ background: 'rgba(244,63,94,0.08)', border: '1px solid rgba(244,63,94,0.15)' }}
        >
          <p className="text-[11px] text-red-400 mb-1">No backtest data</p>
          <p className="text-[10px] text-white/30">Run backtest to see results</p>
        </div>
      )}

      {/* Data */}
      {!loading && !error && stats && (
        <>
          {/* Hero metrics */}
          <div className="grid grid-cols-2 gap-2 mb-3">
            {[
              {
                label: 'Net P/L',
                value: fmtPct(stats.net_pnl_pct, true),
                sub:   fmt$(stats.net_pnl),
                color: stats.net_pnl >= 0 ? '#2dd4bf' : '#f43f5e',
                bg:    stats.net_pnl >= 0 ? 'rgba(45,212,191,0.07)' : 'rgba(244,63,94,0.07)',
                border:stats.net_pnl >= 0 ? 'rgba(45,212,191,0.15)' : 'rgba(244,63,94,0.15)',
              },
              {
                label: 'Win Rate',
                value: fmtPct(stats.win_rate_pct),
                sub:   `${stats.wins}W / ${stats.losses}L`,
                color: stats.win_rate_pct >= 55 ? '#2dd4bf'
                     : stats.win_rate_pct >= 50 ? '#f59e0b' : '#f43f5e',
                bg:    'rgba(255,255,255,0.03)',
                border:'rgba(255,255,255,0.06)',
              },
            ].map(m => (
              <div
                key={m.label}
                className="rounded-lg p-2.5 text-center"
                style={{ background: m.bg, border: `1px solid ${m.border}` }}
              >
                <p className="text-[9px] text-white/30 mb-1 tracking-wider uppercase">{m.label}</p>
                <p className="mono text-sm font-bold" style={{ color: m.color }}>{m.value}</p>
                <p className="text-[10px] text-white/30 mt-0.5">{m.sub}</p>
              </div>
            ))}
          </div>

          {/* Stats rows */}
          <SectionLabel>Backtest Stats</SectionLabel>
          <StatRow
            label="Final Capital"
            value={fmt$(stats.final_capital)}
            color="rgba(255,255,255,0.7)"
          />
          <StatRow
            label="Profit Factor"
            value={stats.profit_factor.toFixed(2)}
            color={stats.profit_factor >= 1.5 ? '#2dd4bf'
                 : stats.profit_factor >= 1.0 ? '#f59e0b' : '#f43f5e'}
          />
          <StatRow
            label="Max Drawdown"
            value={fmtPct(stats.max_drawdown_pct)}
            color={stats.max_drawdown_pct <= 10 ? '#2dd4bf'
                 : stats.max_drawdown_pct <= 20 ? '#f59e0b' : '#f43f5e'}
          />
          <StatRow
            label="Total Signals"
            value={stats.total_signals.toLocaleString()}
          />
          <StatRow
            label="Total Fees"
            value={fmt$(stats.total_fees_paid)}
            color="rgba(255,255,255,0.35)"
          />

          {/* Monte Carlo */}
          {stats.monte_carlo && (
            <>
              <SectionLabel>Monte Carlo</SectionLabel>
              <StatRow
                label="P95 Max Drawdown"
                value={fmtPct(stats.monte_carlo.p95_max_dd_pct)}
                color={stats.monte_carlo.p95_max_dd_pct <= 15 ? '#2dd4bf' : '#f59e0b'}
                sub="worst-case in 95% of sequences"
              />
              <StatRow
                label="Profitable Runs"
                value={fmtPct(stats.monte_carlo.pct_profitable)}
                color={stats.monte_carlo.pct_profitable >= 90 ? '#2dd4bf' : '#f59e0b'}
              />
              <StatRow
                label="Median DD"
                value={fmtPct(stats.monte_carlo.median_max_dd_pct)}
                color="rgba(255,255,255,0.5)"
              />
            </>
          )}

          {/* Footer */}
          <div
            className="mt-3 pt-2 flex items-center justify-between"
            style={{ borderTop: '1px solid rgba(255,255,255,0.04)' }}
          >
            <span className="text-[10px] text-white/20">Last updated</span>
            <span className="text-[10px] text-white/30">{relTime}</span>
          </div>
        </>
      )}
    </div>
  )

  return tile
}