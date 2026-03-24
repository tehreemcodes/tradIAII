'use client'

import { useEffect, useState } from 'react'
import { api, Signal, signalColor, fmtPrice, fmtConfidence } from '@/lib/api'

// ── Confidence Arc SVG ─────────────────────────────────────────
function ConfidenceArc({ value, color }: { value: number; color: string }) {
  const r    = 24
  const cx   = 32
  const cy   = 32
  const circ = 2 * Math.PI * r
  const arc  = (value / 100) * circ * 0.75

  return (
    <svg width={64} height={64} viewBox="0 0 64 64">
      {/* Track */}
      <circle
        cx={cx} cy={cy} r={r}
        fill="none" stroke="rgba(255,255,255,0.07)" strokeWidth={4}
        strokeDasharray={`${circ * 0.75} ${circ}`}
        strokeDashoffset={-circ * 0.125}
        strokeLinecap="round"
        transform="rotate(135 32 32)"
      />
      {/* Fill */}
      <circle
        cx={cx} cy={cy} r={r}
        fill="none" stroke={color} strokeWidth={4}
        strokeDasharray={`${arc} ${circ}`}
        strokeDashoffset={-circ * 0.125}
        strokeLinecap="round"
        transform="rotate(135 32 32)"
        style={{ filter: `drop-shadow(0 0 5px ${color}88)` }}
      />
      <text
        x={cx} y={cy + 5}
        textAnchor="middle"
        fill="white"
        fontSize={11}
        fontFamily="JetBrains Mono, monospace"
        fontWeight={700}
      >
        {Math.round(value)}%
      </text>
    </svg>
  )
}

// ── RR Bar ─────────────────────────────────────────────────────
function RRBar({ entry, sl, tp, direction }: {
  entry:     number
  sl:        number
  tp:        number
  direction: 'BUY' | 'SELL'
}) {
  const isBuy   = direction === 'BUY'
  const risk    = Math.abs(entry - sl)
  const reward  = Math.abs(tp - entry)
  const rrRatio = risk > 0 ? (reward / risk).toFixed(1) : '—'

  return (
    <div className="px-3 py-2 rounded-lg bg-black/30 border border-white/5">
      <div className="flex justify-between mb-2">
        <span className="text-white/30 text-[11px]">Risk : Reward</span>
        <span className="mono text-xs font-bold text-white">1 : {rrRatio}</span>
      </div>
      <div className="flex gap-0.5 h-1.5 rounded overflow-hidden">
        <div className="flex-1 rounded" style={{ background: '#f43f5e66' }} />
        <div
          className="rounded"
          style={{
            flex: parseFloat(rrRatio) || 2,
            background: '#2dd4bf66',
          }}
        />
      </div>
    </div>
  )
}

// ── ICT Checklist ──────────────────────────────────────────────
function ICTChecklist({ signal }: { signal: Signal }) {
  const htf    = signal.htf_bias
  const pat    = signal.pattern
  const isBuy  = signal.signal === 'BUY'
  const isSell = signal.signal === 'SELL'

  const checks = [
    { label: 'Daily Bias Aligned',       pass: htf ? htf.d1 === (isBuy ? 1 : -1) : false },
    { label: '4H Bias Aligned',          pass: htf ? htf.h4 === (isBuy ? 1 : -1) : false },
    { label: 'Full HTF Confluence',      pass: htf?.full_confluence ?? false },
    { label: 'Swing Detected',           pass: pat?.swing_price != null },
    { label: 'FVG Present',              pass: pat?.fvg_top != null },
    { label: 'Signal Generated',         pass: isBuy || isSell },
    { label: 'Confidence >= 60%',        pass: signal.confidence >= 0.60 },
    { label: 'SL at Structure',          pass: signal.sl != null },
  ]

  const passed = checks.filter(c => c.pass).length

  return (
    <div className="rounded-xl border border-white/5 bg-white/[0.03] p-4">
      <p className="text-[10px] font-semibold tracking-widest text-white/30 uppercase mb-3">
        ICT Checklist
      </p>

      {checks.map(item => (
        <div
          key={item.label}
          className="flex items-center justify-between py-2 border-b border-white/[0.04] last:border-0"
        >
          <span
            className="text-xs"
            style={{ color: item.pass ? 'rgba(255,255,255,0.65)' : 'rgba(255,255,255,0.25)' }}
          >
            {item.label}
          </span>
          <div
            className="w-5 h-5 rounded-full flex items-center justify-center text-[10px] flex-shrink-0"
            style={{
              background: item.pass ? 'rgba(45,212,191,0.12)' : 'rgba(244,63,94,0.1)',
              border: `1px solid ${item.pass ? 'rgba(45,212,191,0.3)' : 'rgba(244,63,94,0.25)'}`,
              color: item.pass ? '#2dd4bf' : '#f43f5e',
            }}
          >
            {item.pass ? '✓' : '✗'}
          </div>
        </div>
      ))}

      <div
        className="mt-3 px-3 py-2 rounded-lg flex items-center justify-between"
        style={{
          background: passed >= 6 ? 'rgba(45,212,191,0.07)' : 'rgba(244,63,94,0.05)',
          border: `1px solid ${passed >= 6 ? 'rgba(45,212,191,0.2)' : 'rgba(244,63,94,0.15)'}`,
        }}
      >
        <span
          className="text-xs font-semibold"
          style={{ color: passed >= 6 ? '#2dd4bf' : '#f43f5e' }}
        >
          {passed} / {checks.length} passed
        </span>
        <span
          className="text-[10px] px-2 py-0.5 rounded font-bold"
          style={{
            background: passed >= 6 ? 'rgba(45,212,191,0.15)' : 'rgba(244,63,94,0.12)',
            color: passed >= 6 ? '#2dd4bf' : '#f43f5e',
          }}
        >
          {passed >= 7 ? 'A+ SETUP' : passed >= 5 ? 'GOOD' : 'LOW QUALITY'}
        </span>
      </div>
    </div>
  )
}

// ── Main SignalCard ────────────────────────────────────────────
export default function SignalCard({ capital = 10000, timeframe = '1h' }: { capital?: number, timeframe?: string }) {
  const [signal,  setSignal]  = useState<Signal | null>(null)
  const [loading, setLoading] = useState(true)
  const [error,   setError]   = useState<string | null>(null)

  useEffect(() => {
    const fetch = async () => {
      try {
        setLoading(true) // add loading state wrapper for better UX when switching TFs
        const data = await api.signal(capital, timeframe)
        setSignal(data)
        setError(null)
      } catch (e: any) {
        setError(e.message)
      } finally {
        setLoading(false)
      }
    }

    fetch()
    const iv = setInterval(fetch, 60_000)   // refresh every minute
    return () => clearInterval(iv)
  }, [capital, timeframe])

  if (loading) {
    return (
      <div className="flex items-center justify-center h-40 text-white/20 text-xs">
        Loading signal...
      </div>
    )
  }

  if (error) {
    return (
      <div className="rounded-xl border border-rose/20 bg-rose/5 p-4 text-rose/70 text-xs">
        {error}
      </div>
    )
  }

  if (!signal) return null

  const color    = signalColor(signal.signal)
  const isActive = signal.signal !== 'NO TRADE'

  return (
    <div className="flex flex-col gap-3">

      {/* Hero signal card */}
      <div
        className="rounded-xl p-4"
        style={{
          background:  `linear-gradient(140deg, ${color}18 0%, rgba(13,17,23,0.9) 60%)`,
          border:      `1px solid ${color}30`,
          borderTop:   `3px solid ${color}`,
        }}
      >
        {/* Header */}
        <p className="text-[10px] font-semibold tracking-widest text-white/30 uppercase mb-3">
          Current Signal
        </p>

        <div className="flex items-start justify-between mb-4">
          <div>
            <div
              className="text-3xl font-black tracking-tight leading-none"
              style={{
                color,
                textShadow: `0 0 24px ${color}66`,
              }}
            >
              {signal.signal}
            </div>
            <div className="text-white/30 text-xs mt-1.5">
              {signal.pair} · {signal.timeframe} · {
                new Date(signal.timestamp).toLocaleTimeString('en-US', {
                  hour: '2-digit', minute: '2-digit'
                })
              }
            </div>
          </div>
          <ConfidenceArc value={signal.confidence * 100} color={color} />
        </div>

        {/* Levels */}
        {isActive && signal.entry ? (
          <div className="flex flex-col gap-1.5">
            {[
              { label: 'Entry',        value: signal.entry, color: 'rgba(255,255,255,0.8)' },
              { label: 'Stop Loss',    value: signal.sl,    color: '#f43f5e' },
              { label: 'Take Profit',  value: signal.tp,    color: '#2dd4bf' },
            ].map(row => (
              <div
                key={row.label}
                className="flex items-center justify-between px-3 py-2 rounded-lg"
                style={{ background: 'rgba(0,0,0,0.3)', border: '1px solid rgba(255,255,255,0.05)' }}
              >
                <span className="text-white/30 text-[11px]">{row.label}</span>
                <span
                  className="mono text-xs font-bold"
                  style={{ color: row.color }}
                >
                  ${fmtPrice(row.value)}
                </span>
              </div>
            ))}

            {signal.entry && signal.sl && signal.tp && (
              <RRBar
                entry={signal.entry}
                sl={signal.sl}
                tp={signal.tp}
                direction={signal.signal as 'BUY' | 'SELL'}
              />
            )}

            {/* Position sizing */}
            <div className="grid grid-cols-2 gap-1.5 mt-1">
              {[
                { label: 'Risk Amount',   value: `$${fmtPrice(signal.risk_amount)}` },
                { label: 'Position Size', value: signal.position_size.toFixed(5) },
              ].map(s => (
                <div
                  key={s.label}
                  className="px-3 py-2 rounded-lg text-center"
                  style={{ background: 'rgba(0,0,0,0.3)', border: '1px solid rgba(255,255,255,0.05)' }}
                >
                  <div className="mono text-xs font-semibold text-white">{s.value}</div>
                  <div className="text-[10px] text-white/25 mt-0.5">{s.label}</div>
                </div>
              ))}
            </div>
          </div>
        ) : (
          <div
            className="px-4 py-4 rounded-lg text-center"
            style={{ background: 'rgba(0,0,0,0.3)', border: '1px solid rgba(255,255,255,0.05)' }}
          >
            <div className="text-white/25 text-xs leading-relaxed">
              {signal.error
                ? signal.error
                : 'No high-confidence setup detected.\nWaiting for A+ entry...'}
            </div>
          </div>
        )}
      </div>

      {/* ICT Checklist */}
      {signal && <ICTChecklist signal={signal} />}

    </div>
  )
}
