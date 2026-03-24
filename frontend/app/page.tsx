'use client'

import { useEffect, useState } from 'react'
import dynamic from 'next/dynamic'
import { api, Signal, HealthResponse } from '@/lib/api'
import HTFBiasPanel        from '@/components/HTFBiasPanel'
import SignalCard          from '@/components/SignalCard'
import RiskPanel           from '@/components/RiskPanel'
import ConfluenceChecklist from '@/components/ConfluenceChecklist'
import ExchangePanel       from '@/components/ExchangePanel'
import StatusBar           from '@/components/StatusBar'


// Chart uses browser APIs — must be client-only
const Chart = dynamic(() => import('@/components/Chart'), { ssr: false })

// ── Topbar ─────────────────────────────────────────────────────
function Topbar({
  health,
  price,
  priceUp,
}: {
  health:  HealthResponse | null
  price:   number | null
  priceUp: boolean
}) {
  const [time, setTime] = useState(new Date())

  useEffect(() => {
    const iv = setInterval(() => setTime(new Date()), 1000)
    return () => clearInterval(iv)
  }, [])

  const sessions = [
    { name: 'London',   active: (() => { const h = time.getUTCHours(); return h >= 7 && h < 10 })() },
    { name: 'New York', active: (() => { const h = time.getUTCHours(); return h >= 12 && h < 15 })() },
    { name: 'Asia',     active: (() => { const h = time.getUTCHours(); return h >= 0 && h < 3 })() },
  ]

  return (
    <header
      className="h-[52px] flex items-center justify-between px-6 flex-shrink-0"
      style={{
        background:   'rgba(255,255,255,0.03)',
        borderBottom: '1px solid rgba(255,255,255,0.06)',
        backdropFilter: 'blur(20px)',
      }}
    >
      {/* Logo */}
      <div className="flex items-center gap-3">
        <div
          className="w-8 h-8 rounded-lg flex items-center justify-center text-base font-black"
          style={{ background: 'linear-gradient(135deg, #2dd4bf, #3b82f6)', color: '#0d1117' }}
        >
          ⟁
        </div>
        <span className="mono font-bold text-sm tracking-wide text-white">
          trad<span style={{ color: '#2dd4bf' }}>IA</span>
        </span>
        <div
          className="px-2 py-0.5 rounded-full text-[9px] font-bold tracking-wider"
          style={{
            background: health?.model_ready ? 'rgba(45,212,191,0.12)' : 'rgba(244,63,94,0.12)',
            border:     `1px solid ${health?.model_ready ? 'rgba(45,212,191,0.25)' : 'rgba(244,63,94,0.25)'}`,
            color:      health?.model_ready ? '#2dd4bf' : '#f43f5e',
          }}
        >
          {health?.model_ready ? 'LIVE' : 'NO MODEL'}
        </div>
      </div>

      {/* Price */}
      <div className="flex items-center gap-5">
        <span className="text-white/30 text-xs">BTC / USDT</span>
        {price && (
          <span
            className="mono text-xl font-bold transition-colors duration-300"
            style={{
              color:      priceUp ? '#2dd4bf' : '#f43f5e',
              textShadow: `0 0 16px ${priceUp ? 'rgba(45,212,191,0.3)' : 'rgba(244,63,94,0.3)'}`,
            }}
          >
            ${price.toLocaleString('en-US', { minimumFractionDigits: 2 })}
          </span>
        )}
      </div>

      {/* Sessions + Clock */}
      <div className="flex items-center gap-5">
        {sessions.map(s => (
          <div key={s.name} className="flex items-center gap-1.5">
            <div
              className="w-1.5 h-1.5 rounded-full transition-all duration-300"
              style={{
                background:  s.active ? '#2dd4bf' : 'rgba(255,255,255,0.15)',
                boxShadow:   s.active ? '0 0 6px #2dd4bf' : 'none',
              }}
            />
            <span
              className="text-xs transition-colors duration-300"
              style={{ color: s.active ? 'rgba(255,255,255,0.7)' : 'rgba(255,255,255,0.25)' }}
            >
              {s.name}
            </span>
          </div>
        ))}
        <span
          className="mono text-xs border-l pl-5"
          style={{ color: 'rgba(255,255,255,0.25)', borderColor: 'rgba(255,255,255,0.08)' }}
        >
          {time.toUTCString().slice(17, 25)} UTC
        </span>
      </div>
    </header>
  )
}

// ── Main Dashboard ─────────────────────────────────────────────
export default function Dashboard() {
  const [activeTf, setActiveTf] = useState<string>('1h')
  const [health,  setHealth]  = useState<HealthResponse | null>(null)
  const [signal,  setSignal]  = useState<Signal | null>(null)
  const [price,   setPrice]   = useState<number | null>(null)
  const [priceUp, setPriceUp] = useState(true)

  // Poll health
  useEffect(() => {
    const fetch = async () => {
      try { setHealth(await api.health()) } catch { /* backend offline */ }
    }
    fetch()
    const iv = setInterval(fetch, 30_000)
    return () => clearInterval(iv)
  }, [])

  // Poll signal (also drives price)
  useEffect(() => {
    const fetch = async () => {
      try {
        const s = await api.signal(undefined, activeTf)
        setSignal(s)
        if (s.entry) {
          setPriceUp(prev => s.entry! >= (price ?? s.entry!))
          setPrice(s.entry)
        }
      } catch { /* API not ready */ }
    }
    fetch()
    const iv = setInterval(fetch, 60_000)
    return () => clearInterval(iv)
  }, [price, activeTf])

  return (
    <div className="flex flex-col h-screen bg-navy-900">

      <Topbar health={health} price={price} priceUp={priceUp} />

      {/* 3-Column Layout */}
      <div
        className="flex-1 overflow-y-auto p-4"
      >
        <StatusBar />
        <div
          className="grid"
          style={{ gridTemplateColumns: '256px 1fr 280px', height: 'calc(100% - 68px)' }}
        >

        {/* ── LEFT SIDEBAR ──────────────────────────────── */}
        <aside
          className="overflow-y-auto p-4 flex flex-col gap-3"
          style={{ borderRight: '1px solid rgba(255,255,255,0.05)' }}
        >
          <HTFBiasPanel />
          <ConfluenceChecklist />

          {/* Model info tile */}
          <div
            className="rounded-xl border border-white/5 bg-white/[0.03] p-4"
          >
            <p className="text-[10px] font-semibold tracking-widest text-white/30 uppercase mb-3">
              Model
            </p>
            {[
              { label: 'Status',     value: health?.model_ready ? 'Ready' : 'Not trained', ok: health?.model_ready },
              { label: 'Signal TF',  value: health?.signal_tf   ?? '1H' },
              { label: 'Version',    value: health?.version      ?? '—' },
            ].map(r => (
              <div
                key={r.label}
                className="flex justify-between py-1.5 border-b border-white/[0.04] last:border-0"
              >
                <span className="text-white/30 text-xs">{r.label}</span>
                <span
                  className="text-xs font-medium"
                  style={{
                    color: r.ok === undefined
                      ? 'rgba(255,255,255,0.6)'
                      : r.ok ? '#2dd4bf' : '#f43f5e',
                  }}
                >
                  {r.value}
                </span>
              </div>
            ))}
          </div>
        </aside>

        {/* ── CENTER ────────────────────────────────────── */}
        <main className="flex flex-col overflow-hidden">

          {/* Chart */}
          <div
            className="flex-1 bg-navy-900"
            style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}
          >
            <Chart 
              activeSignal={signal} 
              activeTf={activeTf as any} 
              onTfChange={(tf) => setActiveTf(tf)} 
            />
          </div>

          {/* Bottom info strip */}
          <div
            className="h-[48px] flex items-center gap-6 px-5 flex-shrink-0"
            style={{ background: 'rgba(255,255,255,0.02)' }}
          >
            {signal && signal.signal !== 'NO TRADE' && signal.entry && (
              <>
                {[
                  { label: 'Entry',  v: signal.entry, c: 'rgba(255,255,255,0.6)' },
                  { label: 'SL',     v: signal.sl,    c: '#f43f5e' },
                  { label: 'TP',     v: signal.tp,    c: '#2dd4bf' },
                ].map(l => (
                  <div key={l.label} className="flex items-center gap-2">
                    <span className="text-[10px] text-white/20">{l.label}</span>
                    <span className="mono text-xs font-semibold" style={{ color: l.c }}>
                      {l.v ? `$${l.v.toLocaleString('en-US', { minimumFractionDigits: 2 })}` : '—'}
                    </span>
                  </div>
                ))}
                <div className="flex items-center gap-2">
                  <span className="text-[10px] text-white/20">Confidence</span>
                  <span
                    className="mono text-xs font-semibold"
                    style={{
                      color: signal.confidence >= 0.7 ? '#2dd4bf' :
                             signal.confidence >= 0.6 ? '#f59e0b' : '#f43f5e',
                    }}
                  >
                    {(signal.confidence * 100).toFixed(1)}%
                  </span>
                </div>
              </>
            )}
            {(!signal || signal.signal === 'NO TRADE') && (
              <span className="text-white/15 text-xs">
                No active signal — monitoring market...
              </span>
            )}
          </div>
        </main>

        {/* ── RIGHT SIDEBAR ─────────────────────────────── */}
        <aside
          className="overflow-y-auto p-4 flex flex-col gap-3"
          style={{ borderLeft: '1px solid rgba(255,255,255,0.05)' }}
        >
          <SignalCard capital={10000} timeframe={activeTf} />
          <RiskPanel />
          <ExchangePanel />
        </aside>

      </div>  {/* inner grid */}
      </div>  {/* outer flex-1 */}
    </div>
  )
}
