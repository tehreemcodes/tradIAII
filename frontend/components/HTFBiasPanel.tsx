'use client'

import { useEffect, useState } from 'react'
import { api, HTFBiasResponse, biasColor, biasLabel } from '@/lib/api'

interface KillzoneStatus {
  name:   string
  utc:    string
  active: boolean
  next:   string
}

function getKillzones(): KillzoneStatus[] {
  const now = new Date()
  const h   = now.getUTCHours()
  const m   = now.getUTCMinutes()
  const dec = h + m / 60

  const inRange = (start: number, end: number) =>
    dec >= start && dec < end

  const nextIn = (start: number): string => {
    const diff = (start - dec + 24) % 24
    const hrs  = Math.floor(diff)
    const mins = Math.round((diff - hrs) * 60)
    if (diff < 0.1) return 'Now'
    return hrs > 0 ? `in ${hrs}h ${mins}m` : `in ${mins}m`
  }

  return [
    {
      name:   'Asia Open',
      utc:    '00:00 – 03:00',
      active: inRange(0, 3),
      next:   inRange(0, 3) ? 'Active' : nextIn(0),
    },
    {
      name:   'London Open',
      utc:    '07:00 – 10:00',
      active: inRange(7, 10),
      next:   inRange(7, 10) ? 'Active' : nextIn(7),
    },
    {
      name:   'NY Open',
      utc:    '12:00 – 15:00',
      active: inRange(12, 15),
      next:   inRange(12, 15) ? 'Active' : nextIn(12),
    },
    {
      name:   'NY Close',
      utc:    '19:00 – 21:00',
      active: inRange(19, 21),
      next:   inRange(19, 21) ? 'Active' : nextIn(19),
    },
  ]
}

export default function HTFBiasPanel() {
  const [bias, setBias]       = useState<HTFBiasResponse | null>(null)
  const [kzs, setKzs]         = useState<KillzoneStatus[]>(getKillzones())
  const [loading, setLoading] = useState(true)
  const [error, setError]     = useState<string | null>(null)

  useEffect(() => {
    const fetchBias = async () => {
      try {
        const data = await api.htfBias()
        setBias(data)
        setError(null)
      } catch (e: any) {
        setError(e.message)
      } finally {
        setLoading(false)
      }
    }

    fetchBias()
    const iv1 = setInterval(fetchBias, 60_000)     // refresh every minute
    const iv2 = setInterval(() => setKzs(getKillzones()), 30_000)
    return () => { clearInterval(iv1); clearInterval(iv2) }
  }, [])

  const biasRows = bias ? [
    { label: 'Daily',  value: bias.daily },
    { label: '4 Hour', value: bias.h4    },
    { label: '1 Hour', value: bias.h1    },
  ] : []

  const verdictColor =
    bias?.verdict === 'BULLISH' ? '#2dd4bf' :
    bias?.verdict === 'BEARISH' ? '#f43f5e' : '#64748b'

  return (
    <div className="flex flex-col gap-3">

      {/* HTF Bias */}
      <div className="rounded-xl border border-white/5 bg-white/[0.03] p-4">
        <p className="text-[10px] font-semibold tracking-widest text-white/30 uppercase mb-3">
          HTF Bias
        </p>

        {loading && (
          <div className="text-white/20 text-xs text-center py-4">Loading...</div>
        )}
        {error && (
          <div className="text-rose/70 text-xs">{error}</div>
        )}

        {!loading && !error && biasRows.map(row => (
          <div
            key={row.label}
            className="flex items-center justify-between px-3 py-2 mb-1.5 rounded-lg bg-white/[0.03] border"
            style={{ borderColor: `${biasColor(row.value)}22` }}
          >
            <span className="text-white/40 text-xs">{row.label}</span>
            <div className="flex items-center gap-2">
              <span style={{ color: biasColor(row.value) }} className="text-[10px]">
                {row.value === 1 ? '▲' : row.value === -1 ? '▼' : '—'}
              </span>
              <span
                className="text-xs font-semibold"
                style={{ color: biasColor(row.value) }}
              >
                {biasLabel(row.value)}
              </span>
            </div>
          </div>
        ))}

        {bias && (
          <div
            className="mt-2 px-3 py-2 rounded-lg text-center text-xs font-semibold"
            style={{
              background: `${verdictColor}12`,
              border: `1px solid ${verdictColor}30`,
              color: verdictColor,
            }}
          >
            {bias.verdict === 'BULLISH' && '✓ Full Confluence — Longs Preferred'}
            {bias.verdict === 'BEARISH' && '✓ Full Confluence — Shorts Preferred'}
            {bias.verdict === 'MIXED'   && '⚠ Mixed — Wait for Clarity'}
          </div>
        )}
      </div>

      {/* Killzones */}
      <div className="rounded-xl border border-white/5 bg-white/[0.03] p-4">
        <p className="text-[10px] font-semibold tracking-widest text-white/30 uppercase mb-3">
          Killzones
        </p>

        {kzs.map(kz => (
          <div
            key={kz.name}
            className="flex items-center justify-between py-2 border-b border-white/[0.04] last:border-0"
          >
            <div>
              <div className="flex items-center gap-1.5">
                {kz.active && (
                  <span className="w-1.5 h-1.5 rounded-full bg-teal animate-pulse-slow" />
                )}
                <span
                  className="text-xs"
                  style={{ color: kz.active ? 'white' : 'rgba(255,255,255,0.3)' }}
                >
                  {kz.name}
                </span>
              </div>
              <span className="mono text-[10px] text-white/20">{kz.utc}</span>
            </div>
            <span
              className="text-[10px] font-semibold mono"
              style={{ color: kz.active ? '#2dd4bf' : 'rgba(255,255,255,0.2)' }}
            >
              {kz.next}
            </span>
          </div>
        ))}
      </div>

    </div>
  )
}
