'use client'

import { useEffect, useState } from 'react'
import { api, ZonesResponse, FVGZone } from '@/lib/api'

function ZoneRow({ zone }: { zone: FVGZone }) {
  const bull  = zone.direction === 'bullish'
  const color = bull ? '#2dd4bf' : '#f43f5e'

  return (
    <div
      className="px-3 py-2 mb-1.5 rounded-lg"
      style={{
        background:  'rgba(255,255,255,0.02)',
        border:      '1px solid rgba(255,255,255,0.05)',
        borderLeft:  `3px solid ${color}`,
        opacity:     zone.filled ? 0.4 : 1,
      }}
    >
      <div className="flex items-center justify-between">
        <span className="text-xs font-semibold" style={{ color }}>
          {bull ? 'Bull' : 'Bear'} FVG
        </span>
        <span className="text-[9px] text-white/25">
          {zone.filled ? 'Filled' : '1H'}
        </span>
      </div>
      <div className="mono text-[10px] text-white/30 mt-0.5">
        {zone.bot.toLocaleString()} – {zone.top.toLocaleString()}
      </div>
    </div>
  )
}

export default function ConfluenceChecklist() {
  const [zones,   setZones]   = useState<FVGZone[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetch = async () => {
      try {
        const data = await api.zones()
        setZones(data.zones.filter(z => !z.filled).slice(-8))
      } catch {
        // Zones are a best-effort overlay
      } finally {
        setLoading(false)
      }
    }
    fetch()
    const iv = setInterval(fetch, 120_000)
    return () => clearInterval(iv)
  }, [])

  return (
    <div className="rounded-xl border border-white/5 bg-white/[0.03] p-4">
      <p className="text-[10px] font-semibold tracking-widest text-white/30 uppercase mb-3">
        Active Zones
      </p>

      {loading && (
        <div className="text-white/20 text-xs text-center py-3">Loading...</div>
      )}

      {!loading && zones.length === 0 && (
        <div className="text-white/20 text-xs text-center py-3">No active zones</div>
      )}

      {zones.map((z, i) => <ZoneRow key={i} zone={z} />)}
    </div>
  )
}
