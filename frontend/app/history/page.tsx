'use client'

import { useEffect, useState } from 'react'
import Link from 'next/link'
import { api, Trade, TradesResponse, fmtPrice, signalColor } from '@/lib/api'

// Calculation for ROI based on 20x leverage
const calcROI = (trade: Trade) => {
  if (trade.pnl == null || trade.entry_price == null || trade.size == null) return 0
  const marginUsed = (trade.entry_price * trade.size) / 20
  return (trade.pnl / marginUsed) * 100
}

export default function HistoryPage() {
  const [data, setData] = useState<TradesResponse | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchTrades = async () => {
      try {
        const res = await api.trades()
        setData(res)
      } catch (err) {
        console.error('Failed to fetch trades', err)
      } finally {
        setLoading(false)
      }
    }
    fetchTrades()
  }, [])

  if (loading) {
    return (
      <div className="flex h-screen items-center justify-center bg-navy-900">
        <div className="flex flex-col items-center gap-4">
          <div className="w-12 h-12 rounded-full border-4 border-teal-500/20 border-t-teal-500 animate-spin" />
          <p className="text-white/40 text-xs mono tracking-widest uppercase">Loading History...</p>
        </div>
      </div>
    )
  }

  const trades = data?.closed || []

  return (
    <div className="min-h-screen bg-navy-900 text-white font-sans selection:bg-teal-500/30">
      
      {/* Header */}
      <header className="h-[64px] flex items-center justify-between px-8 border-b border-white/5 bg-navy-950/50 sticky top-0 backdrop-blur-xl z-50">
        <div className="flex items-center gap-4">
          <Link href="/" className="group flex items-center gap-2 text-white/40 hover:text-white transition-colors duration-200">
            <span className="text-lg">←</span>
            <span className="text-xs font-bold tracking-wider uppercase">Dashboard</span>
          </Link>
          <div className="h-4 w-[1px] bg-white/10" />
          <h1 className="text-sm font-black tracking-widest uppercase bg-gradient-to-r from-teal-400 to-blue-500 bg-clip-text text-transparent">
            Trade History
          </h1>
        </div>
        
        <div className="flex items-center gap-6">
          <div className="flex flex-col items-end">
            <p className="text-[9px] uppercase tracking-widest text-white/20 font-bold">Total P&L</p>
            <p className={`mono text-sm font-bold ${data?.stats.total_pnl && data.stats.total_pnl >= 0 ? 'text-teal-400' : 'text-rose-500'}`}>
              {data?.stats.total_pnl ? (data.stats.total_pnl >= 0 ? '+' : '') : ''}${data?.stats.total_pnl?.toLocaleString() ?? '0.00'}
            </p>
          </div>
          <div className="flex flex-col items-end">
            <p className="text-[9px] uppercase tracking-widest text-white/20 font-bold">Win Rate</p>
            <p className="mono text-sm font-bold text-white/80">
              {data?.stats.win_rate_pct?.toFixed(1) ?? '0.0'}%
            </p>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto p-8">
        <div className="rounded-2xl border border-white/5 bg-white/[0.02] overflow-hidden shadow-2xl">
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="bg-white/[0.03]">
                  {[
                    'Time (UTC)', 'Symbol', 'Side', 'Size', 'Entry', 'Exit', 'PnL', 'ROI (20x)', 'Outcome'
                  ].map(h => (
                    <th key={h} className="px-6 py-4 text-[10px] font-black uppercase tracking-widest text-white/30 border-b border-white/5">
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-white/[0.02]">
                {trades.length === 0 ? (
                  <tr>
                    <td colSpan={9} className="px-6 py-12 text-center text-white/20 mono text-xs italic">
                      No closed trades in history
                    </td>
                  </tr>
                ) : (
                  [...trades].reverse().map((t, idx) => {
                    const roi = calcROI(t)
                    const isWin = t.outcome === 'TP'
                    const pnlColor = (t.pnl || 0) >= 0 ? 'text-teal-400' : 'text-rose-500'
                    
                    return (
                      <tr key={t.id + idx} className="hover:bg-white/[0.01] transition-colors group">
                        <td className="px-6 py-4 mono text-[11px] text-white/40 whitespace-nowrap">
                          {t.closed_at ? new Date(t.closed_at).toISOString().replace('T', ' ').slice(0, 19) : '—'}
                        </td>
                        <td className="px-6 py-4 font-bold text-xs tracking-tight text-white/80">
                          {t.symbol}
                        </td>
                        <td className="px-6 py-4">
                          <span className={`text-[10px] font-black px-2 py-1 rounded-md uppercase tracking-widest ${
                            t.direction === 'BUY' ? 'bg-teal-500/10 text-teal-400 border border-teal-500/20' : 'bg-rose-500/10 text-rose-400 border border-rose-500/20'
                          }`}>
                            {t.direction}
                          </span>
                        </td>
                        <td className="px-6 py-4 mono text-[11px] text-white/60">
                          {t.size?.toFixed(4)}
                        </td>
                        <td className="px-6 py-4 mono text-[11px] text-white/60">
                          {fmtPrice(t.entry_price)}
                        </td>
                        <td className="px-6 py-4 mono text-[11px] text-white/60">
                          {fmtPrice(t.close_price)}
                        </td>
                        <td className={`px-6 py-4 mono text-xs font-bold ${pnlColor}`}>
                          {(t.pnl || 0) >= 0 ? '+' : ''}{t.pnl?.toFixed(2)}
                        </td>
                        <td className={`px-6 py-4 mono text-xs font-bold ${pnlColor}`}>
                          {roi >= 0 ? '+' : ''}{roi.toFixed(1)}%
                        </td>
                        <td className="px-6 py-4">
                          <div className="flex items-center gap-2">
                             <span className={`w-1.5 h-1.5 rounded-full ${isWin ? 'bg-teal-400 shadow-[0_0_8px_rgba(45,212,191,0.5)]' : 'bg-rose-500 shadow-[0_0_8px_rgba(244,63,94,0.5)]'}`} />
                             <span className={`text-[10px] font-black uppercase tracking-widest ${isWin ? 'text-teal-400' : 'text-rose-500'}`}>
                               {t.outcome}
                             </span>
                          </div>
                        </td>
                      </tr>
                    )
                  })
                )}
              </tbody>
            </table>
          </div>
        </div>
      </main>
    </div>
  )
}
