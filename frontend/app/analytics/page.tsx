'use client'

import { useEffect, useState } from 'react'
import Link from 'next/link'
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area 
} from 'recharts'
import { 
  ArrowLeft, TrendingUp, DollarSign, Activity, Percent, Shield, AlertTriangle 
} from 'lucide-react'
import { api, AnalyticsSummary, AnalyticsTrade, fmtPrice } from '@/lib/api'

export default function AnalyticsPage() {
  const [summary, setSummary] = useState<AnalyticsSummary | null>(null)
  const [trades, setTrades] = useState<AnalyticsTrade[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [sumRes, tradesRes] = await Promise.all([
          api.analyticsSummary(),
          api.analyticsTrades(50)
        ])
        setSummary(sumRes)
        setTrades(tradesRes.trades)
      } catch (err) {
        console.error('Failed to fetch analytics data', err)
      } finally {
        setLoading(false)
      }
    }
    fetchData()
  }, [])

  if (loading) {
    return (
      <div className="flex h-screen items-center justify-center bg-navy-950">
        <div className="flex flex-col items-center gap-4">
          <div className="w-12 h-12 rounded-full border-4 border-teal-500/20 border-t-teal-500 animate-spin" />
          <p className="text-white/40 text-[10px] mono tracking-[0.2em] uppercase">Auditing Performance...</p>
        </div>
      </div>
    )
  }

  // Process history for Chart
  let cumulative = 0
  const chartData = summary?.history.map((h, i) => {
    cumulative += h.pnl
    return {
      name: i + 1,
      pnl: cumulative,
      date: new Date(h.at).toLocaleDateString([], { month: 'short', day: 'numeric' })
    }
  }) || []

  const stats = [
    { label: 'Net P&L', value: `${(summary?.net_pnl ?? 0) >= 0 ? '+' : ''}${(summary?.net_pnl ?? 0).toFixed(2)}`, sub: 'USDT', icon: DollarSign, color: (summary?.net_pnl || 0) >= 0 ? 'text-teal-400' : 'text-rose-500' },
    { label: 'Win Rate', value: `${(summary?.win_rate_pct ?? 0).toFixed(1)}`, sub: '%', icon: Percent, color: 'text-blue-400' },
    { label: 'Total Fees', value: `-${(summary?.total_fees ?? 0).toFixed(2)}`, sub: 'USDT', icon: Activity, color: 'text-amber-400' },
    { label: 'Total Trades', value: `${summary?.total_trades ?? 0}`, sub: 'Executed', icon: TrendingUp, color: 'text-indigo-400' },
  ]

  return (
    <div className="min-h-screen bg-navy-950 text-white font-sans selection:bg-teal-500/30 pb-12">
      
      {/* Header */}
      <header className="h-[64px] flex items-center justify-between px-8 border-b border-white/5 bg-navy-900/50 sticky top-0 backdrop-blur-xl z-50">
        <div className="flex items-center gap-4">
          <Link href="/" className="group flex items-center gap-2 text-white/40 hover:text-white transition-colors duration-200">
            <ArrowLeft className="w-4 h-4" />
            <span className="text-[10px] font-black tracking-[0.1em] uppercase">Dashboard</span>
          </Link>
          <div className="h-4 w-[1px] bg-white/10" />
          <h1 className="text-xs font-black tracking-[0.2em] uppercase text-white/90">
            Performance <span className="text-teal-400">Analytics</span>
          </h1>
        </div>
      </header>

      <main className="max-w-[1400px] mx-auto p-8 grid gap-8">
        
        {/* Top Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {stats.map((s, idx) => (
            <div key={idx} className="bg-white/[0.03] border border-white/5 rounded-2xl p-6 flex items-center gap-5 transition-all hover:bg-white/[0.05] hover:border-white/10">
              <div className={`p-3 rounded-xl bg-white/[0.03] ${s.color}`}>
                <s.icon className="w-5 h-5" />
              </div>
              <div>
                <p className="text-[10px] font-bold uppercase tracking-widest text-white/20 mb-1">{s.label}</p>
                <p className={`text-2xl font-black mono leading-none ${s.color}`}>
                  {s.value}<span className="text-[10px] ml-1 opacity-40">{s.sub}</span>
                </p>
              </div>
            </div>
          ))}
        </div>

        {/* Chart Section */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          <div className="lg:col-span-2 bg-white/[0.02] border border-white/5 rounded-3xl p-8 relative overflow-hidden group">
            <div className="absolute top-0 right-0 p-8 opacity-5">
              <TrendingUp className="w-48 h-48" />
            </div>
            <h2 className="text-[10px] font-black uppercase tracking-[0.2em] text-white/30 mb-8 flex items-center gap-2">
              <Activity className="w-3 h-3 text-teal-400" />
              Equity Growth Curve
            </h2>
            <div className="h-[300px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={chartData}>
                  <defs>
                    <linearGradient id="colorPnl" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#2dd4bf" stopOpacity={0.3}/>
                      <stop offset="95%" stopColor="#2dd4bf" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.03)" />
                  <XAxis dataKey="date" hide />
                  <YAxis hide domain={['auto', 'auto']} />
                  <Tooltip 
                    contentStyle={{ background: '#0d1117', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '12px' }}
                    labelStyle={{ color: 'rgba(255,255,255,0.3)', fontSize: '10px' }}
                  />
                  <Area type="monotone" dataKey="pnl" stroke="#2dd4bf" strokeWidth={3} fillOpacity={1} fill="url(#colorPnl)" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Key Insights Widget */}
          <div className="bg-white/[0.04] border border-white/5 rounded-3xl p-8 flex flex-col justify-between">
            <div>
              <h3 className="text-[10px] font-black uppercase tracking-[0.2em] text-white/60 mb-6">Execution Audit</h3>
              <div className="space-y-6">
                <div className="flex items-start gap-4">
                  <Shield className="w-5 h-5 text-teal-400 shrink-0 mt-0.5" />
                  <div>
                    <h4 className="text-xs font-bold mb-1">Risk Controlled</h4>
                    <p className="text-[11px] text-white/30 leading-relaxed">System is strictly enforcing 2% risk per trade with 5x leverage.</p>
                  </div>
                </div>
                <div className="flex items-start gap-4">
                  <Percent className="w-5 h-5 text-blue-400 shrink-0 mt-0.5" />
                  <div>
                    <h4 className="text-xs font-bold mb-1">Fee Attrition</h4>
                    <p className="text-[11px] text-white/30 leading-relaxed">Fees represent approx {((summary?.total_fees || 0) / Math.abs(summary?.net_pnl || 1) * 100).toFixed(1)}% of net profit. Monitor taker costs.</p>
                  </div>
                </div>
                <div className="flex items-start gap-4">
                  <AlertTriangle className="w-5 h-5 text-amber-500 shrink-0 mt-0.5" />
                  <div>
                    <h4 className="text-xs font-bold mb-1">Slippage Log</h4>
                    <p className="text-[11px] text-white/30 leading-relaxed">Tracking market order gaps. Limit orders enabled to reduce entry drag.</p>
                  </div>
                </div>
              </div>
            </div>
            <div className="mt-8 pt-8 border-t border-white/5">
              <div className="flex justify-between items-center bg-teal-500/10 p-4 rounded-xl border border-teal-500/20">
                <span className="text-[10px] font-black uppercase tracking-widest text-teal-400">System Goal</span>
                <span className="text-[10px] font-bold text-white/80">Positive Expectancy</span>
              </div>
            </div>
          </div>
        </div>

        {/* Detailed Logs Table */}
        <div className="bg-white/[0.02] border border-white/5 rounded-3xl overflow-hidden mt-4">
          <div className="px-8 py-6 border-b border-white/5 bg-white/[0.01] flex justify-between items-center">
            <h2 className="text-[10px] font-black uppercase tracking-[0.2em] text-white/40">Audit Event Log (Detailed P&L)</h2>
            <div className="px-3 py-1 bg-white/5 rounded-full text-[9px] mono font-bold text-white/30">LATEST 50 TRADES</div>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-left">
              <thead className="bg-white/[0.03]">
                <tr className="text-[9px] font-black uppercase tracking-[0.15em] text-white/20">
                  <th className="px-8 py-4">Status</th>
                  <th className="px-8 py-4">Symbol</th>
                  <th className="px-8 py-4">Entry (Signal vs Actual)</th>
                  <th className="px-8 py-4">PnL (Expected vs Actual)</th>
                  <th className="px-8 py-4">Fees</th>
                  <th className="px-8 py-4">R:R</th>
                  <th className="px-8 py-4 text-right">Outcome</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/[0.02]">
                {trades.length === 0 ? (
                  <tr>
                    <td colSpan={7} className="px-8 py-16 text-center text-white/10 mono text-xs">No analytics records found</td>
                  </tr>
                ) : (
                  trades.map((t, idx) => {
                    const isWin = t.outcome === 'TP'
                    const pnlColor = (t.actual_pnl || 0) >= 0 ? 'text-teal-400' : 'text-rose-500'
                    return (
                      <tr key={t.id} className="hover:bg-white/[0.01] transition-all group">
                        <td className="px-8 py-5">
                           <span className={`text-[8px] font-black px-2 py-0.5 rounded border ${
                             t.status === 'closed' ? 'bg-white/5 border-white/10 text-white/40' : 'bg-teal-500/20 border-teal-500/50 text-teal-400 animate-pulse'
                           }`}>
                             {t.status.toUpperCase()}
                           </span>
                        </td>
                        <td className="px-8 py-5">
                          <p className="text-xs font-bold leading-none mb-1">{t.symbol}</p>
                          <p className="text-[9px] text-white/20 mono">{t.closed_at ? new Date(t.closed_at).toLocaleTimeString() : 'OPEN'}</p>
                        </td>
                        <td className="px-8 py-5">
                          <div className="flex flex-col gap-1">
                            <div className="flex items-center gap-2">
                              <span className="text-[9px] text-white/20 w-8">Exp:</span>
                              <span className="mono text-[11px] text-white/60">{fmtPrice(t.entry_price)}</span>
                            </div>
                            <div className="flex items-center gap-2">
                              <span className="text-[9px] text-white/20 w-8">Act:</span>
                              <span className="mono text-[11px] text-white/80">{fmtPrice(t.exit_price)}</span>
                            </div>
                          </div>
                        </td>
                        <td className="px-8 py-5">
                           <div className="flex flex-col gap-1">
                            <div className="flex items-center gap-2">
                              <span className="text-[9px] text-white/20 w-8">Exp:</span>
                              <span className="mono text-[11px] text-white/40">{t.expected_profit.toFixed(2)} / {t.expected_loss.toFixed(2)}</span>
                            </div>
                            <div className="flex items-center gap-2">
                              <span className="text-[9px] text-white/20 w-8 font-bold">Res:</span>
                              <span className={`mono text-[11px] font-bold ${pnlColor}`}>{(t.actual_pnl || 0).toFixed(2)} USDT</span>
                            </div>
                          </div>
                        </td>
                        <td className="px-8 py-5 mono text-[11px] text-rose-400/60">
                          {t.fees ? `-${t.fees.toFixed(3)}` : '0.000'}
                        </td>
                        <td className="px-8 py-5 mono text-[11px] text-white/50">
                          1:{t.rr_ratio}
                        </td>
                        <td className="px-8 py-5 text-right">
                          <span className={`text-[10px] font-black uppercase ${isWin ? 'text-teal-400' : 'text-rose-500'}`}>
                            {t.outcome || 'PENDING'}
                          </span>
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
