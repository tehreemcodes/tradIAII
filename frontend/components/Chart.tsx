'use client'

import { useEffect, useRef, useState, useCallback } from 'react'
import { api, Candle, Signal } from '@/lib/api'

type TF = '15m' | '1h' | '4h' | '1d'

const TF_LABELS: Record<TF, string> = {
  '15m': '15m',
  '1h':  '1H',
  '4h':  '4H',
  '1d':  '1D',
}

interface ChartProps {
  activeSignal?: Signal | null
  activeTf:      TF
  onTfChange:    (tf: TF) => void
}

export default function Chart({ activeSignal, activeTf, onTfChange }: ChartProps) {
  const containerRef = useRef<HTMLDivElement>(null)

  // FIX: track chart instance AND a "disposed" flag separately
  // to prevent ResizeObserver from calling into a removed chart
  const chartRef    = useRef<any>(null)
  const disposedRef = useRef<boolean>(false)

  const [candles, setCandles] = useState<Candle[]>([])
  const [loading, setLoading] = useState(true)
  const [error,   setError]   = useState<string | null>(null)
  const [price,   setPrice]   = useState<number | null>(null)
  const [priceUp, setPriceUp] = useState(true)

  // ── Fetch Candles ────────────────────────────────────────────
  const fetchCandles = useCallback(async (timeframe: TF) => {
    try {
      setLoading(true)
      const data = await api.candles(timeframe, 200)
      setCandles(data.candles)
      if (data.candles.length > 0) {
        const last = data.candles[data.candles.length - 1]
        const prev = data.candles[data.candles.length - 2]
        setPriceUp(last.close >= (prev?.close ?? last.close))
        setPrice(last.close)
      }
      setError(null)
    } catch (e: any) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchCandles(activeTf)
    const iv = setInterval(() => fetchCandles(activeTf), 60_000)
    return () => clearInterval(iv)
  }, [activeTf, fetchCandles])

  // ── Build Chart ──────────────────────────────────────────────
  useEffect(() => {
    if (!containerRef.current || candles.length === 0) return
    if (typeof window === 'undefined') return

    import('lightweight-charts').then(({ createChart, CrosshairMode }) => {
      // Guard: component may have unmounted during async import
      if (!containerRef.current) return

      // Destroy previous chart instance safely
      if (chartRef.current) {
        try { chartRef.current.remove() } catch { /* already disposed */ }
        chartRef.current = null
      }

      // Reset disposed flag for the new chart instance
      disposedRef.current = false

      const container = containerRef.current!
      const chart = createChart(container, {
        width:  container.clientWidth,
        height: container.clientHeight || 260,
        layout: {
          background:  { color: 'transparent' },
          textColor:   'rgba(255,255,255,0.4)',
          fontSize:    11,
        },
        grid: {
          vertLines: { color: 'rgba(255,255,255,0.04)' },
          horzLines: { color: 'rgba(255,255,255,0.04)' },
        },
        crosshair: {
          mode:     CrosshairMode.Normal,
          vertLine: { color: 'rgba(255,255,255,0.2)', labelBackgroundColor: '#1a2440' },
          horzLine: { color: 'rgba(255,255,255,0.2)', labelBackgroundColor: '#1a2440' },
        },
        rightPriceScale: { borderColor: 'rgba(255,255,255,0.06)' },
        timeScale: {
          borderColor:    'rgba(255,255,255,0.06)',
          timeVisible:    true,
          secondsVisible: false,
        },
      })

      chartRef.current = chart

      // ── Candlestick Series ──────────────────────────────────
      const candleSeries = chart.addCandlestickSeries({
        upColor:         '#2dd4bf',
        downColor:       '#f43f5e',
        borderUpColor:   '#2dd4bf',
        borderDownColor: '#f43f5e',
        wickUpColor:     '#2dd4bf',
        wickDownColor:   '#f43f5e',
      })

      candleSeries.setData(
        candles.map(c => ({
          time:  c.time as any,
          open:  c.open,
          high:  c.high,
          low:   c.low,
          close: c.close,
        }))
      )

      // ── Volume Series ───────────────────────────────────────
      const volSeries = chart.addHistogramSeries({
        color:        'rgba(100,116,139,0.3)',
        priceScaleId: 'vol',
        priceFormat:  { type: 'volume' },
      })
      chart.priceScale('vol').applyOptions({
        scaleMargins: { top: 0.85, bottom: 0 },
      })
      volSeries.setData(
        candles.map(c => ({
          time:  c.time as any,
          value: c.volume,
          color: c.close >= c.open
            ? 'rgba(45,212,191,0.3)'
            : 'rgba(244,63,94,0.3)',
        }))
      )

      // ── Signal Markers ──────────────────────────────────────
      const markers = candles
        .filter(c => c.signal === 2 || c.signal === 0)
        .map(c => ({
          time:     c.time as any,
          position: c.signal === 2 ? 'belowBar' : 'aboveBar',
          color:    c.signal === 2 ? '#2dd4bf' : '#f43f5e',
          shape:    c.signal === 2 ? 'arrowUp'  : 'arrowDown',
          text:     c.signal === 2 ? 'BUY'      : 'SELL',
          size:     1,
        }))

      if (markers.length > 0) {
        candleSeries.setMarkers(markers as any)
      }

      // ── FVG Zone Lines ──────────────────────────────────────
      candles
        .filter(c => (c.bull_fvg || c.bear_fvg) && c.fvg_top != null && c.fvg_bot != null)
        .slice(-10)
        .forEach(c => {
          const color = c.bull_fvg ? '#2dd4bf' : '#f43f5e'
          const opts  = {
            color,
            lineWidth:        1 as const,
            lineStyle:        2,
            priceLineVisible: false,
            lastValueVisible: false,
          }
          chart.addLineSeries(opts).setData([{ time: c.time as any, value: c.fvg_top! }])
          chart.addLineSeries(opts).setData([{ time: c.time as any, value: c.fvg_bot! }])
        })

      // ── Active Signal SL/TP Lines ───────────────────────────
      if (activeSignal?.sl && activeSignal?.tp && candles.length >= 10) {
        const last = candles[candles.length - 1]
        const prev = candles[candles.length - 10]

        const lineOpts = (color: string, title: string) => ({
          color,
          lineWidth:        1 as const,
          lineStyle:        3,
          priceLineVisible: false,
          lastValueVisible: true,
          title,
        })

        chart.addLineSeries(lineOpts('#f43f5e88', 'SL')).setData([
          { time: prev.time as any, value: activeSignal.sl },
          { time: last.time as any, value: activeSignal.sl },
        ])
        chart.addLineSeries(lineOpts('#2dd4bf88', 'TP')).setData([
          { time: prev.time as any, value: activeSignal.tp },
          { time: last.time as any, value: activeSignal.tp },
        ])
      }

      chart.timeScale().fitContent()

      // ── Resize Observer ─────────────────────────────────────
      // FIX: Check disposedRef before calling chart methods.
      // ResizeObserver can fire AFTER chart.remove() during cleanup.
      const ro = new ResizeObserver(entries => {
        // Guard against disposed chart — this was causing "Object is disposed"
        if (disposedRef.current || !chartRef.current) return

        for (const entry of entries) {
          const w = entry.contentRect.width
          if (w > 0) {
            try {
              chartRef.current.applyOptions({ width: w })
            } catch {
              // Chart was disposed between the guard check and applyOptions
              // This can happen in strict mode double-invoke — safe to ignore
            }
          }
        }
      })

      if (containerRef.current) {
        ro.observe(containerRef.current)
      }

      // Cleanup: mark as disposed BEFORE removing the chart
      return () => {
        disposedRef.current = true   // FIX: set flag before ro.disconnect
        ro.disconnect()
        if (chartRef.current) {
          try { chartRef.current.remove() } catch { /* already disposed */ }
          chartRef.current = null
        }
      }
    })
  }, [candles, activeSignal])

  return (
    <div className="flex flex-col h-full">

      {/* Chart Header */}
      <div className="flex items-center justify-between px-4 py-3 flex-shrink-0">
        <div className="flex items-center gap-2">
          {(Object.keys(TF_LABELS) as TF[]).map(t => (
            <button
              key={t}
              onClick={() => onTfChange(t)}
              className="px-3 py-1.5 rounded-md text-[11px] transition-all duration-150 cursor-pointer"
              style={{
                background: t === activeTf ? 'rgba(45,212,191,0.15)' : 'transparent',
                border:     `1px solid ${t === activeTf ? 'rgba(45,212,191,0.35)' : 'rgba(255,255,255,0.07)'}`,
                color:      t === activeTf ? '#2dd4bf' : 'rgba(255,255,255,0.35)',
                fontFamily: 'JetBrains Mono, monospace',
              }}
            >
              {TF_LABELS[t]}
            </button>
          ))}
        </div>

        <div className="flex items-center gap-5">
          {[
            { color: '#2dd4bf', label: 'Bull FVG / BUY'  },
            { color: '#f43f5e', label: 'Bear FVG / SELL' },
          ].map(l => (
            <div key={l.label} className="flex items-center gap-1.5">
              <div className="w-2.5 h-0.5 rounded" style={{ background: l.color, opacity: 0.7 }} />
              <span className="text-[10px] text-white/25">{l.label}</span>
            </div>
          ))}

          {price && (
            <span
              className="mono text-sm font-bold transition-colors duration-300"
              style={{ color: priceUp ? '#2dd4bf' : '#f43f5e' }}
            >
              ${price.toLocaleString('en-US', { minimumFractionDigits: 2 })}
            </span>
          )}
        </div>
      </div>

      {/* Chart Container */}
      <div className="relative flex-1 min-h-0">
        {loading && (
          <div className="absolute inset-0 flex items-center justify-center text-white/20 text-xs z-10 pointer-events-none">
            Loading chart...
          </div>
        )}
        {error && (
          <div className="absolute inset-0 flex items-center justify-center text-rose-400/50 text-xs z-10 pointer-events-none">
            {error}
          </div>
        )}
        {/* FIX: explicit pixel height avoids ResizeObserver loop on 0-height containers */}
        <div
          ref={containerRef}
          className="absolute inset-0 w-full h-full"
        />
      </div>

    </div>
  )
}