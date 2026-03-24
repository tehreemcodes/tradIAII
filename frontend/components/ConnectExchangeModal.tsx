'use client'

/**
 * ConnectExchangeModal
 * =====================
 * Allows users to connect their Binance account via API keys.
 * Keys are submitted once and never returned to the frontend.
 * The session_id token is stored in localStorage for subsequent requests.
 */

import { useState } from 'react'

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000'

interface Props {
  onConnected: (sessionId: string, balance: number, exchange: string) => void
  onClose:     () => void
}

type Step = 'form' | 'connecting' | 'success' | 'error'

export default function ConnectExchangeModal({ onConnected, onClose }: Props) {
  const [step,      setStep]      = useState<Step>('form')
  const [apiKey,    setApiKey]    = useState('')
  const [apiSecret, setApiSecret] = useState('')
  const [testnet,   setTestnet]   = useState(false)
  const [showSecret,setShowSecret]= useState(false)
  const [error,     setError]     = useState<string | null>(null)
  const [result,    setResult]    = useState<any>(null)

  const handleConnect = async () => {
    if (!apiKey.trim() || !apiSecret.trim()) {
      setError('Both API Key and Secret are required.')
      return
    }
    setStep('connecting')
    setError(null)

    try {
      const res = await fetch(`${API_BASE}/api/exchange/connect`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({
          api_key:    apiKey.trim(),
          api_secret: apiSecret.trim(),
          exchange:   'binance',
          testnet,
        }),
      })

      const data = await res.json()

      if (!res.ok) {
        setError(data.detail ?? 'Connection failed.')
        setStep('error')
        return
      }

      // Store session token — never the raw keys
      localStorage.setItem('tradia_session_id', data.session_id)

      setResult(data)
      setStep('success')

      // Clear key fields from state immediately
      setApiKey('')
      setApiSecret('')

      setTimeout(() => {
        onConnected(data.session_id, data.balance, data.exchange)
        onClose()
      }, 1800)

    } catch (e: any) {
      setError('Network error. Is the backend running?')
      setStep('error')
    }
  }

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 z-40"
        style={{ background: 'rgba(0,0,0,0.7)', backdropFilter: 'blur(6px)' }}
        onClick={onClose}
      />

      {/* Modal */}
      <div
        className="fixed z-50 top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-full max-w-md"
        style={{ padding: '0 1rem' }}
      >
        <div
          className="rounded-2xl p-6"
          style={{
            background:   '#0d1117',
            border:       '1px solid rgba(255,255,255,0.08)',
            boxShadow:    '0 24px 80px rgba(0,0,0,0.6)',
          }}
        >
          {/* Header */}
          <div className="flex items-center justify-between mb-6">
            <div>
              <h2 className="text-white font-bold text-base">Connect Exchange</h2>
              <p className="text-white/30 text-xs mt-0.5">
                Keys are encrypted and never stored in plaintext
              </p>
            </div>
            <button
              onClick={onClose}
              className="w-7 h-7 rounded-lg flex items-center justify-center text-white/30 hover:text-white/60 transition-colors"
              style={{ background: 'rgba(255,255,255,0.05)' }}
            >
              ✕
            </button>
          </div>

          {/* Form */}
          {(step === 'form' || step === 'error') && (
            <div className="space-y-4">

              {/* Exchange selector */}
              <div>
                <label className="text-[10px] text-white/30 uppercase tracking-wider mb-1.5 block">
                  Exchange
                </label>
                <div
                  className="flex items-center gap-2 px-3 py-2.5 rounded-lg"
                  style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)' }}
                >
                  <span className="text-sm text-white/80">🟡 Binance</span>
                  <span
                    className="ml-auto text-[9px] px-1.5 py-0.5 rounded font-bold"
                    style={{ background: 'rgba(45,212,191,0.10)', color: '#2dd4bf' }}
                  >
                    FUTURES
                  </span>
                </div>
              </div>

              {/* API Key */}
              <div>
                <label className="text-[10px] text-white/30 uppercase tracking-wider mb-1.5 block">
                  API Key
                </label>
                <input
                  type="text"
                  value={apiKey}
                  onChange={e => setApiKey(e.target.value)}
                  placeholder="Paste your Binance API key"
                  autoComplete="off"
                  spellCheck={false}
                  className="w-full px-3 py-2.5 rounded-lg text-sm font-mono outline-none transition-all"
                  style={{
                    background:   'rgba(255,255,255,0.04)',
                    border:       `1px solid ${apiKey ? 'rgba(45,212,191,0.25)' : 'rgba(255,255,255,0.08)'}`,
                    color:        'rgba(255,255,255,0.8)',
                  }}
                />
              </div>

              {/* API Secret */}
              <div>
                <label className="text-[10px] text-white/30 uppercase tracking-wider mb-1.5 block">
                  API Secret
                </label>
                <div className="relative">
                  <input
                    type={showSecret ? 'text' : 'password'}
                    value={apiSecret}
                    onChange={e => setApiSecret(e.target.value)}
                    placeholder="Paste your Binance API secret"
                    autoComplete="off"
                    className="w-full px-3 py-2.5 pr-10 rounded-lg text-sm font-mono outline-none transition-all"
                    style={{
                      background: 'rgba(255,255,255,0.04)',
                      border:     `1px solid ${apiSecret ? 'rgba(45,212,191,0.25)' : 'rgba(255,255,255,0.08)'}`,
                      color:      'rgba(255,255,255,0.8)',
                    }}
                  />
                  <button
                    type="button"
                    onClick={() => setShowSecret(s => !s)}
                    className="absolute right-2.5 top-1/2 -translate-y-1/2 text-white/25 hover:text-white/50 text-xs transition-colors"
                  >
                    {showSecret ? '🙈' : '👁'}
                  </button>
                </div>
              </div>

              {/* Testnet toggle */}
              <div
                className="flex items-center justify-between px-3 py-2.5 rounded-lg"
                style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.05)' }}
              >
                <div>
                  <p className="text-xs text-white/50">Testnet / Demo mode</p>
                  <p className="text-[10px] text-white/25 mt-0.5">
                    {testnet ? 'Using testnet — fake money' : 'Using mainnet — real money'}
                  </p>
                </div>
                <button
                  onClick={() => setTestnet(t => !t)}
                  className="w-10 h-5 rounded-full transition-all relative"
                  style={{
                    background: testnet ? '#2dd4bf' : 'rgba(255,255,255,0.1)',
                  }}
                >
                  <div
                    className="absolute top-0.5 w-4 h-4 rounded-full bg-white transition-all"
                    style={{ left: testnet ? '22px' : '2px' }}
                  />
                </button>
              </div>

              {/* Error */}
              {error && (
                <div
                  className="px-3 py-2.5 rounded-lg text-xs"
                  style={{ background: 'rgba(244,63,94,0.08)', border: '1px solid rgba(244,63,94,0.15)', color: '#f43f5e' }}
                >
                  {error}
                </div>
              )}

              {/* Security note */}
              <div
                className="px-3 py-2.5 rounded-lg"
                style={{ background: 'rgba(245,158,11,0.06)', border: '1px solid rgba(245,158,11,0.12)' }}
              >
                <p className="text-[10px] text-yellow-500/70 leading-relaxed">
                  ⚠️ Make sure withdrawals are <strong>disabled</strong> on your API key.
                  Only <strong>Enable Reading</strong> and <strong>Enable Spot & Margin Trading</strong> are needed.
                </p>
              </div>

              {/* Connect button */}
              <button
                onClick={handleConnect}
                disabled={!apiKey || !apiSecret}
                className="w-full py-3 rounded-xl text-sm font-bold transition-all"
                style={{
                  background: (!apiKey || !apiSecret)
                    ? 'rgba(255,255,255,0.05)'
                    : 'linear-gradient(135deg, #2dd4bf, #3b82f6)',
                  color:  (!apiKey || !apiSecret) ? 'rgba(255,255,255,0.2)' : '#0d1117',
                  cursor: (!apiKey || !apiSecret) ? 'not-allowed' : 'pointer',
                }}
              >
                Connect Binance Account
              </button>
            </div>
          )}

          {/* Connecting state */}
          {step === 'connecting' && (
            <div className="py-8 text-center">
              <div
                className="w-10 h-10 rounded-full border-2 border-white/10 animate-spin mx-auto mb-4"
                style={{ borderTopColor: '#2dd4bf' }}
              />
              <p className="text-white/60 text-sm">Validating credentials...</p>
              <p className="text-white/25 text-xs mt-1">Connecting to Binance</p>
            </div>
          )}

          {/* Success state */}
          {step === 'success' && result && (
            <div className="py-6 text-center">
              <div
                className="w-12 h-12 rounded-full flex items-center justify-center mx-auto mb-4 text-2xl"
                style={{ background: 'rgba(45,212,191,0.12)', border: '1px solid rgba(45,212,191,0.25)' }}
              >
                ✓
              </div>
              <p className="text-white font-semibold text-sm mb-1">Connected!</p>
              <p className="text-white/40 text-xs mb-3">
                {result.exchange?.toUpperCase()} · {result.testnet ? 'TESTNET' : 'MAINNET'}
              </p>
              <div
                className="rounded-xl px-4 py-3"
                style={{ background: 'rgba(45,212,191,0.06)', border: '1px solid rgba(45,212,191,0.15)' }}
              >
                <p className="text-[10px] text-white/30 mb-0.5">Available Balance</p>
                <p className="mono text-xl font-bold" style={{ color: '#2dd4bf' }}>
                  ${result.balance?.toLocaleString('en-US', { minimumFractionDigits: 2 })} USDT
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </>
  )
}