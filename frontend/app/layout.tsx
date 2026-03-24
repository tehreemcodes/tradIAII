import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'TradIA — ICT/SMC Trading Assistant',
  description: 'Production-grade AI crypto trading assistant powered by ICT/SMC methodology',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className="bg-navy-900 text-white overflow-hidden h-screen">
        {children}
      </body>
    </html>
  )
}
