/** @type {import('next').NextConfig} */
const API_URL = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000'

const nextConfig = {
  // Required for Docker standalone image
  output: 'standalone',

  // Expose the backend URL to the browser bundle
  env: {
    NEXT_PUBLIC_API_URL: API_URL,
  },
  async rewrites() {
    return [
      {
        // Proxy /api/backend/* → Railway backend (useful for SSR routes)
        source: '/api/backend/:path*',
        destination: `${API_URL}/api/:path*`,
      },
    ]
  },
}

module.exports = nextConfig
