/** @type {import('tailwindcss').Config} */
module.exports = {
  // Content paths are relative to where tailwind.config.js lives (frontend/)
  content: [
    './app/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './lib/**/*.{js,ts,jsx,tsx,mdx}',
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        navy: {
          950: '#060a12',
          900: '#0d1117',
          800: '#111827',
          700: '#1a2440',
          600: '#1e3a5f',
        },
        teal:  { DEFAULT: '#2dd4bf', dim: '#1a7a6e' },
        rose:  { DEFAULT: '#f43f5e', dim: '#7a1e2e' },
        amber: { DEFAULT: '#f59e0b', dim: '#7a4e05' },
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
        sans: ['DM Sans', 'system-ui', 'sans-serif'],
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'fade-in':    'fadeIn 0.3s ease-in-out',
      },
      keyframes: {
        fadeIn: {
          '0%':   { opacity: '0', transform: 'translateY(4px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
      },
    },
  },
  plugins: [],
}