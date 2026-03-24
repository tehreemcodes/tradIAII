# ─────────────────────────────────────────────────────────────
# TradIA Frontend — Dockerfile
# Standalone Next.js output for minimal production image
# ─────────────────────────────────────────────────────────────

FROM node:20-alpine AS deps
WORKDIR /app
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --only=production

# ─────────────────────────────────────────────────────────────
FROM node:20-alpine AS builder
WORKDIR /app

# Build-time ARG so next.config.js can bake in the Railway/VPS URL
ARG NEXT_PUBLIC_API_URL=http://localhost:8000
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL

COPY frontend/ .
COPY --from=deps /app/node_modules ./node_modules

RUN npm run build

# ─────────────────────────────────────────────────────────────
FROM node:20-alpine AS runtime
WORKDIR /app

ENV NODE_ENV=production \
    PORT=3000

RUN addgroup -g 1001 -S nodejs && adduser -S nextjs -u 1001

# Copy only what Next.js needs to run
COPY --from=builder /app/public ./public
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

USER nextjs
EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD wget -qO- http://localhost:3000 || exit 1

CMD ["node", "server.js"]
