#!/bin/bash
# ─────────────────────────────────────────────────────────────
# TradIA — VPS Deployment Script
# Tested on Ubuntu 22.04 with SSH root access
#
# Usage (run this on your LOCAL machine):
#   chmod +x deploy.sh
#   ./deploy.sh YOUR_VPS_IP YOUR_DOMAIN
#
# Example:
#   ./deploy.sh 123.45.67.89 tradia.yourdomain.com
# ─────────────────────────────────────────────────────────────

set -e  # Exit on any error

VPS_IP="${1:?Usage: ./deploy.sh VPS_IP DOMAIN}"
DOMAIN="${2:?Usage: ./deploy.sh VPS_IP DOMAIN}"
SSH_USER="root"
APP_DIR="/opt/tradia"
REPO_URL="https://github.com/tehreemcodes/tradIAII.git"

echo "═══════════════════════════════════════════════"
echo "  TradIA VPS Deployment"
echo "  VPS: $VPS_IP   Domain: $DOMAIN"
echo "═══════════════════════════════════════════════"

# ── Step 1: Install Docker on VPS if needed ──────────────────
echo ""
echo "[1/7] Installing Docker & dependencies on VPS..."
ssh "$SSH_USER@$VPS_IP" bash << 'REMOTE'
set -e
if ! command -v docker &>/dev/null; then
    apt-get update -qq
    apt-get install -y -qq ca-certificates curl gnupg git
    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    chmod a+r /etc/apt/keyrings/docker.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
        https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" \
        | tee /etc/apt/sources.list.d/docker.list > /dev/null
    apt-get update -qq
    apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-compose-plugin
    systemctl enable --now docker
    echo "Docker installed ✓"
else
    echo "Docker already installed ✓"
fi
REMOTE

# ── Step 2: Clone or update repo on VPS ──────────────────────
echo ""
echo "[2/7] Cloning/updating repo on VPS..."
ssh "$SSH_USER@$VPS_IP" bash << REMOTE
set -e
if [ -d "$APP_DIR/.git" ]; then
    cd "$APP_DIR"
    git pull origin master
    echo "Repo updated ✓"
else
    git clone "$REPO_URL" "$APP_DIR"
    echo "Repo cloned ✓"
fi
REMOTE

# ── Step 3: Create .env on VPS ───────────────────────────────
echo ""
echo "[3/7] Checking .env on VPS..."
ssh "$SSH_USER@$VPS_IP" bash << REMOTE
if [ ! -f "$APP_DIR/.env" ]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  No .env file found on VPS."
    echo "  Copy .env.example to .env and fill in your values:"
    echo ""
    echo "  ssh root@$VPS_IP"
    echo "  cp $APP_DIR/.env.example $APP_DIR/.env"
    echo "  nano $APP_DIR/.env"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    exit 1
fi
echo ".env found ✓"
REMOTE

# ── Step 4: Replace domain placeholder in nginx.conf ─────────
echo ""
echo "[4/7] Configuring nginx for domain: $DOMAIN..."
ssh "$SSH_USER@$VPS_IP" bash << REMOTE
set -e
sed -i "s/YOUR_DOMAIN/$DOMAIN/g" "$APP_DIR/nginx.conf"
echo "Nginx configured ✓"
REMOTE

# ── Step 5: Get SSL certificate via Certbot ──────────────────
echo ""
echo "[5/7] Obtaining SSL certificate (Let's Encrypt)..."
ssh "$SSH_USER@$VPS_IP" bash << REMOTE
set -e
mkdir -p "$APP_DIR/ssl"
if [ ! -f "$APP_DIR/ssl/fullchain.pem" ]; then
    apt-get install -y -qq certbot
    certbot certonly --standalone --non-interactive --agree-tos \
        -m "admin@$DOMAIN" -d "$DOMAIN" -d "www.$DOMAIN" \
        --cert-path "$APP_DIR/ssl/fullchain.pem" \
        --key-path  "$APP_DIR/ssl/privkey.pem" || \
    certbot certonly --standalone --non-interactive --agree-tos \
        -m "admin@$DOMAIN" -d "$DOMAIN"
    cp /etc/letsencrypt/live/$DOMAIN/fullchain.pem $APP_DIR/ssl/
    cp /etc/letsencrypt/live/$DOMAIN/privkey.pem   $APP_DIR/ssl/
    echo "SSL certificate obtained ✓"
else
    echo "SSL certificate already exists ✓"
fi
REMOTE

# ── Step 6: Build and start Docker containers ─────────────────
echo ""
echo "[6/7] Building and starting Docker containers..."
ssh "$SSH_USER@$VPS_IP" bash << REMOTE
set -e
cd "$APP_DIR"
docker compose pull --quiet 2>/dev/null || true
docker compose build --no-cache
docker compose up -d --remove-orphans
echo "Containers started ✓"
REMOTE

# ── Step 7: Run health check ──────────────────────────────────
echo ""
echo "[7/7] Waiting for health check..."
sleep 15
ssh "$SSH_USER@$VPS_IP" bash << REMOTE
set -e
STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/api/health)
if [ "$STATUS" = "200" ]; then
    echo "Backend health check passed ✓"
else
    echo "Health check returned HTTP $STATUS"
    docker compose logs backend --tail 30
fi
docker compose ps
REMOTE

echo ""
echo "═══════════════════════════════════════════════"
echo "  ✅ TradIA deployed successfully!"
echo ""
echo "  Dashboard: https://$DOMAIN"
echo "  API:       https://$DOMAIN/api/health"
echo "  Logs:      ssh root@$VPS_IP 'docker compose -f /opt/tradia/docker-compose.yml logs -f'"
echo "═══════════════════════════════════════════════"
