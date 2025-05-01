#!/usr/bin/env bash
set -e

DOMAIN=garmentiq.ly.gd.edu.kg
IMAGE=garmentiq_magicbox
CONTAINER=garmentiq_magicbox
PORT=8888

echo "🛠️  Generating/Installing mkcert CA and certs for $DOMAIN & localhost"
mkcert -install
mkcert -cert-file localhost.pem -key-file localhost-key.pem $DOMAIN localhost

echo "🐳 Building Docker image '$IMAGE'…"
docker build -t $IMAGE .

echo "🚀 Running container '$CONTAINER'…"
docker rm -f $CONTAINER 2>/dev/null || true
docker run -d \
  --name $CONTAINER \
  -p $PORT:$PORT \
  -v "$(pwd)/working:/app/working" \
  $IMAGE

echo "🔐 Patching /etc/hosts for $DOMAIN → 127.0.0.1 (requires sudo)"
if ! grep -q "$DOMAIN" /etc/hosts; then
  echo "127.0.0.1 $DOMAIN" | sudo tee -a /etc/hosts
fi

echo ""
echo "✅ Setup complete!"
echo "Open your browser to: https://$DOMAIN/magicbox"
