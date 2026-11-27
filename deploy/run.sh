#!/usr/bin/env bash
# Offline start on a Linux host: load the images saved by save-images.sh and start everything. GPU brain if the NVIDIA toolkit works, else CPU.
# Usage: deploy/run.sh [images.tar.gz]
set -euo pipefail
cd "$(dirname "$0")/.."
[ -n "${1:-}" ] && { echo "loading images..."; gunzip -c "$1" | docker load; }
FILES="-f docker-compose.yml"
if docker run --rm --gpus all --network host busybox true >/dev/null 2>&1 && command -v nvidia-smi >/dev/null; then
  echo "GPU available: brain runs on the GPU"; FILES="$FILES -f docker-compose.gpu.yml"
else
  echo "no usable GPU in docker: brain runs on the CPU (slower under load; see README)"
fi
docker compose $FILES up -d --no-build
echo "waiting for health..."
for i in $(seq 1 60); do
  b=$(curl -s -m 2 -o /dev/null -w '%{http_code}' http://127.0.0.1:5031/health || true)
  p=$(curl -s -m 2 -o /dev/null -w '%{http_code}' http://127.0.0.1:5032/health || true)
  [ "$b" = 200 ] && [ "$p" = 200 ] && break; sleep 3
done
echo "brain: $(curl -s -m 3 http://127.0.0.1:5031/health | head -c 200)"
echo "physics: $(curl -s -m 3 http://127.0.0.1:5032/health | head -c 200)"
echo "open http://127.0.0.1:5030 (keep the browser tab in front: hidden tabs pause the 3D view)"
