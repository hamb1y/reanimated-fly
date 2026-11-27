#!/usr/bin/env bash
# Build the three images and save them to one file, for a machine without internet access.
# Usage: deploy/save-images.sh [out.tar.gz]
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${1:-flybrain-images.tar.gz}"
docker compose build            # uses build.network: host
docker save flybrain-brain flybrain-physics flybrain-web | gzip -1 > "$OUT"
ls -lh "$OUT"
echo "Copy $OUT plus the whole repo folder (docker-compose.yml, docker-compose.gpu.yml, data/, deploy/) to the offline machine."
