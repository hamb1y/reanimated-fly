#!/usr/bin/env bash
# flybrain: one-command install and run.
#
#   ./install.sh           fetch the connectome data (files already on disk are verified and skipped),
#                          build the images, start the stack, print the URL
#   ./install.sh --check   no downloads, no builds: test the source URLs, verify data/ against
#                          data/SHA256SUMS, report Docker and GPU status
#   ./install.sh --cpu     use the CPU brain image even when a GPU is usable
#   ./install.sh --gpu     require the GPU brain image (stops if Docker cannot give containers the GPU)
#
# Safe to re-run: verified data files are skipped and compose rebuilds only what changed.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA="$ROOT/data"
WEB_URL="http://127.0.0.1:5030"
BRAIN_HEALTH="http://127.0.0.1:5031/health"

# Upstream sources, pinned to commits (the sha256 values are in data/SHA256SUMS).
#   philshiu/Drosophila_brain_model (MIT): Shiu et al. 2024 model and the FlyWire v783 connectome tables.
#   flyconnectome/flywire_annotations: the FlyWire annotation supertable (Supplemental_file1).
SHIU="https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/91bdd1e7dcf193f3e7ca5a8933497fcef63b7960"
ANN="https://raw.githubusercontent.com/flyconnectome/flywire_annotations/a83b2776d60d5764cef36b927f5f9679c16c47a2"

# path under data/ | source URL | step after download
# crlf: the upstream CSV has LF line endings; the copy this stack was tested with has CRLF
# (same rows, same order). The conversion makes the bytes match data/SHA256SUMS.
SOURCES=(
  "2025_Connectivity_783.parquet|$SHIU/Connectivity_783.parquet|none"
  "2025_Completeness_783.csv|$SHIU/Completeness_783.csv|crlf"
  "sez_neurons.pickle|$SHIU/sez_neurons.pickle|none"
  "positions/Supplemental_file1_neuron_annotations.tsv|$ANN/supplemental_files/Supplemental_file1_neuron_annotations.tsv|none"
)

CHECK=0
FORCE=auto
MODE=cpu

say()  { printf '%s\n' "$*"; }
step() { printf '\n== %s\n' "$*"; }
warn() { printf 'warning: %s\n' "$*" >&2; }
die()  { printf 'error: %s\n' "$*" >&2; exit 1; }

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'
  else shasum -a 256 "$1" | awk '{print $1}'
  fi
}

# expected sha256 for a path under data/, from data/SHA256SUMS
expected_sha() {
  awk -v f="$1" '$2 == f {print $1; found = 1} END {exit !found}' "$DATA/SHA256SUMS"
}

url_ok() {  # HEAD request, follows redirects
  local code
  code="$(curl -sIL -m 30 -o /dev/null -w '%{http_code}' "$1" || true)"
  if [[ "$code" == 200 ]]; then say "  url ok      $1"; return 0; fi
  say "  url FAILED  ($code) $1"; return 1
}

fetch_one() {  # $1 path under data/, $2 url, $3 post step
  local rel="$1" url="$2" post="$3" dest want got tmp
  dest="$DATA/$rel"
  want="$(expected_sha "$rel")" || die "$rel is not listed in $DATA/SHA256SUMS"
  if [[ -f "$dest" ]]; then
    got="$(sha256_of "$dest")"
    if [[ "$got" == "$want" ]]; then say "  ok          $rel (sha256 matches, skipped)"; return 0; fi
    warn "$rel exists but its sha256 is different; it will be replaced"
  fi
  if [[ "$CHECK" == 1 ]]; then say "  would fetch $rel"; return 0; fi
  mkdir -p "$(dirname "$dest")"
  tmp="$dest.part"
  say "  fetching    $rel"
  curl -fL --retry 3 --retry-delay 5 -o "$tmp" "$url" || { rm -f "$tmp"; die "download failed: $url"; }
  case "$post" in
    none) ;;
    crlf) awk '{printf "%s\r\n", $0}' "$tmp" > "$tmp.crlf" && mv "$tmp.crlf" "$tmp" ;;
    *) die "unknown post step: $post" ;;
  esac
  got="$(sha256_of "$tmp")"
  if [[ "$got" != "$want" ]]; then
    rm -f "$tmp"
    die "$rel: sha256 $got does not match $want (data/SHA256SUMS). Not installed."
  fi
  mv "$tmp" "$dest"
  say "  ok          $rel"
}

check_docker() {
  local fail=0
  if ! command -v docker >/dev/null 2>&1; then
    say "  docker       NOT FOUND (install Docker Engine: https://docs.docker.com/engine/install/)"; fail=1
  elif ! docker compose version >/dev/null 2>&1; then
    say "  compose      NOT FOUND (needs the Compose v2 plugin: https://docs.docker.com/compose/install/)"; fail=1
  else
    say "  docker       $(docker version --format '{{.Server.Version}}' 2>/dev/null || echo 'daemon not reachable')"
    say "  compose      $(docker compose version --short 2>/dev/null || echo unknown)"
    if ! docker info >/dev/null 2>&1; then
      say "  daemon       NOT REACHABLE. Start it (sudo systemctl start docker), add your user to the docker group"
      say "               (sudo usermod -aG docker \$USER, then log in again), or run this script with sudo."
      fail=1
    fi
  fi
  if [[ "$fail" == 1 ]]; then
    [[ "$CHECK" == 1 ]] || die "Docker is not ready (see above)."
  fi
}

choose_mode() {  # sets MODE to gpu or cpu
  if [[ "$FORCE" == cpu ]]; then MODE=cpu; say "  brain        cpu (forced by --cpu)"; return 0; fi
  local name=""
  if command -v nvidia-smi >/dev/null 2>&1 && name="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)" && [[ -n "$name" ]]; then
    say "  host GPU     $name"
  else
    [[ "$FORCE" == gpu ]] && die "--gpu: nvidia-smi does not see an NVIDIA GPU on this host."
    MODE=cpu; say "  brain        cpu (no NVIDIA GPU visible to the host)"; return 0
  fi
  if [[ "$CHECK" == 1 ]]; then
    say "  brain        gpu if Docker can give containers the GPU (probed on a real run), otherwise cpu"
    MODE=gpu; return 0
  fi
  if ! docker image inspect busybox >/dev/null 2>&1; then docker pull -q busybox >/dev/null 2>&1 || true; fi
  if docker run --rm --gpus all busybox true >/dev/null 2>&1; then
    MODE=gpu; say "  brain        gpu (Docker containers get the GPU)"; return 0
  fi
  [[ "$FORCE" == gpu ]] && die "--gpu: Docker cannot give containers the GPU. Install nvidia-container-toolkit and configure Docker (see README, troubleshooting)."
  MODE=cpu
  warn "Docker cannot give containers the GPU (nvidia-container-toolkit is not set up); using the CPU brain."
  say "  brain        cpu (Docker has no GPU access)"
}

wait_http() {  # url, timeout seconds
  local url="$1" secs="$2" i code
  for ((i = 0; i < secs; i += 3)); do
    code="$(curl -s -o /dev/null -m 3 -w '%{http_code}' "$url" || true)"
    [[ "$code" == 200 ]] && return 0
    sleep 3
  done
  return 1
}

main() {
  for arg in "$@"; do
    case "$arg" in
      --check|--dry-run) CHECK=1 ;;
      --cpu) FORCE=cpu ;;
      --gpu) FORCE=gpu ;;
      -h|--help) sed -n '2,11p' "$0"; exit 0 ;;
      *) die "unknown option: $arg (see --help)" ;;
    esac
  done
  cd "$ROOT"
  [[ "$CHECK" == 1 ]] && say "CHECK MODE: nothing is downloaded or built."

  step "1/4 Docker"
  check_docker

  step "2/4 Connectome data in $DATA"
  [[ -f "$DATA/SHA256SUMS" ]] || die "$DATA/SHA256SUMS is missing (it is part of the repository)."
  local bad=0 entry rel url post
  for entry in "${SOURCES[@]}"; do
    IFS='|' read -r rel url post <<< "$entry"
    if [[ "$CHECK" == 1 ]]; then url_ok "$url" || bad=$((bad + 1)); fi
    fetch_one "$rel" "$url" "$post"
  done

  step "3/4 GPU"
  choose_mode

  if [[ "$CHECK" == 1 ]]; then
    step "check result"
    say "  would run: docker compose -f docker-compose.yml -f docker-compose.$MODE.yml up -d --build"
    if [[ "$bad" -gt 0 ]]; then die "$bad source URL(s) failed"; fi
    say "  all source URLs resolve. Run ./install.sh (without --check) to build and start."
    exit 0
  fi

  step "4/4 Build and start the stack (brain on $MODE)"
  docker compose -f docker-compose.yml -f "docker-compose.$MODE.yml" up -d --build
  say "waiting for the web UI ..."
  wait_http "$WEB_URL" 180 || warn "the web UI did not answer within 3 minutes; see: docker compose logs web"
  say "waiting for the brain (the first start loads the connectome and can take a few minutes) ..."
  if wait_http "$BRAIN_HEALTH" 600; then
    say "  brain: $(curl -s -m 3 "$BRAIN_HEALTH" | head -c 300)"
  else
    warn "the brain is not healthy yet; see: docker compose logs brain"
  fi

  say ""
  say "flybrain is running:  $WEB_URL"
  say "  logs:  docker compose logs -f      (from this folder)"
  say "  stop:  docker compose down"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
