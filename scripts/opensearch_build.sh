#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Build OpenSearch from GitHub and run a single-node dev instance (Linux recommended)
# Usage:
#   chmod +x scripts/opensearch_build.sh
#   ./scripts/opensearch_build.sh                # build + extract + print run instructions
#   ./scripts/opensearch_build.sh run            # start single-node (nohup)
#   ./scripts/opensearch_build.sh stop           # stop instance

set -Eeuo pipefail

OPENSEARCH_BRANCH=${OPENSEARCH_BRANCH:-main}
SRC_DIR=${SRC_DIR:-$HOME/opensearch_src}
BUILD_OUT=${BUILD_OUT:-$HOME/opensearch_build}
RUNTIME_DIR=${RUNTIME_DIR:-$HOME/opensearch_runtime}
JAVA_HOME_HINT=${JAVA_HOME_HINT:-}
JVM_HEAP_GB=${JVM_HEAP_GB:-2}

mkdir -p "$SRC_DIR" "$BUILD_OUT" "$RUNTIME_DIR"

step_build() {
  if [[ ! -d "$SRC_DIR/OpenSearch" ]]; then
    git clone --depth=1 -b "$OPENSEARCH_BRANCH" https://github.com/opensearch-project/OpenSearch.git "$SRC_DIR/OpenSearch"
  fi
  cd "$SRC_DIR/OpenSearch"
  echo "[build] Using branch: $OPENSEARCH_BRANCH"
  # Ensure Java is available (prefer 21 or 17)
  if ! command -v java >/dev/null 2>&1; then
    echo "[build] Java not found. Please install JDK (17 or 21). On Debian/Ubuntu: sudo apt-get install -y openjdk-21-jdk"
    exit 1
  fi
  java -version || true
  # Build distributions (skip tests for speed)
  ./gradlew assemble -Dbuild.snapshot=false -DskipTests
  # Locate linux tar 
  TAR=$(ls distribution/archives/linux-tar/build/distributions/opensearch-*.tar.gz | head -n1)
  if [[ -z "$TAR" ]]; then
    echo "[build] Could not find linux tar distribution. Check gradle output."
    exit 1
  fi
  cp -f "$TAR" "$BUILD_OUT/"
  echo "[build] Copied: $TAR -> $BUILD_OUT/"
  
  # Extract to runtime
  mkdir -p "$RUNTIME_DIR"
  rm -rf "$RUNTIME_DIR/opensearch"
  tar -xzf "$BUILD_OUT/$(basename "$TAR")" -C "$RUNTIME_DIR"
  mv "$RUNTIME_DIR"/opensearch-* "$RUNTIME_DIR/opensearch"
  echo "[build] Extracted to $RUNTIME_DIR/opensearch"

  # Minimal single-node config
  cat > "$RUNTIME_DIR/opensearch/config/opensearch.yml" << 'YAML'
cluster.name: aura-dev
node.name: node-1
network.host: 0.0.0.0
http.port: 9200
transport.port: 9300
transport.host: 127.0.0.1
plugins.security.disabled: true
plugins.security.ssl.http.enabled: false
plugins.security.ssl.transport.enabled: false
# dev index settings
cluster.routing.allocation.disk.watermark.low: 95%
cluster.routing.allocation.disk.watermark.high: 97%
cluster.routing.allocation.disk.watermark.flood_stage: 98%
YAML

  echo "[build] Wrote single-node config."
  echo "[build] To start: $(basename "$0") run"
}

step_run() {
  cd "$RUNTIME_DIR/opensearch"
  export OPENSEARCH_JAVA_OPTS="-Xms${JVM_HEAP_GB}g -Xmx${JVM_HEAP_GB}g"
  if [[ -n "$JAVA_HOME_HINT" ]]; then
    export OPENSEARCH_JAVA_HOME="$JAVA_HOME_HINT"
  fi
  echo "[run] OPENSEARCH_JAVA_OPTS=$OPENSEARCH_JAVA_OPTS"
  echo "[run] Starting in background..."
  nohup ./bin/opensearch >/tmp/opensearch.out 2>&1 &
  echo $! > /tmp/opensearch.pid
  echo "[run] PID: $(cat /tmp/opensearch.pid)"
  echo "[run] Logs: tail -f /tmp/opensearch.out"
  echo "[run] Health: curl -s http://127.0.0.1:9200 | jq ."
}

step_stop() {
  if [[ -f /tmp/opensearch.pid ]]; then
    kill "$(cat /tmp/opensearch.pid)" || true
    rm -f /tmp/opensearch.pid
    echo "[stop] Stopped."
  else
    pkill -f "opensearch" || true
    echo "[stop] Sent kill to opensearch."
  fi
}

case "${1:-build}" in
  build) step_build ;;
  run) step_run ;;
  stop) step_stop ;;
  *) echo "Usage: $0 [build|run|stop]"; exit 1 ;;
 esac
