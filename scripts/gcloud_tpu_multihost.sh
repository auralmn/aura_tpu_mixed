#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Create and manage a multi-host TPU VM pod and launch Aura pretraining with JAX distributed
# Requirements: gcloud, jq
# Usage examples:
#   export PROJECT=<gcp-project>
#   export ZONE=<tpu-zone>                # e.g. us-central2-b, europe-west4-a
#   export NAME=aura-v4-32
#   ./scripts/gcloud_tpu_multihost.sh versions
#   ./scripts/gcloud_tpu_multihost.sh create                # create TPU VM pod (defaults: v4-32)
#   ./scripts/gcloud_tpu_multihost.sh describe              # show worker IPs
#   ./scripts/gcloud_tpu_multihost.sh setup --repo_url https://github.com/auralmn/aura_tpu_mixed.git
#   ./scripts/gcloud_tpu_multihost.sh launch_pretrain_hf     # start distributed run (edit args below or pass env)
#   ./scripts/gcloud_tpu_multihost.sh stop                   # stop python on all workers
#   ./scripts/gcloud_tpu_multihost.sh delete                 # delete TPU pod

set -Eeuo pipefail

PROJECT=${PROJECT:-$(gcloud config get-value project 2>/dev/null || echo "")}
ZONE=${ZONE:-}
NAME=${NAME:-aura-v4-32}
ACCEL=${ACCEL:-v4-32}
# To see available versions for your zone, run: ./scripts/gcloud_tpu_multihost.sh versions
VERSION=${VERSION:-}
PORT=${PORT:-12355}
REPO_URL=${REPO_URL:-https://github.com/auralmn/aura_tpu_mixed.git}
REPO_DIR=${REPO_DIR:-$HOME/aura_tpu}
PYTHON_BIN=${PYTHON_BIN:-python3}
VENV_ACTIVATE=${VENV_ACTIVATE:-"source $REPO_DIR/.venv/bin/activate"}
COORDINATOR_WORKER=${COORDINATOR_WORKER:-0}

# Training args (edit or override via env)
DATASET=${DATASET:-allenai/c4}
CONFIG=${CONFIG:-en}
SPLIT=${SPLIT:-train}
TEXT_KEY=${TEXT_KEY:-text}
SPM_MODEL=${SPM_MODEL:-models/spm/spiece.model}
STEPS=${STEPS:-1200}
LR=${LR:-8e-4}
SEQ_LEN=${SEQ_LEN:-256}
BATCH_SIZE=${BATCH_SIZE:-128}
DTYPE=${DTYPE:-bf16}
PER_DEVICE_BATCH=${PER_DEVICE_BATCH:-8}
CKPT_OUT=${CKPT_OUT:-models/aura/adapter_ckpt_c4.pkl}
# Optional mix dataset (leave empty to disable)
DATASET2=${DATASET2:-}
CONFIG2=${CONFIG2:-}
SPLIT2=${SPLIT2:-}
MIX_P2=${MIX_P2:-0.10}

ensure() { command -v "$1" >/dev/null 2>&1 || { echo "[error] Missing command: $1"; exit 1; }; }
need_proj_zone() {
  [[ -n "$PROJECT" && -n "$ZONE" ]] || { echo "[error] Set PROJECT and ZONE env vars"; exit 1; }
}

# Use GA if available, else fall back to alpha
pick_tpu_cmd() {
  if gcloud compute tpus tpu-vm --help >/dev/null 2>&1; then
    echo "gcloud compute tpus tpu-vm"
  else
    echo "gcloud alpha compute tpus tpu-vm"
  fi
}

versions() {
  need_proj_zone
  TPU_CMD=$(pick_tpu_cmd)
  $TPU_CMD versions list --project "$PROJECT" --zone "$ZONE"
}

create() {
  need_proj_zone
  TPU_CMD=$(pick_tpu_cmd)
  if [[ -z "$VERSION" ]]; then
    echo "[info] VERSION not set. Listing available versions for $ZONE..."
    $TPU_CMD versions list --project "$PROJECT" --zone "$ZONE"
    echo "[hint] Export VERSION=<one-of-above> and re-run create"
    exit 1
  fi
  echo "[create] $NAME in $ZONE type=$ACCEL version=$VERSION"
  $TPU_CMD create "$NAME" \
    --project "$PROJECT" --zone "$ZONE" \
    --accelerator-type "$ACCEL" \
    --version "$VERSION"
}

delete_tpu() {
  need_proj_zone
  TPU_CMD=$(pick_tpu_cmd)
  echo "[delete] $NAME"
  $TPU_CMD delete "$NAME" --project "$PROJECT" --zone "$ZONE" --quiet
}

get_desc_json() {
  need_proj_zone
  TPU_CMD=$(pick_tpu_cmd)
  $TPU_CMD describe "$NAME" --project "$PROJECT" --zone "$ZONE" --format json
}

describe() {
  ensure jq
  js=$(get_desc_json)
  echo "$js" | jq '{name:.name, zone:.zone, acceleratorType:.acceleratorType, networkEndpoints: [.networkEndpoints[] | {worker:.ipAddress}]}'
}

get_worker_count() {
  ensure jq
  js=$(get_desc_json)
  echo "$js" | jq -r '.networkEndpoints | length'
}

get_worker_ip() {
  # $1 = worker index
  ensure jq
  js=$(get_desc_json)
  echo "$js" | jq -r ".networkEndpoints[$1].ipAddress"
}

ssh_all() {
  need_proj_zone
  TPU_CMD=$(pick_tpu_cmd)
  $TPU_CMD ssh "$NAME" --project "$PROJECT" --zone "$ZONE" --worker=all --command "$*"
}

ssh_worker() {
  # $1=idx, $2=command
  need_proj_zone
  TPU_CMD=$(pick_tpu_cmd)
  local idx="$1"; shift
  $TPU_CMD ssh "$NAME" --project "$PROJECT" --zone "$ZONE" --worker=worker-$idx --command "$*"
}

setup() {
  # Clone repo and run local setup script on all workers
  ensure jq
  local cmd=
  read -r -d '' cmd <<CMD || true
set -e
if [[ ! -d "$REPO_DIR" ]]; then
  git clone "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"
# optional: checkout your branch
# git fetch && git checkout <branch>
chmod +x scripts/tpu_setup.sh || true
REPO_DIR="$REPO_DIR" bash scripts/tpu_setup.sh
CMD
  ssh_all "$cmd"
  echo "[setup] completed on all workers"
}

stop() {
  # kill training on all workers
  ssh_all "pkill -f build_aura_model.py || true; pkill -f python || true"
  echo "[stop] sent kill to all workers"
}

launch_pretrain_hf() {
  ensure jq
  local N=$(get_worker_count)
  local COORD_IP=$(get_worker_ip "$COORDINATOR_WORKER")
  echo "[launch] workers=$N coordinator=worker-$COORDINATOR_WORKER ip=$COORD_IP port=$PORT"

  for ((i=0;i< N;i++)); do
    # Build remote command
    # Use nohup to detach; stdout/stderr to logs per worker
    read -r -d '' REMOTE <<RCMD || true
set -e
cd "$REPO_DIR"
$VENV_ACTIVATE || true
export JAX_PLATFORMS=tpu
mkdir -p logs
RUN_LOG=logs/train_worker_${i}.out
if [[ -n "$DATASET2" ]]; then
  nohup $PYTHON_BIN src/aura/self_teaching_llm/build_aura_model.py pretrain_hf \
    --dataset "$DATASET" --config "$CONFIG" --split "$SPLIT" --text_key "$TEXT_KEY" \
    --dataset2 "$DATASET2" --config2 "$CONFIG2" --split2 "$SPLIT2" --mix_p2 "$MIX_P2" \
    --spm_model "$SPM_MODEL" \
    --steps "$STEPS" --lr "$LR" --seq_len "$SEQ_LEN" \
    --batch_size "$BATCH_SIZE" --dtype "$DTYPE" \
    --init_distributed --coordinator ${COORD_IP}:$PORT --process_count $N --process_index $i \
    --pmap --per_device_batch "$PER_DEVICE_BATCH" \
    --ckpt_out "$CKPT_OUT" > "$RUN_LOG" 2>&1 & echo \$! > logs/train_worker_${i}.pid
else
  nohup $PYTHON_BIN src/aura/self_teaching_llm/build_aura_model.py pretrain_hf \
    --dataset "$DATASET" --config "$CONFIG" --split "$SPLIT" --text_key "$TEXT_KEY" \
    --spm_model "$SPM_MODEL" \
    --steps "$STEPS" --lr "$LR" --seq_len "$SEQ_LEN" \
    --batch_size "$BATCH_SIZE" --dtype "$DTYPE" \
    --init_distributed --coordinator ${COORD_IP}:$PORT --process_count $N --process_index $i \
    --pmap --per_device_batch "$PER_DEVICE_BATCH" \
    --ckpt_out "$CKPT_OUT" > "$RUN_LOG" 2>&1 & echo \$! > logs/train_worker_${i}.pid
fi
RCMD
    ssh_worker "$i" "$REMOTE"
    echo "[launch] started worker-$i"
  done
  echo "[launch] all workers started. Tail logs with: ./scripts/gcloud_tpu_multihost.sh tail"
}

tail_logs() {
  need_proj_zone
  TPU_CMD=$(pick_tpu_cmd)
  echo "[tail] Ctrl-C to stop"
  # stream from coordinator by default
  $TPU_CMD ssh "$NAME" --project "$PROJECT" --zone "$ZONE" --worker=worker-$COORDINATOR_WORKER --command "tail -f $REPO_DIR/logs/train_worker_${COORDINATOR_WORKER}.out"
}

case "${1:-help}" in
  versions) versions ;;
  create) create ;;
  describe) describe ;;
  setup) shift; while [[ $# -gt 0 ]]; do case "$1" in --repo_url) REPO_URL="$2"; shift 2;; *) shift;; esac; done; setup ;;
  launch_pretrain_hf) launch_pretrain_hf ;;
  stop) stop ;;
  tail) tail_logs ;;
  delete) delete_tpu ;;
  *) echo "Usage: $0 {versions|create|describe|setup|launch_pretrain_hf|tail|stop|delete}"; exit 1 ;;
 esac
