#!/usr/bin/env bash
set -euo pipefail

# Usage (run on EACH host):
#   bash setup_tpu_host.sh \
#     --repo-root /home/nick/aura_tpu_mixed \
#     --data /home/nick/aura_tpu_mixed/data/json/emotions.jsonl \
#     --coordinator 10.130.0.23:12355 \
#     --num-proc 4 \
#     --proc-id 0
# Then start training:
#   COORDINATOR_ADDRESS=${COORDINATOR} NUM_PROCESSES=${NUM_PROC} PROCESS_ID=${PROC_ID} \
#   python ${REPO_ROOT}/train_tpu_v4.py \
#     --data ${DATA} --epochs 10 --batch-size 128 --lr 3e-5 \
#     --coordinator-address ${COORDINATOR} --num-processes ${NUM_PROC} --process-id ${PROC_ID}

REPO_ROOT=""
DATA=""
COORDINATOR=""
NUM_PROC=""
PROC_ID=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-root) REPO_ROOT="$2"; shift 2;;
    --data) DATA="$2"; shift 2;;
    --coordinator) COORDINATOR="$2"; shift 2;;
    --num-proc) NUM_PROC="$2"; shift 2;;
    --proc-id) PROC_ID="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

if [[ -z "${REPO_ROOT}" || -z "${DATA}" || -z "${COORDINATOR}" || -z "${NUM_PROC}" || -z "${PROC_ID}" ]]; then
  echo "Missing required args. See header usage." >&2
  exit 1
fi

if [[ ! -f "${REPO_ROOT}/train_tpu_v4.py" ]]; then
  echo "train_tpu_v4.py not found at ${REPO_ROOT}." >&2
  exit 1
fi

# Ensure user local bin is in PATH
export PATH="$HOME/.local/bin:$PATH"
# Set JAX to use TPU platform
export JAX_PLATFORMS=tpu

# 1) Python deps — pin compatible versions and ensure NumPy ≥ 2.0
# Use python3.12 -m pip to ensure we're using Python 3.12 (default is 3.10)
python3.12 -m pip install -U numpy>=2.0.0 --quiet --user

# Install JAX with TPU support (try jax[tpu] first, fallback to jax if it fails)
if python3.12 -m pip install -U "jax[tpu]==0.4.31" jaxlib==0.4.31 --quiet --user 2>&1; then
  echo "Installed jax[tpu] successfully"
else
  echo "jax[tpu] install failed, trying jax without TPU extra (libtpu should be provided by TPU runtime)"
  python3.12 -m pip install -U jax==0.4.31 jaxlib==0.4.31 --quiet --user
fi

python3.12 -m pip install -U flax optax sentence-transformers spacy scikit-learn --quiet --user

# 2) SpaCy model (silent ok)
python3.12 -m spacy download en_core_web_sm >/dev/null 2>&1 || true

# 3) Sanity checks
python3.12 - <<'PY'
import numpy as np
print("NumPy version:", np.__version__)
import jax
print("JAX version:", jax.__version__)
print("Devices:", jax.devices())
PY

# 4) Echo next-step run command tailored for this host
PYTHON_RUNNER="python3.12"

cat <<EOF

Setup complete on host (proc-id=${PROC_ID}/${NUM_PROC}).
Run this command to start training:

JAX_PLATFORMS=tpu COORDINATOR_ADDRESS=${COORDINATOR} NUM_PROCESSES=${NUM_PROC} PROCESS_ID=${PROC_ID} \
${PYTHON_RUNNER} ${REPO_ROOT}/train_tpu_v4.py \
  --data ${DATA} --epochs 10 --batch-size 128 --lr 3e-5 \
  --coordinator-address ${COORDINATOR} --num-processes ${NUM_PROC} --process-id ${PROC_ID}

Tip: use tmux to keep the session alive:  tmux new -s train
EOF
