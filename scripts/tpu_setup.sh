#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# TPU VM setup for Aura (JAX/Flax, datasets, SentencePiece, GCS)
# Usage:
#   chmod +x scripts/tpu_setup.sh
#   ./scripts/tpu_setup.sh

set -Eeuo pipefail

# Configurable via env
TPU_SETUP_VENV="${TPU_SETUP_VENV:-$HOME/.venvs/aura-tpu}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
REPO_DIR="${REPO_DIR:-$HOME/aura_tpu}"
EXTRA_PIP="${EXTRA_PIP:-}"

printf "[setup] venv: %s\n" "$TPU_SETUP_VENV"
printf "[setup] python: %s\n" "$PYTHON_BIN"
printf "[setup] repo dir: %s\n" "$REPO_DIR"

# Optional base packages (Debian/Ubuntu on TPU VM)
sudo apt-get update -y || true
sudo apt-get install -y python3-venv git wget screen tmux || true

# Create venv
if [[ ! -d "$TPU_SETUP_VENV" ]]; then
  "$PYTHON_BIN" -m venv "$TPU_SETUP_VENV"
fi
# shellcheck source=/dev/null
source "$TPU_SETUP_VENV/bin/activate"

python -m pip install --upgrade pip setuptools wheel

# Install JAX TPU build (libtpu) and core deps
pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install -U flax optax sentencepiece datasets gcsfs google-cloud-storage einops tqdm rich
# Optional extras
if [[ -n "${EXTRA_PIP}" ]]; then
  pip install -U ${EXTRA_PIP}
fi

# Optional project .env (for convenience)
if [[ -d "$REPO_DIR" ]]; then
  cat > "$REPO_DIR/.env.tpu" << 'EOF'
# Aura TPU defaults
export JAX_PLATFORMS=tpu
# Example for multihost runs (uncomment and set appropriately):
# export AURA_COORDINATOR="10.0.0.2:12355"
# export AURA_PROCESS_COUNT=4
# export AURA_PROCESS_INDEX=0
# export PYTHONPATH="$(pwd):$PYTHONPATH"
EOF
  printf "[setup] wrote %s/.env.tpu\n" "$REPO_DIR"
fi

# Smoke test
python - << 'PY'
import jax
print("[aura] backend:", jax.default_backend())
print("[aura] device_count:", jax.device_count())
print("[aura] local_devices:", jax.local_device_count())
import jax.numpy as jnp
x = jnp.ones((8,8), dtype=jnp.float32)
print("[aura] sum:", float(jnp.sum(x)))
PY

cat << 'NOTE'
[setup] Done.
Next steps:
- Activate venv:   source "$TPU_SETUP_VENV/bin/activate"
- Single host run example:
  JAX_PLATFORMS=tpu python3 src/aura/self_teaching_llm/build_aura_model.py pretrain_hf \
    --dataset allenai/c4 --config en --split train --text_key text \
    --spm_model models/spm/spiece.model --steps 200 --seq_len 256 \
    --batch_size 128 --dtype bf16 --pmap --per_device_batch 8 \
    --ckpt_out models/aura/adapter_ckpt_smoke.pkl
- Multihost add (per host):
  --init_distributed --coordinator ${AURA_COORDINATOR} \
  --process_count ${AURA_PROCESS_COUNT} --process_index ${AURA_PROCESS_INDEX}
NOTE
