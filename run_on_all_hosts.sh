
#!/usr/bin/env bash
set -euo pipefail

# Run commands on all TPU v4-32 hosts automatically (auto-assigns PROC_ID 0,1,2,3)
#
# Usage (PROC_ID is auto-replaced with 0,1,2,3 for each host):
#   bash run_on_all_hosts.sh \
#     --hosts "10.130.0.23,10.130.0.20,10.130.0.21,10.130.0.22" \
#     --user nick \
#     --command "bash /home/nick/aura_tpu_mixed/setup_tpu_host.sh --repo-root /home/nick/aura_tpu_mixed --data /home/nick/aura_tpu_mixed/data/json/emotions.jsonl --coordinator 10.130.0.23:12355 --num-proc 4 --proc-id \$PROC_ID"
#
# Training example (parallel, all start within 60s):
#   bash run_on_all_hosts.sh \
#     --hosts "10.130.0.23,10.130.0.20,10.130.0.21,10.130.0.22" \
#     --user nick \
#     --parallel \
#     --command "COORDINATOR_ADDRESS=10.130.0.23:12355 NUM_PROCESSES=4 PROCESS_ID=\$PROC_ID uv run python /home/nick/aura_tpu_mixed/train_tpu_v4.py --data /home/nick/aura_tpu_mixed/data/json/emotions.jsonl --epochs 10 --batch-size 128 --lr 3e-5 --coordinator-address 10.130.0.23:12355 --num-processes 4 --process-id \$PROC_ID"
#
# Or use gcloud compute tpus tpu-vm ssh (for TPU VMs):
#   bash run_on_all_hosts.sh \
#     --tpu-vm "aura-tpuv" \
#     --zone us-central2-b \
#     --project auragcloudtpu \
#     --command "bash /home/nick/aura_tpu_mixed/setup_tpu_host.sh --repo-root /home/nick/aura_tpu_mixed --data /home/nick/aura_tpu_mixed/data/json/emotions.jsonl --coordinator 10.130.0.23:12355 --num-proc 4 --proc-id \$PROC_ID"
#
# Or use gcloud compute ssh (if using GCE VMs):
#   bash run_on_all_hosts.sh \
#     --gce-names "tpu-vm-0,tpu-vm-1,tpu-vm-2,tpu-vm-3" \
#     --zone us-central2-b \
#     --user nick \
#     --command "bash /home/nick/aura_tpu_mixed/setup_tpu_host.sh --repo-root /home/nick/aura_tpu_mixed --data /home/nick/aura_tpu_mixed/data/json/emotions.jsonl --coordinator 10.130.0.23:12355 --num-proc 4 --proc-id \$PROC_ID"
#
# Options:
#   --parallel: Run on all hosts in parallel (default: sequential)
#   --output-dir: Directory to save per-host logs (default: /tmp/tpu_hosts_logs)

HOSTS_IPS=""
GCE_NAMES=""
TPU_VM=""
ZONE=""
PROJECT=""
USER="nick"
COMMAND=""
PARALLEL=false
OUTPUT_DIR="/tmp/tpu_hosts_logs"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --hosts) HOSTS_IPS="$2"; shift 2;;
    --gce-names) GCE_NAMES="$2"; shift 2;;
    --tpu-vm) TPU_VM="$2"; shift 2;;
    --zone) ZONE="$2"; shift 2;;
    --project) PROJECT="$2"; shift 2;;
    --user) USER="$2"; shift 2;;
    --command) COMMAND="$2"; shift 2;;
    --parallel) PARALLEL=true; shift 1;;
    --output-dir) OUTPUT_DIR="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

if [[ -z "${COMMAND}" ]]; then
  echo "Missing required --command" >&2
  exit 1
fi

# Determine host list
if [[ -n "${TPU_VM}" ]]; then
  if [[ -z "${ZONE}" ]]; then
    echo "Missing --zone when using --tpu-vm" >&2
    exit 1
  fi
  if [[ -z "${PROJECT}" ]]; then
    PROJECT=$(gcloud config get-value project 2>/dev/null || true)
    if [[ -z "${PROJECT}" ]]; then
      echo "Missing --project when using --tpu-vm (and no default gcloud project)" >&2
      exit 1
    fi
  fi
  USE_TPU_VM=true
  HOST_COUNT=4
elif [[ -n "${GCE_NAMES}" ]]; then
  if [[ -z "${ZONE}" ]]; then
    echo "Missing --zone when using --gce-names" >&2
    exit 1
  fi
  IFS=',' read -ra GCE_ARRAY <<< "${GCE_NAMES}"
  USE_GCE=true
elif [[ -n "${HOSTS_IPS}" ]]; then
  IFS=',' read -ra HOST_ARRAY <<< "${HOSTS_IPS}"
  USE_GCE=false
  USE_TPU_VM=false
else
  echo "Provide either --tpu-vm, --hosts, or --gce-names" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

# Function to run command on a host
run_on_host() {
  local host_id=$1
  local host_info=$2
  local proc_id=$3
  local log_file="${OUTPUT_DIR}/host_${proc_id}.log"
  
  # Auto-replace PROC_ID in command
  local cmd_with_proc_id="${COMMAND//\$PROC_ID/${proc_id}}"
  cmd_with_proc_id="${cmd_with_proc_id//PROC_ID/${proc_id}}"
  
  echo "[Host ${proc_id}] Starting on ${host_info}..."
  
  if [[ "${USE_TPU_VM}" == "true" ]]; then
    # Use gcloud compute tpus tpu-vm ssh with --worker flag
    gcloud compute tpus tpu-vm ssh "${TPU_VM}" \
      --zone "${ZONE}" \
      --project "${PROJECT}" \
      --worker "${proc_id}" \
      --command "${cmd_with_proc_id}" \
      > "${log_file}" 2>&1
  elif [[ "${USE_GCE}" == "true" ]]; then
    # Use gcloud compute ssh
    gcloud compute ssh "${host_info}" \
      --zone "${ZONE}" \
      --command "${cmd_with_proc_id}" \
      --user "${USER}" \
      > "${log_file}" 2>&1
  else
    # Try to find VM name from IP and use gcloud compute ssh (handles auth automatically)
    local vm_name=$(gcloud compute instances list --filter="networkInterfaces[0].networkIP:${host_info}" --format="value(name)" --project "$(gcloud config get-value project 2>/dev/null || echo '')" 2>/dev/null | head -n1 || echo "")
    local vm_zone=$(gcloud compute instances list --filter="networkInterfaces[0].networkIP:${host_info}" --format="value(zone)" --project "$(gcloud config get-value project 2>/dev/null || echo '')" 2>/dev/null | head -n1 || echo "")
    
    if [[ -n "${vm_name}" && -n "${vm_zone}" ]]; then
      echo "[Host ${proc_id}] Found GCE VM: ${vm_name} in ${vm_zone}, using gcloud compute ssh"
      gcloud compute ssh "${vm_name}" \
        --zone "${vm_zone}" \
        --command "${cmd_with_proc_id}" \
        --user "${USER}" \
        > "${log_file}" 2>&1
    else
      # Fall back to direct SSH (requires SSH keys to be set up)
      echo "[Host ${proc_id}] Using direct SSH (ensure SSH keys are configured)"
      ssh -o StrictHostKeyChecking=no \
          -o UserKnownHostsFile=/dev/null \
          -o ConnectTimeout=30 \
          "${USER}@${host_info}" \
          "${cmd_with_proc_id}" \
          > "${log_file}" 2>&1
    fi
  fi
  
  local exit_code=$?
  if [[ ${exit_code} -eq 0 ]]; then
    echo "[Host ${proc_id}] SUCCESS (log: ${log_file})"
  else
    echo "[Host ${proc_id}] FAILED (exit ${exit_code}, log: ${log_file})"
  fi
  return ${exit_code}
}

# Run on all hosts
if [[ "${USE_TPU_VM}" == "true" ]]; then
  # TPU VM: workers are 0,1,2,3
  HOST_LIST=(0 1 2 3)
elif [[ "${USE_GCE}" == "true" ]]; then
  HOST_LIST=("${GCE_ARRAY[@]}")
else
  HOST_LIST=("${HOST_ARRAY[@]}")
fi

PROC_ID=0
if [[ "${PARALLEL}" == "true" ]]; then
  echo "Running on all hosts in parallel..."
  PIDS=()
  for host in "${HOST_LIST[@]}"; do
    run_on_host "${host}" "${host}" "${PROC_ID}" &
    PIDS+=($!)
    PROC_ID=$((PROC_ID + 1))
  done
  
  # Wait for all jobs
  EXIT_CODE=0
  for pid in "${PIDS[@]}"; do
    wait ${pid} || EXIT_CODE=$?
  done
  exit ${EXIT_CODE}
else
  echo "Running on all hosts sequentially..."
  EXIT_CODE=0
  for host in "${HOST_LIST[@]}"; do
    run_on_host "${host}" "${host}" "${PROC_ID}" || EXIT_CODE=$?
    PROC_ID=$((PROC_ID + 1))
  done
  exit ${EXIT_CODE}
fi

