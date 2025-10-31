#!/usr/bin/env bash
set -euo pipefail

# End-to-end XPK launch script for TPU v4-32 training
#
# Requirements:
# - gcloud auth configured; Artifact Registry enabled
# - docker installed and logged in to Artifact Registry
# - xpk installed (https://github.com/google/xpk)
#
# Usage:
#   bash xpk_e2e.sh \
#     --project YOUR_GCP_PROJECT \
#     --region us-central1 \
#     --repo aura \
#     --bucket YOUR_GCS_BUCKET \
#     --data gs://YOUR_GCS_BUCKET/emotions.jsonl \
#     --epochs 10 \
#     --batch-size 128 \
#     --lr 3e-5 \
#     --image train:latest
#     --image train:latest \
#     [--service-account YOUR_SA@PROJECT.iam.gserviceaccount.com] \
#     [--cluster xpk-orchestrator] [--location us-central1]
#
# Notes:
# - If --project/--region are omitted, they are read from gcloud config.
# - Ensures required services and Artifact Registry repo exist.
# - Optionally grants bucket IAM to the job service account.
# - Builds and pushes REGION-docker.pkg.dev/PROJECT/REPO/IMAGE and submits XPK job.

PROJECT=""
REGION=""
REPO="aura"
IMAGE="train:latest"
BUCKET=""
DATA_GCS=""
EPOCHS="10"
BATCH_SIZE="128"
LR="3e-5"
REPO_ROOT_DIR="/app"
CONTEXT_DIR="."
SERVICE_ACCOUNT=""
CLUSTER_NAME=""
LOCATION_OVERRIDE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project) PROJECT="$2"; shift 2;;
    --region) REGION="$2"; shift 2;;
    --repo) REPO="$2"; shift 2;;
    --image) IMAGE="$2"; shift 2;;
    --bucket) BUCKET="$2"; shift 2;;
    --data) DATA_GCS="$2"; shift 2;;
    --epochs) EPOCHS="$2"; shift 2;;
    --batch-size) BATCH_SIZE="$2"; shift 2;;
    --lr) LR="$2"; shift 2;;
    --context) CONTEXT_DIR="$2"; shift 2;;
    --service-account) SERVICE_ACCOUNT="$2"; shift 2;;
    --cluster) CLUSTER_NAME="$2"; shift 2;;
    --location) LOCATION_OVERRIDE="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

# Auto-detect project/region from gcloud if not provided
if [[ -z "${PROJECT}" ]]; then
  PROJECT=$(gcloud config get-value core/project 2>/dev/null || true)
fi
if [[ -z "${REGION}" ]]; then
  REGION=$(gcloud config get-value compute/region 2>/dev/null || true)
fi

# If a location override is provided, prefer it (XPK uses location)
if [[ -n "${LOCATION_OVERRIDE}" ]]; then
  REGION="${LOCATION_OVERRIDE}"
fi

if [[ -z "${PROJECT}" || -z "${REGION}" || -z "${BUCKET}" || -z "${DATA_GCS}" ]]; then
  echo "Missing required args (project/region/bucket/data). Provide flags or set gcloud config." >&2
  exit 1
fi

ARTIFACT_IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/${IMAGE}"
JOB_NAME="aura-train-$(date +%Y%m%d-%H%M%S)"

# 0) Enable required services and ensure Artifact Registry repo exists
echo "[0/6] Ensuring services and AR repo"
gcloud services enable artifactregistry.googleapis.com --project "${PROJECT}" --quiet || true
gcloud services enable tpu.googleapis.com --project "${PROJECT}" --quiet || true
gcloud artifacts repositories describe "${REPO}" --location "${REGION}" --project "${PROJECT}" >/dev/null 2>&1 || \
gcloud artifacts repositories create "${REPO}" --repository-format docker --location "${REGION}" --project "${PROJECT}" --quiet

# 0.1) Ensure kubectl and kjobctl (XPK prerequisite)
echo "[0/6] Checking kubectl and kjobctl prerequisites"

# Detect OS/ARCH
OS=$(uname | tr '[:upper:]' '[:lower:]')
ARCH_RAW=$(uname -m)
case "$ARCH_RAW" in
  x86_64) ARCH=amd64;;
  aarch64|arm64) ARCH=arm64;;
  *) ARCH=amd64;;
esac

# Install kubectl if missing
if ! command -v kubectl >/dev/null 2>&1; then
  echo "Installing kubectl for ${OS}/${ARCH}..."
  KUBECTL_URL="https://dl.k8s.io/release/$(curl -sL https://dl.k8s.io/release/stable.txt)/bin/${OS}/${ARCH}/kubectl"
  curl -sL "${KUBECTL_URL}" -o kubectl && \
  chmod +x kubectl && \
  sudo mv kubectl /usr/local/bin/kubectl
fi

# Install krew (kubectl plugin manager) if missing
if ! kubectl krew version >/dev/null 2>&1; then
  echo "Installing krew for ${OS}/${ARCH}..."
  KREW_TAR="krew-${OS}_${ARCH}.tar.gz"
  curl -sLO "https://github.com/kubernetes-sigs/krew/releases/latest/download/${KREW_TAR}" && \
  tar zxvf "${KREW_TAR}" >/dev/null && \
  "./krew-${OS}_${ARCH}" install krew >/dev/null && \
  rm -f "${KREW_TAR}"
  export PATH="${KREW_ROOT:-$HOME/.krew}/bin:$PATH"
fi

# Ensure PATH has krew bin for current shell
export PATH="${KREW_ROOT:-$HOME/.krew}/bin:$PATH"

# Install kjobctl plugin if missing (provides `kubectl kjob`)
if ! kubectl kjob --help >/dev/null 2>&1; then
  echo "Installing kjobctl plugin via krew..."
  kubectl krew install kjobctl || true
fi

# Optional: grant bucket IAM to service account
if [[ -n "${SERVICE_ACCOUNT}" ]]; then
  echo "Granting roles/storage.objectViewer on gs://${BUCKET} to ${SERVICE_ACCOUNT}"
  gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" \
    --member="serviceAccount:${SERVICE_ACCOUNT}" \
    --role="roles/storage.objectViewer" --quiet || true
fi

# 1) Build container (expects Dockerfile in context)
echo "[1/6] Building image ${ARTIFACT_IMAGE} from ${CONTEXT_DIR}"
docker build -t "${ARTIFACT_IMAGE}" "${CONTEXT_DIR}"

# 2) Push image
echo "[2/6] Pushing image"
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet
docker push "${ARTIFACT_IMAGE}"

# 3) Ensure kube context (GKE credentials)
echo "[3/6] Ensuring kube context for region=${REGION}"
if [[ -z "${CLUSTER_NAME}" ]]; then
  CLUSTER_NAME=$(gcloud container clusters list --region "${REGION}" --project "${PROJECT}" --format="value(name)" | head -n1 || true)
fi
if [[ -z "${CLUSTER_NAME}" ]]; then
  echo "No GKE cluster found in ${REGION}. Create one: gcloud container clusters create-auto xpk-orchestrator --region ${REGION} --project ${PROJECT}" >&2
  exit 1
fi
echo "Using cluster: ${CLUSTER_NAME}"
gcloud container clusters get-credentials "${CLUSTER_NAME}" --region "${REGION}" --project "${PROJECT}" --dns-endpoint
kubectl config current-context || true

# 4) Generate XPK job spec (tmp file)
YAML_FILE="/tmp/${JOB_NAME}.yaml"

# Optional serviceAccount YAML snippet
SA_YAML=""
if [[ -n "${SERVICE_ACCOUNT}" ]]; then
  SA_YAML="serviceAccount: ${SERVICE_ACCOUNT}"
fi

cat > "${YAML_FILE}" <<EOF
apiVersion: xpk.dev/v1alpha
kind: Job
metadata:
  name: ${JOB_NAME}
spec:
  replicas: 4
  template:
    spec:
      image: ${ARTIFACT_IMAGE}
      tpu:
        type: v4-32
      ${SA_YAML}
      env:
        - name: NUM_PROCESSES
          value: "4"
        - name: PROCESS_ID
          valueFrom:
            fieldRef: xpk.dev/rank
        - name: COORDINATOR_ADDRESS
          valueFrom:
            fieldRef: xpk.dev/host0AddressWithPort:12355
      command:
        - /bin/sh
        - -lc
        - |
          uv run python ${REPO_ROOT_DIR}/train_tpu_v4.py \
            --data ${DATA_GCS} \
            --epochs ${EPOCHS} \
            --batch-size ${BATCH_SIZE} \
            --lr ${LR} \
            --coordinator-address \$COORDINATOR_ADDRESS \
            --num-processes \$NUM_PROCESSES \
            --process-id \$PROCESS_ID
EOF

echo "[4/6] XPK job spec written to ${YAML_FILE}"

# 4.5) Create XPK ConfigMaps if missing (required for existing clusters)
echo "[4.5/6] Ensuring XPK ConfigMaps exist"
if ! kubectl get configmap xpk-orchestrator-resources-configmap >/dev/null 2>&1; then
  echo "Creating required XPK ConfigMaps with expected structure..."
  # Create ConfigMaps with structure XPK expects (with 'map' in the data for parsing)
  kubectl apply -f - <<EOF || true
apiVersion: v1
kind: ConfigMap
metadata:
  name: xpk-orchestrator-resources-configmap
  namespace: default
data:
  system_characteristics: '{"cluster_resources": {}}'
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: xpk-orchestrator-metadata-configmap
  namespace: default
data:
  cluster_metadata: '{"region": "${REGION}", "cluster": "${CLUSTER_NAME}"}'
EOF
  echo "ConfigMaps created"
fi

# 5) Submit job
echo "[5/6] Submitting XPK job ${JOB_NAME}"
xpk run "${YAML_FILE}" --cluster "${CLUSTER_NAME}"

# 6) Tail logs
echo "[6/6] Tailing logs (Ctrl-C to stop)"
# Try exact workload name first; if missing, pick newest workload starting with job name
if xpk workload logs -f "${JOB_NAME}" --cluster "${CLUSTER_NAME}" 2>/dev/null; then
  :
else
  CANDIDATE=$(xpk workload ls --cluster "${CLUSTER_NAME}" 2>/dev/null | awk 'NR>1 {print $1, $4}' | sort -k2r | awk '{print $1}' | grep -m1 "^${JOB_NAME}") || true
  if [[ -n "${CANDIDATE}" ]]; then
    xpk workload logs -f "${CANDIDATE}" --cluster "${CLUSTER_NAME}" || true
  else
    echo "Could not stream logs for ${JOB_NAME}. Showing workload list:"
    xpk workload ls --cluster "${CLUSTER_NAME}" || true
    echo "Use: xpk workload logs -f <workload-name> --cluster ${CLUSTER_NAME}"
  fi
fi

echo "Done"
