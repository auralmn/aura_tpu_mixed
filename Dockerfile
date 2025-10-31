# Minimal TPU-ready training image
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# UV (fast Python package manager)
RUN pip install --no-cache-dir uv

# Workdir
WORKDIR /app

# Copy project files
# Expect train_tpu_v4.py to be in the build context root
COPY train_tpu_v4.py /app/train_tpu_v4.py

# Install Python deps (TPU-compatible). Need libtpu-nightly (pre-release) to satisfy TPU extras
RUN uv pip install --system -U numpy>=2.0.0 && \
    uv pip install --system -U jax==0.4.31 jaxlib==0.4.31 flax optax sentence-transformers spacy scikit-learn && \
    uv run python -m ensurepip && \
    python -m spacy download en_core_web_sm

# Default command can be overridden by XPK
CMD ["python", "/app/train_tpu_v4.py", "--help"]
