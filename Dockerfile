# ==========================================
# GRACE — Whole Head Segmentation (GPU)
# Multi-stage build; CUDA-enabled PyTorch
# ==========================================

# ---------- Stage 1: builder (installs deps into a venv) ----------
# Pick a base matching your host driver stack; CUDA 12.1 is widely supported.
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100

WORKDIR /workspace

# Copy only requirements first to maximize Docker layer caching
COPY requirements.txt /workspace/requirements.txt

# Create an isolated virtualenv and install deps (incl. MONAI/PyTorch add-ons from requirements.txt)
RUN python -m venv /opt/venv && \
    . /opt/venv/bin/activate && \
    pip install --upgrade pip wheel && \
    pip install -r /workspace/requirements.txt

# ---------- Stage 2: runtime (final, smaller image) ----------
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime AS runtime

# tini = clean signal handling; ca-certificates for HTTPS; jq/coreutils handy for ops
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates tini gosu jq coreutils && \
    rm -rf /var/lib/apt/lists/*

# Bring the prebuilt venv from builder
COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    RUNTIME_DEVICE=auto \
    PYTORCH_ENABLE_MPS_FALLBACK=1

# Project source (your repo root). Contains your Python code + the three scripts.
WORKDIR /workspace
COPY . /workspace

# A tiny generic preprocessor (fallback) that can do the full pipeline if desired.
# You can ignore it—your three scripts are used by default.
RUN mkdir -p /opt/grace/bin

# Entrypoint that dispatches: preprocess | train | test
COPY ./entrypoint.sh /opt/grace/entrypoint.sh

RUN chmod +x /opt/grace/entrypoint.sh

# Defaults (can be overridden at runtime)
ENV GRACE_DATA_DIR=/data \
    GRACE_WORKDIR=/workspace \
    GRACE_PREPROCESS_SCRIPT=/workspace/preprocess.py \
    GRACE_TRAIN_SCRIPT=/workspace/train.py \
    GRACE_TEST_SCRIPT=/workspace/test.py

# tini = PID1 for clean signal handling
ENTRYPOINT ["/usr/bin/tini",  "--", "/opt/grace/entrypoint.sh"]
CMD ["--help"]

