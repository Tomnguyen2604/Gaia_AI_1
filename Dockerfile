# ==============================
# STAGE 1: Builder (CUDA 13.0 + PyTorch Nightly)
# ==============================
FROM nvcr.io/nvidia/cuda:13.0.0-devel-ubuntu22.04 AS builder

# Install system deps
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    ca-certificates \
    git \
    python3.11 \
    python3.11-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python3.11 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch Nightly cu130 (RTX 50-series optimized)
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/nightly/cu130 \
    torch torchvision torchaudio

# Copy requirements and install
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Install huggingface-cli
RUN pip install --no-cache-dir huggingface_hub[cli]

# ==============================
# STAGE 2: Runtime (Minimal)
# ==============================
FROM nvcr.io/nvidia/cuda:13.0.0-runtime-ubuntu22.04

# Install minimal runtime deps
RUN apt-get update && apt-get install -y \
    python3.11 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Create app directory
WORKDIR /app

# Copy application code
COPY Scripts/ ./Scripts/
COPY templates/ ./templates/
COPY data/ ./data/
COPY requirements.txt ./

# Create output directories
RUN mkdir -p /app/models /app/checkpoints

# Expose ports (Streamlit and Gradio)
EXPOSE 8501
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python3 -c "import torch; exit(0) if torch.cuda.is_available() else exit(1)"

# Default command: Launch Streamlit chat UI
CMD ["streamlit", "run", "Scripts/chat_streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]