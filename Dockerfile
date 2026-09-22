# Hugging Face Spaces Dockerfile for ApexDecision External Benchmark Center
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=7860

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch (CPU fallback / CUDA compatible)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    torch>=2.1 \
    numpy>=1.24 \
    fastapi>=0.115 \
    uvicorn>=0.27 \
    pydantic>=2.0 \
    tokenizers>=0.15 \
    httpx>=0.27 \
    pyyaml>=6.0 \
    psutil>=5.9 \
    gradio>=6.0

# Copy repository files
COPY . /app

# Ensure non-root permissions for Hugging Face Spaces (user 1000)
RUN useradd -m -u 1000 user && \
    chown -R user:user /app
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

EXPOSE 7860

CMD ["python", "benchmark/app.py"]
