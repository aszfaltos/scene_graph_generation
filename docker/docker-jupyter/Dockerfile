# Scene Graph Generation Jupyter Docker Image with uv
# CUDA 12.4 + Python 3.13 + PyTorch 2.x + Jupyter

FROM nvidia/cuda:12.4.0-cudnn-devel-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    ca-certificates \
    build-essential \
    cmake \
    ninja-build \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /workspace

# Copy project files
COPY pyproject.toml uv.lock .python-version ./

# Install Python and dependencies + Jupyter
RUN uv python install 3.13 && \
    uv sync --frozen && \
    uv add jupyterlab notebook

# Copy source code
COPY . .

# Build C++ extensions with CUDA
ENV FORCE_CUDA=1
RUN uv run python setup.py build_ext --inplace

# Create notebooks directory
RUN mkdir -p /notebooks
WORKDIR /notebooks

# Expose Jupyter port
EXPOSE 8888

# Start Jupyter
CMD ["uv", "run", "jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
