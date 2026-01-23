# Scene Graph Generation Docker Image with uv
# CUDA 12.4 + Python 3.13 + PyTorch 2.x

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

# Install Python and dependencies
RUN uv python install 3.13 && uv sync --frozen

# Copy source code
COPY . .

# Build C++ extensions with CUDA
ENV FORCE_CUDA=1
RUN uv run python setup.py build_ext --inplace

# Set default command
CMD ["uv", "run", "python"]
