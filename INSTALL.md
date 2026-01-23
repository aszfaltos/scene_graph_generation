## Installation

### Requirements
- Python >= 3.13
- PyTorch >= 2.5.0
- torchvision >= 0.20.0
- CUDA 12.x (for GPU support)
- GCC >= 9

### Step-by-step installation with uv (Recommended)

[uv](https://docs.astral.sh/uv/) is a fast Python package manager that replaces conda/pip.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/aszfaltos/scene_graph_generation.git
cd scene_graph_generation

# Install Python 3.13 and create environment
uv python install 3.13
uv sync

# Build C++ extensions
uv run python setup.py build_ext --inplace

# Install third-party dependencies (optional, for full functionality)
mkdir -p third_party
cd third_party

# Install pycocotools (already included via uv, but for reference)
# The pycocotools package is installed automatically via pyproject.toml

# Clone Scene-Graph-Benchmark if needed
git clone https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch.git
cd ..

# Verify installation
uv run python -c "import torch; import pysgg; print(f'PyTorch: {torch.__version__}')"
```

### Running commands

With uv, prefix your Python commands with `uv run`:

```bash
# Run training
uv run python train.py

# Run evaluation
uv run python eval.py

# Start IPython
uv run ipython
```

### Alternative: Legacy conda installation

If you prefer conda (not recommended for new installations):

```bash
conda create -y --name pysgg python=3.13
conda activate pysgg

# Install dependencies via pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-legacy.txt

# Build extensions
python setup.py build develop
```

### System Dependencies

For CUDA support, ensure you have the CUDA Toolkit installed:

**Ubuntu/Debian:**
```bash
# CUDA 12.4
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install cuda-toolkit-12-4
```

**macOS (CPU only):**
CUDA is not supported on macOS. The package will use CPU-only PyTorch automatically.

### Troubleshooting

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues and solutions.
