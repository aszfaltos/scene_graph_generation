# Troubleshooting

Here is a compilation of common issues that you might face
while compiling / running this code:

## uv sync fails with dependency resolution errors

If `uv sync` fails, try:
```bash
# Clear uv cache
uv cache clean

# Try with verbose output
uv sync -v
```

## C++ extension fails to build

If you see errors during `uv run python setup.py build_ext --inplace`:

1. **Check GCC version** (need >= 9):
   ```bash
   gcc --version
   ```

2. **Check CUDA version** (need 12.x for GPU support):
   ```bash
   nvcc --version
   ```

3. **Clean and rebuild**:
   ```bash
   rm -rf build/ pysgg/*.so
   uv run python setup.py build_ext --inplace
   ```

## ImportError: dlopen ... Library not loaded: @rpath/libc10.dylib (macOS)

This happens when the C++ extension can't find PyTorch libraries. Rebuild with:
```bash
rm -rf build/ pysgg/*.so
uv run python setup.py build_ext --inplace
```

The setup.py includes proper rpath settings for macOS.

## ImportError: Undefined symbol: __cudaPopCallConfiguration

This usually means CUDA version mismatch. Ensure your system CUDA matches the PyTorch CUDA version:

```bash
# Check system CUDA
nvcc --version

# Check PyTorch CUDA
uv run python -c "import torch; print(torch.version.cuda)"
```

Both should match (e.g., both 12.4).

## Compilation errors with old GCC

If you see errors like:
```
/usr/include/c++/6/type_traits:1558:8: note: provided for 'template<class _From, class _To> struct std::is_convertible'
```

You need GCC >= 9. On Ubuntu:
```bash
sudo apt-get install gcc-11 g++-11
export CC=gcc-11
export CXX=g++-11
```

## torch.cuda.is_available() returns False

1. **Check NVIDIA driver**:
   ```bash
   nvidia-smi
   ```

2. **Check CUDA installation**:
   ```bash
   nvcc --version
   ```

3. **Reinstall PyTorch with CUDA**:
   ```bash
   uv sync --reinstall
   ```

## Segmentation fault (core dumped)

This usually means ABI incompatibility. Clean and rebuild:
```bash
rm -rf build/ pysgg/*.so .venv/
uv sync
uv run python setup.py build_ext --inplace
```

## Out of memory errors during training

Reduce batch size in your config file or use gradient accumulation.

## SLURM job fails

Check that the SLURM script uses the correct Python path:
```bash
# For uv installations
uv run python train.py

# Or with explicit venv
source /path/to/project/.venv/bin/activate
python train.py
```
