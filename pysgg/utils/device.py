# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Device utility module for cross-platform support (CUDA, MPS, CPU).

This module provides a unified interface for device management, allowing the
codebase to run on NVIDIA GPUs (CUDA), Apple Silicon (MPS), or CPU.

Usage:
    from pysgg.utils.device import get_device, to_device, supports_amp

    # Get the current device
    device = get_device()

    # Move a tensor or module to the appropriate device
    tensor = to_device(tensor)
    model = to_device(model)

    # Check if AMP is supported
    if supports_amp():
        with torch.cuda.amp.autocast():
            ...

Environment Variables:
    SGG_DEVICE: Override device selection ("cuda", "mps", "cpu", or "auto")
"""

import os
import torch
from typing import Optional, Union

# Cache for device to avoid repeated detection
_cached_device: Optional[torch.device] = None


def get_device(cfg_device: Optional[str] = None) -> torch.device:
    """
    Get the appropriate torch.device based on configuration and availability.

    Priority order:
    1. SGG_DEVICE environment variable
    2. cfg_device parameter
    3. Auto-detection (CUDA > MPS > CPU)

    Args:
        cfg_device: Device string from config (e.g., "cuda", "mps", "cpu", "auto")

    Returns:
        torch.device: The selected device
    """
    global _cached_device

    # Check environment variable first
    env_device = os.environ.get("SGG_DEVICE", "").lower()

    if env_device:
        device_str = env_device
    elif cfg_device:
        device_str = cfg_device.lower()
    else:
        device_str = "auto"

    # Handle "auto" detection
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    # Validate and return specific device
    if device_str == "cuda":
        if not torch.cuda.is_available():
            print("Warning: CUDA requested but not available, falling back to CPU")
            return torch.device("cpu")
        return torch.device("cuda")
    elif device_str == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            print("Warning: MPS requested but not available, falling back to CPU")
            return torch.device("cpu")
        return torch.device("mps")
    elif device_str == "cpu":
        return torch.device("cpu")
    else:
        # Try to parse as a specific device string (e.g., "cuda:0")
        try:
            return torch.device(device_str)
        except Exception:
            print(f"Warning: Unknown device '{device_str}', falling back to CPU")
            return torch.device("cpu")


def is_cuda() -> bool:
    """Check if the current device is CUDA."""
    device = get_device()
    return device.type == "cuda"


def is_mps() -> bool:
    """Check if the current device is MPS (Apple Silicon)."""
    device = get_device()
    return device.type == "mps"


def is_cpu() -> bool:
    """Check if the current device is CPU."""
    device = get_device()
    return device.type == "cpu"


def to_device(
    tensor_or_module: Union[torch.Tensor, torch.nn.Module],
    device: Optional[torch.device] = None
) -> Union[torch.Tensor, torch.nn.Module]:
    """
    Move a tensor or module to the appropriate device.

    This is a drop-in replacement for .cuda() calls.

    Args:
        tensor_or_module: The tensor or module to move
        device: Target device (if None, uses get_device())

    Returns:
        The tensor or module on the target device
    """
    if device is None:
        device = get_device()
    return tensor_or_module.to(device)


def supports_amp() -> bool:
    """
    Check if automatic mixed precision (AMP) is supported on the current device.

    Currently, AMP is only fully supported on CUDA devices.
    PyTorch 2.0+ has experimental MPS AMP support, but it's not stable.

    Returns:
        bool: True if AMP is supported
    """
    return is_cuda()


def synchronize() -> None:
    """
    Synchronize the current device.

    This is a device-appropriate replacement for torch.cuda.synchronize().
    """
    if is_cuda():
        torch.cuda.synchronize()
    elif is_mps():
        # MPS synchronization
        if hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()
    # CPU doesn't need synchronization


def empty_cache() -> None:
    """
    Empty the device cache to free memory.

    This is a device-appropriate replacement for torch.cuda.empty_cache().
    """
    if is_cuda():
        torch.cuda.empty_cache()
    elif is_mps():
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
    # CPU doesn't have a cache to empty


def max_memory_allocated() -> int:
    """
    Get the maximum memory allocated on the current device.

    Returns:
        int: Maximum memory allocated in bytes (0 for CPU/MPS)
    """
    if is_cuda():
        return torch.cuda.max_memory_allocated()
    # MPS and CPU don't have this API
    return 0


def set_device(local_rank: int) -> None:
    """
    Set the current device for distributed training.

    This is only meaningful for CUDA devices with multiple GPUs.

    Args:
        local_rank: The local rank/GPU index to use
    """
    if is_cuda():
        torch.cuda.set_device(local_rank)
    # MPS and CPU don't need device setting


def manual_seed(seed: int) -> None:
    """
    Set the random seed for the current device.

    Args:
        seed: The random seed to set
    """
    torch.manual_seed(seed)
    if is_cuda():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    elif is_mps():
        # MPS uses the same manual_seed as CPU
        pass


def get_distributed_backend() -> str:
    """
    Get the appropriate distributed backend for the current device.

    Returns:
        str: "nccl" for CUDA, "gloo" for MPS/CPU
    """
    if is_cuda():
        return "nccl"
    else:
        return "gloo"


def get_device_properties() -> dict:
    """
    Get properties of the current device.

    Returns:
        dict: Device properties including name, type, and memory info
    """
    device = get_device()
    props = {
        "type": device.type,
        "device": str(device),
    }

    if is_cuda():
        cuda_props = torch.cuda.get_device_properties(device)
        props.update({
            "name": cuda_props.name,
            "total_memory": cuda_props.total_memory,
            "major": cuda_props.major,
            "minor": cuda_props.minor,
        })
    elif is_mps():
        props["name"] = "Apple Silicon (MPS)"
    else:
        props["name"] = "CPU"

    return props
