# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import PIL
import sys
import torch


def get_pil_version():
    return "\n        Pillow ({})".format(PIL.__version__)


def collect_env_info():
    """Collect environment info, with fallback for Python 3.13+ compatibility."""
    try:
        from torch.utils.collect_env import get_pretty_env_info
        env_str = get_pretty_env_info()
    except (AttributeError, TypeError) as e:
        # Fallback for Python 3.13+ where get_pip_packages may fail
        env_str = (
            f"PyTorch version: {torch.__version__}\n"
            f"Python version: {sys.version}\n"
            f"CUDA available: {torch.cuda.is_available()}\n"
        )
        if torch.cuda.is_available():
            env_str += f"CUDA version: {torch.version.cuda}\n"
            env_str += f"GPU: {torch.cuda.get_device_name(0)}\n"
    env_str += get_pil_version()
    return env_str
