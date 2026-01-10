# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Tests for the device utility module.
"""

import os
import unittest
import torch


class TestDeviceModule(unittest.TestCase):
    """Tests for pysgg.utils.device module."""

    def setUp(self):
        """Clear any cached device and environment variables before each test."""
        # Clear environment variable
        if "SGG_DEVICE" in os.environ:
            del os.environ["SGG_DEVICE"]

    def tearDown(self):
        """Clean up after each test."""
        if "SGG_DEVICE" in os.environ:
            del os.environ["SGG_DEVICE"]

    def test_get_device_returns_torch_device(self):
        """get_device should return a torch.device object."""
        from pysgg.utils.device import get_device
        device = get_device()
        self.assertIsInstance(device, torch.device)

    def test_get_device_auto_detection(self):
        """get_device with 'auto' should detect available device."""
        from pysgg.utils.device import get_device
        device = get_device("auto")
        self.assertIsInstance(device, torch.device)
        # Should be one of cuda, mps, or cpu
        self.assertIn(device.type, ["cuda", "mps", "cpu"])

    def test_get_device_respects_env_variable(self):
        """get_device should respect SGG_DEVICE environment variable."""
        from pysgg.utils.device import get_device
        os.environ["SGG_DEVICE"] = "cpu"
        device = get_device("cuda")  # Config says cuda, but env says cpu
        self.assertEqual(device.type, "cpu")

    def test_get_device_cpu_always_available(self):
        """get_device with 'cpu' should always return cpu device."""
        from pysgg.utils.device import get_device
        device = get_device("cpu")
        self.assertEqual(device.type, "cpu")

    def test_is_cuda_returns_bool(self):
        """is_cuda should return a boolean."""
        from pysgg.utils.device import is_cuda
        result = is_cuda()
        self.assertIsInstance(result, bool)

    def test_is_mps_returns_bool(self):
        """is_mps should return a boolean."""
        from pysgg.utils.device import is_mps
        result = is_mps()
        self.assertIsInstance(result, bool)

    def test_is_cpu_returns_bool(self):
        """is_cpu should return a boolean."""
        from pysgg.utils.device import is_cpu
        result = is_cpu()
        self.assertIsInstance(result, bool)

    def test_to_device_moves_tensor(self):
        """to_device should move a tensor to the specified device."""
        from pysgg.utils.device import to_device
        tensor = torch.zeros(10)
        device = torch.device("cpu")
        result = to_device(tensor, device)
        self.assertEqual(result.device.type, "cpu")

    def test_to_device_with_none_uses_default(self):
        """to_device with device=None should use get_device()."""
        from pysgg.utils.device import to_device, get_device
        os.environ["SGG_DEVICE"] = "cpu"
        tensor = torch.zeros(10)
        result = to_device(tensor)
        expected_device = get_device()
        self.assertEqual(result.device.type, expected_device.type)

    def test_to_device_works_with_module(self):
        """to_device should work with nn.Module."""
        from pysgg.utils.device import to_device
        module = torch.nn.Linear(10, 10)
        device = torch.device("cpu")
        result = to_device(module, device)
        # Check the module's parameters are on the right device
        for param in result.parameters():
            self.assertEqual(param.device.type, "cpu")

    def test_supports_amp_returns_bool(self):
        """supports_amp should return a boolean."""
        from pysgg.utils.device import supports_amp
        result = supports_amp()
        self.assertIsInstance(result, bool)

    def test_supports_amp_false_for_cpu(self):
        """supports_amp should return False for CPU device."""
        from pysgg.utils.device import supports_amp
        os.environ["SGG_DEVICE"] = "cpu"
        # Need to reload module to pick up new env
        import importlib
        import pysgg.utils.device as device_module
        importlib.reload(device_module)
        result = device_module.supports_amp()
        self.assertFalse(result)

    def test_synchronize_does_not_raise(self):
        """synchronize should not raise an exception."""
        from pysgg.utils.device import synchronize
        # Should not raise
        synchronize()

    def test_empty_cache_does_not_raise(self):
        """empty_cache should not raise an exception."""
        from pysgg.utils.device import empty_cache
        # Should not raise
        empty_cache()

    def test_max_memory_allocated_returns_int(self):
        """max_memory_allocated should return an integer."""
        from pysgg.utils.device import max_memory_allocated
        result = max_memory_allocated()
        self.assertIsInstance(result, int)
        self.assertGreaterEqual(result, 0)

    def test_set_device_does_not_raise(self):
        """set_device should not raise an exception."""
        from pysgg.utils.device import set_device
        # Should not raise even on CPU
        set_device(0)

    def test_manual_seed_does_not_raise(self):
        """manual_seed should not raise an exception."""
        from pysgg.utils.device import manual_seed
        # Should not raise
        manual_seed(42)

    def test_get_distributed_backend_returns_string(self):
        """get_distributed_backend should return a string."""
        from pysgg.utils.device import get_distributed_backend
        result = get_distributed_backend()
        self.assertIsInstance(result, str)
        self.assertIn(result, ["nccl", "gloo"])

    def test_get_distributed_backend_gloo_for_cpu(self):
        """get_distributed_backend should return 'gloo' for CPU."""
        os.environ["SGG_DEVICE"] = "cpu"
        import importlib
        import pysgg.utils.device as device_module
        importlib.reload(device_module)
        result = device_module.get_distributed_backend()
        self.assertEqual(result, "gloo")

    def test_get_device_properties_returns_dict(self):
        """get_device_properties should return a dictionary."""
        from pysgg.utils.device import get_device_properties
        result = get_device_properties()
        self.assertIsInstance(result, dict)
        self.assertIn("type", result)
        self.assertIn("device", result)
        self.assertIn("name", result)


class TestDeviceFallback(unittest.TestCase):
    """Tests for device fallback behavior."""

    def test_invalid_device_falls_back_to_cpu(self):
        """Invalid device string should fall back to CPU."""
        os.environ["SGG_DEVICE"] = "invalid_device_xyz"
        import importlib
        import pysgg.utils.device as device_module
        importlib.reload(device_module)
        device = device_module.get_device()
        self.assertEqual(device.type, "cpu")

    def tearDown(self):
        if "SGG_DEVICE" in os.environ:
            del os.environ["SGG_DEVICE"]


if __name__ == "__main__":
    unittest.main()
