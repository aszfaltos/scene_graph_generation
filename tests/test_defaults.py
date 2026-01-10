# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Tests for config/defaults.py device configuration.
"""

import unittest
import os


class TestDefaultsDeviceConfig(unittest.TestCase):
    """Tests for defaults.py device configuration."""

    def test_default_device_is_auto(self):
        """Default device in config should be 'auto'."""
        filepath = "pysgg/config/defaults.py"
        with open(filepath, 'r') as f:
            content = f.read()
        self.assertIn('_C.MODEL.DEVICE = "auto"', content,
                     "Default device should be 'auto'")


if __name__ == "__main__":
    unittest.main()
