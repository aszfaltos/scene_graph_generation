# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Tests for inference.py device compatibility.
"""

import unittest
import os


class TestInferenceDeviceCompatibility(unittest.TestCase):
    """Tests for inference.py device handling."""

    def test_imports_device_utility(self):
        """inference.py should import device utilities."""
        filepath = "pysgg/engine/inference.py"
        with open(filepath, 'r') as f:
            content = f.read()
        self.assertIn('from pysgg.utils.device import', content,
                     "Should import from pysgg.utils.device")


if __name__ == "__main__":
    unittest.main()
