# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Tests for comm.py device compatibility.
"""

import unittest
import os


class TestCommDeviceCompatibility(unittest.TestCase):
    """Tests for comm.py device handling."""

    def test_no_hardcoded_cuda_string(self):
        """comm.py should not have hardcoded 'cuda' device strings."""
        filepath = "pysgg/utils/comm.py"
        with open(filepath, 'r') as f:
            content = f.read()
        # Should not have to_device = "cuda" hardcoded
        self.assertNotIn('to_device = "cuda"', content,
                        "Should not have hardcoded cuda device string")

    def test_imports_device_utility(self):
        """comm.py should import device utilities."""
        filepath = "pysgg/utils/comm.py"
        with open(filepath, 'r') as f:
            content = f.read()
        self.assertIn('from pysgg.utils.device import', content,
                     "Should import from pysgg.utils.device")


if __name__ == "__main__":
    unittest.main()
