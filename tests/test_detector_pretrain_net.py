# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Tests for detector_pretrain_net.py device compatibility.
"""

import unittest
import os


class TestDetectorPretrainNetDeviceCompatibility(unittest.TestCase):
    """Tests for detector_pretrain_net.py device handling."""

    def test_imports_device_utility(self):
        """detector_pretrain_net.py should import device utilities."""
        filepath = "tools/detector_pretrain_net.py"
        with open(filepath, 'r') as f:
            content = f.read()
        self.assertIn('from pysgg.utils.device import', content,
                     "Should import from pysgg.utils.device")


if __name__ == "__main__":
    unittest.main()
