# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Tests for relation_test_net.py device compatibility.
"""

import unittest
import os


class TestRelationTestNetDeviceCompatibility(unittest.TestCase):
    """Tests for relation_test_net.py device handling."""

    def test_imports_device_utility(self):
        """relation_test_net.py should import device utilities."""
        filepath = "tools/relation_test_net.py"
        with open(filepath, 'r') as f:
            content = f.read()
        self.assertIn('from pysgg.utils.device import', content,
                     "Should import from pysgg.utils.device")


if __name__ == "__main__":
    unittest.main()
