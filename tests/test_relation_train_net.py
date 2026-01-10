# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Tests for relation_train_net.py device compatibility.
"""

import unittest
import os


class TestRelationTrainNetDeviceCompatibility(unittest.TestCase):
    """Tests for relation_train_net.py device handling."""

    def test_imports_device_utility(self):
        """relation_train_net.py should import device utilities."""
        filepath = "tools/relation_train_net.py"
        with open(filepath, 'r') as f:
            content = f.read()
        self.assertIn('from pysgg.utils.device import', content,
                     "Should import from pysgg.utils.device")

    def test_uses_supports_amp(self):
        """relation_train_net.py should use supports_amp for conditional AMP."""
        filepath = "tools/relation_train_net.py"
        with open(filepath, 'r') as f:
            content = f.read()
        self.assertIn('supports_amp', content,
                     "Should use supports_amp for conditional AMP")


if __name__ == "__main__":
    unittest.main()
