# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Tests for attribute_head/loss.py device compatibility.
"""

import unittest
import os


class TestLossDeviceCompatibility(unittest.TestCase):
    """Tests for loss.py device handling."""

    def test_no_hardcoded_cuda_calls(self):
        """loss.py should not have hardcoded .cuda() calls."""
        filepath = "pysgg/modeling/roi_heads/attribute_head/loss.py"
        cuda_lines = []
        with open(filepath, 'r') as f:
            for i, line in enumerate(f, 1):
                stripped = line.strip()
                if stripped.startswith('#'):
                    continue
                if '.cuda()' in line and 'torch.cuda' not in line:
                    cuda_lines.append(i)
        self.assertEqual(cuda_lines, [],
                        f"Found .cuda() calls at lines {cuda_lines}")

    def test_imports_device_utility(self):
        """loss.py should import device utilities."""
        filepath = "pysgg/modeling/roi_heads/attribute_head/loss.py"
        with open(filepath, 'r') as f:
            content = f.read()
        self.assertIn('from pysgg.utils.device import', content,
                     "Should import from pysgg.utils.device")


if __name__ == "__main__":
    unittest.main()
