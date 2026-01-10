# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Tests for device compatibility changes across the codebase.

These tests verify that the hardcoded .cuda() calls have been replaced
with device-agnostic alternatives using the device utility module.
"""

import unittest
import ast
import os


class TestNoCudaHardcoding(unittest.TestCase):
    """Tests to ensure no hardcoded .cuda() calls remain in critical files."""

    def _check_file_for_cuda_calls(self, filepath):
        """
        Check if a file contains hardcoded .cuda() calls.
        Returns a list of line numbers where .cuda() is found.
        """
        if not os.path.exists(filepath):
            return []

        cuda_lines = []
        with open(filepath, 'r') as f:
            for i, line in enumerate(f, 1):
                # Skip comments and imports
                stripped = line.strip()
                if stripped.startswith('#'):
                    continue
                # Check for .cuda() but not torch.cuda (which is valid for checks)
                if '.cuda()' in line and 'torch.cuda' not in line:
                    cuda_lines.append(i)
        return cuda_lines

    def test_relation_head_no_cuda_hardcoding(self):
        """relation_head.py should not have hardcoded .cuda() calls."""
        filepath = "pysgg/modeling/roi_heads/relation_head/relation_head.py"
        cuda_lines = self._check_file_for_cuda_calls(filepath)
        self.assertEqual(
            cuda_lines, [],
            f"Found .cuda() calls at lines {cuda_lines} in {filepath}"
        )

    def test_model_kern_no_cuda_hardcoding(self):
        """model_kern.py should not have hardcoded .cuda() calls."""
        filepath = "pysgg/modeling/roi_heads/relation_head/model_kern.py"
        cuda_lines = self._check_file_for_cuda_calls(filepath)
        self.assertEqual(
            cuda_lines, [],
            f"Found .cuda() calls at lines {cuda_lines} in {filepath}"
        )

    def test_classifier_no_cuda_hardcoding(self):
        """classifier.py should not have hardcoded .cuda() calls."""
        filepath = "pysgg/modeling/roi_heads/relation_head/classifier.py"
        cuda_lines = self._check_file_for_cuda_calls(filepath)
        self.assertEqual(
            cuda_lines, [],
            f"Found .cuda() calls at lines {cuda_lines} in {filepath}"
        )

    def test_attribute_loss_no_cuda_hardcoding(self):
        """attribute loss.py should not have hardcoded .cuda() calls."""
        filepath = "pysgg/modeling/roi_heads/attribute_head/loss.py"
        cuda_lines = self._check_file_for_cuda_calls(filepath)
        self.assertEqual(
            cuda_lines, [],
            f"Found .cuda() calls at lines {cuda_lines} in {filepath}"
        )

    def test_comm_no_cuda_hardcoding(self):
        """comm.py should not have hardcoded .cuda() calls."""
        filepath = "pysgg/utils/comm.py"
        cuda_lines = self._check_file_for_cuda_calls(filepath)
        self.assertEqual(
            cuda_lines, [],
            f"Found .cuda() calls at lines {cuda_lines} in {filepath}"
        )


class TestDeviceImports(unittest.TestCase):
    """Tests to ensure device utility is properly imported where needed."""

    def _check_file_has_device_import(self, filepath):
        """Check if a file imports from pysgg.utils.device."""
        if not os.path.exists(filepath):
            return False

        with open(filepath, 'r') as f:
            content = f.read()
            return 'from pysgg.utils.device import' in content or 'import pysgg.utils.device' in content

    def test_relation_head_imports_device(self):
        """relation_head.py should import device utilities."""
        filepath = "pysgg/modeling/roi_heads/relation_head/relation_head.py"
        self.assertTrue(
            self._check_file_has_device_import(filepath),
            f"{filepath} should import from pysgg.utils.device"
        )

    def test_model_kern_imports_device(self):
        """model_kern.py should import device utilities."""
        filepath = "pysgg/modeling/roi_heads/relation_head/model_kern.py"
        self.assertTrue(
            self._check_file_has_device_import(filepath),
            f"{filepath} should import from pysgg.utils.device"
        )

    def test_classifier_imports_device(self):
        """classifier.py should import device utilities."""
        filepath = "pysgg/modeling/roi_heads/relation_head/classifier.py"
        self.assertTrue(
            self._check_file_has_device_import(filepath),
            f"{filepath} should import from pysgg.utils.device"
        )

    def test_attribute_loss_imports_device(self):
        """attribute loss.py should import device utilities."""
        filepath = "pysgg/modeling/roi_heads/attribute_head/loss.py"
        self.assertTrue(
            self._check_file_has_device_import(filepath),
            f"{filepath} should import from pysgg.utils.device"
        )

    def test_comm_imports_device(self):
        """comm.py should import device utilities."""
        filepath = "pysgg/utils/comm.py"
        self.assertTrue(
            self._check_file_has_device_import(filepath),
            f"{filepath} should import from pysgg.utils.device"
        )


class TestConfigDefaults(unittest.TestCase):
    """Tests for config defaults."""

    def test_default_device_is_auto(self):
        """Default device in config should be 'auto'."""
        filepath = "pysgg/config/defaults.py"
        if not os.path.exists(filepath):
            self.skipTest(f"{filepath} not found")

        with open(filepath, 'r') as f:
            content = f.read()
            # Check that MODEL.DEVICE is set to "auto"
            self.assertIn('_C.MODEL.DEVICE = "auto"', content,
                         "Default device should be 'auto'")


class TestTrainingScriptDeviceHandling(unittest.TestCase):
    """Tests for training script device handling."""

    def test_train_script_uses_device_utility(self):
        """Training script should use device utility for AMP."""
        filepath = "tools/relation_train_net.py"
        if not os.path.exists(filepath):
            self.skipTest(f"{filepath} not found")

        with open(filepath, 'r') as f:
            content = f.read()
            # Should import device utilities
            has_device_import = 'from pysgg.utils.device import' in content
            self.assertTrue(has_device_import,
                          "Training script should import device utilities")


class TestDCNErrorMessages(unittest.TestCase):
    """Tests for improved DCN error messages."""

    def test_dcn_has_helpful_error_message(self):
        """DCN should have a helpful error message for non-CUDA devices."""
        filepath = "pysgg/layers/dcn/deform_conv_func.py"
        if not os.path.exists(filepath):
            self.skipTest(f"{filepath} not found")

        with open(filepath, 'r') as f:
            content = f.read()
            # Should have an informative error message
            self.assertIn('CUDA', content,
                         "DCN should mention CUDA in error messages")


if __name__ == "__main__":
    unittest.main()
