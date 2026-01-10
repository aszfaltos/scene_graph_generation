# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""Tests for tools/runner.py - GPU memory placeholder utility."""

import unittest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestRunner(unittest.TestCase):
    """Test the runner.py GPU utility."""

    def test_runner_has_cuda_guard(self):
        """Test that runner.py checks for CUDA availability before running."""
        runner_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'tools', 'runner.py'
        )
        with open(runner_path, 'r') as f:
            content = f.read()

        # Should check for CUDA availability
        self.assertIn('torch.cuda.is_available()', content,
                      "runner.py should check CUDA availability")

    def test_runner_imports_gpustat_conditionally(self):
        """Test that gpustat is imported only when CUDA is available."""
        runner_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'tools', 'runner.py'
        )
        with open(runner_path, 'r') as f:
            content = f.read()

        # gpustat import should come after CUDA check
        cuda_check_pos = content.find('torch.cuda.is_available()')
        gpustat_import_pos = content.find('import gpustat')

        self.assertGreater(gpustat_import_pos, cuda_check_pos,
                          "gpustat should be imported after CUDA check")


if __name__ == '__main__':
    unittest.main()
