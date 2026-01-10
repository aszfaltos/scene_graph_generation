# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Tests for deform_conv_func.py error messages.
"""

import unittest
import os


class TestDeformConvFuncErrorMessages(unittest.TestCase):
    """Tests for deform_conv_func.py error handling."""

    def test_has_helpful_error_message(self):
        """deform_conv_func.py should have helpful error messages."""
        filepath = "pysgg/layers/dcn/deform_conv_func.py"
        with open(filepath, 'r') as f:
            content = f.read()
        # Should mention that DCN requires CUDA
        self.assertIn('Deformable Convolution requires CUDA', content,
                     "Should have helpful error message about CUDA requirement")


if __name__ == "__main__":
    unittest.main()
