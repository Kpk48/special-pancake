"""Unit tests for hierarchical conditioning model structures."""

from __future__ import annotations

import unittest
import torch

from waste_classifier.hierarchical.stage1_model import Stage1Model
from waste_classifier.hierarchical.stage2_model import Stage2Model
from waste_classifier.hierarchical.stage3_model import Stage3Model


class TestHierarchicalConditioning(unittest.TestCase):
    """Tests verify model dimension shapes and correct conditioning logic routing."""

    def test_stage1_output_shape(self) -> None:
        model = Stage1Model()
        x = torch.randn(4, 3, 128, 128)
        logits = model(x)
        self.assertEqual(logits.shape, (4, 2))

    def test_stage2_conditioning_influences_logits(self) -> None:
        model = Stage2Model()
        x = torch.randn(2, 3, 128, 128)
        
        # Condition on stage1 class 0
        cond0 = torch.tensor([0, 0])
        logits_cond0 = model(x, cond0)
        self.assertEqual(logits_cond0.shape, (2, 6))

        # Condition on stage1 class 1
        cond1 = torch.tensor([1, 1])
        logits_cond1 = model(x, cond1)
        self.assertEqual(logits_cond1.shape, (2, 6))

        # The logits MUST differ when conditioning class is changed
        # indicating stage1 predictions are actually routed and used in the linear head
        self.assertFalse(torch.allclose(logits_cond0, logits_cond1, atol=1e-5))

    def test_stage3_conditioning_influences_logits_standard(self) -> None:
        model = Stage3Model(use_prototypical=False)
        x = torch.randn(2, 3, 128, 128)
        
        cond0 = torch.tensor([0, 0])
        logits_cond0 = model(x, cond0)
        self.assertEqual(logits_cond0.shape, (2, 11))

        cond5 = torch.tensor([5, 5])
        logits_cond5 = model(x, cond5)
        self.assertEqual(logits_cond5.shape, (2, 11))

        self.assertFalse(torch.allclose(logits_cond0, logits_cond5, atol=1e-5))

    def test_stage3_conditioning_influences_logits_prototypical(self) -> None:
        model = Stage3Model(use_prototypical=True)
        x = torch.randn(2, 3, 128, 128)
        
        cond0 = torch.tensor([0, 0])
        logits_cond0 = model(x, cond0)
        self.assertEqual(logits_cond0.shape, (2, 11))

        cond5 = torch.tensor([5, 5])
        logits_cond5 = model(x, cond5)
        self.assertEqual(logits_cond5.shape, (2, 11))

        self.assertFalse(torch.allclose(logits_cond0, logits_cond5, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
