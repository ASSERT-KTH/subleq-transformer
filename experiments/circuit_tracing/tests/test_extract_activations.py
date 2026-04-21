"""
Tests for extract_activations.py.

Verifies:
  1. Output shapes are correct for both model types.
  2. Sum of head_out matches the total attention output.
  3. ffn_pre matches a manual computation through the FFN's first linear.
  4. Residuals are consistent with the model's own forward() output.
  5. get_head_ov_weights returns correct shapes and values.
"""

import sys
import os
import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F

# Path setup
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_CT_DIR = os.path.dirname(_TESTS_DIR)
_EXP_DIR = os.path.dirname(_CT_DIR)
_REPO = os.path.dirname(_EXP_DIR)

for p in [_EXP_DIR, _REPO,
          os.path.join(_REPO, 'round2_trained'),
          os.path.join(_REPO, 'round1_constructed'),
          _CT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from circuit_tracing.extract_activations import (
    get_all_activations,
    get_head_ov_weights,
    _split_head_outputs_constrained,
    _split_head_outputs_hardcoded,
)
from fixtures import MockConstrainedModel

# Alias used throughout this file
_MockConstrainedModel = MockConstrainedModel


# ─── Test cases ───────────────────────────────────────────────────────────────

class TestExtractActivationsShapes(unittest.TestCase):
    """Verify output shapes for the mock constrained model."""

    def setUp(self):
        torch.manual_seed(0)
        self.B, self.T = 3, 5
        self.D, self.H = 8, 2
        self.d_ff = 16
        self.n_layers = 2
        self.model = _MockConstrainedModel(
            d_model=self.D, n_heads=self.H, n_layers=self.n_layers,
            d_ff=self.d_ff, vocab_size=16, seq_len=self.T)
        self.inputs = torch.randint(0, 16, (self.B, self.T))

    def test_all_keys_present(self):
        acts = get_all_activations(self.model, self.inputs)
        required = ['embedding', 'logits']
        for i in range(self.n_layers):
            required += [
                f'attn_weights_{i}', f'head_out_{i}',
                f'residual_post_attn_{i}', f'ffn_pre_{i}',
                f'ffn_post_{i}', f'residual_{i}',
            ]
        for key in required:
            self.assertIn(key, acts, f'Missing key: {key}')

    def test_embedding_shape(self):
        acts = get_all_activations(self.model, self.inputs)
        self.assertEqual(acts['embedding'].shape, (self.B, self.T, self.D))

    def test_attn_weights_shape(self):
        acts = get_all_activations(self.model, self.inputs)
        for i in range(self.n_layers):
            self.assertEqual(acts[f'attn_weights_{i}'].shape,
                             (self.B, self.H, self.T, self.T))

    def test_head_out_shape(self):
        acts = get_all_activations(self.model, self.inputs)
        for i in range(self.n_layers):
            self.assertEqual(acts[f'head_out_{i}'].shape,
                             (self.B, self.H, self.T, self.D))

    def test_ffn_pre_shape(self):
        acts = get_all_activations(self.model, self.inputs)
        for i in range(self.n_layers):
            self.assertEqual(acts[f'ffn_pre_{i}'].shape,
                             (self.B, self.T, self.d_ff))

    def test_logits_shape(self):
        acts = get_all_activations(self.model, self.inputs)
        self.assertEqual(acts['logits'].shape, (self.B, self.T, 16))


class TestHeadOutputConsistency(unittest.TestCase):
    """
    Sum of per-head contributions must equal the total attention module output
    (minus the bias, which is handled separately).
    """

    def setUp(self):
        torch.manual_seed(1)
        self.B, self.T, self.D, self.H = 2, 5, 8, 2
        self.model = _MockConstrainedModel(
            d_model=self.D, n_heads=self.H, n_layers=1,
            d_ff=16, vocab_size=16, seq_len=self.T)
        self.inputs = torch.randint(0, 16, (self.B, self.T))

    def test_head_sum_equals_attn_output(self):
        """head_out.sum(dim=1) + bias == full attention output."""
        model = self.model
        layer = model.layers[0]
        x = (model.token_emb(self.inputs)
             + model.pos_emb(model.pos_indices[:, :self.T].expand(self.B, -1))
             + model.type_emb(model.type_indices[:, :self.T].expand(self.B, -1)))

        normed = layer.norm1(x)
        attn_w, head_out = _split_head_outputs_constrained(layer, normed)

        # What the actual attention module returns
        true_out = layer.attn(normed)

        # Sum of per-head outputs + bias
        reconstructed = head_out.sum(1)   # (B, T, D)
        if layer.attn.out_proj.bias is not None:
            reconstructed = reconstructed + layer.attn.out_proj.bias

        self.assertTrue(
            torch.allclose(reconstructed, true_out, atol=1e-5),
            f'Max diff: {(reconstructed - true_out).abs().max():.2e}')

    def test_attn_weights_sum_to_one(self):
        """Attention weights must sum to 1 over the key dimension."""
        acts = get_all_activations(self.model, self.inputs)
        for i in range(len(self.model.layers)):
            w = acts[f'attn_weights_{i}']
            row_sums = w.sum(-1)   # (B, H, T)
            self.assertTrue(torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5))


class TestFFNPreActivation(unittest.TestCase):
    """ffn_pre must match manual W1 @ LN(residual) + b1."""

    def setUp(self):
        torch.manual_seed(2)
        self.B, self.T, self.D = 2, 5, 8
        self.model = _MockConstrainedModel(
            d_model=self.D, n_heads=2, n_layers=1,
            d_ff=16, vocab_size=16, seq_len=self.T)
        self.inputs = torch.randint(0, 16, (self.B, self.T))

    def test_ffn_pre_matches_manual(self):
        acts = get_all_activations(self.model, self.inputs)
        layer = self.model.layers[0]

        residual_post_attn = acts['residual_post_attn_0']
        normed = layer.norm2(residual_post_attn)
        expected_pre = layer.ffn.w1(normed)

        self.assertTrue(
            torch.allclose(acts['ffn_pre_0'], expected_pre, atol=1e-5),
            f'Max diff: {(acts["ffn_pre_0"] - expected_pre).abs().max():.2e}')

    def test_ffn_post_is_relu_of_pre(self):
        acts = get_all_activations(self.model, self.inputs)
        expected_post = F.relu(acts['ffn_pre_0'])
        self.assertTrue(
            torch.allclose(acts['ffn_post_0'], expected_post, atol=1e-5))


class TestResidualConsistency(unittest.TestCase):
    """Final residual stream must reproduce model forward() logits."""

    def setUp(self):
        torch.manual_seed(3)
        self.B, self.T = 2, 5
        self.model = _MockConstrainedModel(
            d_model=8, n_heads=2, n_layers=2,
            d_ff=16, vocab_size=16, seq_len=self.T)
        self.inputs = torch.randint(0, 16, (self.B, self.T))

    def test_logits_match_model_forward(self):
        with torch.no_grad():
            expected = self.model(self.inputs)
        acts = get_all_activations(self.model, self.inputs)
        self.assertTrue(
            torch.allclose(acts['logits'], expected, atol=1e-5),
            f'Max diff: {(acts["logits"] - expected).abs().max():.2e}')


class TestOVWeights(unittest.TestCase):
    """W_OV shape and simple value check."""

    def setUp(self):
        torch.manual_seed(4)
        self.D, self.H = 8, 2
        self.model = _MockConstrainedModel(
            d_model=self.D, n_heads=self.H, n_layers=2,
            d_ff=16, vocab_size=16, seq_len=5)

    def test_ov_shape(self):
        for i in range(2):
            W_OV = get_head_ov_weights(self.model, i)
            self.assertEqual(W_OV.shape, (self.H, self.D, self.D))

    def test_ov_is_composition_of_WV_WO(self):
        """W_OV_h should equal W_V_h @ W_O_h (up to numerical precision)."""
        layer = self.model.layers[0]
        attn = layer.attn
        D, H, d = self.D, self.H, self.D // self.H

        W_OV = get_head_ov_weights(self.model, 0)   # (H, D, D)

        for h in range(H):
            W_V_h = attn.qkv.weight[2 * D + h * d: 2 * D + (h + 1) * d, :]  # (d, D)
            W_O_h = attn.out_proj.weight[:, h * d: (h + 1) * d]              # (D, d)
            expected = W_V_h.T @ W_O_h.T   # (D, D)
            self.assertTrue(
                torch.allclose(W_OV[h], expected, atol=1e-5),
                f'Head {h}: max diff {(W_OV[h] - expected).abs().max():.2e}')


# ─── Integration test with the real hardcoded model ──────────────────────────

class TestHardcodedModelExtraction(unittest.TestCase):
    """
    Smoke test: get_all_activations works on the round1 HandCodedSUBLEQ model
    and returns the correct shapes.
    """

    @classmethod
    def setUpClass(cls):
        try:
            from round1_constructed.model import HandCodedSUBLEQ, SEQ_LEN, VOCAB_SIZE
            cls.model = HandCodedSUBLEQ()
            cls.model.eval()
            cls.SEQ_LEN = SEQ_LEN
            cls.VOCAB_SIZE = VOCAB_SIZE
            cls.available = True
        except Exception as e:
            cls.available = False
            cls.skip_reason = str(e)

    def setUp(self):
        if not self.available:
            self.skipTest(f'HandCodedSUBLEQ not available: {self.skip_reason}')

    def test_shapes(self):
        B, T = 2, self.SEQ_LEN
        inputs = torch.randint(0, self.VOCAB_SIZE, (B, T))
        acts = get_all_activations(self.model, inputs)
        self.assertEqual(acts['embedding'].shape[0], B)
        self.assertEqual(acts['embedding'].shape[1], T)
        for i in range(4):
            self.assertIn(f'attn_weights_{i}', acts)
            self.assertEqual(acts[f'attn_weights_{i}'].shape[:2], (B, 8))

    def test_attn_weights_valid(self):
        B, T = 1, self.SEQ_LEN
        inputs = torch.randint(0, self.VOCAB_SIZE, (B, T))
        acts = get_all_activations(self.model, inputs)
        for i in range(4):
            w = acts[f'attn_weights_{i}']
            self.assertTrue(torch.allclose(w.sum(-1),
                                           torch.ones_like(w.sum(-1)), atol=1e-4))


if __name__ == '__main__':
    unittest.main()
