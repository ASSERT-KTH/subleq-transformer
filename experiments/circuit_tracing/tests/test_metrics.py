"""
Tests for metrics.py.

Verifies:
  1. fetch_head_score = 1.0 for a perfectly concentrated head.
  2. fetch_head_score ≈ 1/T for a uniform head.
  3. broadcast_head_score = 1.0 for a perfectly broadcast head.
  4. broadcast_head_score ≈ 1/T for a uniform head.
  5. neuron_concept_correlation = ±1 for a neuron that exactly tracks a concept.
  6. neuron_concept_correlation ≈ 0 for an independent neuron.
  7. summarise_head_roles correctly labels fetch/broadcast/other heads.
  8. Integration test: hardcoded model fetch heads score high; other heads score low.
"""

import sys
import os
import unittest
import torch
import numpy as np

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

from circuit_tracing.metrics import (
    fetch_head_score,
    broadcast_head_score,
    neuron_concept_correlation,
    summarise_head_roles,
)


def _make_uniform_attn(N, H, T):
    """Attention weights uniformly distributed over keys."""
    return torch.ones(N, H, T, T) / T


def _make_perfect_fetch_attn(N, H, T, target_head, target_positions):
    """
    Returns attention weights where head `target_head` always attends
    exactly to target_positions[n] from query 0.
    All other heads are uniform.
    """
    attn = torch.ones(N, H, T, T) / T
    attn[:, target_head, :, :] = 1.0 / T   # default uniform
    for n in range(N):
        tp = target_positions[n].item()
        attn[n, target_head, 0, :] = 0.0
        attn[n, target_head, 0, tp] = 1.0
    return attn


def _make_perfect_broadcast_attn(N, H, T, broadcast_head, src_pos=0):
    """
    Returns attention weights where head `broadcast_head` has every query
    attending 100% to src_pos.  All other heads are uniform.
    """
    attn = torch.ones(N, H, T, T) / T
    attn[:, broadcast_head, :, :] = 0.0
    attn[:, broadcast_head, :, src_pos] = 1.0
    return attn


# ─── fetch_head_score ─────────────────────────────────────────────────────────

class TestFetchHeadScore(unittest.TestCase):

    def test_perfect_fetch_head_scores_one(self):
        N, H, T = 50, 4, 10
        target_h = 2
        target_pos = torch.randint(1, T, (N,))
        attn = _make_perfect_fetch_attn(N, H, T, target_h, target_pos)

        scores, baseline = fetch_head_score(attn, target_pos, query_pos=0)
        self.assertAlmostEqual(float(scores[target_h]), 1.0, places=5,
                               msg='Perfect fetch head should score 1.0')

    def test_uniform_head_scores_near_baseline(self):
        N, H, T = 200, 4, 10
        target_pos = torch.randint(0, T, (N,))
        attn = _make_uniform_attn(N, H, T)

        scores, baseline = fetch_head_score(attn, target_pos, query_pos=0)
        for h in range(H):
            self.assertAlmostEqual(float(scores[h]), baseline, places=4,
                                   msg=f'Uniform head {h} should score ≈ 1/T')

    def test_non_fetch_heads_unaffected(self):
        """When only head 2 is a fetch head, other heads should still score low."""
        N, H, T = 100, 4, 12
        target_h = 2
        target_pos = torch.randint(1, T, (N,))
        attn = _make_perfect_fetch_attn(N, H, T, target_h, target_pos)

        scores, baseline = fetch_head_score(attn, target_pos, query_pos=0)
        for h in range(H):
            if h != target_h:
                self.assertAlmostEqual(float(scores[h]), baseline, places=4,
                                       msg=f'Non-fetch head {h} should score ≈ baseline')

    def test_returns_H_scores(self):
        N, H, T = 10, 6, 8
        attn = _make_uniform_attn(N, H, T)
        target_pos = torch.zeros(N, dtype=torch.long)
        scores, _ = fetch_head_score(attn, target_pos)
        self.assertEqual(len(scores), H)

    def test_fetch_score_is_nonnegative(self):
        N, H, T = 20, 3, 10
        attn = torch.rand(N, H, T, T)
        attn = attn / attn.sum(-1, keepdim=True)
        target_pos = torch.randint(0, T, (N,))
        scores, _ = fetch_head_score(attn, target_pos)
        self.assertTrue(all(s >= 0 for s in scores))


# ─── broadcast_head_score ─────────────────────────────────────────────────────

class TestBroadcastHeadScore(unittest.TestCase):

    def test_perfect_broadcast_scores_one(self):
        N, H, T = 50, 4, 10
        bc_h = 1
        attn = _make_perfect_broadcast_attn(N, H, T, bc_h, src_pos=0)

        scores, baseline = broadcast_head_score(attn, src_pos=0, exclude_self=True)
        self.assertAlmostEqual(float(scores[bc_h]), 1.0, places=5,
                               msg='Perfect broadcast head should score 1.0')

    def test_uniform_head_scores_near_baseline(self):
        N, H, T = 200, 4, 10
        attn = _make_uniform_attn(N, H, T)

        scores, baseline = broadcast_head_score(attn, src_pos=0)
        for h in range(H):
            self.assertAlmostEqual(float(scores[h]), baseline, places=4)

    def test_non_broadcast_heads_unaffected(self):
        N, H, T = 100, 4, 10
        bc_h = 3
        attn = _make_perfect_broadcast_attn(N, H, T, bc_h)
        scores, baseline = broadcast_head_score(attn)
        for h in range(H):
            if h != bc_h:
                self.assertAlmostEqual(float(scores[h]), baseline, places=4)

    def test_exclude_self_vs_include_self(self):
        """With exclude_self, score should not count the src_pos→src_pos weight."""
        N, H, T = 20, 2, 8
        attn = _make_perfect_broadcast_attn(N, H, T, broadcast_head=0, src_pos=0)
        s_excl, _ = broadcast_head_score(attn, src_pos=0, exclude_self=True)
        s_incl, _ = broadcast_head_score(attn, src_pos=0, exclude_self=False)
        # Both should be 1.0 for a perfect broadcast head (self-attn is also 1)
        self.assertAlmostEqual(float(s_excl[0]), 1.0, places=5)
        self.assertAlmostEqual(float(s_incl[0]), 1.0, places=5)


# ─── neuron_concept_correlation ──────────────────────────────────────────────

class TestNeuronConceptCorrelation(unittest.TestCase):

    def test_perfect_positive_correlation(self):
        """A neuron whose activation equals the concept should have r = 1."""
        N, T, d_ff = 100, 5, 32
        concept = torch.randn(N)
        ffn_pre = torch.zeros(N, T, d_ff)
        ffn_pre[:, 0, 7] = concept   # neuron 7 at position 0

        corr = neuron_concept_correlation(ffn_pre, concept, token_pos=0)
        self.assertAlmostEqual(float(corr[7]), 1.0, places=4)

    def test_perfect_negative_correlation(self):
        N, T, d_ff = 100, 5, 32
        concept = torch.randn(N)
        ffn_pre = torch.zeros(N, T, d_ff)
        ffn_pre[:, 0, 3] = -concept

        corr = neuron_concept_correlation(ffn_pre, concept, token_pos=0)
        self.assertAlmostEqual(float(corr[3]), -1.0, places=4)

    def test_independent_neuron_near_zero(self):
        """Neuron uncorrelated with concept should have |r| ≈ 0."""
        torch.manual_seed(42)
        N, T, d_ff = 500, 5, 16
        concept = torch.randn(N)
        ffn_pre = torch.randn(N, T, d_ff)   # independent of concept

        corr = neuron_concept_correlation(ffn_pre, concept, token_pos=0)
        mean_abs = float(np.abs(corr).mean())
        # With N=500, expected |r| ≈ 1/sqrt(N) ≈ 0.045; use generous threshold
        self.assertLess(mean_abs, 0.2,
                        msg=f'Independent neurons should have low correlation, got {mean_abs:.3f}')

    def test_output_shape(self):
        N, T, d_ff = 50, 5, 24
        ffn_pre = torch.randn(N, T, d_ff)
        concept = torch.randn(N)
        corr = neuron_concept_correlation(ffn_pre, concept, token_pos=0)
        self.assertEqual(corr.shape, (d_ff,))

    def test_token_pos_argument(self):
        """Correlation at token_pos=2 should use activations at position 2."""
        N, T, d_ff = 100, 5, 16
        concept = torch.randn(N)
        ffn_pre = torch.zeros(N, T, d_ff)
        ffn_pre[:, 2, 5] = concept        # only at position 2, neuron 5

        corr_pos0 = neuron_concept_correlation(ffn_pre, concept, token_pos=0)
        corr_pos2 = neuron_concept_correlation(ffn_pre, concept, token_pos=2)

        self.assertLess(abs(float(corr_pos0[5])), 0.1,
                        'Correlation at wrong position should be low')
        self.assertAlmostEqual(float(corr_pos2[5]), 1.0, places=4)


# ─── summarise_head_roles ─────────────────────────────────────────────────────

class TestSummariseHeadRoles(unittest.TestCase):

    def test_fetch_label(self):
        L, H = 2, 4
        rand_bl = 0.1
        fetch = np.zeros((L, H))
        bcast = np.zeros((L, H))
        fetch[0, 2] = 0.8   # well above 5× baseline
        roles = summarise_head_roles(fetch, bcast, rand_bl,
                                     fetch_threshold=5.0, broadcast_threshold=5.0)
        self.assertEqual(roles[0][2], 'fetch')

    def test_broadcast_label(self):
        L, H = 2, 4
        rand_bl = 0.1
        fetch = np.zeros((L, H))
        bcast = np.zeros((L, H))
        bcast[1, 0] = 0.9
        roles = summarise_head_roles(fetch, bcast, rand_bl)
        self.assertEqual(roles[1][0], 'broadcast')

    def test_mixed_label(self):
        L, H = 1, 2
        rand_bl = 0.1
        fetch = np.array([[0.8, 0.0]])
        bcast = np.array([[0.8, 0.0]])
        roles = summarise_head_roles(fetch, bcast, rand_bl)
        self.assertEqual(roles[0][0], 'mixed')

    def test_other_label(self):
        L, H = 1, 3
        rand_bl = 0.1
        fetch = np.zeros((L, H))
        bcast = np.zeros((L, H))
        roles = summarise_head_roles(fetch, bcast, rand_bl)
        for h in range(H):
            self.assertEqual(roles[0][h], 'other')


# ─── Integration: hardcoded model known circuits ──────────────────────────────

class TestHardcodedCircuits(unittest.TestCase):
    """
    Verify that the known circuits of HandCodedSUBLEQ are detected:
      - L1 heads 0/1/2 are fetch heads (a, b, c operands)
      - L1 head 3 is a broadcast head (PC → all)
      - L3 heads 0/1 are broadcast heads (b-addr and write-delta → all)
    """

    @classmethod
    def setUpClass(cls):
        try:
            from round1_constructed.model import HandCodedSUBLEQ, SEQ_LEN, VOCAB_SIZE
            from round1_constructed.interpreter import MEM_SIZE, VALUE_OFFSET
            import random

            cls.model = HandCodedSUBLEQ()
            cls.model.eval()
            cls.SEQ_LEN = SEQ_LEN
            cls.VOCAB_SIZE = VOCAB_SIZE
            cls.MEM_SIZE = MEM_SIZE
            cls.VALUE_OFFSET = VALUE_OFFSET
            cls.available = True

            # Generate test examples
            random.seed(0)
            N = 200
            inputs = []
            pc_vals = []
            a_addrs, b_addrs, c_addrs = [], [], []

            for _ in range(N * 10):
                if len(inputs) >= N:
                    break
                pc = random.randint(0, MEM_SIZE - 4)
                mem = [0] * MEM_SIZE
                a = random.randint(0, MEM_SIZE - 1)
                b = random.randint(0, MEM_SIZE - 1)
                c = random.randint(0, MEM_SIZE - 3)
                mem[pc] = a
                mem[pc + 1] = b
                mem[pc + 2] = c
                # random data values
                for k in range(MEM_SIZE):
                    if k not in {pc, pc+1, pc+2}:
                        mem[k] = random.randint(-32768, 32767)

                tokens = [pc + VALUE_OFFSET] + [v + VALUE_OFFSET for v in mem]
                inputs.append(torch.tensor(tokens, dtype=torch.long))
                pc_vals.append(pc)
                a_addrs.append(a)
                b_addrs.append(b)
                c_addrs.append(c)

            cls.inputs = torch.stack(inputs[:N])
            cls.pc_vals = torch.tensor(pc_vals[:N])
            cls.a_addrs = torch.tensor(a_addrs[:N])
            cls.b_addrs = torch.tensor(b_addrs[:N])
            cls.c_addrs = torch.tensor(c_addrs[:N])

        except Exception as e:
            cls.available = False
            cls.skip_reason = str(e)

    def setUp(self):
        if not self.available:
            self.skipTest(f'HandCodedSUBLEQ not available: {self.skip_reason}')

    def _get_attn_weights_layer(self, layer_idx):
        from circuit_tracing.extract_activations import get_all_activations
        acts = get_all_activations(self.model, self.inputs)
        return acts[f'attn_weights_{layer_idx}']   # (N, H, T, T)

    def test_layer1_fetch_heads_a_b_c(self):
        """
        L1 heads 0/1/2 should have fetch scores >> baseline.

        The hardcoded model's L1 fetch heads use Gaussian addressing with
        addr_dim=DV and offsets 1/2/3.  At query position 0, DV = pc, so
        each head attends to key position pc+offset — NOT to a+1/b+1/c+1.
        (Those are L2 fetch targets, which read mem[a] and mem[b].)
        """
        attn_w = self._get_attn_weights_layer(0)
        T = self.SEQ_LEN
        baseline = 1.0 / T

        # L1H0: target = pc+1 (reads mem[pc] = a_addr)
        target_0 = self.pc_vals + 1
        scores_0, _ = fetch_head_score(attn_w, target_0, query_pos=0)

        # L1H1: target = pc+2 (reads mem[pc+1] = b_addr)
        target_1 = self.pc_vals + 2
        scores_1, _ = fetch_head_score(attn_w, target_1, query_pos=0)

        # L1H2: target = pc+3 (reads mem[pc+2] = c_addr)
        target_2 = self.pc_vals + 3
        scores_2, _ = fetch_head_score(attn_w, target_2, query_pos=0)

        threshold = 10 * baseline
        self.assertGreater(float(scores_0[0]), threshold,
                           f'L1H0 score {scores_0[0]:.3f} <= {threshold:.3f}')
        self.assertGreater(float(scores_1[1]), threshold,
                           f'L1H1 score {scores_1[1]:.3f} <= {threshold:.3f}')
        self.assertGreater(float(scores_2[2]), threshold,
                           f'L1H2 score {scores_2[2]:.3f} <= {threshold:.3f}')

    def test_layer1_broadcast_head(self):
        """L1 head 3 is a broadcast head (all positions attend to pos 0)."""
        attn_w = self._get_attn_weights_layer(0)
        T = self.SEQ_LEN
        baseline = 1.0 / T

        scores, _ = broadcast_head_score(attn_w, src_pos=0, exclude_self=True)
        threshold = 10 * baseline
        self.assertGreater(float(scores[3]), threshold,
                           f'L1H3 broadcast score {scores[3]:.3f} <= {threshold:.3f}')

    def test_layer3_broadcast_heads(self):
        """L3 heads 0 and 1 should be broadcast heads."""
        attn_w = self._get_attn_weights_layer(2)
        T = self.SEQ_LEN
        baseline = 1.0 / T

        scores, _ = broadcast_head_score(attn_w, src_pos=0, exclude_self=True)
        threshold = 10 * baseline
        self.assertGreater(float(scores[0]), threshold,
                           f'L3H0 broadcast score {scores[0]:.3f} <= {threshold:.3f}')
        self.assertGreater(float(scores[1]), threshold,
                           f'L3H1 broadcast score {scores[1]:.3f} <= {threshold:.3f}')

    def test_layer4_attn_inactive(self):
        """L4 attention is all zeros in the hardcoded model — all heads uniform."""
        attn_w = self._get_attn_weights_layer(3)
        T = self.SEQ_LEN
        baseline = 1.0 / T

        target_pos = torch.zeros(self.inputs.shape[0], dtype=torch.long)
        scores_f, _ = fetch_head_score(attn_w, target_pos)
        scores_b, _ = broadcast_head_score(attn_w)

        # All scores should be near baseline (uniform attention after zeroed QK)
        for h in range(scores_f.shape[0]):
            self.assertAlmostEqual(float(scores_f[h]), baseline, delta=0.05,
                                   msg=f'L4H{h} fetch score should be near baseline')


if __name__ == '__main__':
    unittest.main()
