import math
import unittest
import numpy as np
import torch
from jace_rec.adapters.actionpiece import NativeGuide
from jace_rec.adapters.apao import GroupedIndex, Guide
from jace_rec import gating
from jace_rec.tree import ExpertTrie, conditionals


class AdapterTests(unittest.TestCase):
    def test_actionpiece_preserves_full_mass_and_noncanonical_scores(self):
        index = ExpertTrie([[0, 2, 15], [0, 3, 4, 15], [0, 3, 5, 15]], [1, 2, 3], eos=15)
        scores = torch.randn(4, 16, dtype=torch.float64).log_softmax(-1)
        prefixes = torch.tensor([[0, 3], [0, 3], [0, 15], [0, 8]])
        q = conditionals(index, np.array([.2, .6, .8]))
        result = NativeGuide(index, [q], [.4], ensemble=1, beams=4)(prefixes, scores)
        torch.testing.assert_close(result.exp().sum(-1), torch.ones(4, dtype=torch.float64), atol=1e-14, rtol=0)
        others = [i for i in range(16) if i not in (4, 5)]
        torch.testing.assert_close(result[:2, others], scores[:2, others], atol=0, rtol=0)
        torch.testing.assert_close(result[2:], scores[2:], atol=0, rtol=0)
        native = scores[:2, [4, 5]].exp()
        expected = .6*native + .4*native.sum(-1, keepdim=True)*torch.tensor(
            [[math.exp(q[(0, 3)][4]), math.exp(q[(0, 3)][5])]], dtype=torch.float64)
        torch.testing.assert_close(result[:2, [4, 5]].exp(), expected)

    def test_actionpiece_alpha_zero_and_mixed_user_grouping(self):
        index = ExpertTrie([[0, 2, 15], [0, 3, 4, 15], [0, 3, 5, 15]], [1, 2, 3], eos=15)
        scores = torch.randn(12, 16).log_softmax(-1)
        prefixes = torch.tensor([[0, 3]]*12)
        q = conditionals(index, np.array([.2, .6, .8]))
        result = NativeGuide(index, [None, q], [0., .4], ensemble=2, beams=3)(prefixes, scores)
        torch.testing.assert_close(result[:6], scores[:6], atol=0, rtol=0)
        separate = NativeGuide(index, [q], [.4], ensemble=2, beams=3)(prefixes[6:], scores[6:])
        torch.testing.assert_close(result[6:], separate, atol=0, rtol=0)

    def test_apao_shared_sid_aggregates_items_and_selects_source_id(self):
        paths = [[0, 2, 4, 6, 8], [0, 2, 4, 6, 8], [0, 2, 5, 7, 9], [0, 3, 5, 7, 9]]
        index = GroupedIndex(paths, [9, 4, 1, 2], 12)
        q = np.array([.1, .2, .3, .4])
        mass = index.log_masses(np.log(q))
        self.assertEqual(index.representative[tuple(paths[0])], 2)
        self.assertAlmostEqual(np.exp(mass[index.lookup[tuple(paths[0])]]), .3)
        logs = gating.path_logs(torch.tensor(np.log(q)[None]), torch.tensor([0]), index.ancestors)
        self.assertAlmostEqual(float(logs.sum()), math.log(.3))
        extreme = index.log_masses(np.array([-1200., -1000., -10., -.1]))
        self.assertTrue(np.isfinite(extreme).all())

    def test_apao_preserves_legal_mass_and_alpha_zero(self):
        index = GroupedIndex([[0, 2, 4, 6, 8], [0, 2, 5, 7, 9], [0, 3, 5, 7, 9]], [1, 2, 3], 12)
        masses = torch.tensor(np.stack([index.log_masses(np.log([.1, .2, .7]))]*2), dtype=torch.float64)
        scores = torch.randn(4, 12, dtype=torch.float64).log_softmax(-1)
        ids = torch.tensor([[8, 8, 8, 8, 2], [9, 9, 9, 9, 3], [7, 7, 7, 7, 2], [6, 6, 6, 6, 3]])
        result = Guide(index.transitions, 4, 2, masses, torch.tensor([0., .7], dtype=torch.float64))(ids, scores)
        for row in range(4):
            allowed = [4, 5] if ids[row, -1] == 2 else [5]
            self.assertAlmostEqual(float(result[row].exp().sum()), float(scores[row, allowed].exp().sum()))
            if row < 2:
                torch.testing.assert_close(result[row, allowed], scores[row, allowed], atol=0, rtol=0)
        self.assertFalse(torch.isnan(result).any())
