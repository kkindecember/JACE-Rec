import unittest
import numpy as np
import torch
from jace_rec.experts import score_batch, unit_rows, raw_text


class ExpertTests(unittest.TestCase):
    def test_full_visible_history_retains_repeated_candidates(self):
        embeddings = np.eye(3, dtype=np.float32)
        histories = [[1, 2, 1], [2, 3]]
        actual = raw_text(histories, embeddings)
        expected = []
        for history in histories:
            weights = np.asarray([np.exp(-.1*(len(history)-1-j)) for j in range(len(history))], dtype=np.float32)
            pooled = (embeddings[np.asarray(history)-1]*weights[:, None]).sum(0)/weights.sum()
            pooled /= np.linalg.norm(pooled)+1e-8
            scores = pooled @ unit_rows(embeddings).T
            expected.append((scores-scores.min())/(np.ptp(scores)+1e-8))
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-7)
        self.assertEqual(actual[0, 0], 1.)

    def test_target_does_not_change_expert_inputs_or_scores(self):
        class SequenceExpert:
            def predict(self, seq):
                return torch.tensor([[0., 1., 3., 2.]])
        sample = dict(full_history=[1, 2], target=3)
        original = score_batch(SequenceExpert(), np.eye(3, dtype=np.float32), [sample], 'cpu')
        sample['target'] = 1
        np.testing.assert_array_equal(original,
            score_batch(SequenceExpert(), np.eye(3, dtype=np.float32), [sample], 'cpu'))
