"""Mass-preserving expert guidance with the original beam score convention."""
import math
from contextlib import contextmanager
import numpy as np
import torch
from transformers.generation.beam_search import BeamSearchScorer

def mixed_log_values(native, legal_log_mass, expert_log_prob, alpha):
    native = np.asarray(native, dtype=np.float64)
    if alpha == 0:
        return native.copy()
    return np.logaddexp(math.log1p(-alpha) + native,
                        math.log(alpha) + legal_log_mass + np.asarray(expert_log_prob))


class ProbabilityGuide:
    """Called AFTER native trie masking and BEFORE cumulative beam scores.

Let m be native total legal-token mass, and p be its conditional distribution.
Return log(m * ((1-alpha)*p + alpha*q)). Keeping m avoids silently changing
the baseline by renormalizing each parent's legal token probabilities.
The trace permits exact same-rule rescoring of native completed paths.
"""
    def __init__(self, index, expert_logs=None, alpha=0.):
        assert 0 <= alpha < 1
        self.index, self.expert_logs, self.alpha = index, expert_logs, float(alpha)
        self.trace, self.calls = {}, 0

    def __call__(self, input_ids, scores):
        self.calls += 1
        prefixes = [tuple(p) for p in input_ids.detach().cpu().tolist()]
        row_ids, tokens, qlogs, rows = [], [], [], []
        for row, prefix in enumerate(prefixes):
            node = self.index.lookup.get(prefix)
            if node is None:
                assert torch.isneginf(scores[row]).all(), 'Unexpected unmasked prefix'
                continue
            children = self.index.children[node]
            start = len(row_ids)
            for token in children:
                row_ids.append(row); tokens.append(token)
                if self.alpha:
                    qlogs.append(self.expert_logs[prefix][token])
            rows.append((row, prefix, start, len(row_ids)))
        if not row_ids:
            return scores
        rr = torch.tensor(row_ids, device=scores.device, dtype=torch.long)
        tt = torch.tensor(tokens, device=scores.device, dtype=torch.long)
        native = scores[rr, tt]
        assert torch.isfinite(native).all()
        masses = torch.logsumexp(scores, dim=-1)
        cpu_native = native.detach().cpu().double().numpy()
        cpu_masses = masses.detach().cpu().double().numpy()
        for row, prefix, start, stop in rows:
            if prefix not in self.trace:
                self.trace[prefix] = (tokens[start:stop], cpu_native[start:stop].copy(), float(cpu_masses[row]))
        if self.alpha:
            q = torch.tensor(qlogs, device=scores.device, dtype=scores.dtype)
            scores[rr, tt] = torch.logaddexp(native + math.log1p(-self.alpha),
                q + masses[rr] + math.log(self.alpha))
        return scores

    def rescore(self, paths, expert_logs=None, alpha=0.):
        values = []
        for path in paths:
            total = 0.
            for step in range(1, len(path)):
                prefix, token = tuple(path[:step]), int(path[step])
                tokens, native, mass = self.trace[prefix]
                position = tokens.index(token)
                q = 0. if alpha == 0 else expert_logs[prefix][token]
                total += float(mixed_log_values(native[position], mass, q, alpha))
            # Transformers 4.26 BeamHypotheses scores the sequence before EOS,
            # including its decoder start token, with length_penalty=1.
            values.append(total/(len(path)-1))
        return np.asarray(values)


class BulkCopyBeamScorer(BeamSearchScorer):
    """Run the unchanged HF hypothesis bookkeeping on bulk-copied CPU tensors."""
    def process(self, input_ids, next_scores, next_tokens, next_indices, **kwargs):
        assert kwargs.get('beam_indices') is None
        device = input_ids.device
        result = super().process(input_ids.cpu(), next_scores.cpu(), next_tokens.cpu(),
                                 next_indices.cpu(), **kwargs)
        return {key: value.to(device) for key, value in result.items()}

    def finalize(self, input_ids, final_beam_scores, final_beam_tokens,
                 final_beam_indices, **kwargs):
        assert kwargs.get('beam_indices') is None
        self.device = torch.device('cpu')
        result = super().finalize(input_ids.cpu(), final_beam_scores.cpu(),
            final_beam_tokens.cpu(), final_beam_indices.cpu(), **kwargs)
        self.final_sequence_scores = result['sequence_scores']
        return result


@contextmanager
def bulk_beam_bookkeeping(model):
    """Skip unused beam ancestry/step-score retention, retaining final scores."""
    original = model.beam_search
    def beam_search(input_ids, beam_scorer, **kwargs):
        assert type(beam_scorer) is BeamSearchScorer
        beam_scorer.__class__ = BulkCopyBeamScorer
        beam_scorer._done = beam_scorer._done.cpu()
        kwargs['output_scores'] = False
        result = original(input_ids, beam_scorer, **kwargs)
        result.sequences_scores = beam_scorer.final_sequence_scores
        return result
    model.beam_search = beam_search
    try:
        yield
    finally:
        model.beam_search = original


class BatchedTrieMask:
    """Copy all beam prefixes once and scatter the original legal-token mask."""
    def __init__(self, index):
        self.index = index

    def __call__(self, input_ids, scores):
        rows, tokens = [], []
        for row, prefix in enumerate(input_ids.detach().cpu().tolist()):
            node = self.index.lookup.get(tuple(prefix))
            if node is not None:
                children = self.index.children[node]
                rows.extend([row] * len(children))
                tokens.extend(children)
        mask = torch.full_like(scores, -math.inf)
        if rows:
            rr = torch.tensor(rows, dtype=torch.long, device=scores.device)
            tt = torch.tensor(tokens, dtype=torch.long, device=scores.device)
            mask[rr, tt] = 0
        return scores + mask
