"""APAO four-token guidance and collision-aware catalog aggregation."""
from collections import defaultdict
import numpy as np
import torch

class GroupedIndex:
    """Catalog prefix sums with the author's original, occasionally shared SIDs."""
    def __init__(self, paths, author_ids, vocabulary):
        self.paths = [tuple(p) for p in paths]
        assert all(len(p) == 5 and p[0] == 0 for p in self.paths)
        self.nodes, self.lookup, self.children = [(0,)], {(0,): 0}, [{}]
        self.path_to_items = defaultdict(list)
        ancestors = np.empty((5, len(paths)), np.int64)
        for item, path in enumerate(self.paths, 1):
            self.path_to_items[path].append(item)
            ancestors[0, item-1] = 0
            for depth in range(1, 5):
                prefix = path[:depth+1]
                parent = self.lookup[path[:depth]]
                if prefix not in self.lookup:
                    self.lookup[prefix] = len(self.nodes)
                    self.nodes.append(prefix)
                    self.children.append({})
                node = self.lookup[prefix]
                self.children[parent][path[depth]] = node
                ancestors[depth, item-1] = node
        self.ancestors = torch.from_numpy(ancestors)
        self.node_items = ancestors.T.reshape(-1)
        self.item_refs = np.repeat(np.arange(len(paths)), 5)
        self.representative = {p: min(items, key=lambda i: author_ids[i-1])
                               for p, items in self.path_to_items.items()}
        transitions = np.full((len(self.nodes), vocabulary), -1, np.int64)
        for parent, children in enumerate(self.children):
            for token, child in children.items():
                transitions[parent, token] = child
        self.transitions = torch.from_numpy(transitions)

    def log_masses(self, log_q):
        """Sum all catalog items, including every member of a shared SID."""
        log_q = np.asarray(log_q, np.float64)
        assert log_q.shape == (len(self.paths),) and np.isfinite(log_q).all()
        if np.ptp(log_q) < 700:
            weights = np.exp(log_q-log_q.max())
            masses = np.bincount(self.node_items, weights=weights[self.item_refs],
                                 minlength=len(self.nodes))
            assert (masses > 0).all()
            return np.log(masses) + log_q.max()
        values = np.full(len(self.nodes), -np.inf)
        np.logaddexp.at(values, self.node_items, log_q[self.item_refs])
        assert np.isfinite(values).all()
        return values


class Guide:
    """Vectorized legal masking and mass-preserving mixture in HF beam search."""
    def __init__(self, transitions, history_width, beams, log_masses=None, alphas=None):
        self.transitions, self.width, self.beams = transitions, history_width, beams
        self.masses, self.alphas = log_masses, alphas

    def __call__(self, input_ids, scores):
        nodes = torch.zeros(len(input_ids), dtype=torch.long, device=scores.device)
        for depth in range(self.width, input_ids.shape[1]):
            nodes = self.transitions[nodes, input_ids[:, depth]]
        children = self.transitions[nodes]
        legal = children >= 0
        masked = scores.masked_fill(~legal, -torch.inf)
        if self.masses is None:
            return masked
        users = torch.arange(len(input_ids), device=scores.device)//self.beams
        q = self.masses[users[:, None], children.clamp_min(0)]-self.masses[users, nodes, None]
        a = self.alphas[users, None]
        legal_mass = masked.logsumexp(-1, keepdim=True)
        mixed = torch.logaddexp(masked+torch.log1p(-a), legal_mass+q+a.log())
        return torch.where(a == 0, masked, mixed).masked_fill(~legal, -torch.inf)


def history_batch(tokenizer, codes, samples, device):
    texts = [''.join(codes[i] for i in s['full_history'][-20:]) for s in samples]
    batch = tokenizer(texts, padding='longest', add_special_tokens=False,
                      return_attention_mask=True, return_tensors='pt')
    assert batch['input_ids'].shape[1] % 4 == 0
    return {k: batch[k].to(device) for k in ('input_ids', 'attention_mask')}


@torch.no_grad()
def decode(model, tokenizer, batch, transitions, masses=None, alphas=None, native_trie=None):
    from transformers import LogitsProcessorList
    options = dict(max_new_tokens=4, num_beams=20, num_return_sequences=20,
                   output_scores=True, return_dict_in_generate=True,
                   early_stopping=True, pad_token_id=tokenizer.eos_token_id)
    if native_trie is not None:
        from utils.utils import prefix_allowed_tokens_fn_for_decoderonly
        options['prefix_allowed_tokens_fn'] = prefix_allowed_tokens_fn_for_decoderonly(native_trie)
    else:
        options['logits_processor'] = LogitsProcessorList([
            Guide(transitions, batch['input_ids'].shape[1], 20, masses, alphas)])
    output = model.generate(**batch, **options)
    tokens = output.sequences[:, -4:].cpu().reshape(-1, 20, 4).numpy()
    scores = output.sequences_scores.cpu().reshape(-1, 20).numpy()
    del output
    return tokens, scores


@torch.no_grad()
def teacher_logits(model, batch, targets):
    width = batch['input_ids'].shape[1]
    ids = torch.cat((batch['input_ids'], targets[:, :3]), -1)
    mask = torch.cat((batch['attention_mask'], torch.ones_like(targets[:, :3])), -1)
    positions = mask.long().cumsum(-1)-1
    positions.masked_fill_(mask == 0, 1)
    output = model(input_ids=ids, attention_mask=mask, position_ids=positions)
    logs = output.logits[:, width-1:width+3].float().log_softmax(-1)
    del output
    return logs
