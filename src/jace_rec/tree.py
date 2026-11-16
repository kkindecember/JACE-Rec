"""Full-catalog lexical trie and differentiable target-path aggregation."""
import math
import numpy as np
import torch

class PrefixIndex:
    def __init__(self, paths, item_ids, eos=1):
        self.eos = eos
        self.nodes, self.lookup, self.children = [], {}, []
        node_items, item_refs = [], []
        self.path_to_item = {}
        for path, item in zip(paths, item_ids):
            path = tuple(map(int, path))
            assert path[0] == 0 and path[-1] == eos and eos not in path[:-1]
            assert path not in self.path_to_item, 'Ambiguous lexical item path'
            self.path_to_item[path] = int(item)
            parent = None
            for length in range(1, len(path) + 1):
                prefix = path[:length]
                if prefix not in self.lookup:
                    self.lookup[prefix] = len(self.nodes)
                    self.nodes.append(prefix)
                    self.children.append({})
                node = self.lookup[prefix]
                if parent is not None:
                    self.children[parent][path[length - 1]] = node
                node_items.append(node)
                item_refs.append(int(item))
                parent = node
        self.node_items = np.asarray(node_items)
        self.item_refs = np.asarray(item_refs)
        self.lengths = np.asarray([len(p) - int(p[-1] == eos) for p in self.nodes])
        self.root = self.lookup[(0,)]

    def potentials(self, item_rewards):
        best = np.full(len(self.nodes), -np.inf)
        np.maximum.at(best, self.node_items, np.asarray(item_rewards)[self.item_refs])
        assert np.isfinite(best).all()
        potential = self.lengths * best
        potential[self.root] = 0.
        return potential


class ExpertTrie(PrefixIndex):
    def conditional_logs(self, scores, temperature_scale=20.):
        scores = np.asarray(scores, dtype=np.float64)
        assert scores.ndim == 1 and np.isfinite(scores).all()
        weights = np.exp(temperature_scale * (scores - scores.max()))
        masses = np.bincount(self.node_items, weights=weights[self.item_refs-1],
                             minlength=len(self.nodes))
        assert np.all(masses > 0)
        logs = {}
        for node, children in enumerate(self.children):
            if children:
                values = {token: float(np.log(masses[child]/masses[node]))
                          for token, child in children.items()}
                assert abs(sum(math.exp(v) for v in values.values()) - 1.) < 1e-10
                logs[self.nodes[node]] = values
        return logs


class PathMass:
    def __init__(self, paths, device):
        self.index = ExpertTrie(paths, range(1, len(paths)+1))
        depth = max(map(len, paths)) - 1
        ancestors = np.empty((depth+1, len(paths)), dtype=np.int64)
        valid = np.zeros((len(paths), depth), dtype=bool)
        for item, path in enumerate(paths):
            for d in range(depth+1):
                ancestors[d, item] = self.index.lookup[tuple(path[:min(d+1, len(path))])]
            valid[item, :len(path)-1] = True
        self.ancestors = torch.from_numpy(ancestors).to(device)
        self.valid = torch.from_numpy(valid).to(device)

    def logs(self, scores, targets, scale=20.):
        z = scale*scores
        previous = torch.logsumexp(z, -1)
        logs = []
        for d in range(1, len(self.ancestors)):
            nodes = self.ancestors[d]
            mask = nodes[None, :] == nodes[targets, None]
            current = torch.logsumexp(z.masked_fill(~mask, -torch.inf), -1)
            logs.append(current-previous)
            previous = current
        return torch.stack(logs, -1)


class LazyExpertLogs:
    """Use the original float64 masses, calculating only visited conditionals."""
    def __init__(self, index, scores, temperature_scale=20.):
        scores = np.asarray(scores, dtype=np.float64)
        assert scores.ndim == 1 and np.isfinite(scores).all()
        weights = np.exp(temperature_scale * (scores - scores.max()))
        self.masses = np.bincount(index.node_items, weights=weights[index.item_refs-1],
                                  minlength=len(index.nodes))
        assert np.all(self.masses > 0)
        self.index, self.cache = index, {}

    def __getitem__(self, prefix):
        if prefix not in self.cache:
            node = self.index.lookup[prefix]
            self.cache[prefix] = values = {
                token: float(np.log(self.masses[child] / self.masses[node]))
                for token, child in self.index.children[node].items()}
            assert abs(sum(math.exp(v) for v in values.values()) - 1.) < 1e-10
        return self.cache[prefix]

def conditionals(index, scores):
    from .gating import conditionals as stable_conditionals
    scores=np.asarray(scores,np.float64)
    return LazyExpertLogs(index,scores) if float(np.ptp(scores))*20<700 else stable_conditionals(index,scores)
