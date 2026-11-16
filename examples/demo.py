"""CPU example of the JACE distribution and differentiable path objective."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))

import numpy as np
import torch
from jace_rec import gating
from jace_rec.tree import PathMass, conditionals


def main():
    torch.set_num_threads(2)
    experts = np.array([[.7, .8], [.4, .6], [.9, .2], [.2, .7]])
    history = [1, 3, 1]
    paths = [[0, 2, 4, 1], [0, 2, 5, 1], [0, 3, 1], [0, 3, 7, 8, 1]]
    features, seen = gating.features(experts, history)
    # Dataset runs standardize features using fit-user statistics.
    head = gating.Head('joint_user_alpha')
    scores, controls = gating.catalog_scores(head, features, experts, seen)
    probabilities = np.exp(20*scores)
    tree = PathMass(paths, 'cpu')
    branches = conditionals(tree.index, scores)
    recovered = [np.exp(sum(branches[tuple(p[:t])][p[t]] for t in range(1, len(p))))
                 for p in paths]
    np.testing.assert_allclose(recovered, probabilities, atol=1e-12)
    batch = dict(features=torch.tensor(features[None]), experts=torch.tensor(experts[None]),
                 seen=torch.tensor(seen[None]), targets=torch.tensor([1]), valid=tree.valid[[1]],
                 native=torch.full((1, 4), -3., dtype=torch.float64),
                 legal=torch.full((1, 4), -.4, dtype=torch.float64))
    loss = gating.loss(head, batch, tree.ancestors).mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
    print('Trainable parameters:', sum(p.numel() for p in head.parameters()))
    print('Controls:', controls)
    print('Catalog probabilities:', probabilities.tolist())
    print('Historical probability mass:', probabilities[seen].sum())
    print('Path loss:', float(loss))
    print('Prefix reconstruction and joint gradients: passed')


if __name__ == '__main__':
    main()
