"""JACE-Rec: five observable inputs and jointly learned beta, rho, alpha."""
import math

import numpy as np
import torch


ARMS = ('joint_user_alpha',)
FEATURES = ('log_history_length', 'confidence_sas', 'confidence_text',
            'history_mass_sas', 'history_mass_text')
INITIAL = dict(beta=.5, rho=.01, alpha=.3)


def features(experts, history):
    """Target-free statistics of the supplied full-catalog expert scores.

    Both columns are min-max normalized over the full item catalog.
    """
    e = np.asarray(experts, np.float64)
    assert e.ndim == 2 and e.shape[1] == 2 and np.isfinite(e).all()
    h = np.zeros(len(e), bool)
    ids = np.unique(np.asarray(history, np.int64)) - 1
    assert len(ids) and ids.min() >= 0 and ids.max() < len(e)
    h[ids] = True
    assert (~h).any()
    weight = np.exp(20 * (e - e.max(0)))
    x = np.r_[np.log1p(len(history))/5, e.max(0)-e.mean(0),
              weight[h].sum(0)/weight.sum(0)]
    assert x.shape == (5,) and np.isfinite(x).all()
    return x, h


def normalize(x, normalization):
    return (np.asarray(x, np.float64)-normalization['mean'])/normalization['std']


class Head(torch.nn.Module):
    def __init__(self, arm):
        super().__init__()
        assert arm in ARMS
        self.arm = arm
        self.beta_rho = torch.nn.Linear(5, 2, dtype=torch.float64)
        self.alpha = torch.nn.Linear(5, 1, dtype=torch.float64)
        with torch.no_grad():
            self.beta_rho.weight.zero_()
            self.beta_rho.bias.copy_(torch.tensor([math.log(INITIAL[k]/(1-INITIAL[k]))
                                                 for k in ('beta','rho')], dtype=torch.float64))
            self.alpha.weight.zero_()
            self.alpha.bias.fill_(math.log(.3/.7))

    def logits(self, x):
        br = self.beta_rho(x)
        a = self.alpha(x).squeeze(-1)
        return torch.cat((br, a[..., None]), -1)

    def forward(self, x):
        return self.logits(x).sigmoid()


def leaf_logs(logits, experts, seen):
    """Disjoint-group normalization keeps total historical leaf mass rho."""
    beta = logits[:, 0].sigmoid()
    z = 20*((1-beta[:, None])*experts[:, :, 0]+beta[:, None]*experts[:, :, 1])
    zn = torch.logsumexp(z.masked_fill(seen, -torch.inf), -1)
    zh = torch.logsumexp(z.masked_fill(~seen, -torch.inf), -1)
    assert torch.isfinite(zn).all() and torch.isfinite(zh).all()
    nr = torch.nn.functional.logsigmoid(-logits[:, 1])
    hr = torch.nn.functional.logsigmoid(logits[:, 1])
    return z-torch.where(seen, zh[:, None], zn[:, None])+torch.where(seen, hr[:, None], nr[:, None])


def path_logs(log_q, targets, ancestors):
    previous = torch.logsumexp(log_q, -1)
    values = []
    for nodes in ancestors[1:]:
        mask = nodes[None, :] == nodes[targets, None]
        current = torch.logsumexp(log_q.masked_fill(~mask, -torch.inf), -1)
        values.append(current-previous)
        previous = current
    return torch.stack(values, -1)


def loss(head, batch, ancestors):
    logits = head.logits(batch['features'])
    leaves = leaf_logs(logits, batch['experts'], batch['seen'])
    q = path_logs(leaves, batch['targets'], ancestors)
    la = torch.nn.functional.logsigmoid(logits[:, 2])[:, None]
    ln = torch.nn.functional.logsigmoid(-logits[:, 2])[:, None]
    mixed = torch.logaddexp(batch['native']+ln, batch['legal']+q+la)
    return -mixed.masked_fill(~batch['valid'], 0.).sum(-1)/batch['valid'].sum(-1)


@torch.no_grad()
def catalog_scores(head, x, experts, seen):
    logits = head.logits(torch.as_tensor(np.asarray(x)[None], dtype=torch.float64))
    log_q = leaf_logs(logits, torch.as_tensor(np.asarray(experts)[None], dtype=torch.float64),
                      torch.as_tensor(np.asarray(seen)[None], dtype=torch.bool))[0].numpy()
    params = dict(zip(('beta_text','root_rho','alpha'), logits[0].sigmoid().tolist()))
    # sigmoid is mathematically open; protect only rounded floating endpoints.
    params['alpha'] = float(np.clip(params['alpha'], np.nextafter(0.,1.), np.nextafter(1.,0.)))
    np.testing.assert_allclose(np.logaddexp.reduce(log_q), 0., atol=2e-12)
    np.testing.assert_allclose(np.exp(log_q[np.asarray(seen)]).sum(), params['root_rho'], atol=2e-12)
    assert np.isfinite(log_q).all()
    return log_q/20, params


def conditionals(index, scores):
    """Exact tree sums, including finite but extreme sigmoid logits."""
    scores = np.asarray(scores, np.float64)
    if float(np.ptp(scores))*20 < 700:
        return index.conditional_logs(scores)
    mass = np.full(len(index.nodes), -np.inf)
    np.logaddexp.at(mass, index.node_items, 20*scores[index.item_refs-1])
    assert np.isfinite(mass).all()
    return {index.nodes[n]:{token:float(mass[child]-mass[n]) for token,child in children.items()}
            for n,children in enumerate(index.children) if children}
