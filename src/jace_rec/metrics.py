"""Single-target full-catalog Recall and NDCG; history items remain eligible."""
import numpy as np

def rank_entry(ids, scores, target):
    ids, scores = np.asarray(ids), np.asarray(scores, dtype=np.float64)
    order = np.lexsort((ids, -scores))
    ids, scores = ids[order].tolist(), scores[order].tolist()
    assert len(ids) == len(set(ids)) == 50 and np.isfinite(scores).all()
    return dict(ids=ids, scores=scores, rank=ids.index(target)+1 if target in ids else 1 << 30)

def all_metrics(ranks):
    ranks=np.asarray(ranks)
    if ranks.ndim!=1 or not len(ranks) or np.any(ranks<=0):
        raise ValueError('Positive 1-based ranks are required; use a large sentinel for misses')
    result={}
    for k in (1,3,5,10,20,50):
        hit=ranks<=k
        result[f'R{k}']=float(hit.mean())
        result[f'N{k}']=float(np.where(hit,1/np.log2(ranks+1),0).mean())
    return result
