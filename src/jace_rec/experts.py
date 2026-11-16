"""Frozen sequence and full-history semantic experts (LIME-Rec derived)."""
import numpy as np
import torch

def left_pad(histories, width=50):
    out = np.zeros((len(histories), width), dtype=np.int64)
    for i, h in enumerate(histories):
        tail = h[-width:]
        if tail:
            out[i, -len(tail):] = tail
    return out


def unit_rows(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def semantic_users(histories, embeddings):
    out = np.zeros((len(histories), embeddings.shape[1]), dtype=np.float32)
    for i, h in enumerate(histories):
        w = np.asarray([np.exp(-0.1*(len(h)-1-j)) for j in range(len(h))], dtype=np.float32)
        out[i] = (embeddings[np.asarray(h)-1]*w[:, None]).sum(0)/max(w.sum(), 1e-8)
    return unit_rows(out)


def normalize_score_rows(scores):
    scores=np.asarray(scores,np.float32)
    return (scores-scores.min(axis=1,keepdims=True))/(np.ptp(scores,axis=1,keepdims=True)+1e-8)


def raw_text(histories, embeddings, candidates=None):
    candidates = unit_rows(embeddings) if candidates is None else candidates
    return normalize_score_rows(semantic_users(histories, embeddings) @ candidates.T)


@torch.no_grad()
def score_batch(sas, embeddings, samples, device, candidates=None):
    histories=[sample['full_history'] for sample in samples]
    raw=sas.predict(torch.from_numpy(left_pad(histories)).to(device))[:,1:].cpu().numpy()
    sequence=normalize_score_rows(raw)
    candidates=unit_rows(embeddings) if candidates is None else candidates
    text=raw_text(histories,embeddings,candidates)
    return np.stack((sequence,text),axis=-1).astype(np.float32)
