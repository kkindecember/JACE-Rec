"""Native ActionPiece decoding with mass-preserving JACE guidance."""
import numpy as np
import torch
from transformers.modeling_outputs import BaseModelOutput

class VectorGuide:
    """Same scatter and logsumexp as Guide, with vectorized edge construction."""
    def __init__(self,index,logs,alphas,ensemble=5,beams=50):
        import numpy as np
        self.index,self.logs,self.alphas=index,logs,alphas
        self.ensemble,self.beams=ensemble,beams
        self.tokens={}
        self.q={}

    def __call__(self,input_ids,scores):
        import numpy as np
        groups={}
        for r,p in enumerate(input_ids.detach().cpu().tolist()):
            p=tuple(p);node=self.index.lookup.get(p)
            if node is None or not self.index.children[node]:continue
            user=r//(self.ensemble*self.beams)
            groups.setdefault((user,node),[]).append(r)
        masked=torch.full_like(scores,-torch.inf)
        if not groups:return masked
        rr=[];tt=[];qq=[]
        for (user,node),indices in groups.items():
            if node not in self.tokens:self.tokens[node]=np.array(list(self.index.children[node]),dtype=np.int64)
            tokens=self.tokens[node];key=(user,node)
            if key not in self.q:
                if self.alphas[user]==0:self.q[key]=np.zeros(len(tokens),dtype=np.float64 if scores.dtype==torch.float64 else np.float32)
                else:
                    q=self.logs[user][self.index.nodes[node]]
                    self.q[key]=np.fromiter((q[t] for t in tokens),dtype=np.float64 if scores.dtype==torch.float64 else np.float32,count=len(tokens))
            rr.append(np.repeat(np.array(indices,dtype=np.int64),len(tokens)))
            tt.append(np.tile(tokens,len(indices)))
            qq.append(np.tile(self.q[key],len(indices)))
        r=torch.from_numpy(np.concatenate(rr)).to(scores.device)
        t=torch.from_numpy(np.concatenate(tt)).to(scores.device)
        q=torch.from_numpy(np.concatenate(qq)).to(device=scores.device,dtype=scores.dtype)
        native=scores[r,t];masked[r,t]=native
        mass=torch.logsumexp(masked,-1)
        alphas=torch.tensor(self.alphas,device=scores.device,dtype=scores.dtype)
        a=alphas[torch.div(r,self.ensemble*self.beams,rounding_mode='floor')]
        mixed=torch.logaddexp(native+torch.log1p(-a),mass[r]+q+a.log())
        masked[r,t]=torch.where(a==0,native,mixed)
        return masked


class NativeGuide(VectorGuide):
    def __call__(self, input_ids, scores):
        guided = super().__call__(input_ids, scores)
        # VectorGuide marks entries outside the canonical tree with -inf.
        # Native ActionPiece permits those entries: retain, rather than mask them.
        return torch.where(torch.isfinite(guided), guided, scores)


@torch.no_grad()
def beam_search(model, input_ids, attention_mask, max_length=6, num_beams=1,
                num_return_sequences=1, return_score=False, guide=None,
                encoder_hidden=None):
    size = len(input_ids)
    if encoder_hidden is None:
        encoder_hidden = model.t5.get_encoder()(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True).last_hidden_state
    _, mask, decoder, scores, offset = model.prepare_beam_search_inputs(
        input_ids, attention_mask, size, num_beams)
    encoder = BaseModelOutput(last_hidden_state=encoder_hidden.repeat_interleave(num_beams, dim=0))
    while decoder.shape[1] < max_length:
        output = model.t5(encoder_outputs=encoder, attention_mask=mask,
                          decoder_input_ids=decoder, use_cache=False)
        logs = output.logits[:, -1, :].log_softmax(-1)
        del output
        if guide is not None:
            logs = guide(decoder, logs)
        vocab = logs.shape[-1]
        values, tokens = (logs + scores[:, None]).reshape(size, -1).topk(
            2 * num_beams, dim=1, largest=True, sorted=True)
        parents = (tokens[:, :num_beams] // vocab).reshape(-1) + offset
        decoder = torch.cat((decoder[parents], (tokens[:, :num_beams] % vocab).reshape(-1, 1)), dim=-1)
        scores = values[:, :num_beams].reshape(-1)
    selected = (torch.arange(size, device=input_ids.device)[:, None] * num_beams
                + torch.arange(num_return_sequences, device=input_ids.device)).flatten()
    if return_score:
        return decoder[selected], scores[selected] / (decoder.shape[1] - 1)
    return decoder[selected]


class NativePredictor:
    """Reuse encoder states between gate snapshots; delegate fusion to the author."""
    def __init__(self, tokenizer, catalog):
        self.feature_to_item = {
            tuple(tokenizer._tokenize_once([item])[0]): i
            for i, item in enumerate(catalog) if i}
        self.batch = None
        self.hidden = None

    @torch.no_grad()
    def __call__(self, model, batch, index, logs, alphas, ensemble=5, beam=50):
        assert model.config['num_beams'] == beam
        assert len(batch['input_ids']) == len(alphas) * ensemble
        if batch is not self.batch:
            self.batch = batch
            self.hidden = model.t5.get_encoder()(
                input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                return_dict=True).last_hidden_state
        guide = NativeGuide(index, logs, alphas, ensemble=ensemble, beams=beam)
        original = model.beam_search
        def guided(**kwargs):
            return beam_search(model, guide=guide, encoder_hidden=self.hidden, **kwargs)
        model.beam_search = guided
        try:
            features = model.generate(batch, n_return_sequences=beam).cpu().tolist()
        finally:
            model.beam_search = original
        return np.array([[self.feature_to_item.get(tuple(f), -1) for f in row] for row in features])


def canonical_paths(tokenizer, catalog):
    paths=[]
    for item in catalog[1:]:
        state = tokenizer._tokenize_once([item])
        encoded = tokenizer.encode_labels(state)
        decoded = tokenizer.actionpiece.decode_single_state(encoded)
        assert [tokenizer.actionpiece.rank[x] for x in decoded] == state[0].tolist()
        paths.append([0]+list(encoded)+[tokenizer.eos_token])
    assert len(set(map(tuple,paths))) == len(paths)
    return paths


def path_arrays(paths, index):
    depth=max(map(len,paths))-1
    ancestors=np.empty((depth+1,len(paths)),np.int64)
    valid=np.zeros((len(paths),depth),bool)
    for i,p in enumerate(paths):
        valid[i,:len(p)-1]=True
        for d in range(depth+1):
            ancestors[d,i]=index.lookup[tuple(p[:min(d+1,len(p))])]
    return torch.from_numpy(ancestors),valid


def history_batch(tokenizer,catalog,samples,seed,ensemble=5):
    """Target-free input segmentation, stable across methods and batch/shard sizes."""
    import hashlib
    inputs=[]
    for sample in samples:
        key=f'{seed}:{sample["user_id"]}:{sample["full_history"]}'.encode()
        local_seed=int.from_bytes(hashlib.sha256(key).digest()[:4],'little')
        state=tokenizer._tokenize_once([catalog[i] for i in sample['full_history'][-20:]])
        rng=np.random.get_state()
        try:
            np.random.seed(local_seed)
            for _ in range(ensemble):
                inputs.append([tokenizer.bos_token]+tokenizer.actionpiece.encode(state,shuffle='feature')+[tokenizer.eos_token])
        finally:
            np.random.set_state(rng)
    width=max(map(len,inputs))
    ids=torch.zeros((len(inputs),width),dtype=torch.long)
    masks=torch.zeros_like(ids)
    for j,p in enumerate(inputs):ids[j,:len(p)]=torch.tensor(p);masks[j,:len(p)]=1
    # Native generate uses labels only for the grouping count and output device.
    return dict(input_ids=ids,attention_mask=masks,labels=torch.zeros((len(samples),5),dtype=torch.long))
