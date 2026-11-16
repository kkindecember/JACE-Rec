"""Joint gate optimization and frozen-model training-cache construction."""
from pathlib import Path
import time

import numpy as np
import torch

from . import gating as C
from .io import read,write,sha,stamp
from .tree import PathMass


def training_data(project):
    folder=project.path('cache')
    data={key:np.array(value) for key,value in np.load(folder/'training_cache.npz').items()}
    experts=np.load(folder/'experts.npy',mmap_mode='r');seen=np.load(folder/'seen.npy',mmap_mode='r')
    def batch(indices):
        value={key:torch.from_numpy(np.ascontiguousarray(array[indices])) for key,array in data.items()
               if key not in ('fit_indices','calibration_indices')}
        value['experts']=torch.from_numpy(np.array(experts[indices],dtype=np.float64))
        value['seen']=torch.from_numpy(np.array(seen[indices]));return value
    return data,batch


def fit(project,output,arms=C.ARMS,objective='path'):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    cfg=project.config['training'];data,batch=training_data(project);indices=data['fit_indices']
    tree=PathMass(read(project.asset('catalog'))['decoder_paths'],'cpu');traces={};hashes={}
    for arm in arms:
        if objective != 'path': raise ValueError('JACE uses the joint path objective')
        if (output/f'{arm}_s400.pt').exists():raise FileExistsError('Use a new output directory for a new fit')
        head=C.Head(arm)
        parameters=[p for p in head.parameters() if p.requires_grad]
        optimizers=[torch.optim.AdamW(parameters,lr=cfg['lr'],weight_decay=cfg['weight_decay'])]
        def losses(batch_data):
            return C.loss(head,batch_data,tree.ancestors)
        rng=np.random.default_rng(cfg['seed']);order=rng.permutation(indices);cursor=0;trace=[]
        for step in range(cfg['updates']+1):
            if step:
                if cursor>=len(order):order=rng.permutation(indices);cursor=0
                selected=order[cursor:cursor+cfg['batch_size']];cursor+=len(selected)
                optimizer=optimizers[0]
                head.zero_grad(set_to_none=True);value=losses(batch(selected)).mean()
                if not torch.isfinite(value):raise FloatingPointError('Non-finite gate loss')
                value.backward()
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
                optimizer.step()
            if step in cfg['snapshots']:
                with torch.no_grad():
                    values=[]
                    for left in range(0,len(indices),32):values.extend(losses(batch(indices[left:left+32])).tolist())
                path=output/f'{arm}_s{step}.pt'
                torch.save(dict(arm=arm,step=step,state_dict=head.state_dict(),objective=objective,
                    parameters=sum(p.numel() for p in head.parameters()),trainable_parameters=sum(p.numel() for p in parameters),
                    normalization_sha256=sha(project.asset('normalization')),parent_epoch=project.config['parent_epoch'],
                    fit_users=len(indices),features=list(C.FEATURES),training_config=cfg),path)
                hashes[path.name]=sha(path);trace.append(dict(step=step,fit_loss=float(np.mean(values))))
            if step%20==0:
                write(output/'status.json',dict(state='RUNNING',arm=arm,step=step,total_steps=cfg['updates'],at=stamp()))
        traces[arm]=trace
    report=dict(state='COMPLETED',objective=objective,traces=traces,fit_users=len(indices),sha256=hashes,at=stamp())
    write(output/'training.json',report);write(output/'status.json',dict(state='COMPLETED',at=stamp()))
    return report


@torch.no_grad()
def build_cache(project,output,device='cuda:0',runtime_factory=None):
    from .runtime import Runtime
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    runtime=(runtime_factory or Runtime)(project,device);samples=project.samples('train');n=len(samples);v=len(runtime.catalog['items'])-1
    identity=dict(model_version=project.config.get('model_version','legacy_corrected'),gram_sha256=sha(project.path('gram')),sasrec_sha256=sha(project.path('sasrec')),
        catalog_sha256=sha(project.asset('catalog')),embeddings_sha256=sha(project.path('embeddings')),
        samples_sha256=sha(project.path('data')/'train_samples.json'),split_sha256=sha(project.asset('split')))
    if (output/'identity.json').exists():assert read(output/'identity.json')==identity,'Changed cache inputs'
    else:
        assert not (output/'progress.json').exists(),'Unidentified old cache; use a new output directory'
        write(output/'identity.json',identity)
    tree=PathMass(runtime.catalog['decoder_paths'],'cpu');targets=np.asarray([s['target']-1 for s in samples])
    valid=tree.valid[targets].numpy();progress=read(output/'progress.json')['completed_users'] if (output/'progress.json').exists() else 0
    def mapped(name,shape,dtype):
        path=output/name
        return np.lib.format.open_memmap(path,mode='r+' if progress and path.exists() else 'w+',dtype=dtype,shape=shape)
    experts=mapped('experts.npy',(n,v,2),np.float32)
    native=mapped('native.npy',valid.shape,np.float64);legal=mapped('legal.npy',valid.shape,np.float64)
    started=time.monotonic()
    for left in range(progress,n,16):
        block=samples[left:left+16];experts[left:left+len(block)]=runtime.experts(block)
        for offset in range(0,len(block),4):
            group=block[offset:offset+4];paths=[runtime.catalog['decoder_paths'][s['target']-1] for s in group]
            context=runtime.context(group);logs=runtime.teacher(context,paths).float().log_softmax(-1)
            for j,path in enumerate(paths):
                row=left+offset+j;native[row]=0;legal[row]=0
                for t in range(len(path)-1):
                    children=list(runtime.index.children[runtime.index.lookup[tuple(path[:t+1])]])
                    native[row,t]=float(logs[j,t,path[t+1]])
                    legal[row,t]=float(torch.logsumexp(logs[j,t,children],0))
            del context,logs
        experts.flush();native.flush();legal.flush()
        write(output/'progress.json',dict(state='RUNNING',completed_users=left+len(block),total_users=n,
                                         elapsed_seconds=time.monotonic()-started,at=stamp()))
    seen=mapped('seen.npy',(n,v),np.bool_);features=np.empty((n,5))
    for j,sample in enumerate(samples):features[j],seen[j]=C.features(experts[j],sample['full_history'])
    seen.flush();split=read(project.asset('split'));fit_indices=np.asarray(split['fit_indices'])
    cal_indices=np.asarray(split['calibration_indices'])
    norm=dict(names=list(C.FEATURES),mean=features[fit_indices].mean(0).tolist(),
              std=np.maximum(features[fit_indices].std(0),1e-6).tolist(),fit_only=True,fit_users=len(fit_indices))
    write(output/'normalization.json',norm)
    np.savez(output/'training_cache.npz',features=C.normalize(features,norm),targets=targets,valid=valid,
        native=np.array(native),legal=np.array(legal),fit_indices=fit_indices,calibration_indices=cal_indices)
    write(output/'progress.json',dict(state='COMPLETED',completed_users=n,total_users=n,at=stamp()))
    return dict(cache=output,normalization=output/'normalization.json')
