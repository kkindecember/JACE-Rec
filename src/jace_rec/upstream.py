"""Train clean upstream SASRec/GRAM models from the packaged training events."""
import copy
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn import functional as F

from .io import read,rows,write,stamp,sha
from .runtime import initialize_gram,collator,input_record,Runtime
from .sasrec import SASRec


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def state_cpu(model):
    return {key:value.detach().cpu().clone() for key,value in model.state_dict().items()}


def save(path,obj):
    path=Path(path);temporary=path.with_name(path.name+'.tmp');torch.save(obj,temporary);temporary.replace(path)


def sas_batch(sequences,indices):
    inputs=[];targets=[]
    for j in indices:
        seq=sequences[j][-51:];x=seq[:-1];y=seq[1:];padding=50-len(x)
        inputs.append([0]*padding+x);targets.append([0]*padding+y)
    return torch.tensor(inputs,dtype=torch.long),torch.tensor(targets,dtype=torch.long)


def train_sasrec(project,output,device='cuda:0',epochs=200,seed=0):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True);seed_all(seed)
    if (output/'latest.pt').exists() or (output/'sasrec.pt').exists():
        raise FileExistsError('Use a new output directory for upstream training')
    sequence_by_user={r['user_id']:r['history'] for r in rows(project.path('data')/'upstream_sequences.jsonl')}
    sequences=[sequence_by_user[u] for u in sorted(sequence_by_user)]
    catalog=read(project.asset('catalog'));n=len(catalog['items'])-1
    model=SASRec(n,64,50,2,2,.2,architecture='legacy').to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=0.)
    if (output/'sasrec.pt').exists():raise FileExistsError('A trained SASRec already exists in this output directory')
    for epoch in range(1,epochs+1):
        order=list(range(len(sequences)));random.Random(seed+epoch).shuffle(order);model.train();losses=[]
        for left in range(0,len(order),128):
            x,y=sas_batch(sequences,order[left:left+128]);x=x.to(device);y=y.to(device)
            valid=y.ne(0);hidden=model(x);logits=hidden[valid]@model.item_emb.weight.T
            loss=F.cross_entropy(logits,y[valid],label_smoothing=.1)
            assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step();optimizer.zero_grad(set_to_none=True);losses.append(float(loss.detach()))
        save(output/'latest.pt',dict(state_dict=state_cpu(model),optimizer=optimizer.state_dict(),epoch=epoch,training_only=True))
        write(output/'status.json',dict(state='RUNNING',stage='sasrec',seed=seed,epoch=epoch,total_epochs=epochs,loss=float(np.mean(losses)),at=stamp()))
    path=output/'sasrec.pt'
    save(path,dict(state_dict=state_cpu(model),architecture='legacy',num_items=n,epoch=epochs,seed=seed,training_only=True,
        train_complete=True,hidden=64,maxlen=50,num_layers=2,num_heads=2,dropout=.2))
    embeddings=model.item_emb.weight.detach().cpu().numpy()[1:]
    embeddings=embeddings/np.maximum(np.linalg.norm(embeddings,axis=1,keepdims=True),1e-12)
    k={'Toys':5,'Beauty':10,'Sports':10,'Yelp':5}[project.domain];ids=np.arange(1,n+1);neighbors=[]
    for left in range(0,n,128):
        scores=embeddings[left:left+128]@embeddings.T
        scores[np.arange(len(scores)),np.arange(left,left+len(scores))]=-np.inf
        order=np.lexsort((np.broadcast_to(ids,scores.shape),-scores),axis=1)[:,:k];neighbors.append(ids[order])
    neighbors=np.concatenate(neighbors);np.save(output/'neighbor_ids.npy',neighbors)
    text=dict(line.rstrip('\n').split(' ',1) for line in (project.path('data')/'raw/item_plain_text.txt').read_text().splitlines())
    lex=catalog['lexical_ids'];catalog['passages']=['']+[
        f"item: {lex[i]}; similar items: {', '.join(lex[j] for j in neighbors[i-1])}; {text[item]}"
        for i,item in enumerate(catalog['items'][1:],1)]
    write(output/'catalog.json',catalog)
    write(output/'status.json',dict(state='COMPLETED',stage='sasrec',epoch=epochs,checkpoint_sha256=sha(path),at=stamp()))
    return dict(sasrec=path,catalog=output/'catalog.json')


def train_gram(project,output,device='cuda:0',epochs=30,evaluate_every=5,seed=2023):
    """Thirty-epoch FiD-T5 training with native validation N10 selection."""
    if project.config.get('intended_methods') == ['gram'] and 'gram_train' not in project.config['paths']:
        raise ValueError('Native-full evaluation configs require explicit full gram_train inputs before training')
    from transformers import AdamW,get_linear_schedule_with_warmup
    from .metrics import rank_entry,all_metrics
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True);seed_all(seed)
    if (output/'latest.pt').exists() or list(output.glob('epoch_*.pt')):
        raise FileExistsError('Use a new output directory for upstream training')
    if (output/'selection.json').exists():raise FileExistsError('A selected GRAM already exists in this output directory')
    catalog=read(project.asset('catalog'));training=list(rows(project.path('gram_train') if 'gram_train' in project.config['paths'] else project.path('data')/'gram_train.jsonl'))
    model,tokenizer=initialize_gram(project,checkpoint=False);model.to(device);combine=collator(project,tokenizer,'train')
    groups=[dict(params=[p for n,p in model.named_parameters() if not any(s in n for s in ('bias','LayerNorm.weight'))],weight_decay=.01),
            dict(params=[p for n,p in model.named_parameters() if any(s in n for s in ('bias','LayerNorm.weight'))],weight_decay=0.)]
    optimizer=AdamW(groups,lr=.001,eps=1e-6)
    steps_per_epoch=math.ceil(len(training)/128);total=steps_per_epoch*epochs
    scheduler=get_linear_schedule_with_warmup(optimizer,int(.05*total),total)
    candidates=[];step=0
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    for epoch in range(1,epochs+1):
        order=list(range(len(training)));random.Random(seed+epoch).shuffle(order);model.train();losses=[]
        for left in range(0,len(order),128):
            window=order[left:left+128];optimizer.zero_grad(set_to_none=True)
            for offset in range(0,len(window),16):
                selected=window[offset:offset+16]
                batch=combine([input_record(training[j],catalog,target_free=False) for j in selected])
                labels=batch['target_ids'].to(device)
                result=model(input_ids=batch['item_text_ids'].to(device),attention_mask=batch['item_text_masks'].to(device),
                    labels=labels,return_dict=True)
                loss=result.loss;assert torch.isfinite(loss)
                (loss/math.ceil(len(window)/16)).backward();losses.append(float(loss.detach()))
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True);step+=1
            if step%100==0:
                save(output/'latest.pt',dict(state_dict=state_cpu(model),optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(),epoch=epoch,step=step))
                write(output/'status.json',dict(state='RUNNING',stage='gram',epoch=epoch,step=step,total_steps=total,at=stamp()))
        path=output/f'epoch_{epoch:02d}.pt'
        if epoch%evaluate_every==0 or epoch==epochs:
            save(path,dict(state_dict=state_cpu(model),epoch=epoch,seed=seed,training_only=True))
            # Reuse the same encoder/decoder objects; no duplicate model on GPU.
            from .tree import ExpertTrie
            runtime=Runtime.__new__(Runtime);runtime.project=project;runtime.device=torch.device(device)
            runtime.catalog=catalog;runtime.model=model.eval();runtime.tokenizer=tokenizer;runtime.collator=combine
            runtime.index=ExpertTrie(catalog['decoder_paths'],range(1,len(catalog['items'])))
            runtime.max_length=max(map(len,catalog['decoder_paths']))
            validation=project.samples('validation');folder=output/f'validation_epoch_{epoch:02d}';folder.mkdir(exist_ok=True)
            ranks=[]
            tf32=torch.backends.cuda.matmul.allow_tf32;cudnn_tf32=torch.backends.cudnn.allow_tf32
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            with (folder/'predictions.jsonl').open('w') as stream,torch.no_grad():
                for j,sample in enumerate(validation):
                    prediction=runtime.decode(runtime.context([sample]),alpha=0.)
                    ranked=rank_entry(prediction['ids'],prediction['scores'],sample['target']);ranks.append(ranked['rank'])
                    import json
                    stream.write(json.dumps(dict(row_index=j,user_id=sample['user_id'],target=sample['target'],**ranked))+'\n')
                    if (j+1)%25==0:
                        stream.flush();write(output/'status.json',dict(state='RUNNING',stage='validation',epoch=epoch,
                            completed_users=j+1,total_users=len(validation),at=stamp()))
            torch.backends.cuda.matmul.allow_tf32=tf32;torch.backends.cudnn.allow_tf32=cudnn_tf32
            metric=all_metrics(ranks);write(folder/'metrics.json',metric)
            candidates.append(dict(epoch=epoch,checkpoint=str(path),checkpoint_sha256=sha(path),validation_N10=metric['N10']))
    selected=max(candidates,key=lambda c:(c['validation_N10'],-c['epoch']))
    write(output/'selection.json',dict(state='COMPLETED',best=selected,candidates=candidates,
        selection='full native validation N10; earlier epoch tie',at=stamp()))
    write(output/'status.json',dict(state='COMPLETED',stage='gram',best_epoch=selected['epoch'],at=stamp()))
    return dict(gram=Path(selected['checkpoint']))
