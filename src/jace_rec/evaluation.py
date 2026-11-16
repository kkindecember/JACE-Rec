"""Resumable single-GPU prediction, internal calibration and metric export."""
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from . import gating as C
from .io import read,write,sha,stamp
from .metrics import rank_entry,all_metrics
from .runtime import Runtime,load_head,head_path
from .tree import conditionals


def same_query(a,b):
    return all(a[key]==b[key] for key in ('user_id','history','full_history','target'))


def resume(path,samples,indices,names):
    path=Path(path);saved=[];offset=0
    if not path.exists():return saved
    with path.open('rb') as stream:
        while True:
            line=stream.readline()
            if not line:break
            try:row=json.loads(line)
            except (ValueError,UnicodeDecodeError):
                assert not line.endswith(b'\n') and not stream.read(1),'Corrupt complete output'
                path.with_name(path.name+f'.torn.{time.time_ns()}').write_bytes(line)
                with path.open('r+b') as handle:handle.truncate(offset)
                break
            assert len(saved)<len(indices);j=indices[len(saved)]
            assert row['row_index']==j and same_query(row,samples[j]) and set(row['methods'])==set(names)
            for result in row['methods'].values():assert rank_entry(result['ids'],result['scores'],row['target'])==result
            saved.append(row);offset+=len(line)
    if saved and path.stat().st_size:
        with path.open('rb') as stream:stream.seek(-1,2);last=stream.read(1)
        if last!=b'\n':
            with path.open('ab') as stream:stream.write(b'\n')
    return saved


@torch.no_grad()
def evaluate(project,output,role='test',methods=('joint_user_alpha',),device='cuda:0',
             head_directory=None,step=None,start=0,stop=None,calibrate=False,runtime_factory=None,arms=C.ARMS):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    samples=project.samples(role);stop=len(samples) if stop is None else min(stop,len(samples))
    assert 0<=start<stop<=len(samples);indices=list(range(start,stop));heads={}
    if calibrate:
        assert role=='calibration'
        heads={'initial':load_head(project,C.ARMS[0],0,head_directory)}
        heads.update({f'{arm}_s{s}':load_head(project,arm,s,head_directory) for arm in arms for s in (300,400)})
        methods=tuple(heads)
    else:
        for name in methods:
            arm=name[:-7] if name.endswith('_direct') else name
            if arm in C.ARMS and arm not in heads:heads[arm]=load_head(project,arm,step,head_directory)
            elif name=='fixed_default':heads[name]=C.Head('joint_user_alpha').eval().requires_grad_(False)
            elif name not in ('gram','sas_direct','qwen_direct') and arm not in C.ARMS:raise ValueError(name)
    norm=read(project.asset('normalization'))
    identity=dict(model_version=project.config.get('model_version','legacy_corrected'),domain=project.domain,role=role,start=start,stop=stop,methods=list(methods),
        gram_sha256=sha(project.path('gram')),sasrec_sha256=sha(project.path('sasrec')),
        catalog_sha256=sha(project.asset('catalog')),embeddings_sha256=sha(project.path('embeddings')),
        inference_settings={key:project.config[key] for key in ('lexical_type','score_scale','text_recency_decay',
                           'gram_history','sas_history','text_history','beam_size','length_penalty')},
        normalization_sha256=sha(project.asset('normalization')),
        input_sha256=sha(project.path('data')/f'{role}_samples.json'),
        head_sha256={name:('fixed-default-beta=.5-rho=.01-alpha=.3' if name=='fixed_default' else
            sha(head_path(project,C.ARMS[0] if name=='initial' else name.rsplit('_s',1)[0] if calibrate else name,
                0 if name=='initial' else int(name.rsplit('_s',1)[1]) if calibrate else step,head_directory))) for name in heads})
    code_manifest=project.code_root/'manifest.json'
    if code_manifest.exists():identity['code_manifest_sha256']=sha(code_manifest)

    if (output/'identity.json').exists():assert read(output/'identity.json')==identity,'Changed run inputs'
    else:write(output/'identity.json',identity)
    saved=resume(output/'predictions.jsonl',samples,indices,methods);pending=indices[len(saved):]
    if pending:
        runtime=(runtime_factory or Runtime)(project,device);fingerprint=runtime.digest();began=time.monotonic()
        with (output/'predictions.jsonl').open('a') as stream:
            for left in range(0,len(pending),16):
                block=pending[left:left+16];experts=runtime.experts([samples[j] for j in block])
                for k,j in enumerate(block):
                    sample=samples[j];expert=experts[k];raw,seen=C.features(expert,sample['full_history'])
                    x=C.normalize(raw,norm);predictions={};parameters={}
                    need_generation=any(not name.endswith('_direct') for name in methods)
                    context=runtime.context([sample]) if need_generation else None
                    for name in methods:
                        if name=='gram':
                            pred=runtime.decode(context,alpha=0.);result=rank_entry(pred['ids'],pred['scores'],sample['target'])
                        else:
                            if name in ('sas_direct','qwen_direct'):scores=expert[:,int(name=='qwen_direct')]
                            else:
                                key=name[:-7] if name.endswith('_direct') else name
                                scores,param=C.catalog_scores(heads[key],x,expert,seen);parameters[name]=param
                            if name.endswith('_direct'):
                                ids=np.arange(1,len(scores)+1);order=np.lexsort((ids,-scores))[:50]
                                result=rank_entry(ids[order],scores[order],sample['target'])
                            else:
                                pred=runtime.decode(context,conditionals(runtime.index,scores),param['alpha'])
                                result=rank_entry(pred['ids'],pred['scores'],sample['target'])
                        predictions[name]=result
                    row=dict(row_index=j,**{key:sample[key] for key in ('user_id','history','full_history','target')},
                             methods=predictions,parameters=parameters)
                    stream.write(json.dumps(row)+'\n');stream.flush();saved.append(row)
                    if len(saved)==1 or len(saved)%8==0 or len(saved)==len(indices):
                        os.fsync(stream.fileno())
                        write(output/'status.json',dict(state='RUNNING',completed_users=len(saved),total_users=len(indices),
                            elapsed_seconds=time.monotonic()-began,at=stamp()))
                if torch.cuda.is_available() and torch.cuda.memory_reserved()>2*torch.cuda.memory_allocated():torch.cuda.empty_cache()
        assert runtime.digest()==fingerprint
    ranks={name:np.asarray([r['methods'][name]['rank'] for r in saved]) for name in methods}
    metrics={name:all_metrics(value) for name,value in ranks.items()}
    np.savez(output/'ranks.npz',**ranks)
    write(output/'metrics.json',dict(state='COMPLETED',domain=project.domain,role=role,users=len(saved),metrics=metrics,
        prediction_sha256=sha(output/'predictions.jsonl'),history_filter=False,beam_size=50,length_penalty=1.,at=stamp()))
    write(output/'status.json',dict(state='COMPLETED',completed_users=len(saved),total_users=len(indices),at=stamp()))
    return metrics


def select(project,calibration_directory,head_directory,output,arms=C.ARMS):
    report=read(Path(calibration_directory)/'metrics.json');assert report['role']=='calibration'
    assert report['state']=='COMPLETED' and report['domain']==project.domain
    identity=read(Path(calibration_directory)/'identity.json')
    assert identity['model_version']==project.config.get('model_version','legacy_corrected')
    assert identity['normalization_sha256']==sha(project.asset('normalization'))
    assert identity['input_sha256']==sha(project.path('data')/'calibration_samples.json')
    assert identity['start']==0 and identity['stop']==len(project.samples('calibration')), 'Select using the complete internal calibration set'
    chosen={}
    for arm in arms:
        candidates=[(0,'initial')]+[(step,f'{arm}_s{step}') for step in (300,400)]
        step,name=max(candidates,key=lambda pair:(report['metrics'][pair[1]]['N10'],report['metrics'][pair[1]]['R10'],-pair[0]))
        path=Path(head_directory)/f'{arm}_s{step}.pt'
        assert sha(path)==identity['head_sha256'][name], 'Changed calibrated checkpoint'
        chosen[arm]=dict(step=step,checkpoint=str(path),checkpoint_sha256=sha(path))
    write(output,dict(state='COMPLETED',selected=chosen,calibration_users=report['users'],
        selection_rule='internal calibration N10, R10, earlier step',frozen_at=stamp(),primary_arm='joint_user_alpha'))
    return chosen
