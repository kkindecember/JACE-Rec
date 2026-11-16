"""Frozen GRAM/SASRec runtime with target-free user encoding."""
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

from . import gating
from .decoding import BatchedTrieMask,ProbabilityGuide,bulk_beam_bookkeeping
from .experts import score_batch,unit_rows
from .io import read,sha
from .metrics import rank_entry
from .sasrec import SASRec
from .tree import ExpertTrie,conditionals


def vendor_imports(root):
    if not (Path(root)/'vendor/gram/model/gram.py').is_file():
        raise FileNotFoundError('GRAM backend is not installed. See docs/REPRODUCE.md.')
    sys.path.insert(0,str(Path(root)/'vendor'))
    from gram.model.gram import GRAM
    from gram.model.gram_t5_outputs import BaseModelOutputWithPastAndCrossAttentions
    from gram.processor import CollatorGRAM
    return GRAM,BaseModelOutputWithPastAndCrossAttentions,CollatorGRAM


def initialize_gram(project,checkpoint=True):
    from transformers import AutoTokenizer,AutoModelForSeq2SeqLM,T5Config
    GRAM,_,_=vendor_imports(project.code_root)
    path=str(project.path('t5'))
    config=T5Config.from_pretrained(path,local_files_only=True)
    config.max_seq_len=128;config.max_item_num=20;config.use_position_embedding=1;config.sample_num='1'
    tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
    base=AutoModelForSeq2SeqLM.from_pretrained(path,config=config,local_files_only=True)
    model=GRAM(config=config);model.load_t5(base.state_dict());del base
    if checkpoint:
        saved=torch.load(project.path('gram'),map_location='cpu')
        model.load_state_dict(saved.get('state_dict',saved),strict=True)
    return model,tokenizer


def collator(project,tokenizer,mode='test'):
    _,_,CollatorGRAM=vendor_imports(project.code_root)
    args=SimpleNamespace(item_prompt_max_len=128,target_max_len=32,max_his=20,item_id_type='split',
                         hierarchical_id_type=project.config['lexical_type'])
    return CollatorGRAM(tokenizer,args=args,mode=mode)


def input_record(sample,catalog,target_free=True):
    history=sample['history'][-20:][::-1];lex=catalog['lexical_ids']
    prompt='What would user purchase after '+' ; '.join(lex[i] for i in history)+' ?'
    return dict(input=[prompt]+[catalog['passages'][i] for i in history],
        output=lex[1 if target_free else sample['target']],user_id=sample['user_id'],
        history_item_ids=history,target_item_id=0 if target_free else sample['target'])


class Runtime:
    def __init__(self,project,device='cuda:0',load_experts=True):
        self.project=project;self.device=torch.device(device)
        torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        self.catalog=read(project.asset('catalog'))
        self.model,self.tokenizer=initialize_gram(project)
        self.model.eval().requires_grad_(False).to(self.device)
        self.collator=collator(project,self.tokenizer)
        self.index=ExpertTrie(self.catalog['decoder_paths'],range(1,len(self.catalog['items'])))
        self.max_length=max(map(len,self.catalog['decoder_paths']))
        if load_experts:
            saved=torch.load(project.path('sasrec'),map_location='cpu')
            assert saved['training_only'] and saved['architecture']=='legacy'
            self.sas=SASRec(len(self.catalog['items'])-1,64,50,2,2,0.,architecture='legacy')
            self.sas.load_state_dict(saved['state_dict'],strict=True)
            self.sas.eval().requires_grad_(False).to(self.device)
            with np.load(project.path('embeddings')) as packed:
                assert packed['item_ids'].tolist()==self.catalog['items'][1:]
                self.embeddings=packed['embeddings']
            self.candidates=unit_rows(self.embeddings)

    def digest(self):
        digest=hashlib.sha256()
        for key,value in self.model.state_dict().items():
            digest.update(key.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    @torch.no_grad()
    def context(self,samples):
        batch=self.collator([input_record(s,self.catalog,target_free=True) for s in samples])
        ids=batch['item_text_ids'].to(self.device);mask=batch['item_text_masks'].to(self.device)
        self.model.encoder.n_passages=ids.size(1)
        hidden=self.model.encoder(input_ids=ids.reshape(len(samples),-1),
            attention_mask=mask.reshape(len(samples),-1),return_dict=True)[0]
        return dict(ids=ids,mask=mask,hidden=hidden)

    @torch.no_grad()
    def teacher(self,context,paths):
        _,Output,_=vendor_imports(self.project.code_root)
        width=max(len(p)-1 for p in paths)
        ids=torch.zeros((len(paths),width),dtype=torch.long,device=self.device)
        mask=torch.zeros_like(ids,dtype=torch.bool)
        for j,path in enumerate(paths):
            ids[j,:len(path)-1]=torch.tensor(path[:-1],device=self.device);mask[j,:len(path)-1]=True
        return self.model(encoder_outputs=Output(last_hidden_state=context['hidden']),
            attention_mask=context['mask'].reshape(len(paths),-1),decoder_input_ids=ids,
            decoder_attention_mask=mask,use_cache=False,return_dict=True).logits

    @torch.no_grad()
    def experts(self,samples):
        return score_batch(self.sas,self.embeddings,samples,self.device,self.candidates)

    @torch.no_grad()
    def decode(self,context,q=None,alpha=0.):
        from transformers import LogitsProcessorList
        GRAM,Output,_=vendor_imports(self.project.code_root)
        guide=ProbabilityGuide(self.index,q,alpha)
        with bulk_beam_bookkeeping(self.model):
            output=super(GRAM,self.model).generate(input_ids=context['ids'].reshape(1,-1),
                attention_mask=context['mask'].reshape(1,-1),encoder_outputs=Output(last_hidden_state=context['hidden']),
                max_length=self.max_length,num_beams=50,num_return_sequences=50,length_penalty=1.,
                return_dict_in_generate=True,output_scores=False,
                logits_processor=LogitsProcessorList([BatchedTrieMask(self.index),guide]))
        paths=[p[:p.index(1)+1] for p in output.sequences.cpu().tolist()]
        ids=[self.index.path_to_item[tuple(p)] for p in paths]
        scores=output.sequences_scores.detach().cpu().double().numpy()
        np.testing.assert_allclose(guide.rescore(paths,q,alpha),scores,atol=2e-6,rtol=1e-5)
        return dict(ids=ids,scores=scores,paths=paths)

    @torch.no_grad()
    def predict(self,sample,head,normalization,expert=None,context=None):
        expert=self.experts([sample])[0] if expert is None else expert
        features,seen=gating.features(expert,sample['full_history'])
        x=gating.normalize(features,normalization)
        scores,parameters=gating.catalog_scores(head,x,expert,seen)
        q=conditionals(self.index,scores)
        context=self.context([sample]) if context is None else context
        prediction=self.decode(context,q,parameters['alpha'])
        result=rank_entry(prediction['ids'],prediction['scores'],sample['target'])
        return result,parameters


def head_path(project,arm,step=None,directory=None):
    if step is None and directory is None and 'head' in project.config['paths']:
        return project.path('head')
    directory=Path(directory) if directory else project.path('heads')
    if step is None:step=read(project.asset('selection'))['selected'][arm]['step']
    return directory/f'{arm}_s{step}.pt'


def load_head(project,arm,step=None,directory=None):
    path=head_path(project,arm,step,directory);saved=torch.load(path,map_location='cpu')
    assert saved['normalization_sha256']==sha(project.asset('normalization'))
    head=gating.Head(arm);head.load_state_dict(saved['state_dict'],strict=True)
    return head.eval().requires_grad_(False)
