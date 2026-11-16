"""Offline Qwen item encoding in the separate encoder environment."""
from pathlib import Path
import time
from .io import read, write, sha, stamp

def encode(project,output,device='cuda:0'):
    import numpy as np
    import torch
    from transformers import AutoModel,AutoTokenizer
    output=Path(output).resolve();output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():raise FileExistsError('Use a new path rather than overwrite frozen embeddings')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    info=read(project.path('data')/'encoder_texts.json');texts=info['texts']
    identity=dict(texts_sha256=sha(project.path('data')/'encoder_texts.json'),
        model_config_sha256=sha(project.path('qwen')/'config.json'),
        model_revision='5cf2132abc99cad020ac570b19d031efec650f2b',pooling='last token; left padding',batch_size=8)
    identity_path=output.with_suffix('.identity.json')
    if identity_path.exists():assert read(identity_path)==identity,'Changed encoding inputs'
    else:
        assert not output.with_suffix('.progress.json').exists(),'Unidentified previous encoding'
        write(identity_path,identity)
    tokenizer=AutoTokenizer.from_pretrained(str(project.path('qwen')),padding_side='left',local_files_only=True)
    lengths=np.asarray([len(v) for v in tokenizer(texts,truncation=False)['input_ids']]);assert lengths.max()<=2048
    model=AutoModel.from_pretrained(str(project.path('qwen')),local_files_only=True,torch_dtype=torch.float16,
        attn_implementation='sdpa').eval().requires_grad_(False).to(device)
    @torch.inference_mode()
    def vectors(documents):
        batch=tokenizer(documents,padding=True,truncation=False,return_tensors='pt').to(device)
        assert batch['attention_mask'][:,-1].all()
        return torch.nn.functional.normalize(model(**batch).last_hidden_state[:,-1].float(),dim=1).cpu().numpy()
    probe=list(dict.fromkeys([0,int(lengths.argmin()),int(lengths.argmax()),len(texts)-1]))
    mixed=vectors([texts[j] for j in probe]);single=np.concatenate([vectors([texts[j]]) for j in probe])
    cosine=(mixed*single).sum(1);assert cosine.min()>.999
    path=output.with_suffix('.vectors.npy');progress=output.with_suffix('.progress.json')
    done=read(progress)['completed_items'] if progress.exists() else 0
    result=np.lib.format.open_memmap(path,mode='r+' if done else 'w+',dtype=np.float32,shape=(len(texts),2560))
    order=np.argsort(lengths,kind='stable');started=time.monotonic()
    for left in range(done,len(texts),8):
        ix=order[left:left+8];result[ix]=vectors([texts[j] for j in ix]);done=left+len(ix)
        if done%128==0 or done==len(texts):
            result.flush();write(progress,dict(state='RUNNING',completed_items=done,total_items=len(texts),at=stamp()))
    np.testing.assert_allclose(np.linalg.norm(result,axis=1),1.,atol=3e-6)
    np.savez(output,item_ids=np.asarray(info['item_ids']),embeddings=result)
    write(output.with_suffix('.encoding.json'),dict(state='COMPLETED',items=len(texts),embedding_sha256=sha(output),
        model_revision='5cf2132abc99cad020ac570b19d031efec650f2b',pooling='last actual token; left padding',
        model_precision='float16',output_precision='float32',instructions=None,maximum_tokens=int(lengths.max()),
        batch_single_min_cosine=float(cosine.min()),elapsed_seconds=time.monotonic()-started))
    write(progress,dict(state='COMPLETED',completed_items=len(texts),total_items=len(texts),at=stamp()))
