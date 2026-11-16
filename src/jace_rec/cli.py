"""Command line interface for the standalone JACE-Rec checkout."""
import argparse
import json
from pathlib import Path

import sys

from .io import Project,read,sha,write


def main():
    parser=argparse.ArgumentParser(prog='jace-rec')
    parser.add_argument('--config',type=Path,default=Path('configs/current-toys-2023.json'))
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--memory-mib',type=int,default=22000,help='GPU memory reservation for dynamic admission')
    commands=parser.add_subparsers(dest='command',required=True)
    verify=commands.add_parser('verify-assets');verify.add_argument('--full',action='store_true')
    fit=commands.add_parser('train-gate');fit.add_argument('--output',type=Path,required=True)
    fit.add_argument('--arms',nargs='+',default=['joint_user_alpha'])
    fit.add_argument('--objective',choices=['path'],default='path')
    for name in ('evaluate','calibrate'):
        command=commands.add_parser(name);command.add_argument('--output',type=Path,required=True)
        command.add_argument('--device',choices=['auto','cpu'],default='auto');command.add_argument('--head-dir',type=Path)
        if name=='evaluate':
            command.add_argument('--role',choices=['validation','test','calibration'],default='test')
            command.add_argument('--methods',nargs='+',default=['A6'],choices=['A0','A6','A8','joint_user_alpha','gram','fixed_default'])
            command.add_argument('--start',type=int,default=0);command.add_argument('--stop',type=int)
            command.add_argument('--step',type=int);command.add_argument('--selection',type=Path)
    select=commands.add_parser('select');select.add_argument('--calibration',type=Path,required=True)
    select.add_argument('--head-dir',type=Path,required=True);select.add_argument('--output',type=Path,required=True)
    cache=commands.add_parser('cache-training');cache.add_argument('--output',type=Path,required=True)
    cache.add_argument('--device',choices=['auto','cpu'],default='auto');cache.add_argument('--write-config',type=Path)
    encoder=commands.add_parser('encode-items');encoder.add_argument('--output',type=Path,required=True)
    encoder.add_argument('--device',choices=['auto','cpu'],default='auto');encoder.add_argument('--write-config',type=Path)
    for name in ('train-sasrec','train-gram'):
        command=commands.add_parser(name);command.add_argument('--output',type=Path,required=True)
        command.add_argument('--device',choices=['auto','cpu'],default='auto');command.add_argument('--epochs',type=int)
        command.add_argument('--write-config',type=Path)
        command.add_argument('--seed',type=int)
        if name=='train-gram':command.add_argument('--evaluate-every',type=int,default=5)
    args=parser.parse_args()
    project=Project(args.config)
    current=project.config.get('model_version')=='no_selfmatch_v2'
    if not current:
        parser.error('Use a configuration for model_version=no_selfmatch_v2')
    if current and args.command=='train-gate' and (args.arms!=['joint_user_alpha'] or args.objective!='path'):
        parser.error('The current release trains the A6 joint path-likelihood head only')
    if args.command=='calibrate' and current and args.head_dir is None:
        parser.error('Current calibration requires --head-dir from a fresh train-gate run')
    if args.command=='select' and args.output.exists():
        parser.error('Selection output exists; use a new file to preserve the frozen selection')
    if args.command not in ('verify-assets','select'):
        from .release_execution import execute_snapshot
        execute_snapshot(project,args,sys.argv[1:])
    if args.command!='verify-assets':
        import torch
        torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
        from .runtime import Runtime
        factory=Runtime
    if args.command=='verify-assets':
        manifest=read(project.root/'assets/manifest.json');checked=0
        files=manifest['files']
        if current:
            names=manifest['configurations'][project.config_path.name]['files']
            files={name:files[name] for name in names}
        for name,record in files.items():
            path=project.root/name
            assert path.is_file() and path.stat().st_size==record['bytes'],name
            if args.full:assert sha(path)==record['sha256'],name
            checked+=1
        print(json.dumps(dict(state='PASSED',files=checked,hashes_verified=args.full,total_bytes=sum(v['bytes'] for v in files.values()))))
    elif args.command=='train-gate':
        from .training import fit
        print(json.dumps(fit(project,args.output,args.arms,args.objective),indent=2))
    elif args.command in ('evaluate','calibrate'):
        from .evaluation import evaluate
        if args.command=='calibrate':
            result=evaluate(project,args.output,role='calibration',device=args.device,head_directory=args.head_dir,calibrate=True,runtime_factory=factory,arms=('joint_user_alpha',))
        else:
            if args.selection:
                project.config['paths']['selection']=str(args.selection.resolve())
                project.config['paths'].pop('head',None)
            names={'A0':'gram','A6':'joint_user_alpha','A8':'fixed_default'}
            args.methods=[names.get(n,n) for n in args.methods]
            result=evaluate(project,args.output,role=args.role,methods=args.methods,device=args.device,
                head_directory=args.head_dir,step=args.step,start=args.start,stop=args.stop,runtime_factory=factory)
        print(json.dumps(result,indent=2))
    elif args.command=='select':
        from .evaluation import select
        print(json.dumps(select(project,args.calibration,args.head_dir,args.output,arms=('joint_user_alpha',)),indent=2))
    elif args.command=='cache-training':
        from .training import build_cache
        paths=build_cache(project,args.output,args.device,runtime_factory=factory)
        if args.write_config:
            if current:project.config['paths'].pop('head',None)
            project.output_config(args.write_config,**paths)
    elif args.command=='encode-items':
        from .encoding import encode
        destination=args.output/'qwen4_embeddings.npz'
        encode(project,destination,args.device)
        if args.write_config:project.output_config(args.write_config,embeddings=destination)
    else:
        from .upstream import train_sasrec,train_gram
        seed=args.seed if args.seed is not None else project.config.get('expert_gate_seed',0) if args.command=='train-sasrec' else project.config.get('generator_seed',2023)
        if args.command=='train-sasrec':paths=train_sasrec(project,args.output,args.device,args.epochs or 200,seed)
        else:paths=train_gram(project,args.output,args.device,args.epochs or 30,args.evaluate_every,seed)
        if args.write_config:
            if current:project.config['paths'].pop('head',None)
            project.output_config(args.write_config,**paths)


if __name__=='__main__':main()
