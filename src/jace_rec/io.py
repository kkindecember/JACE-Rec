"""Portable project configuration and atomic experiment records."""
import hashlib
import json
import os
from pathlib import Path
import time


def read(path):
    with Path(path).open() as stream:
        return json.load(stream)


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temporary.replace(path)


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):digest.update(block)
    return digest.hexdigest()


def stamp():
    return time.strftime('%Y-%m-%dT%H:%M:%S%z')


def rows(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():yield json.loads(line)


class Project:
    def __init__(self,configuration):
        self.config_path=Path(configuration).resolve()
        self.root=Path(os.environ.get('JACE_PROJECT_ROOT', self.config_path.parent.parent)).resolve()
        self.code_root=Path(os.environ.get('JACE_CODE_ROOT', self.root)).resolve()
        self.config=read(self.config_path)
        self.domain=self.config['domain']
        fixed=dict(score_scale=20.,text_recency_decay=.1,gram_history=20,sas_history=50,
                   text_history='full visible',beam_size=50,length_penalty=1.)
        for key,value in fixed.items():
            if self.config.get(key)!=value:
                raise ValueError(f'This frozen implementation requires {key}={value!r}')

    def path(self,key):
        path=Path(self.config['paths'][key])
        return path if path.is_absolute() else self.root/path

    def samples(self,role):
        return read(self.path('data')/f'{role}_samples.json')

    def asset(self,name):
        return self.path(name) if name in self.config['paths'] else self.path('data')/(name+'.json')

    def output_config(self,path,**updates):
        """Write derived configuration under this project's configs directory."""
        import copy
        destination=Path(path).resolve()
        if destination.parent!=self.root/'configs':
            raise ValueError('Derived configurations belong in the project configs directory')
        config=copy.deepcopy(self.config)
        for key,value in updates.items():
            value=Path(value).resolve()
            try:config['paths'][key]=str(value.relative_to(self.root))
            except ValueError:config['paths'][key]=str(value)
        write(destination,config)
