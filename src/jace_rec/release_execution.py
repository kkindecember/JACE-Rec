"""Immutable code snapshots and account-policy admission for the public CLI."""
import importlib.util
import os
from pathlib import Path
import shutil
import sys

from .io import read, sha, write


def snapshot(project, output):
    output = Path(output).resolve()
    code = output / 'code'
    if output.exists() and any(output.iterdir()) and not (code / 'manifest.json').exists():
        raise FileExistsError('Choose a new output directory; historical runs cannot be overwritten')
    if not code.exists():
        code.mkdir(parents=True)
        for name in ('src', 'vendor'):
            shutil.copytree(project.code_root/name, code/name,
                            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        backend = project.root/'vendor/gram'
        if backend.is_dir() and not (code/'vendor/gram').exists():
            shutil.copytree(backend, code/'vendor/gram', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        (code/'scripts').mkdir()
        for name in ('jace.py', 'gpu_account_policy.py'):
            shutil.copy2(project.code_root/'scripts'/name, code/'scripts'/name)
        (code/'configs').mkdir()
        shutil.copy2(project.config_path, code/'configs/project.json')
        write(code/'manifest.json', dict(files={str(p.relative_to(code)):sha(p)
              for p in sorted(code.rglob('*')) if p.is_file()}))
    manifest = read(code/'manifest.json')
    for name, digest in manifest['files'].items():
        if sha(code/name) != digest:
            raise ValueError(f'Changed frozen code: {name}')
    if sha(project.config_path) != manifest['files']['configs/project.json']:
        raise ValueError('Changed configuration; choose a new output directory')
    return code


def execute_snapshot(project, args, argv):
    """Return only in a verified worker; the parent exits with the child status."""
    gpu = getattr(args, 'device', 'cpu') != 'cpu'
    spec = importlib.util.spec_from_file_location('gpu_account_policy', project.code_root/'scripts/gpu_account_policy.py')
    policy = importlib.util.module_from_spec(spec); spec.loader.exec_module(policy)
    policy_root = gpu_policy_root(project.root)
    if os.environ.get('JACE_RELEASE_WORKER') == '1':
        if gpu:
            policy.require_worker_lease(policy_root)
            args.device = 'cuda:0'
        return
    code = snapshot(project, args.output)
    # Config is frozen while the live project root continues to resolve data paths.
    arguments = list(argv)
    if '--config' in arguments:
        i = arguments.index('--config'); del arguments[i:i+2]
    arguments = [v for v in arguments if not v.startswith('--config=')]
    command = [sys.executable, str(code/'scripts/jace.py'), '--config', str(code/'configs/project.json')] + arguments
    env = dict(os.environ, JACE_PROJECT_ROOT=str(project.root), JACE_CODE_ROOT=str(code),
               JACE_RELEASE_WORKER='1', JACE_GPU_POLICY_ROOT=str(policy_root), OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads))
    if gpu:
        status = policy.run_command(policy_root, command, args.memory_mib, env, cwd=project.root)
    else:
        import subprocess
        env['CUDA_VISIBLE_DEVICES'] = ''
        status = subprocess.call(command, cwd=project.root, env=env)
    raise SystemExit(status)


def gpu_policy_root(root):
    """Reuse a configured account policy, including from a containing checkout."""
    if os.environ.get('JACE_GPU_POLICY_ROOT'):
        return Path(os.environ['JACE_GPU_POLICY_ROOT']).resolve()
    root = Path(root).resolve()
    for parent in root.parents:
        if (parent/'configs/gpu-account-policy.json').is_file():
            return parent
    return root
