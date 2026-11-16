"""Live account GPU admission policy; standard library only, no CUDA initialization.

Call candidates() while holding admission_lock(), then spawn and register the
worker before releasing it. Existing training workers are never preempted here.
"""
from contextlib import contextmanager
import csv
import json
import os
from pathlib import Path
import subprocess
import time
import fcntl


def policy(root):
    path = Path(root) / 'configs/gpu-account-policy.json'
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    assert 1 <= value['preferred_gpu_count'] <= value['max_gpu_count'] <= 8
    assert value['account_uid'] == os.getuid(), 'GPU policy belongs to another account'
    return value


def process_identity(pid):
    path = Path('/proc') / str(pid)
    try:
        fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z':
            return None
        return path.stat().st_uid, fields[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def occupied_gpus(root):
    """Count actual account CUDA contexts AND live workers awaiting allocation."""
    raw = subprocess.check_output([
        'nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'
    ], text=True)
    occupied = set()
    for uuid, pid in csv.reader(raw.splitlines()):
        identity = process_identity(int(pid.strip()))
        if identity and identity[0] == os.getuid():
            occupied.add(uuid.strip())
    for path in (Path(root) / 'status/checkpoint_gpu_leases').glob('GPU-*.json'):
        row = json.loads(path.read_text())
        for entry in row.get('members', [row]):
            identity = process_identity(entry['pid'])
            if identity == (os.getuid(), str(entry['process_start_ticks'])):
                occupied.add(entry.get('gpu', row['gpu'])['uuid'])
    return occupied


def choose(eligible, occupied, settings):
    """Prefer consolidation and read the account limit from live settings."""
    if settings is None:
        return list(eligible)
    if settings.get('hold_new_workers', False):
        return []
    existing = [g for g in eligible if g['uuid'] in occupied]
    if existing:
        return existing
    if len(occupied) >= settings['max_gpu_count']:
        return []
    return list(eligible)


def candidates(root, eligible):
    try:
        settings = policy(root)
        if settings is None:
            return list(eligible)
        occupied = occupied_gpus(root)
        allowed = choose(eligible, occupied, settings)
        decision = dict(at=time.time(), occupied=sorted(occupied),
                        allowed=[g['uuid'] for g in allowed], settings=settings)
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, AssertionError) as exc:
        # A failed NVML query must not unwind a controller and kill its workers.
        allowed = []
        decision = dict(at=time.time(), allowed=[], error=str(exc), state='WAITING_FOR_GPU_ACCOUNT_QUERY')
    directory = Path(root)/'status/gpu_account_policy'
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/f'last_admission_{os.getpid()}.json'
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(decision, indent=2)+'\n')
    temp.replace(path)
    return allowed


@contextmanager
def admission_lock(root):
    directory = Path(root) / 'status/checkpoint_gpu_leases'
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'admission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def register(root, gpu, pid, memory_mib, task):
    identity = process_identity(pid)
    assert identity and identity[0] == os.getuid(), 'Cannot register a foreign/dead worker'
    path = Path(root) / 'status/checkpoint_gpu_leases' / f'{gpu["uuid"]}.{pid}.json'
    value = dict(pid=pid, process_start_ticks=identity[1], gpu=gpu,
                 memory_reservation_mib=memory_mib, task=task, stage='account_policy',
                 supervisor_pid=os.getpid(), at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    temp = path.with_suffix(f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def available_memory(root, devices):
    """Subtract unallocated portions of live reservations, without double counting."""
    raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
                                   '--format=csv,noheader,nounits'], text=True, timeout=20)
    actual = {}
    for uuid, pid, memory in csv.reader(raw.splitlines()):
        key = (uuid.strip(), int(pid.strip()))
        actual[key] = actual.get(key, 0) + int(memory.strip())
    reserved = {}
    for path in (Path(root)/'status/checkpoint_gpu_leases').glob('GPU-*.json'):
        row = json.loads(path.read_text())
        for member in row.get('members', [row]):
            if process_identity(member['pid']) != (os.getuid(), str(member['process_start_ticks'])):
                continue
            key = (member.get('gpu', row['gpu'])['uuid'], member['pid'])
            reserved[key] = max(reserved.get(key, 0), int(member.get('memory_reservation_mib', actual.get(key) or next((g['free_mib'] for g in devices if g['uuid']==key[0]), 0))))
    pending = {}
    for (uuid, pid), amount in reserved.items():
        pending[uuid] = pending.get(uuid, 0) + max(0, amount-actual.get((uuid, pid), 0))
    return [dict(g, free_mib=max(0, g['free_mib']-pending.get(g['uuid'], 0)),
                 pending_reservation_mib=pending.get(g['uuid'], 0)) for g in devices]


def inventory():
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free,utilization.gpu',
                                   '--format=csv,noheader,nounits'], text=True, timeout=20)
    result = [dict(index=int(i), uuid=u.strip(), free_mib=int(m), utilization=int(v))
              for i,u,m,v in csv.reader(raw.splitlines())]
    return sorted(result, key=lambda g:(g['utilization'], -g['free_mib']))


def require_worker_lease(root):
    """Worker waits for its parent's admission lock, then verifies its own lease."""
    uuid = os.environ.get('JACE_ADMITTED_GPU')
    with admission_lock(root):
        path = Path(root)/'status/checkpoint_gpu_leases'/f'{uuid}.{os.getpid()}.json'
        row = json.loads(path.read_text())
        if (row['gpu']['uuid'] != uuid or os.environ.get('CUDA_VISIBLE_DEVICES') != uuid
                or process_identity(row['pid']) != (os.getuid(), str(row['process_start_ticks']))
                or row['pid'] != os.getpid()):
            raise RuntimeError('Missing or stale account GPU admission')


def run_command(root, command, memory_mib, env=None, poll_seconds=5, timeout_seconds=172800, cwd=None):
    """Start one registered worker; query failures defer admission only."""
    if memory_mib <= 0:
        raise ValueError('A positive GPU reservation is required')
    if policy(root) is None:
        raise RuntimeError('Restore configs/gpu-account-policy.json before GPU execution')
    began = time.monotonic()
    while time.monotonic()-began < timeout_seconds:
        with admission_lock(root):
            try:
                devices = available_memory(root, inventory())
                allowed = candidates(root, [g for g in devices if g['free_mib'] >= memory_mib])
            except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
                allowed = []
                print(f'WAITING_FOR_GPU_QUERY: {error}', flush=True)
            if allowed:
                gpu = allowed[0]
                child_env = dict(os.environ if env is None else env,
                                 CUDA_VISIBLE_DEVICES=gpu['uuid'], JACE_ADMITTED_GPU=gpu['uuid'])
                process = subprocess.Popen(command, cwd=cwd or root, env=child_env)
                try:
                    register(root, gpu, process.pid, memory_mib, 'current-release')
                except BaseException:
                    process.terminate()
                    process.wait()
                    raise
                break
        time.sleep(poll_seconds)
    else:
        raise TimeoutError('Timed out waiting for account GPU admission')
    return process.wait()
