"""Create an account policy only when no existing shared policy is configured."""
import argparse
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from jace_rec.release_execution import gpu_policy_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-gpus', type=int, required=True)
    args = parser.parse_args()
    if not 1 <= args.max_gpus <= 8:
        parser.error('Choose an account limit between 1 and 8 GPUs')
    root = gpu_policy_root(Path(__file__).resolve().parents[1])
    path = root/'configs/gpu-account-policy.json'
    if path.exists():
        print('Existing shared policy preserved:', path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(dict(account_uid=os.getuid(), max_gpu_count=args.max_gpus,
                       preferred_gpu_count=args.max_gpus, prefer_already_occupied=True,
                       hold_new_workers=False), stream, indent=2)
        stream.write('\n')
    print('Created account policy:', path)


if __name__ == '__main__':
    main()
