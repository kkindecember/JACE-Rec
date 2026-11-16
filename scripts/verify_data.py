"""Verify the released split and clean training prefixes against source sequences."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from jace_rec.io import Project, read, rows


def check(project):
    data = project.path('data')
    catalog = read(project.asset('catalog'))
    item_index = {item: i for i, item in enumerate(catalog['items']) if i}
    raw = {}
    with (data/'raw/user_sequence.txt').open() as stream:
        for line in stream:
            user, *items = line.split()
            if user in raw:
                raise ValueError('Duplicate source user')
            raw[user] = [item_index[item] for item in items]
    clean = {row['user_id']: row['history'] for row in rows(data/'upstream_sequences.jsonl')}
    assert set(clean) == set(raw), 'User set changed'
    removed = {user for user in clean if clean[user] != raw[user][:-2]}
    assert all(clean[user] == raw[user][:-3] for user in removed)
    pool = read(data/'train_samples.json')
    assert len({s['user_id'] for s in pool}) == len(pool)
    for sample in pool:
        user = sample['user_id']
        assert user in removed and sample['full_history'] == clean[user]
        assert sample['target'] == raw[user][-3]
        assert sample['history'] == sample['full_history'][-20:]
    split = read(data/'split.json')
    fit, calibration = split['fit_indices'], split['calibration_indices']
    assert sorted(fit+calibration) == list(range(len(pool)))
    assert [pool[i] for i in calibration] == read(data/'calibration_samples.json')
    for role, offset in [('validation', 2), ('test', 1)]:
        samples = read(data/f'{role}_samples.json')
        assert {s['user_id'] for s in samples} == set(raw)
        for sample in samples:
            sequence = raw[sample['user_id']]
            assert sample['full_history'] == sequence[:-offset]
            assert sample['target'] == sequence[-offset]
            assert sample['history'] == sample['full_history'][-20:]
    expected = Counter((user, tuple(seq[max(0, i-20):i]), seq[i])
                       for user, seq in clean.items() for i in range(1, len(seq)))
    actual = Counter((r['user_id'], tuple(r['history']), r['target'])
                     for r in rows(data/'gram_train.jsonl'))
    assert actual == expected, 'Clean GRAM training prefixes differ'
    return dict(domain=project.domain, users=len(raw), heldout_events=len(removed),
                gate_fit_users=len(fit), calibration_users=len(calibration),
                clean_training_prefixes=sum(actual.values()), state='PASSED')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/current-toys-2023.json'))
    args = parser.parse_args()
    print(json.dumps(check(Project(args.config)), indent=2))


if __name__ == '__main__':
    main()
