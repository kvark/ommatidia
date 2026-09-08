#!/usr/bin/env python3
"""Summarize the recorded two-seed support study; reject incomplete/mismatched arms."""
import argparse
import hashlib
import json
from pathlib import Path


def read(root, arm, seed):
    matches = list(root.rglob(f'{arm}-{seed}/quality.json'))
    if len(matches) != 1:
        raise ValueError(f'{arm}-{seed}: expected one completed report, found {len(matches)}')
    path = matches[0]
    report = json.loads(path.read_text())
    return path.parent, report


def field_row(arm, seed, directory, result, control_dir):
    for name in ('0-reference.png', '0-fit-reference.png', '0-context.json'):
        if (directory / name).read_bytes() != (control_dir / name).read_bytes():
            raise ValueError(f'unmatched {name}: {arm}-{seed}')
    context = json.loads((directory / '0-context.json').read_text())
    if set(context) != {'bounds', 'views'} or any(set(v) != {'camera', 'rgb'} for v in context['views']):
        raise ValueError('runtime contexts contain fields beyond RGB/camera/bounds')
    score = result['scores'][0]
    return dict(task='field', arm=arm, seed=seed, parameters=result['parameter_count'],
                psnr=score['learned']['compressed_psnr'], linear_mse=score['learned']['linear_mse'],
                log1p_mse=score['learned']['log1p_mse'], geometry=score['diagnostics']['geometry_held'])


def summarize(root):
    rows = []
    for seed in (7, 11):
        control_dir, _ = read(root, 'late-rgb', seed)
        for arm in ('late-rgb', 'visible-rgb'):
            directory, result = read(root, arm, seed)
            rows.append(field_row(arm, seed, directory, result, control_dir))
        _, control = read(root, 'selector-control', seed)
        for arm in ('selector-control', 'selector-projected', 'selector-linear'):
            directory, result = read(root, arm, seed)
            if result['baseline'] != control['baseline']:
                raise ValueError(f'unmatched deterministic baseline: {arm}-{seed}')
            rows.append(dict(task='selector', arm=arm, seed=seed, **result['learned']))
    control_dir, _ = read(root, 'late-rgb', 7)
    directory, result = read(root, 'visible-unsupervised', 7)
    rows.append(field_row('visible-unsupervised', 7, directory, result, control_dir))
    json.dumps(rows, allow_nan=False)
    groups = {}
    for task in ('field', 'selector'):
        for arm in sorted({r['arm'] for r in rows if r['task'] == task}):
            group = [r for r in rows if r['arm'] == arm]
            keys = set.intersection(*[set(r) for r in group]) - {'task', 'arm', 'seed'}
            numeric = {k for k in keys if all(isinstance(r[k], (int, float)) for r in group)}
            groups[arm] = {k: sum(r[k] for r in group) / len(group) for k in sorted(numeric)}
            groups[arm]['optimization_seeds'] = len(group)
    return {'role': 'development; field: one construction scene, not a fresh audit', 'rows': rows, 'means': groups}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    root = parser.parse_args().root
    summary = summarize(root)
    (root / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    for row in summary['rows']:
        print(json.dumps(row, sort_keys=True))
    (root / 'reports.sha256').write_text(''.join(
        f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(root)}\n'
        for p in sorted(root.rglob('quality.json'))))


if __name__ == '__main__':
    main()
