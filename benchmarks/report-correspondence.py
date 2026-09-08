#!/usr/bin/env python3
"""Validate complete paired correspondence / RGB construction experiments."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    result = json.loads(path.read_text())
    def finite(value):
        if isinstance(value, float):
            require(math.isfinite(value), f'nonfinite metric: {path}')
        elif isinstance(value, dict):
            for item in value.values():
                finite(item)
        elif isinstance(value, list):
            for item in value:
                finite(item)
    finite(result)
    return result


def matched(a, b, names):
    for name in names:
        require((a / name).read_bytes() == (b / name).read_bytes(), f'unmatched {name}: {a}, {b}')


def completed(directory, updates):
    with (directory / 'loss.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == updates, f'incomplete optimizer log: {directory}')
    require(all(math.isfinite(float(r['loss'])) for r in rows), f'nonfinite loss: {directory}')
    require((directory / 'model.safetensors').stat().st_size > 100, f'no weights: {directory}')


def verify(root):
    rows = []
    for seed in (7, 11):
        control = root / f'field-visible-{seed}'
        for arm in ('field-visible', 'field-stereo'):
            directory = root / f'{arm}-{seed}'
            d = read(directory / 'quality.json')
            completed(directory, 1024)
            require(d['seed'] == seed and d['steps'] == 1024, 'incorrect field run')
            require(d['rays'] == d['samples'] == 64 and d['image_rays_seen'] == 65536, 'unequal field rays')
            require(d['consistency_rays'] == d['incident_rays'] == 0, 'uncontrolled auxiliary rays')
            require(d['view_fusion'] == ('stereo-rgb' if arm == 'field-stereo' else 'visible-rgb'), 'wrong field variant')
            require(abs(d['surface_weight'] - 0.05) < 1e-7 and abs(d['visibility_weight'] - 0.05) < 1e-7, 'unequal supervision')
            require(d['sampling'] == 'stratified-fixed-intervals', 'wrong sampling')
            matched(control, directory, ('captures.sha256', '0-context.json', '0-reference.png', '0-fit-reference.png'))
            context = read(directory / '0-context.json')
            require(set(context) == {'bounds', 'views'}, 'extra runtime context fields')
            require(all(set(v) == {'rgb', 'camera'} for v in context['views']), 'truth in runtime views')
            c = read(directory / 'model.field.json')
            other = read(control / 'model.field.json')
            require({k: v for k, v in c.items() if k != 'view_fusion'} ==
                    {k: v for k, v in other.items() if k != 'view_fusion'}, 'unmatched model configuration')
            require(len(d['scores']) == 1, 'unexpected construction scene count')
            s = d['scores'][0]
            reference = read(control / 'quality.json')['scores'][0]
            require(s['scene_seed'] == reference['scene_seed'], 'unequal scene')
            # A zero-output matching head must not change the initial control.
            require(abs(s['untrained']['compressed_psnr'] - reference['untrained']['compressed_psnr']) < 1e-4, 'unmatched initial field')
            g = s['diagnostics']['geometry_held']
            rows.append(dict(task='field', arm=arm, seed=seed, psnr=s['learned']['compressed_psnr'],
                             linear_mse=s['learned']['linear_mse'], log1p_mse=s['learned']['log1p_mse'],
                             depth_mae=g['hit_depth_mae_world'], hit_accuracy=g['hit_miss_accuracy'],
                             fitting_psnr=s['diagnostics']['fitting_camera']['compressed_psnr'],
                             parameters=d['parameter_count']))
        control = root / f'selector-lobes-{seed}'
        for arm in ('selector-lobes', 'selector-rgb'):
            directory = root / f'{arm}-{seed}'
            completed(directory, 512)
            d = read(directory / 'quality.json')
            meta = read(directory / 'training.json')
            require(meta['steps'] == 512 and meta['seed'] == seed and meta['unroll'] == 2, 'unequal selector budget')
            require(meta['construction_noise'] and meta['fixed_exposure_loss'], 'wrong construction contract')
            require(meta['rgb_loss'] == (arm == 'selector-rgb'), 'wrong selector objective')
            require(meta['loss_weights'] == dict(compressed=0.0, physical=1.0, low_frequency=0.0, confidence=0.0, temporal=0.0), 'uncontrolled objective')
            require(meta['projected_weight'] is None, 'projected supervision must be disabled')
            require(len(meta['training']['captures']) == len(meta['evaluation']['captures']) == 3, 'incorrect noise split')
            matched(control, directory, ('captures.sha256', 'model.transport.ron'))
            baseline = read(control / 'quality.json')['baseline']
            require(d['baseline'] == baseline, 'unmatched independent causal baseline')
            require(d['role'] == 'same-scenes held-noise construction', 'incorrect evaluation role')
            for partition in ('', 'fitting'):
                report = read(directory / partition / 'quality.json')
                require(report['learned']['frames'] == report['baseline']['frames'] == 24, 'partial selector evaluation')
                require(report['learned']['temporal_frames'] == 18 and report['learned']['resets'] == 6, 'wrong sequence resets')
                pngs = sorted(p.name for p in (control / partition).glob('*reference.png'))
                require(len(pngs) == 24, 'incomplete reference images')
                matched(control / partition, directory / partition, pngs)
                images = sorted(p.name for p in (control / partition).glob('*-base.png'))
                require(len(images) == 24, 'incomplete baseline images')
                matched(control / partition, directory / partition, images)
            rows.append(dict(task='selector', arm=arm, seed=seed, **d['learned'],
                             fitting_psnr=read(directory / 'fitting/quality.json')['learned']['psnr']))
    means = {}
    for arm in sorted({r['arm'] for r in rows}):
        group = [r for r in rows if r['arm'] == arm]
        numeric = [k for k in group[0] if k not in ('task', 'arm', 'seed')]
        means[arm] = {k: fmean(r[k] for r in group) for k in numeric}
    return dict(role='construction only; no new-scene generalization or speed claim', rows=rows, means=means)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    result = verify(args.root)
    result['reports_sha256'] = {str(p.relative_to(args.root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sorted(args.root.rglob('quality.json'))}
    (args.root / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
