#!/usr/bin/env python3
"""Verify the fixed geometry/cross-noise study and summarize all completed arms."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean

METHODS = ('learned', 'fixed-prior', 'observable-risk', 'cross-noise-risk', 'single-oracle', 'convex-oracle')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load(path):
    data = json.loads(path.read_text())
    def finite(value):
        if isinstance(value, float):
            require(math.isfinite(value), f'nonfinite metric in {path}')
        elif isinstance(value, dict):
            for v in value.values():
                finite(v)
        elif isinstance(value, list):
            for v in value:
                finite(v)
    finite(data)
    return data


def fields(root):
    rows = []
    for seed in (7, 11):
        control = root / f'field-control-{seed}'
        for arm in ('field-control', 'field-consistent'):
            directory = root / f'{arm}-{seed}'
            report = load(directory / 'quality.json')
            require(report['seed'] == seed and report['steps'] == 1024, 'incomplete/mismatched optimizer budget')
            require(report['rays'] == report['samples'] == 64 and report['image_rays_seen'] == 65536, 'unmatched ray budget')
            require(report['consistency_rays'] == 8 and report['view_fusion'] == 'visible-rgb', 'unmatched field variant')
            require(abs(report['consistency_weight'] - (0.1 if arm == 'field-consistent' else 0)) < 1e-7, 'incorrect arm')
            require(len((directory / 'loss.csv').read_text().splitlines()) == 1025, 'incomplete loss history')
            for name in ('0-reference.png', '0-fit-reference.png', '0-context.json', 'model.field.json'):
                require((directory / name).read_bytes() == (control / name).read_bytes(), f'unmatched {name}')
            for digest in ('ce16aa43e4d687f7b4775d388bd10abf68d22a199045454ac10b81535f7e57eb',
                           'c2f79ec8d3f8b1873dde970155561824d436342744d4b08848f6aaf7797da63a'):
                require(digest in (directory / 'recipe.txt').read_text(), 'wrong capture hash')
            context = load(directory / '0-context.json')
            require(set(context) == {'bounds', 'views'} and all(set(v) == {'camera', 'rgb'} for v in context['views']), 'truth in runtime context')
            require(len(report['scores']) == 1, 'unexpected construction scene count')
            score = report['scores'][0]
            check = score['source_consistency']
            distances = []
            for row in check['distributions']:
                a, b = row['source_mass'], row['volume_mass']
                require(len(a) == len(b) == 17, 'incorrect categorical layout')
                for mass in (a, b):
                    require(min(mass) >= -1e-6 and abs(sum(mass) - 1) < 1e-4, 'non-normalized distribution')
                ca = cb = error = 0.0
                for x, y in zip(a[:-1], b[:-1]):
                    ca += x
                    cb += y
                    error += (ca - cb) ** 2
                distances.append(error / 16)
            require(len(distances) == check['rays'] == 512, 'incomplete consistency sample')
            require(abs(fmean(distances) - check['cdf_mse']) < 1e-9, 'CDF metric disagrees with distributions')
            geometry = score['diagnostics']['geometry_held']
            rows.append(dict(arm=arm, seed=seed, psnr=score['learned']['compressed_psnr'],
                             linear_mse=score['learned']['linear_mse'], depth_mae=geometry['hit_depth_mae_world'],
                             hit_accuracy=geometry['hit_miss_accuracy'], cdf_mse=check['cdf_mse'],
                             fitting_psnr=score['diagnostics']['fitting_camera']['compressed_psnr']))
    means = {arm: {k: fmean(r[k] for r in rows if r['arm'] == arm) for k in rows[0] if k not in ('arm', 'seed')}
             for arm in ('field-control', 'field-consistent')}
    return {'rows': rows, 'means': means}


def noise(path):
    data = load(path)
    require(data['risk_target'] == 'linear_lobe_mse', 'obsolete log-risk diagnostic')
    require(data['fit_realizations'] == data['held_realizations'] == 3, 'incorrect noise split')
    require(len(data['results']) == 2, 'duplicate modes')
    require({r['mode'] for r in data['results']} == {'reset', 'causal'}, 'incomplete modes')
    rows = []
    for result in data['results']:
        require(len(result['frames']) == 144, 'incomplete frame/method coverage')
        require(result['native_recomposition_max_relative_difference'] < 1e-5, 'native RGB mismatch')
        require(set(result['lobe_mse']) == set(METHODS), 'incomplete methods')
        for method in METHODS:
            frames = [f for f in result['frames'] if f['method'] == method]
            require(len(frames) == 24 and {(f['stream'], f['frame']) for f in frames} ==
                    {(s, f) for s in (3, 4, 5) for f in range(8)}, 'missing/duplicate held frames')
            row = dict(mode=result['mode'], method=method, lobe_mse=result['lobe_mse'][method])
            row.update({k: fmean(f[k] for f in frames) for k in ('psnr', 'energy_ratio', 'linear_mse', 'low_frequency_psnr', 'detail_ratio')})
            rows.append(row)
        for bias, variance, mse in result['spatial_bias2_population_variance_mse']:
            require(abs(bias + variance - mse) < 1e-7 * (1 + mse), 'bias/variance decomposition mismatch')
    return {'rows': rows, 'note': 'Fixed-prior and diagnostic selectors reuse learned history; only learned is causally rolled out.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('field_root', type=Path)
    parser.add_argument('noise_report', type=Path)
    parser.add_argument('--out', type=Path, default=Path('geometry-summary.json'))
    args = parser.parse_args()
    summary = {'role': 'construction diagnostics, not unseen-scene or production-quality claims',
               'field': fields(args.field_root), 'noise': noise(args.noise_report)}
    summary['reports_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in [*sorted(args.field_root.glob('*/quality.json')), args.noise_report]}
    args.out.write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
