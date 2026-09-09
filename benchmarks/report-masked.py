"""Verify all eight fixed-budget masked-selector artifacts (requires NumPy/Pillow)."""
import argparse, csv, json, math
from pathlib import Path
import numpy as np
from PIL import Image


def check(ok, text):
    if not ok:
        raise ValueError(text)


def read(path):
    d = json.loads(path.read_text())
    def finite(v):
        if isinstance(v, float):
            check(math.isfinite(v), f'nonfinite metric {path}')
        elif isinstance(v, list):
            for x in v:
                finite(x)
        elif isinstance(v, dict):
            for x in v.values():
                finite(x)
    finite(d)
    return d


def weights(logits, prior, mode):
    z = np.asarray(logits, np.float64).reshape(6, -1)
    p = np.asarray(prior, np.float64).reshape(z.shape)
    check(np.all(np.isfinite(z)) and np.all(p >= 0) and np.all(p.sum(0) > 0), 'invalid mixture inputs')
    if mode == 'softplus':
        m = np.maximum(np.logaddexp(z, 0), 1e-8)
    else:
        center = np.where(p > 0, z, -np.inf).max(0)
        gain = np.float32(np.log2(np.e)) * np.float32(.5)
        m = np.exp(np.where(p > 0, (z - center) * gain, -np.inf))
    w = p * m
    return w / w.sum(0)


def finite_checkpoint(path):
    import struct
    b = path.read_bytes()
    n = struct.unpack('<Q', b[:8])[0]
    header, payload = json.loads(b[8:8+n]), b[8+n:]
    for name, row in header.items():
        if name == '__metadata__':
            continue
        check(row['dtype'] in ('F32', 'F16'), f'unsupported diagnostic dtype {row["dtype"]}')
        a, z = row['data_offsets']
        arr = np.frombuffer(payload[a:z], '<f4' if row['dtype'] == 'F32' else '<f2')
        check(arr.size == math.prod(row['shape']) and np.isfinite(arr).all(), f'invalid tensor {name}')


def fitting(root):
    rows = []
    for frame in (0, 3):
        batches = []
        for mode in ('softplus', 'masked-softmax'):
            d = root / f'masked-fit-{mode}-{frame}' / f'{mode}-{frame}'
            q, b = read(d / 'quality.json'), read(d / 'frozen-batch.json')
            batches.append(b)
            check(q['frame'] == frame and q['task'] == 'selector-rgb', 'mismatched fit')
            check(q['result']['mixture'] == mode.replace('-', '_'), 'wrong mixture label')
            n = len(b['history']) // 6
            prior = np.asarray(b['prior']).reshape(6, 2*n)
            candidates = np.concatenate([np.asarray(b['spatial']).reshape(5, 6, n), np.asarray(b['history']).reshape(1, 6, n)])
            bound = q['result']['conditional_bound']
            for arm in q['result']['arms']:
                name = arm['name']
                p = read(d / name / 'prediction.json')
                w = weights(p['logits'], b['prior'], mode)
                actual = np.asarray(p['weights']).reshape(6, 2*n)
                error = float(abs(w-actual).max())
                check(error < 3e-6, f'mathematical weight mismatch {mode}/{frame}/{name}: {error}')
                check(np.all(actual[prior == 0] == 0), 'illegal candidate weight')
                check(np.max(abs(actual.sum(0)-1)) < 2e-6, 'lost normalized mass')
                lobes = (np.repeat(w.reshape(6, 2, n), 3, axis=1) * candidates).sum(0)
                check(np.max(abs(lobes.ravel()-p['lobes'])/(1+abs(lobes.ravel()))) < 2e-6, 'radiance mismatch')
                rgb = lobes[:3].ravel()*b['albedo'] + lobes[3:].ravel() + b['emission']
                loss = float(np.mean((rgb-b['target_rgb'])**2))
                check(abs(loss-arm['final_loss']) < 3e-6*(1+loss), 'incorrect final loss')
                check(arm['steps'] == 512 and arm['seed'] == 7 and loss >= bound-1e-6, 'bad budget/bound')
                check(len((d/name/'loss.csv').read_text().splitlines()) == 513, 'partial fitting history')
                finite_checkpoint(d/name/'diagnostic.safetensors')
                rows.append(dict(mode=mode, frame=frame, arm=name, loss=loss, bound=bound,
                    fraction_recovered=(arm['initial_loss']-loss)/(arm['initial_loss']-bound),
                    mathematical_weight_max_error=error,
                    near_zero_legal_weight=float(np.mean(actual[prior > 0] < 1e-6)),
                    max_row_weight_mean=float(actual.max(0).mean()),
                    logit_min=arm['logit_min'], logit_max=arm['logit_max'], psnr=arm['rgb']['compressed_psnr']))
        check(batches[0] == batches[1], 'frozen observations or truth differ between arms')
    return rows


def causal(root):
    rows = []
    for seed in (7, 11):
        pairs = []
        for mode in ('softplus', 'masked-softmax'):
            d = root / f'masked-causal-{mode}-{seed}' / f'{mode}-{seed}'
            q, meta, fit = read(d/'quality.json'), read(d/'training.json'), read(d/'fitting'/'quality.json')
            check(meta['steps'] == 512 and meta['seed'] == seed and meta['unroll'] == 2, 'wrong causal budget')
            check(q['learned']['frames'] == q['baseline']['frames'] == fit['learned']['frames'] == 24, 'incomplete frames')
            check(len((d/'loss.csv').read_text().splitlines()) == 513, 'partial training curve')
            for report, path in ((q, d/'frames.csv'), (fit, d/'fitting'/'frames.csv')):
                frames = list(csv.DictReader(path.open()))
                expected = {(s, f) for s in range(6) for f in range(4)}
                check(len(frames) == 24 and {(int(r['sequence']), int(r['frame'])) for r in frames} == expected, 'incomplete per-frame rows')
                for method in ('baseline', 'learned'):
                    value = sum(float(r[method+'_psnr']) for r in frames)/24
                    check(abs(value-report[method]['psnr']) < 1e-6, 'aggregate PSNR mismatch')
            check('evaluation reads mixture from the checkpoint sidecar' in (d/'admission.log').read_text(), 'missing negative admission evidence')
            side = (d/'model.transport.ron').read_text()
            check(('version: 2' if mode == 'masked-softmax' else 'version: 1') in side, 'wrong sidecar version')
            finite_checkpoint(d/'model.safetensors')
            pairs.append((d, q, meta))
            rows.append(dict(mode=mode, seed=seed, **q['learned']))
        check(pairs[0][1]['baseline'] == pairs[1][1]['baseline'], 'unequal deterministic baseline')
        for key in ('training', 'evaluation', 'loss_weights', 'learning_rate'):
            check(pairs[0][2][key] == pairs[1][2][key], f'unmatched {key}')
        a, b = pairs[0][0], pairs[1][0]
        for p in a.rglob('*.png'):
            if p.name.endswith('-base.png') or 'baseline' in p.name or 'reference' in p.name:
                check(p.read_bytes() == (b/p.relative_to(a)).read_bytes(), f'unmatched {p.name}')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    root = parser.parse_args().root
    results = {'fitting': fitting(root), 'causal': causal(root)}
    images = list(root.rglob('*.png'))
    for p in images:
        with Image.open(p) as image:
            image.load()
    results['decoded_images'] = len(images)
    (root/'summary.json').write_text(json.dumps(results, indent=2, allow_nan=False)+'\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
