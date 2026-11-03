"""Step 6. Inference: one checkpoint or an ensemble (probabilities averaged), with the coverage gate
whenever the geometry file holds valid coverage for an image. Series prediction = argmax of the mean
image probability.

python -m scripts.predict --manifest ext/manifest.jsonl --cache ext/cache --geometry ext/geometry.npz \
    --checkpoints work/cmr_compass_s0/fold*/best.pt --out ext/pred_series.csv     # five-fold ensemble
python -m scripts.predict ... --checkpoints work/cmr_compass_s0/fold0/best.pt --folds work/cache/folds.json \
    --fold 0 --out ...                                                            # held-out fold only

Output CSV: final (gated) prediction and probabilities, then fused and per-branch probabilities.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from cmr_compass import BRANCHES
from cmr_compass.data import (ViewDataset, geometry_input, geometry_stats, load_cache, load_geometry,
                              load_manifest, split_rows)
from cmr_compass.metrics import metrics, series_level, write_series_csv
from cmr_compass.model import coverage_gate, load_checkpoint, predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--cache', type=Path, required=True)
    ap.add_argument('--geometry', type=Path, default=None)
    ap.add_argument('--checkpoints', type=Path, nargs='+', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--folds', type=Path, default=None,
                    help='with --fold: predict only that fold\'s test patients; also used to recompute '
                         'geometry statistics for checkpoints that do not store them')
    ap.add_argument('--fold', type=int, default=None)
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    images, masks, pos = load_cache(a.cache)
    all_rows = [dict(r, _cache=pos[r['sample_id']]) for r in load_manifest(a.manifest) if r['sample_id'] in pos]
    folds = json.loads(a.folds.read_text()) if a.folds else None
    rows = split_rows(all_rows, folds, a.fold)['test'] if a.fold is not None else all_rows
    sample_ids = [None] * len(images)
    for s, i in pos.items():
        sample_ids[i] = s
    X, valid = load_geometry(a.geometry, sample_ids) if a.geometry else (None, None)

    probs = []
    for path in a.checkpoints:
        model, ckpt = load_checkpoint(path, device)
        geom = np.zeros((len(images), 32), np.float32)
        if 'geometric' in model.branches:
            stats = ckpt.get('geometry_stats')
            if stats is None:  # research checkpoints: recompute from the training patients of their fold
                train = split_rows(all_rows, folds, int(ckpt['args']['fold']))['train']
                stats = geometry_stats(X, valid, [r['_cache'] for r in train])
            geom = geometry_input(X, valid, *map(np.asarray, stats))
        dl = DataLoader(ViewDataset(rows, images, masks, geom, False), 256, num_workers=a.workers)
        probs.append(predict(model, dl, device))
    heads = ['fused'] + [b for b in BRANCHES if all(b in p for p in probs)]
    probs = {h: np.mean([p[h] for p in probs], 0) for h in heads}

    final = probs['fused']
    g = np.load(a.geometry) if a.geometry else None
    if 'geometric' in probs and g is not None and 'q' in g.files:
        q_of = {s: (q, v) for s, q, v in zip(g['sample_ids'].tolist(), g['q'], g['q_valid'])}
        q = np.array([q_of.get(r['sample_id'], (np.nan, False))[0] for r in rows], float)
        q_valid = np.array([q_of.get(r['sample_id'], (np.nan, False))[1] for r in rows], bool)
        final = coverage_gate(probs['fused'], probs['geometric'], q, q_valid)
        print(f'coverage gate active for {int(q_valid.sum())} / {len(rows)} images')
    series = series_level(rows, dict(probs, final=final))
    write_series_csv(a.out, series, heads)
    y = [s['label'] for s in series]
    for h in ['final'] + heads:
        m = metrics(y, [s['probs'][h].argmax() for s in series])
        print(f"{h:>10s}: series acc {m['acc']:.4f}  macro-F1 {m['macro_f1']:.4f}  (n={m['n']})")
    print('->', a.out)


if __name__ == '__main__':
    main()
