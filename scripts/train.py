"""Step 5. Train CMR-Compass on one fold (test = fold k, val = fold k+1, train = rest).

python -m scripts.train --manifest data/manifest.jsonl --cache work/cache --geometry work/geometry.npz \
    --fold 0 --seed 0 --out work/cmr_compass_s0                       # repeat for folds 0..4, seeds 0..2
python -m scripts.train ... --branches visual --out work/visual_only   # single-branch ablation

Loss = CE(fused) + 0.3 * sum_b CE(branch b), label smoothing 0.05; AdamW, one warm-up epoch, cosine decay.
The checkpoint with the best validation series macro-F1 is kept (best.pt, including the geometry
normalization statistics of the training fold); its validation/test series predictions are written.
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from cmr_compass import BRANCHES
from cmr_compass.data import (ViewDataset, geometry_input, geometry_stats, load_cache, load_geometry,
                              load_manifest, sampler_weights, split_rows)
from cmr_compass.metrics import metrics, series_level, write_series_csv
from cmr_compass.model import CMRCompass, predict


def evaluate(model, loader, rows, device):
    probs = predict(model, loader, device)
    series = series_level(rows, dict(probs, final=probs['fused']))
    return series, metrics([s['label'] for s in series], [s['probs']['final'].argmax() for s in series])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--cache', type=Path, required=True)
    ap.add_argument('--geometry', type=Path, default=None, help='needed with the geometric branch')
    ap.add_argument('--folds', type=Path, default=None, help='default: <cache>/folds.json')
    ap.add_argument('--fold', type=int, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--branches', nargs='+', choices=BRANCHES, default=BRANCHES)
    ap.add_argument('--aux-weight', type=float, default=0.3)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--label-smoothing', type=float, default=0.05)
    ap.add_argument('--no-pretrained', action='store_true')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out = a.out / f'fold{a.fold}'
    out.mkdir(parents=True, exist_ok=True)

    # ---- data
    images, masks, pos = load_cache(a.cache)
    rows = [dict(r, _cache=pos[r['sample_id']]) for r in load_manifest(a.manifest) if r['sample_id'] in pos]
    parts = split_rows(rows, json.loads((a.folds or a.cache / 'folds.json').read_text()), a.fold)
    print({k: len(v) for k, v in parts.items()}, flush=True)
    sample_ids = [None] * len(images)
    for s, i in pos.items():
        sample_ids[i] = s
    stats, geom = None, np.zeros((len(images), 32), np.float32)
    if 'geometric' in a.branches:
        X, valid = load_geometry(a.geometry, sample_ids)
        stats = geometry_stats(X, valid, [r['_cache'] for r in parts['train']])
        geom = geometry_input(X, valid, *stats)

    def loader(part, train):
        ds = ViewDataset(parts[part], images, masks, geom, train)
        if not train:
            return DataLoader(ds, 256, num_workers=a.workers, pin_memory=True)
        sampler = WeightedRandomSampler(sampler_weights(parts[part]), num_samples=len(ds), replacement=True)
        return DataLoader(ds, a.batch_size, sampler=sampler, num_workers=a.workers, pin_memory=True,
                          drop_last=True, persistent_workers=a.workers > 0)
    train_dl, val_dl, test_dl = loader('train', True), loader('val', False), loader('test', False)

    # ---- model
    model = CMRCompass(a.branches, geom.shape[1], not a.no_pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    steps, warm = a.epochs * len(train_dl), len(train_dl)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda t: (t + 1) / warm if t < warm else 0.5 * (1 + math.cos(math.pi * (t - warm) / max(1, steps - warm))))
    crit = nn.CrossEntropyLoss(label_smoothing=a.label_smoothing)

    history, best = [], -1.0
    for ep in range(a.epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for x, m, g, y, _ in train_dl:
            x, m, g, y = x.to(device, non_blocking=True), m.to(device), g.to(device), y.to(device)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                fused, aux = model(x, m, g)
                loss = crit(fused.float(), y)
                if model.head is not None:
                    loss = loss + a.aux_weight * sum(crit(l.float(), y) for l in aux.values())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step(); sched.step()
            tot += loss.item() * len(y); n += len(y)
        _, vm = evaluate(model, val_dl, parts['val'], device)
        history.append(dict(epoch=ep, train_loss=tot / n, val_series_macro_f1=vm['macro_f1'],
                            val_series_acc=vm['acc'], sec=time.time() - t0))
        print(json.dumps(history[-1]), flush=True)
        if vm['macro_f1'] > best:
            best = vm['macro_f1']
            torch.save(dict(model=model.state_dict(), epoch=ep, fold=a.fold, branches=a.branches,
                            geometry_stats=None if stats is None else [s.tolist() for s in stats],
                            args={k: str(v) for k, v in vars(a).items()}), out / 'best.pt')
    (out / 'history.json').write_text(json.dumps(history, indent=1))

    # ---- validation and test predictions of the selected checkpoint
    model.load_state_dict(torch.load(out / 'best.pt', map_location=device, weights_only=False)['model'])
    res = {}
    for part, dl in (('val', val_dl), ('test', test_dl)):
        series, res[part] = evaluate(model, dl, parts[part], device)
        write_series_csv(out / f'pred_series_{part}.csv', series, ['fused'] + a.branches)
    (out / 'metrics.json').write_text(json.dumps(res, indent=1))
    print('TEST series:', {k: res['test'][k] for k in ('n', 'acc', 'macro_f1')})


if __name__ == '__main__':
    main()
