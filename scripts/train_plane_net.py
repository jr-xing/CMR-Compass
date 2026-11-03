"""Step 3. Train the plane network on all patients outside fold k, then predict the lines of fold k,
so every patient's estimated planes come from a model that never saw that patient.

python -m scripts.train_plane_net --manifest data/manifest.jsonl --cache work/cache \
    --pairing work/pairing.json --fold 0 --out work/plane_net            # repeat for folds 0..4
python -m scripts.train_plane_net ... --checkpoint work/plane_net/fold0/model.pt --out work/plane_net_ext
                                                                        # predict all scans, no training
Outputs (--out/fold{k}/): model.pt, lines.json ({scan_id: {view: [line per SAX slice]}}).
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from cmr_compass.data import load_cache, load_manifest
from cmr_compass.plane_net import HeatmapNet, SliceDataset, cache_affine, heat_loss, predict_scan


def load_scans(manifest, pairing, pos, fold_of):
    """Reference stacks with cache rows, cache-to-3D affine per slice, and targets."""
    tags = {r['sample_id']: r['dicom_tags'] for r in manifest}
    scans = []
    for sc in pairing:
        if sc['reference'] is None:
            continue
        slices = []
        for sl in sc['reference']['slices']:
            ok = [s for s in sl['sample_ids'] if s in pos]
            if ok:
                slices.append(dict(plane_id=sl['plane_id'], frames=[pos[s] for s in ok], **cache_affine(tags[ok[0]])))
        scans.append(dict(scan_id=sc['scan_id'], patient_id=sc['patient_id'], fold=fold_of.get(sc['patient_id']),
                          slices=slices, targets=sc['targets']))
    return scans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--cache', type=Path, required=True)
    ap.add_argument('--pairing', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--fold', type=int, default=None)
    ap.add_argument('--folds', type=Path, default=None, help='default: <cache>/folds.json')
    ap.add_argument('--checkpoint', type=Path, default=None, help='skip training; predict every scan')
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--no-pretrained', action='store_true')
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    images, masks, pos = load_cache(a.cache)
    fold_of = json.loads((a.folds or a.cache / 'folds.json').read_text())['fold_of'] if a.fold is not None else {}
    scans = load_scans(load_manifest(a.manifest), json.loads(a.pairing.read_text()), pos, fold_of)
    out = a.out / (f'fold{a.fold}' if a.fold is not None else 'all')
    out.mkdir(parents=True, exist_ok=True)

    if a.checkpoint:
        model = HeatmapNet(pretrained=False).to(device)
        model.load_state_dict(torch.load(a.checkpoint, map_location=device)['model'])
        test = scans
    else:
        train = [s for s in scans if s['fold'] != a.fold and s['targets']]
        test = [s for s in scans if s['fold'] == a.fold]
        assert not ({s['patient_id'] for s in train} & {s['patient_id'] for s in test})
        print(f'train scans {len(train)}, held-out scans {len(test)}', flush=True)
        dl = DataLoader(SliceDataset(train, images, masks, True), a.batch_size, shuffle=True,
                        num_workers=a.workers, drop_last=True, persistent_workers=a.workers > 0)
        model = HeatmapNet(pretrained=not a.no_pretrained).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * len(dl), pct_start=0.05)
        for ep in range(a.epochs):
            model.train()
            tot, n = 0.0, 0
            for x, H, has in dl:
                x, H, has = x.to(device), H.to(device), has.to(device)
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                    logits = model(x)
                loss = heat_loss(logits, H, has)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step(); sched.step()
                tot += loss.item() * len(x); n += len(x)
            if ep % 10 == 0 or ep == a.epochs - 1:
                print(json.dumps(dict(epoch=ep, train_loss=tot / n)), flush=True)
        torch.save(dict(model=model.state_dict(), args={k: str(v) for k, v in vars(a).items()}), out / 'model.pt')

    ds = SliceDataset(test, images, masks, False)
    lines = {sc['scan_id']: predict_scan(model, ds, sc, device) for sc in test}
    (out / 'lines.json').write_text(json.dumps(lines))
    print(f'lines for {len(lines)} scans -> {out}')


if __name__ == '__main__':
    main()
