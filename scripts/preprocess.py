"""Step 1. Cache every image and its masks on a common grid, and assign patient-level folds.

python -m scripts.preprocess --manifest data/manifest.jsonl --out work/cache

Outputs (--out): images.npy float16 [N, 256, 256], masks.npy uint8 [N, 256, 256] (6-bit masks),
meta.json (sample_ids, failures), folds.json (written only if it does not exist yet).
"""
import argparse
import json
from pathlib import Path

import numpy as np

from cmr_compass.data import CACHE_SIZE, load_manifest, make_folds, preprocess_image, preprocess_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--folds', type=int, default=5)
    a = ap.parse_args()
    rows = load_manifest(a.manifest)
    a.out.mkdir(parents=True, exist_ok=True)

    images = np.lib.format.open_memmap(a.out / 'images.npy', 'w+', np.float16, (len(rows), CACHE_SIZE, CACHE_SIZE))
    masks = np.lib.format.open_memmap(a.out / 'masks.npy', 'w+', np.uint8, (len(rows), CACHE_SIZE, CACHE_SIZE))
    failures = {}
    for i, r in enumerate(rows):
        try:
            ps = r['dicom_tags']['PixelSpacing']
            img = np.load(r['image'])
            if img.ndim != 2:
                raise ValueError(f'expected a 2D image, got shape {img.shape}')
            images[i] = preprocess_image(img, ps)
            masks[i] = preprocess_mask(np.load(r['mask']), ps) if r.get('mask') else 0
        except Exception as e:  # recorded, never silently dropped
            failures[r['sample_id']] = f'{type(e).__name__}: {e}'
        if (i + 1) % 1000 == 0:
            print(f'{i + 1}/{len(rows)}', flush=True)
    images.flush()
    masks.flush()
    (a.out / 'meta.json').write_text(json.dumps(dict(sample_ids=[r['sample_id'] for r in rows], failures=failures), indent=1))
    if not (a.out / 'folds.json').exists():
        (a.out / 'folds.json').write_text(json.dumps(make_folds(rows, a.folds), indent=1))
    print(f'{len(rows)} images, {len(failures)} failures -> {a.out}')


if __name__ == '__main__':
    main()
