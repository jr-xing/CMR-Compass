"""Manifest, patient-level folds and image/mask preprocessing.

manifest.jsonl has one row per 2D image (see README). Images and masks are cached once at
1.25 mm in-plane spacing, center-cropped/padded to 256 px; models see a 224 px crop.
"""
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import zoom
from torch.utils.data import Dataset

from . import CLASSES

SPACING_MM, CACHE_SIZE, CROP = 1.25, 256, 224
MASK_CLASSES = ['LV_endo', 'LV_epi', 'LA_endo', 'RV_endo', 'RA_endo', 'RVOT_endo']  # bit k = class k


def load_manifest(path):
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    for r in rows:
        r['label'] = CLASSES.index(r['view'])
    return rows


# ----------------------------------------------------------------------------- folds
def make_folds(rows, k=5, seed=0):
    """Patient-level folds, stratified by dataset + modality coverage and balanced by series count.
    For fold j: test = j, val = (j + 1) % k, train = the rest."""
    series, mods, dataset = defaultdict(set), defaultdict(set), {}
    for r in rows:
        series[r['patient_id']].add(r['series_uid'])
        mods[r['patient_id']].add(r['modality'])
        dataset[r['patient_id']] = r.get('dataset', '')
    strata = defaultdict(list)
    for p in series:
        strata[f"{dataset[p]}|{'+'.join(sorted(mods[p]))}"].append(p)
    rng = random.Random(seed)
    fold_of, fold_series = {}, [0] * k
    for name in sorted(strata):
        pts = sorted(strata[name])
        rng.shuffle(pts)
        pts.sort(key=lambda p: -len(series[p]))  # big patients first, ties stay shuffled
        in_stratum = [0] * k
        for p in pts:
            f = min(range(k), key=lambda i: (in_stratum[i], fold_series[i], i))
            fold_of[p] = f
            in_stratum[f] += 1
            fold_series[f] += len(series[p])
    folds = []
    for j in range(k):
        part = lambda f: 'test' if f == j else 'val' if f == (j + 1) % k else 'train'
        folds.append({f'{n}_patients': sorted(p for p, f in fold_of.items() if part(f) == n)
                      for n in ('train', 'val', 'test')})
    return dict(fold_of=fold_of, folds=folds)


def split_rows(rows, folds, fold):
    """{'train'|'val'|'test': rows} for one fold."""
    part_of = {p: name for name in ('train', 'val', 'test') for p in folds['folds'][fold][f'{name}_patients']}
    parts = defaultdict(list)
    for r in rows:
        parts[part_of[r['patient_id']]].append(r)
    assert not ({r['patient_id'] for r in parts['train']} & {r['patient_id'] for r in parts['test']})
    return parts


# ----------------------------------------------------------------------------- preprocessing
def center_crop_pad(img, size=CACHE_SIZE):
    out = np.zeros((size, size), img.dtype)
    h, w = img.shape
    sy, sx = max(0, (h - size) // 2), max(0, (w - size) // 2)
    crop = img[sy:sy + size, sx:sx + size]
    dy, dx = (size - crop.shape[0]) // 2, (size - crop.shape[1]) // 2
    out[dy:dy + crop.shape[0], dx:dx + crop.shape[1]] = crop
    return out


def preprocess_image(img, pixel_spacing):
    """Native pixels -> 1.25 mm (bilinear) -> 256 px -> percentile (1, 99.5) of nonzero pixels -> [0, 1]."""
    ps = [float(v) for v in pixel_spacing]  # [row, column] spacing in mm
    img = zoom(np.asarray(img, np.float32), (ps[0] / SPACING_MM, ps[1] / SPACING_MM), order=1)
    img = center_crop_pad(img)
    lo, hi = np.percentile(img[img != 0] if (img != 0).any() else img, [1, 99.5])
    return np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float16)


def preprocess_mask(bitmask, pixel_spacing):
    """Native uint8 bitmask -> same grid as preprocess_image (nearest neighbour)."""
    ps = [float(v) for v in pixel_spacing]
    m = zoom(np.asarray(bitmask, np.uint8), (ps[0] / SPACING_MM, ps[1] / SPACING_MM), order=0)
    return center_crop_pad(m)


def load_cache(cache):
    """images.npy, masks.npy (memory-mapped) and {sample_id: row index}, without failed samples."""
    cache = Path(cache)
    meta = json.loads((cache / 'meta.json').read_text())
    pos = {s: i for i, s in enumerate(meta['sample_ids']) if s not in meta['failures']}
    return np.load(cache / 'images.npy', mmap_mode='r'), np.load(cache / 'masks.npy', mmap_mode='r'), pos


# ----------------------------------------------------------------------------- geometric input
def geometry_stats(X, valid, train_idx):
    """Per-feature mean/std over the training images where the feature is available."""
    tr = np.zeros(len(X), bool)
    tr[train_idx] = True
    mu, sd = np.zeros(X.shape[1], np.float32), np.ones(X.shape[1], np.float32)
    for j in range(X.shape[1]):
        ok = valid[:, j] & tr
        if ok.sum() > 1:
            mu[j], sd[j] = X[ok, j].mean(), max(X[ok, j].std(), 1e-6)
    return mu, sd


def load_geometry(path, sample_ids):
    """Raw descriptor X [N, 16] and availability [N, 16], aligned with sample_ids (missing = unavailable)."""
    d = np.load(path, allow_pickle=True)
    pos = {s: i for i, s in enumerate(d['sample_ids'].tolist())}
    X = np.zeros((len(sample_ids), d['X'].shape[1]), np.float32)
    valid = np.zeros(X.shape, bool)
    for i, s in enumerate(sample_ids):
        if s in pos:
            X[i], valid[i] = d['X'][pos[s]], d['valid'][pos[s]]
    return X, valid & np.isfinite(X)


def geometry_input(X, valid, mu, sd):
    """z-scored features (0 where unavailable) followed by the 16 availability flags -> [N, 32]."""
    Z = np.where(valid, (np.nan_to_num(X) - mu) / sd, 0.0).astype(np.float32)
    return np.concatenate([Z, valid.astype(np.float32)], 1)


# ----------------------------------------------------------------------------- torch dataset
class ViewDataset(Dataset):
    """Rows of one fold part. Training: shared random affine + crop for image and masks, then gamma and
    contrast/brightness on the image. Evaluation: center crop."""

    def __init__(self, rows, images, masks, geom, train):
        self.rows, self.images, self.masks, self.geom, self.train = rows, images, masks, geom, train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        c = self.rows[i]['_cache']
        x = torch.from_numpy(np.asarray(self.images[c], dtype=np.float32))[None]
        m = torch.from_numpy(unpack_mask(self.masks[c]))
        x, m = spatial(x, m, self.train)
        if self.train:
            x = x.clamp(0, 1) ** math.exp(random.uniform(-0.3, 0.3))
            x = x * random.uniform(0.9, 1.1) + random.uniform(-0.05, 0.05)
        x = (x - 0.449) / 0.226  # grayscale ImageNet statistics
        return x, m, torch.from_numpy(self.geom[c]), self.rows[i]['label'], i


def spatial(x, m, train, size=CROP):
    _, h, w = x.shape
    if not train:
        t, l = (h - size) // 2, (w - size) // 2
        return x[:, t:t + size, l:l + size], m[:, t:t + size, l:l + size]
    ang = math.radians(random.uniform(-15, 15))
    s = random.uniform(0.9, 1.1)
    tx, ty = random.uniform(-0.06, 0.06), random.uniform(-0.06, 0.06)
    theta = torch.tensor([[math.cos(ang) / s, -math.sin(ang) / s, tx],
                          [math.sin(ang) / s, math.cos(ang) / s, ty]], dtype=torch.float32)
    grid = F.affine_grid(theta[None], (1, 1, h, w), align_corners=False)
    x = F.grid_sample(x[None], grid, mode='bilinear', padding_mode='zeros', align_corners=False)[0]
    m = F.grid_sample(m[None], grid, mode='nearest', padding_mode='zeros', align_corners=False)[0]
    t, l = random.randint(0, h - size), random.randint(0, w - size)
    return x[:, t:t + size, l:l + size], m[:, t:t + size, l:l + size]


def sampler_weights(rows):
    """Each series counts once, then each label (view) has equal total weight."""
    series_size, label_series = defaultdict(int), defaultdict(set)
    for r in rows:
        series_size[r['series_uid']] += 1
        label_series[r['label']].add(r['series_uid'])
    return [1.0 / series_size[r['series_uid']] / len(label_series[r['label']]) for r in rows]


def unpack_mask(bitmask):
    """uint8 bitmask [..., H, W] -> binary channels [..., 6, H, W]."""
    bm = np.asarray(bitmask)
    return np.stack([(bm >> k) & 1 for k in range(len(MASK_CLASSES))], -3).astype(np.float32)
