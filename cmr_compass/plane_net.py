"""Plane network: predicts where the standard 2CH/3CH/4CH planes cut each SAX slice.

Input = SAX image + its 6 predicted masks; output = one line heatmap per long-axis view.
Targets come only from DICOM geometry: a Gaussian of each pixel's distance to the patient's
labelled on-axis long-axis plane. Predicted lines are mapped to 3D and fitted into one plane
per view by geometry.fit_plane.
"""
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset

from .data import CACHE_SIZE, MASK_CLASSES, SPACING_MM
from .geometry import LAX

SIGMA_MM = 4.0  # width of the target line


def cache_affine(tags):
    """3D position of cache pixel (row r, col c): P = o + er * r + ec * c (same resample + crop as the cache)."""
    iop = np.asarray(tags['ImageOrientationPatient'], float)
    x, y = iop[:3] / np.linalg.norm(iop[:3]), iop[3:] / np.linalg.norm(iop[3:])
    ipp = np.asarray(tags['ImagePositionPatient'], float)
    ps = [float(v) for v in tags['PixelSpacing']]
    h, w = int(tags['Rows']), int(tags['Columns'])
    hz, wz = int(round(h * ps[0] / SPACING_MM)), int(round(w * ps[1] / SPACING_MM))  # scipy zoom output shape
    kr, kc = (h - 1) / max(hz - 1, 1), (w - 1) / max(wz - 1, 1)                      # zoomed index -> native index
    sy, sx = max(0, (hz - CACHE_SIZE) // 2), max(0, (wz - CACHE_SIZE) // 2)
    dy, dx = (CACHE_SIZE - min(hz, CACHE_SIZE)) // 2, (CACHE_SIZE - min(wz, CACHE_SIZE)) // 2
    er, ec = y * ps[0] * kr, x * ps[1] * kc
    return dict(o=ipp + er * (sy - dy) + ec * (sx - dx), er=er, ec=ec)


def target_heatmaps(sl, targets, size=CACHE_SIZE):
    """[3, size, size] heatmaps exp(-d^2 / 2 sigma^2), d = distance (mm) of each pixel to the labelled plane;
    has[k] = 0 when view k has no labelled plane (ignored in the loss)."""
    r, c = np.mgrid[0:size, 0:size].astype(np.float32)
    P = sl['o'][None, None] + r[..., None] * sl['er'] + c[..., None] * sl['ec']
    H, has = np.zeros((len(LAX), size, size), np.float32), np.zeros(len(LAX), np.float32)
    for k, v in enumerate(LAX):
        for t in targets.get(v, []):
            d = (P - np.asarray(t['center'])) @ np.asarray(t['normal'])
            H[k] = np.maximum(H[k], np.exp(-d ** 2 / (2 * SIGMA_MM ** 2)))
            has[k] = 1
    return H, has


class SliceDataset(Dataset):
    """One item per reference SAX slice; training samples a random frame and augments image, masks and targets."""

    def __init__(self, scans, images, masks, train, rot=30.0):
        self.items = [(sc, sl) for sc in scans for sl in sc['slices']]
        self.images, self.masks, self.train, self.rot = images, masks, train, rot

    def __len__(self):
        return len(self.items)

    def inputs(self, idx):
        x = torch.from_numpy(np.asarray(self.images[idx], dtype=np.float32))[None]
        bm = torch.from_numpy(np.asarray(self.masks[idx]).astype(np.int64))
        return torch.cat([x, torch.stack([(bm >> k) & 1 for k in range(len(MASK_CLASSES))]).float()])

    def __getitem__(self, i):
        sc, sl = self.items[i]
        x = self.inputs(random.choice(sl['frames']) if self.train else sl['frames'][0])
        H, has = target_heatmaps(sl, sc['targets'])
        H = torch.from_numpy(H)
        if self.train:
            ang, s = math.radians(random.uniform(-self.rot, self.rot)), random.uniform(0.85, 1.2)
            tx, ty = random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1)
            theta = torch.tensor([[math.cos(ang) / s, -math.sin(ang) / s, tx],
                                  [math.sin(ang) / s, math.cos(ang) / s, ty]], dtype=torch.float32)
            grid = F.affine_grid(theta[None], (1, 1, CACHE_SIZE, CACHE_SIZE), align_corners=False)
            both = F.grid_sample(torch.cat([x[:1], H])[None], grid, mode='bilinear', padding_mode='zeros', align_corners=False)[0]
            m = F.grid_sample(x[None, 1:], grid, mode='nearest', padding_mode='zeros', align_corners=False)[0]
            img = both[:1].clamp(0, 1) ** math.exp(random.uniform(-0.3, 0.3))
            img = img * random.uniform(0.9, 1.1) + random.uniform(-0.05, 0.05)
            x, H = torch.cat([img, m]), both[1:]
        return x, H, torch.from_numpy(has)


class Up(nn.Module):
    def __init__(self, cin, skip, cout):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(cin + skip, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                                  nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    def forward(self, x, skip):
        return self.conv(torch.cat([F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False), skip], 1))


class HeatmapNet(nn.Module):
    """ResNet-18 encoder + U-Net decoder; input = image + 6 mask channels, output = 3 line heatmaps (logits)."""

    def __init__(self, in_ch=1 + len(MASK_CLASSES), pretrained=True):
        super().__init__()
        r = torchvision.models.resnet18(weights='DEFAULT' if pretrained else None)
        w = r.conv1.weight.data
        r.conv1 = nn.Conv2d(in_ch, 64, 7, 2, 3, bias=False)
        r.conv1.weight.data = torch.cat([w.sum(1, keepdim=True), w.mean(1, keepdim=True).repeat(1, in_ch - 1, 1, 1)], 1)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu)
        self.pool, self.l1, self.l2, self.l3, self.l4 = r.maxpool, r.layer1, r.layer2, r.layer3, r.layer4
        self.u3, self.u2, self.u1, self.u0 = Up(512, 256, 256), Up(256, 128, 128), Up(128, 64, 64), Up(64, 64, 32)
        self.out = nn.Conv2d(32, len(LAX), 1)

    def forward(self, x):
        s0 = self.stem(x)
        s1 = self.l1(self.pool(s0))
        s2 = self.l2(s1)
        s3 = self.l3(s2)
        y = self.u0(self.u1(self.u2(self.u3(self.l4(s3), s3), s2), s1), s0)
        return F.interpolate(self.out(y), size=x.shape[-2:], mode='bilinear', align_corners=False)


def heat_loss(logits, H, has):
    """Weighted MSE on sigmoid heatmaps: weight 1 + 9H; views without a labelled plane are ignored."""
    p = logits.float().sigmoid()
    w = (1 + 9 * H) * has[:, :, None, None]
    return (w * (p - H) ** 2).sum() / w.sum().clamp(min=1)


def extract_line(h, sl):
    """Heatmap -> 3D point on the line, 3D unit direction and peak, by weighted PCA of the bright pixels."""
    peak = float(h.max())
    if peak < 0.3:
        return None
    rr, cc = np.nonzero(h > 0.5 * peak)
    if len(rr) < 20:
        return None
    w = h[rr, cc] ** 2
    mu = np.array([np.average(rr, weights=w), np.average(cc, weights=w)])
    X = np.stack([rr - mu[0], cc - mu[1]], 1) * np.sqrt(w)[:, None]
    _, evec = np.linalg.eigh(X.T @ X)
    d = evec[:, 1]  # main direction in (row, col)
    p = sl['o'] + sl['er'] * mu[0] + sl['ec'] * mu[1]
    t = sl['er'] * d[0] + sl['ec'] * d[1]
    return dict(point=p.tolist(), direction=(t / np.linalg.norm(t)).tolist(), peak=peak)


@torch.no_grad()
def predict_scan(model, dataset, sc, device):
    """Lines for every slice of one reference stack (cine: mean heatmap over 3 frames)."""
    model.eval()
    lines = {v: [] for v in LAX}
    for sl in sc['slices']:
        fr = sl['frames']
        idx = sorted({fr[0], fr[len(fr) // 3], fr[2 * len(fr) // 3]})
        h = model(torch.stack([dataset.inputs(i) for i in idx]).to(device)).float().sigmoid().mean(0).cpu().numpy()
        for k, v in enumerate(LAX):
            ln = extract_line(h[k], sl)
            if ln:
                lines[v].append(dict(sax_plane_id=sl['plane_id'], **ln))
    return lines
