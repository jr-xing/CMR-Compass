"""CMR-Compass network and coverage gate.

Branches: visual (ResNet-18 on the image), structural (ResNet-18 on 6 predicted masks),
geometric (MLP on the 16 standardized measurements + 16 availability flags).
Each branch has an auxiliary linear head; the fusion head sees the concatenated features.
A model with a single branch is the single-branch baseline (its auxiliary head is the output).
"""
import numpy as np
import torch
import torch.nn as nn
import torchvision

from . import BRANCHES, CLASSES
from .data import MASK_CLASSES

IMNET_MEAN, IMNET_STD = 0.449, 0.226  # grayscale equivalent of ImageNet statistics


def resnet_encoder(in_ch, pretrained):
    m = torchvision.models.resnet18(weights='DEFAULT' if pretrained else None)
    w = m.conv1.weight.data
    m.conv1 = nn.Conv2d(in_ch, 64, 7, 2, 3, bias=False)
    # 1 channel: sum of RGB filters (= gray replicated); k mask channels: mean filter per channel
    m.conv1.weight.data = w.sum(1, keepdim=True) if in_ch == 1 else w.mean(1, keepdim=True).repeat(1, in_ch, 1, 1)
    m.fc = nn.Identity()
    return m, 512


class CMRCompass(nn.Module):
    def __init__(self, branches=BRANCHES, geom_dim=32, pretrained=True, drop=0.2):
        super().__init__()
        self.branches = [b for b in BRANCHES if b in branches]  # fixed order: visual, structural, geometric
        self.enc, dims = nn.ModuleDict(), {}
        if 'visual' in branches:
            self.enc['visual'], dims['visual'] = resnet_encoder(1, pretrained)
        if 'structural' in branches:
            self.enc['structural'], dims['structural'] = resnet_encoder(len(MASK_CLASSES), pretrained)
        if 'geometric' in branches:
            self.enc['geometric'] = nn.Sequential(nn.Linear(geom_dim, 64), nn.ReLU(inplace=True),
                                                  nn.Linear(64, 64), nn.ReLU(inplace=True))
            dims['geometric'] = 64
        self.aux = nn.ModuleDict({b: nn.Linear(dims[b], len(CLASSES)) for b in self.branches})
        self.head = None if len(self.branches) == 1 else nn.Sequential(
            nn.Dropout(drop), nn.Linear(sum(dims.values()), 256), nn.ReLU(inplace=True),
            nn.Dropout(drop), nn.Linear(256, len(CLASSES)))

    def forward(self, image, masks, geometry):
        """image [B,1,224,224] normalized; masks [B,6,224,224]; geometry [B,32]. Returns logits."""
        inp = dict(visual=image, structural=masks, geometric=geometry)
        feats = {b: self.enc[b](inp[b]) for b in self.branches}
        aux = {b: self.aux[b](feats[b]) for b in self.branches}
        fused = aux[self.branches[0]] if self.head is None else self.head(torch.cat([feats[b] for b in self.branches], 1))
        return fused, aux


@torch.no_grad()
def predict(model, loader, device):
    """Image-level softmax probabilities {'fused': [N, 4], <branch>: [N, 4]} in loader.dataset order."""
    model.eval()
    n = len(loader.dataset)
    out = {k: np.zeros((n, len(CLASSES)), np.float32) for k in ['fused'] + model.branches}
    for x, m, g, _, idx in loader:
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            fused, aux = model(x.to(device), m.to(device), g.to(device))
        out['fused'][idx.numpy()] = fused.float().softmax(1).cpu().numpy()
        for b, logits in aux.items():
            out[b][idx.numpy()] = logits.float().softmax(1).cpu().numpy()
    return out


def coverage_gate(p_fused, p_geometric, q, q_valid):
    """p = q * p_f + (1 - q) * p_G when coverage is valid, else p_f (numpy arrays, probabilities [..., 4])."""
    q = np.asarray(q, float)
    valid = np.asarray(q_valid, bool) & np.isfinite(q)
    alpha = np.where(valid, 1 - np.clip(np.nan_to_num(q), 0, 1), 0.0)
    return (1 - alpha[..., None]) * p_fused + alpha[..., None] * p_geometric


# Original research checkpoints used the names image / anatomy / geom for the three branches.
_OLD_NAMES = {'image': 'visual', 'anatomy': 'structural', 'geom': 'geometric'}


def load_checkpoint(path, device='cpu'):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    state = {}
    for k, v in ckpt['model'].items():
        part = k.split('.')
        if part[0] in ('enc', 'aux') and part[1] in _OLD_NAMES:
            part[1] = _OLD_NAMES[part[1]]
        state['.'.join(part)] = v
    branches = [b for b in BRANCHES if f'aux.{b}.weight' in state]
    geom_dim = state['enc.geometric.0.weight'].shape[1] if 'geometric' in branches else 32
    model = CMRCompass(branches, geom_dim, pretrained=False)
    model.load_state_dict(state)
    return model.to(device).eval(), ckpt
