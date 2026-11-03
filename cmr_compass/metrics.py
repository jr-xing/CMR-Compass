"""Series aggregation and classification metrics."""
import csv
from collections import defaultdict

import numpy as np

from . import CLASSES


def series_level(rows, probs):
    """Mean image probability per series, for every output in probs ({name: [N, 4]})."""
    idx = defaultdict(list)
    for i, r in enumerate(rows):
        idx[r['series_uid']].append(i)
    out = []
    for uid, ii in idx.items():
        r0 = rows[ii[0]]
        out.append(dict(series_uid=uid, patient_id=r0['patient_id'], modality=r0['modality'], view=r0['view'],
                        label=r0['label'], n_images=len(ii), probs={k: v[ii].mean(0) for k, v in probs.items()}))
    return out


def write_series_csv(path, series, heads):
    """One row per series: final prediction and probabilities, then each extra head's probabilities."""
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['series_uid', 'patient_id', 'modality', 'n_images', 'view', 'pred_view'] +
                   [f'p_{c}' for c in CLASSES] + [f'{h}_p_{c}' for h in heads for c in CLASSES])
        for s in series:
            p = s['probs']['final']
            w.writerow([s['series_uid'], s['patient_id'], s['modality'], s['n_images'], s['view'],
                        CLASSES[int(p.argmax())]] + [f'{v:.5f}' for v in p] +
                       [f'{v:.5f}' for h in heads for v in s['probs'][h]])


def metrics(y, p, n_cls=len(CLASSES)):
    """Accuracy, macro-F1 (over classes present in y) and confusion matrix."""
    y, p = np.asarray(y), np.asarray(p)
    cm = np.zeros((n_cls, n_cls), int)
    for a, b in zip(y, p):
        cm[a, b] += 1
    f1 = []
    for c in [c for c in range(n_cls) if (y == c).any()]:
        tp = cm[c, c]
        prec = tp / cm[:, c].sum() if cm[:, c].sum() else 0.0
        rec = tp / cm[c, :].sum()
        f1.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return dict(n=int(len(y)), acc=float((y == p).mean()) if len(y) else float('nan'),
                macro_f1=float(np.mean(f1)) if f1 else float('nan'), confusion=cm.tolist())
