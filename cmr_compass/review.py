"""Expert review queue: mismatches first, then series with high branch disagreement."""
import math

import numpy as np


def disagreement(branch_probs):
    """D = JSD({p_b}) / log |B|: equally weighted Jensen-Shannon divergence of the series-mean branch
    probabilities, in [0, 1]. High when branch distributions differ, even if their argmax agrees."""
    p = np.asarray(branch_probs, float)
    p = p / p.sum(axis=1, keepdims=True)
    m = p.mean(axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        terms = np.where(p > 0, p * np.log(p / np.where(m > 0, m, 1)), 0)
    return float(terms.sum() / len(p) / math.log(len(p)))


def review_queue(series, budget=0.10):
    """series: dicts with series_uid, patient_id, label (reference/description view), pred, branch_probs.

    Step 1: every mismatch (pred != label). Step 2: the top `budget` fraction of all series by D,
    minus series of patients that already have a mismatch (no refill). Ties break by series_uid.
    """
    mismatch = [s for s in series if s['pred'] != s['label']]
    patients = {s['patient_id'] for s in mismatch}
    for s in series:
        s['disagreement'] = disagreement(s['branch_probs'])
    ranked = sorted(series, key=lambda s: (-s['disagreement'], s['series_uid']))
    top = ranked[:math.ceil(len(series) * budget)]
    return mismatch, [s for s in top if s['patient_id'] not in patients]
