"""Step 4. Geometric descriptor (16 values) for every image, and optionally its coverage q.

python -m scripts.build_geometry --manifest data/manifest.jsonl --pairing work/pairing.json \
    --lines work/plane_net/fold*/lines.json --detections data/sax_detections.jsonl \
    [--sax-location data/sax_location.jsonl] --out work/geometry.npz

Images get features only when their examination has a SAX reference stack with enough LV detections
and they share its frame of reference; everything else is marked unavailable.
Output: sample_ids, X [N, 16], valid [N, 16], names, and with --sax-location also q [N], q_valid [N].
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from cmr_compass.coverage import coverage
from cmr_compass.data import load_manifest
from cmr_compass.geometry import FEATURES, LAX, build_planes, cardiac_frame, descriptor, detections_to_3d, fit_plane


def read_jsonl(path):
    return {r['sample_id']: r for r in map(json.loads, Path(path).read_text().splitlines()) if r}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--pairing', type=Path, required=True)
    ap.add_argument('--lines', type=Path, nargs='+', required=True)
    ap.add_argument('--detections', type=Path, required=True)
    ap.add_argument('--sax-location', type=Path, default=None)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()

    rows = load_manifest(a.manifest)
    row_of = {r['sample_id']: r for r in rows}
    planes = build_planes(rows)
    by_id, by_scan = {p['plane_id']: p for p in planes}, defaultdict(list)
    for p in planes:
        by_scan[p['scan_id']].append(p)
    lines = {}
    for f in a.lines:
        lines.update(json.loads(f.read_text()))
    det = read_jsonl(a.detections)
    loc = read_jsonl(a.sax_location) if a.sax_location else None

    feat, cov, reasons = {}, {}, Counter()
    for sc in json.loads(a.pairing.read_text()):
        ref = sc['reference']
        if ref is None or sc['scan_id'] not in lines:
            continue
        # SAX anatomy: one representative frame per reference slice
        rep = [sl['sample_ids'][0] for sl in ref['slices']]
        ref_planes = [by_id[sl['plane_id']] for sl in ref['slices']]
        anat = [detections_to_3d(det.get(s), row_of[s]['dicom_tags']) for s in rep]
        fr = cardiac_frame(ref['normal'], anat)
        if fr is None:
            continue
        est = {v: fit_plane(lines[sc['scan_id']][v]) for v in LAX}
        if loc is not None:
            ref_slices = [dict(mask=np.load(row_of[s]['mask']), tags=row_of[s]['dicom_tags'],
                               position_mm=sl['position_mm'], location=loc.get(s, {}).get('probabilities', {'mid': 0.0}))
                          for s, sl in zip(rep, ref['slices'])]
        for q in by_scan[sc['scan_id']]:
            if q['frame_of_reference'] != ref_planes[0]['frame_of_reference']:
                continue
            f = descriptor(q, fr, est, ref_planes, anat)
            c = coverage(row_of[q['sample_ids'][0]]['dicom_tags'], ref_slices, ref['normal']) if loc is not None else None
            for sid in q['sample_ids']:  # images of one plane share its geometry
                feat[sid], cov[sid] = f, c
                reasons[c[2] if c else None] += 1

    ids = [r['sample_id'] for r in rows]
    X = np.full((len(ids), len(FEATURES)), np.nan, np.float32)
    for i, sid in enumerate(ids):
        if sid in feat:
            X[i] = feat[sid]
    out = dict(sample_ids=np.array(ids), X=np.nan_to_num(X), valid=np.isfinite(X), names=np.array(FEATURES))
    if loc is not None:
        out['q'] = np.array([cov[s][0] if s in cov else np.nan for s in ids], np.float32)
        out['q_valid'] = np.array([cov[s][1] if s in cov else False for s in ids], bool)
        print('coverage:', dict(reasons))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, **out)
    print(f'images with geometric features: {len(feat)} / {len(ids)} -> {a.out}')


if __name__ == '__main__':
    main()
