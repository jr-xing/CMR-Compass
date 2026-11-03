"""Step 2. Choose each examination's SAX reference stack and the plane-network targets.

python -m scripts.pair --manifest data/manifest.jsonl --out work/pairing.json
python -m scripts.pair --manifest ... --sax-views visual_only_predictions.csv --out ...   # external data

SAX series are taken from the manifest's view column (internal: manually reviewed labels) or, with
--sax-views, from a classifier's series predictions (CSV with series_uid, pred_view; external data).
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from cmr_compass.data import load_manifest
from cmr_compass.geometry import build_planes, pair_scan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--sax-views', type=Path, default=None)
    a = ap.parse_args()
    views = None
    if a.sax_views:
        views = {r['series_uid']: r['pred_view'] for r in csv.DictReader(open(a.sax_views))}
    by_scan = defaultdict(list)
    for p in build_planes(load_manifest(a.manifest), views):
        by_scan[(p['patient_id'], p['scan_id'])].append(p)
    scans = [pair_scan(by_scan[k]) for k in sorted(by_scan)]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(scans, indent=1))
    print(f"{len(scans)} examinations, {sum(s['reference'] is not None for s in scans)} with a SAX reference stack")


if __name__ == '__main__':
    main()
