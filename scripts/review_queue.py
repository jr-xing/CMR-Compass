"""Step 7. Expert review queue from series predictions (output of predict.py or train.py).

python -m scripts.review_queue --predictions work/pred_series.csv --out work/review_queue.csv

Step 1 lists every mismatch between prediction and the reference (description) label. Step 2 adds the
top 10% of series by branch disagreement, minus patients that already have a mismatch.
"""
import argparse
import csv
from pathlib import Path

from cmr_compass import BRANCHES, CLASSES
from cmr_compass.review import review_queue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--predictions', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--budget', type=float, default=0.10, help='fraction of all series flagged by disagreement')
    a = ap.parse_args()
    series = [dict(r, label=r['view'], pred=r['pred_view'],
                   branch_probs=[[float(r[f'{b}_p_{c}']) for c in CLASSES] for b in BRANCHES])
              for r in csv.DictReader(open(a.predictions))]
    mismatch, flagged = review_queue(series, a.budget)
    with open(a.out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['step', 'series_uid', 'patient_id', 'view', 'pred_view', 'disagreement'])
        for step, rows in (('1_mismatch', mismatch), ('2_disagreement', flagged)):
            for s in rows:
                w.writerow([step, s['series_uid'], s['patient_id'], s['view'], s['pred_view'], f"{s['disagreement']:.4f}"])
    print(f'{len(mismatch)} mismatches + {len(flagged)} series flagged by disagreement / {len(series)} -> {a.out}')


if __name__ == '__main__':
    main()
