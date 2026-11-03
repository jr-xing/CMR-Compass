"""CPU smoke tests on synthetic inputs (no data or downloads needed):  python -m pytest tests"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from cmr_compass.coverage import coverage
from cmr_compass.geometry import FEATURES, cardiac_frame, descriptor, fit_plane, plane_geometry
from cmr_compass.model import CMRCompass, coverage_gate, load_checkpoint
from cmr_compass.review import disagreement

REPO = Path(__file__).resolve().parents[1]


def tags(x_dir, y_dir, origin, n=64, spacing=1.5, thickness=8.0):
    return dict(ImageOrientationPatient=list(x_dir) + list(y_dir), ImagePositionPatient=list(origin),
                PixelSpacing=[spacing, spacing], Rows=n, Columns=n, SliceThickness=thickness, FrameOfReferenceUID='F1')


def sax_tags(z):  # axial slices centred on the z axis, which plays the LV long axis
    return tags((1, 0, 0), (0, 1, 0), (-48, -48, z))


def lax_tags(angle_deg):  # long-axis plane containing the z axis
    c, s = np.cos(np.radians(angle_deg)), np.sin(np.radians(angle_deg))
    u = np.array([-s, c, 0.0])
    return tags(u, (0, 0, -1), -48 * u + [0, 0, 80])


# ----------------------------------------------------------------------------- model
def test_forward_backward_and_single_branch():
    x, m, g = torch.randn(2, 1, 224, 224), torch.rand(2, 6, 224, 224).round(), torch.randn(2, 32)
    model = CMRCompass(pretrained=False)
    fused, aux = model(x, m, g)
    assert fused.shape == (2, 4) and set(aux) == {'visual', 'structural', 'geometric'}
    (fused.sum() + sum(a.sum() for a in aux.values())).backward()
    fused, aux = CMRCompass(['geometric'], pretrained=False)(x, m, g)
    assert torch.equal(fused, aux['geometric'])


def test_research_checkpoint_names_load(tmp_path):
    model = CMRCompass(pretrained=False).eval()
    old = {k.replace('.visual.', '.image.').replace('.structural.', '.anatomy.').replace('.geometric.', '.geom.'): v
           for k, v in model.state_dict().items()}
    torch.save(dict(model=old, args={}), tmp_path / 'best.pt')
    loaded, _ = load_checkpoint(tmp_path / 'best.pt')
    x, m, g = torch.randn(2, 1, 224, 224), torch.rand(2, 6, 224, 224).round(), torch.randn(2, 32)
    with torch.no_grad():
        assert torch.equal(model(x, m, g)[0], loaded(x, m, g)[0])


def test_coverage_gate():
    pf, pg = np.array([[0.7, 0.1, 0.1, 0.1]] * 3), np.array([[0.1, 0.1, 0.1, 0.7]] * 3)
    out = coverage_gate(pf, pg, np.array([1.0, 0.0, np.nan]), np.array([True, True, False]))
    assert np.allclose(out[0], pf[0]) and np.allclose(out[1], pg[1]) and np.allclose(out[2], pf[2])


def test_disagreement_bounds():
    assert abs(disagreement([[0.25, 0.25, 0.25, 0.25]] * 3)) < 1e-12
    assert abs(disagreement([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]) - 1) < 1e-12


# ----------------------------------------------------------------------------- geometry
def test_fit_plane_recovers_plane():
    rng = np.random.default_rng(0)
    n = np.array([np.cos(0.4), np.sin(0.4), 0.0])  # plane through the z axis
    t = np.cross(n, [0, 0, 1])
    lines = [dict(point=(np.array([0, 0, z]) + rng.normal(0, 0.5, 3)).tolist(), direction=t.tolist(), peak=0.9)
             for z in range(0, 60, 10)]
    fit = fit_plane(lines)
    assert np.degrees(np.arccos(abs(fit['normal'] @ n))) < 2


def test_descriptor_on_synthetic_heart():
    zs = range(0, 60, 10)
    ref_planes = [plane_geometry(sax_tags(z)) for z in zs]
    anat = [dict(lv_center=[0, 0, z], lv_radius_mm=25.0, lv_conf=0.9, rv_center=[30, 0, z], rv_halfsize_mm=20.0,
                 rvip_ant=[20, 15, z], rvip_inf=[20, -15, z]) for z in zs]
    fr = cardiac_frame([0, 0, 1], anat)
    q = plane_geometry(lax_tags(0))  # the plane x = 0, through the LV axis
    est = {'2ch': None, '3ch': None, '4ch': dict(normal=q['normal'], point=np.zeros(3))}
    f = dict(zip(FEATURES, descriptor(q, fr, est, ref_planes, anat)))
    assert np.isnan(f['ang_2ch']) and f['ang_4ch'] < 1e-6 and f['dist_4ch'] < 1e-6
    assert abs(f['tilt_sin'] - 1) < 1e-6 and f['fov_cross'] == 1 and f['d_lv'] < 1e-6 and f['hit_lv'] == 1
    assert abs(f['d_rv'] - 30 / 25) < 1e-6 and f['hit_rv'] == 0


def test_coverage_through_and_beside_the_lv():
    yy, xx = np.mgrid[0:64, 0:64]
    disk = ((yy - 32) ** 2 + (xx - 32) ** 2 <= 8 ** 2).astype(np.uint8)
    mask = disk | (disk << 1)  # bit 0 LV cavity, bit 1 LV epicardium
    ref = [dict(mask=mask, tags=sax_tags(z), position_mm=z, location=dict(apex=0.05, mid=0.9, base=0.05))
           for z in (0, 5, 10, 15)]
    centre = 32 * 1.5 - 48  # LV centre x in mm
    q, ok, _ = coverage(tags((0, 1, 0), (0, 0, -1), (centre, -48, 80), thickness=9.0), ref, [0, 0, 1])
    assert ok and q > 0.99
    q_side, ok, _ = coverage(tags((0, 1, 0), (0, 0, -1), (centre + 8, -48, 80), thickness=9.0), ref, [0, 0, 1])
    assert ok and q_side < 0.9


# ----------------------------------------------------------------------------- end to end
def make_dataset(root, n_patients=6):
    """Each patient: one 6-slice cine SAX stack along z and 2CH/3CH/4CH planes through the z axis."""
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:64, 0:64]
    lv = ((yy - 32) ** 2 + (xx - 32) ** 2 <= 8 ** 2).astype(np.uint8) * 3
    rows, det, loc = [], [], []

    def add(pid, uid, view, t, mask, k):
        sid = f'{uid}_{k}'
        np.save(root / f'{sid}.npy', rng.random((64, 64)).astype(np.float32))
        np.save(root / f'{sid}_mask.npy', mask)
        rows.append(dict(sample_id=sid, patient_id=pid, scan_id=f'{pid}_scan', series_uid=uid, dataset='synthetic',
                         modality='cine', subtype='sequence', view=view, series_description=f'{view} cine',
                         image=str(root / f'{sid}.npy'), mask=str(root / f'{sid}_mask.npy'), dicom_tags=t))
        return sid

    for p in range(n_patients):
        pid = f'P{p}'
        for k, z in enumerate(range(0, 60, 10)):
            sid = add(pid, f'{pid}_sax', 'sax', sax_tags(z), lv, k)
            det.append(dict(sample_id=sid, lv_box=[24, 24, 40, 40, 0.9], rv_box=[40, 26, 52, 38, 0.8],
                            rvip_ant=[40, 26], rvip_inf=[40, 38]))
            loc.append(dict(sample_id=sid, probabilities=dict(apex=0.05, mid=0.9, base=0.05)))
        for view, ang in (('4ch', 0), ('2ch', 90), ('3ch', 135)):
            for k in range(2):
                add(pid, f'{pid}_{view}', view, lax_tags(ang), np.zeros((64, 64), np.uint8), k)
    for name, items in (('manifest', rows), ('sax_detections', det), ('sax_location', loc)):
        (root / f'{name}.jsonl').write_text('\n'.join(json.dumps(r) for r in items))


def run(*args):
    subprocess.run([sys.executable, '-m', *map(str, args)], cwd=REPO, check=True)


def test_pipeline_end_to_end(tmp_path):
    d, w = tmp_path / 'data', tmp_path / 'work'
    d.mkdir()
    make_dataset(d)
    common = ['--manifest', d / 'manifest.jsonl']
    run('scripts.preprocess', *common, '--out', w / 'cache')
    run('scripts.pair', *common, '--out', w / 'pairing.json')
    assert all(s['reference'] and len(s['targets']) == 3 for s in json.loads((w / 'pairing.json').read_text()))
    run('scripts.train_plane_net', *common, '--cache', w / 'cache', '--pairing', w / 'pairing.json', '--fold', 0,
        '--epochs', 1, '--batch-size', 2, '--workers', 0, '--no-pretrained', '--out', w / 'plane_net')
    run('scripts.train_plane_net', *common, '--cache', w / 'cache', '--pairing', w / 'pairing.json',
        '--checkpoint', w / 'plane_net/fold0/model.pt', '--workers', 0, '--out', w / 'plane_net')
    run('scripts.build_geometry', *common, '--pairing', w / 'pairing.json', '--lines', w / 'plane_net/all/lines.json',
        '--detections', d / 'sax_detections.jsonl', '--sax-location', d / 'sax_location.jsonl', '--out', w / 'geometry.npz')
    g = np.load(w / 'geometry.npz')
    assert g['valid'].any(axis=1).all() and g['q_valid'].any()
    run('scripts.train', *common, '--cache', w / 'cache', '--geometry', w / 'geometry.npz', '--fold', 0,
        '--epochs', 1, '--batch-size', 4, '--workers', 0, '--no-pretrained', '--out', w / 'model')
    run('scripts.predict', *common, '--cache', w / 'cache', '--geometry', w / 'geometry.npz', '--workers', 0,
        '--checkpoints', w / 'model/fold0/best.pt', '--out', w / 'pred_series.csv')
    run('scripts.review_queue', '--predictions', w / 'pred_series.csv', '--out', w / 'review_queue.csv')
    assert (w / 'review_queue.csv').read_text().startswith('step,series_uid')
