"""Patient-specific geometry: SAX reference stack, estimated long-axis planes, 16-value descriptor.

Unit of analysis is a *plane*: one slice position inside a series (a cine series is 1 plane x N frames;
an LGE SAX or multi-slice long-axis series has several parallel planes). All points are in patient
coordinates (mm) from the DICOM header.
"""
from collections import defaultdict

import numpy as np

LAX = ('2ch', '3ch', '4ch')
FEATURES = [f'ang_{v}' for v in LAX] + [f'dist_{v}' for v in LAX] + \
           ['tilt_sin', 'tilt_cos', 'along_axis', 'fov_cross', 'd_lv', 'd_rv', 'd_rvip_ant', 'd_rvip_inf', 'hit_lv', 'hit_rv']

# thresholds (degrees / mm)
T = dict(
    stack_parallel_deg=5.0,      # SAX planes closer than this belong to the same stack
    dup_position_mm=1.0,         # stack slices closer than this along the normal are duplicates
    min_ref_slices=5,            # a reference stack needs at least this many slices
    lax_axis_offset_mm=15.0,     # long-axis planes farther than this from the LV axis are off-axis
    on_axis_mm=8.0,              # labelled planes this close to the axis define the plane-network target
    same_plane_deg=3.0,          # two planes this parallel ...
    same_plane_mm=3.0,           # ... and this close are the same plane
    axis_irls_mm=15.0,           # long-axis planes farther than this are down-weighted in axis fitting
)
REF_PRIORITY = {('cine', 'sequence'): 0, ('LGE', 'MAG'): 1, ('LGE', 'PSIR'): 2}
MIN_LV_SLICES = 3                                  # cardiac frame
MIN_SLICES, HALF_LEN_MM, OUTLIER_MM = 3, 25.0, 8.0  # plane fit


def unit(v):
    return v / max(np.linalg.norm(v), 1e-9)


def angle_deg(a, b):
    """Angle between two plane normals (sign-free), in [0, 90]."""
    return float(np.degrees(np.arccos(np.clip(abs(np.dot(a, b)), 0, 1))))


def plane_geometry(tags):
    iop = np.asarray(tags['ImageOrientationPatient'], float)
    x, y = iop[:3] / np.linalg.norm(iop[:3]), iop[3:] / np.linalg.norm(iop[3:])
    n = unit(np.cross(x, y))
    ipp = np.asarray(tags['ImagePositionPatient'], float)
    ps = np.asarray(tags['PixelSpacing'], float)  # [row spacing, column spacing]
    R, C = int(tags['Rows']), int(tags['Columns'])
    ex, ey = x * ps[1] * (C - 1), y * ps[0] * (R - 1)  # full FOV edge vectors
    corners = np.stack([ipp, ipp + ex, ipp + ex + ey, ipp + ey])
    return dict(normal=n, x=x, y=y, origin=ipp, center=ipp + (ex + ey) / 2, corners=corners)


def pixel_to_3d(tags, col, row):
    g, ps = plane_geometry(tags), [float(v) for v in tags['PixelSpacing']]
    return (g['origin'] + g['x'] * ps[1] * col + g['y'] * ps[0] * row).tolist()


# ----------------------------------------------------------------------------- planes and SAX stacks
def build_planes(rows, views=None):
    """Group images into planes: (series_uid, position along the series normal, 0.5 mm bins).
    views: optional {series_uid: view} overriding the manifest view (e.g. classifier-predicted SAX)."""
    by_series = defaultdict(list)
    for r in rows:
        by_series[r['series_uid']].append(r)
    planes = []
    for uid, rs in by_series.items():
        g0 = plane_geometry(rs[0]['dicom_tags'])
        groups = defaultdict(list)
        for r in rs:
            g = plane_geometry(r['dicom_tags'])
            if angle_deg(g['normal'], g0['normal']) > 0.5:
                raise ValueError(f'series {uid} mixes orientations')
            groups[round(float(np.dot(g0['normal'], g['origin'])) * 2) / 2].append(r)
        for k, (_, members) in enumerate(sorted(groups.items())):
            r0 = members[0]
            planes.append(dict(
                plane_id=f'{uid}#{k}', series_uid=uid, patient_id=r0['patient_id'], scan_id=r0['scan_id'],
                modality=r0['modality'], subtype=r0['subtype'], view=(views or {}).get(uid, r0['view']),
                series_description=r0.get('series_description', ''),
                frame_of_reference=r0['dicom_tags'].get('FrameOfReferenceUID'),
                sample_ids=[m['sample_id'] for m in members], **plane_geometry(r0['dicom_tags'])))
    return planes


def build_stacks(sax_planes):
    """SAX planes grouped by (modality, subtype, description), clustered by parallel normals, sorted."""
    groups = defaultdict(list)
    for p in sax_planes:
        if (p['modality'], p['subtype']) in REF_PRIORITY:
            groups[(p['modality'], p['subtype'], ' '.join(str(p['series_description']).lower().split()))].append(p)
    stacks = []
    for (mod, sub, _), ps in groups.items():
        clusters = []
        for p in ps:  # greedy clustering by parallel normals
            for c in clusters:
                if angle_deg(c[0]['normal'], p['normal']) < T['stack_parallel_deg']:
                    c.append(p)
                    break
            else:
                clusters.append([p])
        for c in clusters:
            n = unit(np.stack([p['normal'] * np.sign(np.dot(p['normal'], c[0]['normal'])) for p in c]).mean(0))
            c = sorted(c, key=lambda p: float(np.dot(n, p['center'])))
            pos = np.array([float(np.dot(n, p['center'])) for p in c])
            slices, dups = [], 0
            for p, s in zip(c, pos):
                if slices and s - slices[-1]['position_mm'] < T['dup_position_mm']:
                    dups += 1
                else:
                    slices.append(dict(position_mm=s, plane=p))
            sp = np.diff([s['position_mm'] for s in slices])
            flags = []  # quality flags, used only to rank candidate stacks
            if len(slices) < T['min_ref_slices']:
                flags.append('too_few_slices')
            if len(sp) and (sp.max() > 1.5 * np.median(sp) or sp.min() < 0.5 * np.median(sp)):
                flags.append('irregular_spacing')
            if dups:
                flags.append('duplicate_positions')
            if max(angle_deg(n, p['normal']) for p in c) > 2.0:
                flags.append('normals_spread')
            if len({p['frame_of_reference'] for p in c}) > 1:
                flags.append('mixed_frame_of_reference')
            stacks.append(dict(modality=mod, subtype=sub, normal=n, slices=slices, n_slices=len(slices), flags=flags,
                               centroid=np.mean([s['plane']['center'] for s in slices], 0)))
    return stacks


def choose_reference(stacks):
    """cine > LGE MAG > LGE PSIR; then fewest quality flags, most slices."""
    ok = [s for s in stacks if s['n_slices'] >= T['min_ref_slices']]
    if not ok:
        return None
    return min(ok, key=lambda s: (REF_PRIORITY[(s['modality'], s['subtype'])], len(s['flags']), -s['n_slices']))


def fit_axis(lax_planes, direction, origin):
    """Robust least-squares point on the LV long axis (every long-axis plane should contain it)."""
    d = unit(direction)
    u = unit(np.cross(d, [1.0, 0, 0] if abs(d[0]) < 0.9 else [0, 1.0, 0]))
    v = np.cross(d, u)
    A = np.array([[np.dot(p['normal'], u), np.dot(p['normal'], v)] for p in lax_planes])
    b = np.array([np.dot(p['normal'], p['center'] - origin) for p in lax_planes])
    if len(lax_planes) < 2 or np.linalg.matrix_rank(A, tol=0.2) < 2:
        return None
    w = np.ones(len(b))
    for _ in range(5):  # IRLS: down-weight off-axis planes (e.g. outer slices of multi-slice long-axis series)
        sw = np.sqrt(w)
        ab, *_ = np.linalg.lstsq(A * sw[:, None], b * sw, rcond=None)
        res = np.abs(A @ ab - b)
        w = np.where(res < T['axis_irls_mm'], 1.0, T['axis_irls_mm'] / np.maximum(res, 1e-6))
    return origin + ab[0] * u + ab[1] * v


def pair_scan(planes):
    """Reference SAX stack of one examination, and the labelled on-axis long-axis planes that
    supervise the plane network (targets). Long-axis view labels are used only for targets."""
    ref = choose_reference(build_stacks([p for p in planes if p['view'] == 'sax']))
    out = dict(patient_id=planes[0]['patient_id'], scan_id=planes[0]['scan_id'], reference=None, targets={})
    if ref is None:
        return out
    c0 = float(np.dot(ref['normal'], ref['centroid']))
    out['reference'] = dict(modality=ref['modality'], subtype=ref['subtype'], normal=ref['normal'].tolist(),
                            slices=[dict(plane_id=s['plane']['plane_id'], sample_ids=s['plane']['sample_ids'],
                                         position_mm=s['position_mm'] - c0) for s in ref['slices']])
    lax = [p for p in planes if p['view'] in LAX]
    point = fit_axis(lax, ref['normal'], ref['centroid'])
    if point is None:
        return out
    ref_for = ref['slices'][0]['plane']['frame_of_reference']

    def same_plane_other_view(a):
        return any(b['series_uid'] != a['series_uid'] and b['view'] != a['view'] and
                   angle_deg(a['normal'], b['normal']) < T['same_plane_deg'] and
                   abs(np.dot(a['normal'], b['center'] - a['center'])) < T['same_plane_mm'] and
                   abs(np.dot(b['normal'], a['center'] - b['center'])) < T['same_plane_mm'] for b in planes)

    for v in LAX:
        cand = [(abs(float(np.dot(p['normal'], point - p['center']))), p) for p in lax if p['view'] == v]
        cand = [(off, p) for off, p in cand if off <= T['lax_axis_offset_mm'] and p['frame_of_reference'] == ref_for
                and not same_plane_other_view(p)]
        if cand:
            best = min(off for off, _ in cand)
            out['targets'][v] = [dict(plane_id=p['plane_id'], normal=p['normal'].tolist(), center=p['center'].tolist())
                                 for off, p in cand if off <= max(T['on_axis_mm'], best)]
    return out


# ----------------------------------------------------------------------------- SAX anatomy
def detections_to_3d(det, tags):
    """Per-slice detector output (native pixels) -> 3D LV/RV centres, sizes and RV insertion points."""
    ps = [float(v) for v in tags['PixelSpacing']]
    rec = dict(lv_center=None, lv_radius_mm=None, lv_conf=None, rv_center=None, rv_halfsize_mm=None,
               rvip_ant=None, rvip_inf=None)
    for k, size_key in (('lv', 'lv_radius_mm'), ('rv', 'rv_halfsize_mm')):
        box = (det or {}).get(f'{k}_box')  # [x1, y1, x2, y2, confidence]
        if box:
            x1, y1, x2, y2, conf = box
            rec[f'{k}_center'] = pixel_to_3d(tags, (x1 + x2) / 2, (y1 + y2) / 2)
            rec[f'{k}_conf'] = conf
            rec[size_key] = ((x2 - x1) * ps[1] + (y2 - y1) * ps[0]) / 4  # half of the mean box side
    if (det or {}).get('rvip_ant') and det.get('rvip_inf'):
        rec.update(rvip_ant=pixel_to_3d(tags, *det['rvip_ant']), rvip_inf=pixel_to_3d(tags, *det['rvip_inf']))
    return rec


def cardiac_frame(ref_normal, slices):
    """LV axis (line through the LV centres of the SAX slices), RV direction, LV radius and length."""
    lv = [s for s in slices if s['lv_center'] is not None]
    if len(lv) < MIN_LV_SLICES:
        return None
    P = np.array([s['lv_center'] for s in lv])
    p0 = P.mean(0)
    n = unit(np.asarray(ref_normal, float))
    d = n
    if np.ptp(P @ n) > 20:  # enough extent along the stack to fit a line through the LV centres
        _, _, vt = np.linalg.svd(P - p0)
        cand = unit(vt[0]) * np.sign(vt[0] @ n)
        if np.degrees(np.arccos(np.clip(cand @ n, -1, 1))) < 35:  # a sane LV axis is close to the stack normal
            d = cand
    rv = [np.array(s['rv_center']) - np.array(s['lv_center']) for s in lv if s['rv_center'] is not None]
    ip = [(np.array(s['rvip_ant']) + np.array(s['rvip_inf'])) / 2 - np.array(s['lv_center'])
          for s in lv if s['rvip_ant'] is not None]
    if not rv and not ip:
        return None
    v = np.mean(rv if rv else ip, 0)  # fall back to the RV insertion points (their midpoint is on the RV side)
    e_rv = v - (v @ d) * d
    if np.linalg.norm(e_rv) < 5:
        return None
    along = (P - p0) @ d
    return dict(d=d, p0=p0, e_rv=unit(e_rv), r_lv=float(np.median([s['lv_radius_mm'] for s in lv])),
                lv_length=float(max(np.ptp(along), 20.0)))


def line_on_slice(q, s):
    """Intersection of query plane q with SAX slice plane s: (point on line, unit direction) or None."""
    t = np.cross(q['normal'], s['normal'])
    if np.linalg.norm(t) < 0.2:  # nearly parallel planes
        return None
    t = unit(t)
    A = np.stack([q['normal'], s['normal'], t])
    b = np.array([q['normal'] @ q['center'], s['normal'] @ s['center'], t @ s['center']])
    return np.linalg.solve(A, b), t


def dist_point_line(pt, p, t):
    v = np.asarray(pt) - p
    return float(np.linalg.norm(v - (v @ t) * t))


# ----------------------------------------------------------------------------- estimated planes
def fit_plane(lines):
    """Plane through the predicted lines of a stack (each line contributes 3 points); drops outlier slices."""
    use = [l for l in lines if l['peak'] >= 0.3]
    for _ in range(3):
        if len(use) < MIN_SLICES:
            return None
        pts, w = [], []
        for l in use:
            p, t = np.array(l['point']), np.array(l['direction'])
            pts += [p - HALF_LEN_MM * t, p, p + HALF_LEN_MM * t]
            w += [l['peak']] * 3
        pts, w = np.array(pts), np.array(w)
        c = np.average(pts, axis=0, weights=w)
        _, sv, vt = np.linalg.svd((pts - c) * np.sqrt(w)[:, None], full_matrices=False)
        n = vt[2]
        res = np.array([max(abs((np.array(l['point']) + s * HALF_LEN_MM * np.array(l['direction']) - c) @ n)
                            for s in (-1, 0, 1)) for l in use])
        keep = res <= max(OUTLIER_MM, 2.5 * np.median(res))
        if keep.all():
            break
        use = [l for l, k in zip(use, keep) if k]
    if len(use) < MIN_SLICES or sv[1] < 1e-6:
        return None
    return dict(normal=n, point=c, n_slices=len(use))


def heart_points(plane, fr):
    """Five points of an estimated plane around the LV axis (the J = 5 points of the paper)."""
    n = plane['normal']
    c = fr['p0'] - ((fr['p0'] - plane['point']) @ n) * n  # LV centre projected onto the plane
    a = unit(fr['d'] - (fr['d'] @ n) * n)                  # along the LV axis, inside the plane
    b = np.cross(n, a)
    r = 1.5 * fr['r_lv']
    return np.array([c, c + r * a, c - r * a, c + r * b, c - r * b])


# ----------------------------------------------------------------------------- descriptor
def descriptor(q, fr, est_planes, ref_planes, anat):
    """16 measurements of query plane q (NaN = unavailable).

    q: plane_geometry of the query; fr: cardiac_frame; est_planes: {view: fit_plane or None};
    ref_planes / anat: reference SAX slice planes and their 3D detections, in stack order.
    """
    n = q['normal']
    cos_t = abs(n @ fr['d'])
    f = dict(tilt_sin=float(np.sqrt(max(0.0, 1 - cos_t ** 2))), tilt_cos=float(cos_t),
             along_axis=float(fr['d'] @ (q['center'] - fr['p0'])) / fr['lv_length'])
    for v in LAX:  # g_plane: angle (/ 90 deg) and in-heart distance (/ LV radius, capped at 5)
        p = est_planes.get(v)
        if p is None:
            f[f'ang_{v}'] = f[f'dist_{v}'] = np.nan
        else:
            f[f'ang_{v}'] = angle_deg(n, p['normal']) / 90.0
            f[f'dist_{v}'] = min(5.0, float(np.mean(np.abs((heart_points(p, fr) - q['center']) @ n))) / fr['r_lv'])
    crossed, dd = 0, defaultdict(list)
    for s, a in zip(ref_planes, anat):  # g_landmark and g_extent, via the query's line on each SAX slice
        sd = (s['corners'] - q['origin']) @ n
        crossed += bool(sd.min() < 0 < sd.max())
        ln = line_on_slice(q, s)
        if ln is None or a['lv_center'] is None:
            continue
        dl = dist_point_line(a['lv_center'], *ln)
        dd['d_lv'].append(dl / fr['r_lv'])
        dd['hit_lv'].append(float(dl < a['lv_radius_mm']))
        if a['rv_center'] is not None:
            dr = dist_point_line(a['rv_center'], *ln)
            dd['d_rv'].append(dr / fr['r_lv'])
            dd['hit_rv'].append(float(dr < a['rv_halfsize_mm']))
        if a['rvip_ant'] is not None:
            dd['d_rvip_ant'].append(dist_point_line(a['rvip_ant'], *ln) / fr['r_lv'])
            dd['d_rvip_inf'].append(dist_point_line(a['rvip_inf'], *ln) / fr['r_lv'])
    f['fov_cross'] = crossed / len(ref_planes)
    for k in ['d_lv', 'd_rv', 'd_rvip_ant', 'd_rvip_inf']:
        f[k] = float(np.median(dd[k])) if dd[k] else np.nan
    for k in ['hit_lv', 'hit_rv']:
        f[k] = float(np.mean(dd[k])) if dd[k] else np.nan
    return [f[k] for k in FEATURES]
