"""Anatomical coverage q of a long-axis query (the coverage gate input).

q compares the LV area the query band actually contains with the largest area attainable at the
query's orientation and slice thickness, averaged over eligible mid-ventricular SAX slices.
Areas are approximated by counts of native pixel centres. Independent of labels and predictions.
"""
import numpy as np
from scipy.ndimage import binary_fill_holes, label

from .data import CACHE_SIZE, CROP, SPACING_MM

CFG = dict(min_component_fraction=0.9, duplicate_mm=1.0, min_locations=3,
           min_span_mm=10.0, min_mid_probability=0.8, min_query_angle_deg=50.0)


def basis(tags):
    o = np.asarray(tags['ImagePositionPatient'], float)
    v = np.asarray(tags['ImageOrientationPatient'], float).reshape(2, 3)
    ps = np.asarray(tags['PixelSpacing'], float)
    if not (np.isfinite(o).all() and np.isfinite(v).all() and np.isfinite(ps).all() and (ps > 0).all()):
        raise ValueError('invalid_geometry')
    if not np.allclose(v @ v.T, np.eye(2), atol=1e-3):
        raise ValueError('invalid_orientation')
    v = v / np.linalg.norm(v, axis=1)[:, None]
    return o, v[0], v[1], np.cross(*v)


def world(tags, xy):
    o, x, y, _ = basis(tags)
    ps, xy = tags['PixelSpacing'], np.asarray(xy, float)
    return o + xy[..., 0, None] * ps[1] * x + xy[..., 1, None] * ps[0] * y


def pixel(tags, points):
    o, x, y, _ = basis(tags)
    d, ps = points - o, tags['PixelSpacing']
    return np.stack([d @ x / ps[1], d @ y / ps[0]], axis=-1)


def angle(n, m):
    return float(np.degrees(np.arccos(np.clip(abs(np.dot(n, m)) / np.linalg.norm(n) / np.linalg.norm(m), 0, 1))))


def lv_region(mask):
    """Filled LV region E_i (cavity + myocardium) of one SAX slice, or None with the rejection reason."""
    cavity, myo = (mask & 1) > 0, (mask & 2) > 0
    if not cavity.any() or not myo.any():
        return None, 'missing_lv_structure'
    lab, _ = label(cavity | myo, np.ones((3, 3)))
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    which = int(counts.argmax())
    if counts[which] / (cavity | myo).sum() < CFG['min_component_fraction']:
        return None, 'fragmented_lv'
    region = binary_fill_holes(lab == which)
    if region[0].any() or region[-1].any() or region[:, 0].any() or region[:, -1].any():
        return None, 'boundary_lv'
    return region, 'eligible'


def visible(tags, points):
    """Points inside the query's field of view F after resampling, center crop/pad and the model crop
    (inverts scipy zoom, grid_mode=False; padding is never visible)."""
    xy, valid = pixel(tags, points), np.ones(len(points), bool)
    for axis, (n, ps) in enumerate([(int(tags['Columns']), tags['PixelSpacing'][1]), (int(tags['Rows']), tags['PixelSpacing'][0])]):
        length = int(round(n * ps / SPACING_MM))
        if n < 2 or length < 2:
            raise ValueError('degenerate_resampling')
        native = xy[:, axis]
        scaled = native * (length - 1) / (n - 1)
        src, used = max(0, (length - CACHE_SIZE) // 2), min(length, CACHE_SIZE)
        dst = (CACHE_SIZE - used) // 2
        model = scaled - src + dst - (CACHE_SIZE - CROP) // 2
        valid &= (native >= -0.5) & (native < n - 0.5) & (scaled >= src - 0.5) & (scaled < src + used - 0.5) \
            & (model >= -0.5) & (model < CROP - 0.5)
    return valid


def max_interval_count(values, width):
    a = np.sort(np.asarray(values, float))
    right = np.searchsorted(a, a + width + 1e-8, side='right')
    return int(np.max(right - np.arange(len(a)))) if len(a) else 0


def slice_coverage(points, query_tags):
    """A(E ∩ Q ∩ F) / max_s A(E ∩ Q_s) for one slice; Q = band of slice thickness around the query plane."""
    o, _, _, n = basis(query_tags)
    thick = float(query_tags['SliceThickness'])
    d = (points - o) @ n
    best = max_interval_count(d, thick)
    if best == 0:
        raise ValueError('empty_region')
    inside = (abs(d) <= thick / 2 + 1e-8) & visible(query_tags, points)
    return float(np.clip(inside.sum() / best, 0, 1))


def coverage(query_tags, reference_slices, reference_normal):
    """q and its validity for one query image.

    reference_slices (reference stack order, one representative frame each):
      mask: native-grid uint8 bitmask; tags: DICOM geometry incl. FrameOfReferenceUID;
      position_mm: position along the stack; location: {'apex'|'mid'|'base': probability}
      from the fixed SAX slice-location classifier.
    Returns (q, valid, reason). The caller must only pass queries that have geometric features.
    """
    chosen, regions = [], {}
    for i, s in enumerate(reference_slices):
        loc = s['location']
        if max(loc, key=loc.get) != 'mid' or loc['mid'] < CFG['min_mid_probability']:
            continue  # only confidently mid-ventricular slices
        region, reason = lv_region(np.asarray(s['mask'], np.uint8))
        if reason != 'eligible' or any(abs(s['position_mm'] - c['position_mm']) <= CFG['duplicate_mm'] for c in chosen):
            continue
        yy, xx = np.where(region)
        regions[i] = world(s['tags'], np.stack([xx, yy], axis=1))
        chosen.append(dict(s, index=i))
    pos = [c['position_mm'] for c in chosen]
    try:
        if len(chosen) < CFG['min_locations'] or max(pos) - min(pos) < CFG['min_span_mm']:
            raise ValueError('insufficient_mid_support')
        if angle(basis(query_tags)[3], reference_normal) < CFG['min_query_angle_deg']:
            raise ValueError('not_long_axis_geometry')
        frame = query_tags.get('FrameOfReferenceUID')
        if frame in (None, '', 'None', 'nan') or any(c['tags'].get('FrameOfReferenceUID') != frame for c in chosen):
            raise ValueError('frame_of_reference_mismatch')
        thick = float(query_tags.get('SliceThickness', 0) or 0)
        if not np.isfinite(thick) or thick <= 0:
            raise ValueError('invalid_slice_thickness')
        q = float(np.mean([slice_coverage(regions[c['index']], query_tags) for c in chosen]))
        return q, True, 'valid'
    except (ValueError, KeyError, TypeError) as e:
        return float('nan'), False, str(e)
