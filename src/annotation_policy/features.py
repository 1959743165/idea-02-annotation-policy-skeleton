"""Pixel-only detector features and public expert-disagreement descriptors."""
from __future__ import annotations
from collections import OrderedDict
import hashlib
import numpy as np
from scipy import ndimage
from .annotation_state import consensus_labelmap, independent_reads

PATCH_FEATURE_NAMES = (
    "mean_gray", "gray_std", "min_gray", "max_gray", "dark_deficit", "dark_fraction",
    "gradient_x", "gradient_y", "gradient_anisotropy", "component_elongation",
    "component_area", "context_dark_fraction", "context_contrast", "patch_x", "patch_y",
)
CONTEXT_FEATURE_NAMES = (
    "model_uncertainty", "entropy", "existence_disagreement", "class_disagreement",
    "boundary_disagreement", "boundary_pair_fraction", "repeat_disagreement", "image_quality",
    "image_contrast", "n_annotators", "repeat_read_count", "active", "annotation_foreground_fraction",
    "model_annotation_gap",
)
_FEATURE_CACHE: OrderedDict = OrderedDict()
_MAX_CACHE_SIZE = 1024


def _blocks(array, grid):
    h, w = array.shape
    if grid <= 0 or h % grid or w % grid:
        raise ValueError("Reference patch grid must divide image height and width")
    return array.reshape(grid, h // grid, grid, w // grid).transpose(0, 2, 1, 3)


def extract_patch_features(image, grid_size=8):
    """Engineering features for the CPU fixture, not a learned vision backbone.

    No annotations or metadata are accepted. Caching is keyed by actual pixel
    bytes, so a repeat-image action cannot re-use stale appearance features.
    """
    im = np.asarray(image, dtype=np.float64)
    if im.ndim != 2 or not np.isfinite(im).all() or im.min() < 0 or im.max() > 1:
        raise ValueError("Expected finite grayscale pixels in [0,1]")
    key = (im.shape, int(grid_size), hashlib.sha256(im.tobytes()).hexdigest())
    if key in _FEATURE_CACHE:
        _FEATURE_CACHE.move_to_end(key)
        return _FEATURE_CACHE[key].copy()
    blocks = _blocks(im, grid_size)
    background = ndimage.gaussian_filter(im, max(2, min(im.shape) / 8))
    deficit = np.maximum(background - im, 0)
    # Thresholded connected components are a declared appearance feature, not
    # oracle object proposals. In low contrast/noise this heuristic can fail.
    threshold = max(.018, float(np.std(im) * .55))
    dark = deficit > threshold
    component_map, count = ndimage.label(dark)
    elongation = np.zeros_like(im); area_map = np.zeros_like(im)
    for k in range(1, count + 1):
        mask = component_map == k
        yy, xx = np.where(mask)
        if len(xx) < 3:
            continue
        cov = np.cov(np.stack((xx, yy)))
        eigenvalues = np.linalg.eigvalsh(cov)
        anisotropy = float((eigenvalues[-1] - eigenvalues[0]) / max(eigenvalues.sum(), 1e-8))
        elongation[mask] = anisotropy
        area_map[mask] = min(1.0, len(xx) / (im.size * .20))
    gx = ndimage.sobel(im, axis=1, mode="reflect") / 8
    gy = ndimage.sobel(im, axis=0, mode="reflect") / 8
    ex = _blocks(gx ** 2, grid_size).mean(axis=(2, 3))
    ey = _blocks(gy ** 2, grid_size).mean(axis=(2, 3))
    e_xy = _blocks(gx * gy, grid_size).mean(axis=(2, 3))
    anisotropy = np.sqrt((ex - ey) ** 2 + 4 * e_xy ** 2) / np.maximum(ex + ey, 1e-8)
    context = max(3, int(round(min(im.shape) / grid_size * 2)))
    context_fraction = ndimage.uniform_filter(dark.astype(float), size=context)
    context_contrast = np.sqrt(np.maximum(ndimage.uniform_filter(im ** 2, size=context)
                                         - ndimage.uniform_filter(im, size=context) ** 2, 0))
    y, x = np.mgrid[:grid_size, :grid_size]
    values = [blocks.mean((2, 3)), blocks.std((2, 3)), blocks.min((2, 3)), blocks.max((2, 3)),
              _blocks(deficit, grid_size).mean((2, 3)), _blocks(dark, grid_size).mean((2, 3)),
              np.sqrt(ex), np.sqrt(ey), anisotropy,
              _blocks(elongation, grid_size).mean((2, 3)), _blocks(area_map, grid_size).mean((2, 3)),
              _blocks(context_fraction, grid_size).mean((2, 3)), _blocks(context_contrast, grid_size).mean((2, 3)),
              (x + .5) / grid_size, (y + .5) / grid_size]
    result = np.stack(values, axis=-1).reshape(-1, len(values))
    _FEATURE_CACHE[key] = result.copy()
    if len(_FEATURE_CACHE) > _MAX_CACHE_SIZE:
        _FEATURE_CACHE.popitem(last=False)
    return result


def clear_feature_cache():
    _FEATURE_CACHE.clear()


def describe_case(case, model):
    """Fixed features from pixels, CURRENT rater reads and model probabilities.

    No latent error mode, truth label, future cost, potential reward, test score
    or unpurchased action outcome is read. Rater IDs are not one-hot shortcuts.
    """
    prediction = np.asarray(model.predict(case), dtype=float)
    entropy_grid = -(prediction * np.log(np.maximum(prediction, 1e-12))).sum(axis=-1) / np.log(3)
    # Top-eight entropy locations are selected without reference labels.
    uncertainty = float(np.sort(entropy_grid.ravel())[-min(8, entropy_grid.size):].mean())
    reads = independent_reads(case)
    foreground = [(r.labelmap > 0) & (r.labelmap != 255) for r in reads]
    exists = np.asarray([f.any() for f in foreground], dtype=float)
    existence_disagreement = float(4 * exists.mean() * (1 - exists.mean())) if len(exists) else 0.0
    class_presence = np.array([[bool((r.labelmap == c).any()) for c in (1, 2)] for r in reads], dtype=float)
    class_disagreement = float((4 * class_presence.mean(axis=0) * (1 - class_presence.mean(axis=0))).mean()) if len(reads) else 0.0
    pair_distances = []
    pair_total = len(reads) * (len(reads) - 1) / 2
    for i in range(len(reads)):
        for j in range(i):
            if foreground[i].any() and foreground[j].any():
                union = np.logical_or(foreground[i], foreground[j]).sum()
                intersection = np.logical_and(foreground[i], foreground[j]).sum()
                pair_distances.append(1 - intersection / max(union, 1))
    original_by_rater = {r.annotator_id: r for r in reads}
    repeated = []
    for r in case.annotations:
        if r.is_repeat and r.annotator_id in original_by_rater:
            first = original_by_rater[r.annotator_id].labelmap
            union = ((first > 0) & (first != 255)) | ((r.labelmap > 0) & (r.labelmap != 255))
            repeated.append(float(((first != r.labelmap) & union).sum() / max(union.sum(), 1)))
    im = np.asarray(case.image)
    highpass = im - ndimage.gaussian_filter(im, .8)
    noise = float(np.median(np.abs(highpass - np.median(highpass))))
    contrast = float(np.percentile(im, 95) - np.percentile(im, 5))
    quality = contrast / max(contrast + 4 * noise, 1e-8)
    if case.active:
        consensus = consensus_labelmap(case)
        annotated_fraction = float(((consensus > 0) & (consensus != 255)).mean())
    else:
        annotated_fraction = 0.0
    model_fraction = float((1 - prediction[..., 0]).mean())
    result = {
        "model_uncertainty": uncertainty, "entropy": float(entropy_grid.mean()),
        "existence_disagreement": existence_disagreement, "class_disagreement": class_disagreement,
        "boundary_disagreement": float(np.mean(pair_distances)) if pair_distances else 0.0,
        "boundary_pair_fraction": len(pair_distances) / pair_total if pair_total else 0.0,
        "repeat_disagreement": float(np.mean(repeated)) if repeated else 0.0,
        "image_quality": float(quality), "image_contrast": contrast, "n_annotators": float(len(reads)),
        "repeat_read_count": float(sum(r.is_repeat for r in case.annotations)), "active": float(case.active),
        "annotation_foreground_fraction": annotated_fraction,
        "model_annotation_gap": abs(model_fraction - annotated_fraction),
    }
    if tuple(result) != CONTEXT_FEATURE_NAMES or not np.isfinite(list(result.values())).all():
        raise AssertionError("Invalid public context feature vector")
    return result
