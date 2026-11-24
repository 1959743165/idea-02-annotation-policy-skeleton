"""Independent, specimen-aware risk evaluation for Idea 02.

The reference task has synthetic classes 0=background, 1=round indication,
2=linear indication. Calling class 2 "critical" is a declared software proxy,
not a weld acceptance rule or a claim about physical failure severity.
"""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import hashlib
import json
import math
import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from .schema import load_jsonl

DEFAULT_CONFIG = {"foreground_threshold": .35, "class_weights": [1.0, 1.0, 3.0],
                  "matching_iou": .5, "min_predicted_component_pixels": 1,
                  "n_bootstrap": 0, "bootstrap_seed": 17}


def case_value(case, key, default=None):
    if isinstance(case, dict):
        return case.get(key, case.get("metadata", {}).get(key, default))
    if hasattr(case, key):
        return getattr(case, key)
    return getattr(case, "metadata", {}).get(key, default)


def _case_list(cases):
    return list(cases.values()) if hasattr(cases, "values") else list(cases)


def load_truth(manifest_path, split=None, truth_path=None):
    """Read independent labels ONLY for the requested evaluation role(s).

    Synthetic default: ``manifest_dir/oracle/truth.jsonl``. Every truth row has
    image_id and labelmap_path, whose path is relative to the public manifest
    directory. Real data require an explicitly supplied, independently audited
    truth_path. The returned dict contains label arrays, never clean images or
    future annotation-action outcomes. No policy feature function calls this.
    """
    from PIL import Image
    manifest_path = Path(manifest_path)
    public = load_jsonl(manifest_path)
    roles = None if split is None else {split} if isinstance(split, str) else set(split)
    selected = {r["image_id"]: r for r in public if roles is None or r["split"] in roles}
    if not selected:
        raise ValueError("The requested truth evaluation role is empty")
    if truth_path is None:
        if not all(r.get("is_synthetic") is True for r in selected.values()):
            raise ValueError("Real evaluation requires an explicit independent adjudication truth_path")
        truth_path = manifest_path.parent / "oracle" / "truth.jsonl"
        if not truth_path.is_file():
            legacy = manifest_path.parent / "oracle" / "ground_truth.jsonl"
            if legacy.is_file(): truth_path = legacy
    else:
        truth_path = Path(truth_path)
        if not truth_path.is_absolute(): truth_path = manifest_path.parent / truth_path
    result = {}
    for row in load_jsonl(truth_path):
        ident = row.get("image_id")
        if ident not in selected:
            continue
        if ident in result:
            raise ValueError("Duplicate independent truth image_id: " + ident)
        relative = Path(row["labelmap_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Truth labelmap paths must be relative to manifest root without '..'")
        with Image.open(manifest_path.parent / relative) as im:
            array = np.asarray(im)
        if array.shape != (selected[ident]["height"], selected[ident]["width"]):
            raise ValueError("Independent truth/image shape mismatch: " + ident)
        if not np.isin(array, [0, 1, 2, 255]).all():
            raise ValueError("Reference truth supports class0/1/2 and255 unknown only")
        if row.get("specimen_id") is not None and row["specimen_id"] != selected[ident]["specimen_id"]:
            raise ValueError("Independent truth has inconsistent specimen identity: " + ident)
        if row.get("split") is not None and row["split"] != selected[ident]["split"]:
            raise ValueError("Independent truth has inconsistent split: " + ident)
        result[ident] = np.asarray(array, dtype=np.uint8).copy()
    missing = set(selected) - set(result)
    if missing:
        raise ValueError("Missing independent truth; never substitute consensus labels: " + ", ".join(sorted(missing)[:10]))
    return result


def probability_grid_to_labels(probabilities, height, width, foreground_threshold=.35):
    """Apply the frozen decision rule, then invert uniform patch-grid geometry.

    Select the larger foreground-class probability if it reaches the fixed
    threshold; otherwise output background. This is not threshold tuning.
    """
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 3 or p.shape[-1] != 3 or min(p.shape[:2]) < 1:
        raise ValueError("Prediction must be [grid_h,grid_w,3] probabilities")
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any() or not np.allclose(p.sum(-1), 1, atol=1e-4):
        raise ValueError("Prediction probabilities must be finite, nonnegative and sum to one")
    if not 0 < foreground_threshold < 1 or height <= 0 or width <= 0:
        raise ValueError("Invalid decision threshold or image size")
    fg = np.argmax(p[..., 1:], axis=-1) + 1
    grid = np.where(np.max(p[..., 1:], axis=-1) >= foreground_threshold, fg, 0)
    # Half-open equal-area patch cells map to original pixel centers.
    yy = np.minimum(((np.arange(height) + .5) * p.shape[0] / height).astype(int), p.shape[0] - 1)
    xx = np.minimum(((np.arange(width) + .5) * p.shape[1] / width).astype(int), p.shape[1] - 1)
    return grid[yy[:, None], xx[None, :]].astype(np.uint8)


def component_matches(truth_mask, prediction_mask, matching_iou=.5, min_predicted_pixels=1):
    """Maximum-cardinality eligible component matching, then maximum IoU.

    Four-connectivity and the matching threshold are predeclared. This is a
    component-based dense-map diagnostic, not COCO object-detection AP.
    """
    if not 0 < matching_iou <= 1 or min_predicted_pixels < 1:
        raise ValueError("Invalid component matching configuration")
    structure = ndimage.generate_binary_structure(2, 1)
    gt, ngt = ndimage.label(np.asarray(truth_mask, dtype=bool), structure)
    pr, npr = ndimage.label(np.asarray(prediction_mask, dtype=bool), structure)
    predictions = [pr == i for i in range(1, npr + 1) if int((pr == i).sum()) >= min_predicted_pixels]
    truths = [gt == i for i in range(1, ngt + 1)]
    ious = np.zeros((len(truths), len(predictions)), dtype=float)
    for i, gm in enumerate(truths):
        for j, pm in enumerate(predictions):
            union = np.count_nonzero(gm | pm)
            ious[i, j] = np.count_nonzero(gm & pm) / union if union else 0
    matched = []
    if ious.size:
        eligible = ious >= matching_iou
        score = eligible.astype(float) * (min(ious.shape) + 1) + ious
        left, right = linear_sum_assignment(-score)
        matched = [float(ious[i, j]) for i, j in zip(left, right) if eligible[i, j]]
    return {"true_components": len(truths), "predicted_components": len(predictions),
            "matched_components": len(matched), "matched_ious": matched}


def _divide(numerator, denominator):
    return float(numerator / denominator) if denominator > 0 else None


def _prediction_mapping(predictions):
    if isinstance(predictions, dict):
        return {key: value.get("probabilities", value) if isinstance(value, dict) else value
                for key, value in predictions.items()}
    mapping = {}
    for record in predictions:
        ident = record["image_id"]
        if ident in mapping: raise ValueError("Duplicate prediction image_id")
        mapping[ident] = record["probabilities"]
    return mapping


def evaluate_predictions(predictions, cases, truth, config=None):
    """Evaluate a fixed cohort; no held-out label affects the model or policy.

    ``risk_i`` is a class-normalized weighted pixel error. For each known class
    present in image i, e_c is its misclassified-pixel fraction. The per-image
    loss is sum(w_c e_c)/sum(w_c). Average images within each specimen, then
    average specimens equally. Background e_0 is the false-foreground fraction.

    Missing output gets a predeclared maximal-risk penalty and is retained in
    the cohort; it is never silently dropped. Independent truth is mandatory.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    threshold = float(cfg["foreground_threshold"])
    if not math.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("foreground_threshold must be finite and between zero and one")
    weights = np.asarray(cfg["class_weights"], dtype=float)
    if weights.shape != (3,) or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("class_weights must contain three finite positive weights")
    cases = _case_list(cases)
    if not cases: raise ValueError("Evaluation cohort is empty")
    ids = [case_value(c, "image_id") for c in cases]
    if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("Evaluation case IDs must be nonempty and unique")
    mapping = _prediction_mapping(predictions)
    if set(mapping) - set(ids):
        raise ValueError("Prediction includes images outside the fixed evaluation cohort")
    roles = {case_value(c, "split", "unspecified") for c in cases}
    if len(roles) > 1 and not cfg.get("allow_mixed_splits", False):
        raise ValueError("Evaluate one role at a time; mixed-split summaries need explicit exploratory opt-in")
    per_specimen = {}; per_image = []; matrix = np.zeros((3, 3), dtype=np.int64)
    reference_hashes = defaultdict(dict)
    for case in cases:
        ident = case_value(case, "image_id"); sid = str(case_value(case, "specimen_id", ident))
        if ident not in truth: raise ValueError("Independent truth missing for " + ident)
        value = truth[ident]
        gt = np.asarray(value.get("labelmap") if isinstance(value, dict) else value)
        if gt.ndim != 2 or not np.isin(gt, [0, 1, 2, 255]).all():
            raise ValueError("Independent labels must be a 2D 0/1/2/255 map")
        reference_hashes[sid][ident] = hashlib.sha256(np.asarray(gt,dtype=np.uint8).tobytes() + str(gt.shape).encode()).hexdigest()
        known = gt != 255; missing = ident not in mapping
        pred = np.zeros_like(gt) if missing else probability_grid_to_labels(mapping[ident], *gt.shape, threshold)
        present = [c for c in range(3) if np.any(gt == c)]
        errors = {str(c): float(np.mean(pred[gt == c] != c)) for c in present}
        image_risk = (float(sum(weights[c] * errors[str(c)] for c in present) / sum(weights[c] for c in present))
                      if present else None)
        if missing and image_risk is not None: image_risk = 1.0
        if known.any():
            np.add.at(matrix, (gt[known].astype(int), pred[known].astype(int)), 1)
        true_critical = bool(np.any(gt == 2)); predicted_critical = bool(np.any((pred == 2) & known))
        true_defect = bool(np.any((gt == 1) | (gt == 2)))
        predicted_defect = bool(np.any(((pred == 1) | (pred == 2)) & known))
        critical_match = component_matches(gt == 2, (pred == 2) & known,
                                           float(cfg["matching_iou"]), int(cfg["min_predicted_component_pixels"]))
        parts = per_specimen.setdefault(sid, {
            "risk_sum": 0.0, "image_count": 0, "cohort_images": 0, "critical_misses": 0, "critical_images": 0,
            "false_clears": 0, "defect_images": 0, "critical_instances_matched": 0, "critical_instances": 0,
            "normal_false_alarms": 0, "normal_images": 0, "missing_predictions": 0,
            "known_pixels": 0, "unknown_pixels": 0, "image_ids": []})
        parts["cohort_images"] += 1; parts["image_ids"].append(ident)
        if image_risk is not None:
            parts["risk_sum"] += image_risk; parts["image_count"] += 1
        parts["critical_images"] += int(true_critical)
        parts["critical_misses"] += int(true_critical and not predicted_critical)
        parts["defect_images"] += int(true_defect)
        parts["false_clears"] += int(true_defect and not predicted_defect)
        parts["critical_instances"] += critical_match["true_components"]
        parts["critical_instances_matched"] += critical_match["matched_components"]
        is_normal = known.all() and not true_defect
        parts["normal_images"] += int(is_normal)
        parts["normal_false_alarms"] += int(is_normal and predicted_defect)
        parts["missing_predictions"] += int(missing)
        parts["known_pixels"] += int(known.sum()); parts["unknown_pixels"] += int((~known).sum())
        per_image.append({"image_id": ident, "specimen_id": sid, "risk": image_risk,
                          "class_error_fractions": errors, "known_fraction": float(known.mean()),
                          "truth_has_critical_proxy": true_critical, "prediction_has_critical_proxy": predicted_critical,
                          "truth_has_indication": true_defect, "prediction_has_indication": predicted_defect,
                          "missing_prediction": missing, "critical_component_match": critical_match})
    for sid, parts in per_specimen.items():
        parts["risk"] = _divide(parts["risk_sum"], parts["image_count"])
        parts["reference_sha256"] = hashlib.sha256(json.dumps(reference_hashes[sid],sort_keys=True).encode()).hexdigest()
    risks = [p["risk"] for p in per_specimen.values() if p["risk"] is not None]
    sums = {key: sum(p[key] for p in per_specimen.values()) for key in
            ("image_count", "cohort_images", "critical_misses", "critical_images", "false_clears", "defect_images",
             "critical_instances_matched", "critical_instances", "normal_false_alarms", "normal_images",
             "missing_predictions", "known_pixels", "unknown_pixels")}
    risk = float(np.mean(risks)) if risks else None
    metrics = {"risk": risk,
               "critical_miss_rate": _divide(sums["critical_misses"], sums["critical_images"]),
               "false_clear_rate": _divide(sums["false_clears"], sums["defect_images"]),
               "critical_instance_recall": _divide(sums["critical_instances_matched"], sums["critical_instances"]),
               "normal_false_alarm_rate": _divide(sums["normal_false_alarms"], sums["normal_images"])}
    ious = {}
    for c in range(3):
        union = matrix[c].sum() + matrix[:, c].sum() - matrix[c, c]
        ious[str(c)] = _divide(matrix[c, c], union)
    defined_ious = [x for x in ious.values() if x is not None]
    metrics["mean_iou"] = float(np.mean(defined_ious)) if defined_ious else None
    all_synthetic = all(case_value(c, "is_synthetic", False) is True for c in cases)
    unavailable = {key: "zero_eligible_denominator" for key, value in metrics.items() if value is None}
    report = {**metrics, "status": "SYNTHETIC_SOFTWARE_EVALUATION_ONLY" if all_synthetic else "REAL_EVALUATION_REQUIRES_INDEPENDENT_AUDIT",
              "all_synthetic": all_synthetic, "split": sorted(roles), "counts": sums,
              "specimens": len(per_specimen), "risk_eligible_specimens": len(risks),
              "risk_estimand": "mean_of_within_specimen_mean_class_normalized_pixel_risk",
              "rate_estimand": "pooled_eligible_image_or_component_counts_with_specimen_cluster_inference",
              "config": {k: cfg[k] for k in ("foreground_threshold", "class_weights", "matching_iou", "min_predicted_component_pixels")},
              "per_specimen": per_specimen, "per_image": per_image, "pixel_confusion_matrix": matrix.tolist(),
              "class_iou": ious, "undefined_metrics": unavailable,
              "limitations": ["Class 2 is a synthetic critical-indication proxy, not an engineering acceptance category.",
                              "Critical miss and false-clear rates are image-presence diagnostics; also inspect instance recall and localization.",
                              "Consensus training annotations are not the independent evaluation truth.",
                              "All-unknown images cannot establish correctness; their missing label coverage is reported.",
                              "A missing prediction incurs maximal risk rather than dropping a difficult image.",
                              "One fixed acquisition trajectory does not quantify policy-training or deployment-site uncertainty."]}
    if int(cfg.get("n_bootstrap", 0)) > 0:
        from .statistics import metric_confidence_intervals
        report["confidence_intervals"] = metric_confidence_intervals(per_specimen, int(cfg["n_bootstrap"]), int(cfg.get("bootstrap_seed", 17)))
    return report


def evaluate_detector(model, cases, truth, config=None):
    """Evaluate an already trained model; optimization is outside this function."""
    cohort = _case_list(cases)
    predictions = {case_value(case, "image_id"): model.predict(case) for case in cohort}
    return evaluate_predictions(predictions, cohort, truth, config)
