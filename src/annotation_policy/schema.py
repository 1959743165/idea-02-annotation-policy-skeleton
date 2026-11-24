"""Dependency-light record and action-log validation.

Private ground truth is deliberately outside this public manifest validator.
The private synthetic evaluator/environment can inspect it through a separate
interface; action-policy features must not receive it.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image
from .splits import SPLITS, check_group_leakage

ACTIONS = ("new_image", "boundary", "missing", "class", "review", "reimage")
PRIVATE_FIELDS = {"ground_truth", "truth_labelmap_path", "oracle_labelmap_path", "true_class",
                  "true_cost_minutes", "corruption_type", "repair_outcomes", "potential_outcomes"}


def load_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number}: every JSONL record must be an object")
            rows.append(row)
    return rows


def write_json(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path


def write_jsonl(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    return path


def _relative_path(value):
    return isinstance(value, str) and bool(value) and not Path(value).is_absolute() and ".." not in Path(value).parts


def _finite_positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def _hidden_keys(value):
    found = set()
    if isinstance(value, dict):
        found.update(set(value) & PRIVATE_FIELDS)
        for child in value.values(): found.update(_hidden_keys(child))
    elif isinstance(value, list):
        for child in value: found.update(_hidden_keys(child))
    return found


def validate_dataset(manifest_path, check_files=True, protocol="within_domain", check_hashes=False):
    manifest_path = Path(manifest_path); rows = load_jsonl(manifest_path)
    errors, warnings = [], []
    if not rows:
        return {"ok": False, "errors": ["empty public manifest"], "warnings": []}
    seen = set()
    for i, row in enumerate(rows):
        ident = row.get("image_id", f"row_{i+1}")
        for field in ("image_id", "image_path", "specimen_id", "weld_id", "source_image_id", "device_id", "site_id"):
            if not isinstance(row.get(field), str) or not row.get(field):
                errors.append(f"{ident}: {field} must be a nonempty string")
        if ident in seen: errors.append(f"duplicate image_id: {ident}")
        seen.add(ident)
        if row.get("split") not in SPLITS: errors.append(f"{ident}: invalid split")
        if type(row.get("is_synthetic")) is not bool: errors.append(f"{ident}: is_synthetic must be boolean")
        if type(row.get("initial_labeled")) is not bool: errors.append(f"{ident}: initial_labeled must be boolean")
        dimensions_ok = all(type(row.get(k)) is int and row[k] > 0 for k in ("width", "height"))
        if not dimensions_ok: errors.append(f"{ident}: width/height must be positive integers")
        if not _relative_path(row.get("image_path")): errors.append(f"{ident}: image_path must be relative without '..'")
        if _hidden_keys(row): errors.append(f"{ident}: public record contains forbidden oracle fields {sorted(_hidden_keys(row))}")
        actions = row.get("available_actions")
        if not isinstance(actions, list) or any(a not in ACTIONS for a in (actions or [])):
            errors.append(f"{ident}: invalid available_actions"); actions = []
        if len(actions) != len(set(actions)): errors.append(f"{ident}: duplicate available_actions")
        quotes = row.get("action_quotes_minutes")
        if not isinstance(quotes, dict): errors.append(f"{ident}: missing action_quotes_minutes"); quotes = {}
        for action in actions:
            if not _finite_positive(quotes.get(action)): errors.append(f"{ident}: {action} requires a positive finite public time quote")
        if row.get("initial_labeled") is True and "new_image" in actions:
            errors.append(f"{ident}: new_image cannot reacquire an already active case in the single-action protocol")
        if row.get("initial_labeled") is False and any(a != "new_image" for a in actions):
            errors.append(f"{ident}: repair/review actions require currently visible annotations")
        annotations = row.get("annotations")
        if not isinstance(annotations, list): errors.append(f"{ident}: annotations must be a list"); annotations = []
        if row.get("initial_labeled") is False and annotations:
            errors.append(f"{ident}: unlabelled public cases must not expose hidden initial annotations")
        read_ids = set(); annotators = set()
        for number, read in enumerate(annotations):
            prefix = f"{ident}/read_{number}"
            if not isinstance(read, dict): errors.append(prefix + ": annotation must be an object"); continue
            for field in ("annotator_id", "read_id", "labelmap_path"):
                if not isinstance(read.get(field), str) or not read.get(field): errors.append(f"{prefix}: missing {field}")
            read_key = (read.get("annotator_id"), read.get("read_id"))
            if read_key in read_ids: errors.append(f"{prefix}: duplicate annotator_id/read_id pair")
            read_ids.add(read_key); annotators.add(read.get("annotator_id"))
            if type(read.get("is_repeat")) is not bool: errors.append(f"{prefix}: is_repeat must be boolean")
            if not _relative_path(read.get("labelmap_path")): errors.append(f"{prefix}: labelmap_path must be relative without '..'")
            if check_files and _relative_path(read.get("labelmap_path")) and dimensions_ok:
                try:
                    with Image.open(manifest_path.parent / read["labelmap_path"]) as im:
                        array = np.asarray(im)
                    if array.shape != (row["height"], row["width"]): errors.append(f"{prefix}: labelmap dimensions disagree with image")
                    if not np.isin(array, [0, 1, 2, 255]).all(): errors.append(f"{prefix}: reference labelmap classes must be 0,1,2 or255")
                except (OSError, ValueError) as exc: errors.append(f"{prefix}: missing/invalid labelmap: {exc}")
        historical_seed = (row.get("is_synthetic") is False and row.get("split") == "seed_train" and
                           row.get("annotation_provenance") == "authorized_historical_single_reader_training_only")
        if row.get("initial_labeled") is True and len(annotators) < (1 if historical_seed else 3):
            errors.append(f"{ident}: at least {'1 historical seed reader' if historical_seed else '3 distinct independent annotators'} are required; repeats do not count as new experts")
        if historical_seed:
            warnings.append(f"{ident}: declared historical single-reader seed supervision; excluded from multi-expert disagreement evidence")
        if check_files and _relative_path(row.get("image_path")):
            try:
                path = manifest_path.parent / row["image_path"]
                with Image.open(path) as im:
                    if dimensions_ok and im.size != (row["width"], row["height"]): errors.append(f"{ident}: image dimensions disagree")
                    im.verify()
                if check_hashes:
                    expected = row.get("image_sha256")
                    if expected is None: warnings.append(f"{ident}: image_sha256 not supplied; no identity hash checked")
                    elif hashlib.sha256(path.read_bytes()).hexdigest() != expected: errors.append(f"{ident}: image SHA256 mismatch")
            except (OSError, ValueError) as exc: errors.append(f"{ident}: missing/invalid image: {exc}")
    groups = check_group_leakage(rows, protocol)
    errors.extend(groups["errors"]); warnings.extend(groups["warnings"])
    return {"ok": not errors, "errors": errors, "warnings": warnings, "images": len(rows),
            "specimens": len({r.get("specimen_id") for r in rows}),
            "all_synthetic": all(r.get("is_synthetic") is True for r in rows),
            "public_annotations": sum(len(r.get("annotations", [])) for r in rows if isinstance(r.get("annotations"), list)),
            "group_check": groups, "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "limitations": ["Valid JSON and group IDs do not replace expert annotation quality audits.",
                            "Public quotes are budgets/limits, not observed prospective action costs."]}


def validate_action_logs(records):
    """Validate actual paid action logs, retaining failures and negative rewards."""
    errors = []; warnings = []; seen = set()
    for i, row in enumerate(records):
        ident = row.get("event_id", f"row_{i+1}")
        if not isinstance(row.get("event_id"), str) or not row.get("event_id"): errors.append(f"{ident}: event_id required")
        if ident in seen: errors.append(f"duplicate event_id: {ident}")
        seen.add(ident)
        if not isinstance(row.get("specimen_id"), str) or not row.get("specimen_id"): errors.append(f"{ident}: specimen_id required")
        actions = row.get("available_actions")
        if not isinstance(actions, list) or not actions or len(actions) != len(set(actions)) or any(a not in ACTIONS for a in actions):
            errors.append(f"{ident}: invalid nonempty available_actions"); actions = []
        action = row.get("action")
        if action not in actions: errors.append(f"{ident}: logged action was unavailable")
        propensity = row.get("propensity")
        if not _finite_positive(propensity) or propensity > 1: errors.append(f"{ident}: selected-action propensity must be in (0,1]")
        if not _finite_positive(row.get("cost_minutes")): errors.append(f"{ident}: realized cost must be finite and positive, including failed actions")
        reward = row.get("reward")
        if not isinstance(reward, (int, float)) or isinstance(reward, bool) or not math.isfinite(reward): errors.append(f"{ident}: finite reward required; negative reward is allowed")
        if row.get("split") not in {"policy_train", "policy_val"}: errors.append(f"{ident}: learning/OPE action logs may only use policy_train or policy_val")
        quote = row.get("quote_minutes", row.get("quoted_max_cost_minutes"))
        if quote is not None:
            if not _finite_positive(quote):
                errors.append(f"{ident}: invalid positive public quote")
            elif _finite_positive(row.get("cost_minutes")) and row["cost_minutes"] > quote + 1e-8:
                excess = row["cost_minutes"] - quote
                recorded = row.get("quote_overrun_minutes")
                if row.get("is_synthetic") is False and _finite_positive(recorded) and abs(recorded-excess) <= 1e-8:
                    warnings.append(f"{ident}: audited real-action cost overrun retained in full; do not discard the event")
                else:
                    errors.append(f"{ident}: realized cost exceeds quote without an explicit valid real-action overrun record")
    return {"ok": not errors, "errors": errors, "warnings": warnings, "records": len(records)}
