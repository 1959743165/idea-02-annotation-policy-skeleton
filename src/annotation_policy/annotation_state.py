"""Public annotation state. This module never opens oracle ground truth."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import copy
import json
from pathlib import Path
import numpy as np

ACTIONS = ("new_image", "boundary", "missing", "class", "review", "reimage")
DEVELOPMENT_ACTION_SPLITS = frozenset(("policy_train", "policy_val", "acquisition_pool"))
DETECTOR_TRAINING_SPLITS = frozenset(("seed_train", "policy_train", "policy_val", "acquisition_pool"))


@dataclass
class AnnotationRead:
    annotator_id: str
    read_id: str
    labelmap: np.ndarray
    is_repeat: bool = False
    uncertainty: float = 0.0
    provenance: str = "synthetic_rater_not_human"


@dataclass
class Case:
    image_id: str
    specimen_id: str
    split: str
    device_id: str
    image: np.ndarray
    annotations: list[AnnotationRead]
    active: bool
    available_actions: tuple[str, ...]
    action_quotes_minutes: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)
    effective_override: np.ndarray | None = None
    revision_history: list[dict[str, Any]] = field(default_factory=list)

    @property
    def width(self):
        return int(self.image.shape[1])

    @property
    def height(self):
        return int(self.image.shape[0])

    @property
    def is_synthetic(self):
        return self.metadata.get("is_synthetic") is True

    def clone(self):
        """No action may overwrite the raw public data shared by other arms."""
        return copy.deepcopy(self)

    def save_snapshot(self, path):
        """Persist paid/visible annotation state without any oracle fields."""
        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_suffix(".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        forbidden = {"latent_difficulty", "simulation_error_mode", "ground_truth", "oracle_truth", "true_class"}
        if forbidden.intersection(self.metadata):
            raise ValueError("Private oracle fields cannot be exported in a public Case snapshot")
        read_metadata = []
        arrays = {"image": np.asarray(self.image, dtype=np.float64)}
        for i, read in enumerate(self.annotations):
            key = f"annotation_{i}"
            arrays[key] = np.asarray(read.labelmap, dtype=np.uint8)
            read_metadata.append({"array_key": key, "annotator_id": read.annotator_id,
                                  "read_id": read.read_id, "is_repeat": read.is_repeat,
                                  "uncertainty": read.uncertainty, "provenance": read.provenance})
        if self.effective_override is not None:
            arrays["effective_override"] = np.asarray(self.effective_override, dtype=np.uint8)
        meta = {"snapshot_version": "1.0", "image_id": self.image_id, "specimen_id": self.specimen_id,
                "split": self.split, "device_id": self.device_id, "active": self.active,
                "available_actions": list(self.available_actions), "action_quotes_minutes": self.action_quotes_minutes,
                "metadata": self.metadata, "annotations": read_metadata,
                "has_effective_override": self.effective_override is not None,
                "revision_history": self.revision_history,
                "contains_oracle_truth": False}
        arrays["metadata_json"] = np.asarray(json.dumps(meta, ensure_ascii=False, allow_nan=False))
        np.savez_compressed(path, **arrays)
        return path

    @classmethod
    def load_snapshot(cls, path):
        """Restore only stored public state; no image/label file is re-opened."""
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata_json"]))
            if meta.get("contains_oracle_truth") is not False:
                raise ValueError("Snapshot has no explicit public-state provenance")
            reads = [AnnotationRead(r["annotator_id"], r["read_id"], data[r["array_key"]].copy(),
                                    bool(r["is_repeat"]), float(r["uncertainty"]), r["provenance"])
                     for r in meta["annotations"]]
            return cls(meta["image_id"], meta["specimen_id"], meta["split"], meta["device_id"],
                       data["image"].copy(), reads, bool(meta["active"]), tuple(meta["available_actions"]),
                       {k: float(v) for k, v in meta["action_quotes_minutes"].items()}, meta["metadata"],
                       data["effective_override"].copy() if meta["has_effective_override"] else None,
                       meta["revision_history"])


def independent_reads(case: Case) -> list[AnnotationRead]:
    """One latest non-repeat read per rater; repeated reads are not extra votes."""
    by_rater: dict[str, AnnotationRead] = {}
    for read in case.annotations:
        if not read.is_repeat:
            by_rater[read.annotator_id] = read
    return list(by_rater.values())


def consensus_labelmap(case: Case) -> np.ndarray:
    """Public plurality label map; ties/unobserved pixels are 255 (ignored).

    This is a *training aggregation*, not an independent reference standard.
    Expert disagreement is retained in each original read and separately used
    as an action-policy input. A paid correction is an append-only revision.
    """
    if not case.active:
        raise ValueError("An unlabelled/inactive case has no training consensus")
    if case.effective_override is not None:
        out = np.asarray(case.effective_override, dtype=np.uint8)
        if out.shape != case.image.shape:
            raise ValueError("Revision label map is not aligned with the image")
        return out.copy()
    reads = independent_reads(case)
    if not reads:
        raise ValueError("An active case has no independent annotation reads")
    for read in reads:
        if read.labelmap.shape != case.image.shape:
            raise ValueError("Public annotation geometry does not match the image")
        if not np.isin(read.labelmap, (0, 1, 2, 255)).all():
            raise ValueError("Label maps use 0,1,2 or explicit unknown 255")
    counts = np.stack([sum((r.labelmap == c).astype(np.int16) for r in reads) for c in range(3)], axis=-1)
    maximum = counts.max(axis=-1)
    out = counts.argmax(axis=-1).astype(np.uint8)
    out[((counts == maximum[..., None]).sum(axis=-1) != 1) | (maximum == 0)] = 255
    return out


def labelmap_to_patch_targets(labelmap: np.ndarray, grid_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Area-averaged one-hot targets and observed-pixel fractions per patch.

    It preserves fractional foreground coverage, unlike resizing a categorical
    map with nearest-neighbour and then counting its pixels. Unknown=255 is
    excluded from CE rather than recoded as background. Grid must divide image.
    """
    labels = np.asarray(labelmap)
    if labels.ndim != 2 or not np.isin(labels, (0, 1, 2, 255)).all():
        raise ValueError("Expected a 2-D 0/1/2/255 annotation map")
    height, width = labels.shape
    if grid_size <= 0 or height % grid_size or width % grid_size:
        raise ValueError("Reference grid must divide image dimensions exactly")
    ph, pw = height // grid_size, width // grid_size
    observed = (labels != 255).reshape(grid_size, ph, grid_size, pw).transpose(0, 2, 1, 3)
    counts = np.stack([(labels == c).reshape(grid_size, ph, grid_size, pw).transpose(0, 2, 1, 3).sum(axis=(2, 3))
                       for c in range(3)], axis=-1).astype(np.float64)
    valid_count = observed.sum(axis=(2, 3)).astype(np.float64)
    targets = counts / np.maximum(valid_count[..., None], 1.0)
    fraction = valid_count / (ph * pw)
    return targets.reshape(-1, 3), fraction.reshape(-1)


def public_state_digest(case: Case) -> dict:
    """Small observed-state record for logs; it includes no prospective outcomes."""
    return {"image_id": case.image_id, "specimen_id": case.specimen_id,
            "split": case.split, "active": bool(case.active),
            "n_independent_raters": len(independent_reads(case)),
            "n_reads": len(case.annotations), "revision_count": len(case.revision_history),
            "available_actions": list(case.available_actions),
            "is_synthetic": case.is_synthetic}
