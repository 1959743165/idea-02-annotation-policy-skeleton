"""Actually trained NumPy patch classifier for synthetic annotation experiments."""
from __future__ import annotations

import json
from pathlib import Path
import time
from collections.abc import Mapping
import numpy as np
from .annotation_state import DETECTOR_TRAINING_SPLITS, consensus_labelmap, labelmap_to_patch_targets
from .features import PATCH_FEATURE_NAMES, extract_patch_features


def _softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=1, keepdims=True)


def objective_and_gradient(x, targets, valid_fraction, w, b, class_weights, l2):
    """Weighted soft-target CE with an explicit analytic gradient."""
    weighted_targets = targets * np.asarray(class_weights)[None, :]
    denominator = float((weighted_targets.sum(axis=1) * valid_fraction).sum())
    if denominator <= 0:
        raise ValueError("No observed training pixels")
    logits = x @ w + b
    shifted = logits - logits.max(axis=1, keepdims=True)
    # Stable log-softmax keeps the objective and analytic gradient consistent
    # even for extremely confident errors; clipping probabilities would not.
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    probs = np.exp(log_probs)
    loss = -float((valid_fraction[:, None] * weighted_targets * log_probs).sum()) / denominator
    loss += .5 * l2 * float(np.sum(w ** 2))
    derivative = (probs * weighted_targets.sum(axis=1, keepdims=True) - weighted_targets) * valid_fraction[:, None] / denominator
    return loss, x.T @ derivative + l2 * w, derivative.sum(axis=0)


class Detector:
    def __init__(self, weights, bias, feature_mean, feature_std, grid_size=8, metadata=None, diagnostics=None):
        self.weights = np.asarray(weights, dtype=np.float64)
        self.bias = np.asarray(bias, dtype=np.float64)
        self.feature_mean = np.asarray(feature_mean, dtype=np.float64)
        self.feature_std = np.asarray(feature_std, dtype=np.float64)
        self.grid_size = int(grid_size)
        self.metadata = dict(metadata or {})
        self.diagnostics = dict(diagnostics or {})

    def predict(self, case):
        """Only image pixels enter prediction; annotations are never accessed."""
        raw = extract_patch_features(case.image, self.grid_size)
        x = (raw - self.feature_mean) / self.feature_std
        return _softmax(x @ self.weights + self.bias).reshape(self.grid_size, self.grid_size, 3)

    def save(self, path):
        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_suffix(".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {"grid_size": self.grid_size, "metadata": self.metadata, "diagnostics": self.diagnostics}
        np.savez_compressed(path, weights=self.weights, bias=self.bias, feature_mean=self.feature_mean,
                            feature_std=self.feature_std, metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False, allow_nan=False)))
        return path

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"]))
            return cls(data["weights"].copy(), data["bias"].copy(), data["feature_mean"].copy(),
                       data["feature_std"].copy(), metadata["grid_size"], metadata["metadata"], metadata["diagnostics"])


def train_detector(cases, config=None, seed=17):
    """Fit a deterministic reference model to ACTIVE visible annotation states.

    Pass the same config and initialization seed for every counterfactual
    retraining comparison. The backend rejects real-data claims and held-out
    reward/validation/test cases rather than silently fitting their labels.
    """
    started = time.perf_counter()
    config = dict(config or {})
    cfg = dict(config.get("detector", config))
    if isinstance(cases, Mapping):
        cases = list(cases.values())
    else:
        cases = list(cases)
    cases = [case for case in cases if case.active]
    if not cases:
        raise ValueError("No active public training cases")
    if len({c.image_id for c in cases}) != len(cases):
        raise ValueError("Duplicate training image IDs would double-count an annotation")
    for case in cases:
        if not case.is_synthetic:
            raise ValueError("NumPy fixture backend is labelled synthetic only; use an explicitly validated real detector adapter")
        if case.split not in DETECTOR_TRAINING_SPLITS:
            raise ValueError(f"Forbidden downstream training split: {case.split}")
    grid = int(cfg.get("grid_size", 8)); epochs = int(cfg.get("epochs", 60))
    learning_rate = float(cfg.get("learning_rate", .035)); l2 = float(cfg.get("l2", .001))
    class_weights = np.asarray(cfg.get("class_weights", [1., 3., 4.]), dtype=float)
    if epochs <= 0 or learning_rate <= 0 or l2 < 0 or class_weights.shape != (3,) or (class_weights <= 0).any():
        raise ValueError("Invalid detector hyperparameters")
    raw = np.concatenate([extract_patch_features(c.image, grid) for c in cases])
    targets_and_valid = [labelmap_to_patch_targets(consensus_labelmap(c), grid) for c in cases]
    targets = np.concatenate([x[0] for x in targets_and_valid]); valid = np.concatenate([x[1] for x in targets_and_valid])
    mean = raw.mean(axis=0); std = np.maximum(raw.std(axis=0), 1e-3)
    x = (raw - mean) / std
    rng = np.random.default_rng(int(seed))
    w = rng.normal(0, .015, size=(x.shape[1], 3)); b = np.zeros(3)
    initial_w, initial_b = w.copy(), b.copy()
    initial_loss = objective_and_gradient(x, targets, valid, w, b, class_weights, l2)[0]
    wm, wv, bm, bv = np.zeros_like(w), np.zeros_like(w), np.zeros_like(b), np.zeros_like(b)
    history = []
    for epoch in range(1, epochs + 1):
        loss, dw, db = objective_and_gradient(x, targets, valid, w, b, class_weights, l2)
        if not np.isfinite(loss) or not np.isfinite(dw).all() or not np.isfinite(db).all():
            raise FloatingPointError("Nonfinite detector optimization")
        wm = .9 * wm + .1 * dw; wv = .999 * wv + .001 * dw ** 2
        bm = .9 * bm + .1 * db; bv = .999 * bv + .001 * db ** 2
        w -= learning_rate * (wm / (1 - .9 ** epoch)) / (np.sqrt(wv / (1 - .999 ** epoch)) + 1e-8)
        b -= learning_rate * (bm / (1 - .9 ** epoch)) / (np.sqrt(bv / (1 - .999 ** epoch)) + 1e-8)
        history.append(float(loss))
    final_loss = objective_and_gradient(x, targets, valid, w, b, class_weights, l2)[0]
    diagnostics = {"initial_loss": float(initial_loss), "final_loss": float(final_loss),
                   "parameter_l2_change": float(np.sqrt(np.sum((w - initial_w) ** 2) + np.sum((b - initial_b) ** 2))),
                   "epochs": epochs, "finite_training": bool(np.isfinite(final_loss)),
                   "elapsed_seconds": time.perf_counter() - started, "loss_history": history}
    metadata = {"backend": "numpy_synthetic_patch_classifier", "is_synthetic": True,
                "feature_names": list(PATCH_FEATURE_NAMES), "normalization": "active training pixels only",
                "training_image_ids": sorted(c.image_id for c in cases),
                "training_specimen_ids": sorted({c.specimen_id for c in cases}),
                "seed": int(seed), "config": cfg, "class_weights": class_weights.tolist(),
                "probability_warning": "class-weighted CE scores are not externally calibrated real-world probabilities",
                "source_labels": "public plurality or paid append-only revision; never oracle truth"}
    return Detector(w, b, mean, std, grid, metadata, diagnostics)


def finite_difference_check(seed=17, epsilon=1e-6):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(9, 4)); targets = rng.dirichlet(np.ones(3), size=9)
    valid = rng.uniform(.2, 1, size=9); valid[0] = 0
    w = rng.normal(scale=.1, size=(4, 3)); b = rng.normal(scale=.1, size=3)
    class_weights = np.array([1., 3., 4.]); l2 = .02
    _, dw, db = objective_and_gradient(x, targets, valid, w, b, class_weights, l2)
    errors = []
    for parameter, gradient in ((w, dw), (b, db)):
        for index in np.ndindex(parameter.shape):
            original = parameter[index]
            parameter[index] = original + epsilon
            plus = objective_and_gradient(x, targets, valid, w, b, class_weights, l2)[0]
            parameter[index] = original - epsilon
            minus = objective_and_gradient(x, targets, valid, w, b, class_weights, l2)[0]
            parameter[index] = original
            errors.append(abs((plus - minus) / (2 * epsilon) - gradient[index]))
    maximum = float(max(errors))
    return {"passed": maximum < 2e-6, "max_absolute_gradient_error": maximum,
            "parameters_checked": len(errors), "epsilon": epsilon,
            "scope": "NumPy weighted soft-label CE and L2 analytic derivative only"}
