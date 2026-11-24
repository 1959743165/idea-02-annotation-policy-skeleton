"""Learn conditional annotation gain and time from randomized observed actions.

The learner never receives hidden labels, future outcomes or test-set metrics.
Ridge regression is a transparent reference policy, not a claimed new bandit
algorithm. Static action predictions are surrogates for sequential retraining.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import copy
import json
import math
import numpy as np
from .features import CONTEXT_FEATURE_NAMES

ACTIONS = ("new_image", "boundary", "missing", "class", "review", "reimage")
DISAGREEMENT_FEATURES = {
    "existence_disagreement", "class_disagreement", "boundary_disagreement",
    "boundary_pair_fraction", "repeat_disagreement", "model_annotation_gap",
}


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, records):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for record in records:
            out.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    return path


def validate_learning_logs(records, allowed_split, allowed_feature_names=None):
    if not records:
        raise ValueError("Empty policy-learning logs")
    seen = set()
    for row in records:
        if row.get("split") != allowed_split:
            raise ValueError(f"Expected {allowed_split} logs; train/test/reward mixing is forbidden")
        if row.get("reward_source_split") != "reward":
            raise ValueError("Policy rewards must come from the declared development reward split")
        if row["event_id"] in seen:
            raise ValueError("Duplicate event_id")
        seen.add(row["event_id"])
        available = row["available_actions"]
        if not available or any(a not in ACTIONS for a in available) or row["action"] not in available:
            raise ValueError("Logged action is not available")
        p = float(row["propensity"])
        if not 0 < p <= 1 or not math.isfinite(p):
            raise ValueError("Logging propensity must be known, finite and positive")
        probs = row.get("logging_policy_probs")
        if not isinstance(probs, dict) or set(probs) != set(available):
            raise ValueError("Complete feasible-action logging probabilities are required")
        if any(not math.isfinite(float(v)) or float(v) < 0 for v in probs.values()):
            raise ValueError("Invalid logging action probabilities")
        if not np.isclose(sum(probs.values()), 1) or not np.isclose(probs[row["action"]], p):
            raise ValueError("Inconsistent logging propensity")
        if not -1 <= float(row["reward"]) <= 1 or not math.isfinite(float(row["reward"])):
            raise ValueError("Reward is a measured risk reduction in [-1,1]")
        actual, quote = float(row["cost_minutes"]), float(row["quoted_max_cost_minutes"])
        if not math.isfinite(actual) or not math.isfinite(quote) or actual <= 0 or quote <= 0:
            raise ValueError("Actual cost and predeclared quote must be positive and finite")
        if actual > quote + 1e-9:
            documented_overrun = (row.get("is_synthetic") is False and
                                  math.isclose(float(row.get("quote_overrun_minutes", -1)),actual-quote,abs_tol=1e-8))
            if not documented_overrun:
                raise ValueError("Synthetic cost cannot exceed its maximum quote; real overruns require an explicit preserved record")
        if not row.get("features") or any(not math.isfinite(float(v)) for v in row["features"].values()):
            raise ValueError("Public state features must be finite")
        declared=set(CONTEXT_FEATURE_NAMES if allowed_feature_names is None else allowed_feature_names)
        if not set(row["features"]).issubset(declared):
            raise ValueError("Undeclared policy feature; extensions require an explicit audited allowed_feature_names schema")


@dataclass
class GainCostPolicy:
    feature_names: list[str]
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    inverse_design: np.ndarray
    residual_variance: float
    metadata: dict

    @staticmethod
    def design(z, actions):
        z = np.atleast_2d(np.asarray(z, dtype=float))
        onehot = np.zeros((len(actions), len(ACTIONS)))
        for i, action in enumerate(actions):
            onehot[i, ACTIONS.index(action)] = 1.0
        interactions = np.einsum("nd,na->nad", z, onehot).reshape(len(actions), -1)
        return np.column_stack([np.ones(len(actions)), z, onehot, interactions])

    @classmethod
    def fit(cls, records, config=None, ridge=1.0):
        config = config or {}
        validate_learning_logs(records, "policy_train",config.get("allowed_feature_names"))
        names = sorted(records[0]["features"])
        if any(sorted(r["features"]) != names for r in records):
            raise ValueError("All policy contexts must have the same declared feature schema")
        mode = config.get("feature_mode", "full")
        if mode == "without_disagreement":
            names = [n for n in names if n not in DISAGREEMENT_FEATURES]
        elif mode == "action_only":
            names = []
        elif mode != "full":
            raise ValueError("Unknown policy feature_mode")
        raw = np.asarray([[float(r["features"][name]) for name in names] for r in records], dtype=float)
        mean = raw.mean(0) if names else np.zeros(0)
        scale = np.maximum(raw.std(0), 1e-6) if names else np.ones(0)
        z = (raw - mean) / scale
        x = cls.design(z, [r["action"] for r in records])
        y = np.asarray([[r["reward"], np.log(r["cost_minutes"])] for r in records], dtype=float)
        if config.get("propensity_weighting", True):
            weights = np.minimum([1 / r["propensity"] for r in records], float(config.get("max_inverse_propensity", 10)))
        else:
            weights = np.ones(len(records))
        weights = weights / weights.mean()
        if ridge <= 0:
            raise ValueError("ridge must be positive")
        penalty = np.eye(x.shape[1]) * float(ridge)
        penalty[0, 0] = 1e-8
        system = x.T @ (weights[:, None] * x) + penalty
        inv = np.linalg.inv(system)
        coef = np.linalg.solve(system, x.T @ (weights[:, None] * y))
        residual = y[:, 0] - (x @ coef)[:, 0]
        variance = float(np.average(residual ** 2, weights=weights))
        # exp(E[log cost]) is not E[cost]. A training-only multiplicative residual
        # correction reduces retransformation bias; it is still a point predictor,
        # not a guaranteed conditional expectation under heterogeneous noise.
        log_cost_residual=y[:,1]-(x @ coef)[:,1]
        cost_factor=float(np.average(np.exp(np.clip(log_cost_residual,-8,8)),weights=weights))
        if not config.get("cost_residual_correction",True): cost_factor=1.0
        metadata = {
            "backend":"numpy_contextual_gain_cost_ridge", "ridge":float(ridge), "config":dict(config),
            "training_specimen_ids":sorted({r["specimen_id"] for r in records}),
            "training_event_ids":[r["event_id"] for r in records],
            "observed_actions":sorted({r["action"] for r in records}),
            "training_samples":len(records), "all_synthetic":all(r.get("is_synthetic") is True for r in records),
            "normalization_fitted_on":"policy_train only", "reward_definition":"before risk minus after risk on reward-development specimens",
            "residual_gain_mse":variance, "feature_mode":mode,
            "uncertainty_is_calibrated_confidence_interval":False,
            "log_cost_residual_factor":cost_factor,
            "cost_estimator":"positive point prediction with training-only residual correction; not an exact conditional mean guarantee",
            "propensity_weighting_scope":"supervised ridge loss weighting, not off-policy IPS evaluation",
        }
        return cls(names, mean, scale, coef, inv, variance, metadata)

    def predict(self, features, actions, quotes=None):
        if not actions:
            return {}
        values = np.array([float(features[n]) for n in self.feature_names])
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite policy context")
        z = np.repeat(((values - self.mean) / self.scale)[None, :], len(actions), axis=0)
        x = self.design(z, actions)
        y = x @ self.coefficients
        variance = np.maximum(0, np.einsum("nd,df,nf->n", x, self.inverse_design, x) * self.residual_variance)
        cost_floor = float(self.metadata["config"].get("minimum_predicted_cost", 0.05))
        result = {}
        for i, action in enumerate(actions):
            cost = max(cost_floor, float(np.exp(np.clip(y[i, 1], -8, 8))) * self.metadata.get("log_cost_residual_factor",1.0))
            if quotes is not None and self.metadata["config"].get("cap_predicted_cost_at_quote",self.metadata.get("all_synthetic",False)):
                cost = min(cost, float(quotes[action]))
            gain = float(np.clip(y[i, 0], -1, 1))
            std = float(np.sqrt(variance[i]))
            score = (gain - float(self.metadata["config"].get("uncertainty_penalty", 0)) * std) / cost
            identified = action in self.metadata["observed_actions"]
            result[action] = {"predicted_gain":gain,"predicted_cost_minutes":cost,
                              "gain_standard_error_heuristic":std,"gain_per_minute":score,
                              "action_observed_in_training":identified}
        return result

    def action_probabilities(self, features, actions, quotes=None, temperature=0.1):
        """A within-case policy ONLY. This is not a case-selection probability."""
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        predictions = self.predict(features, actions, quotes)
        observed = [a for a in actions if predictions[a]["action_observed_in_training"]]
        if not observed:
            raise ValueError("No available action has policy-training support")
        scores = np.asarray([predictions[a]["gain_per_minute"] for a in observed]) / temperature
        probabilities = np.exp(scores - scores.max()); probabilities /= probabilities.sum()
        result = {a:0.0 for a in actions}
        result.update({a:float(p) for a,p in zip(observed,probabilities)})
        return result

    def save(self, path):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as out:
            np.savez_compressed(out, mean=self.mean, scale=self.scale, coefficients=self.coefficients,
                                inverse_design=self.inverse_design, residual_variance=self.residual_variance,
                                feature_names=np.asarray(self.feature_names), metadata=json.dumps(self.metadata, ensure_ascii=False))
        return path

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            return cls(data["feature_names"].tolist(), data["mean"], data["scale"], data["coefficients"],
                       data["inverse_design"], float(data["residual_variance"]), json.loads(str(data["metadata"])))


def fit_policy(train_records, val_records, config=None):
    config = config or {}
    validate_learning_logs(train_records, "policy_train",config.get("allowed_feature_names"))
    validate_learning_logs(val_records, "policy_val",config.get("allowed_feature_names"))
    if {r["specimen_id"] for r in train_records} & {r["specimen_id"] for r in val_records}:
        raise ValueError("A specimen crosses policy training and validation")
    reward_scale = max(float(np.std([r["reward"] for r in train_records])), 1e-3)
    cost_scale = max(float(np.std(np.log([r["cost_minutes"] for r in train_records]))), 1e-3)
    candidates = []
    models = []
    for alpha in config.get("ridge_grid", [0.1, 1, 10]):
        model = GainCostPolicy.fit(train_records, config, alpha)
        gain_errors, cost_errors = [], []
        for row in val_records:
            out = model.predict(row["features"], [row["action"]])[row["action"]]
            gain_errors.append(((out["predicted_gain"] - row["reward"]) / reward_scale) ** 2)
            cost_errors.append(((np.log(out["predicted_cost_minutes"]) - np.log(row["cost_minutes"])) / cost_scale) ** 2)
        score = float(np.mean(gain_errors) + float(config.get("cost_loss_weight", .05)) * np.mean(cost_errors))
        candidates.append({"ridge":float(alpha),"validation_score":score,
                           "validation_gain_mse_standardized":float(np.mean(gain_errors)),
                           "validation_log_cost_mse_standardized":float(np.mean(cost_errors))})
        models.append(model)
    selected = int(np.argmin([r["validation_score"] for r in candidates]))
    model = models[selected]
    model.metadata["selection"] = {"split":"policy_val","candidate_scores":candidates,
                                  "selected_ridge":candidates[selected]["ridge"],
                                  "validation_specimen_ids":sorted({r["specimen_id"] for r in val_records})}
    return model


def heuristic_action(features, available):
    """Frozen public-disagreement rule used by strong non-learning baselines."""
    if available == ["new_image"] or set(available) == {"new_image"}:
        return "new_image"
    scores = {
        "missing":float(features.get("existence_disagreement", 0)),
        "class":float(features.get("class_disagreement", 0)),
        "boundary":float(features.get("boundary_disagreement", 0)),
        "review":float(features.get("repeat_disagreement", 0)) + 0.05,
        "reimage":max(0.0, 0.25 - float(features.get("image_quality", 1))),
        "new_image":0.0,
    }
    return max(available, key=lambda action:(scores[action], -ACTIONS.index(action)))


def choose_case_action(contexts, policy, strategy, remaining_minutes, rng):
    """Pure selector: NO oracle, outcomes, ground truth, or test metrics arguments.

    contexts = list of {image_id,features,available_actions,action_quotes_minutes}.
    Feasibility uses a public maximum quote; realized future cost is unavailable.
    """
    if not math.isfinite(remaining_minutes) or remaining_minutes < 0:
        raise ValueError("Remaining budget must be finite and nonnegative")
    if strategy == "no_action":
        return None
    candidates = []
    for context in contexts:
        available = [a for a in context["available_actions"]
                     if 0 < context["action_quotes_minutes"][a] <= remaining_minutes + 1e-9]
        if strategy.startswith("fixed_"):
            action = strategy.removeprefix("fixed_")
            if action not in ACTIONS:
                raise ValueError("Unknown fixed action")
            available = [a for a in available if a == action]
        if not available:
            continue
        if strategy in ("uncertainty_per_minute", "disagreement_rule"):
            available = [heuristic_action(context["features"], available)]
        predicted = policy.predict(context["features"], available, context["action_quotes_minutes"])
        for action in available:
            estimates = predicted[action]
            if strategy in ("learned_gain_per_minute", "learned_gain_without_cost") and not estimates["action_observed_in_training"]:
                continue
            uncertainty = float(context["features"].get("model_uncertainty", 0))
            if strategy == "random":
                score = float(rng.random())
            elif strategy == "cheapest":
                score = -estimates["predicted_cost_minutes"]
            elif strategy == "learned_gain_per_minute":
                score = estimates["gain_per_minute"]
            elif strategy == "learned_gain_without_cost":
                score = estimates["predicted_gain"]
            elif strategy == "disagreement_rule":
                strength = max(float(context["features"].get(k, 0)) for k in DISAGREEMENT_FEATURES)
                score = strength / estimates["predicted_cost_minutes"]
            elif strategy == "uncertainty_per_minute" or strategy.startswith("fixed_"):
                score = uncertainty / estimates["predicted_cost_minutes"]
            else:
                raise ValueError("Unknown strategy: " + strategy)
            candidates.append({"image_id":context["image_id"],"action":action,"score":float(score),
                               "quoted_max_cost_minutes":float(context["action_quotes_minutes"][action]),
                               **estimates})
    if not candidates:
        return None
    # Random secondary key prevents identifier order determining exact-score ties.
    for c in candidates:
        c["_tie"] = float(rng.random())
    chosen = max(candidates, key=lambda c:(c["score"], c["_tie"]))
    chosen.pop("_tie")
    return chosen
