"""Paid synthetic annotation-action environment; NEVER a deployable selector.

Only this simulation environment and independent evaluation may read oracle/.
Action rewards are NOT assigned here: root experiment code must actually
retrain a detector and measure held-out development-risk change.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
from PIL import Image
from scipy import ndimage
from .annotation_state import AnnotationRead, Case, consensus_labelmap, DEVELOPMENT_ACTION_SPLITS
from .dataset import stable_seed, make_simulated_read, _objects, _deform


@dataclass
class ActionOutcome:
    case: Case
    cost_minutes: float
    success: bool
    action: str
    event_id: str
    details: dict

    @property
    def updated_case(self):
        return self.case


def _match_component(mask, objects):
    """Find an oracle object near an existing public component, simulation only."""
    if not objects:
        return None
    ys, xs = np.where(mask); center = np.array([ys.mean(), xs.mean()])
    ranked = []
    for class_id, target in objects:
        intersection = np.logical_and(mask, target).sum()
        union = np.logical_or(mask, target).sum()
        yy, xx = np.where(target)
        distance = np.linalg.norm(center - [yy.mean(), xx.mean()])
        ranked.append((intersection / max(union, 1), -distance, class_id, target))
    ranked.sort(key=lambda r: (r[0], r[1]), reverse=True)
    overlap, negative_distance, class_id, target = ranked[0]
    if overlap > 0 or negative_distance > -6:
        return class_id, target
    return None


class SyntheticOracle:
    """Private fixture oracle with deterministic, paid potential outcomes.

    At most one action per case is supported in the primary experiment. A task
    can fail or find nothing to change, and still incurs its realized cost.
    Different policies receive the same outcome for the same seed/case/action;
    event numbering and candidate traversal do not alter a potential outcome.
    """
    def __init__(self, manifest_path, seed=17):
        self.manifest_path = Path(manifest_path).resolve()
        self.root = self.manifest_path.parent
        self.seed = int(seed)
        public = [json.loads(line) for line in self.manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not public or not all(r.get("is_synthetic") is True for r in public):
            raise ValueError("SyntheticOracle cannot be used as a source of real expert corrections")
        path = self.root / "oracle/truth.jsonl"
        self._private_records = {r["image_id"]: r for r in
                                 (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())}
        self.access_log = []

    def apply(self, case: Case, action: str, event_id="paid_action") -> ActionOutcome:
        if not case.is_synthetic:
            raise ValueError("Real images require a human/action adapter, not SyntheticOracle")
        if case.split not in DEVELOPMENT_ACTION_SPLITS:
            raise ValueError("Actions are forbidden on seed/reward/validation/test roles")
        if case.revision_history:
            raise ValueError("Primary protocol allows at most one paid action per case")
        if action not in case.available_actions:
            raise ValueError(f"Action {action!r} is not publicly available for this case")
        if action == "new_image" and case.active:
            raise ValueError("new_image requires an unlabelled existing-image candidate")
        if action != "new_image" and not case.active:
            raise ValueError("Correction/review actions require existing public annotations")
        quote = float(case.action_quotes_minutes[action])
        if not np.isfinite(quote) or quote <= 0:
            raise ValueError("A positive public maximum-duration quote is required")
        private = self._private_records[case.image_id]
        if private["split"] != case.split:
            raise ValueError("Private/public role disagreement")
        with Image.open(self.root / private["labelmap_path"]) as im:
            truth = np.asarray(im.convert("L"), dtype=np.uint8)
        rng = np.random.default_rng(stable_seed(self.seed, case.image_id, action, "potential_outcome"))
        difficulty = float(private["latent_difficulty"])
        realized_cost = quote * float(rng.uniform(.56, .96))
        completed = bool(rng.random() >= .035 + .10 * difficulty)
        revised = case.clone()
        annotation_changed, image_changed = False, False
        before = consensus_labelmap(case) if case.active else None
        # Raw reads never change; corrections create a separate current revision.
        if completed:
            if action == "new_image":
                for j in range(3):
                    labelmap, uncertainty = make_simulated_read(truth, revised.image, rng, mode="mixed",
                                                               quality=.74, difficulty=difficulty)
                    revised.annotations.append(AnnotationRead(f"SYNTH_PAID_LABELLER_{j}", str(event_id), labelmap,
                                                              False, uncertainty, "synthetic_paid_initial_read"))
                revised.active = True
                annotation_changed = True
            elif action == "review":
                labelmap, uncertainty = make_simulated_read(truth, revised.image, rng, mode="mixed",
                                                           quality=.84, difficulty=difficulty)
                revised.annotations.append(AnnotationRead("SYNTH_EXTRA_REVIEWER", str(event_id), labelmap,
                                                          False, uncertainty, "synthetic_additional_independent_review"))
                annotation_changed = not np.array_equal(consensus_labelmap(revised), before)
            elif action == "reimage":
                with Image.open(self.root / private["clean_image_path"]) as im:
                    clean = np.asarray(im.convert("L"), dtype=float) / 255
                revised.image = np.clip(clean + rng.normal(0, .006 + .010 * difficulty, clean.shape), 0, 1)
                # Geometry is unchanged by construction. Real reimaging needs
                # independent registration/annotation validation, not this trick.
                image_changed = not np.array_equal(revised.image, case.image)
            else:
                current = before.copy()
                existing = _objects(np.where(current == 255, 0, current))
                true_objects = _objects(truth)
                if action == "boundary":
                    for reported_class, component in existing:
                        match = _match_component(component, true_objects)
                        if match is None or rng.random() < .10 + .18 * difficulty:
                            continue
                        _, target = match
                        # Keep the pre-existing reported class: this action does
                        # not silently combine class correction or missing search.
                        corrected = _deform(target, rng, 1) if rng.random() < .12 else target.copy()
                        current[component] = 0
                        current[corrected] = reported_class
                elif action == "class":
                    for _, component in existing:
                        match = _match_component(component, true_objects)
                        if match is not None and rng.random() >= .08 + .15 * difficulty:
                            true_class, _ = match
                            current[component] = true_class
                elif action == "missing":
                    public_foreground = (current > 0) & (current != 255)
                    for true_class, target in true_objects:
                        coverage = np.logical_and(public_foreground, target).sum() / max(target.sum(), 1)
                        if coverage < .25 and rng.random() >= .10 + .22 * difficulty:
                            found = _deform(target, rng, 1) if rng.random() < .2 else target.copy()
                            reported = 3 - true_class if rng.random() < .06 else true_class
                            current[found] = reported
                else:
                    raise ValueError("Unknown action")
                revised.effective_override = current
                annotation_changed = not np.array_equal(current, before)
        revised.available_actions = ()
        # Keep quote history for audit, but no further action remains eligible.
        details = {"is_synthetic": True, "task_completed": completed,
                   "annotation_changed": bool(annotation_changed), "image_changed": bool(image_changed),
                   "quoted_max_minutes": quote, "outcome_semantics": "completion is not positive utility",
                   "reward_assigned_by_oracle": False,
                   "new_image_semantics": "annotate an existing acquired image" if action == "new_image" else None}
        revision = {"event_id": str(event_id), "action": action, "cost_minutes": realized_cost,
                    "success": completed, **details}
        revised.revision_history.append(revision)
        self.access_log.append({"image_id": case.image_id, "split": case.split, "action": action,
                                "event_id": str(event_id), "reason": "paid_synthetic_action_execution"})
        return ActionOutcome(revised, realized_cost, completed, action, str(event_id), details)
