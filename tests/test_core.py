"""Core numerical/data-boundary tests, no GPU or real-data claim.

Run from package root: python -m unittest discover -s tests -p 'test_core.py' -v
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from annotation_policy.annotation_state import (ACTIONS, AnnotationRead, Case,
    consensus_labelmap, independent_reads, labelmap_to_patch_targets)
from annotation_policy.dataset import generate_synthetic, load_public_cases
from annotation_policy.detector import Detector, finite_difference_check, objective_and_gradient, train_detector
from annotation_policy.features import CONTEXT_FEATURE_NAMES, describe_case, extract_patch_features
from annotation_policy.oracle import SyntheticOracle


class CoreTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="idea02_core_test_")
        cls.directory = Path(cls.temporary.name)
        cls.manifest = generate_synthetic(cls.directory / "data", seed=17, n_specimens=180)
        cls.cases = load_public_cases(cls.manifest)
        cls.seed_cases = [c for c in cls.cases.values() if c.split == "seed_train"]
        cls.model = train_detector(cls.seed_cases, {"epochs": 45}, seed=17)
        cls.oracle = SyntheticOracle(cls.manifest, seed=17)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def eligible(self, action):
        return next(c for c in self.cases.values() if c.split == "policy_train" and action in c.available_actions)

    def test_analytic_gradient_with_soft_targets_and_missing_pixels(self):
        result = finite_difference_check()
        self.assertTrue(result["passed"])
        self.assertEqual(result["parameters_checked"], 15)
        self.assertLess(result["max_absolute_gradient_error"], 2e-6)

    def test_confident_errors_use_stable_unclipped_cross_entropy(self):
        x = np.array([[1.0]])
        w = np.array([[500.0, -500.0, 0.0]])
        targets = np.array([[0.0, 1.0, 0.0]])
        loss, dw, db = objective_and_gradient(x, targets, np.ones(1), w, np.zeros(3), np.ones(3), 0.0)
        self.assertAlmostEqual(loss, 1000.0)
        self.assertTrue(np.isfinite(dw).all())
        np.testing.assert_allclose(db, [1, -1, 0], atol=1e-12)

    def test_training_performs_nontrivial_updates_and_save_load_is_exact(self):
        self.assertGreater(self.model.diagnostics["parameter_l2_change"], .1)
        self.assertLess(self.model.diagnostics["final_loss"], self.model.diagnostics["initial_loss"])
        path = self.model.save(self.directory / "model_roundtrip.npz")
        restored = Detector.load(path)
        case = next(c for c in self.cases.values() if c.split == "test")
        actual = restored.predict(case)
        np.testing.assert_array_equal(actual, self.model.predict(case))
        self.assertEqual(actual.shape, (8, 8, 3))
        np.testing.assert_allclose(actual.sum(-1), 1, atol=1e-14)

    def test_predictions_depend_on_pixels_without_annotation_access(self):
        class ImageOnly:
            def __init__(self, image):
                self.image = image
            def __getattr__(self, name):
                raise AssertionError(f"Predict accessed a non-image field: {name}")
        case = self.seed_cases[0]
        np.testing.assert_array_equal(self.model.predict(case), self.model.predict(ImageOnly(case.image)))
        changed = np.clip(case.image * .7, 0, 1)
        self.assertGreater(float(abs(self.model.predict(ImageOnly(changed)) - self.model.predict(case)).max()), 1e-4)

    def test_supervision_really_influences_learned_parameters(self):
        changed = [c.clone() for c in self.seed_cases]
        for case in changed:
            case.effective_override = np.zeros_like(case.image, dtype=np.uint8)
        all_background = train_detector(changed, {"epochs": 45}, seed=17)
        self.assertGreater(float(np.linalg.norm(self.model.weights - all_background.weights)), .1)

    def test_fixed_seed_retraining_is_deterministic(self):
        repeated = train_detector(self.seed_cases, {"epochs": 45}, seed=17)
        np.testing.assert_array_equal(self.model.weights, repeated.weights)
        np.testing.assert_array_equal(self.model.bias, repeated.bias)

    def test_heldout_and_real_training_cases_are_rejected(self):
        heldout = self.seed_cases[0].clone()
        heldout.split = "reward"
        with self.assertRaisesRegex(ValueError, "Forbidden"):
            train_detector([heldout], {"epochs": 1})
        real = self.seed_cases[0].clone()
        real.metadata["is_synthetic"] = False
        with self.assertRaisesRegex(ValueError, "synthetic only"):
            train_detector([real], {"epochs": 1})
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            train_detector([self.seed_cases[0], self.seed_cases[0]], {"epochs": 1})

    def test_area_targets_do_not_drop_small_indications_or_recode_unknown(self):
        labels = np.zeros((8, 8), dtype=np.uint8)
        labels[0, 0] = 2
        labels[0:2, 4:8] = 255
        targets, valid = labelmap_to_patch_targets(labels, 2)
        self.assertAlmostEqual(targets[0, 2], 1 / 16)
        self.assertAlmostEqual(targets[1, 0], 1)
        self.assertAlmostEqual(valid[1], .5)
        unknown = np.full((8, 8), 255, dtype=np.uint8)
        target2, valid2 = labelmap_to_patch_targets(unknown, 2)
        np.testing.assert_array_equal(target2, 0)
        np.testing.assert_array_equal(valid2, 0)

    def test_repeat_reads_do_not_become_extra_independent_votes(self):
        case = self.seed_cases[0].clone()
        shape = case.image.shape
        one = np.ones(shape, dtype=np.uint8)
        two = np.full(shape, 2, dtype=np.uint8)
        case.annotations = [AnnotationRead("a", "first", one), AnnotationRead("b", "first", two)]
        case.annotations.extend(AnnotationRead("a", f"repeat_{i}", one, is_repeat=True) for i in range(9))
        self.assertEqual(len(independent_reads(case)), 2)
        self.assertTrue((consensus_labelmap(case) == 255).all())

    def test_role_counts_specimens_and_external_device_are_disjoint(self):
        counts = Counter(c.split for c in self.cases.values())
        self.assertEqual(dict(counts), {"seed_train": 32, "policy_train": 32, "policy_val": 16,
            "reward": 20, "acquisition_pool": 32, "detector_val": 16, "test": 16, "device_test": 16})
        self.assertEqual(len({c.specimen_id for c in self.cases.values()}), len(self.cases))
        external = {c.device_id for c in self.cases.values() if c.split == "device_test"}
        development = {c.device_id for c in self.cases.values() if c.split != "device_test"}
        self.assertFalse(external.intersection(development))
        for case in self.cases.values():
            if case.active:
                self.assertGreaterEqual(len(independent_reads(case)), 3)
            if case.split in {"reward", "detector_val", "test", "device_test"}:
                self.assertFalse(case.active)
                self.assertEqual(case.annotations, [])
                self.assertEqual(case.available_actions, ())

    def test_public_loader_never_opens_private_truth(self):
        real_open = Path.open
        def checked_open(path, *args, **kwargs):
            self.assertNotIn("oracle", Path(path).parts)
            return real_open(path, *args, **kwargs)
        with patch.object(Path, "open", checked_open):
            public = load_public_cases(self.manifest)
        self.assertEqual(len(public), 180)

    def test_public_loader_rejects_injected_oracle_reference(self):
        row = json.loads(self.manifest.read_text(encoding="utf-8").splitlines()[0])
        row["image_path"] = f"oracle/masks/{row['image_id']}.png"
        bad = self.manifest.parent / "bad_public_manifest.jsonl"
        bad.write_text(json.dumps(row) + "\n", encoding="utf-8")
        try:
            with self.assertRaisesRegex(ValueError, "oracle"):
                load_public_cases(bad)
        finally:
            bad.unlink()

    def test_context_uses_public_state_not_private_metadata(self):
        case = self.eligible("boundary")
        first = describe_case(case, self.model)
        changed = case.clone()
        changed.metadata.update({"latent_difficulty": .999, "simulation_error_mode": "class",
                                 "oracle_truth": 2, "future_cost": .001, "test_risk": -100})
        second = describe_case(changed, self.model)
        self.assertEqual(first, second)
        self.assertEqual(tuple(first), CONTEXT_FEATURE_NAMES)
        self.assertTrue(np.isfinite(list(first.values())).all())

    def test_pixel_cache_invalidates_when_pixels_change(self):
        original = self.seed_cases[0].image
        first = extract_patch_features(original)
        changed = original.copy()
        changed[10:20, 10:20] = .05
        second = extract_patch_features(changed)
        self.assertGreater(float(np.linalg.norm(first - second)), .01)
        first[:] = 999
        self.assertLess(float(extract_patch_features(original).max()), 999)

    def test_action_availability_uses_public_state(self):
        inactive = self.eligible("new_image")
        self.assertFalse(inactive.active)
        self.assertEqual(inactive.available_actions, ("new_image",))
        self.assertEqual(inactive.annotations, [])
        for case in self.cases.values():
            if case.active:
                self.assertNotIn("new_image", case.available_actions)
        with self.assertRaisesRegex(ValueError, "not publicly available"):
            self.oracle.apply(inactive, "class")

    def test_all_actions_charge_bounded_cost_and_preserve_raw_reads(self):
        for action in ACTIONS:
            with self.subTest(action=action):
                original = self.eligible(action)
                raw_before = [r.labelmap.copy() for r in original.annotations]
                result = self.oracle.apply(original, action, "unit_action")
                self.assertGreater(result.cost_minutes, 0)
                self.assertLessEqual(result.cost_minutes, original.action_quotes_minutes[action])
                self.assertFalse(result.details["reward_assigned_by_oracle"])
                self.assertEqual(original.revision_history, [])
                self.assertEqual(result.case.available_actions, ())
                for i, before in enumerate(raw_before):
                    np.testing.assert_array_equal(original.annotations[i].labelmap, before)
                    np.testing.assert_array_equal(result.case.annotations[i].labelmap, before)
                with self.assertRaisesRegex(ValueError, "at most one"):
                    self.oracle.apply(result.case, action, "forbidden_second_action")

    def test_potential_outcomes_ignore_action_order_and_event_identifiers(self):
        case = self.eligible("class")
        first = self.oracle.apply(case, "class", "orderA_0")
        self.oracle.apply(self.eligible("review"), "review", "irrelevant_order")
        second = self.oracle.apply(case, "class", "orderB_99")
        self.assertEqual(first.cost_minutes, second.cost_minutes)
        self.assertEqual(first.success, second.success)
        np.testing.assert_array_equal(first.case.image, second.case.image)
        np.testing.assert_array_equal(first.case.effective_override, second.case.effective_override)
        for r1, r2 in zip(first.case.annotations, second.case.annotations):
            np.testing.assert_array_equal(r1.labelmap, r2.labelmap)

    def test_failed_action_is_still_paid_and_cannot_be_retried(self):
        case = self.eligible("review")
        failure = None
        for seed in range(150):
            outcome = SyntheticOracle(self.manifest, seed=seed).apply(case, "review", "test_failure")
            if not outcome.success:
                failure = outcome
                break
        self.assertIsNotNone(failure, "Controlled fixture must exercise at least one failure branch")
        self.assertGreater(failure.cost_minutes, 0)
        self.assertEqual(len(failure.case.annotations), len(case.annotations))
        np.testing.assert_array_equal(failure.case.image, case.image)
        self.assertEqual(len(failure.case.revision_history), 1)
        self.assertEqual(failure.case.available_actions, ())

    def test_snapshot_roundtrip_preserves_observed_state_and_excludes_truth(self):
        for action in ACTIONS:
            with self.subTest(action=action):
                outcome = self.oracle.apply(self.eligible(action), action, f"snapshot_{action}")
                path = outcome.case.save_snapshot(self.directory / f"snapshot_{action}.npz")
                restored = Case.load_snapshot(path)
                np.testing.assert_array_equal(restored.image, outcome.case.image)
                self.assertEqual(restored.metadata, outcome.case.metadata)
                self.assertEqual(restored.revision_history, outcome.case.revision_history)
                self.assertEqual(restored.active, outcome.case.active)
                self.assertEqual(restored.available_actions, ())
                for a, b in zip(restored.annotations, outcome.case.annotations):
                    np.testing.assert_array_equal(a.labelmap, b.labelmap)
                    self.assertEqual(a.annotator_id, b.annotator_id)
                    self.assertEqual(a.read_id, b.read_id)
                    self.assertEqual(a.is_repeat, b.is_repeat)
                if outcome.case.effective_override is not None:
                    np.testing.assert_array_equal(restored.effective_override, outcome.case.effective_override)
                with np.load(path, allow_pickle=False) as stored:
                    meta = json.loads(str(stored["metadata_json"]))
                    self.assertFalse(meta["contains_oracle_truth"])
                    self.assertNotIn("simulation_error_mode", meta["metadata"])
                    self.assertNotIn("latent_difficulty", meta["metadata"])
                np.testing.assert_array_equal(self.model.predict(restored), self.model.predict(outcome.case))

    def test_oracle_cannot_correct_evaluation_roles(self):
        case = next(c for c in self.cases.values() if c.split == "test")
        with self.assertRaisesRegex(ValueError, "forbidden"):
            self.oracle.apply(case, "review", "should_never_happen")


if __name__ == "__main__":
    unittest.main(verbosity=2)
