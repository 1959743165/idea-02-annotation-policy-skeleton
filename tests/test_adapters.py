"""Real integration boundaries exercised using explicit synthetic test fixtures."""
import copy,json,sys,tempfile,unittest
from pathlib import Path
import numpy as np
from annotation_policy.adapters import (person_minutes,make_randomized_pilot_request,import_human_completion,
                                        attach_measured_reward,DetectorCommandAdapter,export_current_training_manifest,sha256_file)
from annotation_policy.dataset import generate_synthetic,load_public_cases
from annotation_policy.oracle import SyntheticOracle
from annotation_policy.experiment import write_json
from annotation_policy.policy import read_jsonl

ROOT=Path(__file__).resolve().parents[1]


class HumanAdapterTest(unittest.TestCase):
    def test_person_time_sums_parallel_people_but_rejects_same_person_overlap(self):
        intervals=[{"annotator_id":p,"start_utc":"2026-01-01T00:00:00Z","end_utc":"2026-01-01T00:02:00Z"} for p in ("synthetic_A","synthetic_B")]
        self.assertEqual(person_minutes(intervals),4.)
        intervals[1]["annotator_id"]="synthetic_A"
        with self.assertRaises(ValueError): person_minutes(intervals)

    def test_pending_requests_do_not_claim_work_and_failed_overrun_is_charged(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            context={"image_id":"synthetic_i","specimen_id":"synthetic_s","split":"policy_train","is_synthetic":True,
                     "features":{"model_uncertainty":.5},"available_actions":["review"],"action_quotes_minutes":{"review":1.}}
            make_randomized_pilot_request(context,root/"request.json",protocol_id="fixture",state_version="v0")
            request=json.loads((root/"request.json").read_text())
            self.assertEqual(request["status"],"REQUESTED_NOT_EXECUTED"); self.assertEqual(request["propensity"],1.)
            completion={"event_id":request["event_id"],"status":"failed","reason":"Synthetic failure fixture, no real work",
                        "active_intervals":[{"annotator_id":"synthetic_rater","start_utc":"2026-01-01T00:00:00Z","end_utc":"2026-01-01T00:02:00Z"}]}
            write_json(root/"completion.json",completion)
            paid=import_human_completion(root/"request.json",root/"completion.json",root/"ledger.jsonl")
            self.assertEqual(paid["cost_minutes"],2.); self.assertEqual(paid["quote_overrun_minutes"],1.); self.assertIsNone(paid["reward"])
            with self.assertRaises(ValueError): import_human_completion(root/"request.json",root/"completion.json",root/"ledger.jsonl")

    def test_measured_reward_requires_matching_development_specimens_and_seed(self):
        paid={"split":"policy_train","specimen_id":"candidate","logging_policy_probs":{"class":1.},"propensity":1.,"status":"completed"}
        before={"risk":.3,"split":"reward","specimen_ids":["r1","r2"],"training_specimen_ids":["candidate"],"seed":17,"model_sha256":"fixture_before"}
        after={**before,"risk":.32,"model_sha256":"fixture_after"}
        row=attach_measured_reward(paid,before,after,reward_panel="reward_fit")
        self.assertAlmostEqual(row["reward"],-.02)
        for changed in ({"split":"test"},{"specimen_ids":["candidate"]},{"seed":99}):
            with self.assertRaises(ValueError): attach_measured_reward(paid,before,{**after,**changed},reward_panel="reward_fit")


class CommandAdapterTest(unittest.TestCase):
    def test_current_paid_state_can_train_and_inference_receives_only_images(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); manifest=generate_synthetic(root/"data",n_specimens=40)
            cases=load_public_cases(manifest)
            base=[c for c in cases.values() if c.split=="seed_train"]
            # Select a completed synthetic fixture to exercise activation; failure
            # retention is tested separately and no policy score is assessed here.
            oracle=SyntheticOracle(manifest)
            pairs=[(c,oracle.apply(c,"new_image",event_id="adapter_fixture").case)
                   for c in cases.values() if "new_image" in c.available_actions]
            new,paid=next((original,revised) for original,revised in pairs if revised.active)
            self.assertTrue(paid.active); self.assertFalse(paid.metadata["initial_labeled"])
            training=export_current_training_manifest([*base,paid],root/"current")
            exported=read_jsonl(training); newly=next(r for r in exported if r["image_id"]==new.image_id)
            self.assertTrue(newly["active"]); self.assertFalse(newly["initial_labeled"])
            adapter=DetectorCommandAdapter([sys.executable,str(ROOT/"scripts/reference_adapter_worker.py")])
            fitted=adapter.fit(training,root/"fit",config={"epochs":5})
            # Prediction stays executable with every private oracle file unavailable.
            (root/"data/oracle").rename(root/"data/private_truth_unavailable")
            predicted=adapter.predict(manifest,fitted["model_path"],root/"predict")
            self.assertEqual(len(read_jsonl(predicted["predictions_path"])),40)
            request=json.loads((root/"predict/adapter_request.json").read_text())
            inference=read_jsonl(request["inference_manifest"])
            self.assertTrue(all(not r["annotations"] for r in inference))
            self.assertTrue(all("current_labelmap_path" not in r for r in inference))

    def test_string_command_and_test_role_training_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError): DetectorCommandAdapter("python worker.py")._invoke({},d)
            root=Path(d); manifest=generate_synthetic(root/"data",n_specimens=40)
            cases=load_public_cases(manifest)
            test=next(c for c in cases.values() if c.split=="test"); test.active=True
            with self.assertRaises(ValueError): export_current_training_manifest([test],root/"bad")

if __name__=="__main__": unittest.main()
