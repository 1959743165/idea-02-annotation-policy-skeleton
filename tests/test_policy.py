"""Meaningful boundary tests for observed-action policy learning and selection."""
import copy,tempfile,unittest
from pathlib import Path
import numpy as np
from annotation_policy.policy import GainCostPolicy,fit_policy,choose_case_action,validate_learning_logs,ACTIONS


def fixture_logs(split="policy_train",n=36):
    rows=[]
    for i in range(n):
        action=ACTIONS[i%6]; available=list(ACTIONS)
        features={"model_uncertainty":(i%9)/9,"class_disagreement":(i%7)/7}
        cost=.7+.2*(i%6)+.01*(i%5)
        rows.append({"event_id":f"{split}_event_{i}","image_id":f"{split}_image_{i}",
                     "specimen_id":f"{split}_specimen_{i}","split":split,"is_synthetic":True,
                     "features":features,"action":action,"available_actions":available,
                     "propensity":1/6,"logging_policy_probs":{a:1/6 for a in available},
                     "cost_minutes":cost,"quoted_max_cost_minutes":3.,
                     "reward":.03*features["model_uncertainty"]-.005*(i%6),
                     "reward_source_split":"reward","reward_specimen_ids":[f"{split}_reward_group"],
                     "reward_panel":"reward_fit" if split=="policy_train" else "reward_tune"})
    return rows


class PolicyTest(unittest.TestCase):
    def test_policy_actually_fits_different_rewards(self):
        a=fixture_logs(); b=copy.deepcopy(a)
        for row in b: row["reward"]=-row["reward"]
        pa,pb=GainCostPolicy.fit(a),GainCostPolicy.fit(b)
        self.assertGreater(np.linalg.norm(pa.coefficients-pb.coefficients),1e-4)
        self.assertGreater(pa.metadata["log_cost_residual_factor"],0)

    def test_normalization_is_train_only_and_roundtrip_is_exact(self):
        train,val=fixture_logs(),fixture_logs("policy_val",12)
        for row in val: row["features"]["model_uncertainty"]+=100
        policy=fit_policy(train,val)
        self.assertLess(float(policy.mean.max()),1)
        with tempfile.TemporaryDirectory() as d:
            policy.save(Path(d)/"policy.npz"); restored=GainCostPolicy.load(Path(d)/"policy.npz")
            self.assertEqual(policy.predict(train[0]["features"],list(ACTIONS)),restored.predict(train[0]["features"],list(ACTIONS)))

    def test_test_reward_wrong_propensity_and_oracle_features_are_rejected(self):
        for field,value in (("split","test"),("reward_source_split","test"),("propensity",0.0)):
            rows=fixture_logs(); rows[0][field]=value
            with self.assertRaises(ValueError): GainCostPolicy.fit(rows)
        rows=fixture_logs(); rows[0]["features"]["latent_difficulty"]=.8
        with self.assertRaises(ValueError): GainCostPolicy.fit(rows)

    def test_train_and_validation_specimens_cannot_overlap(self):
        train,val=fixture_logs(),fixture_logs("policy_val",6)
        val[0]["specimen_id"]=train[0]["specimen_id"]
        with self.assertRaises(ValueError): fit_policy(train,val)

    def test_failed_negative_reward_is_valid_and_real_overrun_is_retained(self):
        rows=fixture_logs(); rows[0].update(success=False,reward=-.2,is_synthetic=False,cost_minutes=4.,quote_overrun_minutes=1.)
        validate_learning_logs(rows,"policy_train")
        self.assertEqual(GainCostPolicy.fit(rows).metadata["training_samples"],len(rows))
        rows[0]["is_synthetic"]=True
        with self.assertRaises(ValueError): validate_learning_logs(rows,"policy_train")

    def test_feature_and_cost_ablations_change_the_trained_model(self):
        rows=fixture_logs()
        full=GainCostPolicy.fit(rows); nod=GainCostPolicy.fit(rows,{"feature_mode":"without_disagreement"})
        only=GainCostPolicy.fit(rows,{"feature_mode":"action_only"})
        self.assertIn("class_disagreement",full.feature_names)
        self.assertNotIn("class_disagreement",nod.feature_names); self.assertEqual(only.feature_names,[])
        self.assertTrue(np.isfinite(only.predict({},list(ACTIONS))["class"]["gain_per_minute"]))

    def test_selector_uses_public_quote_and_cannot_select_unavailable_action(self):
        policy=GainCostPolicy.fit(fixture_logs())
        context={"image_id":"case","features":fixture_logs()[0]["features"],"available_actions":["class","review"],
                 "action_quotes_minutes":{"class":1.5,"review":3.0}}
        d=choose_case_action([context],policy,"learned_gain_per_minute",2.,np.random.default_rng(17))
        self.assertEqual(d["action"],"class")
        self.assertIsNone(choose_case_action([context],policy,"random",1.,np.random.default_rng(17)))

    def test_unobserved_actions_are_not_treated_as_identified_policy_gains(self):
        rows=[r for r in fixture_logs() if r["action"]=="class"]
        policy=GainCostPolicy.fit(rows)
        context={"image_id":"case","features":rows[0]["features"],"available_actions":["review"],"action_quotes_minutes":{"review":1.}}
        self.assertIsNone(choose_case_action([context],policy,"learned_gain_per_minute",2.,np.random.default_rng(17)))
        with self.assertRaises(ValueError): policy.action_probabilities(rows[0]["features"],["review"])

    def test_real_cost_predictions_are_not_forcibly_capped_at_underestimated_quotes(self):
        rows=fixture_logs()
        for row in rows: row.update(is_synthetic=False,cost_minutes=8.,quoted_max_cost_minutes=2.,quote_overrun_minutes=6.)
        policy=GainCostPolicy.fit(rows)
        self.assertGreater(policy.predict(rows[0]["features"],["class"],{"class":2.})["class"]["predicted_cost_minutes"],2.)

if __name__=="__main__": unittest.main()
