import copy
import unittest
from annotation_policy.offpolicy import evaluate_ope
from annotation_policy.schema import validate_action_logs


def fixture():
    rows=[{'event_id':'e1','specimen_id':'s1','split':'policy_val','is_synthetic':True,
           'available_actions':['boundary','class'],'action':'boundary','propensity':.5,
           'logging_policy_probs':{'boundary':.5,'class':.5},'cost_minutes':2.,'reward':.2},
          {'event_id':'e2','specimen_id':'s2','split':'policy_val','is_synthetic':True,
           'available_actions':['boundary','class'],'action':'class','propensity':.5,
           'logging_policy_probs':{'boundary':.5,'class':.5},'cost_minutes':4.,'reward':-.1}]
    target={'e1':{'boundary':.75,'class':.25},'e2':{'boundary':.5,'class':.5}}
    return rows,target

FIXED={'mode':'predeclared','description':'Fixed probabilities chosen before observing the logged rewards'}
HELD={'mode':'held_out','training_specimen_ids':['independent_train'],'tuning_specimen_ids':[]}


class OffPolicyTests(unittest.TestCase):
    def call(self,rows=None,target=None,**kwargs):
        rr,tt=fixture();kwargs.setdefault('target_policy_provenance',FIXED);kwargs.setdefault('n_bootstrap',50)
        return evaluate_ope(rows if rows is not None else rr,target if target is not None else tt,**kwargs)

    def test_hand_calculated_ips_snips_and_cost(self):
        result=self.call()
        self.assertAlmostEqual(result['ips'],.1)
        self.assertAlmostEqual(result['snips'],.08)
        self.assertAlmostEqual(result['expected_cost_ips'],3.5)
        self.assertAlmostEqual(result['reward_per_expected_minute'],.1/3.5)
        self.assertAlmostEqual(result['effective_sample_size'],2.5**2/3.25)
        self.assertIsNone(result['dr'])

    def test_hand_calculated_dr_with_external_predictions(self):
        q={'e1':{'boundary':.1,'class':.3},'e2':{'boundary':.1,'class':-.2}}
        result=self.call(q_predictions=q,q_provenance=HELD)
        self.assertAlmostEqual(result['dr'],.175)

    def test_dr_rejects_unverified_and_in_sample_q(self):
        q={'e1':{'boundary':.1,'class':.3},'e2':{'boundary':.1,'class':-.2}}
        with self.assertRaisesRegex(ValueError,'provenance'):
            self.call(q_predictions=q)
        bad={**HELD,'training_specimen_ids':['s1']}
        with self.assertRaisesRegex(ValueError,'fitting/tuning'):
            self.call(q_predictions=q,q_provenance=bad)

    def test_target_tuned_on_eval_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'fitting/tuning'):
            self.call(target_policy_provenance={**HELD,'tuning_specimen_ids':['s2']})

    def test_support_checks_unobserved_actions_too(self):
        rows,target=fixture();rows[0]['logging_policy_probs']={'boundary':1.,'class':0.};rows[0]['propensity']=1.
        with self.assertRaisesRegex(ValueError,'overlap'):
            self.call(rows,target)

    def test_complete_vectors_and_propensity_consistency(self):
        rows,target=fixture();rows[0]['logging_policy_probs']={'boundary':.5}
        with self.assertRaisesRegex(ValueError,'complete'):
            self.call(rows,target)
        rows,target=fixture();rows[0]['propensity']=.4
        with self.assertRaisesRegex(ValueError,'disagrees'):
            self.call(rows,target)
        rows,target=fixture();target['e1']={'boundary':.5,'class':.3,'review':.2}
        with self.assertRaisesRegex(ValueError,'exactly'):
            self.call(rows,target)

    def test_zero_negative_or_nonfinite_cost_rejected(self):
        for cost in [0,-1,float('nan')]:
            rows,target=fixture();rows[0]['cost_minutes']=cost
            with self.assertRaisesRegex(ValueError,'cost'):
                self.call(rows,target)

    def test_duplicates_and_test_logs_rejected(self):
        rows,target=fixture();rows[1]['event_id']='e1'
        with self.assertRaisesRegex(ValueError,'duplicate'):
            self.call(rows,target)
        rows,target=fixture();rows[0]['split']='test'
        with self.assertRaisesRegex(ValueError,'policy_train or policy_val'):
            self.call(rows,target)

    def test_clipping_report_preserves_unclipped_estimate(self):
        result=self.call(max_weight=1.)
        self.assertAlmostEqual(result['ips'],.05)
        self.assertAlmostEqual(result['unclipped']['ips'],.1)
        self.assertEqual(result['clipped_events'],1)

    def test_zero_realized_target_overlap_is_not_false_success(self):
        rows,target=fixture();target={'e1':{'boundary':0.,'class':1.},'e2':{'boundary':1.,'class':0.}}
        result=self.call(rows,target)
        self.assertEqual(result['effective_sample_size'],0)
        self.assertIsNone(result['snips'])
        self.assertIsNone(result['reward_per_expected_minute'])

    def test_audited_real_cost_overrun_retained(self):
        rows,_=fixture();rows[0].update({'is_synthetic':False,'quoted_max_cost_minutes':1.,'quote_overrun_minutes':1.})
        result=validate_action_logs(rows);self.assertTrue(result['ok']);self.assertTrue(result['warnings'])
        rows[0]['is_synthetic']=True;self.assertFalse(validate_action_logs(rows)['ok'])

    def test_single_specimen_cluster_does_not_gain_false_precision(self):
        rows,target=fixture();rows[1]['specimen_id']='s1'
        result=self.call(rows,target)
        self.assertIsNone(result['confidence_intervals']['ips']['ci'])

if __name__=='__main__': unittest.main()
