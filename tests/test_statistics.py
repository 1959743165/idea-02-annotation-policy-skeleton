import copy
import unittest
from annotation_policy.statistics import (step_risk_auc,trajectory_summary,compare_trajectories,
    cluster_ratio_ci,paired_cluster_difference,holm_adjust,estimate_paired_power,summarize_independent_runs)


def event(t, risks):
    per={s:{'risk_sum':v,'image_count':1,'image_ids':[s+'_img'],
            'critical_misses':0,'critical_images':1,'false_clears':0,'defect_images':1,
            'critical_instances_matched':1,'critical_instances':1,'normal_false_alarms':0,'normal_images':0}
         for s,v in risks.items()}
    return {'minutes':t,'risk':sum(risks.values())/len(risks),'per_specimen':per}


class StatisticsTests(unittest.TestCase):
    def test_step_area_no_future_interpolation(self):
        result=step_risk_auc([{'minutes':0,'risk':1},{'minutes':5,'risk':0}],10)
        self.assertEqual(result['normalized_auc'],.5)
        self.assertEqual(result['carry_forward_minutes'],5)

    def test_action_after_horizon_has_no_credit(self):
        result=step_risk_auc([{'minutes':0,'risk':.8},{'minutes':12,'risk':0}],10)
        self.assertEqual(result['normalized_auc'],.8)
        self.assertEqual(result['future_events_ignored'],1)
        self.assertEqual(result['completed_actions'],0)

    def test_endpoint_action_does_not_retroactively_reduce_auc(self):
        result=step_risk_auc([{'minutes':0,'risk':1},{'minutes':10,'risk':0}],10)
        self.assertEqual(result['normalized_auc'],1)
        self.assertEqual(result['final_risk'],0)

    def test_invalid_times_and_missing_baseline_rejected(self):
        for events in ([{'minutes':1,'risk':.5}], [{'minutes':0,'risk':.5},{'minutes':0,'risk':.4}],
                       [{'minutes':0,'risk':.5},{'minutes':-1,'risk':.4}]):
            with self.assertRaises(ValueError): step_risk_auc(events,10)

    def test_null_risk_not_zero(self):
        result=step_risk_auc([{'minutes':0,'risk':None}],10)
        self.assertIsNone(result['normalized_auc'])

    def test_cluster_zero_denominator(self):
        result=cluster_ratio_ci({'a':[0,0],'b':[0,0]},n_bootstrap=50)
        self.assertIsNone(result['estimate']);self.assertIsNone(result['ci'])

    def test_same_specimen_pairing_identical_difference(self):
        a={'x':[1,2],'y':[0,1],'z':[2,3]}
        result=paired_cluster_difference(a,a,n_bootstrap=100,n_permutations=100)
        self.assertEqual(result['difference_B_minus_A'],0)
        self.assertEqual(result['ci'],[0,0])
        self.assertEqual(result['paired_permutation_p_two_sided'],1)

    def test_paired_sets_must_match(self):
        with self.assertRaises(ValueError): paired_cluster_difference({'a':[1,2]},{'b':[1,2]})

    def test_cluster_nauc_and_same_cohort(self):
        curve=[event(0,{'a':1,'b':0}),event(5,{'a':0,'b':0})]
        result=trajectory_summary(curve,10,n_bootstrap=50)
        self.assertEqual(result['normalized_auc'],.25)
        self.assertEqual(result['per_specimen_nauc']['a'],[.5,1])
        broken=copy.deepcopy(curve);broken[1]['per_specimen']['a']['image_ids']=['changed']
        with self.assertRaisesRegex(ValueError,'cohort'):
            trajectory_summary(broken,10,n_bootstrap=20)

    def test_reported_risk_must_match_contributions(self):
        row=event(0,{'a':1,'b':0});row['risk']=.8
        with self.assertRaisesRegex(ValueError,'disagrees'):
            trajectory_summary([row],10,n_bootstrap=20)

    def test_comparison_direction_full_minus_baseline(self):
        a=[event(0,{'a':.8,'b':.6})]
        b=[event(0,{'a':.8,'b':.6}),event(5,{'a':.4,'b':.4})]
        result=compare_trajectories(a,b,10,n_bootstrap=50,n_permutations=50)
        self.assertAlmostEqual(result['primary_normalized_auc']['difference_B_minus_A'],-.15)

    def test_holm_keeps_undefined_family_member(self):
        self.assertEqual(holm_adjust({'a':.01,'b':.04,'c':None}),{'a':.03,'b':.08,'c':None})

    def test_power_is_assumptions_only_reproducible(self):
        result=estimate_paired_power([150],repetitions=50,seed=4)
        again=estimate_paired_power([150],repetitions=50,seed=4)
        self.assertEqual(result,again)
        self.assertEqual(result['analytic_approximation']['complete_specimens'],126)
        self.assertEqual(result['status'],'PLANNING_ASSUMPTIONS_NOT_EMPIRICAL_RESULTS')

    def test_run_uncertainty_is_separate_from_test_bootstrap(self):
        rows=[{'run_id':str(i),'acquisition_seed':i,'heldout_cohort_id':'test-v1','horizon_minutes':60,
               'difference_B_minus_A':v} for i,v in enumerate([-.03,-.01,-.02])]
        result=summarize_independent_runs(rows)
        self.assertAlmostEqual(result['mean_difference_B_minus_A'],-.02)
        self.assertAlmostEqual(result['run_seed_sd'],.01)
        rows[1]['acquisition_seed']=0
        with self.assertRaisesRegex(ValueError,'distinct'):
            summarize_independent_runs(rows)

if __name__=='__main__': unittest.main()
