import unittest
import numpy as np
from annotation_policy.evaluation import evaluate_predictions, evaluate_detector, probability_grid_to_labels, component_matches


def case(i, specimen=None):
    return {'image_id':i,'specimen_id':specimen or i,'split':'test','is_synthetic':True}


def probability(labels):
    return np.eye(3)[np.asarray(labels,dtype=int)]


class EvaluationTests(unittest.TestCase):
    def test_perfect_map(self):
        gt=np.array([[0,1],[2,0]])
        result=evaluate_predictions({'a':probability(gt)},[case('a')],{'a':gt})
        self.assertEqual(result['risk'],0)
        self.assertEqual(result['critical_miss_rate'],0)
        self.assertEqual(result['false_clear_rate'],0)
        self.assertEqual(result['critical_instance_recall'],1)
        self.assertEqual(result['mean_iou'],1)

    def test_class_normalized_weighted_risk(self):
        gt=np.array([[0,0],[1,2]])
        result=evaluate_predictions({'a':probability(np.zeros_like(gt))},[case('a')],{'a':gt})
        self.assertAlmostEqual(result['risk'],4/5)
        self.assertEqual(result['critical_miss_rate'],1)
        self.assertEqual(result['false_clear_rate'],1)

    def test_specimen_macro_not_image_micro(self):
        cases=[case('a','shared'),case('b','shared'),case('c','alone')]
        truth={c['image_id']:np.zeros((2,2),int) for c in cases}
        pred={'a':probability(np.zeros((2,2))), 'b':probability(np.zeros((2,2))), 'c':probability(np.ones((2,2)))}
        result=evaluate_predictions(pred,cases,truth)
        self.assertAlmostEqual(result['risk'],.5)

    def test_zero_denominator_is_null(self):
        gt=np.zeros((2,2),int)
        result=evaluate_predictions({'a':probability(gt)},[case('a')],{'a':gt})
        self.assertIsNone(result['critical_miss_rate'])
        self.assertIsNone(result['false_clear_rate'])
        self.assertIn('critical_miss_rate',result['undefined_metrics'])

    def test_unknown_coverage_not_success(self):
        result=evaluate_predictions({'a':probability(np.zeros((2,2)))},[case('a')],{'a':np.full((2,2),255)})
        self.assertIsNone(result['risk'])
        self.assertEqual(result['counts']['unknown_pixels'],4)
        self.assertEqual(result['risk_eligible_specimens'],0)

    def test_missing_prediction_penalty_preserves_cohort(self):
        result=evaluate_predictions({},[case('a')],{'a':np.zeros((2,2),int)})
        self.assertEqual(result['risk'],1)
        self.assertEqual(result['counts']['missing_predictions'],1)
        self.assertEqual(result['per_specimen']['a']['image_count'],1)

    def test_independent_truth_cannot_be_missing(self):
        with self.assertRaisesRegex(ValueError,'truth missing'):
            evaluate_predictions({},[case('a')],{})

    def test_extraneous_or_duplicate_predictions_rejected(self):
        with self.assertRaisesRegex(ValueError,'outside'):
            evaluate_predictions({'x':probability([[0]])},[case('a')],{'a':np.zeros((1,1))})
        with self.assertRaisesRegex(ValueError,'Duplicate prediction'):
            evaluate_predictions([{'image_id':'a','probabilities':probability([[0]])}]*2,[case('a')],{'a':np.zeros((1,1))})

    def test_mapping_cases_and_detector_interface(self):
        class Model:
            def predict(self,c): return probability([[0]])
        result=evaluate_detector(Model(),{'a':case('a')},{'a':np.zeros((1,1),int)})
        self.assertEqual(result['risk'],0)

    def test_probability_validation_and_fixed_threshold(self):
        with self.assertRaises(ValueError): probability_grid_to_labels(np.array([[[.5,.5,.5]]]),2,2)
        p=np.array([[[.5,.1,.4]]])
        self.assertTrue((probability_grid_to_labels(p,2,2,.35)==2).all())
        self.assertTrue((probability_grid_to_labels(p,2,2,.45)==0).all())
        with self.assertRaises(ValueError):
            evaluate_predictions({},[case('a')],{'a':np.zeros((1,1))},{'foreground_threshold':float('nan')})

    def test_component_recall_counts_instances_not_presence(self):
        gt=np.zeros((6,6),bool); gt[0:2,0:2]=True;gt[4:6,4:6]=True
        pred=gt.copy();pred[4:6,4:6]=False
        result=component_matches(gt,pred)
        self.assertEqual(result['true_components'],2)
        self.assertEqual(result['matched_components'],1)

    def test_mixed_splits_rejected(self):
        rows=[case('a'),{**case('b'),'split':'reward'}]
        with self.assertRaisesRegex(ValueError,'one role'):
            evaluate_predictions({},rows,{'a':np.zeros((1,1)),'b':np.zeros((1,1))})

if __name__=='__main__': unittest.main()
