import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from annotation_policy.splits import check_group_leakage
from annotation_policy.schema import validate_dataset,write_jsonl


def row(i,split='policy_train',device='D1'):
    return {'image_id':i,'specimen_id':i,'weld_id':'w_'+i,'source_image_id':'src_'+i,
            'image_path':i+'.png','device_id':device,'site_id':'S1','split':split,'width':4,'height':4,
            'is_synthetic':True,'initial_labeled':True,
            'annotations':[{'annotator_id':'E'+str(k),'read_id':'read0','labelmap_path':i+f'_{k}.png','is_repeat':False} for k in range(3)],
            'available_actions':['boundary'],'action_quotes_minutes':{'boundary':2}}


class SplitTests(unittest.TestCase):
    def validate(self,rows,check_files=False):
        with tempfile.TemporaryDirectory() as d:
            path=write_jsonl(Path(d)/'manifest.jsonl',rows)
            return validate_dataset(path,check_files=check_files)

    def test_unique_expert_and_read_pair_not_read_id_alone(self):
        self.assertTrue(self.validate([row('a')])['ok'])
        a=row('a'); a['annotations'].append(copy.deepcopy(a['annotations'][0]))
        result=self.validate([a])
        self.assertFalse(result['ok'])
        self.assertTrue(any('duplicate annotator_id/read_id' in e for e in result['errors']))

    def test_repeated_same_expert_does_not_count_as_three(self):
        a=row('a')
        for k,read in enumerate(a['annotations']):read['annotator_id']='E0';read['read_id']=f'read{k}';read['is_repeat']=k>0
        result=self.validate([a]);self.assertFalse(result['ok'])
        self.assertTrue(any('3 distinct' in e for e in result['errors']))

    def test_explicit_historical_single_reader_is_seed_only(self):
        a=row('historical','seed_train'); a['is_synthetic']=False; a['annotations']=a['annotations'][:1]
        a['annotation_provenance']='authorized_historical_single_reader_training_only'
        a['available_actions']=[]; a['action_quotes_minutes']={}
        result=self.validate([a]); self.assertTrue(result['ok'])
        self.assertTrue(any('historical single-reader' in w for w in result['warnings']))
        a['split']='policy_train'
        self.assertFalse(self.validate([a])['ok'])

    def test_specimen_cross_split_detected(self):
        a,b=row('a'),row('b','test');b['specimen_id']=a['specimen_id']
        self.assertFalse(check_group_leakage([a,b])['ok'])

    def test_transitive_source_and_weld_group_detected(self):
        a,b,c=row('a'),row('b'),row('c','test')
        b['source_image_id']=a['source_image_id'];c['weld_id']=b['weld_id']
        result=check_group_leakage([a,b,c]);self.assertFalse(result['ok'])
        self.assertEqual(len(result['leaked_groups'][0]['image_ids']),3)

    def test_parent_view_link_isolated(self):
        a,b=row('a'),row('b','test');b['parent_image_id']='a'
        self.assertFalse(check_group_leakage([a,b])['ok'])

    def test_device_test_never_shares_device(self):
        self.assertFalse(check_group_leakage([row('a'),row('b','device_test')])['ok'])
        self.assertTrue(check_group_leakage([row('a'),row('b','device_test','D2')])['ok'])

    def test_main_device_holdout_enforced_only_when_declared(self):
        rows=[row('a'),row('b','test')]
        self.assertTrue(check_group_leakage(rows)['ok'])
        self.assertFalse(check_group_leakage(rows,'device_holdout')['ok'])

    def test_hidden_oracle_metadata_rejected_recursively(self):
        a=row('a');a['extra']={'nested':{'true_class':2}}
        self.assertFalse(self.validate([a])['ok'])

    def test_unlabelled_cases_cannot_expose_hidden_labels(self):
        a=row('a');a['initial_labeled']=False;a['available_actions']=['new_image'];a['action_quotes_minutes']={'new_image':3}
        self.assertFalse(self.validate([a])['ok'])
        a['annotations']=[];self.assertTrue(self.validate([a])['ok'])

    def test_hash_tampering_caught(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);a=row('a');a['image_sha256']='0'*64
            for name in ['a.png']+[r['labelmap_path'] for r in a['annotations']]:
                Image.fromarray(np.zeros((4,4),dtype=np.uint8)).save(root/name)
            path=write_jsonl(root/'manifest.jsonl',[a])
            result=validate_dataset(path,check_hashes=True)
            self.assertFalse(result['ok']);self.assertTrue(any('SHA256' in e for e in result['errors']))

if __name__=='__main__': unittest.main()
