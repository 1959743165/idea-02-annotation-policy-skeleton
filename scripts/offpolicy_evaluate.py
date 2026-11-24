#!/usr/bin/env python3
"""Within-case action OPE. Does not estimate adaptive budget-policy value."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from annotation_policy.offpolicy import evaluate_ope
from annotation_policy.schema import load_jsonl,write_json

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--logs',required=True); p.add_argument('--train-logs')
    p.add_argument('--target-probabilities'); p.add_argument('--target-provenance')
    p.add_argument('--q-predictions'); p.add_argument('--q-provenance')
    p.add_argument('--ridge',type=float,default=1.); p.add_argument('--temperature',type=float,default=.1)
    p.add_argument('--max-weight',type=float); p.add_argument('--n-bootstrap',type=int,default=1000)
    p.add_argument('--seed',type=int,default=17); p.add_argument('--output',default='runs/ope.json')
    a=p.parse_args(); rows=load_jsonl(a.logs)
    read=lambda path:json.loads(Path(path).read_text(encoding='utf-8')) if path else None
    if a.train_logs:
        if any((a.target_probabilities,a.target_provenance,a.q_predictions,a.q_provenance)):
            p.error('--train-logs cannot be combined with externally supplied predictions/provenance')
        if any(r.get('split')!='policy_val' for r in rows): p.error('--train-logs mode evaluates policy_val only')
        from annotation_policy.policy import GainCostPolicy
        train=load_jsonl(a.train_logs)
        # Crucially do NOT call fit_policy: it tunes on policy_val rewards.
        model=GainCostPolicy.fit(train,config={'feature_mode':'full'},ridge=a.ridge)
        target={}; q={}
        for row in rows:
            quotes=row.get('action_quotes_minutes')
            target[row['event_id']]=model.action_probabilities(row['features'],row['available_actions'],quotes,a.temperature)
            q[row['event_id']]={action:pred['predicted_gain'] for action,pred in model.predict(row['features'],row['available_actions'],quotes).items()}
        provenance={'mode':'held_out','training_specimen_ids':sorted({r['specimen_id'] for r in train}),
                    'tuning_specimen_ids':[],'evaluation_rewards_used_for_selection':False,
                    'configuration':f'fixed ridge={a.ridge}, temperature={a.temperature}; no policy_val tuning',
                    'model_distinct_from_main_policy_tuned_on_policy_val':True}
        target_provenance=q_provenance=provenance
    else:
        target,q=read(a.target_probabilities),read(a.q_predictions)
        target_provenance,q_provenance=read(a.target_provenance),read(a.q_provenance)
        if target_provenance is None: p.error('--target-provenance is required for external target probabilities')
    result=evaluate_ope(rows,target,target_policy_provenance=target_provenance,q_predictions=q,q_provenance=q_provenance,
                        max_weight=a.max_weight,n_bootstrap=a.n_bootstrap,seed=a.seed)
    write_json(a.output,result)
    print(json.dumps({'estimand':result['estimand'],'ips':result['ips'],'snips':result['snips'],'dr':result['dr'],
                      'effective_sample_size':result['effective_sample_size'],'output':a.output},ensure_ascii=False))
if __name__=='__main__': main()
