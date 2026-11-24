#!/usr/bin/env python3
"""Execute the declared synthetic baseline and ablation matrix, with shared pilot."""
import argparse,copy,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from annotation_policy.experiment import (read_config,write_json,collect_development_logs,train_policy_from_logs,
                                         run_budget,evaluate_locked_timeline)
from annotation_policy.statistics import trajectory_summary,compare_trajectories

ROOT=Path(__file__).resolve().parents[1]

def main():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument("--matrix",default=str(ROOT/"configs/matrix.yaml"))
 p.add_argument("--manifest",required=True); p.add_argument("--pilot-dir",help="Reuse already measured shared randomized logs")
 p.add_argument("--output-dir",required=True); p.add_argument("--budget-minutes",type=float)
 a=p.parse_args(); matrix=read_config(a.matrix); base=Path(matrix["base_config"])
 cfg=read_config(base if base.is_absolute() else ROOT/base)
 if a.budget_minutes is not None: cfg["budget_minutes"]=a.budget_minutes
 out=Path(a.output_dir).resolve(); out.mkdir(parents=True,exist_ok=True)
 pilot=Path(a.pilot_dir).resolve() if a.pilot_dir else out/"pilot"
 if not a.pilot_dir: collect_development_logs(a.manifest,pilot,cfg)
 train,val=pilot/"policy_train_logs.jsonl",pilot/"policy_val_logs.jsonl"
 locked={}; names=set()
 for variant in matrix["variants"]:
  name=variant["name"]
  if name in names or Path(name).name!=name: raise ValueError("Unique safe variant names required")
  names.add(name); policy_cfg={**cfg["policy"],**variant.get("policy_overrides",{})}
  policy=train_policy_from_logs(train,val,out/name/"policy.npz",policy_cfg)
  locked[name]=run_budget(a.manifest,policy,out/name,cfg,variant["strategy"])
  write_json(out/name/"variant_config.json",{"variant":variant,"policy":policy_cfg,"shared_pilot":str(pilot)})
 # Test truth is first accessed here, after the entire matrix is frozen.
 table=[]; trajectories={}
 for name,timeline in locked.items():
  for split in matrix.get("evaluate_splits",["test"]):
   result=evaluate_locked_timeline(a.manifest,out/name/"locked_timeline.json",out/name/split,cfg["evaluation"],split)
   stat=trajectory_summary(result["events"],cfg["budget_minutes"],cfg["bootstrap_repetitions"],cfg["seed"])
   write_json(out/name/split/"trajectory_statistics.json",stat); trajectories[(name,split)]=result
   table.append({"variant":name,"split":split,"paid_actions":timeline["paid_actions"],"failed_actions":timeline["failed_actions"],
                 "spent_minutes":timeline["spent_minutes"],"normalized_auc":stat["normalized_auc"],"final_risk":stat["final_risk"]})
 primary={}
 for split in matrix.get("evaluate_splits",["test"]):
  if ("learned_full",split) in trajectories and ("uncertainty_per_minute",split) in trajectories:
   primary[split]=compare_trajectories(trajectories[("uncertainty_per_minute",split)]["events"],
                                     trajectories[("learned_full",split)]["events"],cfg["budget_minutes"],
                                     cfg["bootstrap_repetitions"],cfg["bootstrap_repetitions"],cfg["seed"])
 result={"status":"SYNTHETIC_MATRIX_SOFTWARE_CHECK_NOT_PUBLICATION_RESULTS","variants":len(locked),"rows":table,
         "all_variants_locked_before_test":True,"primary_learned_minus_uncertainty":primary,
         "no_superiority_assertion":True,"real_detector_or_human_experiment_performed":False}
 write_json(out/"matrix_summary.json",result)
 print(json.dumps({"status":result["status"],"variants":len(locked),"summary":str(out/"matrix_summary.json")},ensure_ascii=False))
if __name__=="__main__": main()
