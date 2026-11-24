#!/usr/bin/env python3
"""Train contextual gain/time regressors from audited policy_train/policy_val logs."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from annotation_policy.experiment import read_config,train_policy_from_logs

def main():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument("--config",default="configs/smoke.yaml")
 p.add_argument("--train-logs",required=True); p.add_argument("--val-logs",required=True)
 p.add_argument("--output",default="runs/policy.npz"); a=p.parse_args()
 model=train_policy_from_logs(a.train_logs,a.val_logs,a.output,read_config(a.config).get("policy",{}))
 print(json.dumps({"checkpoint":a.output,"metadata":model.metadata},ensure_ascii=False,indent=2))
if __name__=="__main__": main()
