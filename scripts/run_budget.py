#!/usr/bin/env python3
"""Execute and lock a SYNTHETIC paid-action trajectory without test feedback."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from annotation_policy.policy import GainCostPolicy
from annotation_policy.experiment import read_config,run_budget

def main():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument("--config",default="configs/smoke.yaml")
 p.add_argument("--manifest"); p.add_argument("--policy",required=True)
 p.add_argument("--strategy",default="learned_gain_per_minute"); p.add_argument("--output-dir",required=True)
 p.add_argument("--budget-minutes",type=float); p.add_argument("--seed",type=int)
 a=p.parse_args(); cfg=read_config(a.config)
 if a.budget_minutes is not None: cfg["budget_minutes"]=a.budget_minutes
 result=run_budget(a.manifest or cfg["dataset"]["manifest"],GainCostPolicy.load(a.policy),a.output_dir,cfg,a.strategy,a.seed)
 print(json.dumps({k:v for k,v in result.items() if k!="events"},ensure_ascii=False,indent=2))
if __name__=="__main__": main()
