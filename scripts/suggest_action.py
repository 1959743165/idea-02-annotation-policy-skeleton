#!/usr/bin/env python3
"""Suggest one feasible action from public contexts; save a local pending request."""
import argparse,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from annotation_policy.policy import GainCostPolicy,choose_case_action,read_jsonl
from annotation_policy.adapters import make_action_request

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument("--contexts",required=True); p.add_argument("--policy",required=True)
 p.add_argument("--remaining-minutes",required=True,type=float); p.add_argument("--output",required=True)
 p.add_argument("--strategy",default="learned_gain_per_minute"); p.add_argument("--seed",default=17,type=int)
 p.add_argument("--protocol-id",required=True); p.add_argument("--state-version",required=True)
 a=p.parse_args(); contexts=read_jsonl(a.contexts); policy=GainCostPolicy.load(a.policy)
 decision=choose_case_action(contexts,policy,a.strategy,a.remaining_minutes,np.random.default_rng(a.seed))
 context=next((r for r in contexts if decision and r["image_id"]==decision["image_id"]),{})
 make_action_request(decision,context,a.output,protocol_id=a.protocol_id,state_version=a.state_version)
 print(json.dumps({"request":a.output,"decision":decision,"human_action_executed":False},ensure_ascii=False))
if __name__=="__main__": main()
