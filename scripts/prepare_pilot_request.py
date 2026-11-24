#!/usr/bin/env python3
"""Randomize an action for one predeclared case and save its known propensity."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from annotation_policy.adapters import make_randomized_pilot_request

def main():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument("--context",required=True); p.add_argument("--output",required=True)
 p.add_argument("--protocol-id",required=True); p.add_argument("--state-version",required=True); p.add_argument("--seed",required=True,type=int)
 a=p.parse_args(); c=json.loads(Path(a.context).read_text(encoding="utf-8"))
 out=make_randomized_pilot_request(c,a.output,protocol_id=a.protocol_id,state_version=a.state_version,seed=a.seed)
 print(json.dumps({"request":str(out),"human_action_executed":False},ensure_ascii=False))
if __name__=="__main__": main()
