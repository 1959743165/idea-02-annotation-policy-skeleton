#!/usr/bin/env python3
"""Validate an already completed human action and append actual paid time."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from annotation_policy.adapters import import_human_completion

def main():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument("--request",required=True)
 p.add_argument("--completion",required=True); p.add_argument("--ledger",required=True); a=p.parse_args()
 row=import_human_completion(a.request,a.completion,a.ledger)
 print(json.dumps({"event_id":row["event_id"],"status":row["status"],"cost_minutes":row["cost_minutes"],
                   "quote_overrun_minutes":row["quote_overrun_minutes"],"reward_status":row["reward_status"]},ensure_ascii=False))
if __name__=="__main__": main()
