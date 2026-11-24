#!/usr/bin/env python3
"""Attach measured paired development risk to one already paid pilot event."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from annotation_policy.adapters import attach_measured_reward
from annotation_policy.policy import read_jsonl,write_jsonl

def main():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument("--ledger",required=True); p.add_argument("--event-id",required=True)
 p.add_argument("--before-metrics",required=True); p.add_argument("--after-metrics",required=True)
 p.add_argument("--reward-panel",choices=["reward_fit","reward_tune"],required=True); p.add_argument("--output",required=True)
 a=p.parse_args(); rows=[r for r in read_jsonl(a.ledger) if r["event_id"]==a.event_id]
 if len(rows)!=1: raise ValueError("Exactly one matching paid event is required")
 load=lambda p:json.loads(Path(p).read_text(encoding="utf-8"))
 row=attach_measured_reward(rows[0],load(a.before_metrics),load(a.after_metrics),reward_panel=a.reward_panel)
 if Path(a.output).exists(): raise ValueError("Use a new output file; combine validated events explicitly later")
 write_jsonl(a.output,[row]); print(json.dumps({"output":a.output,"reward":row["reward"]},ensure_ascii=False))
if __name__=="__main__": main()
