#!/usr/bin/env python3
"""Evaluate a fully locked trajectory on independent test specimens."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from annotation_policy.experiment import read_config,evaluate_locked_timeline

def main():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument("--config",default="configs/smoke.yaml")
 p.add_argument("--manifest",required=True); p.add_argument("--timeline",required=True)
 p.add_argument("--output-dir",required=True); p.add_argument("--split",choices=["test","device_test"],default="test")
 a=p.parse_args(); out=evaluate_locked_timeline(a.manifest,a.timeline,a.output_dir,read_config(a.config).get("evaluation",{}),a.split)
 print(json.dumps({"status":out["status"],"output":str(Path(a.output_dir)/"evaluated_timeline.json")},ensure_ascii=False))
if __name__=="__main__": main()
