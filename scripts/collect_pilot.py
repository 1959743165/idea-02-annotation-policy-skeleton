#!/usr/bin/env python3
"""Collect randomized SYNTHETIC action logs with measured retraining rewards."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from annotation_policy.experiment import read_config,collect_development_logs

def main():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument("--config",default="configs/smoke.yaml")
 p.add_argument("--manifest"); p.add_argument("--output-dir",default="runs/pilot")
 a=p.parse_args(); cfg=read_config(a.config)
 out=collect_development_logs(a.manifest or cfg["dataset"]["manifest"],a.output_dir,cfg)
 print(json.dumps({k:{"path":v["path"],"events":len(v["rows"])} for k,v in out.items()},ensure_ascii=False,indent=2))
if __name__=="__main__": main()
