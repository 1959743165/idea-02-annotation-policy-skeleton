#!/usr/bin/env python3
"""Complete SYNTHETIC measured pilot -> policy -> budget -> independent evaluation."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from annotation_policy.experiment import read_config,run_experiment

def main():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument("--config",default="configs/smoke.yaml")
 p.add_argument("--manifest"); p.add_argument("--output-dir"); a=p.parse_args(); cfg=read_config(a.config)
 out=run_experiment(a.manifest or cfg["dataset"]["manifest"],a.output_dir or cfg["output_dir"],cfg)
 print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
