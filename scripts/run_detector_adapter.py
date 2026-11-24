#!/usr/bin/env python3
"""Run the local command protocol using an explicitly supplied detector wrapper."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from annotation_policy.adapters import DetectorCommandAdapter

def main():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument("--operation",choices=["fit","predict"],required=True)
 p.add_argument("--command-json",required=True,help='JSON list, e.g. ["python","scripts/reference_adapter_worker.py"]')
 p.add_argument("--manifest",required=True); p.add_argument("--output-dir",required=True); p.add_argument("--model")
 p.add_argument("--seed",default=17,type=int); p.add_argument("--config-json",default="{}"); a=p.parse_args()
 adapter=DetectorCommandAdapter(json.loads(a.command_json))
 if a.operation=="fit": result=adapter.fit(a.manifest,a.output_dir,seed=a.seed,config=json.loads(a.config_json))
 else:
  if not a.model: p.error("predict requires --model")
  result=adapter.predict(a.manifest,a.model,a.output_dir)
 print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
