#!/usr/bin/env python3
"""Validate public expert reads, identities, provenance groups and device holdout."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from annotation_policy.schema import validate_dataset,write_json

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True); p.add_argument('--output',default='runs/validation.json')
    p.add_argument('--protocol',choices=['within_domain','device_holdout','site_holdout'],default='within_domain')
    p.add_argument('--check-hashes',action='store_true'); p.add_argument('--no-files',action='store_true')
    a=p.parse_args(); result=validate_dataset(a.manifest,not a.no_files,a.protocol,a.check_hashes); write_json(a.output,result)
    print(json.dumps({'ok':result['ok'],'errors':result['errors'],'output':a.output},ensure_ascii=False))
    return 0 if result['ok'] else 2
if __name__=='__main__': raise SystemExit(main())
