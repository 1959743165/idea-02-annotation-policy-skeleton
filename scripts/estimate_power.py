#!/usr/bin/env python3
"""Planning-only paired specimen power simulation; never empirical results."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from annotation_policy.statistics import estimate_paired_power
from annotation_policy.schema import write_json

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--n-specimens',type=int,nargs='+',default=[100,150,240,300])
    p.add_argument('--effect',type=float,default=.02); p.add_argument('--paired-sd',type=float,default=.08)
    p.add_argument('--attrition',type=float,default=.15); p.add_argument('--alpha',type=float,default=.05)
    p.add_argument('--target-power',type=float,default=.8); p.add_argument('--repetitions',type=int,default=2000)
    p.add_argument('--seed',type=int,default=17); p.add_argument('--output',default='runs/power_planning.json')
    a=p.parse_args(); result=estimate_paired_power(a.n_specimens,a.effect,a.paired_sd,a.attrition,a.alpha,a.target_power,a.repetitions,a.seed)
    write_json(a.output,result); print(json.dumps({'status':result['status'],'analytic_approximation':result['analytic_approximation'],'output':a.output}))
if __name__=='__main__': main()
