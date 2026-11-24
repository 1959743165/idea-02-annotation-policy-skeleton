#!/usr/bin/env python3
"""Plot independently evaluated, already locked annotation-action trajectories."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from annotation_policy.visualization import load_trajectories,plot_trajectories

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--timelines',nargs='+',required=True); p.add_argument('--output-dir',required=True)
    p.add_argument('--horizon-minutes',type=float,required=True); p.add_argument('--n-bootstrap',type=int,default=1000)
    p.add_argument('--seed',type=int,default=17)
    a=p.parse_args(); files=plot_trajectories(load_trajectories(a.timelines),a.output_dir,a.horizon_minutes,a.n_bootstrap,a.seed)
    print(json.dumps({'files':files},ensure_ascii=False))
if __name__=='__main__': main()
