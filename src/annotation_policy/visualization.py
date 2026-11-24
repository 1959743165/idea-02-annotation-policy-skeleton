"""Export honest, fixed-horizon risk/time figures from already locked runs."""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .statistics import trajectory_summary
from .schema import write_json


def load_trajectories(paths):
    result = {}
    for path in paths:
        row = json.loads(Path(path).read_text(encoding='utf-8'))
        if not isinstance(row.get('events'), list): raise ValueError(f'{path}: evaluated trajectory events required')
        label = f"{row.get('strategy', Path(path).stem)} [seed {row.get('seed', '?')}]"
        if label in result: raise ValueError('Duplicate trajectory label: ' + label)
        result[label] = row
    return result


def plot_trajectories(trajectories, output_dir, horizon_minutes, n_bootstrap=1000, seed=17):
    """Input maps legend labels to evaluated_timeline dicts (or event lists).

    Every comparison uses the same fixed H and exact specimen/image cohort.
    Confidence intervals resample test specimens only; no policy-seed certainty
    is implied. The files never invent real annotation costs or real outcomes.
    """
    if not trajectories: raise ValueError('At least one evaluated trajectory is required')
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    prepared = {}; fingerprints = set(); splits = set(); synthetic = []
    for label, data in trajectories.items():
        events = data if isinstance(data, list) else data['events']
        summary = trajectory_summary(events, horizon_minutes, n_bootstrap, seed)
        if summary['normalized_auc'] is None:
            raise ValueError('Undefined risk at a required budget state; cannot plot it as zero or interpolate through it')
        fingerprints.add(summary['cohort_sha256'])
        if isinstance(data, dict):
            splits.add(data.get('split', 'unspecified'))
            synthetic.append(str(data.get('status', '')).startswith('SYNTHETIC'))
        else: synthetic.append(True)
        prepared[label] = {'events': events, 'summary': summary}
    if len(fingerprints) != 1 or len(splits) > 1:
        raise ValueError('Plots require one identical held-out specimen/image cohort and split')
    note = ('SYNTHETIC SOFTWARE TEST - NOT REAL DATA OR EXPERT RESULTS' if all(synthetic)
            else 'Held-out evaluation; intervals condition on frozen acquisition trajectories')
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    colors = plt.get_cmap('tab10')(np.linspace(0,.9,max(1,len(prepared))))
    files = []
    fig, ax = plt.subplots(figsize=(9.4,5.8))
    for color, (label, data) in zip(colors, prepared.items()):
        summary = data['summary']
        ax.step(summary['step_times'], summary['step_risks'], where='post', label=label, color=color, linewidth=2)
        ax.scatter(summary['step_times'][:-1], summary['step_risks'][:-1], color=color, s=16, alpha=.7)
    ax.set(xlabel='Cumulative completed-action expert person-minutes', ylabel='Specimen-macro risk (lower is better)',
           xlim=(0,horizon_minutes), ylim=(0,1), title='Risk under a common expert-time budget')
    ax.grid(alpha=.18); ax.legend(loc='best', fontsize=8)
    fig.text(.5,.02,note,ha='center',fontsize=8,color='#8b1e24'); fig.tight_layout(rect=(0,.055,1,1))
    target=output_dir/'risk_vs_expert_minutes.png'; fig.savefig(target,dpi=180); plt.close(fig); files.append(str(target))
    fig, ax = plt.subplots(figsize=(9.4,max(4.0,.7*len(prepared)+2)))
    for y, (color, (label, data)) in enumerate(zip(colors,prepared.items())):
        interval=data['summary']['nauc_confidence_interval']; estimate=interval['estimate']; ci=interval['ci']
        if estimate is None: continue
        ax.plot(estimate,y,'o',color=color)
        if ci is not None: ax.hlines(y,ci[0],ci[1],color=color,linewidth=2)
    ax.set_yticks(range(len(prepared)),list(prepared)); ax.invert_yaxis()
    ax.set(xlabel=f'Normalized risk AUC at H={horizon_minutes:g} person-minutes (lower is better)',
           xlim=(0,1),title='Conditional 95% test-specimen bootstrap intervals')
    ax.grid(axis='x',alpha=.18)
    fig.text(.5,.035,note+'\nPolicy-training, expert-panel and acquisition-run variability are not included.',ha='center',fontsize=8,color='#8b1e24')
    fig.tight_layout(rect=(0,.095,1,1)); target=output_dir/'nauc_specimen_ci.png'; fig.savefig(target,dpi=180); plt.close(fig); files.append(str(target))
    actions=('new_image','boundary','missing','class','review','reimage'); labels=list(prepared)
    fig, ax = plt.subplots(figsize=(9.4,max(4.,.65*len(labels)+2)))
    totals=defaultdict(lambda:np.zeros(len(labels)))
    for i, data in enumerate(prepared.values()):
        for event in data['events']:
            if event.get('action') in actions and event['minutes'] <= horizon_minutes:
                value=event.get('cost_minutes')
                if value is None or not np.isfinite(value) or value <= 0:
                    raise ValueError('Action allocation requires actual positive event costs, including failures')
                totals[event['action']][i] += value
    left=np.zeros(len(labels))
    for action in actions:
        values=totals[action]; ax.barh(labels,values,left=left,label=action); left+=values
    ax.invert_yaxis(); ax.set(xlabel='Actual expert person-minutes charged by action',xlim=(0,horizon_minutes),title='Purchased annotation actions (failed actions retain their cost)')
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.16),ncol=3,fontsize=8); ax.grid(axis='x',alpha=.16)
    fig.text(.5,.02,note,ha='center',fontsize=8,color='#8b1e24'); fig.tight_layout(rect=(0,.12,1,1))
    target=output_dir/'action_time_allocation.png'; fig.savefig(target,dpi=180); plt.close(fig); files.append(str(target))
    path=write_json(output_dir/'trajectory_statistics.json',{'status':'SYNTHETIC_DESCRIPTIVE_CHECK' if all(synthetic) else 'LOCKED_HELD_OUT_EVALUATION',
                       'horizon_minutes':horizon_minutes,'trajectories':{k:v['summary'] for k,v in prepared.items()},'figure_note':note})
    files.append(str(path)); return files
