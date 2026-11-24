#!/usr/bin/env python3
"""Run a reproducible, honest CPU pipeline; no real model/data are downloaded."""
from __future__ import annotations
import argparse,hashlib,importlib.metadata,json,os,platform,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from annotation_policy.dataset import generate_synthetic,load_public_cases
from annotation_policy.detector import Detector
from annotation_policy.policy import GainCostPolicy,read_jsonl
from annotation_policy.schema import validate_dataset
from annotation_policy.experiment import read_config,write_json
from annotation_policy.statistics import compare_trajectories

ROOT=Path(__file__).resolve().parents[1]

def main():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument("--config",default=str(ROOT/"configs/smoke.yaml"))
 p.add_argument("--output-dir",default=str(ROOT/"runs/smoke")); p.add_argument("--skip-unit-tests",action="store_true")
 a=p.parse_args(); cfg=read_config(a.config); out=Path(a.output_dir).resolve(); out.mkdir(parents=True,exist_ok=True)
 checks=[]; commands=[]; start=time.perf_counter()
 def check(name,condition,detail):
  checks.append({"name":name,"passed":bool(condition),"detail":detail})
  if not condition: raise AssertionError(name+": "+str(detail))
 def run(name,args):
  command=[sys.executable,*map(str,args)]; env={**os.environ,"PYTHONPATH":str(ROOT/"src"),"PYTHONDONTWRITEBYTECODE":"1"}
  t=time.perf_counter(); process=subprocess.run(command,cwd=ROOT,env=env,capture_output=True,text=True)
  log=out/"logs"/(name+".log"); log.parent.mkdir(parents=True,exist_ok=True)
  log.write_text("COMMAND "+json.dumps(command)+"\nEXIT "+str(process.returncode)+"\n"+process.stdout+process.stderr,encoding="utf-8")
  commands.append({"name":name,"command":command,"exit_code":process.returncode,"wall_seconds":time.perf_counter()-t,"log":str(log)})
  check(name+"_exit",process.returncode==0,{"log":str(log)})
  print(name+": passed",flush=True)
 try:
  manifest=generate_synthetic(out/"data",seed=cfg["seed"],n_specimens=cfg["dataset"]["n_specimens"],image_size=cfg["dataset"]["image_size"])
  validation=validate_dataset(manifest,check_files=True,check_hashes=True); write_json(out/"dataset_validation.json",validation)
  check("public_manifest_hashes_groups",validation["ok"],{"images":validation["images"],"errors":validation["errors"]})
  run("experiment",[ROOT/"scripts/run_experiment.py","--config",Path(a.config).resolve(),"--manifest",manifest,"--output-dir",out/"experiment"])
  summary=json.loads((out/"experiment/experiment_summary.json").read_text())
  check("all_methods_locked_before_test",summary["all_methods_locked_before_test_evaluation"],summary["status"])
  timelines=[]
  for strategy in cfg["strategies"]:
   timeline=json.loads((out/"experiment"/strategy/"locked_timeline.json").read_text())
   events=timeline["events"]; costs=sum(e.get("cost_minutes",0) for e in events)
   check("budget_"+strategy,abs(costs-timeline["spent_minutes"])<1e-8 and costs<=cfg["budget_minutes"]+1e-8 and
         timeline["paid_actions"]==timeline["completed_actions"]+timeline["failed_actions"]==len(events)-1,
         {"spent":costs,"paid":timeline["paid_actions"],"failed":timeline["failed_actions"]})
   timelines.append(out/"experiment"/strategy/"test/evaluated_timeline.json")
  initial=Detector.load(out/"experiment"/cfg["strategies"][0]/"checkpoints/model_000.npz")
  diag=initial.diagnostics
  check("detector_is_actually_optimized",diag["finite_training"] and diag["parameter_l2_change"]>1e-6 and diag["final_loss"]<diag["initial_loss"],diag)
  train=read_jsonl(out/"experiment/pilot/policy_train_logs.jsonl"); val=read_jsonl(out/"experiment/pilot/policy_val_logs.jsonl")
  reward_fit={x for r in train for x in r["reward_specimen_ids"]}; reward_tune={x for r in val for x in r["reward_specimen_ids"]}
  check("measured_rewards_and_disjoint_panels",not bool(reward_fit & reward_tune) and all(
   abs(r["reward"]-(r["before_risk"]-r["after_risk"]))<1e-12 and r["reward_is_measured_by_detector_retraining"] for r in train+val),
   {"train_events":len(train),"val_events":len(val),"reward_fit_specimens":len(reward_fit),"reward_tune_specimens":len(reward_tune)})
  cases=load_public_cases(manifest,splits="test"); prediction={iid:initial.predict(case) for iid,case in cases.items()}
  oracle=out/"data/oracle"; unavailable=out/"data/oracle_not_available_during_prediction"
  oracle.rename(unavailable)
  try:
   public=load_public_cases(manifest,splits="test")
   invariant=all(np.array_equal(prediction[iid],initial.predict(case)) for iid,case in public.items())
  finally: unavailable.rename(oracle)
  check("inference_without_any_private_truth_files",invariant,{"test_images":len(cases)})
  baseline=json.loads((out/"experiment/uncertainty_per_minute/test/evaluated_timeline.json").read_text())
  learned=json.loads((out/"experiment/learned_gain_per_minute/test/evaluated_timeline.json").read_text())
  comparison=compare_trajectories(baseline["events"],learned["events"],cfg["budget_minutes"],cfg["bootstrap_repetitions"],cfg["bootstrap_repetitions"],cfg["seed"])
  write_json(out/"paired_comparison_SYNTHETIC.json",comparison)
  check("paired_specimen_statistics",comparison["difference_direction"]=="B_minus_A",comparison["primary_normalized_auc"])
  run("device_holdout_evaluation",[ROOT/"scripts/evaluate.py","--config",Path(a.config).resolve(),"--manifest",manifest,
      "--timeline",out/"experiment/learned_gain_per_minute/locked_timeline.json","--split","device_test","--output-dir",out/"device_test"])
  run("offpolicy_single_step",[ROOT/"scripts/offpolicy_evaluate.py","--train-logs",out/"experiment/pilot/policy_train_logs.jsonl",
      "--logs",out/"experiment/pilot/policy_val_logs.jsonl","--n-bootstrap",cfg["bootstrap_repetitions"],"--output",out/"ope_SYNTHETIC.json"])
  run("visualization",[ROOT/"scripts/visualize.py","--timelines",*timelines,"--horizon-minutes",cfg["budget_minutes"],
      "--n-bootstrap",cfg["bootstrap_repetitions"],"--output-dir",out/"figures"])
  run("power_planning",[ROOT/"scripts/estimate_power.py","--repetitions",300,"--output",out/"power_PLANNING_NOT_OBSERVED.json"])
  if not a.skip_unit_tests: run("unit_tests",["-m","unittest","discover","-s",ROOT/"tests","-v"])
  passed=True
 except Exception as exc:
  passed=False; checks.append({"name":"uncaught_failure","passed":False,"detail":repr(exc)})
 finally:
  result={"status":"PASSED_SYNTHETIC_SOFTWARE_SMOKE" if passed else "FAILED_SMOKE","passed":passed,
          "created_utc":datetime.now(timezone.utc).isoformat(),"python":sys.version,"platform":platform.platform(),
          "dependencies":{name:importlib.metadata.version(name) for name in ("numpy","scipy","Pillow","matplotlib","PyYAML")},
          "wall_seconds":time.perf_counter()-start,"checks":checks,"commands":commands,
          "real_training_or_real_expert_work_performed":False,"scientific_superiority_is_not_asserted":True,
          "meaning":"Checks software integrity and evaluation separation; results do not establish natural disagreement or policy effectiveness."}
  write_json(out/"smoke_results.json",result)
  print(json.dumps({"status":result["status"],"checks":len(checks),"wall_seconds":result["wall_seconds"],"report":str(out/"smoke_results.json")},ensure_ascii=False),flush=True)
 if not passed: sys.exit(1)
if __name__=="__main__": main()
