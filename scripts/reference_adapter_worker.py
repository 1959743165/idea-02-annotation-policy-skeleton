#!/usr/bin/env python3
"""Executable adapter-contract example using the CPU reference model only.

Replace its two operations with a locally installed real detector trainer. This
worker does not claim to implement YOLO, Weld-LLaVA, RT-DETR, or RapidX.
"""
import argparse,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from annotation_policy.dataset import load_public_cases
from annotation_policy.detector import train_detector,Detector
from annotation_policy.adapters import sha256_file
from annotation_policy.experiment import write_json
from annotation_policy.policy import write_jsonl

def main():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument("--request",required=True); p.add_argument("--response",required=True)
 a=p.parse_args(); r=json.loads(Path(a.request).read_text(encoding="utf-8")); out=Path(r["output_dir"])
 response={"request_id":r["request_id"],"status":"completed","backend":"CPU_NUMPY_REFERENCE_NOT_REAL_MODEL"}
 if r["operation"]=="fit":
  cases=load_public_cases(r["training_manifest"])
  for line in Path(r["training_manifest"]).read_text(encoding="utf-8").splitlines():
   row=json.loads(line); case=cases[row["image_id"]]
   case.active=bool(row.get("active",row["initial_labeled"]))
   if row.get("current_labelmap_path"):
    target=(Path(r["training_manifest"]).parent/row["current_labelmap_path"]).resolve()
    if not target.is_relative_to(Path(r["training_manifest"]).parent.resolve()) or "oracle" in target.parts:
     raise ValueError("Current supervision must be a visible, locally exported training target")
    case.effective_override=np.asarray(Image.open(target),dtype=np.uint8)
  model=train_detector(list(cases.values()),r["config"],seed=r["seed"]); artifact=model.save(out/"model.npz")
  response.update(model_path=str(artifact),model_sha256=sha256_file(artifact),diagnostics=model.diagnostics,
                  training_specimen_ids=sorted({c.specimen_id for c in cases.values()}))
 elif r["operation"]=="predict":
  model=Detector.load(r["model_path"]); cases=load_public_cases(r["inference_manifest"])
  rows=[]
  for iid,case in cases.items():
   probability=model.predict(case)
   rows.append({"image_id":iid,"format":"grid_class_probabilities","grid_height":int(probability.shape[0]),
                "grid_width":int(probability.shape[1]),"class_order":[0,1,2],"probabilities":probability.tolist()})
  artifact=write_jsonl(out/"predictions.jsonl",rows)
  response.update(predictions_path=str(artifact),predictions_sha256=sha256_file(artifact))
 else: raise ValueError("Unsupported adapter operation")
 write_json(a.response,response); print(response["status"])
if __name__=="__main__": main()
