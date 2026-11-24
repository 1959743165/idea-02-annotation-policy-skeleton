"""Local-file contracts for real detector wrappers and paid expert actions.

No network, expert messaging, model download, scanner control, or hidden labels
are implemented here. External detector wrappers are explicitly supplied by the
researcher. A completed human action is imported only after its outcome exists.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import hashlib
import json
import math
import subprocess
import sys
import uuid
import shutil
import numpy as np
from PIL import Image
from .policy import ACTIONS, GainCostPolicy, choose_case_action, read_jsonl
from .annotation_state import DETECTOR_TRAINING_SPLITS, consensus_labelmap
from .experiment import write_json


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _iso_time(value):
    stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if stamp.tzinfo is None:
        raise ValueError("Use timezone-aware ISO8601 timestamps")
    return stamp


def person_minutes(intervals):
    """Sum active work over experts; parallel work by two experts counts twice.

    Pauses must be excluded by opening separate active intervals. Overlapping
    intervals for the same person are rejected instead of double charging them.
    """
    if not intervals:
        raise ValueError("Completed/failed actions need actual active-time records")
    by_person = {}
    total_seconds = 0.0
    for interval in intervals:
        person = interval.get("annotator_id")
        if not isinstance(person, str) or not person:
            raise ValueError("Every interval needs a pseudonymous annotator_id")
        start, end = _iso_time(interval["start_utc"]), _iso_time(interval["end_utc"])
        if end <= start:
            raise ValueError("Active intervals must have positive duration")
        by_person.setdefault(person, []).append((start, end))
        total_seconds += (end - start).total_seconds()
    for person, values in by_person.items():
        values.sort()
        if any(right[0] < left[1] for left, right in zip(values, values[1:])):
            raise ValueError(f"Overlapping work intervals for the same expert: {person}")
    return total_seconds / 60.0


def make_action_request(decision, context, output_path, *, protocol_id, state_version):
    """Write a reviewable local request; this does not contact an expert."""
    if decision is None:
        return write_json(output_path, {"status":"NO_FEASIBLE_ACTION", "protocol_id":protocol_id})
    if decision["image_id"] != context["image_id"] or decision["action"] not in context["available_actions"]:
        raise ValueError("Decision must match the visible eligible context")
    action = decision["action"]
    quote = float(context["action_quotes_minutes"][action])
    if not math.isfinite(quote) or quote <= 0:
        raise ValueError("Positive finite expert-person-minute quote required")
    record = {"schema_version":"1.0", "event_id":str(uuid.uuid4()),
              "status":"REQUESTED_NOT_EXECUTED", "protocol_id":protocol_id,
              "image_id":context["image_id"], "specimen_id":context["specimen_id"],
              "split":context["split"], "is_synthetic":bool(context.get("is_synthetic", False)),
              "action":action,"available_actions":list(context["available_actions"]),
              "features":dict(context["features"]), "selection":decision,
              "quoted_max_cost_minutes":quote,"before_state_version":state_version,
              "required_return_fields":["event_id","status","active_intervals","after_state_version",
                                         "outcome_artifact_path","outcome_artifact_sha256","reason"],
              "reimage_execution_note":"A local request is not authorization or control of acquisition equipment."}
    if record["split"] not in {"policy_train", "policy_val", "acquisition_pool"}:
        raise ValueError("Expert action requests cannot target test/reward specimens")
    return write_json(output_path, record)


def make_randomized_pilot_request(context, output_path, *, protocol_id, state_version, seed=17):
    """Uniform action conditional on one predeclared pilot case (not case selection)."""
    if context.get("split") not in {"policy_train", "policy_val"}:
        raise ValueError("Randomized learning pilots require policy_train/policy_val cases")
    available=list(context["available_actions"])
    if not available or len(set(available)) != len(available) or any(a not in ACTIONS for a in available):
        raise ValueError("Declare the complete nonempty feasible action set before randomization")
    action=available[int(np.random.default_rng(seed).integers(len(available)))]
    decision={"image_id":context["image_id"],"action":action,"strategy":"uniform_action_given_predeclared_case",
              "quoted_max_cost_minutes":float(context["action_quotes_minutes"][action])}
    path=make_action_request(decision,context,output_path,protocol_id=protocol_id,state_version=state_version)
    request=json.loads(path.read_text(encoding="utf-8"))
    request.update(propensity=1/len(available),logging_policy_probs={a:1/len(available) for a in available},
                   action_quotes_minutes=context["action_quotes_minutes"],randomization_seed=int(seed),
                   randomization_unit="action conditional on fixed pilot case; no joint case-policy propensity")
    return write_json(path,request)


def attach_measured_reward(paid_event, before_metrics, after_metrics, *, reward_panel):
    """Join independently produced, paired development metrics to a paid pilot.

    Metrics must carry `risk`, `split`, `specimen_ids`, `model_sha256`, `seed`,
    and `training_specimen_ids`. Actual detector outputs remain auditable files.
    A shared init seed alone does not remove stochastic training variability;
    repeated seeds are required by the real statistical plan.
    """
    if paid_event.get("split") not in {"policy_train","policy_val"}:
        raise ValueError("Only randomized development pilots supply policy-learning rewards")
    if not paid_event.get("logging_policy_probs") or not paid_event.get("propensity"):
        raise ValueError("Unknown retrospective propensities cannot be fabricated")
    expected_panel="reward_fit" if paid_event["split"]=="policy_train" else "reward_tune"
    if reward_panel!=expected_panel: raise ValueError("Use separate reward panels for fit and tuning")
    for metrics in (before_metrics,after_metrics):
        if metrics.get("split")!="reward" or not metrics.get("model_sha256"):
            raise ValueError("Reward metrics require a development-only model provenance")
        value=metrics.get("risk")
        if not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<=value<=1:
            raise ValueError("Undefined risk cannot be imputed as zero")
        if not metrics.get("specimen_ids") or set(metrics["specimen_ids"]) & set(metrics.get("training_specimen_ids",[])):
            raise ValueError("Risk specimens must be recorded and absent from detector training")
        if paid_event["specimen_id"] in metrics["specimen_ids"]:
            raise ValueError("Action candidate cannot also be a reward specimen")
    if sorted(before_metrics["specimen_ids"])!=sorted(after_metrics["specimen_ids"]):
        raise ValueError("Before/after risk must use exactly the same reward specimens")
    if before_metrics.get("seed") is None or before_metrics["seed"]!=after_metrics.get("seed"):
        raise ValueError("Record matched detector initialization seeds")
    row={**paid_event,"reward":float(before_metrics["risk"]-after_metrics["risk"]),
         "before_risk":before_metrics["risk"],"after_risk":after_metrics["risk"],
         "reward_source_split":"reward","reward_panel":reward_panel,
         "reward_specimen_ids":sorted(before_metrics["specimen_ids"]),"reward_status":"MEASURED_PAIRED_RETRAINING",
         "before_model_sha256":before_metrics["model_sha256"],"after_model_sha256":after_metrics["model_sha256"],
         "detector_seed_before_after":before_metrics["seed"],"success":paid_event["status"]=="completed"}
    return row


def export_current_training_manifest(cases, output_dir):
    """Export paid image/label revisions, while preserving raw rater provenance.

    `current_labelmap_path` is an effective training target, NOT another expert
    read and NOT independent ground truth. A real bbox backend exports its current
    detection target analogously instead of rasterizing boxes as exact boundaries.
    """
    output=Path(output_dir).resolve(); output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for case in cases:
        if not case.active: continue
        if case.split not in DETECTOR_TRAINING_SPLITS:
            raise ValueError("Cannot export held-out cases into training")
        iid=case.image_id
        if Path(iid).name!=iid or iid in {".",".."}: raise ValueError("Use filesystem-safe image IDs")
        directory=output/"visible"/iid; directory.mkdir(parents=True,exist_ok=True)
        image_path=directory/"image.png"
        Image.fromarray(np.rint(np.clip(case.image,0,1)*255).astype(np.uint8)).save(image_path)
        reads=[]
        for index,read in enumerate(case.annotations):
            label_path=directory/f"raw_read_{index:03d}.png"; Image.fromarray(read.labelmap).save(label_path)
            reads.append({"annotator_id":read.annotator_id,"read_id":read.read_id,"is_repeat":read.is_repeat,
                          "uncertainty":read.uncertainty,"provenance":read.provenance,
                          "labelmap_path":str(label_path.relative_to(output))})
        target=directory/"current_training_target.png"; Image.fromarray(consensus_labelmap(case)).save(target)
        row={"image_id":iid,"specimen_id":case.specimen_id,"split":case.split,"device_id":case.device_id,
             "width":case.width,"height":case.height,"is_synthetic":case.is_synthetic,
             "image_path":str(image_path.relative_to(output)),"image_sha256":sha256_file(image_path),
             "initial_labeled":bool(case.metadata.get("initial_labeled",case.active)),"active":True,
             "annotations":reads,"available_actions":[],"action_quotes_minutes":{},
             "state_version":f"visible_revision_count_{len(case.revision_history)}",
             "current_labelmap_path":str(target.relative_to(output)),
             "current_target_semantics":"effective supervision, not an independent expert or reference standard",
             "revision_history":case.revision_history}
        rows.append(row)
    if not rows: raise ValueError("No active training cases to export")
    from .policy import write_jsonl
    return write_jsonl(output/"current_training_manifest.jsonl",rows)


def import_human_completion(request_path, completion_path, ledger_path):
    """Append an immutable paid event. Retain failures and maximum-quote overruns.

    Future measured detector risk is deliberately not invented here. A separate
    paired-retraining job adds the before/after development-risk reward later.
    Single writer required: use a transactional database when running in parallel.
    """
    request = json.loads(Path(request_path).read_text(encoding="utf-8"))
    completion = json.loads(Path(completion_path).read_text(encoding="utf-8"))
    if request.get("status") != "REQUESTED_NOT_EXECUTED" or completion.get("event_id") != request.get("event_id"):
        raise ValueError("Completion must match one actual pending request")
    if completion.get("status") not in {"completed", "failed", "cancelled_after_work"}:
        raise ValueError("Completion status must preserve success/failure/cancelled-after-work")
    cost = person_minutes(completion["active_intervals"])
    ledger = Path(ledger_path); ledger.parent.mkdir(parents=True, exist_ok=True)
    existing = read_jsonl(ledger) if ledger.exists() else []
    if any(r["event_id"] == request["event_id"] for r in existing):
        raise ValueError("Duplicate completed event_id; do not silently overwrite paid work")
    path = completion.get("outcome_artifact_path")
    artifact = None
    if path:
        artifact = Path(path)
        if not artifact.is_absolute():
            artifact = Path(completion_path).resolve().parent / artifact
        if not artifact.is_file() or artifact.stat().st_size == 0:
            raise ValueError("Outcome artifact must exist and be nonempty")
        if completion.get("outcome_artifact_sha256") != sha256_file(artifact):
            raise ValueError("Outcome artifact SHA256 mismatch")
    if completion["status"] == "completed" and (artifact is None or not completion.get("after_state_version")):
        raise ValueError("Completed revisions require an actual artifact and after-state version")
    if not isinstance(completion.get("reason"), str) or not completion["reason"].strip():
        raise ValueError("Record what changed, or why the paid action failed")
    quote = float(request["quoted_max_cost_minutes"])
    record = {**request, "status":completion["status"], "cost_minutes":cost,
              "quote_overrun_minutes":max(0.0,cost-quote),"active_intervals":completion["active_intervals"],
              "after_state_version":completion.get("after_state_version",request["before_state_version"]),
              "outcome_artifact_path":str(artifact) if artifact is not None else None,
              "outcome_artifact_sha256":sha256_file(artifact) if artifact is not None else None,
              "reason":completion["reason"],"reward":None,"reward_status":"NOT_MEASURED",
              "action_completed_after_max_quote":cost>quote+1e-9,
              "cost_unit":"expert_person_minutes_sum_of_active_intervals",
              "previous_ledger_sha256":sha256_file(ledger) if ledger.exists() else None}
    with ledger.open("a",encoding="utf-8") as stream:
        stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+"\n")
    return record


@dataclass
class DetectorCommandAdapter:
    """Execute researcher-supplied `[command..., --request file, --response file]`.

    Training input contains visible annotations only; held-out test images may be
    passed only to predict. The wrapper must honor input data boundaries. File
    validation is useful but does not create OS-level isolation from test truth.
    """
    command: list[str]
    timeout_seconds: float = 3600.0

    def _invoke(self, request, output_dir):
        if not isinstance(self.command,(list,tuple)) or not self.command or not all(isinstance(s,str) and s for s in self.command):
            raise ValueError("command must be a nonempty list, not a shell expression")
        output = Path(output_dir).resolve(); output.mkdir(parents=True,exist_ok=True)
        request_path, response_path = output/"adapter_request.json",output/"adapter_response.json"
        if response_path.exists():
            raise ValueError("Use a new run directory; do not accept a stale model response")
        write_json(request_path,request)
        with (output/"adapter_stdout.log").open("w",encoding="utf-8") as stdout, (output/"adapter_stderr.log").open("w",encoding="utf-8") as stderr:
            # No shell=True. The executable and its implementation are locally supplied.
            result = subprocess.run([*self.command,"--request",str(request_path),"--response",str(response_path)],
                                    stdout=stdout,stderr=stderr,timeout=self.timeout_seconds,check=False)
        for name in ("adapter_stdout.log","adapter_stderr.log"):
            path=output/name
            if path.stat().st_size==0: path.write_text("(no output)\n",encoding="utf-8")
        if result.returncode:
            raise RuntimeError(f"Detector wrapper failed with exit {result.returncode}; inspect {output}")
        if not response_path.is_file():
            raise ValueError("Detector wrapper did not return its required response JSON")
        response=json.loads(response_path.read_text(encoding="utf-8"))
        if response.get("request_id") != request["request_id"] or response.get("status") != "completed":
            raise ValueError("Mismatched or incomplete detector response")
        return response

    def fit(self, manifest_path, output_dir, *, seed=17, config=None):
        manifest = Path(manifest_path).resolve(); rows=read_jsonl(manifest)
        if not rows or any(r.get("split") not in DETECTOR_TRAINING_SPLITS or not r.get("active",r.get("initial_labeled")) for r in rows):
            raise ValueError("Training manifests must contain only active, permitted training roles")
        if any(any(k in r for k in ("truth_labelmap_path","ground_truth","oracle_labelmap_path")) for r in rows):
            raise ValueError("Private truth fields are forbidden in detector training manifests")
        for row in rows:
            paths=[row["image_path"],*[a["labelmap_path"] for a in row.get("annotations",[])]]
            if row.get("current_labelmap_path"): paths.append(row["current_labelmap_path"])
            for value in paths:
                p=Path(value); resolved=(manifest.parent/p).resolve()
                if p.is_absolute() or ".." in p.parts or not resolved.is_relative_to(manifest.parent) or "oracle" in resolved.relative_to(manifest.parent).parts:
                    raise ValueError("Training pixels/labels must be visible relative files outside oracle/")
                if not resolved.is_file() or not resolved.stat().st_size:
                    raise ValueError("Missing current training artifact")
        request={"request_id":str(uuid.uuid4()),"operation":"fit", "training_manifest":str(manifest),
                 "output_dir":str(Path(output_dir).resolve()),"seed":int(seed),"config":config or {},
                 "training_specimen_ids":sorted({r["specimen_id"] for r in rows}),
                 "training_manifest_sha256":sha256_file(manifest),"input_scope":"visible current training annotations only"}
        response=self._invoke(request,output_dir)
        artifact=Path(response["model_path"])
        if not artifact.is_file() or not artifact.stat().st_size or response.get("model_sha256") != sha256_file(artifact):
            raise ValueError("Missing/corrupt detector artifact")
        if response.get("training_specimen_ids") != request["training_specimen_ids"]:
            raise ValueError("Wrapper must echo its exact training specimen set")
        return response

    def predict(self, manifest_path, model_path, output_dir):
        manifest=Path(manifest_path).resolve(); rows=read_jsonl(manifest)
        if not rows: raise ValueError("Empty inference manifest")
        # Materialize a whitelist: the inference worker gets no rater labels or
        # oracle paths, even if the original public manifest contains annotations.
        input_root=Path(output_dir).resolve()/"inference_inputs"; input_root.mkdir(parents=True,exist_ok=True)
        safe=[]
        for index,row in enumerate(rows):
            source=(manifest.parent/row["image_path"]).resolve()
            if not source.is_relative_to(manifest.parent) or "oracle" in source.relative_to(manifest.parent).parts:
                raise ValueError("Inference image paths cannot escape the dataset or enter oracle/")
            target=input_root/f"image_{index:06d}{source.suffix}"; shutil.copyfile(source,target)
            fields=("image_id","specimen_id","split","device_id","width","height","is_synthetic")
            public={k:row[k] for k in fields}
            public.update(image_path=target.name,annotations=[],initial_labeled=False,
                          available_actions=[],action_quotes_minutes={})
            safe.append(public)
        from .policy import write_jsonl
        inference_manifest=write_jsonl(input_root/"images_only.jsonl",safe)
        request={"request_id":str(uuid.uuid4()),"operation":"predict","inference_manifest":str(inference_manifest),
                 "model_path":str(Path(model_path).resolve()),"output_dir":str(Path(output_dir).resolve()),
                 "input_scope":"image pixels only; labels must not be accessed by predict"}
        response=self._invoke(request,output_dir)
        path=Path(response["predictions_path"])
        if not path.is_file() or not path.stat().st_size or response.get("predictions_sha256") != sha256_file(path):
            raise ValueError("Missing/corrupt prediction artifact")
        predicted=read_jsonl(path)
        expected_ids=[r["image_id"] for r in rows]; actual_ids=[r.get("image_id") for r in predicted]
        if sorted(expected_ids) != sorted(actual_ids):
            raise ValueError("Predictions must cover every requested image exactly once")
        return response
