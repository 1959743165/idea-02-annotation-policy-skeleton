"""Measured annotation-action pilots and budget-constrained prospective simulation.

Learning rewards are actual before/after detector risk differences on explicitly
separate development specimens. Final test evaluation occurs only after every
action in a budget trajectory is locked. No policy performance is guaranteed.
"""
from __future__ import annotations
from pathlib import Path
from collections import Counter
import hashlib
import json
import math
import time
import numpy as np
import yaml
from .policy import GainCostPolicy, fit_policy, choose_case_action, write_jsonl, read_jsonl


def read_config(path):
    """CPU configuration paths are relative to the invoking working directory."""
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return path


def _core():
    from .dataset import load_public_cases
    from .detector import train_detector, Detector
    from .features import describe_case
    from .oracle import SyntheticOracle
    from .evaluation import load_truth, evaluate_detector
    return load_public_cases, train_detector, Detector, describe_case, SyntheticOracle, load_truth, evaluate_detector


def _reward_partitions(cases, seed):
    groups = sorted({c.specimen_id for c in cases.values() if c.split == "reward"})
    if len(groups) < 4:
        raise ValueError("At least four reward specimens are needed for disjoint fit/tuning reward panels")
    order = np.random.default_rng(seed + 9001).permutation(len(groups))
    first = {groups[i] for i in order[:len(groups)//2]}
    second = set(groups) - first
    return first, second


def collect_development_logs(manifest_path, output_dir, config):
    load_public_cases, train_detector, _, describe_case, SyntheticOracle, load_truth, evaluate_detector = _core()
    from .schema import validate_dataset
    validation = validate_dataset(manifest_path, check_files=True, check_hashes=True)
    if not validation["ok"]:
        raise ValueError("Dataset rejected: " + json.dumps(validation["errors"], ensure_ascii=False))
    cases = load_public_cases(manifest_path)
    if not all(c.is_synthetic for c in cases.values()):
        raise ValueError("This collector uses a synthetic oracle. Real cases require a recorded human executor and detector adapter.")
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    seed = int(config.get("seed", 17))
    baseline_cases = [c for c in cases.values() if c.split == "seed_train"]
    if not baseline_cases:
        raise ValueError("No seed_train specimens")
    reward_fit_ids, reward_tune_ids = _reward_partitions(cases, seed)
    truth = load_truth(manifest_path, split="reward")
    oracle = SyntheticOracle(manifest_path, seed=seed)
    detector_config = config.get("detector", {})
    evaluation_config = config.get("evaluation", {})
    all_logs = {}
    measured = []
    for role, reward_ids, offset in (("policy_train", reward_fit_ids, 0), ("policy_val", reward_tune_ids, 1)):
        reward_cases = {iid:c for iid,c in cases.items() if c.split == "reward" and c.specimen_id in reward_ids}
        candidates = [c for c in cases.values() if c.split == role]
        rng = np.random.default_rng(seed + 100 + offset)
        order = rng.permutation(len(candidates))
        limit = config.get("pilot", {}).get("max_train_cases" if role == "policy_train" else "max_val_cases")
        if limit is not None:
            order = order[:int(limit)]
        rows = []
        for j, index in enumerate(order):
            case = candidates[int(index)]
            before_cases = baseline_cases + ([case] if case.active else [])
            before = train_detector(before_cases, detector_config, seed=seed)
            features = describe_case(case, before)
            available = list(case.available_actions)
            if not available:
                continue
            action = available[int(rng.integers(len(available)))]
            event_id = f"{role}_{j:04d}_{case.image_id}"
            start = time.perf_counter()
            pre = evaluate_detector(before, reward_cases, truth, evaluation_config)
            result = oracle.apply(case, action, event_id=event_id)
            after = train_detector(baseline_cases + ([result.case] if result.case.active else []), detector_config, seed=seed)
            post = evaluate_detector(after, reward_cases, truth, evaluation_config)
            if pre["risk"] is None or post["risk"] is None:
                raise ValueError("A development reward is undefined; do not replace it with zero")
            reward = float(pre["risk"] - post["risk"])
            snapshot = result.case.save_snapshot(output / "paid_states" / f"{event_id}.npz")
            row = {"event_id":event_id,"image_id":case.image_id,"specimen_id":case.specimen_id,
                   "split":role,"is_synthetic":True,"features":features,"available_actions":available,
                   "action":action,"propensity":1.0/len(available),
                   "logging_policy_probs":{a:1.0/len(available) for a in available},
                   "logging_policy":"uniform action conditional on predeclared case; not a sequential case-selection propensity",
                   "quoted_max_cost_minutes":float(case.action_quotes_minutes[action]),
                   "action_quotes_minutes":dict(case.action_quotes_minutes),
                   "cost_minutes":float(result.cost_minutes),"success":bool(result.success),
                   "reward":reward,"before_risk":float(pre["risk"]),"after_risk":float(post["risk"]),
                   "reward_source_split":"reward","reward_specimen_ids":sorted(reward_ids),
                   "reward_panel":"reward_fit" if role == "policy_train" else "reward_tune",
                   "reward_is_measured_by_detector_retraining":True,
                   "detector_seed_before_after":seed,"detector_init_matched":True,
                   "wall_seconds_compute_only":time.perf_counter()-start,
                   "paid_state_snapshot":str(snapshot),"action_details":result.details}
            rows.append(row)
        target = write_jsonl(output / f"{role}_logs.jsonl", rows)
        all_logs[role] = {"path":str(target),"rows":rows}
        measured.append({"split":role,"events":len(rows),"action_counts":dict(Counter(r["action"] for r in rows)),
                         "simulated_expert_minutes":sum(r["cost_minutes"] for r in rows),
                         "reward_specimens":sorted(reward_ids)})
    summary = {"status":"SYNTHETIC_RANDOMIZED_ACTION_PILOT_ONLY","real_experts_used":False,
               "reward_fit_and_tune_disjoint":not bool(reward_fit_ids & reward_tune_ids),
               "final_test_used_for_rewards":False,"measured_risk_changes_not_scripted_utilities":True,
               "dataset_validation":validation,"roles":measured,
               "scope":"Each development case receives one randomized action. The estimated marginal gain is conditional on the common seed training set, not a guaranteed sequential treatment effect."}
    write_json(output / "pilot_summary.json", summary)
    return all_logs


def train_policy_from_logs(train_log_path, val_log_path, output_path, config=None):
    train_rows, val_rows = read_jsonl(train_log_path), read_jsonl(val_log_path)
    fit_ids = {x for r in train_rows for x in r.get("reward_specimen_ids", [])}
    val_ids = {x for r in val_rows for x in r.get("reward_specimen_ids", [])}
    if not fit_ids or not val_ids or fit_ids & val_ids:
        raise ValueError("Policy fit/tune rewards require explicitly recorded disjoint development reward specimen sets")
    model = fit_policy(train_rows, val_rows, config)
    model.metadata["reward_fit_specimen_ids"] = sorted(fit_ids)
    model.metadata["reward_tune_specimen_ids"] = sorted(val_ids)
    model.metadata["training_logs_sha256"] = hashlib.sha256(Path(train_log_path).read_bytes()).hexdigest()
    model.metadata["validation_logs_sha256"] = hashlib.sha256(Path(val_log_path).read_bytes()).hexdigest()
    model.save(output_path)
    write_json(Path(output_path).with_suffix(".json"), model.metadata)
    return model


def run_budget(manifest_path, policy, output_dir, config, strategy="learned_gain_per_minute", seed=None):
    """Lock the entire paid sequence and checkpoints before loading test truth."""
    load_public_cases, train_detector, _, describe_case, SyntheticOracle, _, _ = _core()
    cases = load_public_cases(manifest_path, splits=["seed_train", "acquisition_pool"])
    if not cases or not all(c.is_synthetic for c in cases.values()):
        raise ValueError("Built-in budget executor is synthetic only; use the real human/detector contracts")
    seed = int(config.get("seed", 17) if seed is None else seed)
    rng = np.random.default_rng(seed + 700)
    seed_cases = [c.clone() for c in cases.values() if c.split == "seed_train"]
    pool = {iid:c.clone() for iid,c in cases.items() if c.split == "acquisition_pool"}
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    budget = float(config.get("budget_minutes", 20))
    if not math.isfinite(budget) or budget <= 0:
        raise ValueError("Positive finite budget required")
    oracle = SyntheticOracle(manifest_path, seed=seed)
    detector_config = config.get("detector", {})
    detector = train_detector(seed_cases + [c for c in pool.values() if c.active], detector_config, seed=seed)
    initial = detector.save(output / "checkpoints/model_000.npz")
    events = [{"step":0,"minutes":0.0,"checkpoint":str(initial.relative_to(output)),"action":None,"image_id":None,
               "test_metrics_available_to_selector":False}]
    used, spent = set(), 0.0
    paid_log = []
    started = time.perf_counter()
    while True:
        contexts = [{"image_id":iid,"features":describe_case(case,detector),
                     "available_actions":list(case.available_actions),"action_quotes_minutes":case.action_quotes_minutes}
                    for iid,case in sorted(pool.items()) if iid not in used]
        decision = choose_case_action(contexts, policy, strategy, budget-spent, rng)
        if decision is None:
            break
        iid, action = decision["image_id"], decision["action"]
        case = pool[iid]
        features_before = next(x["features"] for x in contexts if x["image_id"] == iid)
        step = len(events); event_id = f"budget_{step:03d}_{iid}_{action}"
        result = oracle.apply(case, action, event_id=event_id)
        actual = float(result.cost_minutes)
        if not 0 < actual <= decision["quoted_max_cost_minutes"] + 1e-9:
            raise ValueError("Executor exceeded its public maximum quote. Record all real cost as an overrun; do not silently discard or retrospectively select another action.")
        spent += actual
        if spent > budget + 1e-9:
            raise AssertionError("Strict synthetic budget exceeded")
        pool[iid] = result.case; used.add(iid)
        snapshot = result.case.save_snapshot(output / f"paid_states/{event_id}.npz")
        detector = train_detector(seed_cases + [c for c in pool.values() if c.active], detector_config, seed=seed)
        checkpoint = detector.save(output / f"checkpoints/model_{step:03d}.npz")
        event = {"step":step,"minutes":float(spent),"checkpoint":str(checkpoint.relative_to(output)),"image_id":iid,
                 "specimen_id":case.specimen_id,"action":action,"cost_minutes":actual,
                 "quoted_max_cost_minutes":decision["quoted_max_cost_minutes"],"success":bool(result.success),
                 "paid_state_snapshot":str(snapshot.relative_to(output)),"test_metrics_available_to_selector":False}
        events.append(event)
        paid_log.append({"event_id":event_id,"image_id":iid,"specimen_id":case.specimen_id,
                         "split":"acquisition_pool","is_synthetic":True,"strategy":strategy,
                         "features_before":features_before,"selection":decision,"event":event,
                         "action_details":result.details,"is_randomized_logging_policy":False,
                         "propensity_for_offpolicy_evaluation":None})
    timeline = {"status":"SYNTHETIC_LOCKED_ACQUISITION_TRAJECTORY","strategy":strategy,"seed":seed,
                "budget_minutes":budget,"spent_minutes":float(spent),"unspent_minutes":float(budget-spent),
                "stop_reason":"no_remaining_supported_case_action_fits_strategy_and_public_quote",
                "artifact_base":str(output.resolve()),
                "artifact_path_resolution":"relative to locked_timeline.json parent; artifact_base is an in-memory runtime fallback",
                "actions_per_case_limit":1,"paid_actions":len(paid_log),
                "completed_actions":sum(bool(e["event"]["success"]) for e in paid_log),
                "failed_actions":sum(not bool(e["event"]["success"]) for e in paid_log),"events":events,
                "paid_cost_includes_failed_actions":True,"future_realized_cost_used_for_selection":False,
                "test_used_during_selection":False,"selector_inputs":"public context, available actions, maximum quotes, trained policy, remaining budget",
                "wall_seconds_compute_only":time.perf_counter()-started,
                "sunk_initial_and_pilot_annotation_time_in_incremental_budget":False,
                "initial_and_pilot_cost_note":"Common pre-existing annotations/pilot are excluded from the incremental budget but must be reported separately in a real total-human-effort analysis."}
    if paid_log:
        write_jsonl(output / "paid_action_log.jsonl", paid_log)
    else:
        write_json(output / "no_paid_actions.json", {"status":"NO_FEASIBLE_PAID_ACTIONS","spent_minutes":0})
    write_json(output / "locked_timeline.json", timeline)
    return timeline


def evaluate_locked_timeline(manifest_path, timeline, output_dir, evaluation_config=None, split="test"):
    """Only this post-lock stage reads independent test truth and generates metrics."""
    load_public_cases, _, Detector, _, _, load_truth, evaluate_detector = _core()
    if isinstance(timeline, (str, Path)):
        artifact_base=Path(timeline).resolve().parent
        timeline = json.loads(Path(timeline).read_text(encoding="utf-8"))
    else:
        artifact_base=Path(timeline["artifact_base"])
    if timeline.get("test_used_during_selection") is not False:
        raise ValueError("No locked trajectory provenance")
    if split not in ("test", "device_test"):
        raise ValueError("Final budget evaluation accepts locked test/device_test only")
    cases = load_public_cases(manifest_path, splits=[split])
    if not cases:
        raise ValueError("Empty final evaluation split")
    truth = load_truth(manifest_path, split=split)
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    events = []
    for event in timeline["events"]:
        checkpoint=Path(event["checkpoint"])
        model = Detector.load(checkpoint if checkpoint.is_absolute() else artifact_base/checkpoint)
        report = evaluate_detector(model, cases, truth, evaluation_config)
        events.append({**event,"risk":report["risk"],"per_specimen":report["per_specimen"],
                       "critical_miss_rate":report["critical_miss_rate"],"false_clear_rate":report["false_clear_rate"]})
        write_json(output / f"metrics_step_{event['step']:03d}.json", report)
    result = {"status":"SYNTHETIC_TEST_REPLAY_AFTER_POLICY_LOCK","strategy":timeline["strategy"],"seed":timeline["seed"],
              "split":split,"horizon_minutes":timeline["budget_minutes"],"spent_minutes":timeline["spent_minutes"],
              "test_specimen_ids":sorted({c.specimen_id for c in cases.values()}),"events":events,
              "real_training_or_human_annotation_performed":False,
              "test_feedback_used_for_selection":False,
              "interpretation":"Software-test trajectory only; independent final evaluation occurs after selection is frozen."}
    write_json(output / "evaluated_timeline.json", result)
    return result


def run_experiment(manifest_path, output_dir, config):
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    write_json(output / "resolved_config.json", config)
    logs = collect_development_logs(manifest_path, output / "pilot", config)
    policy = train_policy_from_logs(logs["policy_train"]["path"], logs["policy_val"]["path"], output / "policy.npz", config.get("policy", {}))
    results = {}
    # Lock ALL methods before any test scoring, not merely each action before its score.
    locked = {}
    for strategy in config.get("strategies", ["random", "uncertainty_per_minute", "learned_gain_per_minute"]):
        locked[strategy] = run_budget(manifest_path, policy, output / strategy, config, strategy)
    for strategy, trajectory in locked.items():
        results[strategy] = evaluate_locked_timeline(manifest_path, trajectory, output / strategy / "test", config.get("evaluation", {}))
    summary = {"status":"COMPLETED_SYNTHETIC_ANNOTATION_POLICY_PIPELINE","real_training_performed":False,
               "real_expert_annotations_collected":False,"policy":str(output / "policy.npz"),
               "train_logs":logs["policy_train"]["path"],"val_logs":logs["policy_val"]["path"],
               "strategies":{k:{"trajectory":str(output/k/"test/evaluated_timeline.json"),
                                   "paid_actions":locked[k]["paid_actions"],"completed_actions":locked[k]["completed_actions"],
                                   "failed_actions":locked[k]["failed_actions"],"spent_minutes":locked[k]["spent_minutes"]}
                             for k in results},
               "all_methods_locked_before_test_evaluation":True,
               "scientific_hypotheses_confirmed":False}
    write_json(output / "experiment_summary.json", summary)
    return summary
