"""Specimen-cluster inference and completed-action expert-time curves.

Intervals in this file condition on the fitted policy and its acquisition
trajectory. They do NOT make a single trajectory into many independent runs.
Repeat independent acquisition runs/held-out pools for policy-level uncertainty.
"""
from __future__ import annotations
import hashlib
import json
import math
import numpy as np
from scipy import stats

METRIC_KEYS = {
    "critical_miss_rate": ("critical_misses", "critical_images"),
    "false_clear_rate": ("false_clears", "defect_images"),
    "critical_instance_recall": ("critical_instances_matched", "critical_instances"),
    "normal_false_alarm_rate": ("normal_false_alarms", "normal_images"),
}


def _ratio(array):
    total = np.asarray(array, dtype=float).sum(axis=0)
    return float(total[0] / total[1]) if total[1] > 0 else None


def _validate_contributions(clusters):
    if not isinstance(clusters, dict): raise ValueError("Contributions must be keyed by independent specimen")
    keys = sorted(clusters)
    array = np.asarray([clusters[k] for k in keys], dtype=float).reshape(-1, 2)
    if not np.isfinite(array).all() or (array[:, 1] < 0).any():
        raise ValueError("Cluster contributions must be finite [numerator,nonnegative denominator]")
    return keys, array


def metric_contributions(per_specimen, metric):
    if metric == "risk":
        return {s: [float(p["risk_sum"] / p["image_count"]), 1.0] if p["image_count"] > 0 else [0.0, 0.0]
                for s, p in per_specimen.items()}
    if metric not in METRIC_KEYS: raise ValueError("Unknown metric: " + metric)
    numerator, denominator = METRIC_KEYS[metric]
    return {s: [float(p[numerator]), float(p[denominator])] for s, p in per_specimen.items()}


def cluster_ratio_ci(clusters, n_bootstrap=2000, seed=17, confidence=.95):
    keys, array = _validate_contributions(clusters)
    if n_bootstrap < 0 or not 0 < confidence < 1: raise ValueError("Invalid bootstrap settings")
    point = _ratio(array) if len(keys) else None
    result = {"estimate": point, "ci": None, "confidence": confidence, "unit": "specimen",
              "n_specimens": len(keys), "eligible_specimens": int((array[:, 1] > 0).sum()),
              "numerator": float(array[:, 0].sum()), "denominator": float(array[:, 1].sum()),
              "bootstrap_repetitions": n_bootstrap, "valid_bootstrap_draws": 0,
              "method": "percentile_specimen_cluster_bootstrap"}
    if point is None:
        result["reason"] = "zero_eligible_denominator"; return result
    if len(keys) < 2 or n_bootstrap < 20:
        result["reason"] = "at_least_2_specimens_and_20_draws_needed_for_interval"; return result
    rng = np.random.default_rng(seed); samples = []
    for start in range(0, n_bootstrap, 256):
        ids = rng.integers(0, len(keys), size=(min(256, n_bootstrap - start), len(keys)))
        sums = array[ids].sum(axis=1); valid = sums[:, 1] > 0
        samples.extend((sums[valid, 0] / sums[valid, 1]).tolist())
    result["valid_bootstrap_draws"] = len(samples)
    if len(samples) >= 20:
        alpha = (1 - confidence) / 2
        result["ci"] = np.quantile(samples, [alpha, 1 - alpha]).astype(float).tolist()
    else:
        result["reason"] = "too_few_defined_bootstrap_draws"
    warnings = []
    if samples and np.ptp(samples) == 0:
        warnings.append("degenerate_interval_does_not_prove_zero_population_risk")
    if len(samples) < .9 * n_bootstrap:
        warnings.append("many_bootstrap_samples_have_zero_denominator")
    if warnings: result["warnings"] = warnings
    return result


def metric_confidence_intervals(per_specimen, n_bootstrap=2000, seed=17):
    return {metric: cluster_ratio_ci(metric_contributions(per_specimen, metric), n_bootstrap, seed)
            for metric in ("risk", *METRIC_KEYS)}


def paired_cluster_difference(clusters_a, clusters_b, n_bootstrap=2000,
                              n_permutations=2000, seed=17, confidence=.95):
    """B minus A, resampling or exchanging whole paired specimens together."""
    ka, a = _validate_contributions(clusters_a); kb, b = _validate_contributions(clusters_b)
    if ka != kb: raise ValueError("Paired comparisons require exactly the same specimen IDs")
    if n_bootstrap < 0 or n_permutations < 0 or not 0 < confidence < 1:
        raise ValueError("Invalid inference settings")
    pa, pb = (_ratio(x) if len(ka) else None for x in (a, b))
    result = {"difference_B_minus_A": None if pa is None or pb is None else pb - pa,
              "estimate_A": pa, "estimate_B": pb, "ci": None, "confidence": confidence,
              "paired_permutation_p_two_sided": None, "n_specimens": len(ka),
              "unit": "specimen", "valid_bootstrap_draws": 0,
              "method": "paired_specimen_bootstrap_and_block_label_exchange"}
    if pa is None or pb is None:
        result["reason"] = "zero_eligible_denominator"; return result
    if len(ka) < 2:
        result["reason"] = "fewer_than_two_independent_specimens"; return result
    rng = np.random.default_rng(seed); deltas = []
    for start in range(0, n_bootstrap, 256):
        ids = rng.integers(0, len(ka), size=(min(256, n_bootstrap-start), len(ka)))
        aa, bb = a[ids].sum(1), b[ids].sum(1)
        valid = (aa[:, 1] > 0) & (bb[:, 1] > 0)
        deltas.extend((bb[valid, 0] / bb[valid, 1] - aa[valid, 0] / aa[valid, 1]).tolist())
    result["valid_bootstrap_draws"] = len(deltas)
    if len(deltas) >= 20:
        alpha = (1-confidence)/2
        result["ci"] = np.quantile(deltas, [alpha, 1-alpha]).astype(float).tolist()
    if deltas and np.ptp(deltas) == 0:
        result["warning"] = "degenerate_difference_interval_conditions_on_observed_test_specimens"
    extreme, defined = 0, 0
    for _ in range(n_permutations):
        flip = rng.random(len(ka)) < .5
        perm_a, perm_b = np.where(flip[:, None], b, a), np.where(flip[:, None], a, b)
        aa, bb = _ratio(perm_a), _ratio(perm_b)
        if aa is None or bb is None: continue
        defined += 1
        extreme += abs(bb-aa) >= abs(pb-pa) - 1e-12
    if defined:
        result["paired_permutation_p_two_sided"] = float((extreme+1)/(defined+1))
    result["valid_permutation_draws"] = defined
    result["permutation_assumption"] = "whole-specimen model-label exchangeability under the null; not automatic randomization of adaptive policies"
    return result


def holm_adjust(p_values):
    """Keep the full declared family size, even for unavailable endpoints."""
    valid = []
    for key, value in p_values.items():
        if value is None: continue
        if not math.isfinite(value) or not 0 <= value <= 1: raise ValueError("Invalid p value")
        valid.append((key, value))
    valid.sort(key=lambda kv: kv[1]); result = {k: None for k in p_values}; previous = 0.0
    for index, (key, value) in enumerate(valid):
        previous = max(previous, min(1.0, (len(p_values)-index)*value)); result[key] = previous
    return result


def summarize_independent_runs(records, confidence=.95):
    """Summarize paired policy differences over distinct acquisition-run seeds.

    Every row is one complete paired trajectory comparison at the same fixed H
    and test cohort, not a budget point, test image or bootstrap draw. This
    estimates run-randomness variability conditional on that test cohort.
    Synthetic seeds cannot establish variability of real expert panels/sites.
    """
    rows = list(records)
    if not rows or not 0 < confidence < 1: raise ValueError('Nonempty independent run records and valid confidence are required')
    for field in ('run_id','acquisition_seed'):
        values = [r.get(field) for r in rows]
        if any(v is None for v in values) or len(set(values)) != len(values):
            raise ValueError(f'Each whole acquisition run needs a distinct {field}')
    cohorts = {r.get('heldout_cohort_id') for r in rows}
    horizons = {r.get('horizon_minutes') for r in rows}
    if None in cohorts or len(cohorts) != 1 or None in horizons or len(horizons) != 1:
        raise ValueError('Run summaries must share an explicit heldout_cohort_id and fixed horizon_minutes')
    differences = np.asarray([r['difference_B_minus_A'] for r in rows], dtype=float)
    if not np.isfinite(differences).all() or (np.abs(differences)>1).any():
        raise ValueError('Each run difference must be finite and within [-1,1] for normalized risk AUC')
    n=len(rows); mean=float(differences.mean()); sd=float(differences.std(ddof=1)) if n>1 else None
    interval = None
    if n>1:
        half=float(stats.t.ppf((1+confidence)/2,n-1))*sd/math.sqrt(n)
        interval=[mean-half,mean+half]
    return {'n_independent_runs':n,'mean_difference_B_minus_A':mean,'run_seed_sd':sd,
            'mean_difference_ci':interval,'confidence':confidence,
            'method':'paired_run_mean_t_interval_requires_independent_runs_and_adequate_distributional_approximation',
            'heldout_cohort_id':next(iter(cohorts)),'horizon_minutes':next(iter(horizons)),
            'uncertainty_scope':'policy/acquisition-run randomness conditional on one fixed test cohort',
            'not_covered':['test-population sampling uncertainty','unrepresented expert/site/domain variability'],
            'warning':'Run IDs and distinct seeds do not themselves prove independent real data or human panels; inspect the run design.'
                       + (' Fewer than 10 runs: treat the t interval as descriptive and fragile.' if n<10 else '')}


def _event_time(event):
    if "minutes" in event: value = event["minutes"]
    elif "cumulative_cost_minutes" in event: value = event["cumulative_cost_minutes"]
    else: raise ValueError("Every event needs cumulative completed-action minutes")
    if "minutes" in event and "cumulative_cost_minutes" in event and abs(event["minutes"]-event["cumulative_cost_minutes"]) > 1e-8:
        raise ValueError("Conflicting cumulative-time fields")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("Event minutes must be finite and nonnegative")
    return float(value)


def _event_eval(event):
    return event.get("evaluation", event)


def _validate_events(events):
    events = list(events)
    if not events or abs(_event_time(events[0])) > 1e-10:
        raise ValueError("The first trajectory event must be a pre-action baseline at minute zero")
    previous = -1.0
    for event in events:
        minute = _event_time(event)
        if minute <= previous: raise ValueError("Completed-action times must be strictly increasing")
        risk = _event_eval(event).get("risk")
        if risk is not None and (not isinstance(risk, (int, float)) or isinstance(risk, bool) or not math.isfinite(risk) or not 0 <= risk <= 1):
            raise ValueError("Risk must be a finite [0,1] loss or explicit null")
        previous = minute
    return events


def step_risk_auc(events, horizon_minutes):
    """Integrate the right-continuous, completed-action risk step function.

    Outcomes completed after H are ignored, not linearly interpolated into H.
    A policy that stops early keeps its last observable risk to the common H.
    This is risk versus expert person-minutes, not GPU/annotator wall time.
    """
    if isinstance(horizon_minutes, bool) or not isinstance(horizon_minutes, (int, float)) or not math.isfinite(horizon_minutes) or horizon_minutes <= 0:
        raise ValueError("The common budget horizon must be finite and positive")
    events = _validate_events(events); horizon = float(horizon_minutes)
    prefix = [e for e in events if _event_time(e) <= horizon]
    if any(_event_eval(e).get("risk") is None for e in prefix):
        return {"auc": None, "normalized_auc": None, "final_risk": None,
                "horizon_minutes": horizon, "reason": "undefined_risk_at_a_required_budget_state"}
    auc = 0.0
    for index, event in enumerate(prefix):
        left = _event_time(event)
        right = min(horizon, _event_time(prefix[index+1])) if index+1 < len(prefix) else horizon
        auc += (right-left)*float(_event_eval(event)["risk"])
    times = [_event_time(e) for e in prefix]; risks = [float(_event_eval(e)["risk"]) for e in prefix]
    if times[-1] < horizon:
        times.append(horizon); risks.append(risks[-1])
    return {"auc": float(auc), "normalized_auc": float(auc/horizon), "final_risk": risks[-1],
            "horizon_minutes": horizon, "completed_actions": len(prefix)-1,
            "last_completed_action_minutes": _event_time(prefix[-1]),
            "carry_forward_minutes": horizon-_event_time(prefix[-1]),
            "future_events_ignored": len(events)-len(prefix),
            "step_times": times, "step_risks": risks,
            "integration": "right_continuous_completed_action_step_function_no_linear_interpolation"}


def trajectory_summary(events, horizon_minutes, n_bootstrap=2000, seed=17):
    """nAUC and specimen-conditioned CI for one already frozen trajectory."""
    events = _validate_events(events)
    summary = step_risk_auc(events, horizon_minutes)
    prefix = [e for e in events if _event_time(e) <= horizon_minutes]
    initial = _event_eval(prefix[0]).get("per_specimen")
    if not isinstance(initial, dict) or not initial:
        raise ValueError("Specimen contributions are required for trajectory inference")
    ids = sorted(initial); fingerprint_data = []
    for sid in ids:
        p = initial[sid]
        fingerprint_data.append([sid, sorted(p.get("image_ids", [])), p["image_count"], p.get("reference_sha256")])
    for event in prefix:
        current = _event_eval(event).get("per_specimen")
        if not isinstance(current, dict) or sorted(current) != ids:
            raise ValueError("Every budget state must evaluate exactly the same specimen cohort")
        for sid in ids:
            if (current[sid]["image_count"] != initial[sid]["image_count"] or
                sorted(current[sid].get("image_ids", [])) != sorted(initial[sid].get("image_ids", [])) or
                current[sid].get("reference_sha256") != initial[sid].get("reference_sha256")):
                raise ValueError("The evaluation cohort or truth-eligible denominator changes across budget states")
            for key in ('critical_images','defect_images','critical_instances','normal_images','known_pixels','unknown_pixels'):
                if current[sid].get(key) != initial[sid].get(key):
                    raise ValueError("Independent reference denominators change across budget states")
        actual = [p["risk_sum"]/p["image_count"] for p in current.values() if p["image_count"] > 0]
        reported = _event_eval(event).get("risk")
        if actual and (reported is None or abs(float(np.mean(actual))-reported) > 1e-8):
            raise ValueError("Reported overall risk disagrees with specimen-macro contributions")
    contributions = {}
    for sid in ids:
        if initial[sid]["image_count"] == 0:
            contributions[sid] = [0.0, 0.0]; continue
        curve = [{"minutes": _event_time(e), "risk": _event_eval(e)["per_specimen"][sid]["risk_sum"] / initial[sid]["image_count"]}
                 for e in prefix]
        contributions[sid] = [step_risk_auc(curve, horizon_minutes)["normalized_auc"], 1.0]
    interval = cluster_ratio_ci(contributions, n_bootstrap, seed)
    if summary["normalized_auc"] is not None and interval["estimate"] is not None and abs(interval["estimate"]-summary["normalized_auc"]) > 1e-8:
        raise ValueError("Trajectory nAUC and clustered nAUC disagree")
    summary.update({"nauc_confidence_interval": interval, "per_specimen_nauc": contributions,
                    "cohort_sha256": hashlib.sha256(json.dumps(fingerprint_data, sort_keys=True).encode()).hexdigest(),
                    "endpoint_confidence_intervals": metric_confidence_intervals(_event_eval(prefix[-1])["per_specimen"], n_bootstrap, seed),
                    "uncertainty_scope": "test-specimen sampling conditional on this fixed acquisition trajectory and trained policy",
                    "not_covered": ["policy-training randomness", "different action outcomes/expert panels", "new acquisition pools", "new device/site populations"]})
    return summary


def compare_trajectories(events_a, events_b, horizon_minutes, n_bootstrap=2000,
                         n_permutations=2000, seed=17):
    a = trajectory_summary(events_a, horizon_minutes, n_bootstrap, seed)
    b = trajectory_summary(events_b, horizon_minutes, n_bootstrap, seed)
    if a["cohort_sha256"] != b["cohort_sha256"]:
        raise ValueError("Paired trajectory comparison requires the same held-out cohort")
    primary = paired_cluster_difference(a["per_specimen_nauc"], b["per_specimen_nauc"], n_bootstrap, n_permutations, seed)
    pa = [_event_eval(e) for e in events_a if _event_time(e) <= horizon_minutes][-1]["per_specimen"]
    pb = [_event_eval(e) for e in events_b if _event_time(e) <= horizon_minutes][-1]["per_specimen"]
    endpoint = {metric: paired_cluster_difference(metric_contributions(pa, metric), metric_contributions(pb, metric),
                                                 n_bootstrap, n_permutations, seed)
                for metric in ("risk", *METRIC_KEYS)}
    family = {"normalized_auc": primary["paired_permutation_p_two_sided"],
              "critical_miss_rate": endpoint["critical_miss_rate"]["paired_permutation_p_two_sided"],
              "false_clear_rate": endpoint["false_clear_rate"]["paired_permutation_p_two_sided"]}
    return {"difference_direction": "B_minus_A", "horizon_minutes": horizon_minutes,
            "primary_normalized_auc": primary, "endpoint_comparisons": endpoint,
            "predeclared_family_holm_p": holm_adjust(family), "paired_unit": "specimen",
            "cohort_sha256": a["cohort_sha256"],
            "limitations": ["Confidence intervals condition on each already-frozen trajectory.",
                            "Repeat independent acquisition runs and expert outcomes to study policy-level uncertainty.",
                            "Repeated budget points are correlated measurements, not extra independent samples.",
                            "No test-risk information may be fed back into action selection, early stopping or threshold choice."]}


def _wilson(success, total, confidence=.95):
    z = float(stats.norm.ppf(.5+confidence/2)); phat = success/total
    denom = 1+z*z/total; center=(phat+z*z/(2*total))/denom
    radius=z*math.sqrt(phat*(1-phat)/total+z*z/(4*total*total))/denom
    return [center-radius, center+radius]


def estimate_paired_power(n_specimens=(100, 150, 240, 300), effect=.02, paired_sd=.08,
                          attrition=.15, alpha=.05, target_power=.8, repetitions=2000, seed=17):
    """Planning-only normal paired-specimen nAUC-difference simulation.

    Paired SD is at the specimen level AFTER constructing each specimen's full
    trajectory nAUC. Do not insert a pixel/patch SD and call it specimen power.
    This model does not capture policy-run uncertainty or multiple endpoints.
    """
    if effect <= 0 or paired_sd <= 0 or not 0 <= attrition < 1 or not 0 < alpha < 1 or not 0 < target_power < 1 or repetitions < 20:
        raise ValueError("Invalid planning assumptions")
    if any(type(n) is not int or n < 4 for n in n_specimens): raise ValueError("Each candidate n must be integer >=4")
    z = float(stats.norm.ppf(1-alpha/2) + stats.norm.ppf(target_power))
    complete = math.ceil((z*paired_sd/effect)**2)
    rng = np.random.default_rng(seed); runs = []
    for enrolled in n_specimens:
        n = math.floor(enrolled*(1-attrition))
        if n < 3: raise ValueError("Too few specimens remain after planned attrition")
        cutoff = float(stats.t.ppf(1-alpha/2, n-1)); found = 0
        for start in range(0, repetitions, 256):
            differences = rng.normal(-effect, paired_sd, size=(min(256,repetitions-start),n))
            statistics = differences.mean(1)/(differences.std(1,ddof=1)/np.sqrt(n))
            found += int((statistics < -cutoff).sum())
        runs.append({"enrolled_specimens": enrolled, "analyzed_specimens": n,
                     "power_estimate": found/repetitions, "monte_carlo_ci_95": _wilson(found,repetitions),
                     "simulation_repetitions": repetitions})
    return {"status": "PLANNING_ASSUMPTIONS_NOT_EMPIRICAL_RESULTS", "seed": seed,
            "assumptions": {"meaningful_nauc_reduction": effect, "paired_specimen_sd": paired_sd,
                            "attrition": attrition, "two_sided_alpha": alpha, "target_power": target_power,
                            "difference_model": "normal Full-minus-baseline specimen nAUC differences with mean=-effect"},
            "analytic_approximation": {"complete_specimens": complete, "enrolled_specimens": math.ceil(complete/(1-attrition))},
            "simulation": runs,
            "limitations": ["Planning SD and effect must be replaced by independent pilot estimates.",
                            "Monte Carlo intervals describe simulation error only, not uncertainty about real-world power.",
                            "Conditions on fixed policies/acquisition trajectories; additional independent runs are needed for policy uncertainty.",
                            "Does not include multiple-comparison correction or device/site clustering.",
                            "The normal approximation is a planning model, not fabricated bounded empirical nAUC observations."]}
