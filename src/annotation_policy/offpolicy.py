"""Single-step, within-case annotation-action OPE with explicit overlap checks.

This is NOT an estimator of adaptive case selection or a budget trajectory.
Provenance assertions are auditable metadata, not proof of causal assumptions.
"""
from __future__ import annotations
from collections import defaultdict
import math
import numpy as np
from .schema import validate_action_logs


def _probability_vector(values, available, label):
    if not isinstance(values, dict) or set(values) != set(available):
        raise ValueError(f"{label}: complete probabilities on exactly the available actions are required")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in values.values()):
        raise ValueError(f"{label}: probabilities must be finite and nonnegative")
    if abs(sum(values.values()) - 1) > 1e-8:
        raise ValueError(f"{label}: probabilities must sum to one")
    return {a: float(values[a]) for a in available}


def _event_values(rows, supplied, key):
    if supplied is None:
        return {r['event_id']: r.get(key) for r in rows}
    if not isinstance(supplied, dict) or set(supplied) != {r['event_id'] for r in rows}:
        raise ValueError(f"{key}: supplied values must be keyed by exactly all evaluated event IDs")
    return supplied


def _check_provenance(provenance, evaluation_ids, label, allow_predeclared=False):
    if not isinstance(provenance, dict):
        raise ValueError(f"{label}: explicit held-out training/tuning provenance is required")
    mode = provenance.get('mode')
    if mode == 'predeclared' and allow_predeclared:
        if not isinstance(provenance.get('description'), str) or not provenance['description'].strip():
            raise ValueError(f"{label}: a predeclared policy needs a description")
        if provenance.get('training_specimen_ids') or provenance.get('tuning_specimen_ids'):
            raise ValueError(f"{label}: a fitted policy cannot be labelled predeclared")
        return
    if mode != 'held_out':
        raise ValueError(f"{label}: this implementation supports external group-held-out provenance only; use independent folds before calling it")
    for field in ('training_specimen_ids', 'tuning_specimen_ids'):
        values = provenance.get(field)
        if not isinstance(values, list) or any(not isinstance(v, str) or not v for v in values):
            raise ValueError(f"{label}: explicit {field} list is required, including an empty tuning list when appropriate")
        overlap = set(values) & evaluation_ids
        if overlap:
            raise ValueError(f"{label}: evaluated specimens were used for fitting/tuning: {sorted(overlap)}")
    if not provenance['training_specimen_ids']:
        raise ValueError(f"{label}: fitted model needs nonempty training specimen provenance")
    if provenance.get('evaluation_rewards_used_for_selection') is True:
        raise ValueError(f"{label}: evaluation rewards cannot select the model or target policy")


def _estimates(reward, cost, weights, q_target=None, q_observed=None, max_weight=None):
    w = np.minimum(weights, max_weight) if max_weight is not None else weights
    total_weight = float(w.sum()); n = len(w)
    ips = float(np.mean(w * reward))
    cost_ips = float(np.mean(w * cost))
    result = {'ips': ips, 'snips': float(np.sum(w * reward) / total_weight) if total_weight > 0 else None,
              'dr': float(np.mean(q_target + w * (reward - q_observed))) if q_target is not None else None,
              'expected_cost_ips': cost_ips,
              'expected_cost_snips': float(np.sum(w * cost) / total_weight) if total_weight > 0 else None,
              'reward_per_expected_minute': ips / cost_ips if cost_ips > 0 else None,
              'effective_sample_size': float(total_weight**2 / np.sum(w**2)) if np.sum(w**2) > 0 else 0.,
              'weight_sum': total_weight, 'mean_weight': total_weight/n, 'max_observed_weight': float(w.max()),
              'nonzero_weight_events': int(np.sum(w > 0)),
              'clipped_events': int(np.sum(weights > max_weight)) if max_weight is not None else 0,
              'max_weight': max_weight}
    return result


def evaluate_ope(records, target_probabilities=None, *, target_policy_provenance=None,
                 q_predictions=None, q_provenance=None, max_weight=None,
                 n_bootstrap=2000, seed=17):
    """IPS/SNIPS and optional held-out DR, averaging over logged case contexts.

    ``target_probabilities`` and ``q_predictions`` are event_id -> action -> value.
    Alternatively records can carry target_policy_probs and q_by_action.
    Realized action cost must be positive even for failure; reward may be negative.
    All target-supported actions need positive logger support in every context.
    Bootstrap samples specimen clusters and keeps repeated contexts together.
    """
    rows = list(records)
    if not rows: raise ValueError('OPE requires nonempty randomized action logs')
    validation = validate_action_logs(rows)
    if not validation['ok']: raise ValueError('Invalid action logs: ' + '; '.join(validation['errors']))
    if n_bootstrap < 0 or not isinstance(n_bootstrap, int): raise ValueError('n_bootstrap must be a nonnegative integer')
    if max_weight is not None and (isinstance(max_weight, bool) or not math.isfinite(max_weight) or max_weight <= 0):
        raise ValueError('max_weight must be finite and positive')
    evaluation_ids = {r['specimen_id'] for r in rows}
    _check_provenance(target_policy_provenance, evaluation_ids, 'target policy', allow_predeclared=True)
    target = _event_values(rows, target_probabilities, 'target_policy_probs')
    q_values = _event_values(rows, q_predictions, 'q_by_action')
    has_q = any(v is not None for v in q_values.values())
    if has_q:
        if any(v is None for v in q_values.values()): raise ValueError('DR needs reward predictions for every evaluated event')
        _check_provenance(q_provenance, evaluation_ids, 'DR reward model')
    elif q_provenance is not None:
        raise ValueError('q_provenance supplied without q predictions')
    weights, rewards, costs, qt, qo, props = [], [], [], [], [], []
    clusters = defaultdict(list)
    for index, row in enumerate(rows):
        eid, actions, chosen = row['event_id'], row['available_actions'], row['action']
        logger = _probability_vector(row.get('logging_policy_probs'), actions, eid + ' logger')
        policy = _probability_vector(target[eid], actions, eid + ' target')
        if abs(logger[chosen] - row['propensity']) > 1e-8:
            raise ValueError(eid + ': chosen-action propensity disagrees with logging_policy_probs')
        unsupported = [a for a in actions if policy[a] > 0 and logger[a] <= 0]
        if unsupported: raise ValueError(eid + ': target action lacks logger overlap: ' + ','.join(unsupported))
        weights.append(policy[chosen] / logger[chosen]); props.extend(logger[a] for a in actions if policy[a] > 0)
        rewards.append(row['reward']); costs.append(row['cost_minutes']); clusters[row['specimen_id']].append(index)
        if has_q:
            q = q_values[eid]
            if not isinstance(q, dict) or set(q) != set(actions) or any(isinstance(v, bool) or not isinstance(v, (int,float)) or not math.isfinite(v) for v in q.values()):
                raise ValueError(eid + ': q predictions must be finite for exactly all available actions')
            qt.append(sum(policy[a]*q[a] for a in actions)); qo.append(q[chosen])
    w, r, c = map(lambda a: np.asarray(a, dtype=float), (weights, rewards, costs))
    qt, qo = (np.asarray(qt), np.asarray(qo)) if has_q else (None, None)
    result = _estimates(r, c, w, qt, qo, max_weight)
    result.update({'estimand': 'single-step within-case action value over the logged case-context distribution',
                   'not_estimable_here': 'adaptive joint case/action selection, repeated repairs, or fixed-budget risk trajectories',
                   'records': len(rows), 'specimens': len(clusters),
                   'all_synthetic': all(row.get('is_synthetic') is True for row in rows),
                   'dr_status': 'external_group_held_out_predictions' if has_q else 'not_computed_no_independent_reward_model',
                   'target_policy_provenance': target_policy_provenance,
                   'q_provenance': q_provenance if has_q else None,
                   'minimum_logger_probability_on_target_support': min(props),
                   'largest_unclipped_weight': float(w.max()),
                   'unclipped': _estimates(r, c, w, qt, qo),
                   'clipping_sensitivity': {str(cap): _estimates(r, c, w, qt, qo, cap) for cap in (2.,5.,10.)}})
    metrics = ('ips','snips','dr','expected_cost_ips','expected_cost_snips','reward_per_expected_minute')
    samples = {m: [] for m in metrics}; keys = sorted(clusters); rng = np.random.default_rng(seed)
    if len(keys) >= 2 and n_bootstrap >= 20:
        for _ in range(n_bootstrap):
            draw = rng.integers(0, len(keys), len(keys))
            indices = np.concatenate([np.asarray(clusters[keys[int(i)]], dtype=int) for i in draw])
            estimate = _estimates(r[indices],c[indices],w[indices],qt[indices] if has_q else None,qo[indices] if has_q else None,max_weight)
            for metric in metrics:
                if estimate[metric] is not None: samples[metric].append(estimate[metric])
    result['confidence_intervals'] = {m: {'estimate': result[m], 'ci': np.quantile(samples[m],[.025,.975]).tolist() if len(samples[m]) >= 20 else None,
                                        'valid_draws': len(samples[m]), 'method': 'percentile_specimen_cluster_bootstrap', 'confidence': .95}
                                       for m in metrics}
    result['bootstrap_repetitions'] = n_bootstrap
    warnings = ['OPE conditions on fixed target policy, reward model, development reward panel and logged context population.',
                'Specimen bootstrap does not include shared reward-panel sampling or new policy/acquisition-run uncertainty.',
                'Metadata exclusion checks cannot prove correct logging randomization, hidden confounding absence or honest provenance.',
                'Reward per expected minute is a ratio of single-step expectations, not value under an adaptive budget.',
                'Self-normalization and weight clipping introduce finite-sample bias; clipped DR is not generally doubly robust.']
    if result['effective_sample_size'] < max(10, .1*len(rows)): warnings.append('Low overlap precision: effective sample size is small; do not make a confirmatory claim.')
    if min(props) < .02: warnings.append('Very small logging probability on target support; predeclare a restricted estimand or improve exploration.')
    if result['weight_sum'] == 0: warnings.append('No observed selected action received target-policy weight: SNIPS and reward/cost are undefined, not zero.')
    if len(clusters) < 2 or n_bootstrap < 20: warnings.append('Too few specimen clusters or bootstrap draws for an interval.')
    if max_weight is not None: warnings.append('Primary returned estimators are clipped; inspect the unclipped estimate and full sensitivity report.')
    result['warnings'] = validation.get('warnings', []) + warnings
    return result
