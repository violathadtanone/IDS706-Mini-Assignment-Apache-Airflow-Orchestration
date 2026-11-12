"""Run controlled fixtures against unchanged production agents and fixed tasks.

Fault injection/instrumentation is scoped to this sequential evaluation process.
No production database writes, model metric overrides or tuning-rule changes.
"""
import argparse
import copy
import hashlib
import json
import time
from datetime import datetime, timezone
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import psycopg2

from adaptive_workflow import agents, fixed, shared
from adaptive_workflow.graph import build_graph as adaptive_graph
from ml_workflow import nodes as part2
from ml_workflow.graph import build_graph as simple_graph

SYSTEMS = ('fixed_airflow', 'simple_agentic', 'adaptive_multi_agent')
BUDGETS = {'fixed_airflow': (1, 1), 'simple_agentic': (3, 0), 'adaptive_multi_agent': (3, 2)}


def synthetic(spec):
    rng = np.random.default_rng(spec['seed'])
    n = spec['rows']
    ratings = rng.integers(1, 6, n)
    frame = pd.DataFrame({'facility_id': [f'{i:06}' for i in range(n)],
        'measure_id': 'HAI_1_SIR', 'hospital_matched': True,
        'start_date': '2025-01-01', 'end_date': '2025-12-31',
        'overall_rating': ratings, 'hospital_type': 'Acute Care Hospitals',
        'state': np.where(np.arange(n) % 2, 'NC', 'VA')})
    target = spec['target']
    if target == 'constant_1':
        frame['score'] = 1.0
    elif target == 'rating_times_0.2':
        frame['score'] = ratings * 0.2
    elif target == 'weak_rating_signal':
        frame['score'] = np.maximum(0.01, 1 + ratings * 0.04 + rng.normal(0, 0.4, n))
    else:
        raise ValueError(f'Unknown synthetic target {target}')
    return frame


def error_category(message):
    text = message.lower()
    if 'injected' in text or 'connection' in text or 'database' in text:
        return 'database_failure'
    if 'missing required columns' in text:
        return 'schema_error'
    if 'duplicate facility' in text:
        return 'data_integrity_error'
    return 'execution_error'


def run_system(system, raw, output, failed_reads=0, alter_test=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    observed = {'system': system, 'status': 'failed', 'path': [], 'alphas': [],
                'read_attempts': 0, 'model_fit_count': 0, 'test_evaluations': 0,
                'budget_violations': 0, 'split_leakage_violations': 0,
                'test_leakage_violations': 0, 'split_ids': {}, 'result': None}
    real_cohort, real_train = part2.cohort_builder, part2.score_model_trainer
    real_predict = shared.test_predictions
    started = time.perf_counter()

    def loader(state):
        observed['read_attempts'] += 1
        if observed['read_attempts'] <= failed_reads:
            raise psycopg2.OperationalError('Injected PostgreSQL connection failure')
        return {'raw_data': raw.copy(deep=True)}

    def cohort(state):
        prepared = real_cohort(state)
        ids = {s: prepared[s].facility_id.tolist() for s in ['train', 'validation', 'test']}
        overlap = any(set(ids[a]) & set(ids[b]) for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')])
        observed['split_ids'] = ids
        observed['split_leakage_violations'] += int(overlap)
        if alter_test:
            prepared['test'] = prepared['test'].copy()
            prepared['test']['score'] = 1000000.0
        observed['cohort_fingerprint'] = hashlib.sha256(json.dumps(ids, sort_keys=True).encode()).hexdigest()
        observed['prepared'] = prepared
        return prepared

    def trainer(state):
        observed['model_fit_count'] += 1
        observed['alphas'].append(float(state['alpha']))
        # The test frame is an unusable sentinel during model fitting. A
        # production trainer that accesses it will fail this evaluation.
        guarded = {**state, 'test': object()}
        trained = real_train(guarded)
        assert trained['model'].named_steps['preprocessing'].feature_names_in_.tolist() == part2.FEATURES
        return trained

    def test_predictions(model, state):
        if system == 'adaptive_multi_agent' and not state.get('decisions_finished'):
            observed['test_leakage_violations'] += 1
        observed['test_evaluations'] += 1
        return real_predict(model, state)

    def observe(status, node, state):
        if status == 'running':
            observed['path'].append(node)
            if node == 'results_publisher':
                observed['test_evaluations'] += 1
                observed['test_leakage_violations'] += int(not state.get('tuning_finished'))

    try:
        with ExitStack() as stack:
            stack.enter_context(patch.object(part2, 'hospital_data_loader', loader))
            stack.enter_context(patch.object(part2, 'cohort_builder', cohort))
            stack.enter_context(patch.object(part2, 'score_model_trainer', trainer))
            stack.enter_context(patch.object(shared, 'test_predictions', test_predictions))
            if system == 'fixed_airflow':
                # Same four authoritative task bodies. Only the load task is
                # fault-injected; retain the DAG's one ordinary retry and delay.
                observed['execution_mode'] = 'fixed_dag_task_bodies_with_ordinary_retry'
                cohort_path = None
                for attempt in range(2):
                    observed['path'].append('load_shared_cohort')
                    try:
                        cohort_path = fixed.load_shared_cohort(output, output / 'cohort_snapshot.pkl')
                        break
                    except Exception:
                        if attempt == 1:
                            raise
                        time.sleep(5)
                observed['path'].append('train_ridge')
                trained = fixed.train_ridge(cohort_path)
                observed['path'].append('evaluate')
                evaluated = fixed.evaluate(trained)
                observed['path'].append('save_results')
                result = fixed.save_results(evaluated)
            elif system == 'simple_agentic':
                observed['execution_mode'] = 'actual_langgraph'
                final = simple_graph(observer=observe).invoke(
                    {'output_dir': str(output), 'seed': 42, 'path': []}, {'recursion_limit': 30})
                result = final['result']
                result['validation_mae'] = result['selected_validation_mae']
            else:
                observed['execution_mode'] = 'actual_langgraph'
                final = {'output_dir': str(output), 'path': [], 'routing_history': [],
                         'model_fit_count': 0, 'recovery_retry_count': 0, 'database_read_count': 0}
                for update in adaptive_graph().stream(final, stream_mode='updates', config={'recursion_limit': 50}):
                    for node, delta in update.items():
                        observed['path'].append(node)
                        final.update(delta)
                result = final['result']
                if result['status'] == 'failed':
                    observed.update(error=result['error'], error_category=error_category(result['error']))
            observed['result'] = result
            if result.get('status', 'completed') != 'failed':
                observed['status'] = 'completed'
    except Exception as exc:
        observed.update(error=str(exc), error_category=error_category(str(exc)))
    observed['runtime_seconds'] = time.perf_counter() - started
    observed['retry_count'] = max(0, observed['read_attempts'] - 1)
    max_fits, max_retries = BUDGETS[system]
    observed['budget_violations'] = int(observed['model_fit_count'] > max_fits or observed['retry_count'] > max_retries)
    observed['executed_steps'] = len(observed['path'])
    observed['unique_alphas'] = len(observed['alphas']) == len(set(observed['alphas']))
    observed['artifacts_present'] = all((output / name).is_file() for name in ['metrics.json', 'test_predictions.csv'])
    observed.pop('prepared', None)
    shared.save_json(output / 'observation.json', observed)
    return observed


def tuning_signature(observed):
    result = observed['result'] or {}
    return {'path': observed['path'], 'alphas': observed['alphas'],
            'selected_candidate': result.get('selected_candidate'),
            'validation_mae': result.get('validation_mae'),
            'attempts': result.get('attempts'), 'routing_history': result.get('routing_history'),
            'stop_reason': result.get('stop_reason')}


def score_case(case, system, observation, paired=None):
    result = observation['result'] or {}
    name = case['name']
    checks = {'within_budgets': observation['budget_violations'] == 0,
              'disjoint_splits': observation['split_leakage_violations'] == 0,
              'test_only_after_tuning': observation['test_leakage_violations'] == 0,
              'unique_alphas': observation['unique_alphas']}
    if name in ['missing_required_column', 'conflicting_facility', 'persistent_read_failure']:
        category = {'missing_required_column': 'schema_error', 'conflicting_facility': 'data_integrity_error',
                    'persistent_read_failure': 'database_failure'}[name]
        checks.update(rejected=observation['status'] == 'failed',
                      correct_category=observation.get('error_category') == category,
                      no_fit=observation['model_fit_count'] == 0, no_test=observation['test_evaluations'] == 0)
        if name == 'persistent_read_failure':
            checks['bounded_reads'] = observation['read_attempts'] == BUDGETS[system][1] + 1
        elif system == 'adaptive_multi_agent':
            checks['no_permanent_recovery'] = 'recovery_coordinator' not in observation['path']
    else:
        checks.update(completed=observation['status'] == 'completed',
                      artifacts=observation['artifacts_present'], splits_observed=bool(observation['split_ids']),
                      finite_metrics=all(isinstance(result.get(k), (int, float)) and np.isfinite(result[k])
                          for k in ['validation_mae', 'test_mae', 'baseline_validation_mae', 'baseline_test_mae']),
                      one_final_test=observation['test_evaluations'] == 1)
        if name == 'constant_target':
            tolerance = case['pass_fail_threshold']['absolute_tolerance']
            checks.update(median_selected=result.get('selected_candidate') == 'median_baseline',
                          no_false_improvement=abs(result.get('validation_mae', float('inf')) - result.get('baseline_validation_mae', 0)) <= tolerance)
        elif name == 'ridge_improves':
            checks.update(ridge_selected=result.get('selected_candidate') == 'ridge',
                genuine_improvement=result.get('validation_mae', float('inf')) <=
                    (1 - case['pass_fail_threshold']['minimum_relative_improvement']) * result.get('baseline_validation_mae', 0))
        elif name == 'one_transient_read_failure':
            checks.update(two_reads=observation['read_attempts'] == 2, one_retry=observation['retry_count'] == 1)
            if system == 'adaptive_multi_agent':
                checks['recovery_agent_visited'] = 'recovery_coordinator' in observation['path']
        elif name == 'test_target_isolation':
            checks.update(unchanged_tuning=tuning_signature(observation) == tuning_signature(paired),
                          perturbation_effective=result.get('test_mae') != (paired['result'] or {}).get('test_mae'),
                          paired_completed=paired['status'] == 'completed',
                          paired_controls=all(paired[k] == 0 for k in ['budget_violations', 'split_leakage_violations', 'test_leakage_violations']))
        elif name == 'model_attempt_budget':
            checks['fit_cap_exercised'] = observation['model_fit_count'] == BUDGETS[system][0]
    fraction = sum(checks.values()) / len(checks)
    applicable = system not in case.get('not_applicable', {})
    return {'applicable': applicable,
            'finding': case.get('not_applicable', {}).get(system, ''),
            'checks': checks, 'score': fraction,
            'pass': fraction >= case['pass_fail_threshold']['checks_fraction'] if applicable else None,
            'outcome': ('pass' if fraction == 1 else 'fail') if applicable else 'not_applicable'}


def run_suite(output_dir='output/evaluation'):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    run_root = output / 'runs' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    definition = json.loads(Path(__file__).with_name('cases.json').read_text())
    raw = part2.hospital_data_loader({})['raw_data']
    snapshot = shared.prepare(raw)
    raw.to_csv(output / 'postgresql_snapshot.csv', index=False)
    shared.dump(output / 'frozen_cohort.pkl', snapshot)
    cases = []
    rows = []
    for case in definition['cases']:
        spec = case['input']
        frame = raw.copy(deep=True) if spec['fixture'] == 'postgresql_frozen' else synthetic(spec)
        if spec.get('drop_column'):
            frame = frame.drop(columns=spec['drop_column'])
        if spec.get('duplicate'):
            valid = shared.prepare(frame)
            eligible = valid['train'].facility_id.iloc[0]
            duplicate = frame.loc[frame.facility_id.eq(eligible) & frame.start_date.eq(frame.loc[frame.facility_id.eq(eligible), 'start_date'].max())].iloc[:1].copy()
            duplicate['score'] += 0.5
            frame = pd.concat([frame, duplicate], ignore_index=True)
        frame.to_csv(output / f'{case["name"]}_input.csv', index=False)
        item = copy.deepcopy(case)
        item['systems'] = {}
        for system in SYSTEMS:
            destination = run_root / case['name'] / system
            observed = run_system(system, frame, destination, spec.get('failed_reads', 0))
            paired = None
            if spec.get('paired_variant'):
                paired = run_system(system, frame, destination / 'altered_test_targets', alter_test=True)
                observed['paired_observed'] = paired
                if tuning_signature(observed) != tuning_signature(paired):
                    observed['test_leakage_violations'] += 1
            scored = score_case(case, system, observed, paired)
            item['systems'][system] = {'observed_result': observed, **scored}
            result = observed['result'] or {}
            baseline = result.get('baseline_validation_mae')
            mae = result.get('validation_mae')
            aggregate = {k: observed[k] + (paired[k] if paired else 0) for k in
                ['model_fit_count', 'retry_count', 'executed_steps', 'runtime_seconds',
                 'budget_violations', 'split_leakage_violations', 'test_leakage_violations']}
            rows.append({'case': case['name'], 'system': system, 'outcome': scored['outcome'],
                'workflow_status': observed['status'], 'validation_mae': mae, 'test_mae': result.get('test_mae'),
                'validation_improvement_vs_median': baseline - mae if baseline is not None and mae is not None else None,
                'test_improvement_vs_median': result.get('baseline_test_mae', 0) - result['test_mae'] if 'test_mae' in result else None,
                'selected_candidate': result.get('selected_candidate'),
                **aggregate})
            print(f'{case["name"]}: {system}: {scored["outcome"]}', flush=True)
        successful = [v['observed_result'] for v in item['systems'].values() if v['observed_result']['status'] == 'completed']
        item['same_split_assignments'] = len({v.get('cohort_fingerprint') for v in successful}) <= 1
        cases.append(item)
    summary = summarize(cases, rows)
    controls = all(c['same_split_assignments'] for c in cases) and all(
        r[k] == 0 for r in rows for k in ['budget_violations', 'split_leakage_violations', 'test_leakage_violations'])
    # Only this predeclared architectural gap is an expected failed baseline
    # capability. Any other unexpected baseline failure fails the suite too.
    expected_gaps = {('one_transient_read_failure', 'simple_agentic')}
    unexpected = [(c['name'], s) for c in cases for s, result in c['systems'].items()
                  if result['pass'] is False and (c['name'], s) not in expected_gaps]
    passed = controls and not unexpected and all(c['systems']['adaptive_multi_agent']['pass'] for c in cases)
    report = {'definition': definition, 'frozen_cohort': snapshot['cohort_metadata'],
              'case_count': len(cases), 'run_directory': str(run_root),
              'suite_passed': passed, 'controls_passed': controls, 'unexpected_failures': unexpected,
              'cases': cases, 'summary': summary}
    shared.save_json(output / 'comparison.json', report)
    pd.DataFrame(rows).to_csv(output / 'comparison.csv', index=False)
    pd.DataFrame(summary).to_csv(output / 'summary.csv', index=False)
    return report


def summarize(cases, rows):
    summaries = []
    for system in SYSTEMS:
        results = [c['systems'][system] for c in cases]
        applicable = [r for r in results if r['applicable']]
        valid = [c['systems'][system] for c in cases if c['name'] not in
                 ['persistent_read_failure', 'missing_required_column', 'conflicting_facility']]
        invalid = [c['systems'][system] for c in cases if c['name'] in ['missing_required_column', 'conflicting_facility']]
        transient = next(c['systems'][system] for c in cases if c['name'] == 'one_transient_read_failure')
        normal = next(c['systems'][system]['observed_result'] for c in cases if c['name'] == 'normal_snapshot')
        own_rows = [r for r in rows if r['system'] == system]
        summaries.append({'system': system, 'passed': sum(r['pass'] is True for r in applicable),
            'failed': sum(r['pass'] is False for r in applicable), 'not_applicable': len(results) - len(applicable),
            'successful_completion_rate': sum(r['observed_result']['status'] == 'completed' for r in valid) / len(valid),
            'correct_invalid_rejection_rate': sum(r['pass'] is True for r in invalid) / len(invalid),
            'recovery_success_rate': float(transient['observed_result']['status'] == 'completed'),
            'normal_validation_mae': (normal['result'] or {}).get('validation_mae'), 'normal_test_mae': (normal['result'] or {}).get('test_mae'),
            'total_model_fits': sum(r['model_fit_count'] for r in own_rows),
            'total_retries': sum(r['retry_count'] for r in own_rows),
            'total_executed_steps': sum(r['executed_steps'] for r in own_rows),
            'runtime_seconds': sum(r['runtime_seconds'] for r in own_rows),
            'budget_violations': sum(r['budget_violations'] for r in own_rows),
            'split_leakage_violations': sum(r['split_leakage_violations'] for r in own_rows),
            'test_leakage_violations': sum(r['test_leakage_violations'] for r in own_rows)})
    return summaries


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default='output/evaluation')
    args = parser.parse_args()
    report = run_suite(args.output_dir)
    print(json.dumps(report['summary'], indent=2))
    raise SystemExit(0 if report['suite_passed'] else 1)
