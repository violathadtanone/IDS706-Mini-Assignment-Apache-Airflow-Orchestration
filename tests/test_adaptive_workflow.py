"""Part 3 behavioral tests, not the final cross-system evaluation suite."""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
import pytest

from ml_workflow import nodes as part2
from adaptive_workflow import agents, fixed, shared
from adaptive_workflow.graph import build_graph, run_workflow


@pytest.fixture
def records():
    rng = np.random.default_rng(21)
    return pd.DataFrame({
        'facility_id': [f'{i:06}' for i in range(100)],
        'measure_id': 'HAI_1_SIR', 'hospital_matched': True,
        'start_date': '2025-01-01', 'end_date': '2025-12-31',
        'overall_rating': rng.integers(1, 6, 100),
        'hospital_type': 'Acute Care Hospitals', 'state': ['NC', 'VA'] * 50,
        'score': rng.uniform(0.1, 2, 100),
    })


def mock_loader(monkeypatch, records):
    monkeypatch.setattr(part2, 'hospital_data_loader', lambda s: {'raw_data': records.copy()})


def invoke(monkeypatch, records, tmp_path, failures=0):
    mock_loader(monkeypatch, records)
    return run_workflow(tmp_path / f'run-{failures}', tmp_path / 'snapshot.pkl',
                        inject_db_failures=failures)


def test_shared_cohort_matches_part2_and_freezes_splits(records, tmp_path):
    expected = part2.cohort_builder({'raw_data': records, 'seed': 42})
    actual = shared.prepare(records, tmp_path / 'snapshot.pkl')
    for split in ['train', 'validation', 'test']:
        pd.testing.assert_frame_equal(actual[split], expected[split])
    assert actual['baseline_validation_mae'] == expected['baseline_validation_mae']
    sets = [set(actual[s].facility_id) for s in ['train', 'validation', 'test']]
    assert [len(s) for s in sets] == [60, 20, 20]
    assert all(not sets[i] & sets[j] for i, j in [(0, 1), (0, 2), (1, 2)])
    changed = records.copy()
    changed.loc[0, 'score'] += 1
    with pytest.raises(ValueError, match='frozen shared snapshot'):
        shared.prepare(changed, tmp_path / 'snapshot.pkl')


def test_fixed_exactly_one_fit_and_same_preprocessing(records, tmp_path, monkeypatch):
    mock_loader(monkeypatch, records)
    original = shared.train_ridge
    calls = []
    def train(state, alpha):
        calls.append(alpha)
        model = original(state, alpha)
        assert model.named_steps['preprocessing'].feature_names_in_.tolist() == part2.FEATURES
        return model
    monkeypatch.setattr(shared, 'train_ridge', train)
    cohort = fixed.load_shared_cohort(tmp_path / 'fixed', tmp_path / 'snapshot.pkl')
    trained = fixed.train_ridge(cohort)
    metrics = fixed.save_results(fixed.evaluate(trained))
    assert calls == [1.0]
    assert metrics['alpha'] == 1 and metrics['model_fit_count'] == 1
    assert metrics['selected_candidate'] == 'ridge'
    assert metrics['path'] == ['load_shared_cohort', 'train_ridge', 'evaluate', 'save_results']
    assert metrics['fingerprint'] == shared.prepare(records)['cohort_metadata']['fingerprint']
    assert all(np.isfinite(metrics[k]) for k in ['validation_mae', 'test_mae', 'baseline_validation_mae', 'baseline_test_mae'])
    assert json.loads((tmp_path / 'fixed/metrics.json').read_text()) == metrics


def test_fixed_dag_has_only_linear_dependencies():
    spec = importlib.util.spec_from_file_location('fixed_dag_test', 'dags/healthcare_fixed_ml_baseline.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    dag = module.healthcare_fixed_ml_baseline()
    assert set(dag.task_ids) == {'load_shared_cohort', 'train_ridge', 'evaluate', 'save_results'}
    assert {t.task_id: t.downstream_task_ids for t in dag.tasks} == {
        'load_shared_cohort': {'train_ridge'}, 'train_ridge': {'evaluate'},
        'evaluate': {'save_results'}, 'save_results': set()}
    assert all(t.retries == 1 for t in dag.tasks)


def test_normal_graph_selects_specialists_and_saves_evidence(records, monkeypatch, tmp_path):
    final = invoke(monkeypatch, records, tmp_path)
    result = final['result']
    assert result['status'] == 'completed'
    assert result['candidate_history'][0]['candidate'] == 'median_baseline'
    assert result['path'].count('median_specialist') == 1
    assert 1 <= result['model_fit_count'] <= 3
    assert len(result['used_alphas']) == len(set(result['used_alphas'])) == result['model_fit_count']
    assert result['path'].count('ridge_specialist') == result['model_fit_count']
    for i, name in enumerate(result['path']):
        if name in ['ridge_specialist', 'median_specialist']:
            assert result['path'][i - 1] == 'workflow_dispatcher'
    assert final['decisions_finished']
    assert result['path'][-1] == 'results_reporter'
    assert result['recovery_retry_count'] == 0
    assert np.isfinite(result['test_mae'])
    assert (tmp_path / 'run-0/routing_history.json').is_file()
    assert (tmp_path / 'run-0/test_predictions.csv').is_file()
    assert json.loads((tmp_path / 'run-0/metrics.json').read_text()) == result


def test_predictive_fixture_selects_ridge(records, monkeypatch, tmp_path):
    records['score'] = records.overall_rating * 0.2
    result = invoke(monkeypatch, records, tmp_path)['result']
    assert result['selected_candidate'] == 'ridge'
    assert result['model_fit_count'] == 2
    assert result['stop_reason'] == 'validation_improvement_at_least_5_percent'


def test_constant_target_selects_median_and_stops_on_stall(records, monkeypatch, tmp_path):
    records['score'] = 1.0
    result = invoke(monkeypatch, records, tmp_path)['result']
    assert result['selected_candidate'] == 'median_baseline'
    assert result['model_fit_count'] == 2
    assert result['test_mae'] == 0
    assert result['stop_reason'] == 'baseline_preferred_after_stalled_ridge_attempts'


@pytest.mark.parametrize('faults,reads,retries,status', [(1, 2, 1, 'completed'),
    (2, 3, 2, 'completed'), (20, 3, 2, 'failed')])
def test_transient_recovery_and_cap(records, monkeypatch, tmp_path, faults, reads, retries, status):
    final = invoke(monkeypatch, records, tmp_path, faults)
    result = final['result']
    assert result['status'] == status
    assert result['database_read_count'] == reads
    assert result['recovery_retry_count'] == retries
    assert 'recovery_coordinator' in result['path']
    assert result['path'][:4] == ['cohort_steward', 'workflow_dispatcher', 'recovery_coordinator', 'cohort_steward']
    if status == 'failed':
        assert result['model_fit_count'] == 0
        assert result['stop_reason'] == 'recovery_retry_budget_exhausted'
        assert 'test_mae' not in result


@pytest.mark.parametrize('problem', ['missing', 'duplicate', 'insufficient', 'period'])
def test_permanent_input_errors_never_retry(records, monkeypatch, tmp_path, problem):
    if problem == 'missing':
        records = records.drop(columns='score')
    elif problem == 'duplicate':
        records = pd.concat([records, records.iloc[:1]])
    elif problem == 'insufficient':
        records = records.iloc[:10]
    else:
        records['end_date'] = 'invalid'
    result = invoke(monkeypatch, records, tmp_path)['result']
    assert result['status'] == 'failed'
    assert result['database_read_count'] == 1
    assert result['recovery_retry_count'] == 0
    assert 'recovery_coordinator' not in result['path']
    assert result['model_fit_count'] == 0
    assert result['error']


def decision_state(attempts, baseline=1.0):
    return {'median_evaluated': True, 'model_fit_count': len(attempts),
            'attempts': attempts, 'baseline_validation_mae': baseline,
            'used_alphas': [a['alpha'] for a in attempts], 'path': []}


def test_dispatcher_uses_evidence_not_test_and_no_alpha_repeats():
    state = decision_state([{'alpha': 1.0, 'train_mae': 0.5, 'validation_mae': 1.2}])
    assert agents.workflow_dispatcher(state)['alpha'] == 10
    state['test'] = object()  # completely unusable held-out set
    state['test_mae'] = -1000
    a = agents.workflow_dispatcher(state)
    state['test_mae'] = 1e9
    assert agents.workflow_dispatcher(state) == a
    state['attempts'].append({'alpha': 10.0, 'train_mae': 1.0, 'validation_mae': 1.05})
    state.update(used_alphas=[1.0, 10.0], model_fit_count=2)
    # Proposed weaker alpha 1 is already used; opposite 100 must be selected.
    assert agents.workflow_dispatcher(state)['alpha'] == 100
    state.update(model_fit_count=3)
    decision = agents.workflow_dispatcher(state)
    assert decision['next_agent'] == 'results_reporter'
    assert decision['stop_reason'] == 'model_fit_budget_exhausted'


def test_guards_prevent_unscheduled_or_excess_fits_and_early_report():
    with pytest.raises(ValueError, match='not selected'):
        agents.ridge_specialist({})
    with pytest.raises(ValueError, match='budget exhausted'):
        agents.ridge_specialist({'next_agent': 'ridge_specialist', 'model_fit_count': 3})
    with pytest.raises(ValueError, match='already used'):
        agents.ridge_specialist({'next_agent': 'ridge_specialist', 'model_fit_count': 1,
                                'alpha': 1.0, 'used_alphas': [1.0]})
    with pytest.raises(ValueError, match='before adaptive decisions'):
        agents.results_reporter({})


def test_database_failure_classification():
    assert agents.transient_database_error(psycopg2.OperationalError('connection refused'))
    assert not agents.transient_database_error(psycopg2.errors.UndefinedColumn('missing score'))
    assert not agents.transient_database_error(psycopg2.OperationalError('password authentication failed'))
    assert not agents.transient_database_error(ValueError('bad data'))


def test_failed_fit_consumes_budget(records, monkeypatch):
    state = {**shared.prepare(records), 'next_agent': 'ridge_specialist',
             'model_fit_count': 0, 'used_alphas': [], 'alpha': 1.0}
    def fail(*args):
        raise RuntimeError('Training failed')
    monkeypatch.setattr(shared, 'train_ridge', fail)
    patch = agents.ridge_specialist(state)
    assert patch['model_fit_count'] == 1 and patch['used_alphas'] == [1.0]
    state.update(patch)
    assert agents.workflow_dispatcher(state)['next_agent'] == 'results_reporter'
    assert not patch['recoverable']


def test_model_budget_reached_in_real_graph(records, monkeypatch, tmp_path):
    original = agents.score_auditor
    def controlled(state):
        patch = original(state)
        if state['candidate_kind'] == 'ridge':
            # Improving, but not beating baseline: no plateau early exit.
            n = state['model_fit_count']
            patch['attempts'][-1]['validation_mae'] = 10 - n
        return patch
    monkeypatch.setattr(agents, 'score_auditor', controlled)
    result = invoke(monkeypatch, records, tmp_path)['result']
    assert result['model_fit_count'] == 3
    assert len(set(result['used_alphas'])) == 3
    assert result['stop_reason'] == 'model_fit_budget_exhausted'


def test_final_test_evaluation_only_once_after_dispatch(records, monkeypatch, tmp_path):
    original = shared.test_predictions
    calls = []
    def predict(model, state):
        assert state['decisions_finished']
        assert state['next_agent'] == 'results_reporter'
        calls.append(1)
        return original(model, state)
    monkeypatch.setattr(shared, 'test_predictions', predict)
    final = invoke(monkeypatch, records, tmp_path)
    assert calls == [1]
    with pytest.raises(ValueError, match='already performed'):
        agents.results_reporter(final)


def test_changed_test_targets_do_not_change_real_graph_decisions(records, monkeypatch, tmp_path):
    first = invoke(monkeypatch, records, tmp_path / 'first')['result']
    original = shared.load_prepared
    def altered(*args, **kwargs):
        prepared = original(*args, **kwargs)
        prepared['test'] = prepared['test'].copy()
        prepared['test']['score'] = 1e9
        return prepared
    monkeypatch.setattr(shared, 'load_prepared', altered)
    second = invoke(monkeypatch, records, tmp_path / 'second')['result']
    for key in ['routing_history', 'used_alphas', 'selected_candidate', 'stop_reason', 'attempts']:
        assert first[key] == second[key]
    assert first['test_mae'] != second['test_mae']
