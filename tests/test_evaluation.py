"""Evaluation definitions and scoring must preserve honest capability findings."""
import json
from pathlib import Path

import pytest

from evaluation import run


def test_case_contract_and_baseline_provenance():
    definition = json.loads(Path('evaluation/cases.json').read_text())
    assert len(definition['cases']) == 9
    assert len({c['name'] for c in definition['cases']}) == 9
    for case in definition['cases']:
        assert {'name', 'input', 'expected_outcome', 'scoring_criterion', 'pass_fail_threshold'} <= case.keys()
        assert case['pass_fail_threshold']['checks_fraction'] == 1.0
    assert 'adaptation' in definition['systems']['fixed_airflow']
    assert 'adaptation' in definition['systems']['simple_agentic']


def test_transient_failure_is_not_redefined_as_baseline_success(tmp_path, monkeypatch):
    definition = json.loads(Path('evaluation/cases.json').read_text())
    case = next(c for c in definition['cases'] if c['name'] == 'one_transient_read_failure')
    frame = run.synthetic({'seed': 21, 'rows': 100, 'target': 'constant_1'})
    observation = run.run_system('simple_agentic', frame, tmp_path, failed_reads=1)
    scored = run.score_case(case, 'simple_agentic', observation)
    assert observation['read_attempts'] == 1
    assert observation['model_fit_count'] == 0
    assert scored['pass'] is False
    assert scored['outcome'] == 'fail'


def test_fixed_always_ridge_is_explicitly_not_applicable_for_selection(tmp_path):
    definition = json.loads(Path('evaluation/cases.json').read_text())
    case = next(c for c in definition['cases'] if c['name'] == 'constant_target')
    observation = run.run_system('fixed_airflow', run.synthetic(case['input']), tmp_path)
    assert observation['status'] == 'completed'
    assert observation['result']['selected_candidate'] == 'ridge'
    assert observation['result']['validation_mae'] == 0
    scored = run.score_case(case, 'fixed_airflow', observation)
    assert scored['outcome'] == 'not_applicable'
    assert scored['pass'] is None
    assert scored['finding']


@pytest.mark.parametrize('system', run.SYSTEMS)
def test_isolation_signatures_use_real_runs(tmp_path, system):
    frame = run.synthetic({'seed': 21, 'rows': 100, 'target': 'rating_times_0.2'})
    first = run.run_system(system, frame, tmp_path / 'first')
    altered = run.run_system(system, frame, tmp_path / 'altered', alter_test=True)
    assert first['status'] == altered['status'] == 'completed'
    assert run.tuning_signature(first) == run.tuning_signature(altered)
    assert first['result']['test_mae'] != altered['result']['test_mae']


@pytest.mark.parametrize('system', run.SYSTEMS)
def test_budget_fixture_really_exercises_fit_cap(tmp_path, system):
    case = next(c for c in json.loads(Path('evaluation/cases.json').read_text())['cases']
                if c['name'] == 'model_attempt_budget')
    observed = run.run_system(system, run.synthetic(case['input']), tmp_path)
    assert run.score_case(case, system, observed)['pass']
    assert observed['model_fit_count'] == run.BUDGETS[system][0]
