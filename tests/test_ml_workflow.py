"""Behavioral checks for the healthcare graph, without a database dependency."""
import copy

import numpy as np
import pandas as pd
import pytest

from ml_workflow import nodes
from ml_workflow.graph import build_graph


@pytest.fixture
def records():
    rng = np.random.default_rng(12)
    return pd.DataFrame({
        'facility_id': [f'{i:06}' for i in range(100)],
        'measure_id': 'HAI_1_SIR', 'hospital_matched': True,
        'start_date': '2025-01-01', 'end_date': '2025-12-31',
        'overall_rating': rng.integers(1, 6, 100),
        'hospital_type': 'Acute Care Hospitals', 'state': ['NC', 'VA'] * 50,
        'score': rng.uniform(0.1, 2, 100),
    })


def prepared(records):
    return nodes.cohort_builder({'raw_data': records, 'seed': 42})


def test_splits_and_train_only_preprocessing(records):
    state = prepared(records)
    sets = [set(state[k].facility_id) for k in ['train', 'validation', 'test']]
    assert [len(s) for s in sets] == [60, 20, 20]
    assert len(set.union(*sets)) == 100
    assert all(not sets[i] & sets[j] for i, j in [(0, 1), (0, 2), (1, 2)])
    again = prepared(records.sample(frac=1, random_state=7))
    assert list(state['train'].facility_id) == list(again['train'].facility_id)
    state['validation']['state'] = 'UNSEEN'
    state['test']['overall_rating'] = 100000
    patch = nodes.score_model_trainer(state)
    pre = patch['model'].named_steps['preprocessing']
    assert pre.feature_names_in_.tolist() == nodes.FEATURES
    assert pre.named_transformers_['rating'].named_steps['scale'].mean_[0] == pytest.approx(state['train'].overall_rating.mean())
    assert 'UNSEEN' not in pre.named_transformers_['categories'].named_steps['encode'].categories_[1]


def review_state(history=None, baseline=1.0):
    return {'attempts': history or [], 'alpha': 1.0,
            'baseline_validation_mae': baseline, 'best_validation_mae': float('inf'),
            'path': []}


def test_adjuster_never_reads_test_and_avoids_repeats():
    state = review_state([{'alpha': 1.0, 'train_mae': 0.5, 'validation_mae': 1.0}])
    assert nodes.regularization_adjuster(state)['alpha'] == 10
    # Reversing direction would revisit 1; choose the unused opposite instead.
    state['attempts'].append({'alpha': 10.0, 'train_mae': 1.0, 'validation_mae': 1.0})
    state['test'] = object()  # unusable test data cannot affect tuning
    assert nodes.regularization_adjuster(state)['alpha'] == 100
    state['attempts'] = [{'alpha': 1.0, 'train_mae': 1.0, 'validation_mae': 1.0}]
    assert nodes.regularization_adjuster(state)['alpha'] == 0.1


class ConstantModel:
    def __init__(self, value):
        self.value = value

    def predict(self, frame):
        return np.full(len(frame), self.value)


def test_best_selected_by_validation_and_stopping(records):
    state = prepared(records)
    state['validation']['score'] = 1.0
    state['baseline_validation_mae'] = 1.0
    state['model'] = ConstantModel(0.5)
    state.update(nodes.error_reviewer(state))
    assert not state['tuning_finished']  # even a good first model must loop
    winner = state['best_model']
    state.update(alpha=10.0, model=ConstantModel(0.1), test=object())
    state.update(nodes.error_reviewer(state))
    assert state['best_model'] is winner
    assert state['best_validation_mae'] == 0.5
    assert state['tuning_finished']
    assert state['stop_reason'] == 'validation_improvement_at_least_5_percent'


@pytest.mark.parametrize('constant,expected_attempts', [(True, 3), (False, 2)])
def test_full_graph_loop_and_publisher_order(records, tmp_path, monkeypatch, constant, expected_attempts):
    if constant:
        records['score'] = 1.0  # perfect baseline: cannot improve, reach cap
    else:
        records['score'] = records.overall_rating * 0.2
    monkeypatch.setattr(nodes, 'hospital_data_loader', lambda s: {
        'raw_data': records, 'path': nodes.trace(s, 'hospital_data_loader')})
    original = nodes.results_publisher
    calls = []

    def publish(state):
        assert state['tuning_finished']
        assert len(state['attempts']) == expected_attempts
        calls.append(1)
        return original(state)

    monkeypatch.setattr(nodes, 'results_publisher', publish)
    final = build_graph().invoke({'output_dir': str(tmp_path)}, {'recursion_limit': 30})
    assert len(calls) == 1
    alphas = [a['alpha'] for a in final['attempts']]
    assert len(alphas) == len(set(alphas)) == expected_attempts
    assert final['path'].count('regularization_adjuster') == expected_attempts - 1
    assert final['path'][-1] == 'results_publisher'
    assert final['path'][-2] == 'error_reviewer'
    assert (tmp_path / 'actual_vs_predicted.png').is_file()
    if constant:
        assert final['result']['selected_candidate'] == 'median_baseline'
        assert final['stop_reason'] == 'maximum_three_attempts'
    with pytest.raises(ValueError, match='already performed'):
        original(final)


def test_test_outcomes_cannot_change_tuning(records):
    a = prepared(records)
    b = copy.deepcopy(a)
    b['test']['score'] = 1e9
    for state in [a, b]:
        for _ in range(3):
            state.update(nodes.score_model_trainer(state))
            state.update(nodes.error_reviewer(state))
            if state['tuning_finished']:
                break
            state.update(nodes.regularization_adjuster(state))
    assert a['attempts'] == b['attempts']
    assert a['best_alpha'] == b['best_alpha']
    assert a['stop_reason'] == b['stop_reason']


def test_early_test_evaluation_rejected():
    with pytest.raises(ValueError, match='before tuning finishes'):
        nodes.results_publisher({'tuning_finished': False})


@pytest.mark.parametrize('problem,match', [
    ('missing_column', 'Missing required columns'), ('empty', 'No healthcare records'),
    ('small', 'Insufficient cohort'), ('duplicate', 'Duplicate facility'),
    ('dates', 'reporting period'), ('missing_id', 'Missing facility'),
    ('no_target', 'reporting period'),
])
def test_invalid_input(records, problem, match):
    if problem == 'missing_column':
        records = records.drop(columns='score')
    elif problem == 'empty':
        records = records.iloc[:0]
    elif problem == 'small':
        records = records.iloc[:10]
    elif problem == 'duplicate':
        records = pd.concat([records, records.iloc[:1]])
    elif problem == 'dates':
        records['end_date'] = 'invalid'
    elif problem == 'missing_id':
        records.loc[0, 'facility_id'] = None
    else:
        records['measure_id'] = 'OTHER'
    with pytest.raises(ValueError, match=match):
        prepared(records)


def test_single_period_selected(records):
    old = records.copy()
    old['start_date'], old['end_date'] = '2024-01-01', '2024-12-31'
    final = prepared(pd.concat([old, records]))
    assert final['cohort_metadata']['rows'] == 100
    assert final['cohort_metadata']['end_date'] == '2025-12-31'
