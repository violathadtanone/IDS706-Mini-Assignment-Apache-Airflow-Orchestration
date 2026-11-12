"""Our healthcare cohort, regression agents, and evidence-based adjustment policy."""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FEATURES = ['overall_rating', 'hospital_type', 'state']
REQUIRED = ['facility_id', 'measure_id', 'score', 'hospital_matched',
            'start_date', 'end_date', *FEATURES]


def trace(state, name):
    return [*state.get('path', []), name]


def validate_input(frame):
    missing = set(REQUIRED) - set(frame.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')
    if frame.empty:
        raise ValueError('No healthcare records available')


def hospital_data_loader(state):
    url = state.get('database_url') or os.environ.get('HEALTHCARE_DATABASE_URL')
    if not url:
        raise ValueError('HEALTHCARE_DATABASE_URL is required')
    # One query reads a consistent snapshot; do not change the Part 1 database.
    with psycopg2.connect(url) as conn, conn.cursor() as cursor:
        cursor.execute('''SELECT facility_id, measure_id, score, hospital_matched,
            start_date, end_date, overall_rating, hospital_type, state
            FROM hospital_infections WHERE measure_id = %s''', ('HAI_1_SIR',))
        frame = pd.DataFrame(cursor.fetchall(), columns=[c.name for c in cursor.description])
    validate_input(frame)
    return {'raw_data': frame, 'path': trace(state, 'hospital_data_loader')}


def cohort_builder(state):
    frame = state['raw_data'].copy()
    validate_input(frame)
    frame = frame.loc[frame.measure_id.eq('HAI_1_SIR') & frame.hospital_matched.eq(True)].copy()
    for column in ['start_date', 'end_date']:
        frame[column] = pd.to_datetime(frame[column], errors='coerce')
    if frame.empty or frame[['start_date', 'end_date']].isna().any().any():
        raise ValueError('No valid HAI_1_SIR reporting period available')
    if (frame.start_date > frame.end_date).any():
        raise ValueError('Reporting period starts after it ends')
    # Latest period is selected using dates only, never outcome magnitude.
    period = frame.sort_values(['end_date', 'start_date']).iloc[-1]
    frame = frame.loc[frame.start_date.eq(period.start_date) & frame.end_date.eq(period.end_date)].copy()
    for column in ['score', 'overall_rating']:
        frame[column] = pd.to_numeric(frame[column], errors='coerce')
    frame = frame.loc[np.isfinite(frame.score) & frame.overall_rating.isin([1, 2, 3, 4, 5])].copy()
    if frame.facility_id.isna().any() or frame.facility_id.astype(str).str.strip().eq('').any():
        raise ValueError('Missing facility ID in cohort')
    frame['facility_id'] = frame.facility_id.astype(str).str.strip()
    if frame.facility_id.duplicated().any():
        raise ValueError('Duplicate facility records in selected reporting period')
    if len(frame) < 50:
        raise ValueError(f'Insufficient cohort: {len(frame)} facilities; at least 50 required')
    for column in ['hospital_type', 'state']:
        frame[column] = frame[column].fillna('').astype(str).str.strip().replace('', np.nan)
    frame = frame.sort_values('facility_id').reset_index(drop=True)
    seed = state.get('seed', 42)
    train, remaining = train_test_split(frame, test_size=0.4, random_state=seed)
    validation, test = train_test_split(remaining, test_size=0.5, random_state=seed)
    median = float(train.score.median())
    baseline = float(mean_absolute_error(validation.score, np.full(len(validation), median)))
    return {
        'train': train, 'validation': validation, 'test': test,
        'baseline_median': median, 'baseline_validation_mae': baseline,
        'alpha': 1.0, 'attempts': [], 'best_validation_mae': float('inf'),
        'tuning_finished': False,
        'cohort_metadata': {'rows': len(frame), 'facilities': frame.facility_id.nunique(),
                            'start_date': str(period.start_date.date()),
                            'end_date': str(period.end_date.date()),
                            'measure_id': 'HAI_1_SIR', 'seed': seed,
                            'split_sizes': {k: len(v) for k, v in [('train', train), ('validation', validation), ('test', test)]}},
        'path': trace(state, 'cohort_builder'),
    }


def score_model_trainer(state):
    if state['tuning_finished'] or len(state['attempts']) >= 3:
        raise ValueError('Training requested after tuning finished')
    if state['alpha'] in [a['alpha'] for a in state['attempts']]:
        raise ValueError('Alpha already attempted')
    preprocessing = ColumnTransformer([
        ('rating', Pipeline([('impute', SimpleImputer(strategy='median')),
                             ('scale', StandardScaler())]), ['overall_rating']),
        ('categories', Pipeline([('impute', SimpleImputer(strategy='constant', fill_value='Unknown')),
                                 ('encode', OneHotEncoder(handle_unknown='ignore'))]),
         ['hospital_type', 'state']),
    ])
    model = Pipeline([('preprocessing', preprocessing),
                      ('ridge', Ridge(alpha=state['alpha'], solver='lsqr'))])
    model.fit(state['train'][FEATURES], state['train'].score)
    return {'model': model, 'path': trace(state, 'score_model_trainer')}


def error_reviewer(state):
    model = state['model']
    attempt = {'alpha': state['alpha']}
    for name in ['train', 'validation']:
        frame = state[name]
        attempt[f'{name}_mae'] = float(mean_absolute_error(frame.score, model.predict(frame[FEATURES])))
    attempts = [*state['attempts'], attempt]
    patch = {'attempts': attempts, 'path': trace(state, 'error_reviewer')}
    best = state['best_validation_mae']
    if attempt['validation_mae'] < best:
        best = attempt['validation_mae']
        patch.update(best_model=model, best_alpha=state['alpha'], best_validation_mae=best)
    baseline = state['baseline_validation_mae']
    # Require a positive improvement even if baseline MAE is zero.
    sufficient = best < baseline and best <= baseline * 0.95
    reason = None
    if len(attempts) >= 2 and sufficient:
        reason = 'validation_improvement_at_least_5_percent'
    elif len(attempts) >= 3:
        reason = 'maximum_three_attempts'
    patch.update(tuning_finished=reason is not None, stop_reason=reason or '')
    return patch


def regularization_adjuster(state):
    latest = state['attempts'][-1]
    stronger = latest['validation_mae'] - latest['train_mae'] > 0.1 * state['baseline_validation_mae']
    factor = 10 if stronger else 0.1
    used = {a['alpha'] for a in state['attempts']}
    for candidate in [latest['alpha'] * factor, latest['alpha'] / factor]:
        candidate = float(f'{candidate:.10g}')
        if candidate not in used:
            return {'alpha': candidate, 'path': trace(state, 'regularization_adjuster')}
    raise ValueError('No unused adjacent alpha available')


def results_publisher(state):
    if not state['tuning_finished']:
        raise ValueError('Cannot evaluate test data before tuning finishes')
    if state.get('result'):
        raise ValueError('Final test evaluation already performed')
    baseline_wins = state['best_validation_mae'] >= state['baseline_validation_mae']
    test = state['test']
    baseline_predictions = np.full(len(test), state['baseline_median'])
    predictions = baseline_predictions if baseline_wins else state['best_model'].predict(test[FEATURES])
    result = {
        **state['cohort_metadata'], 'attempts': state['attempts'],
        'baseline_validation_mae': state['baseline_validation_mae'],
        'best_ridge_alpha': state['best_alpha'], 'best_ridge_validation_mae': state['best_validation_mae'],
        'selected_candidate': 'median_baseline' if baseline_wins else 'ridge',
        'selected_validation_mae': state['baseline_validation_mae'] if baseline_wins else state['best_validation_mae'],
        'test_mae': float(mean_absolute_error(test.score, predictions)),
        'baseline_test_mae': float(mean_absolute_error(test.score, baseline_predictions)),
        'stop_reason': state['stop_reason'], 'path': trace(state, 'results_publisher'),
        'interpretation': 'Retrospective association; ratings may reflect related safety information. Not clinical forecasting.',
    }
    output = Path(state['output_dir'])
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(state['attempts']).to_csv(output / 'attempts.csv', index=False)
    pd.DataFrame({'facility_id': test.facility_id, 'actual_score': test.score,
                  'predicted_score': predictions, 'baseline_prediction': baseline_predictions}).to_csv(
        output / 'test_predictions.csv', index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(test.score, predictions, alpha=0.5, s=18)
    low = min(test.score.min(), predictions.min())
    high = max(test.score.max(), predictions.max())
    ax.plot([low, high], [low, high], '--', color='gray', label='Perfect agreement')
    ax.set(xlabel='Actual HAI_1_SIR score', ylabel='Predicted score',
           title=f'Held-out facilities: {result["selected_candidate"]}')
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / 'actual_vs_predicted.png', dpi=150)
    plt.close(fig)
    (output / 'metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    return {'result': result, 'path': result['path']}
