"""Seven deterministic agents with evidence-based selection and typed recovery."""
from pathlib import Path

import psycopg2
from sklearn.metrics import mean_absolute_error

from ml_workflow import nodes as part2
from . import shared

MAX_FITS = 3
MAX_RETRIES = 2
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)


def path(state, name):
    return [*state.get('path', []), name]


def transient_database_error(exc):
    if not isinstance(exc, (psycopg2.OperationalError, psycopg2.InterfaceError)):
        return False
    if any(marker in str(exc).lower() for marker in [
            'authentication failed', 'no password supplied', 'permission denied',
            'does not exist', 'invalid dsn']):
        return False
    # Missing SQLSTATE covers connection loss/refusal before PostgreSQL replies.
    # Authentication/schema errors must not be retried as connection failures.
    return exc.pgcode is None or exc.pgcode.startswith('08') or exc.pgcode in {'40001', '40P01', '57P01', '57P02', '57P03'}


def cohort_steward(state):
    reads = state.get('database_read_count', 0) + 1
    patch = {'path': path(state, 'cohort_steward'), 'database_read_count': reads}
    try:
        # Explicit demo/test hook. Every non-injected read uses the real loader.
        if reads <= state.get('inject_db_failures', 0):
            raise psycopg2.OperationalError('Injected transient PostgreSQL connection failure')
        prepared = shared.load_prepared(state.get('snapshot_path'), state.get('database_url'))
    except Exception as exc:
        return {**patch, 'error': str(exc), 'recoverable': transient_database_error(exc),
                'failed_operation': 'cohort_steward'}
    return {**prepared, **patch, 'error': '', 'recoverable': False,
            'failed_operation': '', 'attempts': [], 'used_alphas': [],
            'model_fit_count': 0, 'candidate_history': [], 'best_model': None,
            'best_candidate': '', 'best_validation_mae': float('inf'),
            'median_evaluated': False, 'decisions_finished': False}


def workflow_dispatcher(state):
    next_agent, reason, alpha = None, '', state.get('alpha', 1.0)
    if state.get('error'):
        next_agent = 'recovery_coordinator' if state.get('recoverable') else 'results_reporter'
        reason = '' if state.get('recoverable') else 'permanent_execution_failure'
    elif not state['median_evaluated']:
        next_agent = 'median_specialist'
    else:
        attempts = state['attempts']
        baseline = state['baseline_validation_mae']
        if state['model_fit_count'] >= MAX_FITS:
            reason = 'model_fit_budget_exhausted'
        elif len(attempts) >= 2:
            best_ridge = min(a['validation_mae'] for a in attempts)
            if best_ridge < baseline and best_ridge <= baseline * 0.95:
                reason = 'validation_improvement_at_least_5_percent'
            elif best_ridge >= baseline and attempts[-2]['validation_mae'] - attempts[-1]['validation_mae'] <= 0.01 * baseline:
                reason = 'baseline_preferred_after_stalled_ridge_attempts'
        if not reason:
            unused = set(ALPHAS) - set(state['used_alphas'])
            if not unused:
                reason = 'unused_alpha_configurations_exhausted'
            else:
                if not attempts:
                    alpha = 1.0
                else:
                    last = attempts[-1]
                    factor = 10 if last['validation_mae'] - last['train_mae'] > 0.1 * baseline else 0.1
                    choices = [float(f'{last["alpha"] * factor:.10g}'),
                               float(f'{last["alpha"] / factor:.10g}')]
                    alpha = next((a for a in choices if a in unused), min(unused))
                next_agent = 'ridge_specialist'
    if reason:
        next_agent = 'results_reporter'
    decision = {'next_agent': next_agent, 'reason': reason or 'specialist_required',
                'model_fit_count': state.get('model_fit_count', 0),
                'recovery_retry_count': state.get('recovery_retry_count', 0),
                'alpha': alpha if next_agent == 'ridge_specialist' else None}
    return {'path': path(state, 'workflow_dispatcher'), 'next_agent': next_agent,
            'alpha': alpha, 'decisions_finished': next_agent == 'results_reporter',
            'stop_reason': reason, 'routing_history': [*state.get('routing_history', []), decision]}


def median_specialist(state):
    return {'candidate_kind': 'median_baseline', 'candidate_model': None,
            'path': path(state, 'median_specialist')}


def ridge_specialist(state):
    if state.get('next_agent') != 'ridge_specialist' or state.get('decisions_finished'):
        raise ValueError('Ridge specialist was not selected by the dispatcher')
    if state['model_fit_count'] >= MAX_FITS:
        raise ValueError('Model fit budget exhausted')
    alpha = state['alpha']
    if alpha not in ALPHAS or alpha in state['used_alphas']:
        raise ValueError('Ridge alpha is disallowed or already used')
    patch = {'path': path(state, 'ridge_specialist'), 'candidate_kind': 'ridge',
             'model_fit_count': state['model_fit_count'] + 1,
             'used_alphas': [*state['used_alphas'], alpha]}
    try:
        model = shared.train_ridge(state, alpha)
    except Exception as exc:
        # An attempted fit spends budget even if it raises. DB recovery does
        # not silently retry arbitrary model failures.
        return {**patch, 'error': str(exc), 'recoverable': False,
                'failed_operation': 'ridge_specialist'}
    return {**patch, 'candidate_model': model}


def score_auditor(state):
    model = state['candidate_model']
    row = {'candidate': state['candidate_kind'],
           'train_mae': shared.score(model, state['train'], state['baseline_median']),
           'validation_mae': shared.score(model, state['validation'], state['baseline_median'])}
    patch = {'path': path(state, 'score_auditor')}
    if row['candidate'] == 'ridge':
        row['alpha'] = state['alpha']
        patch['attempts'] = [*state['attempts'], {k: row[k] for k in ['alpha', 'train_mae', 'validation_mae']}]
    else:
        patch['median_evaluated'] = True
        patch['baseline_validation_mae'] = row['validation_mae']
    if row['validation_mae'] < state['best_validation_mae']:
        patch.update(best_candidate=row['candidate'], best_model=model,
                     best_validation_mae=row['validation_mae'],
                     best_alpha=row.get('alpha'))
    patch['candidate_history'] = [*state['candidate_history'], row]
    return patch


def recovery_coordinator(state):
    patch = {'path': path(state, 'recovery_coordinator')}
    if not state.get('recoverable') or state.get('failed_operation') != 'cohort_steward':
        raise ValueError('Recovery requested for an ineligible operation')
    retries = state.get('recovery_retry_count', 0)
    if retries >= MAX_RETRIES:
        return {**patch, 'next_agent': 'results_reporter', 'decisions_finished': True,
                'stop_reason': 'recovery_retry_budget_exhausted'}
    return {**patch, 'recovery_retry_count': retries + 1, 'next_agent': 'cohort_steward'}


def results_reporter(state):
    if not state.get('decisions_finished'):
        raise ValueError('Results requested before adaptive decisions finished')
    if state.get('result'):
        raise ValueError('Final test evaluation already performed')
    output = Path(state['output_dir'])
    output.mkdir(parents=True, exist_ok=True)
    result = {**state.get('cohort_metadata', {}), 'system': 'adaptive_multi_agent',
              'path': path(state, 'results_reporter'), 'routing_history': state.get('routing_history', []),
              'model_fit_count': state.get('model_fit_count', 0),
              'recovery_retry_count': state.get('recovery_retry_count', 0),
              'database_read_count': state.get('database_read_count', 0),
              'attempts': state.get('attempts', []), 'used_alphas': state.get('used_alphas', []),
              'candidate_history': state.get('candidate_history', []),
              'stop_reason': state['stop_reason']}
    if state.get('error'):
        result.update(status='failed', error=state['error'])
    else:
        predictions = shared.test_predictions(state['best_model'], state)
        result.update(status='completed', selected_candidate=state['best_candidate'],
                      selected_alpha=state.get('best_alpha'),
                      validation_mae=state['best_validation_mae'],
                      baseline_validation_mae=state['baseline_validation_mae'],
                      test_mae=float(mean_absolute_error(predictions.actual_score, predictions.predicted_score)),
                      baseline_test_mae=float(mean_absolute_error(predictions.actual_score, predictions.baseline_prediction)))
        predictions.to_csv(output / 'test_predictions.csv', index=False)
    shared.save_json(output / 'metrics.json', result)
    shared.save_json(output / 'routing_history.json', {'path': result['path'], 'decisions': result['routing_history']})
    return {'result': result, 'path': result['path']}
