"""Four fixed Airflow stages. Ridge is always the evaluated predictor."""
from pathlib import Path
from sklearn.metrics import mean_absolute_error

from . import shared


def load_shared_cohort(output_dir, snapshot_path, database_url=None):
    state = shared.load_prepared(snapshot_path, database_url)
    return shared.dump(Path(output_dir) / 'cohort.pkl', state)


def train_ridge(cohort_path):
    state = shared.restore(cohort_path)
    state['fixed_model'] = shared.train_ridge(state, 1.0)
    return shared.dump(Path(cohort_path).with_name('trained.pkl'), state)


def evaluate(trained_path):
    state = shared.restore(trained_path)
    model = state['fixed_model']
    predictions = shared.test_predictions(model, state)
    metrics = {
        **state['cohort_metadata'], 'system': 'fixed_airflow',
        'alpha': 1.0, 'model_fit_count': 1, 'selected_candidate': 'ridge',
        'validation_mae': shared.score(model, state['validation']),
        'test_mae': float(mean_absolute_error(predictions.actual_score, predictions.predicted_score)),
        'baseline_validation_mae': state['baseline_validation_mae'],
        'baseline_test_mae': float(mean_absolute_error(predictions.actual_score, predictions.baseline_prediction)),
        'path': ['load_shared_cohort', 'train_ridge', 'evaluate', 'save_results'],
        'stop_reason': 'fixed_single_fit_complete',
    }
    return shared.dump(Path(trained_path).with_name('evaluated.pkl'),
                       {'metrics': metrics, 'predictions': predictions})


def save_results(evaluated_path):
    result = shared.restore(evaluated_path)
    output = Path(evaluated_path).parent
    result['predictions'].to_csv(output / 'test_predictions.csv', index=False)
    shared.save_json(output / 'metrics.json', result['metrics'])
    return result['metrics']
