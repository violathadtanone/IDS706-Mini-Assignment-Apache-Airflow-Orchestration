"""Reuse Part 2's cohort and preprocessing without changing its implementation.

Pickles are internal artifacts created by this application, not upload inputs.
The shared snapshot locks the selected records and split assignments; a later
database read must match it rather than silently changing the experiment.
"""
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from ml_workflow import nodes as part2


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_bytes(pickle.dumps(value))
    temporary.replace(path)
    return str(path)


def restore(path):
    return pickle.loads(Path(path).read_bytes())


def prepare(raw, snapshot_path=None):
    state = part2.cohort_builder({'raw_data': raw, 'seed': 42})
    digest = hashlib.sha256()
    ids = {}
    for name in ['train', 'validation', 'test']:
        frame = state[name]
        ids[name] = frame.facility_id.tolist()
        # JSON normalizes pandas dates/nulls while retaining scores and order.
        digest.update(frame.to_json(orient='split', date_format='iso', double_precision=15).encode())
    if any(set(ids[a]) & set(ids[b]) for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]):
        raise ValueError('Facility leakage between splits')
    state['cohort_metadata'] = {**state['cohort_metadata'], 'fingerprint': digest.hexdigest()}
    if snapshot_path:
        if Path(snapshot_path).exists():
            frozen = restore(snapshot_path)
            if frozen['cohort_metadata'] != state['cohort_metadata']:
                raise ValueError('Database cohort differs from the frozen shared snapshot')
            state = frozen
        else:
            dump(snapshot_path, state)
    # Part 3 owns its execution trace, not the reused helper's trace.
    state['path'] = []
    return state


def load_prepared(snapshot_path=None, database_url=None):
    initial = {'database_url': database_url} if database_url else {}
    return prepare(part2.hospital_data_loader(initial)['raw_data'], snapshot_path)


def train_ridge(state, alpha):
    # Exactly the same ColumnTransformer, imputation, encoding, scaling and
    # Ridge(lsqr) implementation as Part 2; no test data is used here.
    return part2.score_model_trainer({**state, 'alpha': alpha,
                                     'tuning_finished': False, 'attempts': []})['model']


def score(model, frame, median=None):
    predictions = np.full(len(frame), median) if model is None else model.predict(frame[part2.FEATURES])
    return float(mean_absolute_error(frame.score, predictions))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def test_predictions(model, state):
    frame = state['test']
    predictions = (np.full(len(frame), state['baseline_median']) if model is None
                   else model.predict(frame[part2.FEATURES]))
    baseline = np.full(len(frame), state['baseline_median'])
    return pd.DataFrame({'facility_id': frame.facility_id, 'actual_score': frame.score,
                         'predicted_score': predictions, 'baseline_prediction': baseline})
