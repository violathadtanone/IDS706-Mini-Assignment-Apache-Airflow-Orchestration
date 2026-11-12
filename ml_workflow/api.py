"""One local run at a time; polling retains every real node boundary event."""
import json
from pathlib import Path
from threading import Lock, Thread
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from .graph import build_graph

app = FastAPI(title='Healthcare ML workflow')
lock = Lock()
current = None
LABELS = {
    'hospital_data_loader': 'Hospital Data Loader',
    'cohort_builder': 'Cohort Builder',
    'score_model_trainer': 'Score Model Trainer',
    'error_reviewer': 'Error Reviewer',
    'regularization_adjuster': 'Regularization Adjuster',
    'results_publisher': 'Results Publisher',
}


@app.get('/api/graph')
def graph_metadata():
    graph = build_graph().get_graph()
    return {'nodes': [{'id': name, 'label': label} for name, label in LABELS.items()],
            'edges': [{'source': e.source, 'target': e.target} for e in graph.edges
                      if e.source in LABELS and e.target in LABELS]}


def execute(run):
    output = Path('output/ml') / run['id']

    def observe(status, node, state):
        from datetime import datetime, timezone
        with lock:
            run['events'].append({'sequence': len(run['events']) + 1,
                                  'node': node, 'status': status,
                                  'timestamp': datetime.now(timezone.utc).isoformat(),
                                  'attempt': len(state.get('attempts', [])) + (1 if node == 'score_model_trainer' else 0),
                                  'alpha': state.get('alpha')})
            run['nodes'][node] = status
            run['active_node'] = node if status == 'running' else None
            run['attempts'] = state.get('attempts', [])
            run['cohort'] = state.get('cohort_metadata')
            run['baseline_validation_mae'] = state.get('baseline_validation_mae')
            run['alpha'] = state.get('alpha')
            run['attempt'] = len(state.get('attempts', [])) + (1 if node == 'score_model_trainer' else 0)

    try:
        final = build_graph(observer=observe).invoke(
            {'output_dir': str(output), 'seed': 42, 'path': []}, {'recursion_limit': 30})
        with lock:
            run.update(status='completed', result=final['result'], active_node=None)
    except Exception as exc:
        with lock:
            run.update(status='failed', error=str(exc), active_node=None)
    finally:
        output.mkdir(parents=True, exist_ok=True)
        with lock:
            (output / 'execution.json').write_text(json.dumps(run, indent=2))


@app.post('/api/runs', status_code=202)
def start_run():
    global current
    with lock:
        if current and current['status'] == 'running':
            raise HTTPException(409, 'A workflow run is already running')
        current = {'id': uuid4().hex, 'status': 'running',
                   'nodes': {name: 'waiting' for name in LABELS}, 'events': [],
                   'attempts': [], 'active_node': None, 'result': None}
        run = current
    Thread(target=execute, args=(run,), daemon=True).start()
    return {'id': run['id']}


@app.get('/api/runs/{run_id}')
def run_status(run_id: str):
    with lock:
        if not current or current['id'] != run_id:
            raise HTTPException(404, 'Run not found in this server session')
        return json.loads(json.dumps(current))


static = Path(__file__).with_name('static')
if static.exists():
    app.mount('/', StaticFiles(directory=static, html=True), name='frontend')
