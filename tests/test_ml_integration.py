"""Observation hook and API expose actual agent execution, not replayed statuses."""
import time
from threading import Event

import pytest
from fastapi.testclient import TestClient

from ml_workflow import api, nodes
from ml_workflow.graph import build_graph


def test_observer_wraps_real_node_and_reports_failure(monkeypatch):
    events = []
    def fail(state):
        assert events == [('running', 'hospital_data_loader')]
        raise ValueError('Invalid fixture input')
    monkeypatch.setattr(nodes, 'hospital_data_loader', fail)
    with pytest.raises(ValueError, match='Invalid fixture input'):
        build_graph(observer=lambda status, node, state: events.append((status, node))).invoke({})
    assert events == [('running', 'hospital_data_loader'), ('failed', 'hospital_data_loader')]


def test_api_rejects_concurrent_run_and_retains_failure(monkeypatch, tmp_path):
    entered, release = Event(), Event()
    def loader(state):
        entered.set()
        assert release.wait(timeout=10)
        raise ValueError('No eligible facilities')
    monkeypatch.setattr(nodes, 'hospital_data_loader', loader)
    monkeypatch.setattr(api, 'current', None)
    monkeypatch.chdir(tmp_path)
    with TestClient(api.app) as client:
        metadata = client.get('/api/graph').json()
        assert len(metadata['nodes']) == 6
        assert {'source': 'regularization_adjuster', 'target': 'score_model_trainer'} in metadata['edges']
        created = client.post('/api/runs')
        assert created.status_code == 202
        run_id = created.json()['id']
        try:
            assert entered.wait(timeout=10)
            running = client.get(f'/api/runs/{run_id}').json()
            assert running['active_node'] == 'hospital_data_loader'
            assert running['nodes']['hospital_data_loader'] == 'running'
            assert client.post('/api/runs').status_code == 409
            assert client.get('/api/runs/missing').status_code == 404
        finally:
            release.set()
        for _ in range(100):
            final = client.get(f'/api/runs/{run_id}').json()
            if final['status'] != 'running':
                break
            time.sleep(0.01)
        assert final['status'] == 'failed'
        assert final['error'] == 'No eligible facilities'
        assert final['nodes']['hospital_data_loader'] == 'failed'
        assert [e['status'] for e in final['events']] == ['running', 'failed']
