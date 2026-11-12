"""Run independently: python -m adaptive_workflow.graph [--inject-db-failures 1]."""
import argparse
import json
from datetime import datetime, timezone
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from . import agents


class AdaptiveState(TypedDict, total=False):
    database_url: str
    snapshot_path: str
    output_dir: str
    inject_db_failures: int
    path: list
    routing_history: list
    raw_data: object
    train: object
    validation: object
    test: object
    cohort_metadata: dict
    baseline_median: float
    baseline_validation_mae: float
    alpha: float
    attempts: list
    used_alphas: list
    model_fit_count: int
    recovery_retry_count: int
    database_read_count: int
    candidate_kind: str
    candidate_model: object
    candidate_history: list
    best_model: object
    best_candidate: str
    best_alpha: float | None
    best_validation_mae: float
    median_evaluated: bool
    decisions_finished: bool
    next_agent: str
    error: str
    recoverable: bool
    failed_operation: str
    stop_reason: str
    result: dict


def build_graph():
    graph = StateGraph(AdaptiveState)
    for name in ['cohort_steward', 'workflow_dispatcher', 'median_specialist',
                 'ridge_specialist', 'score_auditor', 'recovery_coordinator', 'results_reporter']:
        graph.add_node(name, getattr(agents, name))
    graph.add_edge(START, 'cohort_steward')
    graph.add_edge('cohort_steward', 'workflow_dispatcher')
    graph.add_conditional_edges('workflow_dispatcher', lambda s: s['next_agent'],
        {n: n for n in ['median_specialist', 'ridge_specialist', 'recovery_coordinator', 'results_reporter']})
    graph.add_edge('median_specialist', 'score_auditor')
    graph.add_conditional_edges('ridge_specialist',
        lambda s: 'workflow_dispatcher' if s.get('error') else 'score_auditor',
        {n: n for n in ['workflow_dispatcher', 'score_auditor']})
    graph.add_edge('score_auditor', 'workflow_dispatcher')
    graph.add_conditional_edges('recovery_coordinator', lambda s: s['next_agent'],
        {n: n for n in ['cohort_steward', 'results_reporter']})
    graph.add_edge('results_reporter', END)
    return graph.compile()


def run_workflow(output_dir=None, snapshot_path='output/part3/shared/cohort.pkl',
                 database_url=None, inject_db_failures=0):
    initial = {'output_dir': str(output_dir or ('output/part3/adaptive/' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))),
               'snapshot_path': str(snapshot_path), 'inject_db_failures': inject_db_failures,
               'path': [], 'routing_history': [], 'model_fit_count': 0,
               'recovery_retry_count': 0, 'database_read_count': 0}
    if database_url:
        initial['database_url'] = database_url
    return build_graph().invoke(initial, {'recursion_limit': 50})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir')
    parser.add_argument('--snapshot-path', default='output/part3/shared/cohort.pkl')
    parser.add_argument('--inject-db-failures', type=int, default=0)
    args = parser.parse_args()
    final = run_workflow(**vars(args))['result']
    print(json.dumps(final, indent=2))
    raise SystemExit(0 if final['status'] == 'completed' else 1)
