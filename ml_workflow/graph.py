"""Six role-based agents with an explicit regularization back edge.

Run: HEALTHCARE_DATABASE_URL=... python -m ml_workflow.graph
"""
import argparse
import json
from datetime import datetime, timezone
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from . import nodes


class WorkflowState(TypedDict, total=False):
    database_url: str
    output_dir: str
    seed: int
    raw_data: object
    cohort_metadata: dict
    train: object
    validation: object
    test: object
    baseline_median: float
    baseline_validation_mae: float
    alpha: float
    model: object
    attempts: list
    best_model: object
    best_alpha: float
    best_validation_mae: float
    tuning_finished: bool
    stop_reason: str
    path: list
    result: dict


def after_review(state):
    return 'results_publisher' if state['tuning_finished'] else 'regularization_adjuster'


def build_graph(observer=None):
    """Optionally observe real node boundaries without changing agent decisions."""
    graph = StateGraph(WorkflowState)
    agents = {
        'hospital_data_loader': nodes.hospital_data_loader,
        'cohort_builder': nodes.cohort_builder,
        'score_model_trainer': nodes.score_model_trainer,
        'error_reviewer': nodes.error_reviewer,
        'regularization_adjuster': nodes.regularization_adjuster,
        'results_publisher': nodes.results_publisher,
    }
    def observed(name, agent):
        def execute(state):
            observer('running', name, state)
            try:
                patch = agent(state)
            except Exception:
                observer('failed', name, state)
                raise
            observer('completed', name, {**state, **patch})
            return patch
        return execute

    for name, agent in agents.items():
        graph.add_node(name, observed(name, agent) if observer else agent)
    graph.add_edge(START, 'hospital_data_loader')
    graph.add_edge('hospital_data_loader', 'cohort_builder')
    graph.add_edge('cohort_builder', 'score_model_trainer')
    graph.add_edge('score_model_trainer', 'error_reviewer')
    graph.add_conditional_edges('error_reviewer', after_review, {
        name: name for name in ('regularization_adjuster', 'results_publisher')})
    graph.add_edge('regularization_adjuster', 'score_model_trainer')
    graph.add_edge('results_publisher', END)
    return graph.compile()


def run_workflow(database_url=None, output_dir=None):
    output_dir = output_dir or ('output/ml/' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    initial = {'output_dir': str(output_dir), 'seed': 42, 'path': []}
    if database_url:
        initial['database_url'] = database_url
    return build_graph().invoke(initial, config={'recursion_limit': 30})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir')
    args = parser.parse_args()
    print(json.dumps(run_workflow(output_dir=args.output_dir)['result'], indent=2))
