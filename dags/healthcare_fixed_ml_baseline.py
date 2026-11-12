"""Part 3 comparison baseline: four predefined tasks, one Ridge configuration."""
from datetime import datetime, timedelta, timezone
import hashlib

from airflow.sdk import dag, task


@dag(dag_id='healthcare_fixed_ml_baseline', schedule=None,
     start_date=datetime(2026, 1, 1, tzinfo=timezone.utc), catchup=False,
     max_active_runs=1, tags=['ids706', 'part3', 'fixed-ml-baseline'],
     default_args={'retries': 1, 'retry_delay': timedelta(seconds=5)})
def healthcare_fixed_ml_baseline():
    @task
    def load_shared_cohort():
        from airflow.sdk import get_current_context
        from adaptive_workflow import fixed
        run = hashlib.sha256(get_current_context()['run_id'].encode()).hexdigest()[:16]
        return fixed.load_shared_cohort(
            f'/opt/airflow/output/part3/fixed/{run}',
            '/opt/airflow/output/part3/shared/cohort.pkl')

    @task
    def train_ridge(path):
        from adaptive_workflow import fixed
        return fixed.train_ridge(path)

    @task
    def evaluate(path):
        from adaptive_workflow import fixed
        return fixed.evaluate(path)

    @task
    def save_results(path):
        from adaptive_workflow import fixed
        return fixed.save_results(path)

    save_results(evaluate(train_ridge(load_shared_cohort())))


healthcare_fixed_ml_baseline()
