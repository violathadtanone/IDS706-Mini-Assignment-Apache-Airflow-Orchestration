"""Daily hospital/infection ingestion through PostgreSQL-backed visualization."""
import os
from datetime import datetime, timedelta, timezone

from airflow.sdk import dag, task


@dag(
    dag_id="healthcare_pipeline",
    schedule=os.environ.get("HEALTHCARE_SCHEDULE", "0 0 * * *"),
    start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
    catchup=False,
    max_active_runs=1,
    max_active_tasks=4,
    default_args={"retries": 1, "retry_delay": timedelta(seconds=30)},
    tags=["ids706", "healthcare"],
)
def healthcare_pipeline():
    @task
    def ingest_dataset(kind: str):
        from healthcare_tasks import ingest
        from airflow.sdk import get_current_context
        return ingest(kind, get_current_context()["run_id"])

    @task
    def clean_dataset(path: str, kind: str):
        from healthcare_tasks import clean
        return clean(path, kind)

    @task
    def merge_datasets(hospitals: str, infections: str):
        from healthcare_tasks import merge
        return merge(hospitals, infections)

    @task
    def load_postgres(path: str):
        from healthcare_tasks import load_database
        return load_database(path)

    @task
    def generate_visualization(loaded: dict):
        from healthcare_tasks import analyze
        return analyze(loaded)

    hospitals = clean_dataset.override(task_id="clean_hospitals")(
        ingest_dataset.override(task_id="ingest_hospitals")("hospitals"), "hospitals"
    )
    infections = clean_dataset.override(task_id="clean_infections")(
        ingest_dataset.override(task_id="ingest_infections")("infections"), "infections"
    )
    generate_visualization(load_postgres(merge_datasets(hospitals, infections)))


healthcare_pipeline()
