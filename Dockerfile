FROM apache/airflow:3.1.0-python3.12
RUN pip install --no-cache-dir "apache-airflow==3.1.0" "pandas==2.2.3" "matplotlib==3.10.1" "psycopg2-binary==2.9.10"
