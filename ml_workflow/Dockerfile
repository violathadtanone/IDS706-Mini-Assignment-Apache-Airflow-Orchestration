FROM node:22-alpine AS frontend
WORKDIR /ui
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM ids706-healthcare-airflow:3.1.0
WORKDIR /workspace
COPY ml_workflow/requirements.txt /tmp/ml-requirements.txt
RUN python -m venv --system-site-packages /home/airflow/ml-env && /home/airflow/ml-env/bin/pip install --no-cache-dir -r /tmp/ml-requirements.txt
COPY ml_workflow/ ./ml_workflow/
COPY --from=frontend /ui/dist/ ./ml_workflow/static/
ENV MPLCONFIGDIR=/tmp/matplotlib PYTHONDONTWRITEBYTECODE=1
ENTRYPOINT ["/home/airflow/ml-env/bin/python", "-m", "uvicorn"]
CMD ["ml_workflow.api:app", "--host", "0.0.0.0", "--port", "8001"]
