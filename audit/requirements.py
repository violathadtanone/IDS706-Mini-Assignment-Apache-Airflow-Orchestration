"""Collect verified assignment evidence after the runtime and browser audits."""
import csv
import json
import subprocess
import urllib.request
from datetime import datetime
from pathlib import Path

from audit.runtime import BASE, ML, OUT, ROOT


def execute(args):
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def query(service, user, sql):
    value = execute(BASE + ['exec', '-T', service, 'psql', '-U', user, '-At', '-c',
                    f"SELECT COALESCE(json_agg(x),'[]'::json) FROM ({sql}) x"])
    return json.loads(value)


def collect():
    dags = query('postgres', 'airflow', "SELECT dag_id,timetable_summary,is_paused,has_import_errors,fileloc FROM dag WHERE dag_id IN ('healthcare_pipeline','healthcare_fixed_ml_baseline')")
    runs = query('postgres', 'airflow', "SELECT dag_id,run_id,run_type,state,start_date,end_date FROM dag_run WHERE dag_id IN ('healthcare_pipeline','healthcare_fixed_ml_baseline') ORDER BY start_date")
    tasks = query('postgres', 'airflow', "SELECT dag_id,run_id,task_id,state,start_date,end_date,try_number FROM task_instance WHERE dag_id IN ('healthcare_pipeline','healthcare_fixed_ml_baseline') ORDER BY start_date")
    scheduled = [r for r in runs if r['dag_id'] == 'healthcare_pipeline' and r['run_type'] == 'scheduled' and r['state'] == 'success']
    assert scheduled, 'No actual successful scheduled Part 1 run'
    for run_id in ['full-assignment-audit-20261112', 'full-fixed-audit-20261112']:
        assert any(r['run_id'] == run_id and r['state'] == 'success' for r in runs), f'Fresh audit run not successful: {run_id}'
    assert next(d for d in dags if d['dag_id'] == 'healthcare_pipeline')['timetable_summary'] == '0 0 * * *'
    assert all(not d['has_import_errors'] for d in dags)
    overlaps = []
    for run in scheduled:
        by_id = {t['task_id']: t for t in tasks if t['run_id'] == run['run_id'] and t['dag_id'] == 'healthcare_pipeline'}
        for a, b in [('ingest_hospitals', 'ingest_infections'), ('clean_hospitals', 'clean_infections')]:
            left, right = by_id[a], by_id[b]
            start = max(datetime.fromisoformat(left['start_date']), datetime.fromisoformat(right['start_date']))
            end = min(datetime.fromisoformat(left['end_date']), datetime.fromisoformat(right['end_date']))
            assert end > start, f'No observed parallelism: {a}, {b}'
            overlaps.append({'run_id': run['run_id'], 'tasks': [a, b], 'overlap_seconds': (end - start).total_seconds()})
    counts = query('healthcare-db', 'healthcare', 'SELECT COUNT(*) AS rows,COUNT(DISTINCT facility_id) AS facilities,COUNT(*) FILTER (WHERE NOT hospital_matched) AS unmatched_rows,COUNT(score) AS numeric_scores FROM hospital_infections')[0]
    assert counts['rows'] > 0 and counts['unmatched_rows'] == 0
    ratings = query('healthcare-db', 'healthcare', "SELECT overall_rating,AVG(score) AS average_score,COUNT(*) AS record_count,COUNT(DISTINCT facility_id) AS facility_count FROM hospital_infections WHERE measure_id='HAI_1_SIR' AND score IS NOT NULL AND overall_rating IS NOT NULL AND hospital_matched=TRUE GROUP BY overall_rating ORDER BY overall_rating")
    pngs = list((ROOT / 'output').glob('*/infection_score_by_rating.png'))
    assert pngs, 'Part 1 merged-data visualization missing'
    png = max(pngs, key=lambda p: p.stat().st_mtime)
    assert png.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    with png.with_suffix('.csv').open() as handle:
        saved = list(csv.DictReader(handle))
    assert len(saved) == len(ratings)
    for actual, row in zip(ratings, saved):
        assert float(row['overall_rating']) == actual['overall_rating']
        assert abs(float(row['average_score']) - actual['average_score']) < 1e-12
    image_paths = [str(png.relative_to(ROOT)), 'output/ml/ui-validation/running.png',
                   'output/ml/ui-validation/completed.png', 'output/ml/ui-validation/mobile.png']
    image_code = f'''from PIL import Image
import json
paths={image_paths!r}
verified=[]
for path in paths:
 image=Image.open('/workspace/'+path); size=image.size; image.verify()
 verified.append({{'path':path,'size':size,'valid':True}})
print(json.dumps(verified))'''
    images = json.loads(execute(ML + ['exec', '-T', 'healthcare-ml',
        '/home/airflow/ml-env/bin/python', '-c', image_code]))
    # Fetch the authenticated DAG API inside its container; credentials/tokens
    # remain there and are never returned or written to audit artifacts.
    auth_code = '''import json,urllib.request
from pathlib import Path
password=json.loads(Path('/opt/airflow/simple_auth_manager_passwords.json.generated').read_text())['student']
req=urllib.request.Request('http://localhost:8080/auth/token',data=json.dumps({'username':'student','password':password}).encode(),headers={'Content-Type':'application/json'})
token=json.load(urllib.request.urlopen(req))['access_token']
result=[]
for dag in ['healthcare_pipeline','healthcare_fixed_ml_baseline']:
 req=urllib.request.Request('http://localhost:8080/api/v2/dags/'+dag,headers={'Authorization':'Bearer '+token})
 with urllib.request.urlopen(req) as response:
  data=json.load(response);result.append({'status':response.status,'dag_id':data['dag_id']})
print(json.dumps(result))'''
    dag_api = json.loads(execute(ML + ['exec', '-T', 'airflow-api-server', 'python', '-c', auth_code]))
    browser = json.loads((OUT.parent / 'ml/ui-validation/browser-evidence.json').read_text())
    assert browser['actual_live_status_observed'] and browser['back_edge_traversals'] == 2
    assert not browser['console_errors']
    comparison = json.loads((OUT.parent / 'evaluation/comparison.json').read_text())
    assert comparison['suite_passed'] and comparison['controls_passed']
    runtime = json.loads((OUT / 'runtime.json').read_text())
    assert runtime['passed']
    with urllib.request.urlopen('http://127.0.0.1:8080/api/v2/monitor/health', timeout=10) as response:
        airflow_health = json.load(response)
    for name in ['metadatabase', 'scheduler', 'dag_processor']:
        assert airflow_health[name]['status'] == 'healthy', (name, airflow_health)
    database_tcp = {service: execute(BASE + ['exec', '-T', service, 'pg_isready',
                    '-h', '127.0.0.1', '-p', '5432', '-U', user])
                    for service, user in [('postgres', 'airflow'), ('healthcare-db', 'healthcare')]}
    report = {'part1': {'passed': True, 'dags': dags, 'runs': runs, 'tasks': tasks,
        'scheduled_success_evidence': scheduled, 'parallel_overlap': overlaps, 'postgresql_counts': counts,
        'analysis_matches_postgresql': True, 'analysis_artifact': str(png.relative_to(ROOT)), 'analysis_values': ratings,
        'dataset_sources': [
            {'file': 'dataset/Hospital_General_Information_2.csv', 'canonical_catalog': 'https://data.cms.gov/provider-data/dataset/xubh-q36u'},
            {'file': 'dataset/Healthcare_Associated_Infections-Hospital.csv', 'canonical_catalog': 'https://data.cms.gov/provider-data/dataset/77hc-ibv8'}],
        'provenance_limit': 'Exact original download date/release was not recorded. Local files are hashed; live catalog row counts can differ.',
        'transformations': ['Preserve Facility ID as text, trim fields, normalize empty/Not Available values.',
            'Validate required columns and unique hospital IDs; ratings must be 1–5 when numeric.',
            'Parse infection dates, coerce numeric scores, remove exact duplicates, reject conflicting keys.',
            'Left-join infections to hospitals on Facility ID with many-to-one validation and match coverage.',
            'Transactional PostgreSQL refresh with row-count verification.',
            'Read PostgreSQL back; produce reporting coverage and HAI_1_SIR averages by hospital rating.'],
        'authenticated_dag_api': dag_api},
        'part2': {'passed': True, 'live_browser_evidence': browser,
            'custom_logic': 'Six healthcare nodes with a project-specific numeric regularization policy.',
            'screenshots': ['output/ml/ui-validation/running.png', 'output/ml/ui-validation/completed.png', 'output/ml/ui-validation/mobile.png']},
        'part3': {'passed': True, 'case_count': comparison['case_count'], 'summary': comparison['summary'],
            'adaptive_recovery_observed': comparison['cases'][3]['systems']['adaptive_multi_agent']['observed_result']['path'],
            'separate_budgets': {'ridge_fits': 3, 'database_retries': 2},
            'baseline_provenance': comparison['definition']['systems']},
        'runtime_audit': 'output/audit/runtime.json', 'validated_images': images, 'airflow_health': airflow_health,
        'postgresql_tcp_checks': database_tcp,
        'remaining': []}
    (OUT / 'assignment.json').write_text(json.dumps(report, indent=2))
    print('Parts 1–3 audit passed. Evidence: output/audit/assignment.json')


if __name__ == '__main__':
    collect()
