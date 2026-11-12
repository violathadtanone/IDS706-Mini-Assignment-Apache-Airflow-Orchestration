"""Build, start and probe real application services; save evidence for README.

Run on the Docker host: python3 audit/runtime.py
No changes are made to README.md or reference materials.
"""
import argparse
import json
import socket
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/audit'
BASE = ['docker', 'compose', '-f', 'docker-compose.yaml', '-f', 'compose.part3.yaml']
ML = BASE + ['-f', 'compose.ml.yaml']
ALL = ML + ['-f', 'compose.evaluation.yaml']


def command(args, timeout=600, check=True):
    print('VERIFY:', ' '.join(args), flush=True)
    started = time.perf_counter()
    result = subprocess.run(args, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=timeout)
    evidence = {'command': args, 'exit_code': result.returncode,
                'seconds': time.perf_counter() - started, 'output': result.stdout}
    COMMANDS.append(evidence)
    if check and result.returncode:
        raise RuntimeError(f'Command failed: {args}\n{result.stdout[-5000:]}')
    return result.stdout


def probe(url):
    with urllib.request.urlopen(url, timeout=10) as response:
        body = response.read()
        result = {'url': url, 'status': response.status, 'bytes': len(body),
                'content_type': response.headers.get('Content-Type'),
                'body_excerpt': body[:1200].decode(errors='replace')}
        if 'application/json' in response.headers.get('Content-Type', ''):
            result['json_body'] = json.loads(body)
        return result


def wait_http(url):
    deadline = time.monotonic() + 180
    while True:
        try:
            return probe(url)
        except (OSError, urllib.error.URLError):
            if time.monotonic() > deadline:
                raise
            time.sleep(2)


COMMANDS = []


def run(skip_oneshots=False):
    OUT.mkdir(parents=True, exist_ok=True)
    config = json.loads(command(ALL + ['config', '--format', 'json']))
    checks = []
    for service, spec in config['services'].items():
        build = spec.get('build')
        if build:
            context = Path(build['context'])
            dockerfile = context / build.get('dockerfile', 'Dockerfile')
            assert context.is_dir() and dockerfile.is_file(), (service, dockerfile)
            checks.append({'service': service, 'dockerfile': str(dockerfile.relative_to(ROOT)), 'exists': True})
        for mount in spec.get('volumes', []):
            if mount['type'] == 'bind':
                assert Path(mount['source']).exists(), (service, mount)
                checks.append({'service': service, 'source': mount['source'],
                               'target': mount['target'], 'source_exists': True})
    # Validate each intended overlay combination; override files are not
    # standalone stacks and must be used with the base Compose file.
    for combination in [BASE[:4], BASE, ML, ALL]:
        command(combination + ['config', '--quiet'])
    command(BASE + ['build', 'airflow-init'])
    command(BASE + ['pull', 'postgres', 'healthcare-db'])
    command(BASE + ['up', '-d', 'postgres', 'healthcare-db', 'airflow-init',
                    'airflow-api-server', 'airflow-scheduler', 'airflow-dag-processor'])
    command(ML + ['build', 'healthcare-ml'])
    command(ML + ['up', '-d', 'healthcare-ml'])
    http = [wait_http('http://127.0.0.1:8080/api/v2/monitor/health'),
            wait_http('http://127.0.0.1:8001/api/graph')]
    http += [probe('http://127.0.0.1:8080/'), probe('http://127.0.0.1:8001/')]
    assert all(item['status'] == 200 for item in http)
    assert http[1]['json_body']['nodes']
    assert '<html' in http[-1]['body_excerpt'].lower()
    for port in [8080, 8001]:
        with socket.create_connection(('127.0.0.1', port), timeout=10):
            checks.append({'host_tcp_port': port, 'listening': True})
    services = {}
    for name in ['postgres', 'healthcare-db', 'airflow-init', 'airflow-api-server',
                 'airflow-scheduler', 'airflow-dag-processor', 'healthcare-ml']:
        cid = command(ML + ['ps', '-aq', name]).strip()
        assert cid, f'No container for {name}'
        inspect = json.loads(command(['docker', 'inspect', cid]))[0]
        state = inspect['State']
        expected = 'exited' if name == 'airflow-init' else 'running'
        assert state['Status'] == expected, (name, state)
        if name == 'airflow-init':
            assert state['ExitCode'] == 0
        workdir = inspect['Config']['WorkingDir'] or '/'
        logs = command(['docker', 'logs', '--tail', '50', cid])
        services[name] = {'expected_state': expected, 'intentional_one_shot': name == 'airflow-init',
            'state': state, 'image': inspect['Config']['Image'],
            'entrypoint': inspect['Config']['Entrypoint'], 'command': inspect['Config']['Cmd'],
            'working_dir': workdir, 'ports': inspect['NetworkSettings']['Ports'], 'logs_tail': logs}
        if name != 'airflow-init':
            command(['docker', 'exec', cid, 'sh', '-c', 'test -d "$1"', 'audit', workdir])
            for path in [m['Destination'] for m in inspect['Mounts']]:
                command(['docker', 'exec', cid, 'sh', '-c', 'test -e "$1"', 'audit', path])
            executable = inspect['Config']['Entrypoint'][0]
            command(['docker', 'exec', cid, 'sh', '-c', 'command -v "$1"', 'audit', executable])
            if name.startswith('airflow-'):
                command(['docker', 'exec', cid, 'sh', '-c', 'command -v airflow && test -x /entrypoint'])
            elif name in ['postgres', 'healthcare-db']:
                command(['docker', 'exec', cid, 'sh', '-c', 'command -v postgres && command -v docker-entrypoint.sh'])
    ui_cid = command(ML + ['ps', '-q', 'healthcare-ml']).strip()
    for path in ['/workspace/ml_workflow/api.py', '/workspace/ml_workflow/graph.py',
                 '/workspace/ml_workflow/static/index.html', '/home/airflow/ml-env/bin/python']:
        command(['docker', 'exec', ui_cid, 'test', '-f', path])
    command(ML + ['exec', '-T', 'healthcare-db', 'pg_isready', '-U', 'healthcare'])
    database = command(ML + ['exec', '-T', 'healthcare-db', 'psql', '-U', 'healthcare', '-At', '-c',
        'SELECT COUNT(*), COUNT(DISTINCT facility_id) FROM hospital_infections'])
    command(ML + ['exec', '-T', 'postgres', 'psql', '-U', 'airflow', '-At', '-c', 'SELECT 1'])
    one_shots = []
    if not skip_oneshots:
        for name in ['adaptive-run', 'evaluation']:
            spec = config['services'][name]
            # Verify module origin and working directory inside the same image
            # with the same mounts before executing the actual default command.
            module = spec['command'][1]
            code = f"import importlib.util,os; p=importlib.util.find_spec({module!r}).origin; assert os.path.isfile(p); assert os.getcwd()=='/workspace'; print(p)"
            command(ALL + ['run', '--rm', '--no-deps', name, '-c', code])
            result = command(ALL + ['run', '--rm', '--no-deps', name], timeout=900)
            one_shots.append({'service': name, 'intentional_one_shot': True,
                              'exit_code': 0, 'output_tail': result[-5000:]})
    time.sleep(2)
    for name in services:
        if name != 'airflow-init':
            cid = command(ML + ['ps', '-q', name]).strip()
            assert command(['docker', 'inspect', '--format', '{{.State.Running}}', cid]).strip() == 'true'
    report = {'verified_at_utc': datetime.now(timezone.utc).isoformat(), 'passed': True,
              'path_checks': checks, 'services': services, 'http': http,
              'healthcare_database_counts': database.strip(), 'one_shots': one_shots,
              'commands': COMMANDS}
    # Config output contains local-development database passwords. Store only
    # path/command summaries, not the complete resolved environment.
    report['commands'] = [{**c, 'output': '[resolved configuration omitted]'}
                          if '--format' in c['command'] and 'config' in c['command'] else
                          {**c, 'output': '[container inspection omitted]'} if c['command'][:2] == ['docker', 'inspect']
                          and '--format' not in c['command'] else c for c in COMMANDS]
    (OUT / 'runtime.json').write_text(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-oneshots', action='store_true')
    args = parser.parse_args()
    try:
        run(args.skip_oneshots)
    except Exception as exc:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / 'runtime_failure.txt').write_text(str(exc))
        raise
    print('Runtime audit passed. Evidence: output/audit/runtime.json')
