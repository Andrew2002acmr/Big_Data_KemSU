"""Сравнение ресурсов, инъекция отказа, canary и проверка наблюдаемости."""

import argparse
from datetime import datetime, timezone
import json
from time import monotonic, sleep
from uuid import uuid4

from deploy import ROOT, WORK, apply_generated, kube, rollout, run

PROFILES = {
    'baseline': (2, '1000m', '2', '1536m', 'm6i.xlarge', .192),
    'tuned': (1, '500m', '1', '768m', 'm6i.large', .096),
}


def wait_for(description, test, timeout=180, interval=3):
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        result = test()
        if result:
            return result
        sleep(interval)
    raise RuntimeError(f'Истекло время ожидания: {description}')


def get(resource, *args):
    return json.loads(kube('get', resource, *args, '-o', 'json', capture=True))


def spark(profile):
    cores, request, limit, memory, instance, rate = PROFILES[profile]
    name = f'hospital-{profile}-{uuid4().hex[:6]}'
    manifest = (ROOT / 'kubernetes/spark.yml').read_text()
    manifest = manifest.replace('hospital-RESOURCE_PROFILE', name)
    for key, value in {'CORES': str(cores), 'REQUEST': request, 'LIMIT': json.dumps(limit),
                       'MEMORY': memory}.items():
        manifest = manifest.replace('EXECUTOR_' + key, value)
    kube('apply', '-f', '-', input_text=manifest)

    def complete():
        status = get('sparkapplication/' + name).get('status', {})
        state = status.get('applicationState', {}).get('state')
        if state in ('FAILED', 'SUBMISSION_FAILED', 'FAILING'):
            raise RuntimeError(f'{name}: {status}')
        return status if state == 'COMPLETED' else None

    status = wait_for(name, complete, timeout=420)
    driver = status['driverInfo']['podName']
    output = kube('logs', driver, capture=True)
    (WORK / f'spark-{profile}.log').write_text(output + '\n', encoding='utf-8')
    result = next(json.loads(line.removeprefix('LAB6_RESULT ')) for line in output.splitlines()
                  if line.startswith('LAB6_RESULT '))
    result.update({'driver': driver, 'instance': instance, 'rate_usd_per_hour': rate,
                   'cost_per_job_usd': result['total_executor_seconds'] * rate / 3600})
    return result


def probe(mode, *args):
    return json.loads(kube('exec', 'deploy/load', '--', 'python', 'load.py', mode, *args, capture=True))


def flink_probe():
    wait_for('Flink RUNNING', lambda: get('flinkdeployment/readmissions').get('status', {})
             .get('jobStatus', {}).get('state') == 'RUNNING', timeout=300)
    return probe('flink')


def healthy(result):
    return result['availability'] >= .999 and result['p95_ms'] is not None and result['p95_ms'] < 50


def outage():
    try:
        run('docker', 'stop', '-t', '0', 'lab6-worker2')
        state = run('docker', 'inspect', '--format', '{{.State.Running}}', 'lab6-worker2', capture=True)
        if state != 'false':
            raise RuntimeError('Spot-узел остался запущен')
        result = probe('load', '--count', '1200', '--rate', '100')
    finally:
        run('docker', 'start', 'lab6-worker2')
    kube('wait', 'node/lab6-worker2', '--for=condition=Ready', '--timeout=90s')
    result['spot_node_stopped'] = True
    return result


def candidate(version, bad=False):
    original = get('deployment/model')
    pod = original['spec']['template']
    pod['metadata']['labels']['track'] = 'canary'
    pod['spec'].pop('topologySpreadConstraints')
    pod['spec']['containers'][0]['env'] = [
        {'name': 'RELEASE_VERSION', 'value': version},
        {'name': 'BAD_RELEASE', 'value': str(int(bad))},
    ]
    manifest = {'apiVersion': 'apps/v1', 'kind': 'Deployment', 'metadata': {'name': 'candidate'},
                'spec': {'replicas': 1, 'selector': {'matchLabels': {'app': 'model', 'track': 'canary'}},
                         'template': pod}}
    kube('apply', '-f', '-', input_text=json.dumps(manifest))
    rollout('deployment/candidate')


def canary():
    kube('set', 'env', 'deployment/model', 'RELEASE_VERSION=v1')
    rollout('deployment/model')
    apply_generated('service', 'clusterip', 'candidate', '--tcp=8000:8000')
    kube('patch', 'service/candidate', '--type=json', '-p',
         '[{"op":"replace","path":"/spec/selector","value":{"app":"model","track":"canary"}}]')
    try:
        candidate('v2')
        accepted = probe('load', '--count', '200', '--rate', '50', '--canary')
        if not (healthy(accepted) and accepted['candidate_success'] == accepted['candidate_requests'] == 10
                and accepted['versions'].get('v2') == 10):
            raise RuntimeError(f'Canary v2 отклонён: {accepted}')
        kube('set', 'env', 'deployment/model', 'RELEASE_VERSION=v2')
        rollout('deployment/model')
        candidate('v3', bad=True)
        rejected = probe('load', '--count', '200', '--rate', '50', '--canary')
        if rejected['candidate_success'] or rejected['status_codes'].get('503') != 10:
            raise RuntimeError(f'Ожидалось 10 ошибок canary v3: {rejected}')
    finally:
        kube('delete', 'deployment/candidate', 'service/candidate', '--ignore-not-found')
    after = probe('load', '--count', '100', '--rate', '50')
    if after['availability'] != 1 or after['versions'] != {'v2': 100}:
        raise RuntimeError(f'Откат v3 не подтверждён: {after}')
    return {'accepted': accepted, 'rejected': rejected, 'after_rollback': after}


def scale():
    queued = probe('queue', '--count', '1400')
    trace = []

    def sample():
        deployment = get('deployment/model')
        replicas = {'desired': deployment['spec']['replicas'], 'ready': deployment.get('status', {}).get('readyReplicas', 0)}
        trace.append(replicas)
        return replicas

    wait_for('увеличение реплик KEDA', lambda: sample()['ready'] > 2, timeout=90)
    wait_for('возврат KEDA к двум репликам', lambda: sample() == {'desired': 2, 'ready': 2}, timeout=180, interval=3)
    return {'queued': queued['queued'], 'peak_desired_replicas': max(row['desired'] for row in trace),
            'peak_ready_replicas': max(row['ready'] for row in trace), 'replica_trace': trace}


def publish(series, grouping=''):
    program = (
        'import sys\nfrom urllib.error import HTTPError\nfrom urllib.request import Request, urlopen\n'
        'data = sys.stdin.buffer.read().replace(b"\\r\\n", b"\\n")\n'
        'request = Request(sys.argv[1], data=data, method="PUT", '
        'headers={"Content-Type": "text/plain; version=0.0.4"})\n'
        'try:\n    urlopen(request, timeout=10).close()\n'
        'except HTTPError as error:\n    sys.exit(error.read().decode())\n'
    )
    target = 'http://pushgateway:9091/metrics/job/lab6' + grouping
    payload = ''.join(f'{key} {value}\n' for key, value in series.items())
    kube('exec', '-i', 'deploy/load', '--', 'python', '-c', program, target, input_text=payload)


def monitoring(flink, jobs, validation, metrics):
    publish({'lab6_flink_checkpoint_seconds': flink['checkpoint_duration_sec'],
             'lab6_schema_compliance': validation['passed'] / validation['checked'],
             **{'lab6_' + name.lower(): value for name, value in metrics.items() if name != 'DATA_DRIFT_SCORE'}})
    for profile, result in jobs.items():
        publish({'lab6_spark_cost_usd': result['cost_per_job_usd'],
                 'lab6_spark_disk_spill_bytes': result['disk_spill_bytes']}, f'/profile/{profile}')
    probe('load', '--drift', '--count', '1400', '--rate', '100')

    def firing():
        result = probe('monitor')
        if result['node_samples'] and result['drift'] > .2 and 'HighDataDrift' in result['alerts']:
            return result
        return None

    return wait_for('PSI, kubelet и алерт Prometheus', firing, timeout=75)


def save(name, result):
    (WORK / name).write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')


def track(jobs, metrics, quality):
    import mlflow

    mlflow.set_tracking_uri('sqlite:///' + (WORK / 'mlflow.db').as_posix())
    name = 'lab6-right-sizing'
    experiment = mlflow.get_experiment_by_name(name)
    experiment_id = (experiment.experiment_id if experiment else
                     mlflow.create_experiment(name, artifact_location=(WORK / 'mlflow-artifacts').as_uri()))
    mlflow.set_experiment(experiment_id=experiment_id)
    with mlflow.start_run():
        mlflow.log_metrics(metrics)
        mlflow.log_param('dataset_sha256', quality['sha256'])
        mlflow.log_param('input_bytes', (WORK / 'hospital.csv').stat().st_size)
        for profile, result in jobs.items():
            mlflow.log_metric(f'{profile}_processing_seconds', result['processing_sec'])
            mlflow.log_metric(f'{profile}_cost_usd', result['cost_per_job_usd'])
        mlflow.log_artifact(str(WORK / 'verification.json'))


def verify():
    jobs = {profile: spark(profile) for profile in PROFILES}
    if jobs['baseline']['groups'] != jobs['tuned']['groups']:
        raise RuntimeError('Результаты двух профилей Spark различаются')
    save('spark.json', jobs)
    tests = {'normal': probe('load', '--count', '1500', '--rate', '50'),
             'peak': probe('load', '--count', '1500', '--rate', '100'), 'spot_outage': outage()}
    save('load.json', tests)
    rollout_test = canary()
    save('release.json', rollout_test)
    flink = flink_probe()
    save('flink.json', flink)
    scaling = scale()
    save('keda.json', scaling)
    return finish(jobs, tests, rollout_test, flink, scaling)


def finish(jobs, tests, rollout_test, flink, scaling):
    quality = json.loads((WORK / 'data.json').read_text(encoding='utf-8'))
    model = json.loads((ROOT.parent / 'lab_5/model/readmission.json').read_text(encoding='utf-8'))
    savings = 1 - jobs['tuned']['cost_per_job_usd'] / jobs['baseline']['cost_per_job_usd']
    availability = sum(test['success'] for test in tests.values()) / sum(test['requests'] for test in tests.values())
    metrics = {'REQUIRED_COST_SAVINGS': 35, 'ACHIEVED_COST_SAVINGS': savings * 100,
               'SYSTEM_AVAILABILITY': availability, 'COST_PER_JOB': jobs['tuned']['cost_per_job_usd']}
    monitor = monitoring(flink, jobs, quality['validation'], metrics)
    save('monitor.json', monitor)
    metrics['DATA_DRIFT_SCORE'] = monitor['drift']
    publish({'lab6_observed_drift_score': monitor['drift']}, '/probe/drift')
    sizes = [int(size) for size in run('docker', 'image', 'inspect', '--format', '{{.Size}}',
                                      'lab6-model-build:1', 'lab6-model:1', capture=True).splitlines()]
    result = {'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'metrics': metrics, 'spark': jobs,
              'load': tests, 'release': rollout_test, 'flink': flink, 'keda': scaling, 'monitor': monitor,
              'data': quality, 'image_bytes': dict(zip(('builder', 'runtime'), sizes, strict=True)),
              'model_fpr': model['test_fpr'],
              'targets': {'cost_passed': savings >= .35, 'load_passed': all(map(healthy, tests.values())),
                          'fpr_passed': model['test_fpr'] < .01}}
    save('verification.json', result)
    track(jobs, metrics, quality)
    print('## METRICS ##')
    for name, value in metrics.items():
        print(f'{name}={value:.8f}')
    if not all(result['targets'].values()):
        raise RuntimeError(f"Контрольные пороги не пройдены: {result['targets']}")
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('check', choices=['all', 'baseline', 'tuned', 'canary', 'scale', 'outage', 'flink', 'load'])
    args = parser.parse_args()
    checks = {'all': verify, 'baseline': lambda: spark('baseline'), 'tuned': lambda: spark('tuned'),
              'canary': canary, 'scale': scale, 'outage': outage, 'flink': flink_probe,
              'load': lambda: probe('load', '--count', '1000', '--rate', '100')}
    result = checks[args.check]()
    if args.check != 'all':
        print(json.dumps(result))
