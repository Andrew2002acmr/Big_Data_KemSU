"""Воспроизводимый стенд lab6 в отдельном кластере kind."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
WORK = ROOT / 'tmp'
CLUSTER = 'lab6'
KUBE = WORK / ('kubeconfig-internal' if os.environ.get('LAB6_IN_CONTAINER') else 'kubeconfig')
NAMESPACE = 'lab6'
TOOLS = ['kubectl', '--kubeconfig', str(KUBE), '--namespace', NAMESPACE]
IMAGES = {
    'model-build': '',
    'model': 'lab6-control-plane,lab6-worker',
    'spark': 'lab6-worker,lab6-worker2',
    'flink': 'lab6-worker',
}
CHARTS = [
    ('spark', 'spark-operator', 'https://kubeflow.github.io/spark-operator', '2.2.1', NAMESPACE,
     ['spark.jobNamespaces={lab6}', 'webhook.enable=true', 'controller.leaderElection.enable=false',
      'webhook.leaderElection.enable=false']),
    ('flink', 'flink-kubernetes-operator', 'https://archive.apache.org/dist/flink/flink-kubernetes-operator-1.12.0/',
     '1.12.0', NAMESPACE, ['webhook.create=false']),
    ('keda', 'keda', 'https://kedacore.github.io/charts', '2.17.2', 'lab6-keda', []),
]


def run(*args, input_text=None, capture=False):
    result = subprocess.run(args, cwd=ROOT, input=input_text, capture_output=capture, check=True, text=True, encoding='utf-8')
    return result.stdout.strip() if capture else None


def kube(*args, input_text=None, capture=False):
    return run(*TOOLS, *args, input_text=input_text, capture=capture)


def apply_generated(*args):
    manifest = kube('create', *args, '--dry-run=client', '-o', 'json', capture=True)
    kube('apply', '-f', '-', input_text=manifest)


def rollout(resource, timeout='240s'):
    kube('rollout', 'status', resource, '--timeout=' + timeout)


def install_chart(release, chart, repo, version, namespace, overrides):
    flags = [part for value in overrides for part in ('--set', value)]
    run('helm', 'upgrade', '--install', release, chart, '--repo', repo, '--version', version,
        '--namespace', namespace, '--create-namespace', '--kubeconfig', str(KUBE), '--wait', '--timeout', '10m', *flags)


def cluster():
    WORK.mkdir(exist_ok=True)
    if CLUSTER not in run('kind', 'get', 'clusters', capture=True).splitlines():
        run('kind', 'create', 'cluster', '--name', CLUSTER, '--image', 'kindest/node:v1.33.1',
            '--config', 'kind.yml', '--kubeconfig', str(KUBE), '--wait', '120s')
    flags = []
    if os.environ.get('LAB6_IN_CONTAINER'):
        container = os.environ['HOSTNAME']
        networks = run('docker', 'inspect', '--format', '{{json .NetworkSettings.Networks}}', container, capture=True)
        if '"kind":' not in networks:
            run('docker', 'network', 'connect', 'kind', container)
        flags.append('--internal')
    KUBE.write_text(run('kind', 'get', 'kubeconfig', '--name', CLUSTER, *flags, capture=True) + '\n', encoding='utf-8')
    apply_generated('namespace', NAMESPACE)
    kube('label', 'node', 'lab6-control-plane', 'lab6-worker', 'capacity=on-demand', '--overwrite')
    kube('label', 'node', 'lab6-worker', 'lab6-store=true', '--overwrite')
    kube('label', 'node', 'lab6-worker2', 'capacity=spot', '--overwrite')
    kube('taint', 'node', 'lab6-worker2', 'spot=true:NoSchedule', '--overwrite')
    run('docker', 'exec', 'lab6-worker', 'sh', '-c', 'mkdir -p /var/local/lab6-flink && chmod 0777 /var/local/lab6-flink')


def images(reuse):
    for target, nodes in IMAGES.items():
        image = f'lab6-{target}:1'
        if not reuse:
            run('docker', 'build', '--target', target, '-t', image, '-f', 'Dockerfile', '..')
        if nodes:
            run('kind', 'load', 'docker-image', image, '--name', CLUSTER, '--nodes', nodes)


def configure_monitoring():
    for name, files in [('prometheus', ['prometheus.yml', 'alerts.yml']),
                        ('grafana', ['datasource.yml', 'dashboards.yml', 'lab6.json'])]:
        apply_generated('configmap', f'{name}-config', *(f'--from-file=monitoring/{name}' for name in files))
    kube('apply', '-f', 'kubernetes/monitoring.yml')
    kube('apply', '-f', 'kubernetes/headlamp.yml')


def topics():
    rollout('statefulset/kafka')
    for topic, partitions in [('hospital-events', 2), ('scale-events', 8)]:
        kube('exec', 'kafka-0', '--', '/opt/kafka/bin/kafka-topics.sh', '--bootstrap-server', 'kafka:9092',
             '--create', '--if-not-exists', '--topic', topic, '--partitions', str(partitions), '--replication-factor', '1')


def setup(source, reuse):
    from data_check import prepare

    prepare(source)
    cluster()
    with ThreadPoolExecutor(max_workers=4) as workers:
        operators = [workers.submit(install_chart, *chart) for chart in CHARTS]
        container_images = workers.submit(images, reuse)
        kube('apply', '-f', 'kubernetes/core.yml')
        configure_monitoring()
        topics()
        container_images.result()
        for operator in operators:
            operator.result()
    kube('apply', '-f', 'kubernetes/flink.yml')
    for resource in ('statefulset/redis', 'deployment/model', 'deployment/load', 'deployment/prometheus',
                     'deployment/grafana', 'deployment/pushgateway'):
        rollout(resource)
    kube('wait', 'scaledobject/model-kafka', '--for=condition=Ready', '--timeout=120s')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['setup', 'verify', 'all', 'clean'])
    parser.add_argument('--data', help='путь к исходному CSV')
    parser.add_argument('--reuse-images', action='store_true')
    args = parser.parse_args()
    if args.action == 'clean':
        run('kind', 'delete', 'cluster', '--name', CLUSTER, '--kubeconfig', str(KUBE))
    if args.action in ('setup', 'all'):
        setup(args.data, args.reuse_images)
    if args.action in ('verify', 'all'):
        from verify import verify

        verify()


if __name__ == '__main__':
    main()
