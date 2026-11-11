"""Нагрузка, проверки Kafka/Flink и сбор метрик внутри kind."""

import argparse
import http.client
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from random import Random
from threading import local
from time import monotonic, perf_counter, sleep, time
from urllib.parse import quote
from uuid import uuid4

import numpy as np
from confluent_kafka import Producer

from common import HTTP, dumps, feature_key, key, redis_client

CONNECTIONS = local()


def http_json(url):
    with HTTP.open(url, timeout=5) as response:
        return json.load(response)


def infer(host, bmi):
    connections = getattr(CONNECTIONS, 'by_host', None)
    if connections is None:
        connections = CONNECTIONS.by_host = {}
    connection = connections.setdefault(host, http.client.HTTPConnection(host, 8000, timeout=4))
    body = dumps({'features': [[50., bmi, 0., 1., 3., 1., 0.]]})
    start = perf_counter()
    try:
        connection.request('POST', '/score', body, {'Content-Type': 'application/json'})
        response = connection.getresponse()
        data = json.loads(response.read())
        status = response.status
    except (OSError, http.client.HTTPException, ValueError):
        connection.close()
        return 0, '', (perf_counter() - start) * 1000
    return status, data.get('version', ''), (perf_counter() - start) * 1000


def load(count, rate, canary=False, drift=False):
    generator = Random(42)
    started = monotonic()
    futures = []
    with ThreadPoolExecutor(max_workers=16) as executor:
        for index in range(count):
            remaining = started + index / rate - monotonic()
            if remaining > 0:
                sleep(remaining)
            candidate = canary and index % 20 == 0
            bmi = generator.uniform(18, 40) + (16 if drift else 0)
            futures.append((candidate, executor.submit(infer, 'candidate' if candidate else 'model', bmi)))
    rows = [(candidate, *future.result()) for candidate, future in futures]
    latencies = [row[3] for row in rows if row[1] == 200]
    statuses = Counter(row[1] for row in rows)
    p50, p95, p99 = map(float, np.percentile(latencies, [50, 95, 99])) if latencies else (None, None, None)
    return {'requests': count, 'success': statuses[200], 'availability': statuses[200] / count,
            'status_codes': dict(statuses), 'p50_ms': p50, 'p95_ms': p95, 'p99_ms': p99,
            'rate': rate, 'throughput': count / (monotonic() - started), 'versions': dict(Counter(row[2] for row in rows)),
            'candidate_requests': sum(row[0] for row in rows), 'candidate_success': sum(row[0] and row[1] == 200 for row in rows)}


def producer():
    return Producer({'bootstrap.servers': 'kafka:9092', 'enable.idempotence': True, 'linger.ms': 1,
                     'socket.nagle.disable': True})


def queue(count):
    client = producer()
    run = uuid4().hex[:10]
    for index in range(count):
        data = {'id': f'{run}-{index}', 'features': [50., 25., 0., 1., 3., 1., 0.]}
        client.produce('scale-events', key=str(index % 8), value=dumps(data).encode())
    assert client.flush(15) == 0
    return {'queued': count, 'run': run}


def flink_probe():
    db = redis_client()
    run = 'lab6-' + uuid4().hex[:12]
    db.set(feature_key(1), '[50,25,0,1]', nx=True)
    event = {'run': run, 'id': 'probe', 'patient': 1, 'event_ms': int(time() * 1000), 'stay': 3, 'destination': 'Rehab'}
    client = producer()
    client.produce('hospital-events', key='1', value=dumps(event).encode())
    assert client.flush(10) == 0
    for _ in range(100):
        outcome = db.get(key(run, 'pred', 'probe'))
        if outcome:
            break
        sleep(.2)
    else:
        raise RuntimeError('Flink did not persist the prediction')
    job = next(row for row in http_json('http://readmissions-rest:8081/jobs/overview')['jobs'] if row['state'] == 'RUNNING')
    for _ in range(25):
        checkpoint = http_json(f"http://readmissions-rest:8081/jobs/{job['jid']}/checkpoints")['latest'].get('completed')
        if checkpoint:
            break
        sleep(1)
    else:
        raise RuntimeError('Flink checkpoint not observed')
    return {'job_id': job['jid'], 'prediction': json.loads(outcome), 'checkpoint_duration_sec': checkpoint['end_to_end_duration'] / 1000,
            'checkpoint_state_bytes': checkpoint['state_size']}


def prometheus(expression):
    data = http_json('http://prometheus:9090/api/v1/query?query=' + quote(expression))['data']['result']
    return [float(row['value'][1]) for row in data]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['load', 'queue', 'flink', 'monitor'])
    parser.add_argument('--count', type=int, default=1000)
    parser.add_argument('--rate', type=int, default=50)
    parser.add_argument('--canary', action='store_true')
    parser.add_argument('--drift', action='store_true')
    args = parser.parse_args()
    if args.mode == 'load':
        result = load(args.count, args.rate, args.canary, args.drift)
    elif args.mode == 'queue':
        result = queue(args.count)
    elif args.mode == 'flink':
        result = flink_probe()
    else:
        result = {'drift': max(prometheus('lab6_data_drift_psi'), default=0),
                  'node_samples': len(prometheus('container_memory_working_set_bytes{container="model"}')),
                  'alerts': [item['labels']['alertname'] for item in http_json('http://prometheus:9090/api/v1/alerts')['data']['alerts']]}
    print(dumps(result))


if __name__ == '__main__':
    main()
