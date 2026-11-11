"""Реальная нагрузка HTTP, shadow, canary и возврат на champion."""

import socket
from collections import Counter
from contextlib import contextmanager
from itertools import count, islice
from threading import Thread
from time import monotonic, perf_counter, sleep

import httpx
import mlflow
import numpy as np
import uvicorn
from scipy.special import expit

from api import create_app, in_canary
from training import MODEL


@contextmanager
def serving(app):
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(app, log_level='warning', access_log=False))
        thread = Thread(target=server.run, kwargs={'sockets': [listener]}, daemon=True)
        thread.start()
        try:
            deadline = monotonic() + 15
            while not server.started and thread.is_alive() and monotonic() < deadline:
                sleep(.02)
            if not server.started:
                raise RuntimeError('API не запустился')
            with httpx.Client(base_url=f'http://127.0.0.1:{port}', timeout=10, trust_env=False) as client:
                yield client
        finally:
            server.should_exit = True
            thread.join(timeout=10)


def measure(client, ids, explain=False):
    timings, responses = [], []
    failures = 0
    start = perf_counter()
    for index, key in enumerate(ids):
        before = perf_counter()
        response = client.post('/predict', json={'patient_id': int(key), 'request_id': f'request-{index}', 'explain': explain})
        timings.append((perf_counter() - before) * 1000)
        if response.is_success:
            responses.append(response.json())
        else:
            failures += 1
    wall = perf_counter() - start
    p50, p95, p99 = map(float, np.percentile(timings, [50, 95, 99]))
    return {'requests': len(ids), 'p50_ms': p50, 'p95_ms': p95, 'p99_ms': p99, 'throughput_rps': len(ids) / wall,
            'errors': failures, 'errors_per_sec': failures / wall}, responses


def acceptance(score, latency, champion_auc):
    return (score['auc'] >= max(.55, champion_auc - .01) and score['fpr'] < .01 and
            latency['p95_ms'] < 50 and latency['errors'] == 0)


def benchmark(work, ids, unknown_id, batch_score, online_score):
    app = create_app(work)
    release = app.state.release
    previous = release.champion[1]
    with serving(app) as client:
        for key in ids[:20]:
            client.post('/predict', json={'patient_id': int(key)}).raise_for_status()
        baseline, plain = measure(client, ids)
        explained, extra = measure(client, ids, explain=True)
        assert baseline['errors'] == explained['errors'] == 0
        np.testing.assert_allclose([row['probability'] for row in plain], [row['probability'] for row in extra], atol=1e-12)
        for row in extra:
            parts = row['explanation']
            assert abs(expit(parts['base_log_odds'] + sum(parts['shap'].values())) - row['probability']) < 1e-9
        invalid = client.post('/predict', json={'patient_id': 0}).status_code
        missing = client.post('/predict', json={'patient_id': unknown_id}).status_code
        assert invalid == 422 and missing == 404

        release.mode = 'shadow'
        shadow_metrics, shadow = measure(client, ids[:80])
        assert shadow_metrics['errors'] == 0 and all(row['version'] == previous for row in shadow)
        release.mode = 'canary'
        canary_metrics, canary = measure(client, ids)
        counts = Counter(row['version'] for row in canary)
        assert canary_metrics['errors'] == 0
        assert counts[release.candidate[1]] == sum(in_canary(f'request-{index}') for index in range(len(ids)))

        # Измерение p95 кандидата при направлении всех запросов тестового клиента в 5%-й бакет.
        request_ids = (f'candidate-{index}' for index in count())
        keys = list(islice(filter(in_canary, request_ids), len(ids)))
        started = perf_counter()
        latencies = []
        for request_id, key in zip(keys, ids, strict=True):
            before = perf_counter()
            response = client.post('/predict', json={'patient_id': int(key), 'request_id': request_id})
            response.raise_for_status()
            assert response.json()['version'] == release.candidate[1]
            latencies.append((perf_counter() - before) * 1000)
        candidate_latency = {'p95_ms': float(np.percentile(latencies, 95)), 'errors': 0,
                             'throughput_rps': len(ids) / (perf_counter() - started)}
        approved = acceptance(online_score, candidate_latency, batch_score['auc'])
        # Отказ инъецируется после canary, чтобы проверить штатный путь отката.
        failed_gate = acceptance(online_score, {**candidate_latency, 'errors': 1}, batch_score['auc'])
        assert not failed_gate
        after = release.finish(failed_gate)
        assert after == previous == str(release.client.get_model_version_by_alias(MODEL, 'champion').version)
        assert client.get('/health').json()['mode'] == 'stable'
        assert client.post('/predict', json={'patient_id': int(ids[0]), 'request_id': keys[0]}).json()['version'] == previous
    with mlflow.start_run(run_name='FastAPI: shadow, canary и rollback'):
        mlflow.log_metrics({'p95_ms': baseline['p95_ms'], 'explanation_p95_ms': explained['p95_ms'],
                            'candidate_p95_ms': candidate_latency['p95_ms'], 'rps': baseline['throughput_rps']})
    return {'baseline': baseline, 'explanation': explained, 'candidate': candidate_latency, 'shadow': shadow_metrics,
            'canary': {**canary_metrics, 'traffic_by_version': dict(counts)}, 'naturally_approved': bool(approved),
            'injected_rollback': True, 'champion_after': after, 'invalid_http': invalid, 'missing_http': missing}
