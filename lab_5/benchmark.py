"""Нагрузка Kafka, проверка окон и восстановление из checkpoint."""

import argparse
import json
import os
from pathlib import Path
from random import Random
from threading import Event, Lock, Thread
from time import monotonic, perf_counter_ns, sleep, time
from urllib.request import Request
from uuid import uuid4

import numpy as np
from confluent_kafka import KafkaException, Producer

from common import DESTINATIONS, HTTP, MODEL_VERSION, build_features, dumps, feature_key, key, redis_client, scores, session
from monitor import BOOTSTRAP, checkpoint, health, job, publish, rest


def until(function, description, timeout=90, interval=.05):
    deadline = monotonic() + timeout
    last_error = None
    while monotonic() < deadline:
        try:
            result = function()
            if result is not None and result is not False:
                return result
        except (OSError, KafkaException) as error:
            last_error = error
        sleep(interval)
    raise TimeoutError(f'{description}: {last_error or "timeout"}')


def seed_profiles(db):
    rng = Random(42)
    profiles = {index: [float(rng.randrange(18, 91)), round(rng.uniform(18, 40), 2),
                        float(rng.random() < .25), float(rng.random() < .3)] for index in range(1, 129)}
    profiles.update({1000 + index: [row[0], row[1] + 12, row[2], row[3]] for index, row in list(profiles.items())})
    with db.pipeline(transaction=False) as pipe:
        for index, row in profiles.items():
            pipe.set(feature_key(index), dumps(row), nx=True)
        pipe.execute()
    saved = [json.loads(row) for row in db.mget([feature_key(index) for index in profiles])]
    if saved != list(profiles.values()):
        raise ValueError('Feature snapshots must be immutable')
    return profiles


def event(run, name, stamp, patient=1, stay=3, destination='Home', **options):
    return {'run': run, 'id': name, 'patient': patient, 'event_ms': stamp, 'stay': stay,
            'destination': destination, **options}


def producer_client():
    return Producer({'bootstrap.servers': BOOTSTRAP, 'linger.ms': 0, 'acks': 'all',
                     'enable.idempotence': True, 'socket.nagle.disable': True})


def send(producer, topic, value, partition=None):
    options = {} if partition is None else {'partition': partition}
    envelope = {**value, 'sent_ms': int(time() * 1000)}
    producer.produce(topic, key=str(value['patient']), value=dumps(envelope).encode(), **options)
    producer.poll(0)


def timed_load(db, producer, topic, rows, rate, timeout=90):
    lock = Lock()
    complete = Event()
    pending, observed, errors = {}, {}, []

    def poll_results():
        try:
            while not complete.is_set():
                with lock:
                    sample = list(pending.items())[:512]
                if sample:
                    values = db.mget([name for name, _ in sample])
                    now = perf_counter_ns()
                    for (name, start), raw in zip(sample, values, strict=True):
                        if raw is not None:
                            result = json.loads(raw)
                            with lock:
                                observed[result['id']] = ((now - start) / 1e6, result)
                                pending.pop(name)
                complete.wait(.001)
        except Exception as error:
            errors.append(error)
            complete.set()

    poller = Thread(target=poll_results, daemon=True)
    poller.start()
    started = monotonic()
    try:
        for index, row in enumerate(rows):
            wait = started + index / rate - monotonic()
            if wait > 0:
                sleep(wait)
            name = key(row['run'], 'pred', row['id'])
            with lock:
                pending[name] = perf_counter_ns()
            send(producer, topic, row)
        if producer.flush(10):
            raise RuntimeError('Kafka producer did not flush')
        until(lambda: not pending or bool(errors), f'{topic}: {len(rows)} predictions', timeout)
        if errors:
            raise errors[0]
    finally:
        complete.set()
        poller.join(6)
    p50, p95, p99 = map(float, np.percentile([latency for latency, _ in observed.values()], [50, 95, 99]))
    ordered = [observed[row['id']][1] for row in rows]
    result = {'rows': len(ordered), 'target_rate': rate, 'throughput_eps': len(ordered) / (monotonic() - started),
              'p50_ms': p50, 'p95_ms': p95, 'p99_ms': p99}
    for field in ('feature_us', 'model_us', 'ingest_ms', 'to_sink_ms'):
        if field in ordered[0]:
            p95_stage, p99_stage = map(float, np.percentile([value[field] for value in ordered], [95, 99]))
            result[field + '_p95'] = p95_stage
            result[field + '_p99'] = p99_stage
    return result, ordered


def parity(model, profiles, rows, actual):
    offline = scores(model, [build_features(profiles[row['patient']], row) for row in rows])
    online = [value['score'] for value in actual]
    np.testing.assert_allclose(offline, online, rtol=0, atol=1e-6)
    assert all(value['model'] == MODEL_VERSION for value in actual)
    return float(np.max(np.abs(offline - online)))


def sensor_psi(reference, shifted):
    edges = np.r_[-np.inf, np.unique(np.quantile(reference, np.arange(1, 10) / 10)), np.inf]
    a = np.maximum(np.histogram(reference, edges)[0] / len(reference), 1e-5)
    b = np.maximum(np.histogram(shifted, edges)[0] / len(shifted), 1e-5)
    return float(((b - a) * np.log(b / a)).sum())


def late_checks(db, producer, run, start):
    topic = 'hospital-events'
    base = (start // 10_000 + 1) * 10_000

    def tick(mark, name):
        for partition in (0, 1):
            send(producer, topic, event(run, name + str(partition), mark, tick=True), partition)
        producer.flush(10)
        sleep(.6)

    first = event(run, 'on-time', base + 1000)
    newer = event(run, 'newer', base + 6000)
    timed_load(db, producer, topic, [first, newer], 50)
    tick(base + 11_000, 'tick-open')
    accepted = event(run, 'accepted-late', base + 2000)
    _, [result] = timed_load(db, producer, topic, [accepted], 50)
    assert result['late'] and result['out_of_order'], result
    send(producer, topic, accepted)
    producer.flush(10)
    tick(base + 17_000, 'tick-close')
    window = until(lambda: db.get(key(run, 'window', f'1-{base + 10_000}')), 'closed event-time window')
    window = json.loads(window)
    assert window['count'] == 3 and window['late'] == 1, window
    send(producer, topic, event(run, 'closed-late', base + 3000))
    send(producer, topic, event(run, 'invalid-stay', base + 18_000, stay=0))
    producer.flush(10)
    late = until(lambda: db.get(key(run, 'late', 'closed-late')), 'late side output')
    dlq = until(lambda: db.get(key(run, 'dlq', 'invalid-stay')), 'DLQ')
    assert not db.exists(key(run, 'pred', 'closed-late'))
    return {'accepted': True, 'out_of_order': True, 'side_output': json.loads(late), 'dlq': json.loads(dlq),
            'window': window, 'duplicate_removed': True}, base + 20_000


def fault_checks(db, producer, topic, run, stamp, job_id=None):
    before = until(lambda: checkpoint(job_id), 'initial Flink checkpoint') if job_id else None
    marker = event(run, 'fault-0', stamp, fault=True)
    send(producer, topic, marker)
    producer.flush(10)
    fault_key = (key(run, 'pred', marker['id']) + ':fault' if job_id else key(run, 'spark-fault', marker['id']))
    until(lambda: db.get(fault_key), 'injected failure marker')
    original = db.get(key(run, 'pred', marker['id']))
    rows = [event(run, f'fault-{index}', stamp + index) for index in range(1, 41)]
    _, observed = timed_load(db, producer, topic, rows, 150, timeout=120)
    assert len(observed) == 40 and original == db.get(key(run, 'pred', marker['id']))
    send(producer, topic, marker)
    producer.flush(10)
    result = {'passed': True, 'rows': 41, 'lost': 0, 'duplicate_result': False, 'injected_after_commit': True}
    if job_id:
        after = until(lambda: (value if (value := checkpoint(job_id)) and value['id'] > before['id'] else None),
                      'Flink checkpoint after restart')
        exceptions = rest(f'/jobs/{job_id}/exceptions')
        assert 'Injected failure after Redis commit' in dumps(exceptions), exceptions
        result.update(checkpoint_before=before['id'], checkpoint_after=after['id'],
                      checkpoint_duration_sec=after['end_to_end_duration'] / 1000)
    return result


def run_benchmark(engine, count, rate):
    db = redis_client()
    model = session()
    profiles = seed_profiles(db)
    producer = producer_client()
    run = uuid4().hex[:10]
    start = max(int(time() * 1000), int(db.get('lab5:virtual-clock') or 0) + 30_000)
    db.set('lab5:virtual-clock', start + 90_000)
    topic = 'hospital-events' if engine == 'flink' else 'spark-events'
    job_id = until(job, 'running Flink job', timeout=150, interval=1) if engine == 'flink' else None
    rng = Random(71)

    def load(name, size, speed, cached=True, offset=0, audit=False):
        nonlocal start
        rows = [event(run, f'{name}-{index}', start + int(index / speed * 1000), rng.randrange(1, 129) + offset,
                      rng.randrange(1, 11), rng.choice(DESTINATIONS), cached=cached, audit=audit) for index in range(size)]
        stats, actual = timed_load(db, producer, topic, rows, speed, timeout=120)
        stats['parity_max_delta'] = parity(model, profiles, rows, actual)
        if audit:
            offline = np.asarray([build_features(profiles[row['patient']], row) for row in rows], dtype=np.float32)
            online = np.asarray([value['features'] for value in actual], dtype=np.float32)
            np.testing.assert_allclose(online, offline, rtol=0, atol=1e-6)
            stats['feature_max_delta'] = float(np.max(np.abs(offline - online)))
        start += int(size / speed * 1000) + 4000
        return stats

    if job_id:
        load('warm-cache', 1200, 2 * rate)
        load('warm-direct', 400, rate, cached=False)
    else:
        load('warm', 30, rate)
    result = {'engine': engine, 'run': run, 'latency_budget_p99_ms': 30}
    if job_id:
        for name, speed, cache in [('baseline', rate, False), ('normal', rate, True), ('peak', 2 * rate, True)]:
            result[name] = load(name, count, speed, cache)
            print(f"{name}: p99={result[name]['p99_ms']:.2f} ms", flush=True)
            result[name]['resources'] = until(lambda: health(job_id), 'Flink health')
            publish('stream-benchmark', {'latency_p99_ms': result[name]['p99_ms'],
                    'latency_p95_ms': result[name]['p95_ms'], 'throughput_eps': result[name]['throughput_eps']}, {'phase': name})
        result['pipeline_plan'] = rest(f'/jobs/{job_id}/plan')['plan']
        result['state_and_resources'] = result['peak']['resources']
        result['drift_load'] = load('drift', 256, rate, offset=1000)
        result['data_drift_psi'] = sensor_psi([profiles[index][1] for index in range(1, 129)],
                                             [profiles[index][1] for index in range(1001, 1129)])
        result['late'], start = late_checks(db, producer, run, start)
        publish('stream-quality', {'drift_psi': result['data_drift_psi'], 'late_fraction': 2 / 4})
        feature = build_features(profiles[1], event(run, 'http', start))
        url = os.getenv('MODEL_URL', 'http://model-server:8000') + '/score'
        request = Request(url, data=dumps({'features': [feature]}).encode(), headers={'Content-Type': 'application/json'})
        with HTTP.open(request, timeout=5) as reply:
            output = json.load(reply)
        np.testing.assert_allclose(output['scores'], scores(model, [feature]), atol=1e-6)
        result['model_http_parity'] = True
    else:
        result['normal'] = load('normal', min(count, 320), rate)
    result['feature_audit'] = load('audit', 128, rate, audit=True)
    result['fault'] = fault_checks(db, producer, topic, run, start, job_id)
    if job_id:
        result['latency_passed'] = all(result[name]['p99_ms'] < 30 for name in ('normal', 'peak'))
        assert result['data_drift_psi'] > .2
    else:
        result['latency_passed'] = result['normal']['p99_ms'] < 30
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', choices=['flink', 'spark'], default='flink')
    parser.add_argument('--rate', type=int, default=50)
    parser.add_argument('--count', type=int, default=1000)
    parser.add_argument('--results', type=Path, default=Path('/results'))
    parser.add_argument('--enforce-slo', action='store_true')
    args = parser.parse_args()
    result = run_benchmark(args.engine, args.count, args.rate)
    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / f'{args.engine}.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('## METRICS ##')
    print('REQUIRED_LATENCY=30')
    print(f"MEASURED_P99_LATENCY={result['normal']['p99_ms']:.2f}")
    if args.engine == 'flink':
        print(f"LATE_DATA_HANDLED={result['late']['accepted']}")
        print(f"CHECKPOINT_DURATION_SEC={result['fault']['checkpoint_duration_sec']:.3f}")
        print(f"DATA_DRIFT_SCORE={result['data_drift_psi']:.4f}")
    print(f"LATENCY_PASSED={result['latency_passed']}")
    if args.enforce_slo and not result['latency_passed']:
        raise SystemExit('p99 must be < 30 ms')


if __name__ == '__main__':
    main()
