"""Метрики Kafka/Flink и публикация замеров в Prometheus Pushgateway."""

import json
import logging
import os
import time

from confluent_kafka import Consumer, KafkaException, TopicPartition
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

from common import HTTP

BOOTSTRAP = os.getenv('KAFKA_BOOTSTRAP', 'kafka:9092')
FLINK = os.getenv('FLINK_URL', 'http://jobmanager:8081')
GATEWAY = os.getenv('PUSHGATEWAY_URL', 'pushgateway:9091')


def rest(path):
    with HTTP.open(FLINK + path, timeout=5) as reply:
        return json.load(reply)


def job():
    for entry in rest('/jobs/overview')['jobs']:
        if entry['state'] == 'RUNNING':
            return entry['jid']
    return None


def metric_consumer():
    return Consumer({'bootstrap.servers': BOOTSTRAP, 'group.id': 'hospital-flink',
                     'enable.auto.commit': False, 'log_level': 0})


def kafka_lag(consumer=None):
    owned = consumer is None
    consumer = metric_consumer() if owned else consumer
    try:
        partitions = [TopicPartition('hospital-events', index) for index in range(2)]
        committed = consumer.committed(partitions, timeout=5)
        return sum(max(0, high - max(low, part.offset)) for part in committed
                   for low, high in [consumer.get_watermark_offsets(part, timeout=5)])
    finally:
        if owned:
            consumer.close()


def checkpoint(job_id):
    return rest(f'/jobs/{job_id}/checkpoints')['latest'].get('completed')


def health(job_id, consumer=None):
    completed = checkpoint(job_id)
    if completed is None:
        return None
    result = {'kafka_lag': kafka_lag(consumer), 'checkpoint_duration_seconds': completed['end_to_end_duration'] / 1000,
              'state_bytes': completed['state_size'], 'checkpoint_id': completed['id']}
    managers = rest('/taskmanagers')['taskmanagers']
    if managers:
        suffix = 'Status.JVM.Memory.Heap.Used,Status.JVM.CPU.Load'
        raw = rest(f"/taskmanagers/{managers[0]['id']}/metrics?get={suffix}")
        for row in raw:
            name = 'task_heap_bytes' if row['id'].endswith('.Used') else 'task_cpu_load'
            result[name] = float(row['value'])
    return result


def publish(name, values, labels=None):
    if not GATEWAY:
        return
    registry = CollectorRegistry()
    for key, value in values.items():
        Gauge(f'lab5_{key}', key.replace('_', ' '), registry=registry).set(value)
    push_to_gateway(GATEWAY, job=name, grouping_key=labels, registry=registry, timeout=5)


def main():
    logging.basicConfig(level=logging.INFO)
    consumer = metric_consumer()
    try:
        while True:
            try:
                job_id = job()
                if job_id:
                    report = health(job_id, consumer)
                    if report:
                        publish('stream-health', report)
            except (OSError, ValueError, RuntimeError, KafkaException) as error:
                logging.warning('Flink metrics unavailable: %s', error)
            time.sleep(3)
    finally:
        consumer.close()


if __name__ == '__main__':
    main()
